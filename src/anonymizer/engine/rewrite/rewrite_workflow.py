# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_DOMAIN,
    COL_ENTITIES_BY_VALUE,
    COL_JUDGE_EVALUATION,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_REPAIR_ITERATIONS,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.rewrite.domain_classification import DomainClassificationWorkflow
from anonymizer.engine.rewrite.evaluate import EvaluateWorkflow
from anonymizer.engine.rewrite.final_judge import FinalJudgeWorkflow
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.rewrite.qa_generation import QAGenerationWorkflow
from anonymizer.engine.rewrite.repair import RepairWorkflow
from anonymizer.engine.rewrite.rewrite_generation import RewriteGenerationWorkflow
from anonymizer.engine.rewrite.sensitivity_disposition import SensitivityDispositionWorkflow
from anonymizer.engine.rewrite.workflow_utils import derive_seed_columns, select_seed_cols
from anonymizer.engine.row_partitioning import merge_and_reorder, split_rows

logger = logging.getLogger("anonymizer.rewrite.workflow")

_PASSTHROUGH_DEFAULTS: dict[str, object] = {
    COL_UTILITY_SCORE: 1.0,
    COL_LEAKAGE_MASS: 0.0,
    COL_ANY_HIGH_LEAKED: False,
    COL_NEEDS_HUMAN_REVIEW: False,
    COL_JUDGE_EVALUATION: None,
    COL_REPAIR_ITERATIONS: 0,
}


def _has_entities(entities_by_value: object) -> bool:
    """Return True if this record has at least one detected entity."""
    entities_by_value = normalize_payload(entities_by_value)
    if not entities_by_value or not isinstance(entities_by_value, dict):
        return False
    items = entities_by_value.get("entities_by_value")
    if not isinstance(items, list):
        return False
    return len(items) > 0


def _join_new_columns(
    target: pd.DataFrame,
    source: pd.DataFrame,
    *,
    overwrite: bool = False,
    seed_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Copy columns from source into target using positional alignment.

    When the adapter drops rows (tracked via ``failed_records``), source
    will be shorter than target. We align on ``RECORD_ID_COLUMN`` to
    drop the failed rows from target, then join the new columns from
    source onto the surviving rows.

    Set ``overwrite=True`` to update existing columns that the workflow
    produced (e.g. re-evaluate after repair). Seed columns passed through
    the adapter are excluded from overwrite to prevent re-serialization artifacts.
    """
    if len(source) != len(target):
        dropped = len(target) - len(source)
        logger.warning(
            "Row count mismatch: target=%d, source=%d; dropping %d failed row(s).",
            len(target),
            len(source),
            dropped,
        )
        surviving_ids = set(source[RECORD_ID_COLUMN].astype(str))
        target = (
            target[target[RECORD_ID_COLUMN].astype(str).isin(surviving_ids)]
            .sort_values(RECORD_ID_COLUMN)
            .reset_index(drop=True)
        )
        source = source.sort_values(RECORD_ID_COLUMN).reset_index(drop=True)

    skip = set(seed_cols) if seed_cols else set()
    for col in source.columns:
        if col in skip:
            continue
        if col not in target.columns or overwrite:
            target[col] = source[col].values
    return target


def _join_judge_columns(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Merge judge columns preserving all rows -- judge is non-critical.

    When judge returns fewer rows than target, the missing rows get
    defaults (``COL_JUDGE_EVALUATION=None``, ``COL_NEEDS_HUMAN_REVIEW=True``)
    instead of being dropped from the result.
    """
    if len(source) == len(target):
        return _join_new_columns(target, source)

    logger.warning(
        "Judge returned %d of %d rows; defaulting missing rows to needs_human_review=True.",
        len(source),
        len(target),
    )
    result = target.copy()
    source_by_id = source.set_index(RECORD_ID_COLUMN)
    known_ids = set(source_by_id.index)

    result[COL_JUDGE_EVALUATION] = None
    result[COL_NEEDS_HUMAN_REVIEW] = True

    for idx, record_id in result[RECORD_ID_COLUMN].items():
        if record_id in known_ids:
            row = source_by_id.loc[record_id]
            result.at[idx, COL_JUDGE_EVALUATION] = row.get(COL_JUDGE_EVALUATION)
            result.at[idx, COL_NEEDS_HUMAN_REVIEW] = row.get(COL_NEEDS_HUMAN_REVIEW, True)

    return result


def _apply_passthrough_defaults(
    df: pd.DataFrame,
    *,
    text_col: str = COL_TEXT,
    defaults: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Set default output columns on passthrough (no-entity) rows."""
    df[COL_REWRITTEN_TEXT] = df[text_col]
    for col, default in (defaults or _PASSTHROUGH_DEFAULTS).items():
        df[col] = default
    return df


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewriteResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RewriteWorkflow:
    """Top-level orchestrator for the rewrite pipeline.

    Chains all sub-workflows in order: domain classification,
    sensitivity disposition, QA generation, rewrite generation,
    evaluate-repair loop, and final judge.
    """

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter
        self._domain_wf = DomainClassificationWorkflow()
        self._disposition_wf = SensitivityDispositionWorkflow()
        self._qa_wf = QAGenerationWorkflow()
        self._rewrite_gen_wf = RewriteGenerationWorkflow()
        self._evaluate_wf = EvaluateWorkflow(adapter)
        self._repair_wf = RepairWorkflow(adapter)
        self._judge_wf = FinalJudgeWorkflow()

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
        replace_model_selection: ReplaceModelSelection,
        privacy_goal: PrivacyGoal,
        evaluation: EvaluationCriteria,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> RewriteResult:
        all_failed: list[FailedRecord] = []

        entity_rows, passthrough_rows = split_rows(dataframe, column=COL_ENTITIES_BY_VALUE, predicate=_has_entities)

        # Fast path: no entities anywhere
        if entity_rows.empty:
            _apply_passthrough_defaults(passthrough_rows)
            result_df = merge_and_reorder(passthrough_rows, attrs=dataframe.attrs)
            return RewriteResult(dataframe=result_df, failed_records=all_failed)

        # --- Step 1: replacement map (needs only detection output) ---
        replace_workflow = LlmReplaceWorkflow(adapter=self._adapter)
        replace_result = replace_workflow.generate_map_only(
            entity_rows,
            model_configs=model_configs,
            selected_models=replace_model_selection,
        )
        entity_rows = _join_new_columns(entity_rows, replace_result.dataframe)
        all_failed.extend(replace_result.failed_records)

        # --- Step 2: domain, disposition, QA, rewrite (single adapter call) ---
        pipeline_columns = [
            *self._domain_wf.columns(selected_models=selected_models, data_summary=data_summary),
            *self._disposition_wf.columns(
                selected_models=selected_models,
                privacy_goal=privacy_goal,
                data_summary=data_summary,
            ),
            *self._qa_wf.columns(selected_models=selected_models),
            *self._rewrite_gen_wf.columns(
                selected_models=selected_models,
                privacy_goal=privacy_goal,
                data_summary=data_summary,
            ),
        ]

        pipeline_seed = select_seed_cols(entity_rows, derive_seed_columns(pipeline_columns, entity_rows))
        pipeline_result = self._adapter.run_workflow(
            pipeline_seed,
            model_configs=model_configs,
            columns=pipeline_columns,
            workflow_name="rewrite-pipeline",
            preview_num_records=preview_num_records,
        )
        entity_rows = _join_new_columns(entity_rows, pipeline_result.dataframe)
        all_failed.extend(pipeline_result.failed_records)

        # --- Step 5: evaluate-repair loop ---
        domain = self._extract_domain(entity_rows)
        entity_rows, eval_repair_failed = self._run_evaluate_repair_loop(
            entity_rows,
            model_configs=model_configs,
            selected_models=selected_models,
            privacy_goal=privacy_goal,
            evaluation=evaluation,
            domain=domain,
            preview_num_records=preview_num_records,
        )
        all_failed.extend(eval_repair_failed)

        # --- Step 6: final judge (non-critical) ---
        entity_rows, judge_failed = self._run_final_judge(
            entity_rows,
            model_configs=model_configs,
            selected_models=selected_models,
            privacy_goal=privacy_goal,
            evaluation=evaluation,
            preview_num_records=preview_num_records,
        )
        all_failed.extend(judge_failed)

        # --- Merge and return ---
        _apply_passthrough_defaults(passthrough_rows)
        combined = merge_and_reorder(entity_rows, passthrough_rows, attrs=dataframe.attrs)
        return RewriteResult(dataframe=combined, failed_records=all_failed)

    # ---------------------------------------------------------------------------
    # Evaluate-repair loop
    # ---------------------------------------------------------------------------

    def _run_evaluate_repair_loop(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        evaluation: EvaluationCriteria,
        domain: str | None,
        preview_num_records: int | None,
    ) -> tuple[pd.DataFrame, list[FailedRecord]]:
        all_failed: list[FailedRecord] = []
        effective_threshold = evaluation.get_effective_threshold(domain)

        if COL_REPAIR_ITERATIONS not in df.columns:
            df[COL_REPAIR_ITERATIONS] = 0

        eval_columns = self._evaluate_wf.columns(
            selected_models=selected_models,
            evaluation=evaluation,
            domain=domain,
        )
        eval_seed_cols = derive_seed_columns(eval_columns, df)

        # Always run initial evaluate -- metrics must be produced even when max_repair_iterations=0
        eval_seed = select_seed_cols(df, eval_seed_cols)
        eval_result = self._adapter.run_workflow(
            eval_seed,
            model_configs=model_configs,
            columns=eval_columns,
            workflow_name="rewrite-evaluate-0",
            preview_num_records=preview_num_records,
        )
        df = _join_new_columns(df, eval_result.dataframe, overwrite=True, seed_cols=eval_seed_cols)
        all_failed.extend(eval_result.failed_records)

        repair_columns = self._repair_wf.columns(
            selected_models=selected_models,
            privacy_goal=privacy_goal,
            effective_threshold=effective_threshold,
        )

        for iteration in range(evaluation.max_repair_iterations):
            needs_repair_mask = df[COL_NEEDS_REPAIR].apply(bool)
            if not needs_repair_mask.any():
                logger.info("Evaluate-repair loop: all rows pass at iteration %d", iteration)
                break

            failing_rows = df[needs_repair_mask].copy()
            passing_rows = df[~needs_repair_mask].copy()

            logger.info(
                "Evaluate-repair loop iteration %d: %d/%d rows need repair",
                iteration,
                len(failing_rows),
                len(df),
            )

            # Repair failing rows only
            repair_seed_cols = derive_seed_columns(repair_columns, failing_rows)
            repair_seed = select_seed_cols(failing_rows, repair_seed_cols)
            repair_result = self._adapter.run_workflow(
                repair_seed,
                model_configs=model_configs,
                columns=repair_columns,
                workflow_name=f"rewrite-repair-{iteration}",
                preview_num_records=preview_num_records,
            )
            all_failed.extend(repair_result.failed_records)

            repaired = repair_result.dataframe
            failing_rows = _join_new_columns(failing_rows, repaired)
            if COL_REWRITTEN_TEXT_NEXT in failing_rows.columns:
                failing_rows[COL_REWRITTEN_TEXT] = failing_rows[COL_REWRITTEN_TEXT_NEXT]
            failing_rows[COL_REPAIR_ITERATIONS] = failing_rows[COL_REPAIR_ITERATIONS].apply(lambda x: int(x) + 1)

            # Re-evaluate only repaired rows (passing rows' metrics are unchanged)
            reeval_seed_cols = derive_seed_columns(eval_columns, failing_rows)
            reeval_seed = select_seed_cols(failing_rows, reeval_seed_cols)
            eval_result = self._adapter.run_workflow(
                reeval_seed,
                model_configs=model_configs,
                columns=eval_columns,
                workflow_name=f"rewrite-evaluate-{iteration + 1}",
                preview_num_records=preview_num_records,
            )
            failing_rows = _join_new_columns(
                failing_rows, eval_result.dataframe, overwrite=True, seed_cols=reeval_seed_cols
            )
            all_failed.extend(eval_result.failed_records)

            df = pd.concat([passing_rows, failing_rows], ignore_index=True)

        return df, all_failed

    # ---------------------------------------------------------------------------
    # Final judge (non-critical)
    # ---------------------------------------------------------------------------

    def _run_final_judge(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        evaluation: EvaluationCriteria,
        preview_num_records: int | None,
    ) -> tuple[pd.DataFrame, list[FailedRecord]]:
        try:
            judge_columns = self._judge_wf.columns(
                selected_models=selected_models,
                privacy_goal=privacy_goal,
                evaluation=evaluation,
            )
            judge_seed = select_seed_cols(df, derive_seed_columns(judge_columns, df))
            judge_result = self._adapter.run_workflow(
                judge_seed,
                model_configs=model_configs,
                columns=judge_columns,
                workflow_name="rewrite-final-judge",
                preview_num_records=preview_num_records,
            )
            df = _join_judge_columns(df, judge_result.dataframe)
            return df, judge_result.failed_records
        except Exception:
            logger.warning("Final judge step failed; populating defaults", exc_info=True)
            df[COL_JUDGE_EVALUATION] = None
            df[COL_NEEDS_HUMAN_REVIEW] = True
            return df, []

    # ---------------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _extract_domain(df: pd.DataFrame) -> str | None:
        """Extract the most common domain from the domain classification column, if present."""
        if COL_DOMAIN not in df.columns:
            return None
        try:
            domains = (
                df[COL_DOMAIN]
                .dropna()
                .apply(
                    lambda raw: (
                        str(raw.domain)
                        if hasattr(raw, "domain")
                        else str(raw.get("domain", ""))
                        if isinstance(raw, dict)
                        else str(raw)
                    )
                )
            )
            domains = domains[domains != ""]
            if domains.empty:
                return None
            return str(domains.mode().iloc[0])
        except Exception:
            logger.debug("Could not extract domain from COL_DOMAIN", exc_info=True)
            return None
