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
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_REPAIR_ITERATIONS,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter
from anonymizer.engine.rewrite.domain_classification import DomainClassificationWorkflow
from anonymizer.engine.rewrite.evaluate import EvaluateWorkflow
from anonymizer.engine.rewrite.final_judge import FinalJudgeWorkflow
from anonymizer.engine.rewrite.qa_generation import QAGenerationWorkflow
from anonymizer.engine.rewrite.repair import RepairWorkflow
from anonymizer.engine.rewrite.rewrite_generation import RewriteGenerationWorkflow, _has_entities
from anonymizer.engine.rewrite.sensitivity_disposition import SensitivityDispositionWorkflow

logger = logging.getLogger("anonymizer.rewrite.workflow")

_PASSTHROUGH_DEFAULTS: dict[str, object] = {
    COL_UTILITY_SCORE: 1.0,
    COL_LEAKAGE_MASS: 0.0,
    COL_ANY_HIGH_LEAKED: False,
    COL_NEEDS_HUMAN_REVIEW: False,
    COL_JUDGE_EVALUATION: None,
    COL_REPAIR_ITERATIONS: 0,
}

_ROW_ORDER_COL = "_anonymizer_row_order"

# Seed columns per adapter call — only these are serialized to parquet.
# RECORD_ID_COLUMN is included in every list so the adapter preserves the
# stable ID from detection rather than recomputing from different column sets.
_SEED_PRE_GENERATION = [RECORD_ID_COLUMN, COL_TEXT, COL_TAGGED_TEXT, COL_FINAL_ENTITIES, COL_LATENT_ENTITIES]
_SEED_EVALUATE = [RECORD_ID_COLUMN, COL_QUALITY_QA, COL_PRIVACY_QA, COL_REWRITTEN_TEXT]
_SEED_REPAIR = [
    RECORD_ID_COLUMN,
    COL_TEXT,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_PRIVACY_QA_REANSWER,
    COL_PRIVACY_QA,
    COL_LEAKAGE_MASS,
    COL_ANY_HIGH_LEAKED,
    COL_UTILITY_SCORE,
]
_SEED_REWRITE_GEN = [
    RECORD_ID_COLUMN,
    COL_TEXT,
    COL_TAGGED_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_ENTITIES_BY_VALUE,
    COL_REPLACEMENT_MAP,
]
_SEED_JUDGE = [RECORD_ID_COLUMN, COL_TEXT, COL_REWRITTEN_TEXT, COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_ANY_HIGH_LEAKED]


# ---------------------------------------------------------------------------
# Row split / merge helpers — TODO(#60): move to shared module
# ---------------------------------------------------------------------------


def _split_by_entities(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition df into (entity_rows, passthrough_rows) with row-order tracking."""
    working = df.copy()
    working[_ROW_ORDER_COL] = range(len(working))
    mask = working[COL_ENTITIES_BY_VALUE].apply(_has_entities)
    return working[mask].copy(), working[~mask].copy()


def _merge_and_reorder(
    *parts: pd.DataFrame,
    attrs: dict,
) -> pd.DataFrame:
    """Concat partitions, restore original row order, drop the tracking column."""
    combined = (
        pd.concat(list(parts), ignore_index=True)
        .sort_values(_ROW_ORDER_COL)
        .drop(columns=[_ROW_ORDER_COL])
        .reset_index(drop=True)
    )
    combined.attrs = {**attrs}
    return combined


def _select_seed_cols(df: pd.DataFrame, seed_cols: list[str]) -> pd.DataFrame:
    """Select only the columns a sub-workflow needs for its adapter call."""
    present = [c for c in seed_cols if c in df.columns]
    return df[present].copy()


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

    result[COL_JUDGE_EVALUATION] = None
    result[COL_NEEDS_HUMAN_REVIEW] = True

    for idx, record_id in result[RECORD_ID_COLUMN].astype(str).items():
        if record_id in source_by_id.index.astype(str):
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
        self._rewrite_gen_wf = RewriteGenerationWorkflow(adapter)
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

        entity_rows, passthrough_rows = _split_by_entities(dataframe)

        # Fast path: no entities anywhere
        if entity_rows.empty:
            _apply_passthrough_defaults(passthrough_rows)
            result_df = _merge_and_reorder(passthrough_rows, attrs=dataframe.attrs)
            return RewriteResult(dataframe=result_df, failed_records=all_failed)

        # --- Steps 1-3: domain, disposition, QA (single adapter call) ---
        pre_gen_columns = [
            *self._domain_wf.columns(selected_models=selected_models, data_summary=data_summary),
            *self._disposition_wf.columns(
                selected_models=selected_models,
                privacy_goal=privacy_goal,
                data_summary=data_summary,
            ),
            *self._qa_wf.columns(selected_models=selected_models),
        ]

        pre_gen_seed = _select_seed_cols(entity_rows, _SEED_PRE_GENERATION)
        pre_gen_result = self._adapter.run_workflow(
            pre_gen_seed,
            model_configs=model_configs,
            columns=pre_gen_columns,
            workflow_name="rewrite-pre-generation",
            preview_num_records=preview_num_records,
        )
        entity_rows = _join_new_columns(entity_rows, pre_gen_result.dataframe)
        all_failed.extend(pre_gen_result.failed_records)

        # --- Step 4: rewrite generation ---
        rewrite_gen_seed = _select_seed_cols(entity_rows, _SEED_REWRITE_GEN)
        rewrite_result = self._rewrite_gen_wf.run(
            rewrite_gen_seed,
            model_configs=model_configs,
            selected_models=selected_models,
            replace_model_selection=replace_model_selection,
            privacy_goal=privacy_goal,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
        )
        entity_rows = _join_new_columns(entity_rows, rewrite_result.dataframe)
        all_failed.extend(rewrite_result.failed_records)

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
        combined = _merge_and_reorder(entity_rows, passthrough_rows, attrs=dataframe.attrs)
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

        # Always run initial evaluate -- metrics must be produced even when max_repair_iterations=0
        eval_seed = _select_seed_cols(df, _SEED_EVALUATE)
        eval_result = self._adapter.run_workflow(
            eval_seed,
            model_configs=model_configs,
            columns=eval_columns,
            workflow_name="rewrite-evaluate-0",
            preview_num_records=preview_num_records,
        )
        df = _join_new_columns(df, eval_result.dataframe, overwrite=True, seed_cols=_SEED_EVALUATE)
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
            repair_seed = _select_seed_cols(failing_rows, _SEED_REPAIR)
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

            df = pd.concat([passing_rows, failing_rows], ignore_index=True)

            # Re-evaluate after repair so metrics reflect the repaired text
            eval_seed = _select_seed_cols(df, _SEED_EVALUATE)
            eval_result = self._adapter.run_workflow(
                eval_seed,
                model_configs=model_configs,
                columns=eval_columns,
                workflow_name=f"rewrite-evaluate-{iteration + 1}",
                preview_num_records=preview_num_records,
            )
            df = _join_new_columns(df, eval_result.dataframe, overwrite=True, seed_cols=_SEED_EVALUATE)
            all_failed.extend(eval_result.failed_records)

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
            judge_seed = _select_seed_cols(df, _SEED_JUDGE)
            judge_result = self._adapter.run_workflow(
                judge_seed,
                model_configs=model_configs,
                columns=self._judge_wf.columns(
                    selected_models=selected_models,
                    privacy_goal=privacy_goal,
                    evaluation=evaluation,
                ),
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
            domains = df[COL_DOMAIN].dropna()
            if domains.empty:
                return None
            raw = domains.mode().iloc[0]
            if hasattr(raw, "domain"):
                return str(raw.domain)
            if isinstance(raw, dict):
                return str(raw.get("domain", ""))
            return str(raw)
        except Exception:
            logger.debug("Could not extract domain from COL_DOMAIN", exc_info=True)
            return None
