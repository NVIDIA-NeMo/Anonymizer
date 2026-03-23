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
    COL_REPAIR_ITERATIONS,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.rewrite.domain_classification import DomainClassificationWorkflow
from anonymizer.engine.rewrite.evaluate import _COL_NEEDS_REPAIR, EvaluateWorkflow
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

        pre_gen_result = self._adapter.run_workflow(
            entity_rows,
            model_configs=model_configs,
            columns=pre_gen_columns,
            workflow_name="rewrite-pre-generation",
            preview_num_records=preview_num_records,
        )
        entity_rows = pre_gen_result.dataframe
        all_failed.extend(pre_gen_result.failed_records)

        # --- Step 4: rewrite generation ---
        rewrite_result = self._rewrite_gen_wf.run(
            entity_rows,
            model_configs=model_configs,
            selected_models=selected_models,
            replace_model_selection=replace_model_selection,
            privacy_goal=privacy_goal,
            data_summary=data_summary,
            preview_num_records=preview_num_records,
        )
        entity_rows = rewrite_result.dataframe
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

        for iteration in range(evaluation.max_repair_iterations):
            # Evaluate
            eval_result = self._adapter.run_workflow(
                df,
                model_configs=model_configs,
                columns=self._evaluate_wf.columns(
                    selected_models=selected_models,
                    evaluation=evaluation,
                    domain=domain,
                ),
                workflow_name=f"rewrite-evaluate-{iteration}",
                preview_num_records=preview_num_records,
            )
            df = eval_result.dataframe
            all_failed.extend(eval_result.failed_records)

            needs_repair_mask = df[_COL_NEEDS_REPAIR].apply(bool)
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
            repair_result = self._adapter.run_workflow(
                failing_rows,
                model_configs=model_configs,
                columns=self._repair_wf.columns(
                    selected_models=selected_models,
                    privacy_goal=privacy_goal,
                    effective_threshold=effective_threshold,
                ),
                workflow_name=f"rewrite-repair-{iteration}",
                preview_num_records=preview_num_records,
            )
            all_failed.extend(repair_result.failed_records)

            repaired = repair_result.dataframe
            repaired[COL_REPAIR_ITERATIONS] = repaired[COL_REPAIR_ITERATIONS].apply(lambda x: int(x) + 1)

            df = pd.concat([passing_rows, repaired], ignore_index=True)
        else:
            # Loop exhausted: run a final evaluate so metrics reflect the last repair
            if evaluation.max_repair_iterations > 0:
                eval_result = self._adapter.run_workflow(
                    df,
                    model_configs=model_configs,
                    columns=self._evaluate_wf.columns(
                        selected_models=selected_models,
                        evaluation=evaluation,
                        domain=domain,
                    ),
                    workflow_name="rewrite-evaluate-final",
                    preview_num_records=preview_num_records,
                )
                df = eval_result.dataframe
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
            judge_result = self._adapter.run_workflow(
                df,
                model_configs=model_configs,
                columns=self._judge_wf.columns(
                    selected_models=selected_models,
                    privacy_goal=privacy_goal,
                    evaluation=evaluation,
                ),
                workflow_name="rewrite-final-judge",
                preview_num_records=preview_num_records,
            )
            return judge_result.dataframe, judge_result.failed_records
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
