# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig

from anonymizer.config.models import EvaluateModelSelection, ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_JUDGE_EVALUATION,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REPAIR_ITERATIONS,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.rewrite._semantic_evaluation import RewriteEvaluationBinding
from anonymizer.engine.rewrite.domain_classification import DomainClassificationWorkflow
from anonymizer.engine.rewrite.evaluate import EvaluateWorkflow
from anonymizer.engine.rewrite.final_judge import FinalJudgeWorkflow
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.rewrite.qa_generation import QAGenerationWorkflow
from anonymizer.engine.rewrite.repair import RepairWorkflow
from anonymizer.engine.rewrite.rewrite_generation import (
    RewriteGenerationWorkflow,
    restore_empty_skipped_span_label_counts,
)
from anonymizer.engine.rewrite.sensitivity_disposition import SensitivityDispositionWorkflow
from anonymizer.engine.rewrite.workflow_utils import derive_seed_columns, select_seed_cols
from anonymizer.engine.row_partitioning import (
    ROW_ORDER_COL,
    merge_and_reorder,
    resolve_source_positions,
    restore_diagnostic_columns,
    split_rows,
)
from anonymizer.engine.schemas import EntitiesByValueSchema
from anonymizer.measurement import stage_timer

logger = logging.getLogger("anonymizer.rewrite.workflow")

_PASSTHROUGH_DEFAULTS: dict[str, object] = {
    COL_UTILITY_SCORE: 1.0,
    COL_LEAKAGE_MASS: 0.0,
    COL_WEIGHTED_LEAKAGE_RATE: 0.0,
    COL_ANY_HIGH_LEAKED: False,
    COL_NEEDS_HUMAN_REVIEW: False,
    COL_REPAIR_ITERATIONS: 0,
}

_EVALUATION_PAYLOAD_COLUMNS = (
    COL_QUALITY_QA_REANSWER,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA_COMPARE,
)


def _detection_valid_fraction(row: pd.Series) -> float | None:
    """Convert bool COL_DETECTION_VALID to a 0–1 fraction for rewrite evaluate output.

    The detection judge stores a bool (all_valid) but rewrite evaluate surfaces
    a fraction so it sits on the same scale as utility_score and leakage_mass.
    Returns None when the score cannot be computed (judge unavailable or entity
    parsing failed).
    """
    valid = row.get(COL_DETECTION_VALID)
    if valid is None:
        return None
    if bool(valid):
        return 1.0
    invalid = row.get(COL_DETECTION_INVALID_ENTITIES)
    invalid_count = len(invalid) if isinstance(invalid, list) else 0
    try:
        total = sum(
            len(e.labels) for e in EntitiesByValueSchema.from_raw(row.get(COL_ENTITIES_BY_VALUE)).entities_by_value
        )
    except Exception:
        logger.warning(
            "Could not parse entities_by_value to compute detection_valid fraction; defaulting to None.",
            exc_info=True,
        )
        return None
    if total == 0 or invalid_count == 0:
        # Reachable only when valid is False (True returns early above).
        # total == 0: judge flagged invalid detections but no entities were found — score is uncomputable.
        # invalid_count == 0: judge flagged the row invalid but listed no specific invalid
        # entities — would spuriously produce 1.0 without this guard.
        return None
    return max(0.0, (total - invalid_count) / total)


def _has_entities(entities_by_value: object) -> bool:
    """Return True if this record has at least one detected entity."""
    entities_by_value = normalize_payload(entities_by_value)
    if not entities_by_value or not isinstance(entities_by_value, dict):
        return False
    items = entities_by_value.get("entities_by_value")
    if not isinstance(items, list):
        return False
    return len(items) > 0


def _normalize_evaluation_payloads(df: pd.DataFrame) -> None:
    for column in _EVALUATION_PAYLOAD_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(normalize_payload)


def _join_new_columns(
    target: pd.DataFrame,
    source: pd.DataFrame,
    *,
    source_positions: tuple[int, ...] | None = None,
    overwrite: bool = False,
    seed_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Copy stage outputs onto their surviving source rows, excluding seed columns."""
    positions = resolve_source_positions(target, source, source_positions)
    target = target.iloc[list(positions)].copy().reset_index(drop=True)
    skip = set(seed_cols or ())
    for col in source.columns:
        if col not in skip and (col not in target.columns or overwrite):
            target[col] = source[col].values
    return target


def _join_preserving(
    target: pd.DataFrame,
    source: pd.DataFrame,
    defaults: dict[str, object],
    *,
    source_positions: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Keep complete-stage trace columns; default diagnostics after row drops."""
    if len(source) == len(target):
        defaults = dict.fromkeys(column for column in source if column not in target)
    return restore_diagnostic_columns(target, source, defaults, source_positions=source_positions)


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
    source_positions: tuple[int, ...] | None = None


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RewriteWorkflow:
    """Top-level orchestrator for the rewrite pipeline.

    Chains all sub-workflows in order: domain classification,
    sensitivity disposition, QA generation, rewrite generation,
    and the evaluate-repair loop. The final judge runs separately
    via evaluate().
    """

    def __init__(self, adapter: NddAdapter, *, semantic_binding: RewriteEvaluationBinding | None = None) -> None:
        self._adapter = adapter
        self._semantic_binding = semantic_binding
        self._domain_wf = DomainClassificationWorkflow()
        self._disposition_wf = SensitivityDispositionWorkflow()
        self._qa_wf = QAGenerationWorkflow()
        self._rewrite_gen_wf = RewriteGenerationWorkflow()
        self._evaluate_wf = EvaluateWorkflow(adapter)
        self._repair_wf = RepairWorkflow(adapter)
        self._judge_wf = FinalJudgeWorkflow()
        self._detection_judge_wf = DetectionJudgeWorkflow(adapter)

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
        strict_entity_protection: bool = False,
    ) -> RewriteResult:
        with stage_timer("RewriteWorkflow.run", input_row_count=len(dataframe)) as measurement:
            all_failed: list[FailedRecord] = []

            entity_rows, passthrough_rows = split_rows(dataframe, column=COL_ENTITIES_BY_VALUE, predicate=_has_entities)
            measurement.update(
                entity_row_count=len(entity_rows),
                passthrough_row_count=len(passthrough_rows),
            )

            # Fast path: no entities anywhere
            if entity_rows.empty:
                _apply_passthrough_defaults(passthrough_rows)
                result_df = merge_and_reorder(passthrough_rows)
                result = RewriteResult(
                    dataframe=result_df,
                    failed_records=all_failed,
                    source_positions=tuple(sorted(passthrough_rows[ROW_ORDER_COL])),
                )
                measurement.update(
                    output_row_count=len(result.dataframe),
                    failed_record_count=len(result.failed_records),
                )
                return result

            # --- Step 1: replacement map (needs only detection output) ---
            replace_workflow = LlmReplaceWorkflow(adapter=self._adapter)
            replace_result = replace_workflow.generate_map_only(
                entity_rows,
                model_configs=model_configs,
                selected_models=replace_model_selection,
            )
            entity_rows = _join_new_columns(
                entity_rows, replace_result.dataframe, source_positions=replace_result.source_positions
            )
            all_failed.extend(replace_result.failed_records)

            # --- Step 2: domain, disposition, QA, rewrite (single adapter call) ---
            pipeline_columns = [
                *self._domain_wf.columns(selected_models=selected_models, data_summary=data_summary),
                *self._disposition_wf.columns(
                    selected_models=selected_models,
                    privacy_goal=privacy_goal,
                    data_summary=data_summary,
                    strict_entity_protection=strict_entity_protection,
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
                **(
                    self._semantic_binding.stage(pipeline_seed, tuple(entity_rows[ROW_ORDER_COL]))
                    if self._semantic_binding
                    else {}
                ),
            )
            restore_empty_skipped_span_label_counts(pipeline_result.dataframe)
            entity_rows = _join_new_columns(
                entity_rows, pipeline_result.dataframe, source_positions=pipeline_result.source_positions
            )
            all_failed.extend(pipeline_result.failed_records)

            # --- Step 5: evaluate-repair loop ---
            entity_rows, eval_repair_failed = self._run_evaluate_repair_loop(
                entity_rows,
                model_configs=model_configs,
                selected_models=selected_models,
                privacy_goal=privacy_goal,
                evaluation=evaluation,
                preview_num_records=preview_num_records,
            )
            all_failed.extend(eval_repair_failed)

            # --- Merge and return ---
            _apply_passthrough_defaults(passthrough_rows)
            combined = merge_and_reorder(entity_rows, passthrough_rows)
            result = RewriteResult(
                dataframe=combined,
                failed_records=all_failed,
                source_positions=tuple(sorted([*entity_rows[ROW_ORDER_COL], *passthrough_rows[ROW_ORDER_COL]])),
            )
            measurement.update(
                output_row_count=len(result.dataframe),
                failed_record_count=len(result.failed_records),
            )
            return result

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
        preview_num_records: int | None,
    ) -> tuple[pd.DataFrame, list[FailedRecord]]:
        all_failed: list[FailedRecord] = []
        binding = self._semantic_binding
        if COL_REPAIR_ITERATIONS not in df.columns:
            df[COL_REPAIR_ITERATIONS] = 0
        eval_columns = self._evaluate_wf.columns(selected_models=selected_models, evaluation=evaluation)
        if binding:
            eval_columns = binding.columns(eval_columns)
        repair_columns = self._repair_wf.columns(
            selected_models=selected_models, privacy_goal=privacy_goal, effective_threshold=evaluation.repair_threshold
        )

        def execute(rows: pd.DataFrame, columns: list[ColumnConfigT], name: str, iteration: int) -> pd.DataFrame:
            seed_cols = derive_seed_columns(columns, rows)
            seed = select_seed_cols(rows, seed_cols)
            result = self._adapter.run_workflow(
                seed,
                model_configs=model_configs,
                columns=columns,
                workflow_name=name,
                preview_num_records=preview_num_records,
                **(binding.stage(seed, tuple(rows[ROW_ORDER_COL]), iteration=iteration) if binding else {}),
            )
            all_failed.extend(result.failed_records)
            return _join_new_columns(
                rows, result.dataframe, source_positions=result.source_positions, overwrite=True, seed_cols=seed_cols
            )

        def unavailable(rows: pd.DataFrame) -> pd.Series:
            return (
                ~rows[COL_REWRITE_REPLACEMENT_READY].apply(bool)
                if COL_REWRITE_REPLACEMENT_READY in rows.columns
                else pd.Series(False, index=rows.index)
            )

        # Initial evaluation remains mandatory, including a zero repair allowance.
        df = execute(df, eval_columns, "rewrite-evaluate-0", 0)
        _normalize_evaluation_payloads(df)
        iterations = binding.iteration_slots if binding else range(evaluation.max_repair_iterations)
        for iteration in iterations:
            if binding:
                failing_rows, passing_rows = binding.repair_rows(df, iteration)
            else:
                selected = df[COL_NEEDS_REPAIR].apply(bool) & ~unavailable(df)
                failing_rows, passing_rows = df[selected].copy(), df[~selected].copy()
            if failing_rows.empty:
                break
            failing_rows = execute(failing_rows, repair_columns, f"rewrite-repair-{iteration}", iteration + 1)
            if binding:
                failing_rows = binding.candidate_rows(failing_rows, iteration)
            elif COL_REWRITTEN_TEXT_NEXT in failing_rows.columns:
                failing_rows[COL_REWRITTEN_TEXT] = failing_rows[COL_REWRITTEN_TEXT_NEXT]
            failing_rows[COL_REPAIR_ITERATIONS] = failing_rows[COL_REPAIR_ITERATIONS].apply(lambda x: int(x) + 1)
            failing_rows = execute(failing_rows, eval_columns, f"rewrite-evaluate-{iteration + 1}", iteration + 1)
            _normalize_evaluation_payloads(failing_rows)
            df = pd.concat([passing_rows, failing_rows], ignore_index=True)
        replacement_unavailable = unavailable(df)

        # Compute needs_human_review from objective metrics after the loop exhausts.
        df.loc[replacement_unavailable, COL_REWRITTEN_TEXT] = None
        needs_review = df[COL_REWRITTEN_TEXT].isna()
        needs_review = needs_review | df[COL_ANY_HIGH_LEAKED].apply(bool)
        if evaluation.flag_utility_below is not None:
            needs_review = needs_review | (df[COL_UTILITY_SCORE].apply(float) < evaluation.flag_utility_below)
        if evaluation.flag_leakage_above is not None:
            needs_review = needs_review | (df[COL_LEAKAGE_MASS].apply(float) > evaluation.flag_leakage_above)
        df[COL_NEEDS_HUMAN_REVIEW] = needs_review

        return df, all_failed

    def _run_final_judge(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        privacy_goal: PrivacyGoal,
        preview_num_records: int | None,
    ) -> tuple[pd.DataFrame, list[FailedRecord]]:
        try:
            judge_columns = self._judge_wf.columns(
                selected_models=selected_models,
                privacy_goal=privacy_goal,
            )
            effective_preview = min(preview_num_records, len(df)) if preview_num_records is not None else None
            judge_seed = select_seed_cols(df, derive_seed_columns(judge_columns, df))
            judge_result = self._adapter.run_workflow(
                judge_seed,
                model_configs=model_configs,
                columns=judge_columns,
                workflow_name="rewrite-final-judge",
                preview_num_records=effective_preview,
            )
            df = _join_preserving(
                df,
                judge_result.dataframe,
                {COL_JUDGE_EVALUATION: None},
                source_positions=judge_result.source_positions,
            )
            return df, judge_result.failed_records
        except Exception:
            logger.debug("Rewrite judge workflow failed; scores may be unavailable.", exc_info=True)
            df[COL_JUDGE_EVALUATION] = None
            return df, []

    # ---------------------------------------------------------------------------
    # Evaluate (detection judge + final judge)
    # ---------------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: EvaluateModelSelection,
        privacy_goal: PrivacyGoal,
        preview_num_records: int | None = None,
        compute_detection_validity: bool = False,
    ) -> RewriteResult:
        """Run detection validity judge and holistic judge on a completed rewrite result.

        Takes the trace dataframe from a prior run() / preview() and appends
        COL_JUDGE_EVALUATION, and optionally COL_DETECTION_VALID /
        COL_DETECTION_INVALID_ENTITIES when ``compute_detection_validity=True``.
        COL_NEEDS_HUMAN_REVIEW is not modified — it was set during run() based on
        objective metrics and judge scores do not influence it.
        """
        entity_rows, passthrough_rows = split_rows(df, column=COL_ENTITIES_BY_VALUE, predicate=_has_entities)

        passthrough_rows = passthrough_rows.copy()
        if compute_detection_validity:
            if not passthrough_rows.empty:
                logger.info(
                    "%d passthrough row(s) have no detected entities — detection_valid set to 1.0 (trivially valid).",
                    len(passthrough_rows),
                )
            passthrough_rows[COL_DETECTION_VALID] = 1.0
            passthrough_rows[COL_DETECTION_INVALID_ENTITIES] = [[] for _ in range(len(passthrough_rows))]
        passthrough_rows[COL_JUDGE_EVALUATION] = None

        if entity_rows.empty:
            combined = merge_and_reorder(passthrough_rows)
            return RewriteResult(dataframe=combined, failed_records=[])

        all_failed: list[FailedRecord] = []

        # --- Detection validity judge (opt-in) ---
        if compute_detection_validity:
            try:
                detection_result = self._detection_judge_wf.evaluate(
                    entity_rows,
                    model_configs=model_configs,
                    selected_models=selected_models,
                    preview_num_records=preview_num_records,
                )
                entity_rows = _join_preserving(
                    entity_rows,
                    detection_result.dataframe,
                    defaults={COL_DETECTION_VALID: None, COL_DETECTION_INVALID_ENTITIES: None},
                    source_positions=detection_result.source_positions,
                )
                all_failed.extend(detection_result.failed_records)
                # Convert bool all_valid → 0–1 fraction so detection validity sits on the
                # same scale as utility_score and leakage_mass in the rewrite scores section.
                entity_rows[COL_DETECTION_VALID] = entity_rows.apply(_detection_valid_fraction, axis=1)
            except Exception:
                logger.debug("Detection validity judge failed; scores may be unavailable.", exc_info=True)
                entity_rows[COL_DETECTION_VALID] = None
                entity_rows[COL_DETECTION_INVALID_ENTITIES] = None

        # --- Holistic judge (privacy / quality / style) ---
        entity_rows, judge_failed = self._run_final_judge(
            entity_rows,
            model_configs=model_configs,
            selected_models=selected_models,
            privacy_goal=privacy_goal,
            preview_num_records=preview_num_records,
        )
        all_failed.extend(judge_failed)

        combined = merge_and_reorder(entity_rows, passthrough_rows)
        return RewriteResult(dataframe=combined, failed_records=all_failed)
