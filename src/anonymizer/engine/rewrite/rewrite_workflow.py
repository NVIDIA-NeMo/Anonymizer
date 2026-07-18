# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
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
    COL_REPAIR_ITERATIONS,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
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


def _join_preserving(
    target: pd.DataFrame,
    source: pd.DataFrame,
    defaults: dict[str, object],
) -> pd.DataFrame:
    """Left-join source columns onto target, preserving all target rows.

    When source returns fewer rows than target (e.g. a judge timed out or the
    NDD adapter dropped a row), missing rows receive the values in ``defaults``
    instead of being removed from the result.  Used for non-critical/diagnostic
    steps where losing a row from the output is worse than surfacing a null.
    """
    if len(source) == len(target):
        return _join_new_columns(target, source)

    logger.warning(
        "Source returned %d of %d rows; defaulting %d missing row(s).",
        len(source),
        len(target),
        len(target) - len(source),
    )
    result = target.copy()
    source_by_id = source.set_index(RECORD_ID_COLUMN)
    known_ids = set(source_by_id.index)

    for col, default in defaults.items():
        result[col] = default

    for idx, record_id in result[RECORD_ID_COLUMN].items():
        if record_id in known_ids:
            row = source_by_id.loc[record_id]
            for col in defaults:
                result.at[idx, col] = row.get(col)

    return result


def _join_judge_columns(target: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
    """Merge holistic judge columns preserving all rows.

    Thin wrapper around ``_join_preserving`` with the holistic-judge defaults.
    ``COL_NEEDS_HUMAN_REVIEW`` is not touched here; it is set by the
    evaluate-repair loop based on objective metrics.
    """
    return _join_preserving(target, source, defaults={COL_JUDGE_EVALUATION: None})


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
    and the evaluate-repair loop. The final judge runs separately
    via evaluate().
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
                result = RewriteResult(dataframe=result_df, failed_records=all_failed)
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
            entity_rows = _join_new_columns(entity_rows, replace_result.dataframe)
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
            )
            entity_rows = _join_new_columns(entity_rows, pipeline_result.dataframe)
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
            result = RewriteResult(dataframe=combined, failed_records=all_failed)
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

        if COL_REPAIR_ITERATIONS not in df.columns:
            df[COL_REPAIR_ITERATIONS] = 0

        eval_columns = self._evaluate_wf.columns(
            selected_models=selected_models,
            evaluation=evaluation,
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
            effective_threshold=evaluation.repair_threshold,
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

        # Compute needs_human_review from objective metrics after the loop exhausts.
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
            df = _join_judge_columns(df, judge_result.dataframe)
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
