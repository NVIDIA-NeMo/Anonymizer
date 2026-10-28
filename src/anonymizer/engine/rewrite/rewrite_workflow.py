# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import pyarrow as pa
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
    RECORD_ID_COLUMN,
)
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.graph_core.stages import run_stages
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
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
    align_on_record_ids,
    merge_and_reorder,
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
    """Copy columns from source into target using positional alignment.

    When the adapter drops rows (tracked via ``failed_records``), source
    will be shorter than target. We align on ``RECORD_ID_COLUMN`` to
    drop the failed rows from target, then join the new columns from
    source onto the surviving rows.

    Set ``overwrite=True`` to update existing columns that the workflow
    produced (e.g. re-evaluate after repair). Seed columns passed through
    the adapter are excluded from overwrite to prevent re-serialization artifacts.
    """
    del source_positions  # accepted for caller compatibility; main's contract aligns on RECORD_ID_COLUMN
    if len(source) != len(target):
        dropped = len(target) - len(source)
        logger.warning(
            "Row count mismatch: target=%d, source=%d; dropping %d failed row(s).",
            len(target),
            len(source),
            dropped,
        )
        target, source = align_on_record_ids(target, source, RECORD_ID_COLUMN)

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
# Kernel sequence
# ---------------------------------------------------------------------------


@dataclass
class _Rows:
    """The rows a sequence of stages works on, and the failures they have reported."""

    frame: pd.DataFrame
    failed: list[FailedRecord] = field(default_factory=list)
    passing: pd.DataFrame = field(default_factory=pd.DataFrame)
    failing: pd.DataFrame = field(default_factory=pd.DataFrame)
    converged: bool = False


def _replacement_unavailable(df: pd.DataFrame) -> pd.Series:
    if COL_REWRITE_REPLACEMENT_READY in df.columns:
        return ~df[COL_REWRITE_REPLACEMENT_READY].apply(bool)
    return pd.Series(False, index=df.index)


def _withhold_rewritten_text(df: pd.DataFrame) -> None:
    """Null the rewritten text of rows whose replacements could not be applied.

    When every rewrite in a DataDesigner batch is withheld, the rewritten column arrives as ``null[pyarrow]``
    and a masked assignment into it aborts the process inside pyarrow. That column holds only nulls, so give it
    the Arrow string dtype a partially withheld batch has: that of the batch's source text.
    """
    column = df[COL_REWRITTEN_TEXT]
    if isinstance(column.dtype, pd.ArrowDtype) and pa.types.is_null(column.dtype.pyarrow_dtype):
        text = df[COL_TEXT].dtype if COL_TEXT in df.columns else None
        dtype = (
            text
            if isinstance(text, pd.ArrowDtype)
            and (pa.types.is_string(text.pyarrow_dtype) or pa.types.is_large_string(text.pyarrow_dtype))
            else pd.ArrowDtype(pa.string())
        )
        df[COL_REWRITTEN_TEXT] = pd.Series(pd.array([None] * len(df), dtype=dtype), index=df.index)
        return


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
            entity_rows, passthrough_rows = split_rows(dataframe, column=COL_ENTITIES_BY_VALUE, predicate=_has_entities)
            measurement.update(
                entity_row_count=len(entity_rows),
                passthrough_row_count=len(passthrough_rows),
            )

            # Fast path: no entities anywhere
            if entity_rows.empty:
                _apply_passthrough_defaults(passthrough_rows)
                result = RewriteResult(
                    dataframe=merge_and_reorder(passthrough_rows),
                    failed_records=[],
                    source_positions=tuple(sorted(passthrough_rows[ROW_ORDER_COL])),
                )
                measurement.update(output_row_count=len(result.dataframe), failed_record_count=0)
                return result

            rows = _Rows(entity_rows)
            run_stages(
                "rewrite",
                [
                    (
                        "replacement-map",
                        lambda: self._replacement_map(rows, model_configs, replace_model_selection),
                    ),
                    (
                        "pipeline",
                        lambda: self._pipeline(
                            rows,
                            model_configs=model_configs,
                            selected_models=selected_models,
                            privacy_goal=privacy_goal,
                            data_summary=data_summary,
                            preview_num_records=preview_num_records,
                            strict_entity_protection=strict_entity_protection,
                        ),
                    ),
                    *self._evaluate_repair_stages(
                        rows,
                        model_configs=model_configs,
                        selected_models=selected_models,
                        privacy_goal=privacy_goal,
                        evaluation=evaluation,
                        preview_num_records=preview_num_records,
                    ),
                ],
            )

            # --- Merge and return ---
            _apply_passthrough_defaults(passthrough_rows)
            result = RewriteResult(
                dataframe=merge_and_reorder(rows.frame, passthrough_rows),
                failed_records=rows.failed,
                source_positions=tuple(sorted([*rows.frame[ROW_ORDER_COL], *passthrough_rows[ROW_ORDER_COL]])),
            )
            measurement.update(
                output_row_count=len(result.dataframe),
                failed_record_count=len(result.failed_records),
            )
            return result

    # --- Step 1: replacement map (needs only detection output) ---
    def _replacement_map(
        self, rows: _Rows, model_configs: list[ModelConfig], replace_model_selection: ReplaceModelSelection
    ) -> None:
        replace_result = LlmReplaceWorkflow(adapter=self._adapter).generate_map_only(
            rows.frame,
            model_configs=model_configs,
            selected_models=replace_model_selection,
        )
        rows.frame = _join_new_columns(
            rows.frame, replace_result.dataframe, source_positions=replace_result.source_positions
        )
        rows.failed.extend(replace_result.failed_records)

    # --- Step 2: domain, disposition, QA, rewrite (single adapter call) ---
    def _pipeline(
        self,
        rows: _Rows,
        *,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        data_summary: str | None,
        preview_num_records: int | None,
        strict_entity_protection: bool,
    ) -> None:
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
        pipeline_seed = select_seed_cols(rows.frame, derive_seed_columns(pipeline_columns, rows.frame))
        pipeline_result = self._adapter.run_workflow(
            pipeline_seed,
            model_configs=model_configs,
            columns=pipeline_columns,
            workflow_name="rewrite-pipeline",
            preview_num_records=preview_num_records,
        )
        restore_empty_skipped_span_label_counts(pipeline_result.dataframe)
        rows.frame = _join_new_columns(
            rows.frame, pipeline_result.dataframe, source_positions=pipeline_result.source_positions
        )
        rows.failed.extend(pipeline_result.failed_records)

    # ---------------------------------------------------------------------------
    # Evaluate-repair loop
    # ---------------------------------------------------------------------------

    def _evaluate_repair_stages(
        self,
        rows: _Rows,
        *,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        evaluation: EvaluationCriteria,
        preview_num_records: int | None,
    ) -> list[tuple[str, Callable[[], None]]]:
        """The initial evaluate, then one repair and one re-evaluate per allowed iteration, then the review flags.

        Every iteration is a stage that does nothing once no row needs repair, so the loop runs at most
        ``max_repair_iterations`` repairs and the initial evaluate runs even when that is 0.
        """
        eval_columns = self._evaluate_wf.columns(selected_models=selected_models, evaluation=evaluation)
        repair_columns = self._repair_wf.columns(
            selected_models=selected_models,
            privacy_goal=privacy_goal,
            effective_threshold=evaluation.repair_threshold,
        )

        def run_columns(df: pd.DataFrame, columns: list, name: str, *, overwrite: bool = False) -> pd.DataFrame:
            seed_cols = derive_seed_columns(columns, df)
            result = self._adapter.run_workflow(
                select_seed_cols(df, seed_cols),
                model_configs=model_configs,
                columns=columns,
                workflow_name=name,
                preview_num_records=preview_num_records,
            )
            rows.failed.extend(result.failed_records)
            return _join_new_columns(
                df, result.dataframe, overwrite=overwrite, seed_cols=seed_cols, source_positions=result.source_positions
            )

        def evaluate(df: pd.DataFrame, iteration: int) -> pd.DataFrame:
            df = run_columns(df, eval_columns, f"rewrite-evaluate-{iteration}", overwrite=True)
            _normalize_evaluation_payloads(df)
            return df

        def initial_evaluate() -> None:
            if COL_REPAIR_ITERATIONS not in rows.frame.columns:
                rows.frame[COL_REPAIR_ITERATIONS] = 0
            rows.frame = evaluate(rows.frame, 0)
            _withhold_rewritten_text(rows.frame)

        def repair(iteration: int) -> None:
            if rows.converged:
                return
            df = rows.frame
            needs_repair_mask = df[COL_NEEDS_REPAIR].apply(bool)
            if not needs_repair_mask.any():
                logger.info("Evaluate-repair loop: all rows pass at iteration %d", iteration)
                rows.converged = True
                return
            rows.failing, rows.passing = df[needs_repair_mask].copy(), df[~needs_repair_mask].copy()
            logger.info(
                "Evaluate-repair loop iteration %d: %d/%d rows need repair",
                iteration,
                len(rows.failing),
                len(df),
            )
            # Repair failing rows only
            repaired = run_columns(rows.failing, repair_columns, f"rewrite-repair-{iteration}")
            if COL_REWRITTEN_TEXT_NEXT in repaired.columns:
                repaired[COL_REWRITTEN_TEXT] = repaired[COL_REWRITTEN_TEXT_NEXT]
            repaired[COL_REPAIR_ITERATIONS] = repaired[COL_REPAIR_ITERATIONS].apply(lambda x: int(x) + 1)
            rows.failing = repaired

        def reevaluate(iteration: int) -> None:
            if rows.converged:
                return
            # Re-evaluate only repaired rows (passing rows' metrics are unchanged)
            rows.frame = pd.concat([rows.passing, evaluate(rows.failing, iteration + 1)], ignore_index=True)

        def flag_for_review() -> None:
            # Compute needs_human_review from objective metrics after the loop exhausts.
            df = rows.frame
            _withhold_rewritten_text(df)
            needs_review = (
                df[COL_REWRITTEN_TEXT].isna() | _replacement_unavailable(df) | df[COL_ANY_HIGH_LEAKED].apply(bool)
            )
            if evaluation.flag_utility_below is not None:
                needs_review = needs_review | (df[COL_UTILITY_SCORE].apply(float) < evaluation.flag_utility_below)
            if evaluation.flag_leakage_above is not None:
                needs_review = needs_review | (df[COL_LEAKAGE_MASS].apply(float) > evaluation.flag_leakage_above)
            df[COL_NEEDS_HUMAN_REVIEW] = needs_review

        stages: list[tuple[str, Callable[[], None]]] = [("evaluate-0", initial_evaluate)]
        for iteration in range(evaluation.max_repair_iterations):
            stages.append((f"repair-{iteration}", lambda i=iteration: repair(i)))
            stages.append((f"evaluate-{iteration + 1}", lambda i=iteration: reevaluate(i)))
        stages.append(("flag-for-review", flag_for_review))
        return stages

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
            logger.debug("Rewrite judge workflow failed; scores may be unavailable.")
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

        rows = _Rows(entity_rows)

        # --- Detection validity judge (opt-in) ---
        def detection_judge() -> None:
            try:
                detection_result = self._detection_judge_wf.evaluate(
                    rows.frame,
                    model_configs=model_configs,
                    selected_models=selected_models,
                    preview_num_records=preview_num_records,
                )
                rows.frame = _join_preserving(
                    rows.frame,
                    detection_result.dataframe,
                    defaults={COL_DETECTION_VALID: None, COL_DETECTION_INVALID_ENTITIES: None},
                    source_positions=detection_result.source_positions,
                )
                rows.failed.extend(detection_result.failed_records)
                # Convert bool all_valid → 0–1 fraction so detection validity sits on the
                # same scale as utility_score and leakage_mass in the rewrite scores section.
                rows.frame[COL_DETECTION_VALID] = rows.frame.apply(_detection_valid_fraction, axis=1)
            except Exception:
                logger.debug("Detection validity judge failed; scores may be unavailable.")
                rows.frame[COL_DETECTION_VALID] = None
                rows.frame[COL_DETECTION_INVALID_ENTITIES] = None

        # --- Holistic judge (privacy / quality / style) ---
        def final_judge() -> None:
            rows.frame, judge_failed = self._run_final_judge(
                rows.frame,
                model_configs=model_configs,
                selected_models=selected_models,
                privacy_goal=privacy_goal,
                preview_num_records=preview_num_records,
            )
            rows.failed.extend(judge_failed)

        stages: list[tuple[str, Callable[[], None]]] = [("final-judge", final_judge)]
        if compute_detection_validity:
            stages.insert(0, ("detection-judge", detection_judge))
        run_stages("rewrite-evaluate", stages)

        combined = merge_and_reorder(rows.frame, passthrough_rows)
        return RewriteResult(dataframe=combined, failed_records=rows.failed)
