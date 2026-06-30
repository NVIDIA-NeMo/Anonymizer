# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanitized W&B logging for benchmark measurement records."""

from __future__ import annotations

import logging
import math
from typing import Any

from measurement_tools.wandb_ingress import MeasurementRecord, MeasurementSnapshot
from measurement_tools.wandb_models import (
    EvaluationTableRow,
    ModelWorkflowTableRow,
    NddWorkflowTableRow,
    RecordTableRow,
    RunTableRow,
    StageTableRow,
    TraceCoverageTableRow,
    WandbHistoryPayload,
    WandbSummaryPayload,
    WandbTablePayload,
)

logger = logging.getLogger("measurement.wandb")

_SCALAR_METRIC_PREFIX = "measurement"
_STRING_SCALAR_FIELD_NAMES = frozenset(
    {
        "mode",
        "stage",
        "status",
        "strategy",
    }
)
_LAST_METRIC_FIELD_NAMES = _STRING_SCALAR_FIELD_NAMES
_SUM_METRIC_FIELD_NAMES = frozenset(
    {
        "elapsed_sec",
        "input_row_count",
        "seed_row_count",
        "output_row_count",
        "failed_record_count",
        "preview_num_records",
        "column_count",
        "observed_input_tokens",
        "observed_output_tokens",
        "observed_total_tokens",
        "observed_reasoning_tokens",
        "observed_successful_requests",
        "observed_failed_requests",
        "observed_total_requests",
        "text_length_chars",
        "text_length_tokens",
        "final_entity_count",
        "ground_truth_entity_count",
        "entity_true_positive_count",
        "entity_false_positive_count",
        "entity_false_negative_count",
        "entity_relaxed_gt_found_count",
        "entity_relaxed_detected_tp_count",
        "entity_relaxed_label_compatible_gt_found_count",
        "entity_relaxed_label_compatible_detected_tp_count",
        "replacement_count",
        "replacement_duplicate_value_count",
        "replacement_missing_final_entity_count",
        "replacement_missing_final_value_count",
        "replacement_synthetic_original_collision_count",
        "replacement_synthetic_original_collision_value_count",
        "original_value_leak_count",
        "detected_candidate_count",
        "validation_chunk_count",
        "llm_calls_estimated_total",
        "detection_invalid_entity_count",
        "type_fidelity_invalid_replacement_count",
        "relational_consistency_invalid_relation_count",
        "attribute_fidelity_invalid_entity_count",
    }
)
_MEAN_METRIC_FIELD_NAMES = frozenset(
    {
        "observed_failed_request_rate",
        "input_rows_per_sec",
        "output_rows_per_sec",
        "observed_tokens_per_sec",
        "observed_requests_per_sec",
        "observed_tokens_per_successful_request",
        "entity_precision",
        "entity_recall",
        "entity_f1",
        "utility_score",
        "leakage_mass",
        "weighted_leakage_rate",
        "repair_iterations",
    }
)
_EVALUATION_BOOL_FIELDS = frozenset(
    {"detection_valid", "type_fidelity_valid", "relational_consistency_valid", "attribute_fidelity_valid"}
)
_TRACE_COVERAGE_COUNT_FIELDS = frozenset(
    {"traced_column_count", "native_trace_column_count", "private_trace_column_count", "unsupported_column_count"}
)


def summarize_benchmark_cases(cases: list[Any]) -> dict[str, float | int]:
    total = len(cases)
    completed = sum(getattr(case, "status").value == "completed" for case in cases)
    errored = sum(getattr(case, "status").value == "error" for case in cases)
    elapsed_values = [case.elapsed_sec for case in cases if case.elapsed_sec is not None]
    if any(not math.isfinite(value) or value < 0 for value in elapsed_values):
        raise ValueError("benchmark case elapsed time must be finite and non-negative")
    metrics: dict[str, float | int] = {
        "benchmark/case_total": total,
        "benchmark/case_completed": completed,
        "benchmark/case_errored": errored,
        "benchmark/case_success_rate": (completed / total) if total else 0.0,
    }
    if elapsed_values:
        metrics["benchmark/case_elapsed_sec_sum"] = float(sum(elapsed_values))
        metrics["benchmark/case_elapsed_sec_mean"] = float(sum(elapsed_values) / len(elapsed_values))
    return metrics


def build_outbound_measurements(
    snapshot: MeasurementSnapshot,
    *,
    cases: list[Any],
    log_tables: bool,
) -> tuple[WandbHistoryPayload, WandbSummaryPayload, tuple[WandbTablePayload, ...]]:
    """Build and validate every SDK-bound measurement value."""
    metrics: dict[str, float | int | bool | str] = dict(summarize_benchmark_cases(cases))
    metrics.update(aggregate_measurement_scalars(snapshot.records))
    history = WandbHistoryPayload(metrics=metrics)
    summary = WandbSummaryPayload(metrics=metrics)
    tables = _typed_tables(snapshot.records) if log_tables else ()
    return history, summary, tables


_TABLE_ROW_MODELS = {
    "run": RunTableRow,
    "stage": StageTableRow,
    "record": RecordTableRow,
    "evaluation_record": EvaluationTableRow,
    "ndd_workflow": NddWorkflowTableRow,
    "model_workflow": ModelWorkflowTableRow,
    "dd_trace_coverage": TraceCoverageTableRow,
}


def _typed_tables(records: tuple[MeasurementRecord, ...]) -> tuple[WandbTablePayload, ...]:
    rows_by_type: dict[str, list[Any]] = {}
    for record in records:
        row_model = _TABLE_ROW_MODELS[record.record_type]
        projected = {
            field: getattr(record, field)
            for field in row_model.model_fields
            if hasattr(record, field)
            and getattr(record, field) is not None
            and field not in {"text_length_chars", "text_length_tokens"}
        }
        row = row_model.model_validate(projected)
        rows_by_type.setdefault(record.record_type, []).append(row)
    return tuple(
        WandbTablePayload(record_type=record_type, rows=tuple(rows)) for record_type, rows in rows_by_type.items()
    )


def extract_scalar_metrics(record: MeasurementRecord) -> dict[str, float | int | bool | str]:
    """Return aggregate metrics from one already-validated measurement record."""
    metrics: dict[str, float | int | bool | str] = {}
    for field_name in (
        _LAST_METRIC_FIELD_NAMES | _SUM_METRIC_FIELD_NAMES | _MEAN_METRIC_FIELD_NAMES | _TRACE_COVERAGE_COUNT_FIELDS
    ):
        value = getattr(record, field_name, None)
        if value is not None:
            metrics[f"{_SCALAR_METRIC_PREFIX}/{record.record_type}/{field_name}"] = value
    return metrics


def aggregate_measurement_scalars(records: tuple[MeasurementRecord, ...]) -> dict[str, float | int | bool | str]:
    """Aggregate scalar metrics across strict measurement records."""
    aggregated: dict[str, float | int | bool | str] = {}
    numeric_sums: dict[str, float] = {}
    numeric_means: dict[str, list[float]] = {}
    for record in records:
        for key, value in extract_scalar_metrics(record).items():
            field_name = key.rsplit("/", maxsplit=1)[-1]
            if isinstance(value, bool | str):
                if field_name in _LAST_METRIC_FIELD_NAMES:
                    aggregated[key] = value
                continue
            numeric_value = float(value)
            if field_name in _MEAN_METRIC_FIELD_NAMES:
                mean_key = f"{key}_mean"
                numeric_means.setdefault(mean_key, []).append(numeric_value)
            elif field_name in _SUM_METRIC_FIELD_NAMES | _TRACE_COVERAGE_COUNT_FIELDS:
                numeric_sums[key] = numeric_sums.get(key, 0.0) + numeric_value
    for key, total in numeric_sums.items():
        aggregated[key] = total
    for key, values in numeric_means.items():
        aggregated[key] = sum(values) / len(values)
    aggregated[f"{_SCALAR_METRIC_PREFIX}/record_count"] = len(records)
    return aggregated
