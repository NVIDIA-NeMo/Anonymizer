# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanitized W&B logging for benchmark measurement records."""

from __future__ import annotations

import logging
import math
from typing import Any

from measurement_tools.wandb_ingress import MeasurementRecord, MeasurementSnapshot
from measurement_tools.wandb_metric_schema import (
    SCALAR_AGGREGATION_BY_FIELD,
    BenchmarkMetric,
    ScalarMetricAccumulator,
    metric_path,
)
from measurement_tools.wandb_models import (
    WANDB_TABLE_ROW_MODELS,
    WandbHistoryPayload,
    WandbSummaryPayload,
    WandbTablePayload,
)

logger = logging.getLogger("measurement.wandb")

_SCALAR_METRIC_PREFIX = "measurement"


def summarize_benchmark_cases(cases: list[Any]) -> dict[str, float | int]:
    total = len(cases)
    completed = sum(getattr(case, "status").value == "completed" for case in cases)
    errored = sum(getattr(case, "status").value == "error" for case in cases)
    elapsed_values = [case.elapsed_sec for case in cases if case.elapsed_sec is not None]
    if any(not math.isfinite(value) or value < 0 for value in elapsed_values):
        raise ValueError("benchmark case elapsed time must be finite and non-negative")
    metrics: dict[str, float | int] = {
        BenchmarkMetric.CASE_TOTAL: total,
        BenchmarkMetric.CASE_COMPLETED: completed,
        BenchmarkMetric.CASE_ERRORED: errored,
        BenchmarkMetric.CASE_SUCCESS_RATE: (completed / total) if total else 0.0,
    }
    if elapsed_values:
        metrics[BenchmarkMetric.CASE_ELAPSED_SEC_SUM] = float(sum(elapsed_values))
        metrics[BenchmarkMetric.CASE_ELAPSED_SEC_MEAN] = float(sum(elapsed_values) / len(elapsed_values))
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


def _typed_tables(records: tuple[MeasurementRecord, ...]) -> tuple[WandbTablePayload, ...]:
    rows_by_type: dict[str, list[Any]] = {}
    for record in records:
        row_model = WANDB_TABLE_ROW_MODELS[record.record_type]
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
    for field_name in SCALAR_AGGREGATION_BY_FIELD:
        value = getattr(record, field_name, None)
        if value is not None:
            namespace = metric_path(_SCALAR_METRIC_PREFIX, record.record_type)
            metrics[metric_path(namespace, field_name)] = value
    return metrics


def aggregate_measurement_scalars(records: tuple[MeasurementRecord, ...]) -> dict[str, float | int | bool | str]:
    """Aggregate scalar metrics across strict measurement records."""
    accumulator = ScalarMetricAccumulator()
    for record in records:
        accumulator.update(extract_scalar_metrics(record))
    aggregated = accumulator.metrics()
    aggregated[metric_path(_SCALAR_METRIC_PREFIX, "record_count")] = len(records)
    return aggregated
