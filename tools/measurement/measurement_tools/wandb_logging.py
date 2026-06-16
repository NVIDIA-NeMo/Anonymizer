# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanitized W&B logging for benchmark measurement records."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from export_measurements import normalize_table, read_measurements

logger = logging.getLogger("measurement.wandb")

PRIVACY_BLOCKED_RECORD_TYPES = frozenset(
    {
        "dd_message_trace",
        "dd_task_trace",
    }
)

PRIVACY_ALLOWED_RECORD_TYPES = frozenset(
    {
        "run",
        "stage",
        "record",
        "evaluation_record",
        "ndd_workflow",
        "model_workflow",
        "dd_trace_coverage",
    }
)

PRIVACY_BLOCKED_FIELD_NAMES = frozenset(
    {
        "text",
        "text_replaced",
        "text_with_spans",
        "final_entities",
        "detected_entities",
        "seed_entities",
        "augmented_entities",
        "ground_truth_entities",
        "replacement_map",
        "messages",
        "prompt",
        "response",
        "reasoning_content",
        "content",
        "detection_invalid_entities",
        "type_fidelity_invalid_replacements",
        "relational_consistency_invalid_relations",
        "attribute_fidelity_invalid_entities",
        "_detection_judge",
        "_type_fidelity_judge",
        "_relational_consistency_judge",
        "_attribute_fidelity_judge",
        "api_key",
        "credential",
        "credentials",
        "model_providers",
        "model_configs",
        "password",
        "secret",
        "token",
    }
)

_SCALAR_METRIC_PREFIX = "measurement"
_STRING_SCALAR_FIELD_NAMES = frozenset(
    {
        "mode",
        "stage",
        "status",
        "strategy",
        "workflow_name",
    }
)
_MEAN_METRIC_FIELD_SUFFIXES = ("_score", "_rate", "_ratio", "_per_sec")


def summarize_benchmark_cases(cases: list[Any]) -> dict[str, float | int]:
    total = len(cases)
    completed = sum(getattr(case, "status").value == "completed" for case in cases)
    errored = sum(getattr(case, "status").value == "error" for case in cases)
    elapsed_values = [case.elapsed_sec for case in cases if case.elapsed_sec is not None]
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


def extract_scalar_metrics(record: dict[str, Any]) -> dict[str, float | int | bool | str]:
    """Return sanitized scalar metrics from one measurement record."""
    record_type = record.get("record_type")
    if not isinstance(record_type, str):
        return {}
    if record_type in PRIVACY_BLOCKED_RECORD_TYPES:
        return {}
    if record_type not in PRIVACY_ALLOWED_RECORD_TYPES:
        return {}

    metrics: dict[str, float | int | bool | str] = {}
    for key, value in record.items():
        if _column_is_blocked(key):
            continue
        scalar = _coerce_scalar(value, field_name=key)
        if scalar is None:
            continue
        metrics[f"{_SCALAR_METRIC_PREFIX}/{record_type}/{key}"] = scalar
    return metrics


def aggregate_measurement_scalars(records: list[dict[str, Any]]) -> dict[str, float | int | bool | str]:
    """Aggregate sanitized scalar metrics across measurement records."""
    aggregated: dict[str, float | int | bool | str] = {}
    numeric_sums: dict[str, float] = {}
    numeric_means: dict[str, list[float]] = {}
    for record in records:
        for key, value in extract_scalar_metrics(record).items():
            if isinstance(value, bool | str):
                aggregated[key] = value
                continue
            numeric_value = float(value)
            if _metric_uses_mean_aggregation(key):
                mean_key = f"{key}_mean"
                numeric_means.setdefault(mean_key, []).append(numeric_value)
            else:
                numeric_sums[key] = numeric_sums.get(key, 0.0) + numeric_value
    for key, total in numeric_sums.items():
        aggregated[key] = total
    for key, values in numeric_means.items():
        aggregated[key] = sum(values) / len(values)
    aggregated[f"{_SCALAR_METRIC_PREFIX}/record_count"] = len(records)
    return aggregated


def _metric_uses_mean_aggregation(metric_name: str) -> bool:
    field_name = metric_name.rsplit("/", maxsplit=1)[-1]
    return field_name.endswith(_MEAN_METRIC_FIELD_SUFFIXES)


def log_benchmark_measurements(
    wandb: Any,
    *,
    measurement_path: Path,
    cases: list[Any],
    log_tables: bool,
    table_dir: Path | None,
) -> None:
    """Log sanitized benchmark summaries to the active W&B run."""
    if wandb.run is None:
        return

    metrics = summarize_benchmark_cases(cases)
    if measurement_path.exists() and measurement_path.stat().st_size > 0:
        records = _read_measurement_records(measurement_path)
        metrics.update(aggregate_measurement_scalars(records))
        _assert_privacy_guardrails(metrics)
        if log_tables:
            _log_sanitized_tables(wandb, records)
    elif table_dir is not None and log_tables:
        _log_sanitized_tables_from_dir(wandb, table_dir)

    try:
        wandb.log(metrics)
        _update_wandb_summary(wandb, metrics)
    except Exception as exc:  # noqa: BLE001 -- observability is best-effort
        logger.warning("Failed to log benchmark measurements to W&B: %s", exc)


def _update_wandb_summary(wandb: Any, metrics: dict[str, Any]) -> None:
    summary = getattr(wandb, "summary", None)
    update = getattr(summary, "update", None)
    if callable(update):
        update(metrics)


def _read_measurement_records(path: Path) -> list[dict[str, Any]]:
    dataframe = read_measurements(path)
    return [row for row in dataframe.to_dict(orient="records") if isinstance(row, dict)]


def _coerce_scalar(value: Any, *, field_name: str) -> float | int | bool | str | None:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, str):
        if field_name not in _STRING_SCALAR_FIELD_NAMES:
            return None
        if len(value) > 256 or "://" in value or value.startswith("/"):
            return None
        return value
    return None


def _log_sanitized_tables(wandb: Any, records: list[dict[str, Any]]) -> None:
    dataframe = pd.DataFrame(records)
    if "record_type" not in dataframe.columns:
        return
    for record_type, rows in dataframe.groupby("record_type", sort=False):
        if record_type in PRIVACY_BLOCKED_RECORD_TYPES:
            continue
        if record_type not in PRIVACY_ALLOWED_RECORD_TYPES:
            continue
        table = normalize_table(rows)
        table = _drop_blocked_columns(table)
        if table.empty:
            continue
        wandb.log({f"measurement_table/{record_type}": wandb.Table(dataframe=table)})


def _log_sanitized_tables_from_dir(wandb: Any, table_dir: Path) -> None:
    if not table_dir.is_dir():
        return
    for parquet_path in sorted(table_dir.glob("*.parquet")):
        record_type = parquet_path.stem
        if record_type in PRIVACY_BLOCKED_RECORD_TYPES:
            continue
        if record_type not in PRIVACY_ALLOWED_RECORD_TYPES:
            continue
        table = _drop_blocked_columns(pd.read_parquet(parquet_path))
        if table.empty:
            continue
        wandb.log({f"measurement_table/{record_type}": wandb.Table(dataframe=table)})


def _drop_blocked_columns(table: pd.DataFrame) -> pd.DataFrame:
    blocked = {column for column in table.columns if _column_is_blocked(column)}
    if not blocked:
        return table
    return table.drop(columns=sorted(blocked), errors="ignore")


def _column_is_blocked(column: str) -> bool:
    parts = _field_name_parts(column)
    return any(part in PRIVACY_BLOCKED_FIELD_NAMES or _part_is_path_like(part) for part in parts)


def _field_name_parts(field_name: str) -> list[str]:
    parts: list[str] = []
    for raw_part in field_name.replace("[", ".").replace("]", "").split("."):
        if not raw_part:
            continue
        normalized = raw_part.lower().replace("-", "_")
        parts.append(normalized)
        parts.extend(part for part in normalized.split("_") if part)
    return parts


def _part_is_path_like(part: str) -> bool:
    return part in {"path", "url"} or part.endswith(("_path", "_url"))


def _assert_privacy_guardrails(metrics: dict[str, Any]) -> None:
    for key in metrics:
        field_name = key.rsplit("/", maxsplit=1)[-1]
        if _column_is_blocked(field_name):
            raise ValueError(f"W&B payload guardrail violated: blocked field in metric {key!r}")
    for record_type in PRIVACY_BLOCKED_RECORD_TYPES:
        blocked_segment = f"/{record_type}/"
        if any(blocked_segment in key for key in metrics):
            raise ValueError(f"W&B payload guardrail violated: blocked record type {record_type!r} present")
