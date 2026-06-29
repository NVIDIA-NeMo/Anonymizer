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

from anonymizer.engine import constants as engine_constants

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

_ENGINE_INTERNAL_FIELD_NAMES = frozenset(
    value
    for name, value in vars(engine_constants).items()
    if name.startswith("COL_") and isinstance(value, str) and value.startswith("_")
)

_ENGINE_RAW_OUTPUT_FIELD_NAMES = frozenset(
    {
        engine_constants.COL_FINAL_ENTITIES,
        engine_constants.COL_DETECTION_INVALID_ENTITIES,
        engine_constants.COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
        engine_constants.COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
        engine_constants.COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    }
)

PRIVACY_BLOCKED_FIELD_NAMES = (
    _ENGINE_INTERNAL_FIELD_NAMES
    | _ENGINE_RAW_OUTPUT_FIELD_NAMES
    | frozenset(
        {
            "text",
            "text_replaced",
            "text_with_spans",
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
)

_SCALAR_METRIC_PREFIX = "measurement"
_STRING_SCALAR_FIELD_NAMES = frozenset(
    {
        "mode",
        "stage",
        "status",
        "strategy",
    }
)
_ALLOWED_DIMENSION_VALUES = {
    "mode": frozenset({"replace", "rewrite"}),
    "stage": frozenset(
        {
            "Anonymizer._run_internal",
            "EntityDetectionWorkflow.run",
            "ReplacementWorkflow.run",
            "RewriteWorkflow.run",
        }
    ),
    "status": frozenset({"completed", "error"}),
    "strategy": frozenset({"annotate", "hash", "redact", "rewrite", "substitute"}),
}
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
_TABLE_NUMERIC_FIELDS = _SUM_METRIC_FIELD_NAMES | _MEAN_METRIC_FIELD_NAMES | _TRACE_COVERAGE_COUNT_FIELDS
_TABLE_RECORD_NUMERIC_FIELDS = _TABLE_NUMERIC_FIELDS - {"text_length_chars", "text_length_tokens"}
_TABLE_FIELDS_BY_RECORD_TYPE = {
    "run": frozenset({"record_type", "mode", "strategy"}) | _TABLE_NUMERIC_FIELDS,
    "stage": frozenset({"record_type", "stage", "status"}) | _TABLE_NUMERIC_FIELDS,
    "record": frozenset(
        {
            "record_type",
            "mode",
            "strategy",
            "row_index",
            "record_hash",
            "text_length_chars_bucket",
            "text_length_tokens_bucket",
        }
    )
    | _TABLE_RECORD_NUMERIC_FIELDS,
    "evaluation_record": frozenset({"record_type", "mode", "strategy", "row_index", "record_hash"})
    | _TABLE_RECORD_NUMERIC_FIELDS
    | _EVALUATION_BOOL_FIELDS,
    "ndd_workflow": frozenset({"record_type", "status"}) | _TABLE_NUMERIC_FIELDS,
    "model_workflow": frozenset({"record_type", "status"}) | _TABLE_NUMERIC_FIELDS,
    "dd_trace_coverage": frozenset({"record_type"}) | _TRACE_COVERAGE_COUNT_FIELDS,
}


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
            field_name = key.rsplit("/", maxsplit=1)[-1]
            if isinstance(value, bool | str):
                if field_name in _LAST_METRIC_FIELD_NAMES:
                    aggregated[key] = value
                continue
            numeric_value = float(value)
            if field_name in _MEAN_METRIC_FIELD_NAMES:
                mean_key = f"{key}_mean"
                numeric_means.setdefault(mean_key, []).append(numeric_value)
            elif field_name in _SUM_METRIC_FIELD_NAMES:
                numeric_sums[key] = numeric_sums.get(key, 0.0) + numeric_value
    for key, total in numeric_sums.items():
        aggregated[key] = total
    for key, values in numeric_means.items():
        aggregated[key] = sum(values) / len(values)
    aggregated[f"{_SCALAR_METRIC_PREFIX}/record_count"] = len(records)
    return aggregated


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

    try:
        metrics = summarize_benchmark_cases(cases)
        tables: dict[str, Any] = {}
        if measurement_path.exists() and measurement_path.stat().st_size > 0:
            records = _read_measurement_records(measurement_path)
            metrics.update(aggregate_measurement_scalars(records))
            _assert_privacy_guardrails(metrics)
            if log_tables:
                tables = _build_sanitized_tables(wandb, records)
        elif table_dir is not None and log_tables:
            tables = _build_sanitized_tables_from_dir(wandb, table_dir)
        wandb.log({**metrics, **tables})
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
        return _safe_dimension_value(field_name, value)
    return None


def _log_sanitized_tables(wandb: Any, records: list[dict[str, Any]]) -> None:
    tables = _build_sanitized_tables(wandb, records)
    if tables:
        wandb.log(tables)


def _build_sanitized_tables(wandb: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    dataframe = pd.DataFrame(records)
    if "record_type" not in dataframe.columns:
        return {}
    tables: dict[str, Any] = {}
    for record_type, rows in dataframe.groupby("record_type", sort=False):
        if record_type in PRIVACY_BLOCKED_RECORD_TYPES:
            continue
        if record_type not in PRIVACY_ALLOWED_RECORD_TYPES:
            continue
        table = normalize_table(rows)
        table = _project_table(table, record_type=record_type)
        if table.empty or list(table.columns) == ["record_type"]:
            continue
        tables[f"measurement_table/{record_type}"] = wandb.Table(dataframe=table)
    return tables


def _log_sanitized_tables_from_dir(wandb: Any, table_dir: Path) -> None:
    tables = _build_sanitized_tables_from_dir(wandb, table_dir)
    if tables:
        wandb.log(tables)


def _build_sanitized_tables_from_dir(wandb: Any, table_dir: Path) -> dict[str, Any]:
    if not table_dir.is_dir():
        return {}
    tables: dict[str, Any] = {}
    for parquet_path in sorted(table_dir.glob("*.parquet")):
        record_type = parquet_path.stem
        if record_type in PRIVACY_BLOCKED_RECORD_TYPES:
            continue
        if record_type not in PRIVACY_ALLOWED_RECORD_TYPES:
            continue
        table = _project_table(pd.read_parquet(parquet_path), record_type=record_type)
        if table.empty or list(table.columns) == ["record_type"]:
            continue
        tables[f"measurement_table/{record_type}"] = wandb.Table(dataframe=table)
    return tables


def _project_table(table: pd.DataFrame, *, record_type: str) -> pd.DataFrame:
    if any(not isinstance(column, str) for column in table.columns):
        raise ValueError(f"W&B table {record_type!r} contains a non-string column name")
    allowed = _TABLE_FIELDS_BY_RECORD_TYPE.get(record_type, frozenset())
    projected = table.loc[:, [column for column in table.columns if column in allowed]].copy()
    for field_name in _STRING_SCALAR_FIELD_NAMES & set(projected.columns):
        projected[field_name] = projected[field_name].map(
            lambda value: _safe_dimension_value(field_name, value) if isinstance(value, str) else None
        )
    for field_name in {"text_length_chars_bucket", "text_length_tokens_bucket"} & set(projected.columns):
        projected[field_name] = projected[field_name].map(_safe_size_bucket)
    if "record_hash" in projected.columns:
        projected["record_hash"] = projected["record_hash"].map(_safe_record_hash)
    return projected.dropna(axis="columns", how="all")


def _safe_dimension_value(field_name: str, value: str) -> str | None:
    allowed = _ALLOWED_DIMENSION_VALUES.get(field_name)
    if allowed is None:
        return None
    comparison = value.casefold() if field_name == "strategy" else value
    return value if comparison in allowed else None


def _safe_size_bucket(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value if value in {"0", "1-127", "128-511", "512-2047", "2048-8191", "8192+"} else None


def _safe_record_hash(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    return value if all(character in "0123456789abcdef" for character in value.lower()) else None


def _column_is_blocked(column: str) -> bool:
    normalized_column = column.lower().replace("-", "_")
    if normalized_column == "run_tags" or normalized_column.startswith(("run_tags.", "run_tags[")):
        return True
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
