# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-model request and token usage analysis for benchmark output."""

from __future__ import annotations

from typing import Any

import pandas as pd

from measurement_tools.benchmark_analysis_math import model_workflow_rows as _model_workflow_rows
from measurement_tools.benchmark_analysis_math import request_failure_rate as _request_failure_rate
from measurement_tools.benchmark_analysis_math import sum_int_or_none as _sum_int_or_none
from measurement_tools.benchmark_analysis_models import ModelUsageAnalysisRow, ModelUsageGroupAnalysisRow
from measurement_tools.stats import median_or_none as _median_or_none
from measurement_tools.stats import none_if_nan as _none_if_nan
from measurement_tools.stats import sum_int_or_zero as _sum_int_or_zero

_MODEL_USAGE_SUFFIXES = {
    ".request_usage.total_requests": "observed_total_requests",
    ".request_usage.successful_requests": "observed_successful_requests",
    ".request_usage.failed_requests": "observed_failed_requests",
    ".token_usage.input_tokens": "observed_input_tokens",
    ".token_usage.output_tokens": "observed_output_tokens",
    ".token_usage.total_tokens": "observed_total_tokens",
    ".token_usage.reasoning_tokens": "observed_reasoning_tokens",
}


_MODEL_USAGE_METADATA_SUFFIXES = {
    ".model_alias": "model_alias",
    ".model_name": "model_name",
    ".model_provider_name": "model_provider_name",
}


def build_model_usage_rows(measurements: pd.DataFrame) -> list[ModelUsageAnalysisRow]:
    model_rows = _model_workflow_rows(measurements)
    if model_rows.empty:
        return []
    usage_keys = model_usage_keys(model_rows.columns)
    rows: list[ModelUsageAnalysisRow] = []
    for _, measurement in model_rows.iterrows():
        data = measurement.to_dict()
        case_id = string_from_row(data, ["run_tags.case_id", "run_id"])
        run_id = string_from_row(data, ["run_id", "run_tags.case_id"])
        if case_id is None or run_id is None:
            continue
        for model_usage_key in usage_keys:
            usage = model_usage_metrics(data, model_usage_key)
            if not has_observed_model_usage(usage):
                continue
            metadata = model_usage_metadata(data, model_usage_key)
            rows.append(
                ModelUsageAnalysisRow(
                    suite_id=string_from_row(data, ["run_tags.suite_id"]),
                    workload_id=string_from_row(data, ["run_tags.workload_id"]),
                    config_id=string_from_row(data, ["run_tags.config_id"]),
                    experimental_detection_strategy=string_from_row(data, ["run_tags.experimental_detection_strategy"]),
                    experimental_replacement_strategy=string_from_row(
                        data, ["run_tags.experimental_replacement_strategy"]
                    ),
                    dd_parser_compat=string_from_row(data, ["run_tags.dd_parser_compat"]),
                    repetition=int_from_row(data, ["run_tags.repetition"]),
                    case_id=case_id,
                    run_id=run_id,
                    workflow_name=string_from_row(data, ["workflow_name"]),
                    model_alias=metadata.get("model_alias"),
                    model_name=metadata.get("model_name") or model_usage_key,
                    model_provider_name=metadata.get("model_provider_name"),
                    ndd_elapsed_sec=float_from_row(data, ["elapsed_sec"]),
                    **usage,
                )
            )
    return rows


def model_usage_keys(columns: pd.Index) -> list[str]:
    keys: set[str] = set()
    for column in columns:
        parsed = model_usage_column_parts(str(column))
        if parsed is not None:
            keys.add(parsed[0])
    return sorted(keys)


def model_usage_column_parts(column: str) -> tuple[str, str] | None:
    prefix = "model_usage."
    if not column.startswith(prefix):
        return None
    for suffix, metric in {**_MODEL_USAGE_SUFFIXES, **_MODEL_USAGE_METADATA_SUFFIXES}.items():
        if column.endswith(suffix):
            return column[len(prefix) : -len(suffix)], metric
    return None


def model_usage_metrics(data: dict[str, Any], model_usage_key: str) -> dict[str, int | float | None]:
    values: dict[str, int | float | None] = {
        "observed_total_requests": 0,
        "observed_successful_requests": 0,
        "observed_failed_requests": 0,
        "observed_input_tokens": 0,
        "observed_output_tokens": 0,
        "observed_total_tokens": 0,
        "observed_reasoning_tokens": None,
        "observed_failed_request_rate": None,
    }
    for suffix, metric in _MODEL_USAGE_SUFFIXES.items():
        value = data.get(f"model_usage.{model_usage_key}{suffix}")
        if value is None or pd.isna(value):
            continue
        values[metric] = coerce_int(value)
    values["observed_failed_request_rate"] = _request_failure_rate(
        failed=values["observed_failed_requests"],
        total=values["observed_total_requests"],
    )
    return values


def model_usage_metadata(data: dict[str, Any], model_usage_key: str) -> dict[str, str | None]:
    values: dict[str, str | None] = {
        "model_alias": None,
        "model_name": None,
        "model_provider_name": None,
    }
    for suffix, field_name in _MODEL_USAGE_METADATA_SUFFIXES.items():
        value = data.get(f"model_usage.{model_usage_key}{suffix}")
        if value is None or pd.isna(value):
            continue
        values[field_name] = str(value)
    return values


def has_observed_model_usage(usage: dict[str, int | float | None]) -> bool:
    return any(value not in (None, 0) for value in usage.values())


def string_from_row(data: dict[str, Any], columns: list[str]) -> str | None:
    for column in columns:
        value = data.get(column)
        if value is not None and not pd.isna(value):
            return str(value)
    return None


def int_from_row(data: dict[str, Any], columns: list[str]) -> int | None:
    value = string_from_row(data, columns)
    return int(float(value)) if value is not None else None


def float_from_row(data: dict[str, Any], columns: list[str]) -> float | None:
    value = string_from_row(data, columns)
    return float(value) if value is not None else None


def coerce_int(value: Any) -> int:
    return int(float(value))


def build_model_usage_group_rows(model_usage: list[ModelUsageAnalysisRow]) -> list[ModelUsageGroupAnalysisRow]:
    if not model_usage:
        return []
    table = pd.DataFrame([row.model_dump() for row in model_usage])
    rows: list[ModelUsageGroupAnalysisRow] = []
    group_columns = [
        "workload_id",
        "config_id",
        "experimental_detection_strategy",
        "experimental_replacement_strategy",
        "dd_parser_compat",
        "workflow_name",
        "model_alias",
        "model_name",
        "model_provider_name",
    ]
    for keys, group in table.groupby(group_columns, dropna=False):
        rows.append(build_model_usage_group_row(keys, group))
    return rows


def build_model_usage_group_row(keys: tuple[Any, ...], group: pd.DataFrame) -> ModelUsageGroupAnalysisRow:
    (
        workload_id,
        config_id,
        detection_strategy,
        replacement_strategy,
        dd_parser_compat,
        workflow_name,
        model_alias,
        model_name,
        provider_name,
    ) = keys
    reasoning_sum = _sum_int_or_none(group, "observed_reasoning_tokens")
    total_requests = _sum_int_or_zero(group, "observed_total_requests")
    failed_requests = _sum_int_or_zero(group, "observed_failed_requests")
    return ModelUsageGroupAnalysisRow(
        workload_id=_none_if_nan(workload_id),
        config_id=_none_if_nan(config_id),
        experimental_detection_strategy=_none_if_nan(detection_strategy),
        experimental_replacement_strategy=_none_if_nan(replacement_strategy),
        dd_parser_compat=_none_if_nan(dd_parser_compat),
        workflow_name=_none_if_nan(workflow_name),
        model_alias=_none_if_nan(model_alias),
        model_name=str(model_name),
        model_provider_name=_none_if_nan(provider_name),
        case_count=int(group["case_id"].nunique()),
        workflow_count=len(group),
        sum_observed_total_requests=total_requests,
        sum_observed_successful_requests=_sum_int_or_zero(group, "observed_successful_requests"),
        sum_observed_failed_requests=failed_requests,
        sum_observed_input_tokens=_sum_int_or_zero(group, "observed_input_tokens"),
        sum_observed_output_tokens=_sum_int_or_zero(group, "observed_output_tokens"),
        sum_observed_total_tokens=_sum_int_or_zero(group, "observed_total_tokens"),
        sum_observed_reasoning_tokens=reasoning_sum,
        observed_failed_request_rate=_request_failure_rate(failed=failed_requests, total=total_requests),
        median_observed_total_requests=_median_or_none(group, "observed_total_requests"),
        median_observed_failed_requests=_median_or_none(group, "observed_failed_requests"),
        median_observed_total_tokens=_median_or_none(group, "observed_total_tokens"),
    )


__all__ = [
    "build_model_usage_group_row",
    "build_model_usage_group_rows",
    "build_model_usage_rows",
    "coerce_int",
    "float_from_row",
    "has_observed_model_usage",
    "int_from_row",
    "model_usage_column_parts",
    "model_usage_keys",
    "model_usage_metadata",
    "model_usage_metrics",
    "string_from_row",
]
