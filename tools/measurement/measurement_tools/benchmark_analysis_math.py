# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared numeric and dataframe operations for benchmark analysis."""

from __future__ import annotations

import json
from typing import cast

import pandas as pd

from measurement_tools.stats import sum_int_or_zero as _sum_int_or_zero
from measurement_tools.stats import sum_or_none as _sum_or_none


def pipeline_stage_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    stages = records_of_type(dataframe, "stage")
    if "stage" not in stages.columns:
        return stages.iloc[0:0]
    return stages[stages["stage"] == "Anonymizer._run_internal"]

def first_float(frames: list[pd.DataFrame], columns: list[str]) -> float | None:
    value = first_value(frames, columns)
    return float(value) if value is not None else None

def non_null_count(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    return int(pd.to_numeric(dataframe[column], errors="coerce").notna().sum())

def sum_optional_numbers(*values: object) -> float | None:
    numeric_values = [optional_number(value) for value in values]
    if any(value is None for value in numeric_values):
        return None
    return sum(cast(float, value) for value in numeric_values)

def sum_bool_or_zero(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    return int(dataframe[column].fillna(False).astype(bool).sum())

def records_of_type(dataframe: pd.DataFrame, record_type: str) -> pd.DataFrame:
    if "record_type" not in dataframe.columns:
        return dataframe.iloc[0:0]
    return dataframe[dataframe["record_type"] == record_type]

def first_int(frames: list[pd.DataFrame], columns: list[str]) -> int | None:
    value = first_value(frames, columns)
    return int(float(value)) if value is not None else None

def optional_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)

def safe_rate(numerator: object, elapsed_sec: object) -> float | None:
    numerator_value = optional_number(numerator)
    elapsed_value = optional_number(elapsed_sec)
    if numerator_value is None or elapsed_value is None or elapsed_value <= 0:
        return None
    return numerator_value / elapsed_value

def first_value(frames: list[pd.DataFrame], columns: list[str]) -> str | None:
    for frame in frames:
        for column in columns:
            if column not in frame.columns:
                continue
            values = frame[column].dropna()
            if not values.empty:
                return str(values.iloc[0])
    return None

def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)

def zero_with_positive_count(dataframe: pd.DataFrame, *, zero_column: str, positive_column: str) -> int:
    if zero_column not in dataframe.columns or positive_column not in dataframe.columns:
        return 0
    zero_values = pd.to_numeric(dataframe[zero_column], errors="coerce")
    positive_values = pd.to_numeric(dataframe[positive_column], errors="coerce")
    return int(((zero_values == 0) & (positive_values > 0)).sum())

def coalesce_number(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None

def coerce_count_mapping(value: object) -> dict[str, int]:
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(payload, dict):
        return {}
    counts: dict[str, int] = {}
    for key, count in payload.items():
        numeric = pd.to_numeric(pd.Series([count]), errors="coerce").dropna()
        if not numeric.empty and numeric.iloc[0]:
            counts[str(key)] = int(numeric.iloc[0])
    return counts

def records_of_types(dataframe: pd.DataFrame, record_types: set[str]) -> pd.DataFrame:
    if "record_type" not in dataframe.columns:
        return dataframe.iloc[0:0]
    return dataframe[dataframe["record_type"].isin(record_types)]

def request_failure_rate(*, failed: object, total: object) -> float | None:
    failed_value = optional_number(failed)
    total_value = optional_number(total)
    if failed_value is None or total_value is None or total_value <= 0:
        return None
    return failed_value / total_value

def positive_count(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    values = pd.to_numeric(dataframe[column], errors="coerce").fillna(0)
    return int((values > 0).sum())

def safe_ratio(numerator: object, denominator: object) -> float | None:
    numerator_value = optional_number(numerator)
    denominator_value = optional_number(denominator)
    if numerator_value is None or denominator_value is None or denominator_value <= 0:
        return None
    return numerator_value / denominator_value

def sum_int_or_none(dataframe: pd.DataFrame, column: str) -> int | None:
    value = _sum_or_none(dataframe, column)
    return int(value) if value is not None else None

def sum_prefixed_ints(dataframe: pd.DataFrame, prefix: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    base_column = prefix.removesuffix(".")
    if base_column in dataframe.columns:
        for value in dataframe[base_column].dropna().tolist():
            for key, count in coerce_count_mapping(value).items():
                totals[key] = totals.get(key, 0) + count
    for column in sorted(col for col in dataframe.columns if col.startswith(prefix)):
        value = _sum_int_or_zero(dataframe, column)
        if value:
            totals[column.removeprefix(prefix)] = value
    return totals

def zero_count(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    return int((values == 0).sum())

def model_workflow_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    return records_of_types(dataframe, {"ndd_workflow", "model_workflow"})


__all__ = ['coalesce_number', 'coerce_count_mapping', 'f1', 'first_float', 'first_int', 'first_value', 'model_workflow_rows', 'non_null_count', 'optional_number', 'pipeline_stage_rows', 'positive_count', 'records_of_type', 'records_of_types', 'request_failure_rate', 'safe_rate', 'safe_ratio', 'sum_bool_or_zero', 'sum_int_or_none', 'sum_optional_numbers', 'sum_prefixed_ints', 'zero_count', 'zero_with_positive_count']
