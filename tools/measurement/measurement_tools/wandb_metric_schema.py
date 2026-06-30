# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""W&B metric paths and aggregation over package-owned measurement fields."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping

from anonymizer.measurement.fields import (
    SCALAR_ADDITIVE_FIELDS,
    SCALAR_AVERAGED_FIELDS,
    SCALAR_LAST_VALUE_FIELDS,
)

ScalarValue = bool | float | int | str


class ScalarAggregation(StrEnum):
    LAST = "last"
    SUM = "sum"
    MEAN = "mean"


class BenchmarkMetric(StrEnum):
    CASE_TOTAL = "benchmark/case_total"
    CASE_COMPLETED = "benchmark/case_completed"
    CASE_ERRORED = "benchmark/case_errored"
    CASE_SUCCESS_RATE = "benchmark/case_success_rate"
    CASE_ELAPSED_SEC_SUM = "benchmark/case_elapsed_sec_sum"
    CASE_ELAPSED_SEC_MEAN = "benchmark/case_elapsed_sec_mean"


def metric_path(namespace: str, name: str) -> str:
    return f"{namespace}/{name}"


def metric_paths(namespace: str, *names: str) -> list[str]:
    return [metric_path(namespace, name) for name in names]


SCALAR_AGGREGATION_BY_FIELD: Mapping[str, ScalarAggregation] = MappingProxyType(
    {
        **dict.fromkeys(SCALAR_LAST_VALUE_FIELDS, ScalarAggregation.LAST),
        **dict.fromkeys(SCALAR_ADDITIVE_FIELDS, ScalarAggregation.SUM),
        **dict.fromkeys(SCALAR_AVERAGED_FIELDS, ScalarAggregation.MEAN),
    }
)
AGGREGATED_MEASUREMENT_FIELDS = frozenset(
    f"{field_name}_mean" if aggregation is ScalarAggregation.MEAN else field_name
    for field_name, aggregation in SCALAR_AGGREGATION_BY_FIELD.items()
)
BENCHMARK_METRIC_NAMES = frozenset(BenchmarkMetric)


@dataclass
class ScalarMetricAccumulator:
    _last_values: dict[str, ScalarValue] = field(default_factory=dict)
    _sums: dict[str, float] = field(default_factory=dict)
    _mean_totals: dict[str, tuple[float, int]] = field(default_factory=dict)

    def update(self, metrics: Mapping[str, ScalarValue]) -> None:
        for key, value in metrics.items():
            field_name = key.rsplit("/", maxsplit=1)[-1]
            aggregation = SCALAR_AGGREGATION_BY_FIELD[field_name]
            if aggregation is ScalarAggregation.LAST:
                self._last_values[key] = value
            elif aggregation is ScalarAggregation.SUM:
                self._sums[key] = self._sums.get(key, 0.0) + float(value)
            elif aggregation is ScalarAggregation.MEAN:
                mean_key = f"{key}_mean"
                total, count = self._mean_totals.get(mean_key, (0.0, 0))
                self._mean_totals[mean_key] = (total + float(value), count + 1)
            else:
                raise AssertionError(f"Unsupported scalar aggregation: {aggregation}")

    def metrics(self) -> dict[str, ScalarValue]:
        means = {key: total / count for key, (total, count) in self._mean_totals.items()}
        return {**self._last_values, **self._sums, **means}
