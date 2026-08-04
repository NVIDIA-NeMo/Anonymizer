# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Metric, column, and panel catalogs shared by W&B report surfaces."""

from __future__ import annotations

from typing import Any

from measurement_tools.wandb_metric_schema import metric_paths
from measurement_tools.wandb_report_models import GroupComparison

_ROW_FLOW_FIELDS = ("input_row_count", "output_row_count", "failed_record_count")
_ROW_THROUGHPUT_FIELDS = ("input_rows_per_sec_mean", "output_rows_per_sec_mean")
_CASE_HEALTH_METRICS = metric_paths(
    "benchmark",
    "case_total",
    "case_completed",
    "case_errored",
    "case_success_rate",
)
_CASE_LATENCY_METRICS = [
    *metric_paths("benchmark", "case_elapsed_sec_mean", "case_elapsed_sec_sum"),
    *metric_paths("measurement/stage", "elapsed_sec"),
]
_NDD_ROW_FLOW_METRICS = metric_paths(
    "measurement/ndd_workflow",
    "seed_row_count",
    *_ROW_FLOW_FIELDS,
    "column_count",
)
_NDD_REQUEST_HEALTH_METRICS = metric_paths(
    "measurement/ndd_workflow",
    "observed_total_requests",
    "observed_successful_requests",
    "observed_failed_requests",
    "observed_failed_request_rate_mean",
)
_NDD_TOKEN_METRICS = metric_paths(
    "measurement/ndd_workflow",
    "observed_input_tokens",
    "observed_output_tokens",
    "observed_total_tokens",
    "observed_tokens_per_successful_request_mean",
)
_NDD_THROUGHPUT_METRICS = metric_paths(
    "measurement/ndd_workflow",
    "elapsed_sec",
    *_ROW_THROUGHPUT_FIELDS,
    "observed_requests_per_sec_mean",
    "observed_tokens_per_sec_mean",
)
_RECORD_PRIVACY_METRICS = metric_paths(
    "measurement/record",
    "final_entity_count",
    "entity_precision_mean",
    "entity_recall_mean",
    "entity_f1_mean",
    "original_value_leak_count",
    "leakage_mass_mean",
    "weighted_leakage_rate_mean",
    "detected_candidate_count",
    "validation_chunk_count",
    "llm_calls_estimated_total",
)
_REWRITE_UTILITY_METRICS = metric_paths(
    "measurement/record",
    "utility_score_mean",
    "repair_iterations_mean",
)
_REPLACEMENT_QUALITY_METRICS = metric_paths(
    "measurement/record",
    "replacement_count",
    "replacement_duplicate_value_count",
    "replacement_missing_final_entity_count",
    "replacement_missing_final_value_count",
    "replacement_synthetic_original_collision_count",
    "replacement_synthetic_original_collision_value_count",
)
_STAGE_THROUGHPUT_METRICS = metric_paths(
    "measurement/stage",
    *_ROW_FLOW_FIELDS,
    *_ROW_THROUGHPUT_FIELDS,
)
_METRIC_PANEL_GROUPS = [
    ("Case Health", _CASE_HEALTH_METRICS),
    ("Case Latency", _CASE_LATENCY_METRICS),
    ("NDD Row Flow", _NDD_ROW_FLOW_METRICS),
    ("NDD Request Health", _NDD_REQUEST_HEALTH_METRICS),
    ("NDD Token Usage", _NDD_TOKEN_METRICS),
    ("NDD Throughput", _NDD_THROUGHPUT_METRICS),
    ("Record Privacy", _RECORD_PRIVACY_METRICS),
    ("Rewrite Utility", _REWRITE_UTILITY_METRICS),
    ("Replacement Quality", _REPLACEMENT_QUALITY_METRICS),
    ("Stage Throughput", _STAGE_THROUGHPUT_METRICS),
]
_MEASUREMENT_TABLE_KEYS = metric_paths(
    "measurement_table",
    "run",
    "stage",
    "ndd_workflow",
    "model_workflow",
    "record",
    "evaluation_record",
)
_WORKSPACE_JOB_FILTERS = ("benchmark", "benchmark-import", "benchmark-sweep")
_WORKSPACE_SUMMARY_SCALARS = metric_paths(
    "benchmark",
    "case_success_rate",
    "case_completed",
    "case_errored",
    "case_elapsed_sec_mean",
)
_WORKSPACE_COMPARISON_COLUMNS = [
    "config:sweep_arm_id",
    "config:benchmark_strategies",
    "config:benchmark_gliner_thresholds",
    "config:benchmark_risk_tolerances",
    "summary:benchmark/case_success_rate",
    "summary:measurement/record/utility_score_mean",
    "summary:measurement/record/weighted_leakage_rate_mean",
    "summary:measurement/ndd_workflow/observed_total_tokens",
]
_WORKSPACE_BAR_SECTIONS = (
    ("Benchmark Summary", (("Case Health", _CASE_HEALTH_METRICS), ("Case Latency", _CASE_LATENCY_METRICS)), True),
    (
        "Privacy",
        (("Privacy Outcomes", _RECORD_PRIVACY_METRICS), ("Replacement Quality", _REPLACEMENT_QUALITY_METRICS)),
        True,
    ),
    ("Utility", (("Rewrite Utility", _REWRITE_UTILITY_METRICS),), True),
    (
        "Cost/Throughput",
        (
            ("NDD Request Health", _NDD_REQUEST_HEALTH_METRICS),
            ("NDD Token Usage", _NDD_TOKEN_METRICS),
            ("NDD Throughput", _NDD_THROUGHPUT_METRICS),
            ("Stage Throughput", _STAGE_THROUGHPUT_METRICS),
        ),
        True,
    ),
)


def single_run_visible_columns() -> list[str]:
    return summary_columns(all_report_metrics())


def group_visible_columns(comparison: GroupComparison, parameter_columns: list[str] | None = None) -> list[str]:
    columns = [
        f"config:{comparison.config_key}",
        "config:run_kind",
        *(["config:sweep_id", "config:sweep_params"] if comparison.run_kind == "sweep_arm" else []),
        *(parameter_columns or []),
        *summary_columns(all_report_metrics()),
    ]
    return list(dict.fromkeys(columns))


def metric_title(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1].replace("_", " ").title()


def benchmark_panels(wr: Any, *, groupby: Any | None = None) -> list[Any]:
    return [
        *metric_panels(wr, groupby=groupby),
        wr.MediaBrowser(title="Sanitized Measurement Tables", media_keys=_MEASUREMENT_TABLE_KEYS, mode="grid"),
    ]


def metric_panels(wr: Any, *, groupby: Any | None) -> list[Any]:
    return [bar_panel(wr, title, metrics, groupby=groupby) for title, metrics in _METRIC_PANEL_GROUPS]


def bar_panel(wr: Any, title: str, metrics: list[str], *, groupby: Any | None = None) -> Any:
    return wr.BarPlot(title=title, metrics=metrics, groupby=groupby)


def all_report_metrics() -> list[str]:
    return list(dict.fromkeys(metric for _, metrics in _METRIC_PANEL_GROUPS for metric in metrics))


def summary_columns(metrics: list[str]) -> list[str]:
    return [f"summary:{metric}" for metric in metrics]


__all__ = [
    "all_report_metrics",
    "bar_panel",
    "benchmark_panels",
    "group_visible_columns",
    "metric_panels",
    "metric_title",
    "single_run_visible_columns",
    "summary_columns",
]
