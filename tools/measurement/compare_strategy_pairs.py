#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare benchmark case-analysis rows for baseline/candidate strategy pairs.

Usage:
    uv run python tools/measurement/compare_strategy_pairs.py analysis/case_analysis.csv \
      --baseline-strategy default --candidate-strategy detector_native_validate_no_augment
    uv run python tools/measurement/compare_strategy_pairs.py analysis/case_analysis.parquet \
      --baseline-config default --candidate-config no-augment --output comparisons.csv
    uv run python tools/measurement/compare_strategy_pairs.py baseline/case_analysis.csv \
      --candidate-case-analysis candidate/case_analysis.csv \
      --baseline-strategy default --candidate-strategy native_single_pass
"""

from __future__ import annotations

import ast
import json
import logging
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import cyclopts
import pandas as pd
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.tables import ExportFormat
from pydantic import BaseModel, Field

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.strategy_pairs")

_SIGNATURE_DETAIL_FIELDS = {
    "label",
    "source",
    "row_index",
    "start_position",
    "end_position",
    "value_length",
}


class SafetyVerdict(StrEnum):
    passed = "pass"
    review = "review"
    fail = "fail"


class PerformanceVerdict(StrEnum):
    improved = "improved"
    mixed = "mixed"
    regressed = "regressed"
    unchanged = "unchanged"
    unknown = "unknown"


class CandidateVerdict(StrEnum):
    candidate_viable = "candidate_viable"
    review = "review"
    reject = "reject"


_MIN_CANDIDATE_OVERLAP_RATIO = 0.8
_MAX_CANDIDATE_BOUNDARY_GAP_CHARS = 8
_MIN_CANDIDATE_BOUNDARY_OVERLAP_CHARS = 8
_BOUNDARY_NORMALIZED_BASELINE_LABELS = {"api_key", "http_cookie", "unique_id"}


class ComparisonSummary(BaseModel):
    comparison_count: int = 0
    value_protection_verdict_counts: dict[str, int] = Field(default_factory=dict)
    signature_parity_verdict_counts: dict[str, int] = Field(default_factory=dict)
    safety_verdict_counts: dict[str, int] = Field(default_factory=dict)
    performance_verdict_counts: dict[str, int] = Field(default_factory=dict)
    candidate_verdict_counts: dict[str, int] = Field(default_factory=dict)
    candidate_viable_workloads: list[str] = Field(default_factory=list)
    review_workloads: list[str] = Field(default_factory=list)
    rejected_workloads: list[str] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    workload_id: str
    baseline_config_id: str
    candidate_config_id: str
    baseline_strategy: str | None = None
    candidate_strategy: str | None = None
    baseline_replacement_strategy: str | None = None
    candidate_replacement_strategy: str | None = None
    baseline_case_count: int
    candidate_case_count: int
    baseline_failed_case_count: float | None = None
    candidate_failed_case_count: float | None = None
    failed_case_count_delta: float | None = None
    baseline_failed_case_rate: float | None = None
    candidate_failed_case_rate: float | None = None
    failed_case_rate_delta: float | None = None
    baseline_pipeline_elapsed_sec: float | None = None
    candidate_pipeline_elapsed_sec: float | None = None
    pipeline_elapsed_sec_delta: float | None = None
    pipeline_elapsed_sec_delta_pct: float | None = None
    baseline_observed_total_requests: float | None = None
    candidate_observed_total_requests: float | None = None
    observed_total_requests_delta: float | None = None
    baseline_observed_total_tokens: float | None = None
    candidate_observed_total_tokens: float | None = None
    observed_total_tokens_delta: float | None = None
    baseline_observed_failed_requests: float | None = None
    candidate_observed_failed_requests: float | None = None
    observed_failed_requests_delta: float | None = None
    baseline_observed_bridge_fallback_requests: float | None = None
    candidate_observed_bridge_fallback_requests: float | None = None
    observed_bridge_fallback_requests_delta: float | None = None
    baseline_observed_non_bridge_total_requests: float | None = None
    candidate_observed_non_bridge_total_requests: float | None = None
    observed_non_bridge_total_requests_delta: float | None = None
    baseline_observed_non_bridge_failed_requests: float | None = None
    candidate_observed_non_bridge_failed_requests: float | None = None
    observed_non_bridge_failed_requests_delta: float | None = None
    baseline_original_value_leak_count: float | None = None
    candidate_original_value_leak_count: float | None = None
    original_value_leak_count_delta: float | None = None
    baseline_original_value_leak_record_count: float | None = None
    candidate_original_value_leak_record_count: float | None = None
    original_value_leak_record_count_delta: float | None = None
    baseline_replacement_missing_final_entity_count: float | None = None
    candidate_replacement_missing_final_entity_count: float | None = None
    replacement_missing_final_entity_count_delta: float | None = None
    baseline_replacement_missing_final_value_count: float | None = None
    candidate_replacement_missing_final_value_count: float | None = None
    replacement_missing_final_value_count_delta: float | None = None
    baseline_replacement_synthetic_original_collision_count: float | None = None
    candidate_replacement_synthetic_original_collision_count: float | None = None
    replacement_synthetic_original_collision_count_delta: float | None = None
    baseline_replacement_synthetic_original_collision_value_count: float | None = None
    candidate_replacement_synthetic_original_collision_value_count: float | None = None
    replacement_synthetic_original_collision_value_count_delta: float | None = None
    value_protection_verdict: SafetyVerdict | None = None
    signature_parity_verdict: SafetyVerdict | None = None
    safety_verdict: SafetyVerdict | None = None
    performance_verdict: PerformanceVerdict | None = None
    candidate_verdict: CandidateVerdict | None = None
    baseline_final_entity_count: float | None = None
    candidate_final_entity_count: float | None = None
    final_entity_count_delta: float | None = None
    baseline_seed_validation_candidate_count: float | None = None
    candidate_seed_validation_candidate_count: float | None = None
    seed_validation_candidate_count_delta: float | None = None
    baseline_augmented_entity_count: float | None = None
    candidate_augmented_entity_count: float | None = None
    augmented_entity_count_delta: float | None = None
    baseline_augmented_new_final_value_count: float | None = None
    candidate_augmented_new_final_value_count: float | None = None
    augmented_new_final_value_count_delta: float | None = None
    baseline_detector_entity_count: float | None = None
    candidate_detector_entity_count: float | None = None
    baseline_augmenter_entity_count: float | None = None
    candidate_augmenter_entity_count: float | None = None
    baseline_only_final_entity_signature_count: int | None = None
    candidate_only_final_entity_signature_count: int | None = None
    shared_final_entity_signature_count: int | None = None
    baseline_only_final_entity_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_only_final_entity_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    shared_final_entity_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_only_candidate_covered_signature_count: int | None = None
    baseline_only_candidate_overlapping_signature_count: int | None = None
    baseline_only_candidate_uncovered_signature_count: int | None = None
    baseline_only_candidate_covered_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_only_candidate_overlapping_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_only_candidate_uncovered_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_only_candidate_label_mismatch_signature_count: int | None = None
    baseline_only_candidate_label_mismatch_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_stable_final_entity_signature_count: int | None = None
    candidate_stable_final_entity_signature_count: int | None = None
    stable_final_entity_signature_count_delta: int | None = None
    baseline_stable_candidate_unstable_final_entity_signature_count: int | None = None
    candidate_stable_baseline_unstable_final_entity_signature_count: int | None = None
    shared_stable_final_entity_signature_count: int | None = None
    baseline_stable_candidate_covered_signature_count: int | None = None
    baseline_stable_candidate_overlapping_signature_count: int | None = None
    baseline_stable_candidate_uncovered_signature_count: int | None = None
    baseline_stable_candidate_covered_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_stable_candidate_overlapping_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_stable_candidate_uncovered_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_stable_candidate_label_mismatch_signature_count: int | None = None
    baseline_stable_candidate_label_mismatch_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_stable_candidate_unstable_final_entity_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_stable_baseline_unstable_final_entity_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    shared_stable_final_entity_signature_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)
    candidate_replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    input_path: str
    candidate_input_path: str | None = None
    baseline_selector: str
    candidate_selector: str
    summary: ComparisonSummary = Field(default_factory=ComparisonSummary)
    comparisons: list[ComparisonRow] = Field(default_factory=list)

    @property
    def comparison_count(self) -> int:
        return len(self.comparisons)


def read_case_analysis(path: Path) -> pd.DataFrame:
    if not path.exists() or path.is_dir():
        raise ValueError(f"case analysis path is not a file: {path}")
    if path.suffix == ".parquet":
        table = pd.read_parquet(path)
    elif path.suffix == ".csv":
        table = pd.read_csv(path)
    elif path.suffix == ".jsonl":
        table = pd.read_json(path, lines=True)
    else:
        raise ValueError(f"unsupported case analysis format: {path.suffix}")
    _validate_case_analysis_columns(table)
    return table


def _validate_case_analysis_columns(table: pd.DataFrame) -> None:
    required = {"workload_id", "config_id", "case_id"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"case analysis is missing required column(s): {', '.join(missing)}")


def compare_case_analysis(
    table: pd.DataFrame,
    *,
    baseline_config: str | None = None,
    candidate_config: str | None = None,
    baseline_strategy: str | None = None,
    candidate_strategy: str | None = None,
) -> list[ComparisonRow]:
    return compare_case_tables(
        table,
        table,
        baseline_config=baseline_config,
        candidate_config=candidate_config,
        baseline_strategy=baseline_strategy,
        candidate_strategy=candidate_strategy,
    )


def compare_case_tables(
    baseline_table: pd.DataFrame,
    candidate_table: pd.DataFrame,
    *,
    baseline_config: str | None = None,
    candidate_config: str | None = None,
    baseline_strategy: str | None = None,
    candidate_strategy: str | None = None,
) -> list[ComparisonRow]:
    baseline = _select_rows(
        baseline_table, config_id=baseline_config, strategy=baseline_strategy, selector_name="baseline"
    )
    candidate = _select_rows(
        candidate_table, config_id=candidate_config, strategy=candidate_strategy, selector_name="candidate"
    )
    return [
        _compare_workload(workload_id, baseline, candidate) for workload_id in _common_workloads(baseline, candidate)
    ]


def summarize_comparisons(rows: list[ComparisonRow]) -> ComparisonSummary:
    return ComparisonSummary(
        comparison_count=len(rows),
        value_protection_verdict_counts=_verdict_counts(rows, "value_protection_verdict", list(SafetyVerdict)),
        signature_parity_verdict_counts=_verdict_counts(rows, "signature_parity_verdict", list(SafetyVerdict)),
        safety_verdict_counts=_verdict_counts(rows, "safety_verdict", list(SafetyVerdict)),
        performance_verdict_counts=_verdict_counts(rows, "performance_verdict", list(PerformanceVerdict)),
        candidate_verdict_counts=_verdict_counts(rows, "candidate_verdict", list(CandidateVerdict)),
        candidate_viable_workloads=_workloads_by_candidate_verdict(rows, CandidateVerdict.candidate_viable),
        review_workloads=_workloads_by_candidate_verdict(rows, CandidateVerdict.review),
        rejected_workloads=_workloads_by_candidate_verdict(rows, CandidateVerdict.reject),
    )


def _verdict_counts(rows: list[ComparisonRow], field: str, values: list[StrEnum]) -> dict[str, int]:
    counts = {value.value: 0 for value in values}
    for row in rows:
        verdict = _verdict_value(getattr(row, field))
        if verdict is not None:
            counts[verdict] = counts.get(verdict, 0) + 1
    return counts


def _workloads_by_candidate_verdict(rows: list[ComparisonRow], verdict: CandidateVerdict) -> list[str]:
    return sorted(row.workload_id for row in rows if _verdict_value(row.candidate_verdict) == verdict.value)


def _verdict_value(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, StrEnum):
        return value.value
    return str(value)


def _select_rows(
    table: pd.DataFrame,
    *,
    config_id: str | None,
    strategy: str | None,
    selector_name: str,
) -> pd.DataFrame:
    if (config_id is None) == (strategy is None):
        raise ValueError(f"{selector_name} selector must specify exactly one of config or strategy")
    column, value = ("config_id", config_id) if config_id is not None else ("experimental_detection_strategy", strategy)
    if column not in table.columns:
        raise ValueError(f"case analysis is missing selector column: {column}")
    selected = table[table[column].astype(str) == str(value)]
    if selected.empty:
        raise ValueError(f"{selector_name} selector matched no rows: {column}={value}")
    _validate_unique_config_per_workload(selected, selector_name=selector_name)
    return selected


def _validate_unique_config_per_workload(rows: pd.DataFrame, *, selector_name: str) -> None:
    counts = rows.groupby("workload_id")["config_id"].nunique()
    ambiguous = sorted(str(index) for index, count in counts.items() if count > 1)
    if ambiguous:
        raise ValueError(f"{selector_name} selector matched multiple configs for workload(s): {', '.join(ambiguous)}")


def _common_workloads(baseline: pd.DataFrame, candidate: pd.DataFrame) -> list[str]:
    baseline_workloads = set(str(value) for value in baseline["workload_id"].dropna())
    candidate_workloads = set(str(value) for value in candidate["workload_id"].dropna())
    common = sorted(baseline_workloads & candidate_workloads)
    if not common:
        raise ValueError("baseline and candidate selectors have no workloads in common")
    return common


def _compare_workload(workload_id: str, baseline: pd.DataFrame, candidate: pd.DataFrame) -> ComparisonRow:
    baseline_summary = _summarize_selector(baseline[baseline["workload_id"].astype(str) == workload_id])
    candidate_summary = _summarize_selector(candidate[candidate["workload_id"].astype(str) == workload_id])
    return ComparisonRow(
        workload_id=workload_id,
        baseline_config_id=str(baseline_summary["config_id"]),
        candidate_config_id=str(candidate_summary["config_id"]),
        baseline_strategy=_optional_string(baseline_summary.get("experimental_detection_strategy")),
        candidate_strategy=_optional_string(candidate_summary.get("experimental_detection_strategy")),
        baseline_replacement_strategy=_optional_string(baseline_summary.get("experimental_replacement_strategy")),
        candidate_replacement_strategy=_optional_string(candidate_summary.get("experimental_replacement_strategy")),
        baseline_case_count=int(baseline_summary["case_count"]),
        candidate_case_count=int(candidate_summary["case_count"]),
        **_comparison_metrics(baseline_summary, candidate_summary),
    )


def _summarize_selector(rows: pd.DataFrame) -> dict[str, object]:
    if rows.empty:
        raise ValueError("selector has no rows for workload")
    case_count = int(rows["case_id"].nunique())
    summary: dict[str, object] = {
        "config_id": _single_string(rows, "config_id"),
        "experimental_detection_strategy": _single_string(rows, "experimental_detection_strategy"),
        "experimental_replacement_strategy": _single_string(rows, "experimental_replacement_strategy"),
        "case_count": case_count,
    }
    for column in _NUMERIC_COLUMNS:
        summary[column] = _median_or_none(rows, column)
    summary["replacement_missing_final_entity_count"] = _sum_or_none(rows, "replacement_missing_final_entity_count")
    summary["replacement_missing_final_value_count"] = _sum_or_none(rows, "replacement_missing_final_value_count")
    summary["replacement_synthetic_original_collision_count"] = _sum_or_none(
        rows,
        "replacement_synthetic_original_collision_count",
    )
    summary["replacement_synthetic_original_collision_value_count"] = _sum_or_none(
        rows,
        "replacement_synthetic_original_collision_value_count",
    )
    summary["original_value_leak_count"] = _sum_or_none(rows, "original_value_leak_count")
    summary["original_value_leak_record_count"] = _sum_or_none(rows, "original_value_leak_record_count")
    summary["original_value_leak_label_counts"] = _sum_label_counts(rows, "original_value_leak_label_counts")
    summary["replacement_synthetic_original_collision_label_counts"] = _sum_label_counts(
        rows,
        "replacement_synthetic_original_collision_label_counts",
    )
    summary["failed_case_count"] = _failed_case_count(rows)
    summary["failed_case_rate"] = _rate(summary["failed_case_count"], case_count)
    summary["artifact_final_entity_signature_hashes"] = _signature_hashes(rows)
    summary["stable_artifact_final_entity_signature_hashes"] = _stable_signature_hashes(rows) if case_count > 1 else []
    summary["artifact_final_entity_signature_labels"] = _signature_labels(rows)
    summary["artifact_final_entity_signature_details"] = _signature_details(rows)
    return summary


def _single_string(rows: pd.DataFrame, column: str) -> str | None:
    if column not in rows.columns:
        return None
    values = sorted({str(value) for value in rows[column].dropna()})
    return values[0] if values else None


_NUMERIC_COLUMNS = [
    "pipeline_elapsed_sec",
    "observed_total_requests",
    "observed_total_tokens",
    "observed_failed_requests",
    "observed_bridge_fallback_requests",
    "observed_non_bridge_total_requests",
    "observed_non_bridge_failed_requests",
    "replacement_missing_final_entity_count",
    "replacement_missing_final_value_count",
    "replacement_synthetic_original_collision_count",
    "replacement_synthetic_original_collision_value_count",
    "final_entity_count",
    "seed_validation_candidate_count",
    "augmented_entity_count",
    "augmented_new_final_value_count",
    "artifact_final_detector_entity_count",
    "artifact_final_augmenter_entity_count",
]


def _median_or_none(rows: pd.DataFrame, column: str) -> float | None:
    if column not in rows.columns:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(values.median()) if not values.empty else None


def _sum_or_none(rows: pd.DataFrame, column: str) -> float | None:
    if column not in rows.columns:
        return None
    values = pd.to_numeric(rows[column], errors="coerce").dropna()
    return float(values.sum()) if not values.empty else None


def _comparison_metrics(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    metrics = _named_metric_deltas(baseline, candidate)
    metrics.update(_source_counts(baseline, candidate))
    metrics.update(_entity_signature_deltas(baseline, candidate))
    metrics["flags"] = _comparison_flags(
        metrics,
        baseline_strategy=_optional_string(baseline.get("experimental_detection_strategy")),
        candidate_strategy=_optional_string(candidate.get("experimental_detection_strategy")),
        baseline_replacement_strategy=_optional_string(baseline.get("experimental_replacement_strategy")),
        candidate_replacement_strategy=_optional_string(candidate.get("experimental_replacement_strategy")),
    )
    metrics.update(_comparison_verdicts(metrics))
    return metrics


def _named_metric_deltas(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    return {
        **_metric_delta("failed_case_count", baseline, candidate),
        **_metric_delta("failed_case_rate", baseline, candidate),
        **_metric_delta("pipeline_elapsed_sec", baseline, candidate, include_pct=True),
        **_metric_delta("observed_total_requests", baseline, candidate),
        **_metric_delta("observed_total_tokens", baseline, candidate),
        **_metric_delta("observed_failed_requests", baseline, candidate),
        **_metric_delta("observed_bridge_fallback_requests", baseline, candidate),
        **_metric_delta("observed_non_bridge_total_requests", baseline, candidate),
        **_metric_delta("observed_non_bridge_failed_requests", baseline, candidate),
        **_metric_delta("original_value_leak_count", baseline, candidate),
        **_metric_delta("original_value_leak_record_count", baseline, candidate),
        **_metric_delta("replacement_missing_final_entity_count", baseline, candidate),
        **_metric_delta("replacement_missing_final_value_count", baseline, candidate),
        **_metric_delta("replacement_synthetic_original_collision_count", baseline, candidate),
        **_metric_delta("replacement_synthetic_original_collision_value_count", baseline, candidate),
        **_metric_delta("final_entity_count", baseline, candidate),
        **_metric_delta("seed_validation_candidate_count", baseline, candidate),
        **_metric_delta("augmented_entity_count", baseline, candidate),
        **_metric_delta("augmented_new_final_value_count", baseline, candidate),
    }


def _metric_delta(
    name: str,
    baseline: dict[str, object],
    candidate: dict[str, object],
    *,
    include_pct: bool = False,
) -> dict[str, float | None]:
    base = _optional_float(baseline.get(name))
    cand = _optional_float(candidate.get(name))
    values = {f"baseline_{name}": base, f"candidate_{name}": cand, f"{name}_delta": _delta(base, cand)}
    if include_pct:
        values[f"{name}_delta_pct"] = _delta_pct(base, cand)
    return values


def _source_counts(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    return {
        "baseline_detector_entity_count": _optional_float(baseline.get("artifact_final_detector_entity_count")),
        "candidate_detector_entity_count": _optional_float(candidate.get("artifact_final_detector_entity_count")),
        "baseline_augmenter_entity_count": _optional_float(baseline.get("artifact_final_augmenter_entity_count")),
        "candidate_augmenter_entity_count": _optional_float(candidate.get("artifact_final_augmenter_entity_count")),
        "baseline_original_value_leak_label_counts": _coerce_count_map(
            baseline.get("original_value_leak_label_counts")
        ),
        "candidate_original_value_leak_label_counts": _coerce_count_map(
            candidate.get("original_value_leak_label_counts")
        ),
        "baseline_replacement_synthetic_original_collision_label_counts": _coerce_count_map(
            baseline.get("replacement_synthetic_original_collision_label_counts")
        ),
        "candidate_replacement_synthetic_original_collision_label_counts": _coerce_count_map(
            candidate.get("replacement_synthetic_original_collision_label_counts")
        ),
    }


def _comparison_flags(
    metrics: dict[str, object],
    *,
    baseline_strategy: str | None,
    candidate_strategy: str | None,
    baseline_replacement_strategy: str | None,
    candidate_replacement_strategy: str | None,
) -> list[str]:
    flags: list[str] = []
    _append_if_positive(flags, metrics, "baseline_failed_case_count", "baseline_case_failures")
    _append_if_positive(flags, metrics, "candidate_failed_case_count", "candidate_case_failures")
    _append_if_positive(flags, metrics, "baseline_original_value_leak_count", "baseline_original_value_leak")
    _append_if_positive(flags, metrics, "candidate_original_value_leak_count", "candidate_original_value_leak")
    _append_if_positive(
        flags,
        metrics,
        "baseline_replacement_synthetic_original_collision_count",
        "baseline_replacement_synthetic_original_collision",
    )
    _append_if_positive(
        flags,
        metrics,
        "candidate_replacement_synthetic_original_collision_count",
        "candidate_replacement_synthetic_original_collision",
    )
    _append_if_positive(
        flags,
        metrics,
        "baseline_replacement_missing_final_entity_count",
        "baseline_replacement_missing_final_entity",
    )
    _append_if_positive(
        flags,
        metrics,
        "candidate_replacement_missing_final_entity_count",
        "candidate_replacement_missing_final_entity",
    )
    _append_if_negative(flags, metrics, "final_entity_count_delta", "entity_count_loss")
    _append_if_positive(flags, metrics, _signature_loss_metric(metrics), "entity_signature_loss")
    _append_if_positive(
        flags,
        metrics,
        "baseline_only_candidate_overlapping_signature_count",
        "span_boundary_mismatch",
    )
    _append_if_positive(
        flags,
        metrics,
        "baseline_only_candidate_label_mismatch_signature_count",
        "covered_label_mismatch",
    )
    _append_if_positive(
        flags,
        metrics,
        _stable_signature_loss_metric(metrics),
        "stable_entity_signature_loss",
    )
    failed_request_delta = (
        "observed_non_bridge_failed_requests_delta"
        if _has_metric_pair(metrics, "observed_non_bridge_failed_requests")
        else "observed_failed_requests_delta"
    )
    _append_if_positive(flags, metrics, failed_request_delta, "failed_request_increase")
    _append_if_positive(flags, metrics, "observed_bridge_fallback_requests_delta", "bridge_fallback_increase")
    _append_if_positive(flags, metrics, "observed_total_tokens_delta", "token_increase")
    _append_if_positive(flags, metrics, "observed_total_requests_delta", "request_increase")
    if _candidate_lacks_detector_entities(metrics):
        flags.append("no_candidate_detector_entities")
    if candidate_strategy in _SKIPS_LLM_VALIDATION_STRATEGIES:
        flags.append("candidate_skips_llm_validation")
    if _replacement_only_detection_instability(
        flags,
        baseline_strategy=baseline_strategy,
        candidate_strategy=candidate_strategy,
        baseline_replacement_strategy=baseline_replacement_strategy,
        candidate_replacement_strategy=candidate_replacement_strategy,
    ):
        flags.append("replacement_only_detection_instability")
    return flags


_DETECTION_INSTABILITY_FLAGS = {
    "covered_label_mismatch",
    "entity_count_loss",
    "entity_signature_loss",
    "span_boundary_mismatch",
    "stable_entity_signature_loss",
}


def _replacement_only_detection_instability(
    flags: list[str],
    *,
    baseline_strategy: str | None,
    candidate_strategy: str | None,
    baseline_replacement_strategy: str | None,
    candidate_replacement_strategy: str | None,
) -> bool:
    if baseline_strategy != candidate_strategy:
        return False
    if not baseline_replacement_strategy or not candidate_replacement_strategy:
        return False
    if baseline_replacement_strategy == candidate_replacement_strategy:
        return False
    return bool(set(flags) & _DETECTION_INSTABILITY_FLAGS)


def _signature_loss_metric(metrics: dict[str, object]) -> str:
    if metrics.get("baseline_only_candidate_uncovered_signature_count") is not None:
        return "baseline_only_candidate_uncovered_signature_count"
    return "baseline_only_final_entity_signature_count"


def _stable_signature_loss_metric(metrics: dict[str, object]) -> str:
    if metrics.get("baseline_stable_candidate_uncovered_signature_count") is not None:
        return "baseline_stable_candidate_uncovered_signature_count"
    return "baseline_stable_candidate_unstable_final_entity_signature_count"


_SKIPS_LLM_VALIDATION_STRATEGIES = {"detector_only"}


def _has_metric_pair(metrics: dict[str, object], name: str) -> bool:
    return (
        _optional_float(metrics.get(f"baseline_{name}")) is not None
        and _optional_float(metrics.get(f"candidate_{name}")) is not None
    )


def _comparison_verdicts(metrics: dict[str, object]) -> dict[str, str]:
    value_protection_verdict = _value_protection_verdict(metrics)
    signature_parity_verdict = _signature_parity_verdict(metrics)
    safety_verdict = _safety_verdict(metrics)
    performance_verdict = _performance_verdict(metrics)
    return {
        "value_protection_verdict": value_protection_verdict.value,
        "signature_parity_verdict": signature_parity_verdict.value,
        "safety_verdict": safety_verdict.value,
        "performance_verdict": performance_verdict.value,
        "candidate_verdict": _candidate_verdict(safety_verdict, performance_verdict).value,
    }


def _value_protection_verdict(metrics: dict[str, object]) -> SafetyVerdict:
    flags = set(_coerce_flag_list(metrics.get("flags")))
    if flags & {
        "candidate_case_failures",
        "candidate_original_value_leak",
        "candidate_replacement_missing_final_entity",
        "candidate_replacement_synthetic_original_collision",
        "entity_signature_loss",
        "stable_entity_signature_loss",
    }:
        return SafetyVerdict.fail
    if _entity_count_loss_without_signature_artifacts(flags, metrics):
        return SafetyVerdict.fail
    if flags & {
        "baseline_case_failures",
        "baseline_original_value_leak",
        "baseline_replacement_missing_final_entity",
        "baseline_replacement_synthetic_original_collision",
    }:
        return SafetyVerdict.review
    return SafetyVerdict.passed


def _signature_parity_verdict(metrics: dict[str, object]) -> SafetyVerdict:
    flags = set(_coerce_flag_list(metrics.get("flags")))
    if flags & {"candidate_case_failures", "entity_signature_loss", "stable_entity_signature_loss"}:
        return SafetyVerdict.fail
    if _entity_count_loss_without_signature_artifacts(flags, metrics):
        return SafetyVerdict.fail
    if flags & {
        "span_boundary_mismatch",
        "covered_label_mismatch",
        "entity_count_loss",
        "baseline_case_failures",
    }:
        return SafetyVerdict.review
    return SafetyVerdict.passed


def _safety_verdict(metrics: dict[str, object]) -> SafetyVerdict:
    flags = set(_coerce_flag_list(metrics.get("flags")))
    if flags & {
        "candidate_case_failures",
        "candidate_original_value_leak",
        "candidate_replacement_missing_final_entity",
        "candidate_replacement_synthetic_original_collision",
        "entity_signature_loss",
        "stable_entity_signature_loss",
    }:
        return SafetyVerdict.fail
    if _entity_count_loss_without_signature_artifacts(flags, metrics):
        return SafetyVerdict.fail
    if flags & {
        "no_candidate_detector_entities",
        "candidate_skips_llm_validation",
        "failed_request_increase",
        "bridge_fallback_increase",
        "span_boundary_mismatch",
        "covered_label_mismatch",
    }:
        return SafetyVerdict.review
    if flags & {
        "baseline_case_failures",
        "baseline_original_value_leak",
        "baseline_replacement_missing_final_entity",
        "baseline_replacement_synthetic_original_collision",
        "entity_count_loss",
    }:
        return SafetyVerdict.review
    return SafetyVerdict.passed


def _entity_count_loss_without_signature_artifacts(flags: set[str], metrics: dict[str, object]) -> bool:
    return "entity_count_loss" in flags and metrics.get("baseline_only_final_entity_signature_count") is None


def _performance_verdict(metrics: dict[str, object]) -> PerformanceVerdict:
    deltas = [
        _optional_float(metrics.get("pipeline_elapsed_sec_delta")),
        _optional_float(metrics.get("observed_total_requests_delta")),
        _optional_float(metrics.get("observed_total_tokens_delta")),
    ]
    known = [value for value in deltas if value is not None]
    if not known:
        return PerformanceVerdict.unknown
    improved = any(value < 0 for value in known)
    regressed = any(value > 0 for value in known)
    if improved and regressed:
        return PerformanceVerdict.mixed
    if improved:
        return PerformanceVerdict.improved
    if regressed:
        return PerformanceVerdict.regressed
    return PerformanceVerdict.unchanged


def _candidate_verdict(
    safety_verdict: SafetyVerdict,
    performance_verdict: PerformanceVerdict,
) -> CandidateVerdict:
    if safety_verdict == SafetyVerdict.fail:
        return CandidateVerdict.reject
    if safety_verdict == SafetyVerdict.passed and performance_verdict == PerformanceVerdict.improved:
        return CandidateVerdict.candidate_viable
    return CandidateVerdict.review


def _coerce_flag_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _candidate_lacks_detector_entities(metrics: dict[str, object]) -> bool:
    final_count = _optional_float(metrics.get("candidate_final_entity_count"))
    if final_count is None or final_count <= 0:
        return False
    detector_count = _optional_float(metrics.get("candidate_detector_entity_count"))
    if detector_count is not None:
        return detector_count == 0
    non_detector_count = _known_non_detector_candidate_count(metrics)
    return non_detector_count is not None and non_detector_count >= final_count


def _known_non_detector_candidate_count(metrics: dict[str, object]) -> float | None:
    known_counts = [
        _optional_float(metrics.get("candidate_augmenter_entity_count")),
    ]
    if all(value is None for value in known_counts):
        return None
    return sum(value or 0.0 for value in known_counts)


def _entity_signature_deltas(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    baseline_signatures = set(_coerce_signature_list(baseline.get("artifact_final_entity_signature_hashes")))
    candidate_signatures = set(_coerce_signature_list(candidate.get("artifact_final_entity_signature_hashes")))
    baseline_labels = _coerce_signature_labels(baseline.get("artifact_final_entity_signature_labels"))
    candidate_labels = _coerce_signature_labels(candidate.get("artifact_final_entity_signature_labels"))
    baseline_details = _coerce_signature_details(baseline.get("artifact_final_entity_signature_details"))
    candidate_details = _coerce_signature_details(candidate.get("artifact_final_entity_signature_details"))
    baseline_signatures.update(baseline_labels)
    candidate_signatures.update(candidate_labels)
    if not baseline_signatures and not candidate_signatures:
        return {
            "baseline_only_final_entity_signature_count": None,
            "candidate_only_final_entity_signature_count": None,
            "shared_final_entity_signature_count": None,
            "baseline_only_final_entity_signature_label_counts": {},
            "candidate_only_final_entity_signature_label_counts": {},
            "shared_final_entity_signature_label_counts": {},
            "baseline_only_candidate_covered_signature_count": None,
            "baseline_only_candidate_overlapping_signature_count": None,
            "baseline_only_candidate_uncovered_signature_count": None,
            "baseline_only_candidate_covered_signature_label_counts": {},
            "baseline_only_candidate_overlapping_signature_label_counts": {},
            "baseline_only_candidate_uncovered_signature_label_counts": {},
            "baseline_only_candidate_label_mismatch_signature_count": None,
            "baseline_only_candidate_label_mismatch_signature_label_counts": {},
        }
    baseline_only = baseline_signatures - candidate_signatures
    candidate_only = candidate_signatures - baseline_signatures
    shared = baseline_signatures & candidate_signatures
    coverage = _candidate_span_coverage(
        baseline_only,
        baseline_details=baseline_details,
        candidate_details=candidate_details,
        baseline_labels=baseline_labels,
    )
    return {
        "baseline_only_final_entity_signature_count": len(baseline_only),
        "candidate_only_final_entity_signature_count": len(candidate_only),
        "shared_final_entity_signature_count": len(shared),
        "baseline_only_final_entity_signature_label_counts": _signature_label_counts(
            baseline_only,
            baseline_labels,
        ),
        "candidate_only_final_entity_signature_label_counts": _signature_label_counts(
            candidate_only,
            candidate_labels,
        ),
        "shared_final_entity_signature_label_counts": _signature_label_counts(shared, baseline_labels),
        **coverage,
        **_stable_entity_signature_deltas(
            baseline,
            candidate,
            baseline_labels,
            candidate_labels,
            baseline_details,
            candidate_details,
        ),
    }


def _stable_entity_signature_deltas(
    baseline: dict[str, object],
    candidate: dict[str, object],
    baseline_labels: dict[str, str],
    candidate_labels: dict[str, str],
    baseline_details: dict[str, dict[str, object]],
    candidate_details: dict[str, dict[str, object]],
) -> dict[str, object]:
    baseline_case_count = _optional_float(baseline.get("case_count"))
    candidate_case_count = _optional_float(candidate.get("case_count"))
    if (
        baseline_case_count is None
        or candidate_case_count is None
        or baseline_case_count < 2
        or candidate_case_count < 2
    ):
        return _empty_stable_signature_deltas()
    baseline_stable = set(_coerce_signature_list(baseline.get("stable_artifact_final_entity_signature_hashes")))
    candidate_stable = set(_coerce_signature_list(candidate.get("stable_artifact_final_entity_signature_hashes")))
    if not baseline_stable and not candidate_stable:
        return _empty_stable_signature_deltas()
    baseline_stable_candidate_unstable = baseline_stable - candidate_stable
    candidate_stable_baseline_unstable = candidate_stable - baseline_stable
    shared_stable = baseline_stable & candidate_stable
    stable_candidate_details = {
        signature: detail for signature, detail in candidate_details.items() if signature in candidate_stable
    }
    coverage = _candidate_span_coverage(
        baseline_stable_candidate_unstable,
        baseline_details=baseline_details,
        candidate_details=stable_candidate_details,
        baseline_labels=baseline_labels,
        prefix="baseline_stable_candidate",
    )
    return {
        "baseline_stable_final_entity_signature_count": len(baseline_stable),
        "candidate_stable_final_entity_signature_count": len(candidate_stable),
        "stable_final_entity_signature_count_delta": len(candidate_stable) - len(baseline_stable),
        "baseline_stable_candidate_unstable_final_entity_signature_count": len(baseline_stable_candidate_unstable),
        "candidate_stable_baseline_unstable_final_entity_signature_count": len(candidate_stable_baseline_unstable),
        "shared_stable_final_entity_signature_count": len(shared_stable),
        "baseline_stable_candidate_unstable_final_entity_signature_label_counts": _signature_label_counts(
            baseline_stable_candidate_unstable,
            baseline_labels,
        ),
        "candidate_stable_baseline_unstable_final_entity_signature_label_counts": _signature_label_counts(
            candidate_stable_baseline_unstable,
            candidate_labels,
        ),
        "shared_stable_final_entity_signature_label_counts": _signature_label_counts(shared_stable, baseline_labels),
        **coverage,
    }


def _candidate_span_coverage(
    baseline_signatures: set[str],
    *,
    baseline_details: dict[str, dict[str, object]],
    candidate_details: dict[str, dict[str, object]],
    baseline_labels: dict[str, str],
    prefix: str = "baseline_only_candidate",
) -> dict[str, object]:
    contained: set[str] = set()
    overlapping: set[str] = set()
    label_mismatch: set[str] = set()
    for signature in baseline_signatures:
        match, labels_mismatch = _candidate_span_match_kind(baseline_details.get(signature), candidate_details.values())
        if match == "contained":
            contained.add(signature)
        elif match == "overlapping":
            overlapping.add(signature)
        if labels_mismatch:
            label_mismatch.add(signature)
    covered = contained | overlapping
    uncovered = baseline_signatures - covered
    return {
        f"{prefix}_covered_signature_count": len(covered),
        f"{prefix}_overlapping_signature_count": len(overlapping),
        f"{prefix}_uncovered_signature_count": len(uncovered),
        f"{prefix}_covered_signature_label_counts": _signature_label_counts(covered, baseline_labels),
        f"{prefix}_overlapping_signature_label_counts": _signature_label_counts(overlapping, baseline_labels),
        f"{prefix}_uncovered_signature_label_counts": _signature_label_counts(uncovered, baseline_labels),
        f"{prefix}_label_mismatch_signature_count": len(label_mismatch),
        f"{prefix}_label_mismatch_signature_label_counts": _signature_label_counts(label_mismatch, baseline_labels),
    }


def _candidate_span_match_kind(
    baseline_detail: dict[str, object] | None,
    candidate_details: object,
) -> tuple[str | None, bool]:
    baseline_row = _optional_int(_detail_value(baseline_detail, "row_index"))
    baseline_start = _optional_int(_detail_value(baseline_detail, "start_position"))
    baseline_end = _optional_int(_detail_value(baseline_detail, "end_position"))
    baseline_label = _optional_string(_detail_value(baseline_detail, "label"))
    if baseline_row is None or baseline_start is None or baseline_end is None:
        return None, False
    baseline_length = baseline_end - baseline_start
    if baseline_length <= 0:
        return None, False
    first_mismatched_match: tuple[str, bool] | None = None
    for candidate_detail in candidate_details:
        candidate_row = _optional_int(_detail_value(candidate_detail, "row_index"))
        candidate_start = _optional_int(_detail_value(candidate_detail, "start_position"))
        candidate_end = _optional_int(_detail_value(candidate_detail, "end_position"))
        if candidate_row != baseline_row or candidate_start is None or candidate_end is None:
            continue
        candidate_label = _optional_string(_detail_value(candidate_detail, "label"))
        labels_mismatch = _labels_mismatch(baseline_label, candidate_label)
        if candidate_start <= baseline_start and candidate_end >= baseline_end:
            if not labels_mismatch:
                return "contained", False
            first_mismatched_match = first_mismatched_match or ("contained", True)
            continue
        overlap = max(0, min(baseline_end, candidate_end) - max(baseline_start, candidate_start))
        if overlap / baseline_length >= _MIN_CANDIDATE_OVERLAP_RATIO:
            if not labels_mismatch:
                return "overlapping", False
            first_mismatched_match = first_mismatched_match or ("overlapping", True)
            continue
        if _is_small_boundary_gap(
            baseline_start=baseline_start,
            baseline_end=baseline_end,
            baseline_label=baseline_label,
            candidate_start=candidate_start,
            candidate_end=candidate_end,
            overlap=overlap,
        ):
            if not labels_mismatch:
                return "overlapping", False
            first_mismatched_match = first_mismatched_match or ("overlapping", True)
    return first_mismatched_match or (None, False)


def _labels_mismatch(baseline_label: str | None, candidate_label: str | None) -> bool:
    return baseline_label is not None and candidate_label is not None and baseline_label != candidate_label


def _is_small_boundary_gap(
    *,
    baseline_start: int,
    baseline_end: int,
    baseline_label: str | None,
    candidate_start: int,
    candidate_end: int,
    overlap: int,
) -> bool:
    if baseline_label not in _BOUNDARY_NORMALIZED_BASELINE_LABELS:
        return False
    if overlap < _MIN_CANDIDATE_BOUNDARY_OVERLAP_CHARS:
        return False
    omitted_left = max(0, candidate_start - baseline_start)
    omitted_right = max(0, baseline_end - candidate_end)
    return omitted_left + omitted_right <= _MAX_CANDIDATE_BOUNDARY_GAP_CHARS


def _detail_value(detail: object, key: str) -> object:
    if not isinstance(detail, dict):
        return None
    return detail.get(key)


def _empty_stable_signature_deltas() -> dict[str, object]:
    return {
        "baseline_stable_final_entity_signature_count": None,
        "candidate_stable_final_entity_signature_count": None,
        "stable_final_entity_signature_count_delta": None,
        "baseline_stable_candidate_unstable_final_entity_signature_count": None,
        "candidate_stable_baseline_unstable_final_entity_signature_count": None,
        "shared_stable_final_entity_signature_count": None,
        "baseline_stable_candidate_covered_signature_count": None,
        "baseline_stable_candidate_overlapping_signature_count": None,
        "baseline_stable_candidate_uncovered_signature_count": None,
        "baseline_stable_candidate_covered_signature_label_counts": {},
        "baseline_stable_candidate_overlapping_signature_label_counts": {},
        "baseline_stable_candidate_uncovered_signature_label_counts": {},
        "baseline_stable_candidate_label_mismatch_signature_count": None,
        "baseline_stable_candidate_label_mismatch_signature_label_counts": {},
        "baseline_stable_candidate_unstable_final_entity_signature_label_counts": {},
        "candidate_stable_baseline_unstable_final_entity_signature_label_counts": {},
        "shared_stable_final_entity_signature_label_counts": {},
    }


def _signature_hashes(rows: pd.DataFrame) -> list[str]:
    hashes: set[str] = set()
    for signature_set in _signature_hash_sets(rows):
        hashes.update(signature_set)
    return sorted(hashes)


def _stable_signature_hashes(rows: pd.DataFrame) -> list[str]:
    signature_sets = _signature_hash_sets(rows)
    if not signature_sets:
        return []
    return sorted(set.intersection(*signature_sets))


def _signature_hash_sets(rows: pd.DataFrame) -> list[set[str]]:
    if "artifact_final_entity_signature_hashes" not in rows.columns:
        return []
    return [set(_coerce_signature_list(value)) for value in rows["artifact_final_entity_signature_hashes"].tolist()]


def _signature_labels(rows: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    if "artifact_final_entity_signature_labels" in rows.columns:
        for value in rows["artifact_final_entity_signature_labels"].tolist():
            labels.update(_coerce_signature_labels(value))
    prefix = "artifact_final_entity_signature_labels."
    for column in rows.columns:
        if not column.startswith(prefix):
            continue
        signature_hash = column.removeprefix(prefix)
        for value in rows[column].tolist():
            if not _is_missing_cell(value):
                labels[signature_hash] = str(value)
    return dict(sorted(labels.items()))


def _signature_details(rows: pd.DataFrame) -> dict[str, dict[str, object]]:
    details: dict[str, dict[str, object]] = {}
    if "artifact_final_entity_signature_details" in rows.columns:
        for value in rows["artifact_final_entity_signature_details"].tolist():
            details.update(_coerce_signature_details(value))
    prefix = "artifact_final_entity_signature_details."
    for column in rows.columns:
        if not column.startswith(prefix):
            continue
        remainder = column.removeprefix(prefix)
        signature_hash, _, field = remainder.partition(".")
        if not signature_hash or not field:
            continue
        if field not in _SIGNATURE_DETAIL_FIELDS:
            continue
        for value in rows[column].tolist():
            if not _is_missing_cell(value):
                details.setdefault(signature_hash, {})[field] = _json_scalar(value)
    return dict(sorted(details.items()))


def _sum_label_counts(rows: pd.DataFrame, field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    if field in rows.columns:
        for value in rows[field].tolist():
            _merge_count_map(counts, _coerce_count_map(value))
    prefix = f"{field}."
    for column in rows.columns:
        if not column.startswith(prefix):
            continue
        total = _sum_or_none(rows, column)
        if total:
            counts[column.removeprefix(prefix)] = counts.get(column.removeprefix(prefix), 0) + int(total)
    return dict(sorted(counts.items()))


def _failed_case_count(rows: pd.DataFrame) -> int:
    if "case_failed" in rows.columns:
        return int(rows["case_failed"].map(_coerce_bool).sum())
    error_columns = [
        column
        for column in ("error_stage_count", "error_ndd_workflow_count", "error_model_workflow_count")
        if column in rows.columns
    ]
    if not error_columns:
        return 0
    error_counts = rows[error_columns].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    return int((error_counts > 0).sum())


def _coerce_bool(value: object) -> bool:
    if _is_missing_cell(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def _rate(count: object, total: object) -> float | None:
    count_value = _optional_float(count)
    total_value = _optional_float(total)
    if count_value is None or total_value is None or total_value <= 0:
        return None
    return count_value / total_value


def _signature_label_counts(signatures: set[str], labels: dict[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for signature_hash in signatures:
        label = labels.get(signature_hash, "unknown")
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _coerce_signature_list(value: object) -> list[str]:
    if _is_missing_cell(value):
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list | tuple | set):
        return [str(item) for item in value if not _is_missing_cell(item)]
    if not isinstance(value, str):
        return []
    text = value.strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return []
    return _coerce_signature_list(parsed)


def _coerce_signature_labels(value: object) -> dict[str, str]:
    if _is_missing_cell(value):
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        return {str(key): str(item) for key, item in value.items() if not _is_missing_cell(item)}
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
    return _coerce_signature_labels(parsed)


def _coerce_signature_details(value: object) -> dict[str, dict[str, object]]:
    if _is_missing_cell(value):
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        details: dict[str, dict[str, object]] = {}
        for signature_hash, raw_detail in value.items():
            if not isinstance(raw_detail, dict):
                continue
            details[str(signature_hash)] = {
                str(key): _json_scalar(item)
                for key, item in raw_detail.items()
                if str(key) in _SIGNATURE_DETAIL_FIELDS and not _is_missing_cell(item)
            }
        return dict(sorted(details.items()))
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
    return _coerce_signature_details(parsed)


def _json_scalar(value: object) -> object:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _coerce_count_map(value: object) -> dict[str, int]:
    if _is_missing_cell(value):
        return {}
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, dict):
        counts: dict[str, int] = {}
        for key, item in value.items():
            if _is_missing_cell(item):
                continue
            count = _optional_float(item)
            if count:
                counts[str(key)] = int(count)
        return dict(sorted(counts.items()))
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return {}
    return _coerce_count_map(parsed)


def _merge_count_map(target: dict[str, int], update: dict[str, int]) -> None:
    for key, value in update.items():
        target[key] = target.get(key, 0) + value


def _is_missing_cell(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value))


def _append_if_negative(flags: list[str], metrics: dict[str, object], field: str, flag: str) -> None:
    value = _optional_float(metrics.get(field))
    if value is not None and value < 0:
        flags.append(flag)


def _append_if_positive(flags: list[str], metrics: dict[str, object], field: str, flag: str) -> None:
    value = _optional_float(metrics.get(field))
    if value is not None and value > 0:
        flags.append(flag)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _optional_int(value: object) -> int | None:
    number = _optional_float(value)
    return int(number) if number is not None else None


def _optional_string(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _delta(baseline: float | None, candidate: float | None) -> float | None:
    return candidate - baseline if baseline is not None and candidate is not None else None


def _delta_pct(baseline: float | None, candidate: float | None) -> float | None:
    if baseline is None or candidate is None or baseline == 0:
        return None
    return ((candidate - baseline) / baseline) * 100


def write_comparisons(rows: list[ComparisonRow], output_path: Path, export_format: ExportFormat) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.json_normalize([row.model_dump() for row in rows], sep=".")
    table = _normalize_table_cells(table)
    if export_format == ExportFormat.parquet:
        table.to_parquet(output_path, index=False)
    elif export_format == ExportFormat.csv:
        table.to_csv(output_path, index=False)
    else:
        table.to_json(output_path, orient="records", lines=True)


def _normalize_table_cells(table: pd.DataFrame) -> pd.DataFrame:
    normalized = table.copy()
    for column in normalized.columns:
        if normalized[column].map(_is_nested_cell).any():
            normalized[column] = normalized[column].map(_json_cell)
    return normalized


def _is_nested_cell(value: object) -> bool:
    return isinstance(value, dict | list)


def _json_cell(value: object) -> object:
    if not _is_nested_cell(value):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def render_result(result: ComparisonResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [
        f"Compared {result.comparison_count} workload(s): "
        f"viable={result.summary.candidate_verdict_counts.get(CandidateVerdict.candidate_viable.value, 0)}, "
        f"review={result.summary.candidate_verdict_counts.get(CandidateVerdict.review.value, 0)}, "
        f"reject={result.summary.candidate_verdict_counts.get(CandidateVerdict.reject.value, 0)}"
    ]
    for row in result.comparisons:
        lines.append(
            f"- {row.workload_id}: verdict={row.candidate_verdict or 'unknown'} "
            f"(safety={row.safety_verdict or 'unknown'}, "
            f"value_protection={row.value_protection_verdict or 'unknown'}, "
            f"signature_parity={row.signature_parity_verdict or 'unknown'}, "
            f"performance={row.performance_verdict or 'unknown'}), "
            f"replacement={row.baseline_replacement_strategy or 'unknown'}"
            f"->{row.candidate_replacement_strategy or 'unknown'}, "
            f"elapsed {_number_pair(row.baseline_pipeline_elapsed_sec, row.candidate_pipeline_elapsed_sec, suffix='s')}, "
            f"entities {row.baseline_final_entity_count}->{row.candidate_final_entity_count}, "
            f"requests {_number_pair(row.baseline_observed_total_requests, row.candidate_observed_total_requests)}, "
            f"tokens {_number_pair(row.baseline_observed_total_tokens, row.candidate_observed_total_tokens)}, "
            f"original_value_leaks "
            f"{_number_pair(row.baseline_original_value_leak_count, row.candidate_original_value_leak_count)}, "
            f"lost_labels={_label_count_summary(row.baseline_only_final_entity_signature_label_counts)}, "
            "covered_label_mismatch_labels="
            f"{_label_count_summary(row.baseline_only_candidate_label_mismatch_signature_label_counts)}, "
            f"unstable_lost_labels={_label_count_summary(row.baseline_stable_candidate_unstable_final_entity_signature_label_counts)}, "
            f"leak_labels={_label_count_summary(row.candidate_original_value_leak_label_counts)}, "
            f"flags={','.join(row.flags) if row.flags else 'none'}"
        )
    return "\n".join(lines)


def _number_pair(baseline: float | None, candidate: float | None, *, suffix: str = "") -> str:
    return f"{_format_number(baseline, suffix=suffix)}->{_format_number(candidate, suffix=suffix)}"


def _format_number(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "unknown"
    return f"{value:.1f}{suffix}"


def _label_count_summary(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ",".join(f"{label}:{count}" for label, count in sorted(counts.items()))


def _selector_label(*, config: str | None, strategy: str | None) -> str:
    if config is not None:
        return f"config:{config}"
    if strategy is not None:
        return f"strategy:{strategy}"
    return "<unset>"


@app.default
def main(
    case_analysis: Path,
    *,
    candidate_case_analysis: Annotated[Path | None, cyclopts.Parameter("--candidate-case-analysis")] = None,
    baseline_config: Annotated[str | None, cyclopts.Parameter("--baseline-config")] = None,
    candidate_config: Annotated[str | None, cyclopts.Parameter("--candidate-config")] = None,
    baseline_strategy: Annotated[str | None, cyclopts.Parameter("--baseline-strategy")] = None,
    candidate_strategy: Annotated[str | None, cyclopts.Parameter("--candidate-strategy")] = None,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.csv,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        baseline_table = read_case_analysis(case_analysis)
        candidate_table = read_case_analysis(candidate_case_analysis) if candidate_case_analysis else baseline_table
        comparisons = compare_case_tables(
            baseline_table,
            candidate_table,
            baseline_config=baseline_config,
            candidate_config=candidate_config,
            baseline_strategy=baseline_strategy,
            candidate_strategy=candidate_strategy,
        )
    except ValueError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    result = ComparisonResult(
        input_path=str(case_analysis),
        candidate_input_path=str(candidate_case_analysis) if candidate_case_analysis else None,
        baseline_selector=_selector_label(config=baseline_config, strategy=baseline_strategy),
        candidate_selector=_selector_label(config=candidate_config, strategy=candidate_strategy),
        summary=summarize_comparisons(comparisons),
        comparisons=comparisons,
    )
    if output is not None:
        write_comparisons(comparisons, output, format)
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
