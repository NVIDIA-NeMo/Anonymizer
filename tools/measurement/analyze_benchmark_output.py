#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze joined benchmark measurements and detection artifact sidecars.

Usage:
    uv run python tools/measurement/analyze_benchmark_output.py benchmark-runs/suite-id
    uv run python tools/measurement/analyze_benchmark_output.py benchmark-runs/suite-id --output analysis
    uv run python tools/measurement/analyze_benchmark_output.py benchmark-runs/suite-id --detection-artifacts current.jsonl
    uv run python tools/measurement/analyze_benchmark_output.py benchmark-runs/suite-id --json
"""

from __future__ import annotations

import json
import logging
import math
import sys
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import cyclopts
import pandas as pd
from pydantic import BaseModel, Field, computed_field

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.benchmark_output")

_SYNC_CLIENT_UNAVAILABLE_ERROR = "SyncClientUnavailableError"
_SIGNATURE_DETAIL_FIELDS = {
    "label",
    "source",
    "row_index",
    "start_position",
    "end_position",
    "value_length",
}


class ExportFormat(StrEnum):
    parquet = "parquet"
    csv = "csv"
    jsonl = "jsonl"


class LogFormat(StrEnum):
    plain = "plain"
    json = "json"


_log_format = LogFormat.plain


class CaseAnalysisRow(BaseModel):
    suite_id: str | None = None
    workload_id: str | None = None
    workload_category: str | None = None
    config_id: str | None = None
    experimental_detection_strategy: str | None = None
    experimental_replacement_strategy: str | None = None
    dd_parser_compat: str | None = None
    entity_label_set_id: str | None = None
    entity_label_count: int | None = None
    gliner_threshold: float | None = None
    repetition: int | None = None
    case_id: str
    run_id: str
    case_failed: bool = False
    error_stage_count: int = 0
    error_ndd_workflow_count: int = 0
    error_model_workflow_count: int = 0
    pipeline_elapsed_sec: float | None = None
    ndd_workflow_count: int = 0
    ndd_elapsed_sec_total: float = 0.0
    observed_total_requests: int = 0
    observed_successful_requests: int = 0
    observed_input_tokens: int = 0
    observed_output_tokens: int = 0
    observed_total_tokens: int = 0
    observed_failed_requests: int = 0
    observed_failed_request_rate: float | None = None
    dd_trace_record_count: int = 0
    dd_trace_error_count: int = 0
    dd_trace_sync_client_unavailable_count: int = 0
    observed_bridge_fallback_requests: int | None = None
    observed_non_bridge_total_requests: int | None = None
    observed_non_bridge_failed_requests: int | None = None
    observed_non_bridge_failed_request_rate: float | None = None
    record_count: int = 0
    input_text_tokens_total: int | None = None
    records_per_pipeline_sec: float | None = None
    records_per_ndd_sec: float | None = None
    input_text_tokens_per_pipeline_sec: float | None = None
    input_text_tokens_per_ndd_sec: float | None = None
    topology_endpoint_count: float | None = None
    topology_gpu_count: float | None = None
    topology_tensor_parallelism: float | None = None
    topology_shard_count: float | None = None
    input_text_tokens_per_endpoint_sec: float | None = None
    input_text_tokens_per_gpu_sec: float | None = None
    route_total_row_count: float | None = None
    route_rule_row_count: float | None = None
    route_fallback_row_count: float | None = None
    final_entity_count: float | None = None
    empty_detection_count: int = 0
    empty_detection_rate: float | None = None
    empty_detection_with_ground_truth_count: int = 0
    empty_detection_with_ground_truth_rate: float | None = None
    ground_truth_record_count: int = 0
    ground_truth_entity_count: float | None = None
    entity_true_positive_count: float | None = None
    entity_false_positive_count: float | None = None
    entity_false_negative_count: float | None = None
    entity_precision: float | None = None
    entity_recall: float | None = None
    entity_f1: float | None = None
    entity_relaxed_gt_found_count: float | None = None
    entity_relaxed_detected_tp_count: float | None = None
    entity_relaxed_label_compatible_gt_found_count: float | None = None
    entity_relaxed_label_compatible_detected_tp_count: float | None = None
    entity_relaxed_precision: float | None = None
    entity_relaxed_recall: float | None = None
    entity_relaxed_f1: float | None = None
    entity_relaxed_label_compatible_precision: float | None = None
    entity_relaxed_label_compatible_recall: float | None = None
    entity_relaxed_label_compatible_f1: float | None = None
    replacement_count: float | None = None
    replacement_missing_final_entity_count: float | None = None
    replacement_missing_final_entity_label_counts: dict[str, int] = Field(default_factory=dict)
    replacement_missing_final_value_count: float | None = None
    replacement_synthetic_original_collision_count: float | None = None
    replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)
    replacement_synthetic_original_collision_value_count: float | None = None
    original_value_leak_count: float | None = None
    original_value_leak_record_count: int = 0
    original_value_leak_label_counts: dict[str, int] = Field(default_factory=dict)
    validation_max_entities_per_call: int | None = None
    detection_artifact_rows: int = 0
    seed_entity_count: float | None = None
    seed_validation_candidate_count: float | None = None
    estimated_seed_validation_chunk_count: float | None = None
    augmented_entity_count: float | None = None
    augmented_new_final_value_count: float | None = None
    artifact_final_entity_count: float | None = None
    artifact_final_detector_entity_count: float | None = None
    artifact_final_rule_entity_count: float | None = None
    artifact_final_augmenter_entity_count: float | None = None
    artifact_final_entity_signature_count: float | None = None
    artifact_final_entity_signature_hashes: list[str] = Field(default_factory=list)
    artifact_final_entity_signature_labels: dict[str, str] = Field(default_factory=dict)
    artifact_final_entity_signature_details: dict[str, dict[str, Any]] = Field(default_factory=dict)


class GroupAnalysisRow(BaseModel):
    workload_id: str | None = None
    workload_category: str | None = None
    config_id: str | None = None
    experimental_detection_strategy: str | None = None
    experimental_replacement_strategy: str | None = None
    entity_label_set_id: str | None = None
    entity_label_count: int | None = None
    gliner_threshold: float | None = None
    case_count: int
    failed_case_count: int = 0
    failed_case_rate: float | None = None
    error_stage_count: int = 0
    error_ndd_workflow_count: int = 0
    error_model_workflow_count: int = 0
    median_pipeline_elapsed_sec: float | None = None
    median_ndd_elapsed_sec_total: float | None = None
    median_observed_total_requests: float | None = None
    median_observed_successful_requests: float | None = None
    median_observed_input_tokens: float | None = None
    median_observed_output_tokens: float | None = None
    median_observed_total_tokens: float | None = None
    median_observed_failed_requests: float | None = None
    median_observed_failed_request_rate: float | None = None
    median_observed_bridge_fallback_requests: float | None = None
    median_observed_non_bridge_total_requests: float | None = None
    median_observed_non_bridge_failed_requests: float | None = None
    median_observed_non_bridge_failed_request_rate: float | None = None
    total_record_count: int = 0
    median_record_count: float | None = None
    total_input_text_tokens: int | None = None
    median_input_text_tokens_total: float | None = None
    median_records_per_pipeline_sec: float | None = None
    median_records_per_ndd_sec: float | None = None
    median_input_text_tokens_per_pipeline_sec: float | None = None
    median_input_text_tokens_per_ndd_sec: float | None = None
    median_topology_endpoint_count: float | None = None
    median_topology_gpu_count: float | None = None
    median_topology_tensor_parallelism: float | None = None
    median_topology_shard_count: float | None = None
    median_input_text_tokens_per_endpoint_sec: float | None = None
    median_input_text_tokens_per_gpu_sec: float | None = None
    median_route_total_row_count: float | None = None
    median_route_rule_row_count: float | None = None
    median_route_fallback_row_count: float | None = None
    median_final_entity_count: float | None = None
    total_empty_detection_count: int = 0
    empty_detection_rate: float | None = None
    total_empty_detection_with_ground_truth_count: int = 0
    empty_detection_with_ground_truth_rate: float | None = None
    total_ground_truth_record_count: int = 0
    sum_ground_truth_entity_count: float | None = None
    sum_entity_true_positive_count: float | None = None
    sum_entity_false_positive_count: float | None = None
    sum_entity_false_negative_count: float | None = None
    micro_entity_precision: float | None = None
    micro_entity_recall: float | None = None
    micro_entity_f1: float | None = None
    sum_entity_relaxed_gt_found_count: float | None = None
    sum_entity_relaxed_detected_tp_count: float | None = None
    sum_entity_relaxed_label_compatible_gt_found_count: float | None = None
    sum_entity_relaxed_label_compatible_detected_tp_count: float | None = None
    micro_entity_relaxed_precision: float | None = None
    micro_entity_relaxed_recall: float | None = None
    micro_entity_relaxed_f1: float | None = None
    micro_entity_relaxed_label_compatible_precision: float | None = None
    micro_entity_relaxed_label_compatible_recall: float | None = None
    micro_entity_relaxed_label_compatible_f1: float | None = None
    median_entity_relaxed_f1: float | None = None
    median_entity_relaxed_label_compatible_f1: float | None = None
    median_replacement_missing_final_entity_count: float | None = None
    median_replacement_missing_final_value_count: float | None = None
    replacement_missing_final_entity_label_counts: dict[str, int] = Field(default_factory=dict)
    median_replacement_synthetic_original_collision_count: float | None = None
    median_replacement_synthetic_original_collision_value_count: float | None = None
    replacement_synthetic_original_collision_label_counts: dict[str, int] = Field(default_factory=dict)
    sum_original_value_leak_count: float | None = None
    leaking_case_count: int = 0
    median_original_value_leak_count: float | None = None
    median_seed_entity_count: float | None = None
    median_seed_validation_candidate_count: float | None = None
    median_estimated_seed_validation_chunk_count: float | None = None
    median_augmented_entity_count: float | None = None
    median_augmented_new_final_value_count: float | None = None
    median_artifact_final_entity_count: float | None = None
    median_artifact_final_detector_entity_count: float | None = None
    median_artifact_final_rule_entity_count: float | None = None
    median_artifact_final_augmenter_entity_count: float | None = None
    median_artifact_final_entity_signature_count: float | None = None


class ModelUsageAnalysisRow(BaseModel):
    suite_id: str | None = None
    workload_id: str | None = None
    config_id: str | None = None
    experimental_detection_strategy: str | None = None
    experimental_replacement_strategy: str | None = None
    dd_parser_compat: str | None = None
    repetition: int | None = None
    case_id: str
    run_id: str
    workflow_name: str | None = None
    model_alias: str | None = None
    model_name: str
    model_provider_name: str | None = None
    ndd_elapsed_sec: float | None = None
    observed_total_requests: int = 0
    observed_successful_requests: int = 0
    observed_failed_requests: int = 0
    observed_input_tokens: int = 0
    observed_output_tokens: int = 0
    observed_total_tokens: int = 0
    observed_reasoning_tokens: int | None = None
    observed_failed_request_rate: float | None = None


class ModelUsageGroupAnalysisRow(BaseModel):
    workload_id: str | None = None
    config_id: str | None = None
    experimental_detection_strategy: str | None = None
    experimental_replacement_strategy: str | None = None
    dd_parser_compat: str | None = None
    workflow_name: str | None = None
    model_alias: str | None = None
    model_name: str
    model_provider_name: str | None = None
    case_count: int
    workflow_count: int
    sum_observed_total_requests: int = 0
    sum_observed_successful_requests: int = 0
    sum_observed_failed_requests: int = 0
    sum_observed_input_tokens: int = 0
    sum_observed_output_tokens: int = 0
    sum_observed_total_tokens: int = 0
    sum_observed_reasoning_tokens: int | None = None
    observed_failed_request_rate: float | None = None
    median_observed_total_requests: float | None = None
    median_observed_failed_requests: float | None = None
    median_observed_total_tokens: float | None = None


class BenchmarkOutputAnalysis(BaseModel):
    benchmark_dir: str
    detection_artifacts_path: str | None = None
    cases: list[CaseAnalysisRow] = Field(default_factory=list)
    groups: list[GroupAnalysisRow] = Field(default_factory=list)
    model_usage: list[ModelUsageAnalysisRow] = Field(default_factory=list)
    model_usage_groups: list[ModelUsageGroupAnalysisRow] = Field(default_factory=list)

    @computed_field
    @property
    def case_count(self) -> int:
        return len(self.cases)

    @computed_field
    @property
    def group_count(self) -> int:
        return len(self.groups)

    @computed_field
    @property
    def model_usage_count(self) -> int:
        return len(self.model_usage)

    @computed_field
    @property
    def model_usage_group_count(self) -> int:
        return len(self.model_usage_groups)


class TableSummary(BaseModel):
    table: str
    rows: int
    path: str


class AnalysisExportResult(BaseModel):
    output_dir: str
    format: ExportFormat
    tables: list[TableSummary]
    manifest_path: str


def configure_logging(log_format: LogFormat) -> None:
    global _log_format

    _log_format = log_format
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_bad_input(error: str) -> None:
    if _log_format == LogFormat.json:
        payload = {"level": "error", "event": "bad_input", "error": error}
        sys.stderr.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return
    logger.error("bad_input error=%s", error)


def analyze_benchmark_output(
    benchmark_dir: Path,
    *,
    detection_artifacts: Path | None = None,
) -> BenchmarkOutputAnalysis:
    measurements = read_jsonl_table(benchmark_dir / "measurements.jsonl", required=True)
    artifacts_path = detection_artifacts or benchmark_dir / "detection-artifacts.jsonl"
    artifacts = read_jsonl_table(artifacts_path, required=detection_artifacts is not None)
    traces = read_trace_summary_table(benchmark_dir / "traces")
    cases = [
        _build_case_row(case_id, measurements, artifacts, traces)
        for case_id in _case_ids(measurements, artifacts, traces)
    ]
    model_usage = build_model_usage_rows(measurements)
    return BenchmarkOutputAnalysis(
        benchmark_dir=str(benchmark_dir),
        detection_artifacts_path=str(artifacts_path) if not artifacts.empty else None,
        cases=cases,
        groups=build_group_rows(cases),
        model_usage=model_usage,
        model_usage_groups=build_model_usage_group_rows(model_usage),
    )


def read_jsonl_table(path: Path, *, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise ValueError(f"input path does not exist: {path}")
        return pd.DataFrame()
    if path.is_dir():
        raise ValueError(f"input path is a directory: {path}")
    raw = pd.read_json(path, lines=True)
    if raw.empty:
        return raw
    return pd.json_normalize(raw.to_dict("records"), sep=".")


def read_trace_summary_table(trace_path: Path) -> pd.DataFrame:
    """Read DD trace files into a sanitized table with no prompt/response text."""
    if not trace_path.exists():
        return pd.DataFrame()
    if trace_path.is_file():
        paths = [trace_path]
    elif trace_path.is_dir():
        paths = sorted(trace_path.rglob("*.jsonl"))
    else:
        raise ValueError(f"trace path is not a file or directory: {trace_path}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("record_type") != "dd_message_trace":
                continue
            run_tags = record.get("run_tags") if isinstance(record.get("run_tags"), dict) else {}
            rows.append(
                {
                    "record_type": "dd_message_trace",
                    "run_id": record.get("run_id"),
                    "run_tags.case_id": run_tags.get("case_id"),
                    "run_tags.workload_id": run_tags.get("workload_id"),
                    "run_tags.config_id": run_tags.get("config_id"),
                    "run_tags.experimental_detection_strategy": run_tags.get("experimental_detection_strategy"),
                    "run_tags.experimental_replacement_strategy": run_tags.get("experimental_replacement_strategy"),
                    "run_tags.dd_parser_compat": run_tags.get("dd_parser_compat"),
                    "run_tags.repetition": run_tags.get("repetition"),
                    "workflow_name": record.get("workflow_name"),
                    "model_alias": record.get("model_alias"),
                    "status": record.get("status"),
                    "error_type": record.get("error_type"),
                    "is_async": record.get("is_async"),
                }
            )
    return pd.DataFrame(rows)


def _case_ids(*frames: pd.DataFrame) -> list[str]:
    values: set[str] = set()
    for dataframe in frames:
        for column in ("run_tags.case_id", "case_id", "run_id"):
            if column in dataframe.columns:
                values.update(str(value) for value in dataframe[column].dropna().tolist())
    return sorted(values)


def _build_case_row(
    case_id: str,
    measurements: pd.DataFrame,
    artifacts: pd.DataFrame,
    traces: pd.DataFrame,
) -> CaseAnalysisRow:
    measurement_rows = _rows_for_case(measurements, case_id)
    artifact_rows = _rows_for_case(artifacts, case_id)
    trace_rows = _rows_for_case(traces, case_id)
    record_rows = _records_of_type(measurement_rows, "record")
    ndd_rows = _records_of_type(measurement_rows, "ndd_workflow")
    model_rows = _model_workflow_rows(measurement_rows)
    stage_rows = _records_of_type(measurement_rows, "stage")
    pipeline_rows = _pipeline_stage_rows(measurement_rows)
    validation_max_entities_per_call = _first_int([measurement_rows], ["detect.validation_max_entities_per_call"])
    request_metrics = _case_request_metrics(model_rows)
    pipeline_elapsed_sec = _sum_or_none(pipeline_rows, "elapsed_sec")
    ndd_elapsed_sec_total = _sum_or_zero(ndd_rows, "elapsed_sec")
    record_count = len(record_rows)
    input_text_tokens_total = _sum_int_or_none(record_rows, "text_length_tokens")
    records_per_pipeline_sec = _safe_rate(record_count, pipeline_elapsed_sec)
    records_per_ndd_sec = _safe_rate(record_count, ndd_elapsed_sec_total)
    input_text_tokens_per_pipeline_sec = _safe_rate(input_text_tokens_total, pipeline_elapsed_sec)
    input_text_tokens_per_ndd_sec = _safe_rate(input_text_tokens_total, ndd_elapsed_sec_total)
    final_entity_count = _coalesce_number(
        _sum_or_none(record_rows, "final_entity_count"),
        _sum_or_none(artifact_rows, "final_entity_count"),
    )
    return CaseAnalysisRow(
        suite_id=_first_value([measurement_rows, artifact_rows, trace_rows], ["run_tags.suite_id", "suite_id"]),
        workload_id=_first_value(
            [measurement_rows, artifact_rows, trace_rows], ["run_tags.workload_id", "workload_id"]
        ),
        workload_category=_first_value(
            [measurement_rows, artifact_rows, trace_rows],
            ["run_tags.workload_category", "run_tags.dataset_category", "workload_category", "dataset_category"],
        ),
        config_id=_first_value([measurement_rows, artifact_rows, trace_rows], ["run_tags.config_id", "config_id"]),
        experimental_detection_strategy=_first_value([measurement_rows], ["run_tags.experimental_detection_strategy"]),
        experimental_replacement_strategy=_first_value(
            [measurement_rows, trace_rows],
            ["run_tags.experimental_replacement_strategy"],
        ),
        dd_parser_compat=_first_value([measurement_rows], ["run_tags.dd_parser_compat"]),
        entity_label_set_id=_first_value(
            [measurement_rows],
            [
                "run_tags.entity_label_set_id",
                "run_tags.entity_label_set",
                "run_tags.label_set",
                "detect.entity_label_source",
            ],
        ),
        entity_label_count=_first_int([measurement_rows], ["run_tags.entity_label_count", "detect.entity_label_count"]),
        gliner_threshold=_first_float([measurement_rows], ["run_tags.gliner_threshold", "detect.gliner_threshold"]),
        repetition=_first_int([measurement_rows, artifact_rows, trace_rows], ["run_tags.repetition", "repetition"]),
        case_id=case_id,
        run_id=_first_value([measurement_rows, artifact_rows, trace_rows], ["run_id"]) or case_id,
        **_case_failure_metrics(stage_rows=stage_rows, ndd_rows=ndd_rows, model_rows=model_rows),
        pipeline_elapsed_sec=pipeline_elapsed_sec,
        ndd_workflow_count=len(ndd_rows),
        ndd_elapsed_sec_total=ndd_elapsed_sec_total,
        **request_metrics,
        **_case_trace_metrics(trace_rows, request_metrics=request_metrics),
        record_count=record_count,
        input_text_tokens_total=input_text_tokens_total,
        records_per_pipeline_sec=records_per_pipeline_sec,
        records_per_ndd_sec=records_per_ndd_sec,
        input_text_tokens_per_pipeline_sec=input_text_tokens_per_pipeline_sec,
        input_text_tokens_per_ndd_sec=input_text_tokens_per_ndd_sec,
        **_case_topology_metrics(
            measurement_rows,
            input_text_tokens_per_pipeline_sec=input_text_tokens_per_pipeline_sec,
        ),
        route_total_row_count=_sum_or_none(model_rows, "route_total_row_count"),
        route_rule_row_count=_sum_or_none(model_rows, "route_rule_row_count"),
        route_fallback_row_count=_sum_or_none(model_rows, "route_fallback_row_count"),
        final_entity_count=final_entity_count,
        **_case_empty_detection_metrics(record_rows, record_count=record_count),
        **_case_ground_truth_metrics(record_rows, final_entity_count=final_entity_count),
        replacement_count=_sum_or_none(record_rows, "replacement_count"),
        replacement_missing_final_entity_count=_sum_or_none(record_rows, "replacement_missing_final_entity_count"),
        replacement_missing_final_entity_label_counts=_sum_prefixed_ints(
            record_rows,
            "replacement_missing_final_entity_label_counts.",
        ),
        replacement_missing_final_value_count=_sum_or_none(record_rows, "replacement_missing_final_value_count"),
        replacement_synthetic_original_collision_count=_sum_or_none(
            record_rows,
            "replacement_synthetic_original_collision_count",
        ),
        replacement_synthetic_original_collision_label_counts=_sum_prefixed_ints(
            record_rows,
            "replacement_synthetic_original_collision_label_counts.",
        ),
        replacement_synthetic_original_collision_value_count=_sum_or_none(
            record_rows,
            "replacement_synthetic_original_collision_value_count",
        ),
        original_value_leak_count=_sum_or_none(record_rows, "original_value_leak_count"),
        original_value_leak_record_count=_positive_count(record_rows, "original_value_leak_count"),
        original_value_leak_label_counts=_sum_prefixed_ints(record_rows, "original_value_leak_label_counts."),
        validation_max_entities_per_call=validation_max_entities_per_call,
        **_case_artifact_metrics(
            artifact_rows,
            validation_max_entities_per_call=validation_max_entities_per_call,
        ),
    )


def _case_request_metrics(model_rows: pd.DataFrame) -> dict[str, int | float | None]:
    observed_total_requests = int(_sum_or_zero(model_rows, "observed_total_requests"))
    observed_failed_requests = int(_sum_or_zero(model_rows, "observed_failed_requests"))
    return {
        "observed_total_requests": observed_total_requests,
        "observed_successful_requests": int(_sum_or_zero(model_rows, "observed_successful_requests")),
        "observed_input_tokens": int(_sum_or_zero(model_rows, "observed_input_tokens")),
        "observed_output_tokens": int(_sum_or_zero(model_rows, "observed_output_tokens")),
        "observed_total_tokens": int(_sum_or_zero(model_rows, "observed_total_tokens")),
        "observed_failed_requests": observed_failed_requests,
        "observed_failed_request_rate": _request_failure_rate(
            failed=observed_failed_requests,
            total=observed_total_requests,
        ),
    }


def _case_trace_metrics(
    trace_rows: pd.DataFrame,
    *,
    request_metrics: dict[str, int | float | None],
) -> dict[str, int | float | None]:
    trace_record_count = len(trace_rows)
    if trace_record_count == 0:
        return {
            "dd_trace_record_count": 0,
            "dd_trace_error_count": 0,
            "dd_trace_sync_client_unavailable_count": 0,
            "observed_bridge_fallback_requests": None,
            "observed_non_bridge_total_requests": None,
            "observed_non_bridge_failed_requests": None,
            "observed_non_bridge_failed_request_rate": None,
        }

    status = trace_rows["status"].astype(str) if "status" in trace_rows.columns else pd.Series(dtype=str)
    error_type = trace_rows["error_type"].astype(str) if "error_type" in trace_rows.columns else pd.Series(dtype=str)
    error_count = int((status == "error").sum())
    bridge_fallbacks = int(((status == "error") & (error_type == _SYNC_CLIENT_UNAVAILABLE_ERROR)).sum())
    observed_total = int(request_metrics["observed_total_requests"] or 0)
    observed_failed = int(request_metrics["observed_failed_requests"] or 0)
    non_bridge_total = max(observed_total - bridge_fallbacks, 0)
    non_bridge_failed = max(observed_failed - bridge_fallbacks, 0)
    return {
        "dd_trace_record_count": trace_record_count,
        "dd_trace_error_count": error_count,
        "dd_trace_sync_client_unavailable_count": bridge_fallbacks,
        "observed_bridge_fallback_requests": bridge_fallbacks,
        "observed_non_bridge_total_requests": non_bridge_total,
        "observed_non_bridge_failed_requests": non_bridge_failed,
        "observed_non_bridge_failed_request_rate": _request_failure_rate(
            failed=non_bridge_failed,
            total=non_bridge_total,
        ),
    }


def _case_failure_metrics(
    *,
    stage_rows: pd.DataFrame,
    ndd_rows: pd.DataFrame,
    model_rows: pd.DataFrame,
) -> dict[str, bool | int]:
    error_stage_count = _error_status_count(stage_rows)
    error_ndd_workflow_count = _error_status_count(ndd_rows)
    error_model_workflow_count = _error_status_count(model_rows)
    return {
        "case_failed": error_stage_count > 0 or error_ndd_workflow_count > 0 or error_model_workflow_count > 0,
        "error_stage_count": error_stage_count,
        "error_ndd_workflow_count": error_ndd_workflow_count,
        "error_model_workflow_count": error_model_workflow_count,
    }


def _case_topology_metrics(
    measurement_rows: pd.DataFrame,
    *,
    input_text_tokens_per_pipeline_sec: float | None,
) -> dict[str, float | None]:
    endpoint_count = _first_float(
        [measurement_rows],
        [
            "run_tags.topology_endpoint_count",
            "run_tags.endpoint_count",
            "run_tags.n_endpoints",
            "run_tags.n_llm_endpoints",
        ],
    )
    gpu_count = _first_float(
        [measurement_rows],
        [
            "run_tags.topology_gpu_count",
            "run_tags.gpu_count",
            "run_tags.n_gpus",
            "run_tags.n_llm_gpus",
        ],
    )
    tensor_parallelism = _first_float(
        [measurement_rows],
        [
            "run_tags.topology_tensor_parallelism",
            "run_tags.tensor_parallelism",
            "run_tags.gpus_per_endpoint",
            "run_tags.tp",
        ],
    )
    shard_count = _first_float(
        [measurement_rows],
        ["run_tags.topology_shard_count", "run_tags.shard_count", "run_tags.n_shards"],
    )
    return {
        "topology_endpoint_count": endpoint_count,
        "topology_gpu_count": gpu_count,
        "topology_tensor_parallelism": tensor_parallelism,
        "topology_shard_count": shard_count,
        "input_text_tokens_per_endpoint_sec": _safe_ratio(input_text_tokens_per_pipeline_sec, endpoint_count),
        "input_text_tokens_per_gpu_sec": _safe_ratio(input_text_tokens_per_pipeline_sec, gpu_count),
    }


def _case_empty_detection_metrics(record_rows: pd.DataFrame, *, record_count: int) -> dict[str, int | float | None]:
    empty_detection_count = _zero_count(record_rows, "final_entity_count")
    ground_truth_record_count = _non_null_count(record_rows, "ground_truth_entity_count")
    empty_detection_with_gt_count = _zero_with_positive_count(
        record_rows,
        zero_column="final_entity_count",
        positive_column="ground_truth_entity_count",
    )
    return {
        "empty_detection_count": empty_detection_count,
        "empty_detection_rate": _safe_ratio(empty_detection_count, record_count),
        "empty_detection_with_ground_truth_count": empty_detection_with_gt_count,
        "empty_detection_with_ground_truth_rate": _safe_ratio(
            empty_detection_with_gt_count,
            ground_truth_record_count,
        ),
        "ground_truth_record_count": ground_truth_record_count,
    }


def _case_ground_truth_metrics(
    record_rows: pd.DataFrame,
    *,
    final_entity_count: float | None,
) -> dict[str, float | None]:
    ground_truth_entity_count = _sum_or_none(record_rows, "ground_truth_entity_count")
    true_positive = _sum_or_none(record_rows, "entity_true_positive_count")
    false_positive = _sum_or_none(record_rows, "entity_false_positive_count")
    false_negative = _sum_or_none(record_rows, "entity_false_negative_count")
    relaxed_gt_found = _sum_or_none(record_rows, "entity_relaxed_gt_found_count")
    relaxed_detected_tp = _sum_or_none(record_rows, "entity_relaxed_detected_tp_count")
    label_compatible_gt_found = _sum_or_none(record_rows, "entity_relaxed_label_compatible_gt_found_count")
    label_compatible_detected_tp = _sum_or_none(
        record_rows,
        "entity_relaxed_label_compatible_detected_tp_count",
    )
    strict_precision = _safe_ratio(true_positive, _sum_optional_numbers(true_positive, false_positive))
    strict_recall = _safe_ratio(true_positive, _sum_optional_numbers(true_positive, false_negative))
    relaxed_precision = _safe_ratio(relaxed_detected_tp, final_entity_count)
    relaxed_recall = _safe_ratio(relaxed_gt_found, ground_truth_entity_count)
    label_compatible_precision = _safe_ratio(label_compatible_detected_tp, final_entity_count)
    label_compatible_recall = _safe_ratio(label_compatible_gt_found, ground_truth_entity_count)
    return {
        "ground_truth_entity_count": ground_truth_entity_count,
        "entity_true_positive_count": true_positive,
        "entity_false_positive_count": false_positive,
        "entity_false_negative_count": false_negative,
        "entity_precision": strict_precision,
        "entity_recall": strict_recall,
        "entity_f1": _f1(strict_precision, strict_recall),
        "entity_relaxed_gt_found_count": relaxed_gt_found,
        "entity_relaxed_detected_tp_count": relaxed_detected_tp,
        "entity_relaxed_label_compatible_gt_found_count": label_compatible_gt_found,
        "entity_relaxed_label_compatible_detected_tp_count": label_compatible_detected_tp,
        "entity_relaxed_precision": relaxed_precision,
        "entity_relaxed_recall": relaxed_recall,
        "entity_relaxed_f1": _f1(relaxed_precision, relaxed_recall),
        "entity_relaxed_label_compatible_precision": label_compatible_precision,
        "entity_relaxed_label_compatible_recall": label_compatible_recall,
        "entity_relaxed_label_compatible_f1": _f1(label_compatible_precision, label_compatible_recall),
    }


def _error_status_count(rows: pd.DataFrame) -> int:
    if "status" not in rows.columns:
        return 0
    statuses = rows["status"].astype(str).str.lower()
    return int(statuses.isin({"error", "failed"}).sum())


def _case_artifact_metrics(
    artifact_rows: pd.DataFrame,
    *,
    validation_max_entities_per_call: int | None,
) -> dict[str, int | float | list[str] | dict[str, str] | None]:
    signature_hashes = _artifact_signature_hashes(artifact_rows)
    return {
        "detection_artifact_rows": len(artifact_rows),
        "seed_entity_count": _sum_or_none(artifact_rows, "seed_entity_count"),
        "seed_validation_candidate_count": _sum_or_none(artifact_rows, "seed_validation_candidate_count"),
        "estimated_seed_validation_chunk_count": _estimated_validation_chunk_count(
            artifact_rows,
            validation_max_entities_per_call=validation_max_entities_per_call,
        ),
        "augmented_entity_count": _sum_or_none(artifact_rows, "augmented_entity_count"),
        "augmented_new_final_value_count": _sum_or_none(artifact_rows, "augmented_new_final_value_count"),
        "artifact_final_entity_count": _sum_or_none(artifact_rows, "final_entity_count"),
        "artifact_final_detector_entity_count": _sum_or_none(artifact_rows, "final_source_counts.detector"),
        "artifact_final_rule_entity_count": _sum_or_none(artifact_rows, "final_source_counts.rule"),
        "artifact_final_augmenter_entity_count": _sum_or_none(artifact_rows, "final_source_counts.augmenter"),
        "artifact_final_entity_signature_count": _signature_count(artifact_rows, signature_hashes=signature_hashes),
        "artifact_final_entity_signature_hashes": signature_hashes,
        "artifact_final_entity_signature_labels": _artifact_signature_labels(artifact_rows),
        "artifact_final_entity_signature_details": _artifact_signature_details(artifact_rows),
    }


def _rows_for_case(dataframe: pd.DataFrame, case_id: str) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe
    masks = [
        dataframe[column].astype(str) == case_id
        for column in ("run_tags.case_id", "case_id", "run_id")
        if column in dataframe.columns
    ]
    if not masks:
        return dataframe.iloc[0:0]
    mask = masks[0]
    for next_mask in masks[1:]:
        mask = mask | next_mask
    return dataframe[mask]


def _records_of_type(dataframe: pd.DataFrame, record_type: str) -> pd.DataFrame:
    if "record_type" not in dataframe.columns:
        return dataframe.iloc[0:0]
    return dataframe[dataframe["record_type"] == record_type]


def _records_of_types(dataframe: pd.DataFrame, record_types: set[str]) -> pd.DataFrame:
    if "record_type" not in dataframe.columns:
        return dataframe.iloc[0:0]
    return dataframe[dataframe["record_type"].isin(record_types)]


def _model_workflow_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    return _records_of_types(dataframe, {"ndd_workflow", "model_workflow"})


def _pipeline_stage_rows(dataframe: pd.DataFrame) -> pd.DataFrame:
    stages = _records_of_type(dataframe, "stage")
    if "stage" not in stages.columns:
        return stages.iloc[0:0]
    return stages[stages["stage"] == "Anonymizer._run_internal"]


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
    model_usage_keys = _model_usage_keys(model_rows.columns)
    rows: list[ModelUsageAnalysisRow] = []
    for _, measurement in model_rows.iterrows():
        data = measurement.to_dict()
        case_id = _string_from_row(data, ["run_tags.case_id", "run_id"])
        run_id = _string_from_row(data, ["run_id", "run_tags.case_id"])
        if case_id is None or run_id is None:
            continue
        for model_usage_key in model_usage_keys:
            usage = _model_usage_metrics(data, model_usage_key)
            if not _has_observed_model_usage(usage):
                continue
            metadata = _model_usage_metadata(data, model_usage_key)
            rows.append(
                ModelUsageAnalysisRow(
                    suite_id=_string_from_row(data, ["run_tags.suite_id"]),
                    workload_id=_string_from_row(data, ["run_tags.workload_id"]),
                    config_id=_string_from_row(data, ["run_tags.config_id"]),
                    experimental_detection_strategy=_string_from_row(
                        data, ["run_tags.experimental_detection_strategy"]
                    ),
                    experimental_replacement_strategy=_string_from_row(
                        data, ["run_tags.experimental_replacement_strategy"]
                    ),
                    dd_parser_compat=_string_from_row(data, ["run_tags.dd_parser_compat"]),
                    repetition=_int_from_row(data, ["run_tags.repetition"]),
                    case_id=case_id,
                    run_id=run_id,
                    workflow_name=_string_from_row(data, ["workflow_name"]),
                    model_alias=metadata.get("model_alias"),
                    model_name=metadata.get("model_name") or model_usage_key,
                    model_provider_name=metadata.get("model_provider_name"),
                    ndd_elapsed_sec=_float_from_row(data, ["elapsed_sec"]),
                    **usage,
                )
            )
    return rows


def _model_usage_keys(columns: pd.Index) -> list[str]:
    keys: set[str] = set()
    for column in columns:
        parsed = _model_usage_column_parts(str(column))
        if parsed is not None:
            keys.add(parsed[0])
    return sorted(keys)


def _model_usage_column_parts(column: str) -> tuple[str, str] | None:
    prefix = "model_usage."
    if not column.startswith(prefix):
        return None
    for suffix, metric in {**_MODEL_USAGE_SUFFIXES, **_MODEL_USAGE_METADATA_SUFFIXES}.items():
        if column.endswith(suffix):
            return column[len(prefix) : -len(suffix)], metric
    return None


def _model_usage_metrics(data: dict[str, Any], model_usage_key: str) -> dict[str, int | float | None]:
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
        values[metric] = _coerce_int(value)
    values["observed_failed_request_rate"] = _request_failure_rate(
        failed=values["observed_failed_requests"],
        total=values["observed_total_requests"],
    )
    return values


def _model_usage_metadata(data: dict[str, Any], model_usage_key: str) -> dict[str, str | None]:
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


def _has_observed_model_usage(usage: dict[str, int | float | None]) -> bool:
    return any(value not in (None, 0) for value in usage.values())


def _string_from_row(data: dict[str, Any], columns: list[str]) -> str | None:
    for column in columns:
        value = data.get(column)
        if value is not None and not pd.isna(value):
            return str(value)
    return None


def _int_from_row(data: dict[str, Any], columns: list[str]) -> int | None:
    value = _string_from_row(data, columns)
    return int(float(value)) if value is not None else None


def _float_from_row(data: dict[str, Any], columns: list[str]) -> float | None:
    value = _string_from_row(data, columns)
    return float(value) if value is not None else None


def _coerce_int(value: Any) -> int:
    return int(float(value))


def _first_value(frames: list[pd.DataFrame], columns: list[str]) -> str | None:
    for frame in frames:
        for column in columns:
            if column not in frame.columns:
                continue
            values = frame[column].dropna()
            if not values.empty:
                return str(values.iloc[0])
    return None


def _first_int(frames: list[pd.DataFrame], columns: list[str]) -> int | None:
    value = _first_value(frames, columns)
    return int(float(value)) if value is not None else None


def _first_float(frames: list[pd.DataFrame], columns: list[str]) -> float | None:
    value = _first_value(frames, columns)
    return float(value) if value is not None else None


def _coalesce_number(*values: float | None) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _artifact_signature_hashes(artifact_rows: pd.DataFrame) -> list[str]:
    if "final_entity_signature_hashes" not in artifact_rows.columns:
        return []
    values: set[str] = set()
    for raw in artifact_rows["final_entity_signature_hashes"].dropna():
        values.update(_coerce_string_list(raw))
    return sorted(values)


def _artifact_signature_labels(artifact_rows: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    if "final_entity_signature_labels" in artifact_rows.columns:
        for raw in artifact_rows["final_entity_signature_labels"].dropna():
            labels.update(_coerce_string_dict(raw))
    for column in artifact_rows.columns:
        prefix = "final_entity_signature_labels."
        if not column.startswith(prefix):
            continue
        signature_hash = column.removeprefix(prefix)
        for value in artifact_rows[column].dropna():
            labels[signature_hash] = str(value)
    return dict(sorted(labels.items()))


def _artifact_signature_details(artifact_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    if "final_entity_signature_details" in artifact_rows.columns:
        for raw in artifact_rows["final_entity_signature_details"].dropna():
            details.update(_coerce_detail_map(raw))
    prefix = "final_entity_signature_details."
    for column in artifact_rows.columns:
        if not column.startswith(prefix):
            continue
        remainder = column.removeprefix(prefix)
        signature_hash, _, field = remainder.partition(".")
        if not signature_hash or not field:
            continue
        if field not in _SIGNATURE_DETAIL_FIELDS:
            continue
        for value in artifact_rows[column].dropna():
            details.setdefault(signature_hash, {})[field] = _json_scalar(value)
    return dict(sorted(details.items()))


def _coerce_detail_map(raw: object) -> dict[str, dict[str, Any]]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    if not isinstance(raw, dict):
        return {}
    details: dict[str, dict[str, Any]] = {}
    for signature_hash, value in raw.items():
        if isinstance(value, dict):
            details[str(signature_hash)] = {
                str(key): _json_scalar(item) for key, item in value.items() if str(key) in _SIGNATURE_DETAIL_FIELDS
            }
    return details


def _json_scalar(value: object) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def _coerce_string_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def _coerce_string_dict(raw: object) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return {}


def _signature_count(artifact_rows: pd.DataFrame, *, signature_hashes: list[str]) -> float | None:
    if signature_hashes:
        return float(len(signature_hashes))
    return _sum_or_none(artifact_rows, "final_entity_signature_count")


def _sum_or_zero(dataframe: pd.DataFrame, column: str) -> float:
    if column not in dataframe.columns:
        return 0.0
    return float(pd.to_numeric(dataframe[column], errors="coerce").fillna(0).sum())


def _sum_or_none(dataframe: pd.DataFrame, column: str) -> float | None:
    if column not in dataframe.columns:
        return None
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


def _positive_count(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    values = pd.to_numeric(dataframe[column], errors="coerce").fillna(0)
    return int((values > 0).sum())


def _zero_count(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    return int((values == 0).sum())


def _non_null_count(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    return int(pd.to_numeric(dataframe[column], errors="coerce").notna().sum())


def _zero_with_positive_count(dataframe: pd.DataFrame, *, zero_column: str, positive_column: str) -> int:
    if zero_column not in dataframe.columns or positive_column not in dataframe.columns:
        return 0
    zero_values = pd.to_numeric(dataframe[zero_column], errors="coerce")
    positive_values = pd.to_numeric(dataframe[positive_column], errors="coerce")
    return int(((zero_values == 0) & (positive_values > 0)).sum())


def _sum_prefixed_ints(dataframe: pd.DataFrame, prefix: str) -> dict[str, int]:
    totals: dict[str, int] = {}
    base_column = prefix.removesuffix(".")
    if base_column in dataframe.columns:
        for value in dataframe[base_column].dropna().tolist():
            for key, count in _coerce_count_mapping(value).items():
                totals[key] = totals.get(key, 0) + count
    for column in sorted(col for col in dataframe.columns if col.startswith(prefix)):
        value = _sum_int_or_zero(dataframe, column)
        if value:
            totals[column.removeprefix(prefix)] = value
    return totals


def _coerce_count_mapping(value: object) -> dict[str, int]:
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


def build_group_rows(cases: list[CaseAnalysisRow]) -> list[GroupAnalysisRow]:
    if not cases:
        return []
    table = pd.DataFrame([case.model_dump() for case in cases])
    rows: list[GroupAnalysisRow] = []
    group_columns = [
        "workload_id",
        "workload_category",
        "config_id",
        "experimental_detection_strategy",
        "experimental_replacement_strategy",
        "entity_label_set_id",
        "entity_label_count",
        "gliner_threshold",
    ]
    for keys, group in table.groupby(group_columns, dropna=False):
        rows.append(_build_group_row(keys, group))
    return rows


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
        rows.append(_build_model_usage_group_row(keys, group))
    return rows


def _build_model_usage_group_row(keys: tuple[Any, ...], group: pd.DataFrame) -> ModelUsageGroupAnalysisRow:
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


def _build_group_row(keys: tuple[Any, ...], group: pd.DataFrame) -> GroupAnalysisRow:
    (
        workload_id,
        workload_category,
        config_id,
        detection_strategy,
        replacement_strategy,
        entity_label_set_id,
        entity_label_count,
        gliner_threshold,
    ) = keys
    case_count = int(group["case_id"].nunique())
    failed_case_count = _sum_bool_or_zero(group, "case_failed")
    total_record_count = _sum_int_or_zero(group, "record_count")
    total_input_text_tokens = _sum_int_or_none(group, "input_text_tokens_total")
    total_empty_detection_count = _sum_int_or_zero(group, "empty_detection_count")
    total_ground_truth_record_count = _sum_int_or_zero(group, "ground_truth_record_count")
    total_empty_detection_with_gt_count = _sum_int_or_zero(group, "empty_detection_with_ground_truth_count")
    final_entity_count = _sum_or_none(group, "final_entity_count")
    ground_truth_entity_count = _sum_or_none(group, "ground_truth_entity_count")
    true_positive = _sum_or_none(group, "entity_true_positive_count")
    false_positive = _sum_or_none(group, "entity_false_positive_count")
    false_negative = _sum_or_none(group, "entity_false_negative_count")
    strict_precision = _safe_ratio(true_positive, _sum_optional_numbers(true_positive, false_positive))
    strict_recall = _safe_ratio(true_positive, _sum_optional_numbers(true_positive, false_negative))
    relaxed_gt_found = _sum_or_none(group, "entity_relaxed_gt_found_count")
    relaxed_detected_tp = _sum_or_none(group, "entity_relaxed_detected_tp_count")
    label_compatible_gt_found = _sum_or_none(group, "entity_relaxed_label_compatible_gt_found_count")
    label_compatible_detected_tp = _sum_or_none(group, "entity_relaxed_label_compatible_detected_tp_count")
    relaxed_precision = _safe_ratio(relaxed_detected_tp, final_entity_count)
    relaxed_recall = _safe_ratio(relaxed_gt_found, ground_truth_entity_count)
    label_compatible_precision = _safe_ratio(label_compatible_detected_tp, final_entity_count)
    label_compatible_recall = _safe_ratio(label_compatible_gt_found, ground_truth_entity_count)
    return GroupAnalysisRow(
        workload_id=_none_if_nan(workload_id),
        workload_category=_none_if_nan(workload_category),
        config_id=_none_if_nan(config_id),
        experimental_detection_strategy=_none_if_nan(detection_strategy),
        experimental_replacement_strategy=_none_if_nan(replacement_strategy),
        entity_label_set_id=_none_if_nan(entity_label_set_id),
        entity_label_count=_int_if_not_nan(entity_label_count),
        gliner_threshold=_float_if_not_nan(gliner_threshold),
        case_count=case_count,
        failed_case_count=failed_case_count,
        failed_case_rate=_request_failure_rate(failed=failed_case_count, total=case_count),
        error_stage_count=_sum_int_or_zero(group, "error_stage_count"),
        error_ndd_workflow_count=_sum_int_or_zero(group, "error_ndd_workflow_count"),
        error_model_workflow_count=_sum_int_or_zero(group, "error_model_workflow_count"),
        median_pipeline_elapsed_sec=_median_or_none(group, "pipeline_elapsed_sec"),
        median_ndd_elapsed_sec_total=_median_or_none(group, "ndd_elapsed_sec_total"),
        median_observed_total_requests=_median_or_none(group, "observed_total_requests"),
        median_observed_successful_requests=_median_or_none(group, "observed_successful_requests"),
        median_observed_input_tokens=_median_or_none(group, "observed_input_tokens"),
        median_observed_output_tokens=_median_or_none(group, "observed_output_tokens"),
        median_observed_total_tokens=_median_or_none(group, "observed_total_tokens"),
        median_observed_failed_requests=_median_or_none(group, "observed_failed_requests"),
        median_observed_failed_request_rate=_median_or_none(group, "observed_failed_request_rate"),
        median_observed_bridge_fallback_requests=_median_or_none(group, "observed_bridge_fallback_requests"),
        median_observed_non_bridge_total_requests=_median_or_none(group, "observed_non_bridge_total_requests"),
        median_observed_non_bridge_failed_requests=_median_or_none(group, "observed_non_bridge_failed_requests"),
        median_observed_non_bridge_failed_request_rate=_median_or_none(
            group,
            "observed_non_bridge_failed_request_rate",
        ),
        total_record_count=total_record_count,
        median_record_count=_median_or_none(group, "record_count"),
        total_input_text_tokens=total_input_text_tokens,
        median_input_text_tokens_total=_median_or_none(group, "input_text_tokens_total"),
        median_records_per_pipeline_sec=_median_or_none(group, "records_per_pipeline_sec"),
        median_records_per_ndd_sec=_median_or_none(group, "records_per_ndd_sec"),
        median_input_text_tokens_per_pipeline_sec=_median_or_none(group, "input_text_tokens_per_pipeline_sec"),
        median_input_text_tokens_per_ndd_sec=_median_or_none(group, "input_text_tokens_per_ndd_sec"),
        median_topology_endpoint_count=_median_or_none(group, "topology_endpoint_count"),
        median_topology_gpu_count=_median_or_none(group, "topology_gpu_count"),
        median_topology_tensor_parallelism=_median_or_none(group, "topology_tensor_parallelism"),
        median_topology_shard_count=_median_or_none(group, "topology_shard_count"),
        median_input_text_tokens_per_endpoint_sec=_median_or_none(group, "input_text_tokens_per_endpoint_sec"),
        median_input_text_tokens_per_gpu_sec=_median_or_none(group, "input_text_tokens_per_gpu_sec"),
        median_route_total_row_count=_median_or_none(group, "route_total_row_count"),
        median_route_rule_row_count=_median_or_none(group, "route_rule_row_count"),
        median_route_fallback_row_count=_median_or_none(group, "route_fallback_row_count"),
        median_final_entity_count=_median_or_none(group, "final_entity_count"),
        total_empty_detection_count=total_empty_detection_count,
        empty_detection_rate=_safe_ratio(total_empty_detection_count, total_record_count),
        total_empty_detection_with_ground_truth_count=total_empty_detection_with_gt_count,
        empty_detection_with_ground_truth_rate=_safe_ratio(
            total_empty_detection_with_gt_count,
            total_ground_truth_record_count,
        ),
        total_ground_truth_record_count=total_ground_truth_record_count,
        sum_ground_truth_entity_count=ground_truth_entity_count,
        sum_entity_true_positive_count=true_positive,
        sum_entity_false_positive_count=false_positive,
        sum_entity_false_negative_count=false_negative,
        micro_entity_precision=strict_precision,
        micro_entity_recall=strict_recall,
        micro_entity_f1=_f1(strict_precision, strict_recall),
        sum_entity_relaxed_gt_found_count=relaxed_gt_found,
        sum_entity_relaxed_detected_tp_count=relaxed_detected_tp,
        sum_entity_relaxed_label_compatible_gt_found_count=label_compatible_gt_found,
        sum_entity_relaxed_label_compatible_detected_tp_count=label_compatible_detected_tp,
        micro_entity_relaxed_precision=relaxed_precision,
        micro_entity_relaxed_recall=relaxed_recall,
        micro_entity_relaxed_f1=_f1(relaxed_precision, relaxed_recall),
        micro_entity_relaxed_label_compatible_precision=label_compatible_precision,
        micro_entity_relaxed_label_compatible_recall=label_compatible_recall,
        micro_entity_relaxed_label_compatible_f1=_f1(label_compatible_precision, label_compatible_recall),
        median_entity_relaxed_f1=_median_or_none(group, "entity_relaxed_f1"),
        median_entity_relaxed_label_compatible_f1=_median_or_none(
            group,
            "entity_relaxed_label_compatible_f1",
        ),
        median_replacement_missing_final_entity_count=_median_or_none(
            group,
            "replacement_missing_final_entity_count",
        ),
        median_replacement_missing_final_value_count=_median_or_none(group, "replacement_missing_final_value_count"),
        replacement_missing_final_entity_label_counts=_sum_prefixed_ints(
            group,
            "replacement_missing_final_entity_label_counts.",
        ),
        median_replacement_synthetic_original_collision_count=_median_or_none(
            group,
            "replacement_synthetic_original_collision_count",
        ),
        median_replacement_synthetic_original_collision_value_count=_median_or_none(
            group,
            "replacement_synthetic_original_collision_value_count",
        ),
        replacement_synthetic_original_collision_label_counts=_sum_prefixed_ints(
            group,
            "replacement_synthetic_original_collision_label_counts.",
        ),
        sum_original_value_leak_count=_sum_or_none(group, "original_value_leak_count"),
        leaking_case_count=_positive_count(group, "original_value_leak_count"),
        median_original_value_leak_count=_median_or_none(group, "original_value_leak_count"),
        median_seed_entity_count=_median_or_none(group, "seed_entity_count"),
        median_seed_validation_candidate_count=_median_or_none(group, "seed_validation_candidate_count"),
        median_estimated_seed_validation_chunk_count=_median_or_none(group, "estimated_seed_validation_chunk_count"),
        median_augmented_entity_count=_median_or_none(group, "augmented_entity_count"),
        median_augmented_new_final_value_count=_median_or_none(group, "augmented_new_final_value_count"),
        median_artifact_final_entity_count=_median_or_none(group, "artifact_final_entity_count"),
        median_artifact_final_detector_entity_count=_median_or_none(group, "artifact_final_detector_entity_count"),
        median_artifact_final_rule_entity_count=_median_or_none(group, "artifact_final_rule_entity_count"),
        median_artifact_final_augmenter_entity_count=_median_or_none(group, "artifact_final_augmenter_entity_count"),
        median_artifact_final_entity_signature_count=_median_or_none(group, "artifact_final_entity_signature_count"),
    )


def _none_if_nan(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def _int_if_not_nan(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(float(cast(Any, value)))


def _float_if_not_nan(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(cast(Any, value))


def _median_or_none(dataframe: pd.DataFrame, column: str) -> float | None:
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def _sum_int_or_zero(dataframe: pd.DataFrame, column: str) -> int:
    return int(_sum_or_zero(dataframe, column))


def _sum_bool_or_zero(dataframe: pd.DataFrame, column: str) -> int:
    if column not in dataframe.columns:
        return 0
    return int(dataframe[column].fillna(False).astype(bool).sum())


def _sum_int_or_none(dataframe: pd.DataFrame, column: str) -> int | None:
    value = _sum_or_none(dataframe, column)
    return int(value) if value is not None else None


def _request_failure_rate(*, failed: object, total: object) -> float | None:
    failed_value = _optional_number(failed)
    total_value = _optional_number(total)
    if failed_value is None or total_value is None or total_value <= 0:
        return None
    return failed_value / total_value


def _safe_rate(numerator: object, elapsed_sec: object) -> float | None:
    numerator_value = _optional_number(numerator)
    elapsed_value = _optional_number(elapsed_sec)
    if numerator_value is None or elapsed_value is None or elapsed_value <= 0:
        return None
    return numerator_value / elapsed_value


def _safe_ratio(numerator: object, denominator: object) -> float | None:
    numerator_value = _optional_number(numerator)
    denominator_value = _optional_number(denominator)
    if numerator_value is None or denominator_value is None or denominator_value <= 0:
        return None
    return numerator_value / denominator_value


def _sum_optional_numbers(*values: object) -> float | None:
    numeric_values = [_optional_number(value) for value in values]
    if any(value is None for value in numeric_values):
        return None
    return sum(cast(float, value) for value in numeric_values)


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def _optional_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _estimated_validation_chunk_count(
    artifact_rows: pd.DataFrame,
    *,
    validation_max_entities_per_call: int | None,
) -> float | None:
    if validation_max_entities_per_call is None or validation_max_entities_per_call <= 0:
        return None
    if "seed_validation_candidate_count" not in artifact_rows.columns:
        return None
    counts = pd.to_numeric(artifact_rows["seed_validation_candidate_count"], errors="coerce").dropna()
    if counts.empty:
        return None
    return float(sum(math.ceil(count / validation_max_entities_per_call) for count in counts if count > 0))


def write_analysis_tables(
    result: BenchmarkOutputAnalysis,
    output_dir: Path,
    export_format: ExportFormat,
) -> AnalysisExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        _write_model_rows(
            result.cases, output_dir / f"case_analysis.{export_format.value}", export_format, CaseAnalysisRow
        ),
        _write_model_rows(
            result.groups, output_dir / f"group_analysis.{export_format.value}", export_format, GroupAnalysisRow
        ),
        _write_model_rows(
            result.model_usage,
            output_dir / f"model_analysis.{export_format.value}",
            export_format,
            ModelUsageAnalysisRow,
        ),
        _write_model_rows(
            result.model_usage_groups,
            output_dir / f"model_group_analysis.{export_format.value}",
            export_format,
            ModelUsageGroupAnalysisRow,
        ),
    ]
    export_result = AnalysisExportResult(
        output_dir=str(output_dir),
        format=export_format,
        tables=tables,
        manifest_path=str(output_dir / "manifest.json"),
    )
    Path(export_result.manifest_path).write_text(export_result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return export_result


def _write_model_rows(
    rows: list[BaseModel],
    path: Path,
    export_format: ExportFormat,
    row_model: type[BaseModel],
) -> TableSummary:
    if rows:
        table = pd.json_normalize([row.model_dump() for row in rows], sep=".")
    else:
        table = pd.DataFrame(columns=list(row_model.model_fields))
    if export_format == ExportFormat.parquet:
        table.to_parquet(path, index=False)
    elif export_format == ExportFormat.csv:
        table.to_csv(path, index=False)
    else:
        table.to_json(path, orient="records", lines=True)
    return TableSummary(table=path.stem, rows=len(table), path=str(path))


def render_result(result: BenchmarkOutputAnalysis, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [
        f"Analyzed {result.case_count} case(s) across {result.group_count} group(s); "
        f"model rows={result.model_usage_count}"
    ]
    for group in result.groups:
        label = (
            f"{group.workload_id}/{group.config_id}/"
            f"{group.experimental_detection_strategy}/{group.experimental_replacement_strategy}"
        )
        lines.append(
            f"- {label}: cases={group.case_count}, median_entities={group.median_final_entity_count}, "
            f"failed_cases={group.failed_case_count}/{group.case_count}, "
            f"median_requests={group.median_observed_total_requests}, median_tokens={group.median_observed_total_tokens}, "
            f"median_input_tok_s={group.median_input_text_tokens_per_pipeline_sec}, "
            f"micro_relaxed_f1={group.micro_entity_relaxed_f1}, "
            f"empty_with_gt={group.total_empty_detection_with_ground_truth_count}, "
            f"median_failed_request_rate={group.median_observed_failed_request_rate}, "
            f"median_aug_new_final={group.median_augmented_new_final_value_count}"
        )
    return "\n".join(lines)


@app.default
def main(
    benchmark_dir: Path,
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    detection_artifacts: Annotated[Path | None, cyclopts.Parameter("--detection-artifacts")] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.parquet,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = analyze_benchmark_output(benchmark_dir, detection_artifacts=detection_artifacts)
        if output is not None:
            write_analysis_tables(result, output, format)
    except ValueError as exc:
        log_bad_input(str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
