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
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import pandas as pd
from measurement_tools.benchmark_analysis_math import (
    coalesce_number as _coalesce_number,
)
from measurement_tools.benchmark_analysis_math import (
    f1 as _f1,
)
from measurement_tools.benchmark_analysis_math import (
    first_float as _first_float,
)
from measurement_tools.benchmark_analysis_math import (
    first_int as _first_int,
)
from measurement_tools.benchmark_analysis_math import (
    first_value as _first_value,
)
from measurement_tools.benchmark_analysis_math import (
    model_workflow_rows as _model_workflow_rows,
)
from measurement_tools.benchmark_analysis_math import (
    non_null_count as _non_null_count,
)
from measurement_tools.benchmark_analysis_math import (
    pipeline_stage_rows as _pipeline_stage_rows,
)
from measurement_tools.benchmark_analysis_math import (
    positive_count as _positive_count,
)
from measurement_tools.benchmark_analysis_math import (
    records_of_type as _records_of_type,
)
from measurement_tools.benchmark_analysis_math import (
    request_failure_rate as _request_failure_rate,
)
from measurement_tools.benchmark_analysis_math import (
    safe_rate as _safe_rate,
)
from measurement_tools.benchmark_analysis_math import (
    safe_ratio as _safe_ratio,
)
from measurement_tools.benchmark_analysis_math import (
    sum_int_or_none as _sum_int_or_none,
)
from measurement_tools.benchmark_analysis_math import (
    sum_optional_numbers as _sum_optional_numbers,
)
from measurement_tools.benchmark_analysis_math import (
    sum_prefixed_ints as _sum_prefixed_ints,
)
from measurement_tools.benchmark_analysis_math import (
    zero_count as _zero_count,
)
from measurement_tools.benchmark_analysis_math import (
    zero_with_positive_count as _zero_with_positive_count,
)
from measurement_tools.benchmark_analysis_models import (
    _EVALUATION_ROLLUPS,
)
from measurement_tools.benchmark_analysis_models import (
    BenchmarkOutputAnalysis as BenchmarkOutputAnalysis,
)
from measurement_tools.benchmark_analysis_models import (
    CaseAnalysisRow as CaseAnalysisRow,
)
from measurement_tools.benchmark_analysis_models import (
    GroupAnalysisRow as GroupAnalysisRow,
)
from measurement_tools.benchmark_analysis_models import (
    ModelUsageAnalysisRow as ModelUsageAnalysisRow,
)
from measurement_tools.benchmark_analysis_models import (
    ModelUsageGroupAnalysisRow as ModelUsageGroupAnalysisRow,
)
from measurement_tools.benchmark_analysis_models import (
    _EvaluationRollup as _EvaluationRollup,
)
from measurement_tools.benchmark_group_analysis import (
    build_group_rows as build_group_rows,
)
from measurement_tools.benchmark_model_usage import (
    build_model_usage_group_rows as build_model_usage_group_rows,
)
from measurement_tools.benchmark_model_usage import (
    build_model_usage_rows as build_model_usage_rows,
)
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.stats import sum_int_or_zero as _sum_int_or_zero
from measurement_tools.stats import sum_or_none as _sum_or_none
from measurement_tools.stats import sum_or_zero as _sum_or_zero
from measurement_tools.tables import AnalysisExportResult, ExportFormat, ModelTableSpec
from measurement_tools.tables import write_analysis_tables as _write_analysis_table_specs

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
    evaluation_rows = _records_of_type(measurement_rows, "evaluation_record")
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
        **_case_evaluation_metrics(evaluation_rows),
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


def _case_evaluation_metrics(evaluation_rows: pd.DataFrame) -> dict[str, int | float | None]:
    metrics: dict[str, int | float | None] = {}
    for rollup in _EVALUATION_ROLLUPS:
        judged_count, valid_count = _evaluation_judged_and_valid_counts(evaluation_rows, rollup.valid_column)
        metrics[f"{rollup.prefix}_judged_record_count"] = judged_count
        metrics[f"{rollup.prefix}_valid_record_count"] = valid_count
        metrics[f"{rollup.prefix}_valid_rate"] = _safe_ratio(valid_count, judged_count)
        metrics[rollup.invalid_count_column] = _sum_int_or_zero(evaluation_rows, rollup.invalid_count_column)
    return metrics


def _evaluation_judged_and_valid_counts(evaluation_rows: pd.DataFrame, valid_column: str) -> tuple[int, int]:
    if valid_column not in evaluation_rows.columns:
        return 0, 0
    verdicts = [_optional_bool(value) for value in evaluation_rows[valid_column].tolist()]
    judged_count = sum(verdict is not None for verdict in verdicts)
    valid_count = sum(verdict is True for verdict in verdicts)
    return judged_count, valid_count


def _optional_bool(value: object) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        return None
    if isinstance(value, int | float):
        return bool(value)
    return None


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
    return _write_analysis_table_specs(
        output_dir,
        export_format,
        [
            ModelTableSpec("case_analysis", result.cases, CaseAnalysisRow),
            ModelTableSpec("group_analysis", result.groups, GroupAnalysisRow),
            ModelTableSpec("model_analysis", result.model_usage, ModelUsageAnalysisRow),
            ModelTableSpec("model_group_analysis", result.model_usage_groups, ModelUsageGroupAnalysisRow),
        ],
    )


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
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
