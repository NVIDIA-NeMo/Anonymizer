# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-case metrics and artifact analysis for benchmark output."""

from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd

from measurement_tools.benchmark_analysis_math import coalesce_number as _coalesce_number
from measurement_tools.benchmark_analysis_math import f1 as _f1
from measurement_tools.benchmark_analysis_math import first_float as _first_float
from measurement_tools.benchmark_analysis_math import first_int as _first_int
from measurement_tools.benchmark_analysis_math import first_value as _first_value
from measurement_tools.benchmark_analysis_math import model_workflow_rows as _model_workflow_rows
from measurement_tools.benchmark_analysis_math import non_null_count as _non_null_count
from measurement_tools.benchmark_analysis_math import pipeline_stage_rows as _pipeline_stage_rows
from measurement_tools.benchmark_analysis_math import positive_count as _positive_count
from measurement_tools.benchmark_analysis_math import records_of_type as _records_of_type
from measurement_tools.benchmark_analysis_math import request_failure_rate as _request_failure_rate
from measurement_tools.benchmark_analysis_math import safe_rate as _safe_rate
from measurement_tools.benchmark_analysis_math import safe_ratio as _safe_ratio
from measurement_tools.benchmark_analysis_math import sum_int_or_none as _sum_int_or_none
from measurement_tools.benchmark_analysis_math import sum_optional_numbers as _sum_optional_numbers
from measurement_tools.benchmark_analysis_math import sum_prefixed_ints as _sum_prefixed_ints
from measurement_tools.benchmark_analysis_math import zero_count as _zero_count
from measurement_tools.benchmark_analysis_math import zero_with_positive_count as _zero_with_positive_count
from measurement_tools.benchmark_analysis_models import _EVALUATION_ROLLUPS, CaseAnalysisRow
from measurement_tools.stats import sum_int_or_zero as _sum_int_or_zero
from measurement_tools.stats import sum_or_none as _sum_or_none
from measurement_tools.stats import sum_or_zero as _sum_or_zero

_SYNC_CLIENT_UNAVAILABLE_ERROR = "SyncClientUnavailableError"


_SIGNATURE_DETAIL_FIELDS = {
    "label",
    "source",
    "row_index",
    "start_position",
    "end_position",
    "value_length",
}


def build_case_row(
    case_id: str,
    measurements: pd.DataFrame,
    artifacts: pd.DataFrame,
    traces: pd.DataFrame,
) -> CaseAnalysisRow:
    measurement_rows = rows_for_case(measurements, case_id)
    artifact_rows = rows_for_case(artifacts, case_id)
    trace_rows = rows_for_case(traces, case_id)
    record_rows = _records_of_type(measurement_rows, "record")
    evaluation_rows = _records_of_type(measurement_rows, "evaluation_record")
    ndd_rows = _records_of_type(measurement_rows, "ndd_workflow")
    model_rows = _model_workflow_rows(measurement_rows)
    stage_rows = _records_of_type(measurement_rows, "stage")
    pipeline_rows = _pipeline_stage_rows(measurement_rows)
    validation_max_entities_per_call = _first_int([measurement_rows], ["detect.validation_max_entities_per_call"])
    request_metrics = case_request_metrics(model_rows)
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
        **case_failure_metrics(stage_rows=stage_rows, ndd_rows=ndd_rows, model_rows=model_rows),
        pipeline_elapsed_sec=pipeline_elapsed_sec,
        ndd_workflow_count=len(ndd_rows),
        ndd_elapsed_sec_total=ndd_elapsed_sec_total,
        **request_metrics,
        **case_trace_metrics(trace_rows, request_metrics=request_metrics),
        record_count=record_count,
        input_text_tokens_total=input_text_tokens_total,
        records_per_pipeline_sec=records_per_pipeline_sec,
        records_per_ndd_sec=records_per_ndd_sec,
        input_text_tokens_per_pipeline_sec=input_text_tokens_per_pipeline_sec,
        input_text_tokens_per_ndd_sec=input_text_tokens_per_ndd_sec,
        **case_topology_metrics(
            measurement_rows,
            input_text_tokens_per_pipeline_sec=input_text_tokens_per_pipeline_sec,
        ),
        final_entity_count=final_entity_count,
        **case_empty_detection_metrics(record_rows, record_count=record_count),
        **case_ground_truth_metrics(record_rows, final_entity_count=final_entity_count),
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
        **case_evaluation_metrics(evaluation_rows),
        validation_max_entities_per_call=validation_max_entities_per_call,
        **case_artifact_metrics(
            artifact_rows,
            validation_max_entities_per_call=validation_max_entities_per_call,
        ),
    )


def case_request_metrics(model_rows: pd.DataFrame) -> dict[str, int | float | None]:
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


def case_trace_metrics(
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


def case_failure_metrics(
    *,
    stage_rows: pd.DataFrame,
    ndd_rows: pd.DataFrame,
    model_rows: pd.DataFrame,
) -> dict[str, bool | int]:
    error_stage_count = error_status_count(stage_rows)
    error_ndd_workflow_count = error_status_count(ndd_rows)
    error_model_workflow_count = error_status_count(model_rows)
    return {
        "case_failed": error_stage_count > 0 or error_ndd_workflow_count > 0 or error_model_workflow_count > 0,
        "error_stage_count": error_stage_count,
        "error_ndd_workflow_count": error_ndd_workflow_count,
        "error_model_workflow_count": error_model_workflow_count,
    }


def case_topology_metrics(
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


def case_empty_detection_metrics(record_rows: pd.DataFrame, *, record_count: int) -> dict[str, int | float | None]:
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


def case_ground_truth_metrics(
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


def error_status_count(rows: pd.DataFrame) -> int:
    if "status" not in rows.columns:
        return 0
    statuses = rows["status"].astype(str).str.lower()
    return int(statuses.isin({"error", "failed"}).sum())


def case_evaluation_metrics(evaluation_rows: pd.DataFrame) -> dict[str, int | float | None]:
    metrics: dict[str, int | float | None] = {}
    for rollup in _EVALUATION_ROLLUPS:
        judged_count, valid_count = evaluation_judged_and_valid_counts(evaluation_rows, rollup.valid_column)
        metrics[f"{rollup.prefix}_judged_record_count"] = judged_count
        metrics[f"{rollup.prefix}_valid_record_count"] = valid_count
        metrics[f"{rollup.prefix}_valid_rate"] = _safe_ratio(valid_count, judged_count)
        metrics[rollup.invalid_count_column] = _sum_int_or_zero(evaluation_rows, rollup.invalid_count_column)
    return metrics


def evaluation_judged_and_valid_counts(evaluation_rows: pd.DataFrame, valid_column: str) -> tuple[int, int]:
    if valid_column not in evaluation_rows.columns:
        return 0, 0
    verdicts = [optional_bool(value) for value in evaluation_rows[valid_column].tolist()]
    judged_count = sum(verdict is not None for verdict in verdicts)
    valid_count = sum(verdict is True for verdict in verdicts)
    return judged_count, valid_count


def optional_bool(value: object) -> bool | None:
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


def case_artifact_metrics(
    artifact_rows: pd.DataFrame,
    *,
    validation_max_entities_per_call: int | None,
) -> dict[str, int | float | list[str] | dict[str, str] | None]:
    signature_hashes = artifact_signature_hashes(artifact_rows)
    return {
        "detection_artifact_rows": len(artifact_rows),
        "seed_entity_count": _sum_or_none(artifact_rows, "seed_entity_count"),
        "seed_validation_candidate_count": _sum_or_none(artifact_rows, "seed_validation_candidate_count"),
        "estimated_seed_validation_chunk_count": estimated_validation_chunk_count(
            artifact_rows,
            validation_max_entities_per_call=validation_max_entities_per_call,
        ),
        "augmented_entity_count": _sum_or_none(artifact_rows, "augmented_entity_count"),
        "augmented_new_final_value_count": _sum_or_none(artifact_rows, "augmented_new_final_value_count"),
        "artifact_final_entity_count": _sum_or_none(artifact_rows, "final_entity_count"),
        "artifact_final_detector_entity_count": _sum_or_none(artifact_rows, "final_source_counts.detector"),
        "artifact_final_augmenter_entity_count": _sum_or_none(artifact_rows, "final_source_counts.augmenter"),
        "artifact_final_entity_signature_count": signature_count(artifact_rows, signature_hashes=signature_hashes),
        "artifact_final_entity_signature_hashes": signature_hashes,
        "artifact_final_entity_signature_labels": artifact_signature_labels(artifact_rows),
        "artifact_final_entity_signature_details": artifact_signature_details(artifact_rows),
    }


def rows_for_case(dataframe: pd.DataFrame, case_id: str) -> pd.DataFrame:
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


def artifact_signature_hashes(artifact_rows: pd.DataFrame) -> list[str]:
    if "final_entity_signature_hashes" not in artifact_rows.columns:
        return []
    values: set[str] = set()
    for raw in artifact_rows["final_entity_signature_hashes"].dropna():
        values.update(coerce_string_list(raw))
    return sorted(values)


def artifact_signature_labels(artifact_rows: pd.DataFrame) -> dict[str, str]:
    labels: dict[str, str] = {}
    if "final_entity_signature_labels" in artifact_rows.columns:
        for raw in artifact_rows["final_entity_signature_labels"].dropna():
            labels.update(coerce_string_dict(raw))
    for column in artifact_rows.columns:
        prefix = "final_entity_signature_labels."
        if not column.startswith(prefix):
            continue
        signature_hash = column.removeprefix(prefix)
        for value in artifact_rows[column].dropna():
            labels[signature_hash] = str(value)
    return dict(sorted(labels.items()))


def artifact_signature_details(artifact_rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    details: dict[str, dict[str, Any]] = {}
    if "final_entity_signature_details" in artifact_rows.columns:
        for raw in artifact_rows["final_entity_signature_details"].dropna():
            details.update(coerce_detail_map(raw))
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
            details.setdefault(signature_hash, {})[field] = json_scalar(value)
    return dict(sorted(details.items()))


def coerce_detail_map(raw: object) -> dict[str, dict[str, Any]]:
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
                str(key): json_scalar(item) for key, item in value.items() if str(key) in _SIGNATURE_DETAIL_FIELDS
            }
    return details


def json_scalar(value: object) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            return value
    return value


def coerce_string_list(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return []


def coerce_string_dict(raw: object) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items()}
    return {}


def signature_count(artifact_rows: pd.DataFrame, *, signature_hashes: list[str]) -> float | None:
    if signature_hashes:
        return float(len(signature_hashes))
    return _sum_or_none(artifact_rows, "final_entity_signature_count")


def estimated_validation_chunk_count(
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


__all__ = [
    "artifact_signature_details",
    "artifact_signature_hashes",
    "artifact_signature_labels",
    "build_case_row",
    "case_artifact_metrics",
    "case_empty_detection_metrics",
    "case_evaluation_metrics",
    "case_failure_metrics",
    "case_ground_truth_metrics",
    "case_request_metrics",
    "case_topology_metrics",
    "case_trace_metrics",
    "coerce_detail_map",
    "coerce_string_dict",
    "coerce_string_list",
    "error_status_count",
    "estimated_validation_chunk_count",
    "evaluation_judged_and_valid_counts",
    "json_scalar",
    "optional_bool",
    "rows_for_case",
    "signature_count",
]
