# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped benchmark case analysis and evaluation rollups."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from measurement_tools.benchmark_analysis_math import (
    f1 as _f1,
)
from measurement_tools.benchmark_analysis_math import (
    positive_count as _positive_count,
)
from measurement_tools.benchmark_analysis_math import (
    request_failure_rate as _request_failure_rate,
)
from measurement_tools.benchmark_analysis_math import (
    safe_ratio as _safe_ratio,
)
from measurement_tools.benchmark_analysis_math import (
    sum_bool_or_zero as _sum_bool_or_zero,
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
from measurement_tools.benchmark_analysis_models import _EVALUATION_ROLLUPS, CaseAnalysisRow, GroupAnalysisRow
from measurement_tools.stats import median_or_none as _median_or_none
from measurement_tools.stats import none_if_nan as _none_if_nan
from measurement_tools.stats import sum_int_or_zero as _sum_int_or_zero
from measurement_tools.stats import sum_or_none as _sum_or_none


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
        rows.append(build_group_row(keys, group))
    return rows


def build_group_row(keys: tuple[Any, ...], group: pd.DataFrame) -> GroupAnalysisRow:
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
    evaluation_metrics = group_evaluation_metrics(group)
    return GroupAnalysisRow(
        workload_id=_none_if_nan(workload_id),
        workload_category=_none_if_nan(workload_category),
        config_id=_none_if_nan(config_id),
        experimental_detection_strategy=_none_if_nan(detection_strategy),
        experimental_replacement_strategy=_none_if_nan(replacement_strategy),
        entity_label_set_id=_none_if_nan(entity_label_set_id),
        entity_label_count=int_if_not_nan(entity_label_count),
        gliner_threshold=float_if_not_nan(gliner_threshold),
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
        **evaluation_metrics,
        median_seed_entity_count=_median_or_none(group, "seed_entity_count"),
        median_seed_validation_candidate_count=_median_or_none(group, "seed_validation_candidate_count"),
        median_estimated_seed_validation_chunk_count=_median_or_none(group, "estimated_seed_validation_chunk_count"),
        median_augmented_entity_count=_median_or_none(group, "augmented_entity_count"),
        median_augmented_new_final_value_count=_median_or_none(group, "augmented_new_final_value_count"),
        median_artifact_final_entity_count=_median_or_none(group, "artifact_final_entity_count"),
        median_artifact_final_detector_entity_count=_median_or_none(group, "artifact_final_detector_entity_count"),
        median_artifact_final_augmenter_entity_count=_median_or_none(group, "artifact_final_augmenter_entity_count"),
        median_artifact_final_entity_signature_count=_median_or_none(group, "artifact_final_entity_signature_count"),
    )


def int_if_not_nan(value: object) -> int | None:
    if pd.isna(value):
        return None
    return int(float(cast(Any, value)))


def float_if_not_nan(value: object) -> float | None:
    if pd.isna(value):
        return None
    return float(cast(Any, value))


def group_evaluation_metrics(group: pd.DataFrame) -> dict[str, int | float | None]:
    metrics: dict[str, int | float | None] = {}
    for rollup in _EVALUATION_ROLLUPS:
        judged_count = _sum_int_or_zero(group, f"{rollup.prefix}_judged_record_count")
        valid_count = _sum_int_or_zero(group, f"{rollup.prefix}_valid_record_count")
        metrics[f"sum_{rollup.prefix}_judged_record_count"] = judged_count
        metrics[f"sum_{rollup.prefix}_valid_record_count"] = valid_count
        metrics[f"micro_{rollup.prefix}_valid_rate"] = _safe_ratio(valid_count, judged_count)
        metrics[f"sum_{rollup.invalid_count_column}"] = _sum_int_or_zero(group, rollup.invalid_count_column)
    return metrics


__all__ = [
    "build_group_row",
    "build_group_rows",
    "float_if_not_nan",
    "group_evaluation_metrics",
    "int_if_not_nan",
]
