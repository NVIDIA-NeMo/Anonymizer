# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for benchmark output analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, computed_field


@dataclass(frozen=True)
class _EvaluationRollup:
    prefix: str
    valid_column: str
    invalid_count_column: str


_EVALUATION_ROLLUPS = (
    _EvaluationRollup("detection", "detection_valid", "detection_invalid_entity_count"),
    _EvaluationRollup("type_fidelity", "type_fidelity_valid", "type_fidelity_invalid_replacement_count"),
    _EvaluationRollup(
        "relational_consistency",
        "relational_consistency_valid",
        "relational_consistency_invalid_relation_count",
    ),
    _EvaluationRollup("attribute_fidelity", "attribute_fidelity_valid", "attribute_fidelity_invalid_entity_count"),
)


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
    detection_judged_record_count: int = 0
    detection_valid_record_count: int = 0
    detection_valid_rate: float | None = None
    detection_invalid_entity_count: int = 0
    type_fidelity_judged_record_count: int = 0
    type_fidelity_valid_record_count: int = 0
    type_fidelity_valid_rate: float | None = None
    type_fidelity_invalid_replacement_count: int = 0
    relational_consistency_judged_record_count: int = 0
    relational_consistency_valid_record_count: int = 0
    relational_consistency_valid_rate: float | None = None
    relational_consistency_invalid_relation_count: int = 0
    attribute_fidelity_judged_record_count: int = 0
    attribute_fidelity_valid_record_count: int = 0
    attribute_fidelity_valid_rate: float | None = None
    attribute_fidelity_invalid_entity_count: int = 0
    validation_max_entities_per_call: int | None = None
    detection_artifact_rows: int = 0
    seed_entity_count: float | None = None
    seed_validation_candidate_count: float | None = None
    estimated_seed_validation_chunk_count: float | None = None
    augmented_entity_count: float | None = None
    augmented_new_final_value_count: float | None = None
    artifact_final_entity_count: float | None = None
    artifact_final_detector_entity_count: float | None = None
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
    sum_detection_judged_record_count: int = 0
    sum_detection_valid_record_count: int = 0
    micro_detection_valid_rate: float | None = None
    sum_detection_invalid_entity_count: int = 0
    sum_type_fidelity_judged_record_count: int = 0
    sum_type_fidelity_valid_record_count: int = 0
    micro_type_fidelity_valid_rate: float | None = None
    sum_type_fidelity_invalid_replacement_count: int = 0
    sum_relational_consistency_judged_record_count: int = 0
    sum_relational_consistency_valid_record_count: int = 0
    micro_relational_consistency_valid_rate: float | None = None
    sum_relational_consistency_invalid_relation_count: int = 0
    sum_attribute_fidelity_judged_record_count: int = 0
    sum_attribute_fidelity_valid_record_count: int = 0
    micro_attribute_fidelity_valid_rate: float | None = None
    sum_attribute_fidelity_invalid_entity_count: int = 0
    median_seed_entity_count: float | None = None
    median_seed_validation_candidate_count: float | None = None
    median_estimated_seed_validation_chunk_count: float | None = None
    median_augmented_entity_count: float | None = None
    median_augmented_new_final_value_count: float | None = None
    median_artifact_final_entity_count: float | None = None
    median_artifact_final_detector_entity_count: float | None = None
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


__all__ = [
    "BenchmarkOutputAnalysis",
    "CaseAnalysisRow",
    "GroupAnalysisRow",
    "ModelUsageAnalysisRow",
    "ModelUsageGroupAnalysisRow",
]
