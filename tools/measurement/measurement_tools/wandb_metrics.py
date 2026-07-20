# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated W&B metric and table projections."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import BaseModel, Field, StrictBool, StrictInt, StrictStr, field_validator, model_validator

from measurement_tools.validation import (
    MeasurementStrategy,
    NonNegativeFloat,
    NonNegativeInt,
    Probability,
    StrictFrozenModel,
)
from measurement_tools.wandb_metadata import SafeScalar
from measurement_tools.wandb_metric_schema import AGGREGATED_MEASUREMENT_FIELDS, BENCHMARK_METRIC_NAMES

__all__ = [
    "EvaluationTableRow",
    "ModelWorkflowTableRow",
    "NddWorkflowTableRow",
    "RecordTableRow",
    "RunTableRow",
    "StageTableRow",
    "TraceCoverageTableRow",
    "WANDB_TABLE_ROW_MODELS",
    "WandbHistoryPayload",
    "WandbSummaryPayload",
    "WandbTablePayload",
    "WandbTableRow",
]

_PUBLICATION_COMPLETE_KEY = "publication/complete"
_PUBLICATION_SEAL_DIGEST_KEY = "publication/completion_seal_sha256"


class WandbHistoryPayload(StrictFrozenModel):
    metrics: dict[StrictStr, SafeScalar]

    @field_validator("metrics")
    @classmethod
    def metric_keys_have_policy(cls, metrics: dict[str, SafeScalar]) -> dict[str, SafeScalar]:
        unknown = [key for key in metrics if not _metric_has_aggregate_policy(key)]
        if unknown:
            raise ValueError(f"W&B metric has no aggregate exposure policy: {unknown!r}")
        return metrics


class WandbSummaryPayload(StrictFrozenModel):
    metrics: dict[StrictStr, SafeScalar]

    @field_validator("metrics")
    @classmethod
    def metric_keys_have_policy(cls, metrics: dict[str, SafeScalar]) -> dict[str, SafeScalar]:
        return WandbHistoryPayload.metric_keys_have_policy(metrics)


class _MetricTableRow(StrictFrozenModel):
    elapsed_sec: NonNegativeFloat | None = None
    input_row_count: NonNegativeInt | None = None
    seed_row_count: NonNegativeInt | None = None
    output_row_count: NonNegativeInt | None = None
    failed_record_count: NonNegativeInt | None = None
    preview_num_records: NonNegativeInt | None = None
    column_count: NonNegativeInt | None = None
    observed_input_tokens: NonNegativeInt | None = None
    observed_output_tokens: NonNegativeInt | None = None
    observed_total_tokens: NonNegativeInt | None = None
    observed_reasoning_tokens: NonNegativeInt | None = None
    observed_successful_requests: NonNegativeInt | None = None
    observed_failed_requests: NonNegativeInt | None = None
    observed_total_requests: NonNegativeInt | None = None
    text_length_chars: NonNegativeInt | None = None
    text_length_tokens: NonNegativeInt | None = None
    final_entity_count: NonNegativeInt | None = None
    ground_truth_entity_count: NonNegativeInt | None = None
    entity_true_positive_count: NonNegativeInt | None = None
    entity_false_positive_count: NonNegativeInt | None = None
    entity_false_negative_count: NonNegativeInt | None = None
    entity_relaxed_gt_found_count: NonNegativeInt | None = None
    entity_relaxed_detected_tp_count: NonNegativeInt | None = None
    entity_relaxed_label_compatible_gt_found_count: NonNegativeInt | None = None
    entity_relaxed_label_compatible_detected_tp_count: NonNegativeInt | None = None
    replacement_count: NonNegativeInt | None = None
    replacement_duplicate_value_count: NonNegativeInt | None = None
    replacement_missing_final_entity_count: NonNegativeInt | None = None
    replacement_missing_final_value_count: NonNegativeInt | None = None
    replacement_synthetic_original_collision_count: NonNegativeInt | None = None
    replacement_synthetic_original_collision_value_count: NonNegativeInt | None = None
    original_value_leak_count: NonNegativeInt | None = None
    detected_candidate_count: NonNegativeInt | None = None
    validation_chunk_count: NonNegativeInt | None = None
    llm_calls_estimated_total: NonNegativeInt | None = None
    detection_invalid_entity_count: NonNegativeInt | None = None
    type_fidelity_invalid_replacement_count: NonNegativeInt | None = None
    relational_consistency_invalid_relation_count: NonNegativeInt | None = None
    attribute_fidelity_invalid_entity_count: NonNegativeInt | None = None
    observed_failed_request_rate: Probability | None = None
    input_rows_per_sec: NonNegativeFloat | None = None
    output_rows_per_sec: NonNegativeFloat | None = None
    observed_tokens_per_sec: NonNegativeFloat | None = None
    observed_requests_per_sec: NonNegativeFloat | None = None
    observed_tokens_per_successful_request: NonNegativeFloat | None = None
    entity_precision: Probability | None = None
    entity_recall: Probability | None = None
    entity_f1: Probability | None = None
    utility_score: Probability | None = None
    leakage_mass: NonNegativeFloat | None = None
    weighted_leakage_rate: Probability | None = None
    repair_iterations: NonNegativeInt | None = None


_METRIC_TABLE_POLICY_FIELDS = (
    "elapsed_sec",
    "input_row_count",
    "seed_row_count",
    "output_row_count",
    "failed_record_count",
    "preview_num_records",
    "column_count",
    "observed_input_tokens",
    "observed_output_tokens",
    "observed_total_tokens",
    "observed_reasoning_tokens",
    "observed_successful_requests",
    "observed_failed_requests",
    "observed_total_requests",
    "text_length_chars",
    "text_length_tokens",
    "final_entity_count",
    "ground_truth_entity_count",
    "entity_true_positive_count",
    "entity_false_positive_count",
    "entity_false_negative_count",
    "entity_relaxed_gt_found_count",
    "entity_relaxed_detected_tp_count",
    "entity_relaxed_label_compatible_gt_found_count",
    "entity_relaxed_label_compatible_detected_tp_count",
    "replacement_count",
    "replacement_duplicate_value_count",
    "replacement_missing_final_entity_count",
    "replacement_missing_final_value_count",
    "replacement_synthetic_original_collision_count",
    "replacement_synthetic_original_collision_value_count",
    "original_value_leak_count",
    "detected_candidate_count",
    "validation_chunk_count",
    "llm_calls_estimated_total",
    "detection_invalid_entity_count",
    "type_fidelity_invalid_replacement_count",
    "relational_consistency_invalid_relation_count",
    "attribute_fidelity_invalid_entity_count",
    "observed_failed_request_rate",
    "input_rows_per_sec",
    "output_rows_per_sec",
    "observed_tokens_per_sec",
    "observed_requests_per_sec",
    "observed_tokens_per_successful_request",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "utility_score",
    "leakage_mass",
    "weighted_leakage_rate",
    "repair_iterations",
)


class RunTableRow(_MetricTableRow):
    record_type: Literal["run"]
    mode: Literal["replace", "rewrite"]
    strategy: MeasurementStrategy


class StageTableRow(_MetricTableRow):
    record_type: Literal["stage"]
    stage: Literal[
        "Anonymizer._run_internal",
        "EntityDetectionWorkflow.run",
        "ReplacementWorkflow.run",
        "RewriteWorkflow.run",
    ]
    status: Literal["completed", "error"]


class RecordTableRow(_MetricTableRow):
    record_type: Literal["record"]
    mode: Literal["replace", "rewrite"]
    strategy: MeasurementStrategy
    row_index: StrictInt | None
    record_hash: Annotated[StrictStr, Field(pattern=r"^[0-9a-fA-F]{64}$")]
    text_length_chars: None = None
    text_length_tokens: None = None
    text_length_chars_bucket: Literal["0", "1-127", "128-511", "512-2047", "2048-8191", "8192+"]
    text_length_tokens_bucket: Literal["0", "1-127", "128-511", "512-2047", "2048-8191", "8192+"]


class EvaluationTableRow(RecordTableRow):
    record_type: Literal["evaluation_record"]
    detection_valid: StrictBool | None = None
    type_fidelity_valid: StrictBool | None = None
    relational_consistency_valid: StrictBool | None = None
    attribute_fidelity_valid: StrictBool | None = None


class NddWorkflowTableRow(_MetricTableRow):
    record_type: Literal["ndd_workflow"]
    status: Literal["completed", "error"]


class ModelWorkflowTableRow(_MetricTableRow):
    record_type: Literal["model_workflow"]
    status: Literal["completed", "error"]


class TraceCoverageTableRow(StrictFrozenModel):
    record_type: Literal["dd_trace_coverage"]
    traced_column_count: NonNegativeInt
    native_trace_column_count: NonNegativeInt
    private_trace_column_count: NonNegativeInt
    unsupported_column_count: NonNegativeInt


WandbTableRow = Annotated[
    RunTableRow
    | StageTableRow
    | RecordTableRow
    | EvaluationTableRow
    | NddWorkflowTableRow
    | ModelWorkflowTableRow
    | TraceCoverageTableRow,
    Field(discriminator="record_type"),
]
WANDB_TABLE_ROW_MODELS: Mapping[str, type[BaseModel]] = MappingProxyType(
    {
        "run": RunTableRow,
        "stage": StageTableRow,
        "record": RecordTableRow,
        "evaluation_record": EvaluationTableRow,
        "ndd_workflow": NddWorkflowTableRow,
        "model_workflow": ModelWorkflowTableRow,
        "dd_trace_coverage": TraceCoverageTableRow,
    }
)
_MEASUREMENT_RECORD_TYPES = frozenset(WANDB_TABLE_ROW_MODELS)


class WandbTablePayload(StrictFrozenModel):
    record_type: StrictStr
    rows: tuple[WandbTableRow, ...]

    @model_validator(mode="after")
    def rows_match_record_type(self) -> WandbTablePayload:
        if any(row.record_type != self.record_type for row in self.rows):
            raise ValueError("W&B table rows must match the table record type")
        return self

    @property
    def name(self) -> str:
        return f"measurement_table/{self.record_type}"

    @property
    def columns(self) -> tuple[str, ...]:
        present = {key for row in self.rows for key in row.model_dump(exclude_none=True)}
        model_fields = type(self.rows[0]).model_fields if self.rows else {}
        return tuple(name for name in model_fields if name in present)

    @property
    def data(self) -> tuple[tuple[SafeScalar | None, ...], ...]:
        columns = self.columns
        return tuple(tuple(getattr(row, column) for column in columns) for row in self.rows)


def _metric_has_aggregate_policy(key: str) -> bool:
    if key in BENCHMARK_METRIC_NAMES or key in {
        "measurement/record_count",
        _PUBLICATION_COMPLETE_KEY,
        _PUBLICATION_SEAL_DIGEST_KEY,
    }:
        return True
    parts = key.split("/")
    return (
        len(parts) == 3
        and parts[0] == "measurement"
        and parts[1] in _MEASUREMENT_RECORD_TYPES
        and parts[2] in AGGREGATED_MEASUREMENT_FIELDS
    )
