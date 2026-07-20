# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration and outbound models for native W&B publication."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, cast

from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from measurement_tools.validation import (
    MeasurementStrategy,
    NonNegativeFloat,
    NonNegativeInt,
    Probability,
    StrictFrozenModel,
    VisibleIdentifier,
)
from measurement_tools.wandb_metadata import (
    BenchmarkMetadata,
    ConfigMetadata,
    DetectMetadata,
    ExecutionMetadata,
    GitMetadata,
    ImportedRunMetadata,
    MatrixMetadata,
    ModelSourcesMetadata,
    ReplaceMetadata,
    RewriteMetadata,
    RuntimeMetadata,
    SafeScalar,
    SlurmJobMetadata,
    SlurmMetadata,
    SweepMetadata,
    WandbConfigPayload,
    WandbRunMetadata,
    WandbTag,
    WorkloadMetadata,
    WorkloadSourceMetadata,
)
from measurement_tools.wandb_metric_schema import AGGREGATED_MEASUREMENT_FIELDS, BENCHMARK_METRIC_NAMES
from measurement_tools.wandb_policy import DataClass, Exposure, FieldPolicy
from measurement_tools.wandb_settings import (
    DEFAULT_WANDB_PROJECT as DEFAULT_WANDB_PROJECT,
)
from measurement_tools.wandb_settings import (
    WANDB_TAG_MAX_LENGTH as WANDB_TAG_MAX_LENGTH,
)
from measurement_tools.wandb_settings import (
    ResolvedWandbConfig as ResolvedWandbConfig,
)
from measurement_tools.wandb_settings import (
    WandbInputs as WandbInputs,
)
from measurement_tools.wandb_settings import (
    WandbMode,
)
from measurement_tools.wandb_settings import (
    generated_wandb_tag as generated_wandb_tag,
)
from measurement_tools.wandb_settings import (
    is_safe_wandb_tag as is_safe_wandb_tag,
)
from measurement_tools.wandb_settings import (
    wandb_tag_value_is_sensitive as wandb_tag_value_is_sensitive,
)

PUBLICATION_COMPLETE_KEY = "publication/complete"
PUBLICATION_SEAL_DIGEST_KEY = "publication/completion_seal_sha256"


class WandbPublicationState(StrEnum):
    created = "created"
    resumed = "resumed"
    already_complete = "already_complete"


class WandbInitPayload(StrictFrozenModel):
    run_id: VisibleIdentifier
    resume: Literal["allow", "never"] = "never"
    project: StrictStr
    name: StrictStr
    mode: WandbMode
    directory: Path
    group: StrictStr
    job_type: StrictStr
    entity: StrictStr | None = None
    tags: tuple[WandbTag, ...] = ()


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


class WandbPublishPayload(StrictFrozenModel):
    init: WandbInitPayload
    config: WandbConfigPayload
    history: WandbHistoryPayload
    summary: WandbSummaryPayload
    tables: tuple[WandbTablePayload, ...] = ()

    @model_validator(mode="after")
    def validate_resume_contract(self) -> WandbPublishPayload:
        marker = self.summary.metrics.get(PUBLICATION_COMPLETE_KEY)
        digest = self.summary.metrics.get(PUBLICATION_SEAL_DIGEST_KEY)
        if self.init.resume == "allow":
            imported = self.config.imported
            if imported is None or marker is not True or digest != imported.completion_seal_sha256:
                raise ValueError("resumable W&B payload requires matching imported publication markers")
        elif marker is not None or digest is not None:
            raise ValueError("fresh W&B payload cannot contain resume publication markers")
        return self


class WandbPublishResult(StrictFrozenModel):
    published: StrictBool
    run_id: StrictStr | None = None
    entity: StrictStr | None = None
    project: StrictStr | None = None
    run_url: StrictStr | None = None
    publication_state: WandbPublicationState | None = None
    measurement_sha256: Annotated[StrictStr | None, Field(pattern=r"^[0-9a-f]{64}$")] = None
    record_count: StrictInt = 0


def _aggregate_policies(
    *names: str,
    data_class: DataClass = DataClass.operational,
    exposure: Exposure = Exposure.aggregate,
) -> dict[str, FieldPolicy]:
    return {name: FieldPolicy(data_class=data_class, exposure=exposure) for name in names}


OUTBOUND_FIELD_POLICIES: dict[type[BaseModel], dict[str, FieldPolicy]] = {
    WandbInitPayload: {
        "run_id": FieldPolicy(data_class=DataClass.pseudonymous, exposure=Exposure.aggregate),
        "resume": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "project": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "name": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "mode": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "directory": FieldPolicy(data_class=DataClass.sensitive, exposure=Exposure.local_only),
        "group": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "job_type": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "entity": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "tags": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
    },
    BenchmarkMetadata: _aggregate_policies(
        "metadata_schema_version",
        "suite_schema_version",
        "wandb_sanitizer_version",
        "measurement_schema_version",
        "suite_id",
        "workload_count",
        "config_count",
        "matrix_entry_count",
        "case_count",
        "case_retries",
        "case_retry_backoff_sec",
        "suite_file_hash",
    ),
    SlurmJobMetadata: _aggregate_policies("role", "job_id", data_class=DataClass.pseudonymous),
    SlurmMetadata: _aggregate_policies(
        "job_id",
        "array_job_id",
        "array_task_id",
        "array_task_count",
        "restart_count",
        "ntasks",
        "job_num_nodes",
        "jobs",
        data_class=DataClass.pseudonymous,
    ),
    ExecutionMetadata: _aggregate_policies("backend", "export", "fail_fast", "dd_trace", "dd_task_trace", "slurm"),
    RuntimeMetadata: _aggregate_policies(
        "anonymizer_version",
        "datadesigner_version",
        "wandb_version",
        "python_version",
        "platform_machine",
        "platform_system",
    ),
    GitMetadata: _aggregate_policies("commit", "branch", "dirty", data_class=DataClass.pseudonymous),
    ModelSourcesMetadata: _aggregate_policies("has_model_configs", "has_model_providers", "has_artifact_path"),
    WorkloadSourceMetadata: _aggregate_policies("kind", "suffix"),
    WorkloadMetadata: _aggregate_policies(
        "id",
        "text_column",
        "has_id_column",
        "has_data_summary",
        "row_limit",
        "row_offset",
        "source",
        data_class=DataClass.pseudonymous,
    ),
    DetectMetadata: _aggregate_policies(
        "entity_label_source",
        "entity_label_count",
        "entity_label_set_hash",
        "gliner_threshold",
        "validation_max_entities_per_call",
        "validation_excerpt_window_chars",
    ),
    ReplaceMetadata: _aggregate_policies(
        "strategy", "normalize_label", "algorithm", "digest_length", "has_format_template", "has_instructions"
    ),
    RewriteMetadata: _aggregate_policies(
        "risk_tolerance",
        "max_repair_iterations",
        "strict_entity_protection",
        "has_privacy_goal",
        "has_protect",
        "has_preserve",
        "has_instructions",
    ),
    ConfigMetadata: _aggregate_policies("id", "mode", "evaluate", "emit_telemetry", "detect", "replace", "rewrite"),
    MatrixMetadata: _aggregate_policies("workload", "config", "repetitions", data_class=DataClass.pseudonymous),
    SweepMetadata: _aggregate_policies("id", "arm_id", "params", data_class=DataClass.pseudonymous),
    ImportedRunMetadata: _aggregate_policies(
        "completion_seal_schema_version",
        "completion_seal_sha256",
        "producer_repository",
        "producer_commit",
        "phase",
        "case_id",
        data_class=DataClass.pseudonymous,
    ),
    WandbRunMetadata: _aggregate_policies(
        "run_kind",
        "benchmark",
        "execution",
        "runtime",
        "git",
        "model_sources",
        "workloads",
        "configs",
        "matrix",
        "sweep",
        "imported",
        data_class=DataClass.pseudonymous,
    ),
    WandbConfigPayload: _aggregate_policies(
        "suite_id",
        "wandb_mode",
        "wandb_log_tables",
        "benchmark_suite_id",
        "benchmark_case_count",
        "benchmark_workload_ids",
        "benchmark_workload_row_limits",
        "benchmark_workload_source_kinds",
        "benchmark_workload_source_suffixes",
        "benchmark_config_ids",
        "benchmark_modes",
        "benchmark_strategies",
        "benchmark_gliner_thresholds",
        "benchmark_entity_label_counts",
        "benchmark_risk_tolerances",
        "native_suite_id",
        "sweep_id",
        "sweep_arm_id",
        "imported_config_id",
        "sweep_params",
    ),
    WandbHistoryPayload: {
        "metrics": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
    },
    WandbSummaryPayload: {
        "metrics": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
    },
    _MetricTableRow: _aggregate_policies(
        *_METRIC_TABLE_POLICY_FIELDS,
        exposure=Exposure.table_opt_in,
    ),
    RunTableRow: _aggregate_policies(
        "record_type",
        "mode",
        "strategy",
        data_class=DataClass.pseudonymous,
        exposure=Exposure.table_opt_in,
    ),
    StageTableRow: _aggregate_policies("record_type", "stage", "status", exposure=Exposure.table_opt_in),
    RecordTableRow: _aggregate_policies(
        "record_type",
        "mode",
        "strategy",
        "row_index",
        "record_hash",
        "text_length_chars",
        "text_length_tokens",
        "text_length_chars_bucket",
        "text_length_tokens_bucket",
        data_class=DataClass.pseudonymous,
        exposure=Exposure.table_opt_in,
    ),
    EvaluationTableRow: _aggregate_policies(
        "record_type",
        "detection_valid",
        "type_fidelity_valid",
        "relational_consistency_valid",
        "attribute_fidelity_valid",
        data_class=DataClass.pseudonymous,
        exposure=Exposure.table_opt_in,
    ),
    NddWorkflowTableRow: _aggregate_policies(
        "record_type",
        "status",
        exposure=Exposure.table_opt_in,
    ),
    ModelWorkflowTableRow: _aggregate_policies(
        "record_type",
        "status",
        exposure=Exposure.table_opt_in,
    ),
    TraceCoverageTableRow: _aggregate_policies(
        "record_type",
        "traced_column_count",
        "native_trace_column_count",
        "private_trace_column_count",
        "unsupported_column_count",
        exposure=Exposure.table_opt_in,
    ),
    WandbTablePayload: {
        "record_type": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.table_opt_in),
        "rows": FieldPolicy(data_class=DataClass.pseudonymous, exposure=Exposure.table_opt_in),
    },
    WandbPublishPayload: {
        "init": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "config": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "history": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "summary": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.aggregate),
        "tables": FieldPolicy(data_class=DataClass.pseudonymous, exposure=Exposure.table_opt_in),
    },
    WandbPublishResult: {
        "published": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.local_only),
        "run_id": FieldPolicy(data_class=DataClass.pseudonymous, exposure=Exposure.local_only),
        "entity": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.local_only),
        "project": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.local_only),
        "run_url": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.local_only),
        "publication_state": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.local_only),
        "measurement_sha256": FieldPolicy(data_class=DataClass.pseudonymous, exposure=Exposure.local_only),
        "record_count": FieldPolicy(data_class=DataClass.operational, exposure=Exposure.local_only),
    },
}


def validate_outbound_field_policies() -> None:
    for model, policies in OUTBOUND_FIELD_POLICIES.items():
        inherited_policies: dict[str, FieldPolicy] = {}
        for parent in reversed(model.mro()):
            inherited_policies.update(OUTBOUND_FIELD_POLICIES.get(cast(type[BaseModel], parent), {}))
        if set(model.model_fields) != set(inherited_policies):
            raise RuntimeError(f"incomplete outbound field policy for {model.__name__}")
        for name, policy in inherited_policies.items():
            if policy.data_class == DataClass.sensitive and policy.exposure != Exposure.local_only:
                raise RuntimeError(f"sensitive outbound field {model.__name__}.{name} is not local-only")
            if policy.exposure == Exposure.never:
                raise RuntimeError(f"never-exposed field present in outbound model: {model.__name__}.{name}")


validate_outbound_field_policies()


def _metric_has_aggregate_policy(key: str) -> bool:
    if key in BENCHMARK_METRIC_NAMES or key in {
        "measurement/record_count",
        PUBLICATION_COMPLETE_KEY,
        PUBLICATION_SEAL_DIGEST_KEY,
    }:
        return True
    parts = key.split("/")
    return (
        len(parts) == 3
        and parts[0] == "measurement"
        and parts[1] in _MEASUREMENT_RECORD_TYPES
        and parts[2] in AGGREGATED_MEASUREMENT_FIELDS
    )
