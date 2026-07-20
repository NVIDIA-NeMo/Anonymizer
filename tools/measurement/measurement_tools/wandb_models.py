# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration and outbound models for native W&B publication."""

from __future__ import annotations

from typing import cast

from pydantic import BaseModel

from measurement_tools.validation import StrictFrozenModel as StrictFrozenModel
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
    SlurmJobMetadata,
    SlurmMetadata,
    SweepMetadata,
    WandbConfigPayload,
    WandbRunMetadata,
    WorkloadMetadata,
    WorkloadSourceMetadata,
)
from measurement_tools.wandb_metadata import (
    SafeScalar as SafeScalar,
)
from measurement_tools.wandb_metadata import (
    WandbTag as WandbTag,
)
from measurement_tools.wandb_metrics import (
    _METRIC_TABLE_POLICY_FIELDS,
    EvaluationTableRow,
    ModelWorkflowTableRow,
    NddWorkflowTableRow,
    RecordTableRow,
    RunTableRow,
    StageTableRow,
    TraceCoverageTableRow,
    WandbHistoryPayload,
    WandbSummaryPayload,
    WandbTablePayload,
    _MetricTableRow,
)
from measurement_tools.wandb_metrics import (
    WANDB_TABLE_ROW_MODELS as WANDB_TABLE_ROW_MODELS,
)
from measurement_tools.wandb_metrics import (
    WandbTableRow as WandbTableRow,
)
from measurement_tools.wandb_policy import DataClass, Exposure, FieldPolicy
from measurement_tools.wandb_publication import (
    PUBLICATION_COMPLETE_KEY as PUBLICATION_COMPLETE_KEY,
)
from measurement_tools.wandb_publication import (
    PUBLICATION_SEAL_DIGEST_KEY as PUBLICATION_SEAL_DIGEST_KEY,
)
from measurement_tools.wandb_publication import (
    WandbInitPayload,
    WandbPublishPayload,
    WandbPublishResult,
)
from measurement_tools.wandb_publication import (
    WandbPublicationState as WandbPublicationState,
)
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
    WandbMode as WandbMode,
)
from measurement_tools.wandb_settings import (
    generated_wandb_tag as generated_wandb_tag,
)
from measurement_tools.wandb_settings import (
    is_safe_wandb_tag as is_safe_wandb_tag,
)
from measurement_tools.wandb_settings import wandb_tag_value_is_sensitive as wandb_tag_value_is_sensitive


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
