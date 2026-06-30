# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed configuration and outbound models for native W&B publication."""

from __future__ import annotations

import re
from collections.abc import Mapping
from enum import StrEnum
from ipaddress import ip_address
from pathlib import Path
from typing import Annotated, Any, Literal, cast
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_WANDB_PROJECT = "nemo-anonymizer-benchmarks"
PUBLICATION_COMPLETE_KEY = "publication/complete"
PUBLICATION_SEAL_DIGEST_KEY = "publication/completion_seal_sha256"
WANDB_TAG_MAX_LENGTH = 64
_SENSITIVE_WANDB_TAG_PARTS = frozenset({"api_key", "credential", "credentials", "password", "secret", "token"})
_SENSITIVE_WANDB_TAG_VALUE_PREFIXES = ("sk-", "ghp_", "github_pat_", "glpat-", "xoxb-", "xoxp-", "xoxa-")


class WandbMode(StrEnum):
    online = "online"
    offline = "offline"
    disabled = "disabled"


class WandbInputs(BaseSettings):
    """Optional operator inputs; dedicated measurement variables only."""

    wandb_mode: WandbMode | None = None
    wandb_project: str | None = None
    wandb_base_url: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: str | None = None
    wandb_log_tables: bool | None = None

    model_config = SettingsConfigDict(
        env_prefix="ANONYMIZER_MEASUREMENT_",
        extra="forbid",
        env_ignore_empty=True,
        populate_by_name=True,
        hide_input_in_errors=True,
    )


class ResolvedWandbConfig(BaseModel):
    """Immutable, fully resolved publisher configuration."""

    wandb_mode: WandbMode = WandbMode.disabled
    wandb_project: str = DEFAULT_WANDB_PROJECT
    wandb_base_url: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_job_type: str | None = None
    wandb_run_name: str | None = None
    wandb_tags: str = ""
    wandb_log_tables: bool = False

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)

    @field_validator(
        "wandb_project",
        "wandb_base_url",
        "wandb_entity",
        "wandb_group",
        "wandb_job_type",
        "wandb_run_name",
    )
    @classmethod
    def nonempty_optional_strings(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("W&B string settings cannot be empty")
        return stripped

    @field_validator("wandb_base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("W&B base URL must be a credential-free HTTP(S) URL")
        if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
            raise ValueError("W&B base URL must use HTTPS unless it targets loopback")
        return value.rstrip("/")

    @field_validator("wandb_tags")
    @classmethod
    def validate_tags(cls, value: str) -> str:
        tags = [tag for tag in (part.strip() for part in value.split(",")) if tag]
        if any(not is_safe_wandb_tag(tag) for tag in tags):
            raise ValueError("W&B tags must be safe identifiers between 1 and 64 characters")
        return value

    @property
    def enabled(self) -> bool:
        return self.wandb_mode != WandbMode.disabled

    @property
    def effective_wandb_project(self) -> str:
        return self.wandb_project

    @property
    def effective_wandb_tags(self) -> list[str]:
        return [tag for tag in (part.strip() for part in self.wandb_tags.split(",")) if tag]

    @classmethod
    def from_env_and_overrides(
        cls,
        *,
        defaults: Mapping[str, Any] | None = None,
        **overrides: Any,
    ) -> ResolvedWandbConfig:
        source = WandbInputs()
        values = dict(defaults or {})
        values.update(source.model_dump(exclude_none=True))
        values.update({key: value for key, value in overrides.items() if value is not None})
        return cls.model_validate(values)

    def validated_update(self, **updates: Any) -> ResolvedWandbConfig:
        values = self.model_dump()
        values.update(updates)
        return type(self).model_validate(values)


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def is_safe_wandb_tag(value: str) -> bool:
    if not value or len(value) > WANDB_TAG_MAX_LENGTH or "://" in value or value.startswith("/"):
        return False
    return not wandb_tag_value_is_sensitive(value)


def wandb_tag_value_is_sensitive(value: str) -> bool:
    normalized = value.lower()
    if normalized.startswith(_SENSITIVE_WANDB_TAG_VALUE_PREFIXES):
        return True
    parts = {part for part in re.split(r"[^a-z0-9]+", normalized) if part}
    if "api" in parts and "key" in parts:
        return True
    return bool(parts & _SENSITIVE_WANDB_TAG_PARTS)


class DataClass(StrEnum):
    operational = "operational"
    pseudonymous = "pseudonymous"
    sensitive = "sensitive"


class Exposure(StrEnum):
    local_only = "local_only"
    aggregate = "aggregate"
    table_opt_in = "table_opt_in"
    never = "never"


class FieldPolicy(BaseModel):
    data_class: DataClass
    exposure: Exposure

    model_config = ConfigDict(extra="forbid", frozen=True)


SafeScalar = StrictBool | StrictInt | FiniteFloat | StrictStr
VisibleIdentifier = Annotated[StrictStr, Field(min_length=1, max_length=256)]
WandbTag = Annotated[StrictStr, Field(min_length=1, max_length=WANDB_TAG_MAX_LENGTH)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]
MeasurementStrategy = Literal["Annotate", "Hash", "Redact", "Rewrite", "Substitute"]
_BENCHMARK_METRIC_NAMES = frozenset(
    {
        "benchmark/case_total",
        "benchmark/case_completed",
        "benchmark/case_errored",
        "benchmark/case_success_rate",
        "benchmark/case_elapsed_sec_sum",
        "benchmark/case_elapsed_sec_mean",
    }
)
_MEASUREMENT_RECORD_TYPES = frozenset(
    {"run", "stage", "record", "evaluation_record", "ndd_workflow", "model_workflow", "dd_trace_coverage"}
)
_MEASUREMENT_METRIC_FIELDS = frozenset(
    {
        "mode",
        "stage",
        "status",
        "strategy",
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
        "observed_failed_request_rate_mean",
        "input_rows_per_sec_mean",
        "output_rows_per_sec_mean",
        "observed_tokens_per_sec_mean",
        "observed_requests_per_sec_mean",
        "observed_tokens_per_successful_request_mean",
        "entity_precision_mean",
        "entity_recall_mean",
        "entity_f1_mean",
        "utility_score_mean",
        "leakage_mass_mean",
        "weighted_leakage_rate_mean",
        "repair_iterations_mean",
        "traced_column_count",
        "native_trace_column_count",
        "private_trace_column_count",
        "unsupported_column_count",
    }
)
_NUMERIC_SWEEP_PARAMETER_TAILS = frozenset(
    {
        "detect_gliner_threshold",
        "detect_validation_excerpt_window_chars",
        "detect_validation_max_entities_per_call",
        "replace_digest_length",
        "rewrite_max_repair_iterations",
    }
)
_BOOLEAN_SWEEP_PARAMETER_TAILS = frozenset(
    {"emit_telemetry", "evaluate", "replace_normalize_label", "rewrite_strict_entity_protection"}
)
_ENUM_SWEEP_PARAMETER_VALUES = {
    "replace_algorithm": frozenset({"md5", "sha1", "sha256"}),
    "rewrite_risk_tolerance": frozenset({"high", "low", "minimal", "moderate"}),
}


class WandbInitPayload(BaseModel):
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

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class BenchmarkMetadata(BaseModel):
    metadata_schema_version: NonNegativeInt | None = None
    suite_schema_version: NonNegativeInt | None = None
    wandb_sanitizer_version: NonNegativeInt | None = None
    measurement_schema_version: NonNegativeInt | None = None
    suite_id: VisibleIdentifier | None = None
    workload_count: NonNegativeInt | None = None
    config_count: NonNegativeInt | None = None
    matrix_entry_count: NonNegativeInt | None = None
    case_count: NonNegativeInt | None = None
    case_retries: NonNegativeInt | None = None
    case_retry_backoff_sec: NonNegativeFloat | None = None
    suite_file_hash: StrictStr | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SlurmMetadata(BaseModel):
    job_id: VisibleIdentifier | None = None
    array_job_id: VisibleIdentifier | None = None
    array_task_id: VisibleIdentifier | None = None
    array_task_count: NonNegativeInt | None = None
    restart_count: NonNegativeInt | None = None
    ntasks: NonNegativeInt | None = None
    job_num_nodes: NonNegativeInt | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExecutionMetadata(BaseModel):
    backend: Literal["local", "slurm"] | None = None
    export: StrictBool | None = None
    fail_fast: StrictBool | None = None
    dd_trace: Literal["none", "last_message", "all_messages"] | None = None
    dd_task_trace: StrictBool | None = None
    slurm: SlurmMetadata | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RuntimeMetadata(BaseModel):
    anonymizer_version: StrictStr | None = None
    datadesigner_version: StrictStr | None = None
    wandb_version: StrictStr | None = None
    python_version: StrictStr | None = None
    platform_machine: StrictStr | None = None
    platform_system: StrictStr | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GitMetadata(BaseModel):
    commit: StrictStr | None = None
    branch: StrictStr | None = None
    dirty: StrictBool | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ModelSourcesMetadata(BaseModel):
    has_model_configs: StrictBool | None = None
    has_model_providers: StrictBool | None = None
    has_artifact_path: StrictBool | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WorkloadSourceMetadata(BaseModel):
    kind: Literal["local_file", "remote_file"] | None = None
    suffix: StrictStr | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WorkloadMetadata(BaseModel):
    id: VisibleIdentifier | None = None
    text_column: VisibleIdentifier | None = None
    has_id_column: StrictBool | None = None
    has_data_summary: StrictBool | None = None
    row_limit: NonNegativeInt | None = None
    row_offset: NonNegativeInt | None = None
    source: WorkloadSourceMetadata | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class DetectMetadata(BaseModel):
    entity_label_source: Literal["custom", "default"] | None = None
    entity_label_count: NonNegativeInt | None = None
    entity_label_set_hash: StrictStr | None = None
    gliner_threshold: Probability | None = None
    validation_max_entities_per_call: NonNegativeInt | None = None
    validation_excerpt_window_chars: NonNegativeInt | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ReplaceMetadata(BaseModel):
    strategy: Literal["annotate", "hash", "redact", "substitute"] | None = None
    normalize_label: StrictBool | None = None
    algorithm: Literal["md5", "sha1", "sha256"] | None = None
    digest_length: NonNegativeInt | None = None
    has_format_template: StrictBool | None = None
    has_instructions: StrictBool | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RewriteMetadata(BaseModel):
    risk_tolerance: Literal["minimal", "low", "moderate", "high"] | None = None
    max_repair_iterations: NonNegativeInt | None = None
    strict_entity_protection: StrictBool | None = None
    has_privacy_goal: StrictBool | None = None
    has_protect: StrictBool | None = None
    has_preserve: StrictBool | None = None
    has_instructions: StrictBool | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ConfigMetadata(BaseModel):
    id: VisibleIdentifier | None = None
    mode: Literal["replace", "rewrite"] | None = None
    evaluate: StrictBool | None = None
    emit_telemetry: StrictBool | None = None
    detect: DetectMetadata | None = None
    replace: ReplaceMetadata | None = None
    rewrite: RewriteMetadata | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MatrixMetadata(BaseModel):
    workload: VisibleIdentifier | None = None
    config: VisibleIdentifier | None = None
    repetitions: NonNegativeInt | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SweepMetadata(BaseModel):
    id: VisibleIdentifier
    arm_id: VisibleIdentifier
    params: dict[StrictStr, SafeScalar]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("params")
    @classmethod
    def validate_params(cls, params: dict[str, SafeScalar]) -> dict[str, SafeScalar]:
        for key, value in params.items():
            if not _sweep_parameter_is_safe(key, value):
                raise ValueError(f"unsupported W&B sweep parameter: {key}")
        return params

    @classmethod
    def from_arm(cls, *, sweep_id: str, arm_id: str, params: dict[str, Any]) -> SweepMetadata:
        safe_params = {
            key: value
            for key, value in params.items()
            if isinstance(value, bool | int | float | str) and _sweep_parameter_is_safe(key, value)
        }
        return cls(id=sweep_id, arm_id=arm_id, params=safe_params)


class ImportedRunMetadata(BaseModel):
    completion_seal_schema_version: NonNegativeInt
    completion_seal_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    producer_repository: VisibleIdentifier
    producer_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40,64}$")
    phase: VisibleIdentifier
    case_id: VisibleIdentifier

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WandbRunMetadata(BaseModel):
    """Validated benchmark metadata that is safe to project into W&B config."""

    run_kind: Literal["native_suite", "sweep_arm", "imported_case"] = "native_suite"
    benchmark: BenchmarkMetadata | None = None
    execution: ExecutionMetadata | None = None
    runtime: RuntimeMetadata | None = None
    git: GitMetadata | None = None
    model_sources: ModelSourcesMetadata | None = None
    workloads: tuple[WorkloadMetadata, ...] = ()
    configs: tuple[ConfigMetadata, ...] = ()
    matrix: tuple[MatrixMetadata, ...] = ()
    sweep: SweepMetadata | None = None
    imported: ImportedRunMetadata | None = None

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("workloads", "configs", "matrix", mode="before")
    @classmethod
    def freeze_sequences(cls, value: Any) -> tuple[Any, ...]:
        if not isinstance(value, list | tuple):
            raise ValueError("W&B metadata sequences must be lists or tuples")
        return tuple(value)

    @model_validator(mode="after")
    def validate_run_kind(self) -> WandbRunMetadata:
        if (self.sweep is not None) != (self.run_kind == "sweep_arm"):
            raise ValueError("sweep metadata and sweep_arm run kind must be set together")
        if (self.imported is not None) != (self.run_kind == "imported_case"):
            raise ValueError("import metadata and imported_case run kind must be set together")
        if self.benchmark is None or self.benchmark.suite_id is None:
            raise ValueError("W&B run metadata requires a benchmark suite identity")
        if self.run_kind == "imported_case":
            config_ids = [config.id for config in self.configs if config.id is not None]
            if len(config_ids) != 1:
                raise ValueError("imported W&B metadata requires exactly one config identity")
        return self

    def comparer_values(self) -> dict[str, SafeScalar]:
        values: dict[str, SafeScalar] = {}
        benchmark = self.benchmark
        if benchmark is not None:
            _set_scalar(values, "benchmark_suite_id", benchmark.suite_id)
            _set_scalar(values, "benchmark_case_count", benchmark.case_count)
        _set_scalar(values, "benchmark_workload_ids", _compact_values(item.id for item in self.workloads))
        _set_scalar(
            values,
            "benchmark_workload_row_limits",
            _compact_values(item.row_limit for item in self.workloads),
        )
        _set_scalar(
            values,
            "benchmark_workload_source_kinds",
            _compact_values(item.source.kind if item.source else None for item in self.workloads),
        )
        _set_scalar(
            values,
            "benchmark_workload_source_suffixes",
            _compact_values(item.source.suffix if item.source else None for item in self.workloads),
        )
        _set_scalar(values, "benchmark_config_ids", _compact_values(item.id for item in self.configs))
        _set_scalar(values, "benchmark_modes", _compact_values(item.mode for item in self.configs))
        _set_scalar(values, "benchmark_strategies", _compact_values(_config_strategy(item) for item in self.configs))
        _set_scalar(
            values,
            "benchmark_gliner_thresholds",
            _compact_values(item.detect.gliner_threshold if item.detect else None for item in self.configs),
        )
        _set_scalar(
            values,
            "benchmark_entity_label_counts",
            _compact_values(item.detect.entity_label_count if item.detect else None for item in self.configs),
        )
        _set_scalar(
            values,
            "benchmark_risk_tolerances",
            _compact_values(item.rewrite.risk_tolerance if item.rewrite else None for item in self.configs),
        )
        return values


class WandbConfigPayload(WandbRunMetadata):
    suite_id: VisibleIdentifier
    wandb_mode: WandbMode
    wandb_log_tables: StrictBool
    benchmark_suite_id: VisibleIdentifier | None = None
    benchmark_case_count: NonNegativeInt | None = None
    benchmark_workload_ids: SafeScalar | None = None
    benchmark_workload_row_limits: SafeScalar | None = None
    benchmark_workload_source_kinds: SafeScalar | None = None
    benchmark_workload_source_suffixes: SafeScalar | None = None
    benchmark_config_ids: SafeScalar | None = None
    benchmark_modes: SafeScalar | None = None
    benchmark_strategies: SafeScalar | None = None
    benchmark_gliner_thresholds: SafeScalar | None = None
    benchmark_entity_label_counts: SafeScalar | None = None
    benchmark_risk_tolerances: SafeScalar | None = None
    native_suite_id: VisibleIdentifier | None = None
    sweep_id: VisibleIdentifier | None = None
    sweep_arm_id: VisibleIdentifier | None = None
    imported_config_id: VisibleIdentifier | None = None
    sweep_params: dict[StrictStr, SafeScalar] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    def sdk_values(self) -> dict[str, Any]:
        values = self.model_dump(mode="json", exclude_none=True)
        values.update({f"sweep_param_{key}": value for key, value in self.sweep_params.items()})
        return values

    @classmethod
    def from_run_metadata(
        cls,
        settings: ResolvedWandbConfig,
        *,
        suite_id: str,
        metadata: WandbRunMetadata,
    ) -> WandbConfigPayload:
        values: dict[str, Any] = {
            "suite_id": suite_id,
            "wandb_mode": settings.wandb_mode,
            "wandb_log_tables": settings.wandb_log_tables,
        }
        if metadata.benchmark is None or metadata.benchmark.suite_id != suite_id:
            raise ValueError("W&B metadata suite identity does not match publication suite")
        values.update({name: getattr(metadata, name) for name in WandbRunMetadata.model_fields})
        values.update(metadata.comparer_values())
        if metadata.run_kind == "native_suite":
            values["native_suite_id"] = metadata.benchmark.suite_id
        if metadata.sweep is not None:
            values.update(
                {
                    "sweep_id": metadata.sweep.id,
                    "sweep_arm_id": metadata.sweep.arm_id,
                    "sweep_params": metadata.sweep.params,
                }
            )
        if metadata.run_kind == "imported_case":
            values["imported_config_id"] = metadata.configs[0].id
        return cls.model_validate(values, strict=True)


class WandbHistoryPayload(BaseModel):
    metrics: dict[StrictStr, SafeScalar]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("metrics")
    @classmethod
    def metric_keys_have_policy(cls, metrics: dict[str, SafeScalar]) -> dict[str, SafeScalar]:
        unknown = [key for key in metrics if not _metric_has_aggregate_policy(key)]
        if unknown:
            raise ValueError(f"W&B metric has no aggregate exposure policy: {unknown!r}")
        return metrics


class WandbSummaryPayload(BaseModel):
    metrics: dict[StrictStr, SafeScalar]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("metrics")
    @classmethod
    def metric_keys_have_policy(cls, metrics: dict[str, SafeScalar]) -> dict[str, SafeScalar]:
        return WandbHistoryPayload.metric_keys_have_policy(metrics)


class _MetricTableRow(BaseModel):
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

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


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


class TraceCoverageTableRow(BaseModel):
    record_type: Literal["dd_trace_coverage"]
    traced_column_count: NonNegativeInt
    native_trace_column_count: NonNegativeInt
    private_trace_column_count: NonNegativeInt
    unsupported_column_count: NonNegativeInt

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


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


class WandbTablePayload(BaseModel):
    record_type: StrictStr
    rows: tuple[WandbTableRow, ...]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

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


class WandbPublishPayload(BaseModel):
    init: WandbInitPayload
    config: WandbConfigPayload
    history: WandbHistoryPayload
    summary: WandbSummaryPayload
    tables: tuple[WandbTablePayload, ...] = ()

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

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


class WandbPublishResult(BaseModel):
    published: StrictBool
    run_id: StrictStr | None = None
    measurement_sha256: Annotated[StrictStr | None, Field(pattern=r"^[0-9a-f]{64}$")] = None
    record_count: StrictInt = 0

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _set_scalar(mapping: dict[str, SafeScalar], key: str, value: Any) -> None:
    if isinstance(value, bool | int | float | str):
        mapping[key] = value


def _compact_values(values: Any) -> SafeScalar | None:
    compacted: list[SafeScalar] = []
    seen: set[tuple[str, SafeScalar]] = set()
    for value in values:
        if not isinstance(value, bool | int | float | str):
            continue
        marker = (type(value).__name__, value)
        if marker in seen:
            continue
        compacted.append(value)
        seen.add(marker)
    if len(compacted) == 1:
        return compacted[0]
    if len(compacted) > 1:
        return ",".join(str(value) for value in compacted)
    return None


def _config_strategy(config: ConfigMetadata) -> str | None:
    if config.replace is not None:
        return config.replace.strategy
    return "rewrite" if config.rewrite is not None else None


def _sweep_parameter_is_safe(key: str, value: SafeScalar) -> bool:
    if not key.startswith("configs_") or not key.isascii() or any(not (char.isalnum() or char == "_") for char in key):
        return False
    if any(key.endswith(f"_{tail}") for tail in _NUMERIC_SWEEP_PARAMETER_TAILS):
        if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
            return False
        return not key.endswith("_detect_gliner_threshold") or value <= 1
    if any(key.endswith(f"_{tail}") for tail in _BOOLEAN_SWEEP_PARAMETER_TAILS):
        return isinstance(value, bool)
    return any(key.endswith(f"_{tail}") and value in allowed for tail, allowed in _ENUM_SWEEP_PARAMETER_VALUES.items())


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
    SlurmMetadata: _aggregate_policies(
        "job_id",
        "array_job_id",
        "array_task_id",
        "array_task_count",
        "restart_count",
        "ntasks",
        "job_num_nodes",
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
    if key in _BENCHMARK_METRIC_NAMES or key in {
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
        and parts[2] in _MEASUREMENT_METRIC_FIELDS
    )
