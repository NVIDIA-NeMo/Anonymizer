# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanitized benchmark metadata and W&B config projection."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    Field,
    FiniteFloat,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from measurement_tools.validation import (
    NonNegativeFloat,
    NonNegativeInt,
    Probability,
    StrictFrozenModel,
    VisibleIdentifier,
    VisibleSlurmIdentifier,
)
from measurement_tools.wandb_settings import WANDB_TAG_MAX_LENGTH, ResolvedWandbConfig, WandbMode

__all__ = [
    "BenchmarkMetadata",
    "ConfigMetadata",
    "DetectMetadata",
    "ExecutionMetadata",
    "GitMetadata",
    "ImportedRunMetadata",
    "MatrixMetadata",
    "ModelSourcesMetadata",
    "ReplaceMetadata",
    "RewriteMetadata",
    "RuntimeMetadata",
    "SafeScalar",
    "SlurmJobMetadata",
    "SlurmMetadata",
    "SweepMetadata",
    "WandbConfigPayload",
    "WandbRunMetadata",
    "WandbTag",
    "WorkloadMetadata",
    "WorkloadSourceMetadata",
]

SafeScalar = StrictBool | StrictInt | FiniteFloat | StrictStr
WandbTag = Annotated[StrictStr, Field(min_length=1, max_length=WANDB_TAG_MAX_LENGTH)]
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


class BenchmarkMetadata(StrictFrozenModel):
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


class SlurmJobMetadata(StrictFrozenModel):
    role: VisibleSlurmIdentifier
    job_id: VisibleSlurmIdentifier


class SlurmMetadata(StrictFrozenModel):
    job_id: VisibleIdentifier | None = None
    array_job_id: VisibleIdentifier | None = None
    array_task_id: VisibleIdentifier | None = None
    array_task_count: NonNegativeInt | None = None
    restart_count: NonNegativeInt | None = None
    ntasks: NonNegativeInt | None = None
    job_num_nodes: NonNegativeInt | None = None
    jobs: Annotated[list[SlurmJobMetadata], Field(max_length=64, exclude_if=lambda value: not value)] = Field(
        default_factory=list
    )


class ExecutionMetadata(StrictFrozenModel):
    backend: Literal["local", "slurm"] | None = None
    export: StrictBool | None = None
    fail_fast: StrictBool | None = None
    dd_trace: Literal["none", "last_message", "all_messages"] | None = None
    dd_task_trace: StrictBool | None = None
    slurm: SlurmMetadata | None = None


class RuntimeMetadata(StrictFrozenModel):
    anonymizer_version: StrictStr | None = None
    datadesigner_version: StrictStr | None = None
    wandb_version: StrictStr | None = None
    python_version: StrictStr | None = None
    platform_machine: StrictStr | None = None
    platform_system: StrictStr | None = None


class GitMetadata(StrictFrozenModel):
    commit: StrictStr | None = None
    branch: StrictStr | None = None
    dirty: StrictBool | None = None


class ModelSourcesMetadata(StrictFrozenModel):
    has_model_configs: StrictBool | None = None
    has_model_providers: StrictBool | None = None
    has_artifact_path: StrictBool | None = None


class WorkloadSourceMetadata(StrictFrozenModel):
    kind: Literal["local_file", "remote_file"] | None = None
    suffix: StrictStr | None = None


class WorkloadMetadata(StrictFrozenModel):
    id: VisibleIdentifier | None = None
    text_column: VisibleIdentifier | None = None
    has_id_column: StrictBool | None = None
    has_data_summary: StrictBool | None = None
    row_limit: NonNegativeInt | None = None
    row_offset: NonNegativeInt | None = None
    source: WorkloadSourceMetadata | None = None


class DetectMetadata(StrictFrozenModel):
    entity_label_source: Literal["custom", "default"] | None = None
    entity_label_count: NonNegativeInt | None = None
    entity_label_set_hash: StrictStr | None = None
    gliner_threshold: Probability | None = None
    validation_max_entities_per_call: NonNegativeInt | None = None
    validation_excerpt_window_chars: NonNegativeInt | None = None


class ReplaceMetadata(StrictFrozenModel):
    strategy: Literal["annotate", "hash", "redact", "substitute"] | None = None
    normalize_label: StrictBool | None = None
    algorithm: Literal["md5", "sha1", "sha256"] | None = None
    digest_length: NonNegativeInt | None = None
    has_format_template: StrictBool | None = None
    has_instructions: StrictBool | None = None


class RewriteMetadata(StrictFrozenModel):
    risk_tolerance: Literal["minimal", "low", "moderate", "high"] | None = None
    max_repair_iterations: NonNegativeInt | None = None
    strict_entity_protection: StrictBool | None = None
    has_privacy_goal: StrictBool | None = None
    has_protect: StrictBool | None = None
    has_preserve: StrictBool | None = None
    has_instructions: StrictBool | None = None


class ConfigMetadata(StrictFrozenModel):
    id: VisibleIdentifier | None = None
    mode: Literal["replace", "rewrite"] | None = None
    evaluate: StrictBool | None = None
    emit_telemetry: StrictBool | None = None
    detect: DetectMetadata | None = None
    replace: ReplaceMetadata | None = None
    rewrite: RewriteMetadata | None = None


class MatrixMetadata(StrictFrozenModel):
    workload: VisibleIdentifier | None = None
    config: VisibleIdentifier | None = None
    repetitions: NonNegativeInt | None = None


class SweepMetadata(StrictFrozenModel):
    id: VisibleIdentifier
    arm_id: VisibleIdentifier
    params: dict[StrictStr, SafeScalar]

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


class ImportedRunMetadata(StrictFrozenModel):
    completion_seal_schema_version: NonNegativeInt
    completion_seal_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    producer_repository: VisibleIdentifier
    producer_commit: StrictStr = Field(pattern=r"^[0-9a-f]{40,64}$")
    phase: VisibleIdentifier
    case_id: VisibleIdentifier


class WandbRunMetadata(StrictFrozenModel):
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
