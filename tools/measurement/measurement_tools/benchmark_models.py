# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark suite models and their validation policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from anonymizer.config.rewrite import RiskTolerance

RESERVED_RUN_TAG_KEYS = frozenset({"suite_id", "workload_id", "config_id", "repetition", "case_id"})


class CaseStatus(StrEnum):
    planned = "planned"
    completed = "completed"
    error = "error"


class DDTraceMode(StrEnum):
    none = "none"
    last_message = "last_message"
    all_messages = "all_messages"


class ReplaceKind(StrEnum):
    redact = "redact"
    hash = "hash"
    annotate = "annotate"
    substitute = "substitute"


class WorkloadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    text_column: str = "text"
    id_column: str | None = None
    data_summary: str | None = None
    row_limit: int | None = Field(default=None, ge=1)
    row_offset: int = Field(default=0, ge=0)


class ReplaceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: ReplaceKind
    format_template: str | None = None
    normalize_label: bool | None = None
    algorithm: str | None = None
    digest_length: int | None = None
    instructions: str | None = None


class RewriteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protect: str | None = None
    preserve: str | None = None
    instructions: str | None = None
    risk_tolerance: RiskTolerance = RiskTolerance.low
    max_repair_iterations: int = 3
    strict_entity_protection: bool = False


class ConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    detect: dict[str, Any] = Field(default_factory=dict)
    replace: str | ReplaceSpec | None = None
    rewrite: RewriteSpec | None = None
    evaluate: bool = False
    emit_telemetry: bool = False

    @model_validator(mode="after")
    def validate_mode(self) -> "ConfigSpec":
        if self.replace is None and self.rewrite is None:
            raise ValueError("config must define replace or rewrite")
        if self.replace is not None and self.rewrite is not None:
            raise ValueError("config cannot define both replace and rewrite")
        if self.evaluate and self.rewrite is not None:
            raise ValueError("evaluate is only supported for replace configs")
        return self


class MatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload: str
    config: str
    repetitions: int = Field(default=1, ge=1)


def duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def duplicate_matrix_entries(matrix: list[MatrixEntry]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()
    for entry in matrix:
        key = (entry.workload, entry.config)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    model_configs: str | None = None
    model_providers: str | None = None
    artifact_path: str | None = None
    run_tags: dict[str, Any] = Field(default_factory=dict)
    case_retries: int = Field(default=0, ge=0)
    case_retry_backoff_sec: float = Field(default=0.0, ge=0.0)
    workloads: list[WorkloadSpec] = Field(min_length=1)
    configs: list[ConfigSpec] = Field(min_length=1)
    matrix: list[MatrixEntry] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_ids(self) -> "BenchmarkSpec":
        workload_ids = [workload.id for workload in self.workloads]
        config_ids = [config.id for config in self.configs]
        if duplicate_workloads := duplicates(workload_ids):
            raise ValueError(f"duplicate workload id(s): {', '.join(duplicate_workloads)}")
        if duplicate_configs := duplicates(config_ids):
            raise ValueError(f"duplicate config id(s): {', '.join(duplicate_configs)}")
        self._validate_matrix_references(set(workload_ids), set(config_ids))
        self._validate_run_tags()
        return self

    def _validate_matrix_references(self, workload_ids: set[str], config_ids: set[str]) -> None:
        if self.matrix is None:
            return
        missing_workloads = sorted({entry.workload for entry in self.matrix} - workload_ids)
        missing_configs = sorted({entry.config for entry in self.matrix} - config_ids)
        if missing_workloads:
            raise ValueError(f"matrix references unknown workload id(s): {', '.join(missing_workloads)}")
        if missing_configs:
            raise ValueError(f"matrix references unknown config id(s): {', '.join(missing_configs)}")
        duplicate_entries = duplicate_matrix_entries(self.matrix)
        if duplicate_entries:
            formatted = ", ".join(f"{workload}/{config}" for workload, config in duplicate_entries)
            raise ValueError(f"duplicate matrix workload/config entry(s): {formatted}; use repetitions for repeats")

    def _validate_run_tags(self) -> None:
        reserved_tags = sorted(set(self.run_tags) & RESERVED_RUN_TAG_KEYS)
        if reserved_tags:
            formatted = ", ".join(reserved_tags)
            raise ValueError(f"run_tags cannot define reserved benchmark tag(s): {formatted}")


class BenchmarkCase(BaseModel):
    suite_id: str
    workload_id: str
    config_id: str
    repetition: int
    case_id: str
    status: CaseStatus = CaseStatus.planned
    elapsed_sec: float | None = None
    measurement_path: str | None = None
    detection_artifact_path: str | None = None
    trace_path: str | None = None
    task_trace_path: str | None = None
    error: str | None = None
    attempt_count: int = 0
    attempt_errors: list[str] = Field(default_factory=list)


class BenchmarkResult(BaseModel):
    suite_id: str
    output_dir: str
    measurement_path: str
    summary_path: str
    table_dir: str | None
    detection_artifact_analysis_path: str | None = None
    cases: list[BenchmarkCase]
    execution: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class CaseRunPaths:
    raw_path: Path
    artifact_output_path: Path
    trace_path: Path | None
    task_trace_path: Path | None
    artifact_snapshot: dict[str, int] | None
    export_detection_artifacts: bool


__all__ = [
    "BenchmarkCase",
    "BenchmarkResult",
    "BenchmarkSpec",
    "CaseRunPaths",
    "CaseStatus",
    "ConfigSpec",
    "DDTraceMode",
    "MatrixEntry",
    "ReplaceKind",
    "ReplaceSpec",
    "RESERVED_RUN_TAG_KEYS",
    "RewriteSpec",
    "WorkloadSpec",
    "duplicate_matrix_entries",
    "duplicates",
]
