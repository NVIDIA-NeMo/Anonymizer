# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sanitized W&B metadata for benchmark suites and cases."""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from anonymizer.config.anonymizer_config import infer_input_source_suffix, is_remote_input_source
from anonymizer.measurement import MEASUREMENT_SCHEMA_VERSION
from measurement_tools.benchmark_models import (
    BenchmarkCase,
    BenchmarkSpec,
    ConfigSpec,
    DDTraceMode,
    ReplaceSpec,
    RewriteSpec,
    WorkloadSpec,
)
from measurement_tools.benchmark_specs import cross_product_matrix
from measurement_tools.execution import benchmark_execution_metadata, stable_metadata_hash
from measurement_tools.wandb_setup import WANDB_SANITIZER_VERSION, WandbRunMetadata

BENCHMARK_SUITE_SCHEMA_VERSION = 1
WANDB_METADATA_SCHEMA_VERSION = 2


def run_tags(case: BenchmarkCase, spec: BenchmarkSpec) -> dict[str, Any]:
    return {
        **spec.run_tags,
        "suite_id": spec.suite_id,
        "workload_id": case.workload_id,
        "config_id": case.config_id,
        "repetition": case.repetition,
        "case_id": case.case_id,
    }


def build_wandb_metadata(
    spec: BenchmarkSpec,
    *,
    spec_path: Path,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
    dd_trace: DDTraceMode,
    dd_task_trace: bool,
    execution_metadata_builder: Callable[..., dict[str, Any]] | None = None,
    git_metadata_builder: Callable[[Path], dict[str, str | bool | None]] | None = None,
    package_versions_builder: Callable[[], dict[str, str | None]] | None = None,
) -> WandbRunMetadata:
    execution_builder = execution_metadata_builder or execution_metadata
    git_builder = git_metadata_builder or git_metadata
    packages_builder = package_versions_builder or package_versions
    sweep = sweep_metadata(spec.run_tags)
    metadata = {
        "run_kind": "sweep_arm" if sweep is not None else "native_suite",
        "benchmark": benchmark_metadata(spec, spec_path=spec_path),
        "execution": execution_builder(
            output_dir=output_dir,
            export=export,
            fail_fast=fail_fast,
            dd_trace=dd_trace,
            dd_task_trace=dd_task_trace,
        ),
        "runtime": runtime_metadata(package_versions_builder=packages_builder),
        "git": git_builder(Path.cwd()),
        "model_sources": model_source_metadata(spec),
        "workloads": [workload_metadata(workload) for workload in spec.workloads],
        "configs": [config_metadata(config) for config in spec.configs],
        "matrix": [entry.model_dump(mode="json") for entry in (spec.matrix or cross_product_matrix(spec))],
        "sweep": sweep,
    }
    return WandbRunMetadata.model_validate(metadata, strict=True)


def benchmark_metadata(spec: BenchmarkSpec, *, spec_path: Path) -> dict[str, Any]:
    matrix = spec.matrix or cross_product_matrix(spec)
    metadata = {
        "metadata_schema_version": WANDB_METADATA_SCHEMA_VERSION,
        "suite_schema_version": BENCHMARK_SUITE_SCHEMA_VERSION,
        "wandb_sanitizer_version": WANDB_SANITIZER_VERSION,
        "measurement_schema_version": MEASUREMENT_SCHEMA_VERSION,
        "suite_id": spec.suite_id,
        "workload_count": len(spec.workloads),
        "config_count": len(spec.configs),
        "matrix_entry_count": len(matrix),
        "case_count": sum(entry.repetitions for entry in matrix),
        "case_retries": spec.case_retries,
        "case_retry_backoff_sec": spec.case_retry_backoff_sec,
    }
    if spec_hash := file_hash(spec_path):
        metadata["suite_file_hash"] = spec_hash
    return metadata


def execution_metadata(
    *,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
    dd_trace: DDTraceMode,
    dd_task_trace: bool,
) -> dict[str, Any]:
    metadata = benchmark_execution_metadata(
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace.value,
        dd_task_trace=dd_task_trace,
    )
    metadata.pop("output_dir_hash", None)
    return metadata


def runtime_metadata(*, package_versions_builder: Callable[[], dict[str, str | None]] | None = None) -> dict[str, Any]:
    versions_builder = package_versions_builder or package_versions
    return {
        **versions_builder(),
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
    }


def package_versions() -> dict[str, str | None]:
    return {
        "anonymizer_version": package_version("nemo-anonymizer"),
        "datadesigner_version": package_version("data-designer"),
        "wandb_version": package_version("wandb"),
    }


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def model_source_metadata(spec: BenchmarkSpec) -> dict[str, bool]:
    return {
        "has_model_configs": spec.model_configs is not None,
        "has_model_providers": spec.model_providers is not None,
        "has_artifact_path": spec.artifact_path is not None,
    }


def workload_metadata(workload: WorkloadSpec) -> dict[str, Any]:
    return {
        "id": workload.id,
        "source": source_metadata(workload.source),
        "text_column": workload.text_column,
        "has_id_column": workload.id_column is not None,
        "has_data_summary": workload.data_summary is not None,
        "row_limit": workload.row_limit,
        "row_offset": workload.row_offset,
    }


def source_metadata(source: str) -> dict[str, str | None]:
    return {
        "kind": "remote_file" if is_remote_input_source(source) else "local_file",
        "suffix": source_suffix(source),
    }


def source_suffix(source: str) -> str | None:
    try:
        return infer_input_source_suffix(source)
    except ValueError:
        return None


def config_metadata(config: ConfigSpec) -> dict[str, Any]:
    return {
        "id": config.id,
        "mode": "rewrite" if config.rewrite is not None else "replace",
        "detect": detect_metadata(config.detect),
        "replace": replace_metadata(config.replace),
        "rewrite": rewrite_metadata(config.rewrite),
        "evaluate": config.evaluate,
        "emit_telemetry": config.emit_telemetry,
    }


def sweep_metadata(run_tags: dict[str, Any]) -> dict[str, Any] | None:
    sweep = run_tags.get("wandb_sweep")
    return sweep if isinstance(sweep, dict) else None


def detect_metadata(detect: dict[str, Any]) -> dict[str, Any]:
    entity_labels = detect.get("entity_labels")
    metadata = {
        "entity_label_source": "custom" if isinstance(entity_labels, list) else "default",
        "entity_label_count": len(entity_labels) if isinstance(entity_labels, list) else None,
    }
    if isinstance(entity_labels, list):
        metadata["entity_label_set_hash"] = stable_hash(",".join(sorted(map(str, entity_labels))))
    for key in ("gliner_threshold", "validation_max_entities_per_call", "validation_excerpt_window_chars"):
        if key in detect:
            metadata[key] = detect[key]
    return metadata


def replace_metadata(replace: str | ReplaceSpec | None) -> dict[str, Any] | None:
    if replace is None:
        return None
    if isinstance(replace, str):
        return {"strategy": replace}
    metadata: dict[str, Any] = {"strategy": replace.strategy.value}
    for key in ("normalize_label", "algorithm", "digest_length"):
        value = getattr(replace, key)
        if value is not None:
            metadata[key] = value
    if replace.format_template is not None:
        metadata["has_format_template"] = True
    if replace.instructions is not None:
        metadata["has_instructions"] = True
    return metadata


def rewrite_metadata(rewrite: RewriteSpec | None) -> dict[str, Any] | None:
    if rewrite is None:
        return None
    return {
        "risk_tolerance": rewrite.risk_tolerance.value,
        "max_repair_iterations": rewrite.max_repair_iterations,
        "strict_entity_protection": rewrite.strict_entity_protection,
        "has_privacy_goal": rewrite.protect is not None or rewrite.preserve is not None,
        "has_protect": rewrite.protect is not None,
        "has_preserve": rewrite.preserve is not None,
        "has_instructions": rewrite.instructions is not None,
    }


def git_metadata(cwd: Path) -> dict[str, str | bool | None]:
    commit = git_output(cwd, "rev-parse", "HEAD")
    branch = git_output(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    status = git_output(cwd, "status", "--short")
    return {"commit": commit, "branch": branch, "dirty": bool(status) if status is not None else None}


def git_output(cwd: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return stable_hash(path.read_text(encoding="utf-8"))


def stable_hash(value: str) -> str:
    return stable_metadata_hash(value)


__all__ = [
    "BENCHMARK_SUITE_SCHEMA_VERSION",
    "WANDB_METADATA_SCHEMA_VERSION",
    "benchmark_metadata",
    "build_wandb_metadata",
    "config_metadata",
    "detect_metadata",
    "execution_metadata",
    "file_hash",
    "git_metadata",
    "git_output",
    "model_source_metadata",
    "package_version",
    "package_versions",
    "replace_metadata",
    "rewrite_metadata",
    "run_tags",
    "runtime_metadata",
    "source_metadata",
    "source_suffix",
    "stable_hash",
    "sweep_metadata",
    "workload_metadata",
]
