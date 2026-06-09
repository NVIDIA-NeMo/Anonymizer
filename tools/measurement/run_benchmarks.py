#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Anonymizer benchmark suites and export measurement tables.

Usage:
    uv run python tools/measurement/run_benchmarks.py suite.yaml --output benchmark-runs/suite
    uv run python tools/measurement/run_benchmarks.py suite.yaml --dry-run --json
"""

import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import pandas as pd
import pyarrow.parquet as pq
import yaml
from analyze_detection_artifacts import (
    analyze_artifacts,
    build_detection_artifact_row_from_entities,
    iter_detection_parquet_files,
)
from data_designer.config.models import ModelProvider
from data_designer.config.utils.io_helpers import load_config_file
from dd_parser_compat import DDParserCompatMode, dd_parser_compat_context
from detection_strategies import (
    ExperimentalDetectionStrategy,
    NativeDetectionRuntime,
    experimental_detection_strategy_context,
)
from export_measurements import export_tables, read_measurements
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.tables import ExportFormat
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    Detect,
    Rewrite,
    infer_input_source_suffix,
    is_remote_input_source,
)
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT, DEFAULT_PROTECT_TEXT, PrivacyGoal, RiskTolerance
from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_FINAL_ENTITIES
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.engine.ndd.model_loader import parse_model_configs, validate_model_alias_references
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.measurement import MeasurementConfig, configured_measurement_session

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.benchmark")


class CaseStatus(StrEnum):
    planned = "planned"
    completed = "completed"
    error = "error"


class DDTraceMode(StrEnum):
    none = "none"
    last_message = "last_message"
    all_messages = "all_messages"


_NATIVE_ENDPOINT_ENV = "ANONYMIZER_BENCH_NATIVE_ENDPOINT"
_NATIVE_MODEL_ENV = "ANONYMIZER_BENCH_NATIVE_MODEL"
_GLINER_ENDPOINT_ENV = "ANONYMIZER_BENCH_GLINER_ENDPOINT"
_GLINER_MODEL_ENV = "ANONYMIZER_BENCH_GLINER_MODEL"


class NativeRuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime_id: str | None = None
    endpoint: str | None = None
    endpoint_env: str | None = _NATIVE_ENDPOINT_ENV
    model: str | None = None
    model_env: str | None = _NATIVE_MODEL_ENV
    provider: str = "native"
    alias: str = "native-direct"
    max_tokens: int = Field(default=4096, gt=0)
    timeout_sec: float = Field(default=180.0, gt=0)
    gliner_endpoint: str | None = None
    gliner_endpoint_env: str | None = _GLINER_ENDPOINT_ENV
    gliner_model: str | None = None
    gliner_model_env: str | None = _GLINER_MODEL_ENV
    gliner_provider: str = "gliner"
    gliner_alias: str = "gliner-direct"
    gliner_api_key_env: str = "NVIDIA_API_KEY"
    gliner_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    max_workers: int = Field(default=4, ge=1)


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
    emit_telemetry: bool = False
    experimental_detection_strategy: ExperimentalDetectionStrategy = ExperimentalDetectionStrategy.default
    native_runtime: NativeRuntimeSpec | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "ConfigSpec":
        if self.replace is None and self.rewrite is None:
            raise ValueError("config must define replace or rewrite")
        if self.replace is not None and self.rewrite is not None:
            raise ValueError("config cannot define both replace and rewrite")
        return self


class MatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload: str
    config: str
    repetitions: int = Field(default=1, ge=1)


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    model_configs: str | None = None
    model_providers: str | None = None
    artifact_path: str | None = None
    dd_parser_compat: DDParserCompatMode = DDParserCompatMode.none
    native_runtime: NativeRuntimeSpec | None = None
    case_retries: int = Field(default=0, ge=0)
    case_retry_backoff_sec: float = Field(default=0.0, ge=0.0)
    workloads: list[WorkloadSpec] = Field(min_length=1)
    configs: list[ConfigSpec] = Field(min_length=1)
    matrix: list[MatrixEntry] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_ids(self) -> "BenchmarkSpec":
        workload_ids = [workload.id for workload in self.workloads]
        config_ids = [config.id for config in self.configs]
        if duplicate_workloads := _duplicates(workload_ids):
            raise ValueError(f"duplicate workload id(s): {', '.join(duplicate_workloads)}")
        if duplicate_configs := _duplicates(config_ids):
            raise ValueError(f"duplicate config id(s): {', '.join(duplicate_configs)}")
        self._validate_matrix_references(set(workload_ids), set(config_ids))
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
        duplicate_entries = _duplicate_matrix_entries(self.matrix)
        if duplicate_entries:
            formatted = ", ".join(f"{workload}/{config}" for workload, config in duplicate_entries)
            raise ValueError(f"duplicate matrix workload/config entry(s): {formatted}; use repetitions for repeats")


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


@dataclass(frozen=True)
class _CaseRunPaths:
    raw_path: Path
    artifact_output_path: Path
    trace_path: Path | None
    task_trace_path: Path | None
    artifact_snapshot: dict[str, int] | None
    export_detection_artifacts: bool


@dataclass(frozen=True)
class _CaseExecution:
    input_data: AnonymizerInput
    trace_dataframe: pd.DataFrame | None = None


_TRACE_FINAL_ARTIFACT_STRATEGIES = {
    ExperimentalDetectionStrategy.native_candidate_validate_no_augment,
    ExperimentalDetectionStrategy.detector_native_validate_no_augment,
    ExperimentalDetectionStrategy.detector_native_validate_native_augment,
    ExperimentalDetectionStrategy.gliner_native_validate_no_augment,
    ExperimentalDetectionStrategy.gliner_native_validate_native_augment,
    ExperimentalDetectionStrategy.native_single_pass,
    ExperimentalDetectionStrategy.native_single_pass_recall,
    ExperimentalDetectionStrategy.native_single_pass_values,
    ExperimentalDetectionStrategy.native_single_pass_values_recall,
}
_NATIVE_RUNTIME_STRATEGIES = {
    ExperimentalDetectionStrategy.native_candidate_validate_no_augment,
    ExperimentalDetectionStrategy.detector_native_validate_no_augment,
    ExperimentalDetectionStrategy.detector_native_validate_native_augment,
    ExperimentalDetectionStrategy.gliner_native_validate_no_augment,
    ExperimentalDetectionStrategy.gliner_native_validate_native_augment,
    ExperimentalDetectionStrategy.native_single_pass,
    ExperimentalDetectionStrategy.native_single_pass_recall,
    ExperimentalDetectionStrategy.native_single_pass_values,
    ExperimentalDetectionStrategy.native_single_pass_values_recall,
}
_GLINER_NATIVE_RUNTIME_STRATEGIES = {
    ExperimentalDetectionStrategy.gliner_native_validate_no_augment,
    ExperimentalDetectionStrategy.gliner_native_validate_native_augment,
}

_FINAL_ARTIFACT_KEYS = {
    "final_entity_count",
    "weak_api_key_shape_count",
    "final_entity_signature_count",
    "final_entity_signature_hashes",
    "final_entity_signature_labels",
    "final_entity_signature_details",
    "weak_api_key_shape_label_counts",
    "final_label_counts",
    "final_source_counts",
}

_FINAL_ARTIFACT_PREFIXES = tuple(f"{key}." for key in _FINAL_ARTIFACT_KEYS if key != "final_entity_signature_hashes")


def load_spec(path: Path) -> BenchmarkSpec:
    if not path.exists() or path.is_dir():
        raise ValueError(f"spec path is not a file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark spec must be a YAML mapping")
    return BenchmarkSpec.model_validate(raw)


def build_cases(spec: BenchmarkSpec) -> list[BenchmarkCase]:
    matrix = spec.matrix or _cross_product_matrix(spec)
    return [
        BenchmarkCase(
            suite_id=spec.suite_id,
            workload_id=entry.workload,
            config_id=entry.config,
            repetition=repetition,
            case_id=f"{entry.workload}__{entry.config}__r{repetition:03d}",
        )
        for entry in matrix
        for repetition in range(entry.repetitions)
    ]


def _cross_product_matrix(spec: BenchmarkSpec) -> list[MatrixEntry]:
    return [
        MatrixEntry(workload=workload.id, config=config.id, repetitions=1)
        for workload in spec.workloads
        for config in spec.configs
    ]


def _duplicate_matrix_entries(matrix: list[MatrixEntry]) -> list[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    duplicates: set[tuple[str, str]] = set()
    for entry in matrix:
        key = (entry.workload, entry.config)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def prepare_output_dir(output_dir: Path, *, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output_dir}")
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise ValueError(f"output directory is not empty: {output_dir}; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)


def preflight_suite(spec: BenchmarkSpec, *, spec_path: Path) -> None:
    """Validate cheap suite inputs before any benchmark case consumes model time."""
    base_dir = spec_path.parent
    errors: list[str] = []
    parsed_models = _preflight_model_configs(spec, base_dir=base_dir, errors=errors)

    _preflight_model_providers_with_errors(spec, base_dir=base_dir, errors=errors)
    errors.extend(_preflight_workload_errors(spec, base_dir=base_dir))
    errors.extend(_preflight_config_errors(spec, parsed_models=parsed_models))
    if errors:
        raise ValueError("Benchmark preflight failed:\n- " + "\n- ".join(errors))


def _preflight_model_configs(spec: BenchmarkSpec, *, base_dir: Path, errors: list[str]) -> Any | None:
    try:
        return parse_model_configs(_resolve_config_source(spec.model_configs, base_dir))
    except Exception as exc:
        errors.append(f"model_configs invalid: {exc}")
        return None


def _preflight_model_providers_with_errors(
    spec: BenchmarkSpec,
    *,
    base_dir: Path,
    errors: list[str],
) -> None:
    try:
        _preflight_model_providers(spec, base_dir=base_dir)
    except Exception as exc:
        errors.append(f"model_providers invalid: {exc}")


def _preflight_workload_errors(spec: BenchmarkSpec, *, base_dir: Path) -> list[str]:
    errors: list[str] = []
    for workload in spec.workloads:
        try:
            _preflight_workload(workload, base_dir=base_dir)
        except Exception as exc:
            errors.append(str(exc))
    return errors


def _preflight_config_errors(spec: BenchmarkSpec, *, parsed_models: Any | None) -> list[str]:
    errors: list[str] = []
    active_config_ids = _active_config_ids(spec)
    for config in spec.configs:
        if config.id not in active_config_ids:
            continue
        try:
            anonymizer_config = build_anonymizer_config(config)
        except Exception as exc:
            errors.append(f"config '{config.id}' invalid: {exc}")
            continue
        try:
            _preflight_native_runtime(config, spec=spec)
        except Exception as exc:
            errors.append(f"config '{config.id}' native_runtime invalid: {exc}")
        if parsed_models is None:
            continue
        try:
            validate_model_alias_references(
                parsed_models.model_configs,
                parsed_models.selected_models,
                check_substitute=isinstance(anonymizer_config.replace, Substitute)
                or anonymizer_config.rewrite is not None,
                check_rewrite=anonymizer_config.rewrite is not None,
            )
        except ValueError as exc:
            errors.append(f"config '{config.id}' model aliases invalid: {exc}")
    return errors


def _active_config_ids(spec: BenchmarkSpec) -> set[str]:
    if spec.matrix is None:
        return {config.id for config in spec.configs}
    return {entry.config for entry in spec.matrix}


def _preflight_native_runtime(config: ConfigSpec, *, spec: BenchmarkSpec) -> None:
    strategy = config.experimental_detection_strategy
    if strategy not in _NATIVE_RUNTIME_STRATEGIES:
        return
    runtime = _resolve_native_runtime_spec(spec, config)
    if not runtime.runtime_id:
        raise ValueError("native strategies require native_runtime.runtime_id")
    if not runtime.endpoint or not runtime.model:
        raise ValueError("native strategies require native_runtime.endpoint and native_runtime.model")
    if strategy in _GLINER_NATIVE_RUNTIME_STRATEGIES:
        if not runtime.gliner_endpoint or not runtime.gliner_model:
            raise ValueError("GLiNER-native strategies require native_runtime.gliner_endpoint and gliner_model")
        if not os.environ.get(runtime.gliner_api_key_env):
            raise ValueError(f"{runtime.gliner_api_key_env} is not set for GLiNER-native strategy")


def _resolve_native_runtime_spec(spec: BenchmarkSpec, config: ConfigSpec) -> NativeRuntimeSpec:
    runtime = spec.native_runtime or NativeRuntimeSpec()
    if config.native_runtime is not None:
        runtime = runtime.model_copy(update=config.native_runtime.model_dump(exclude_unset=True))
    return runtime.model_copy(
        update={
            "endpoint": _resolve_runtime_value(runtime.endpoint, runtime.endpoint_env),
            "model": _resolve_runtime_value(runtime.model, runtime.model_env),
            "gliner_endpoint": _resolve_runtime_value(runtime.gliner_endpoint, runtime.gliner_endpoint_env),
            "gliner_model": _resolve_runtime_value(runtime.gliner_model, runtime.gliner_model_env),
        }
    )


def _resolve_runtime_value(explicit: str | None, env_var: str | None) -> str | None:
    if explicit:
        return explicit
    return os.environ.get(env_var) if env_var else None


def _native_detection_runtime(spec: BenchmarkSpec, config: ConfigSpec) -> NativeDetectionRuntime:
    runtime = _resolve_native_runtime_spec(spec, config)
    if not runtime.runtime_id:
        raise ValueError("native strategies require native_runtime.runtime_id")
    if not runtime.endpoint or not runtime.model:
        raise ValueError("native strategies require native_runtime.endpoint and native_runtime.model")
    return NativeDetectionRuntime(
        endpoint=runtime.endpoint,
        model=runtime.model,
        provider=runtime.provider,
        alias=runtime.alias,
        max_tokens=runtime.max_tokens,
        timeout_sec=runtime.timeout_sec,
        gliner_endpoint=runtime.gliner_endpoint,
        gliner_model=runtime.gliner_model or "",
        gliner_provider=runtime.gliner_provider,
        gliner_alias=runtime.gliner_alias,
        gliner_api_key_env=runtime.gliner_api_key_env,
        gliner_threshold=runtime.gliner_threshold,
        max_workers=runtime.max_workers,
    )


def _preflight_model_providers(spec: BenchmarkSpec, *, base_dir: Path) -> None:
    raw = _resolve_config_source(spec.model_providers, base_dir)
    if raw is None:
        return
    config_source: str | Path = raw
    if isinstance(raw, str) and "\n" not in raw:
        candidate = Path(raw.strip()).expanduser()
        if candidate.suffix in (".yaml", ".yml"):
            if not candidate.is_file():
                raise FileNotFoundError(f"Providers config file not found: {candidate}")
            config_source = candidate
    config_dict = yaml.safe_load(raw) if isinstance(raw, str) and "\n" in raw else load_config_file(config_source)
    raw_providers = config_dict.get("providers") if isinstance(config_dict, dict) else None
    if not isinstance(raw_providers, list):
        raise ValueError("model_providers YAML must contain a top-level 'providers' list.")
    for provider in raw_providers:
        ModelProvider.model_validate(provider)


def _preflight_workload(workload: WorkloadSpec, *, base_dir: Path) -> None:
    resolved_source = _resolve_input_source(workload.source, base_dir)
    if _workload_has_row_slice(workload) and not _is_local_input_source(str(resolved_source)):
        raise ValueError(f"workload '{workload.id}' row slicing requires a local workload source")
    input_data = AnonymizerInput(
        source=str(resolved_source),
        text_column=workload.text_column,
        id_column=workload.id_column,
        data_summary=workload.data_summary,
    )
    columns = _input_columns(input_data.source)
    if columns is None:
        return
    if workload.text_column not in columns:
        raise ValueError(
            f"workload '{workload.id}' text_column '{workload.text_column}' not found in {input_data.source}; "
            f"available columns: {sorted(columns)}"
        )
    if workload.id_column is not None and workload.id_column not in columns:
        raise ValueError(
            f"workload '{workload.id}' id_column '{workload.id_column}' not found in {input_data.source}; "
            f"available columns: {sorted(columns)}"
        )


def _input_columns(source: str) -> set[str] | None:
    suffix = infer_input_source_suffix(source)
    if suffix not in SUPPORTED_IO_FORMATS:
        supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
        raise ValueError(f"Unsupported input format: {suffix}. Use {supported_formats}.")
    if is_remote_input_source(source):
        return None
    if suffix == ".csv":
        return set(pd.read_csv(source, nrows=0).columns)
    return set(pq.ParquetFile(source).schema_arrow.names)


def run_suite(
    spec: BenchmarkSpec,
    *,
    spec_path: Path,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
    dd_trace: DDTraceMode,
    trace_dir: Path | None,
    dd_task_trace: bool = False,
    task_trace_dir: Path | None = None,
) -> BenchmarkResult:
    contexts = _build_contexts(
        spec,
        spec_path=spec_path,
        output_dir=output_dir,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
        dd_task_trace=dd_task_trace,
        task_trace_dir=task_trace_dir,
    )
    anonymizer = Anonymizer(**contexts["anonymizer_kwargs"])
    cases = _run_cases(spec, contexts=contexts, anonymizer=anonymizer, fail_fast=fail_fast, export=export)
    measurement_path = combine_measurements(cases, output_dir / "measurements.jsonl")
    should_export = _should_export_measurements(export=export, measurement_path=measurement_path)
    table_dir = _export_suite_tables(measurement_path, output_dir=output_dir, should_export=should_export)
    artifact_analysis_path = _combine_suite_detection_artifacts(
        cases, output_dir=output_dir, should_export=should_export
    )
    result = _benchmark_result(
        spec,
        output_dir=output_dir,
        measurement_path=measurement_path,
        table_dir=table_dir,
        artifact_analysis_path=artifact_analysis_path,
        cases=cases,
    )
    write_summary(result)
    return result


def _run_cases(
    spec: BenchmarkSpec,
    *,
    contexts: dict[str, Any],
    anonymizer: Anonymizer,
    fail_fast: bool,
    export: bool,
) -> list[BenchmarkCase]:
    return [
        _run_case(
            case,
            spec,
            contexts=contexts,
            anonymizer=anonymizer,
            fail_fast=fail_fast,
            export_detection_artifacts=export,
        )
        for case in build_cases(spec)
    ]


def _should_export_measurements(*, export: bool, measurement_path: Path) -> bool:
    return export and measurement_path.stat().st_size > 0


def _export_suite_tables(measurement_path: Path, *, output_dir: Path, should_export: bool) -> Path | None:
    if not should_export:
        return None
    return export_measurement_tables(measurement_path, output_dir / "tables")


def _combine_suite_detection_artifacts(
    cases: list[BenchmarkCase],
    *,
    output_dir: Path,
    should_export: bool,
) -> Path | None:
    if not should_export:
        return None
    return combine_detection_artifact_analysis(cases, output_dir / "detection-artifacts.jsonl")


def _benchmark_result(
    spec: BenchmarkSpec,
    *,
    output_dir: Path,
    measurement_path: Path,
    table_dir: Path | None,
    artifact_analysis_path: Path | None,
    cases: list[BenchmarkCase],
) -> BenchmarkResult:
    return BenchmarkResult(
        suite_id=spec.suite_id,
        output_dir=str(output_dir),
        measurement_path=str(measurement_path),
        summary_path=str(output_dir / "summary.json"),
        table_dir=str(table_dir) if table_dir is not None else None,
        detection_artifact_analysis_path=str(artifact_analysis_path) if artifact_analysis_path is not None else None,
        cases=cases,
    )


def _build_contexts(
    spec: BenchmarkSpec,
    *,
    spec_path: Path,
    output_dir: Path,
    dd_trace: DDTraceMode,
    trace_dir: Path | None,
    dd_task_trace: bool = False,
    task_trace_dir: Path | None = None,
) -> dict[str, Any]:
    base_dir = spec_path.parent
    artifact_path = _resolve_optional_path(spec.artifact_path, base_dir) or output_dir / "artifacts"
    return {
        "base_dir": base_dir,
        "workloads": {workload.id: workload for workload in spec.workloads},
        "configs": {config.id: config for config in spec.configs},
        "raw_dir": output_dir / "raw",
        "dd_trace": dd_trace,
        "trace_dir": trace_dir or output_dir / "traces",
        "dd_task_trace": dd_task_trace,
        "task_trace_dir": task_trace_dir or output_dir / "task-traces",
        "dd_parser_compat": spec.dd_parser_compat,
        "artifact_path": artifact_path,
        "anonymizer_kwargs": {
            "model_configs": _resolve_config_source(spec.model_configs, base_dir),
            "model_providers": _resolve_config_source(spec.model_providers, base_dir),
            "artifact_path": artifact_path,
        },
    }


def _run_case(
    case: BenchmarkCase,
    spec: BenchmarkSpec,
    *,
    contexts: dict[str, Any],
    anonymizer: Anonymizer,
    fail_fast: bool,
    export_detection_artifacts: bool,
) -> BenchmarkCase:
    started = time.perf_counter()
    attempt_errors: list[str] = []
    max_attempts = 1 if fail_fast else spec.case_retries + 1
    for attempt_number in range(1, max_attempts + 1):
        paths = _case_run_paths(case, contexts=contexts, export_detection_artifacts=export_detection_artifacts)
        try:
            return _run_case_success(
                case,
                spec,
                contexts=contexts,
                anonymizer=anonymizer,
                paths=paths,
                started=started,
                attempt_count=attempt_number,
                attempt_errors=attempt_errors,
            )
        except Exception as exc:
            if fail_fast:
                raise
            attempt_errors.append(str(exc))
            if attempt_number >= max_attempts:
                return _run_case_error(
                    case,
                    contexts=contexts,
                    paths=paths,
                    started=started,
                    error=exc,
                    attempt_count=attempt_number,
                    attempt_errors=attempt_errors,
                )
            _sleep_before_case_retry(spec, case=case, attempt_number=attempt_number, error=exc)

    raise RuntimeError("unreachable benchmark retry state")


def _sleep_before_case_retry(
    spec: BenchmarkSpec,
    *,
    case: BenchmarkCase,
    attempt_number: int,
    error: Exception,
) -> None:
    logger.warning(
        "case %s attempt %d failed; retrying after %.2fs: %s",
        case.case_id,
        attempt_number,
        spec.case_retry_backoff_sec,
        error,
    )
    if spec.case_retry_backoff_sec > 0:
        time.sleep(spec.case_retry_backoff_sec)


def _case_run_paths(
    case: BenchmarkCase,
    *,
    contexts: dict[str, Any],
    export_detection_artifacts: bool,
) -> _CaseRunPaths:
    return _CaseRunPaths(
        raw_path=contexts["raw_dir"] / f"{case.case_id}.jsonl",
        artifact_output_path=contexts["raw_dir"] / f"{case.case_id}.detection-artifacts.jsonl",
        trace_path=_case_trace_path(case, contexts=contexts),
        task_trace_path=_case_task_trace_path(case, contexts=contexts),
        artifact_snapshot=snapshot_detection_artifacts(contexts["artifact_path"])
        if export_detection_artifacts
        else None,
        export_detection_artifacts=export_detection_artifacts,
    )


def _run_case_success(
    case: BenchmarkCase,
    spec: BenchmarkSpec,
    *,
    contexts: dict[str, Any],
    anonymizer: Anonymizer,
    paths: _CaseRunPaths,
    started: float,
    attempt_count: int,
    attempt_errors: list[str],
) -> BenchmarkCase:
    workload = _get_item(contexts["workloads"], case.workload_id, "workload")
    config = _get_item(contexts["configs"], case.config_id, "config")
    execution = _execute_case(
        anonymizer,
        workload,
        config,
        raw_path=paths.raw_path,
        trace_path=paths.trace_path,
        task_trace_path=paths.task_trace_path,
        case=case,
        spec=spec,
        base_dir=contexts["base_dir"],
        dd_trace=contexts["dd_trace"],
        dd_parser_compat=contexts["dd_parser_compat"],
    )
    detection_artifact_path = _case_detection_artifact_path(
        contexts,
        paths,
        case=case,
        config=config,
        execution=execution,
    )
    return _case_with_result(
        case,
        status=CaseStatus.completed,
        started=started,
        raw_path=paths.raw_path,
        detection_artifact_path=detection_artifact_path,
        trace_path=paths.trace_path,
        task_trace_path=paths.task_trace_path,
        attempt_count=attempt_count,
        attempt_errors=attempt_errors,
    )


def _run_case_error(
    case: BenchmarkCase,
    *,
    contexts: dict[str, Any],
    paths: _CaseRunPaths,
    started: float,
    error: Exception,
    attempt_count: int,
    attempt_errors: list[str],
) -> BenchmarkCase:
    detection_artifact_path = _export_case_detection_artifacts_if_requested(
        contexts,
        paths.artifact_output_path,
        case=case,
        artifact_snapshot=paths.artifact_snapshot,
    )
    return _case_with_result(
        case,
        status=CaseStatus.error,
        started=started,
        raw_path=paths.raw_path,
        detection_artifact_path=detection_artifact_path,
        trace_path=paths.trace_path,
        task_trace_path=paths.task_trace_path,
        error=str(error),
        attempt_count=attempt_count,
        attempt_errors=attempt_errors,
    )


def _case_detection_artifact_path(
    contexts: dict[str, Any],
    paths: _CaseRunPaths,
    *,
    case: BenchmarkCase,
    config: ConfigSpec,
    execution: _CaseExecution,
) -> Path | None:
    detection_artifact_path = _export_case_detection_artifacts_if_requested(
        contexts,
        paths.artifact_output_path,
        case=case,
        artifact_snapshot=paths.artifact_snapshot,
    )
    if paths.export_detection_artifacts:
        detection_artifact_path = _trace_final_artifact_path_if_requested(
            config,
            detection_artifact_path,
            paths.artifact_output_path,
            case=case,
            trace_dataframe=execution.trace_dataframe,
        )
    if detection_artifact_path is not None or paths.artifact_snapshot is None:
        return detection_artifact_path
    return None


def _trace_final_artifact_path_if_requested(
    config: ConfigSpec,
    detection_artifact_path: Path | None,
    output_path: Path,
    *,
    case: BenchmarkCase,
    trace_dataframe: pd.DataFrame | None,
) -> Path | None:
    if config.experimental_detection_strategy not in _TRACE_FINAL_ARTIFACT_STRATEGIES:
        return detection_artifact_path
    if trace_dataframe is None:
        return detection_artifact_path
    return patch_case_detection_artifacts_from_trace_dataframe(
        detection_artifact_path or output_path,
        trace_dataframe,
        case=case,
    )


def patch_case_detection_artifacts_from_trace_dataframe(
    output_path: Path,
    trace_dataframe: pd.DataFrame,
    *,
    case: BenchmarkCase | None = None,
) -> Path | None:
    final_rows = _final_entity_artifact_rows_from_trace_dataframe(trace_dataframe)
    if not final_rows:
        return None
    rows = _read_detection_artifact_payloads(output_path) if output_path.exists() else []
    patched = _merge_final_entity_artifact_rows(rows, final_rows)
    if case is not None:
        patched = [_with_case_metadata(row, case=case) for row in patched]
    write_detection_artifact_payloads(patched, output_path)
    return output_path


def _final_entity_artifact_rows_from_trace_dataframe(trace_dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    entities_column = _trace_final_entities_column(trace_dataframe)
    if entities_column is None:
        return []
    return [
        _final_entity_artifact_row(raw_entities, row_index=row_index)
        for row_index, raw_entities in enumerate(trace_dataframe[entities_column])
    ]


def _trace_final_entities_column(trace_dataframe: pd.DataFrame) -> str | None:
    if COL_FINAL_ENTITIES in trace_dataframe.columns:
        return COL_FINAL_ENTITIES
    if COL_DETECTED_ENTITIES in trace_dataframe.columns:
        return COL_DETECTED_ENTITIES
    return None


def _final_entity_artifact_row(raw_entities: object, *, row_index: int) -> dict[str, Any]:
    entities = EntitiesSchema.from_raw(raw_entities).entities
    return build_detection_artifact_row_from_entities(
        workflow_name="entity-detection-final-trace",
        batch_file="trace_dataframe",
        row_index=row_index,
        seed_entities=[],
        seed_validation_candidate_count=0,
        merged_validation_candidate_count=0,
        augmented_entities=[],
        final_entities=entities,
    ).model_dump()


def _read_detection_artifact_payloads(output_path: Path) -> list[dict[str, Any]]:
    with output_path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _merge_final_entity_artifact_rows(
    rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    patched = [_patch_final_entity_artifact_row(row, final_row) for row, final_row in zip(rows, final_rows)]
    return patched + rows[len(final_rows) :] + final_rows[len(rows) :]


def _patch_final_entity_artifact_row(row: dict[str, Any], final_row: dict[str, Any]) -> dict[str, Any]:
    clean_row = _without_final_entity_artifact_fields(row)
    return {**clean_row, **_final_entity_artifact_fields(final_row)}


def _without_final_entity_artifact_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key not in _FINAL_ARTIFACT_KEYS and not key.startswith(_FINAL_ARTIFACT_PREFIXES)
    }


def _final_entity_artifact_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in _FINAL_ARTIFACT_KEYS}


def _case_with_result(
    case: BenchmarkCase,
    *,
    status: CaseStatus,
    started: float,
    raw_path: Path,
    detection_artifact_path: Path | None,
    trace_path: Path | None,
    task_trace_path: Path | None,
    attempt_count: int,
    attempt_errors: list[str],
    error: str | None = None,
) -> BenchmarkCase:
    return case.model_copy(
        update={
            "status": status,
            "elapsed_sec": time.perf_counter() - started,
            "measurement_path": str(raw_path),
            "detection_artifact_path": (str(detection_artifact_path) if detection_artifact_path is not None else None),
            "trace_path": str(trace_path) if trace_path is not None else None,
            "task_trace_path": str(task_trace_path) if task_trace_path is not None else None,
            "error": error,
            "attempt_count": attempt_count,
            "attempt_errors": list(attempt_errors),
        }
    )


def _export_case_detection_artifacts_if_requested(
    contexts: dict[str, Any],
    output_path: Path,
    *,
    case: BenchmarkCase,
    artifact_snapshot: dict[str, int] | None,
) -> Path | None:
    if artifact_snapshot is None:
        return None
    return export_case_detection_artifact_analysis(
        contexts["artifact_path"],
        output_path,
        case=case,
        artifact_snapshot=artifact_snapshot,
    )


def _case_trace_path(case: BenchmarkCase, *, contexts: dict[str, Any]) -> Path | None:
    if contexts["dd_trace"] == DDTraceMode.none:
        return None
    return contexts["trace_dir"] / f"{case.case_id}.jsonl"


def _case_task_trace_path(case: BenchmarkCase, *, contexts: dict[str, Any]) -> Path | None:
    if not contexts["dd_task_trace"]:
        return None
    return contexts["task_trace_dir"] / f"{case.case_id}.jsonl"


def _execute_case(
    anonymizer: Anonymizer,
    workload: WorkloadSpec,
    config: ConfigSpec,
    *,
    raw_path: Path,
    trace_path: Path | None,
    task_trace_path: Path | None,
    case: BenchmarkCase,
    spec: BenchmarkSpec,
    base_dir: Path,
    dd_trace: DDTraceMode,
    dd_parser_compat: DDParserCompatMode,
) -> _CaseExecution:
    anonymizer_config = build_anonymizer_config(config)
    input_data = build_input(
        workload,
        base_dir,
        slice_dir=raw_path.parent / "inputs",
        case_id=case.case_id,
    )
    measurement = MeasurementConfig(
        output_path=raw_path,
        run_id=case.case_id,
        run_tags=_run_tags(case, spec),
        streaming=True,
        keep_records=False,
        dd_trace=dd_trace.value,
        dd_trace_path=trace_path,
        dd_task_trace_path=task_trace_path,
        fail_on_write_error=True,
    )
    with configured_measurement_session(measurement):
        with dd_parser_compat_context(dd_parser_compat):
            detection_context_kwargs: dict[str, Any] = {}
            if config.experimental_detection_strategy in _NATIVE_RUNTIME_STRATEGIES:
                detection_context_kwargs["native_runtime"] = _native_detection_runtime(spec, config)
            with experimental_detection_strategy_context(
                config.experimental_detection_strategy,
                **detection_context_kwargs,
            ):
                result = anonymizer.run(
                    config=anonymizer_config,
                    data=input_data,
                )
    return _CaseExecution(input_data=input_data, trace_dataframe=getattr(result, "trace_dataframe", None))


def build_input(
    workload: WorkloadSpec,
    base_dir: Path,
    *,
    slice_dir: Path | None = None,
    case_id: str | None = None,
) -> AnonymizerInput:
    resolved_source = _resolve_input_source(workload.source, base_dir)
    source = (
        _materialize_sliced_source(workload, resolved_source, slice_dir=slice_dir, case_id=case_id)
        if _workload_has_row_slice(workload)
        else resolved_source
    )
    return AnonymizerInput(
        source=str(source),
        text_column=workload.text_column,
        id_column=workload.id_column,
        data_summary=workload.data_summary,
    )


def _workload_has_row_slice(workload: WorkloadSpec) -> bool:
    return workload.row_limit is not None or workload.row_offset > 0


def _is_local_input_source(source: str) -> bool:
    return "://" not in source


def _materialize_sliced_source(
    workload: WorkloadSpec,
    source: str | Path,
    *,
    slice_dir: Path | None,
    case_id: str | None,
) -> Path:
    if not _is_local_input_source(str(source)):
        raise ValueError(f"workload '{workload.id}' row slicing requires a local workload source")
    if slice_dir is None or case_id is None:
        raise ValueError("row slicing requires slice_dir and case_id")
    source_path = Path(source)
    suffix = infer_input_source_suffix(str(source_path))
    dataframe = _read_local_input_dataframe(source_path, suffix=suffix)
    sliced = dataframe.iloc[_slice_bounds(workload)]
    slice_dir.mkdir(parents=True, exist_ok=True)
    destination = slice_dir / f"{_safe_case_filename(case_id)}{suffix}"
    _write_local_input_dataframe(sliced, destination, suffix=suffix)
    return destination


def _slice_bounds(workload: WorkloadSpec) -> slice:
    start = workload.row_offset
    stop = start + workload.row_limit if workload.row_limit is not None else None
    return slice(start, stop)


def _read_local_input_dataframe(source: Path, *, suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
    raise ValueError(f"Unsupported input format: {suffix}. Use {supported_formats}.")


def _write_local_input_dataframe(dataframe: pd.DataFrame, destination: Path, *, suffix: str) -> None:
    if suffix == ".csv":
        dataframe.to_csv(destination, index=False)
        return
    if suffix == ".parquet":
        dataframe.to_parquet(destination, index=False)
        return
    supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
    raise ValueError(f"Unsupported input format: {suffix}. Use {supported_formats}.")


def _safe_case_filename(case_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in case_id)


def build_anonymizer_config(config: ConfigSpec) -> AnonymizerConfig:
    detect = Detect.model_validate(config.detect)
    if config.replace is not None:
        return AnonymizerConfig(
            detect=detect, replace=build_replace(config.replace), emit_telemetry=config.emit_telemetry
        )
    return AnonymizerConfig(detect=detect, rewrite=build_rewrite(config.rewrite), emit_telemetry=config.emit_telemetry)


def build_replace(raw: str | ReplaceSpec) -> Redact | Hash | Annotate | Substitute:
    spec = ReplaceSpec(strategy=ReplaceKind(raw)) if isinstance(raw, str) else raw
    if spec.strategy == ReplaceKind.redact:
        return Redact(**_present({"format_template": spec.format_template, "normalize_label": spec.normalize_label}))
    if spec.strategy == ReplaceKind.hash:
        return Hash(
            **_present(
                {
                    "format_template": spec.format_template,
                    "algorithm": spec.algorithm,
                    "digest_length": spec.digest_length,
                }
            )
        )
    if spec.strategy == ReplaceKind.annotate:
        return Annotate(**_present({"format_template": spec.format_template}))
    return Substitute(**_present({"instructions": spec.instructions}))


def build_rewrite(spec: RewriteSpec | None) -> Rewrite:
    if spec is None:
        raise ValueError("rewrite config is missing")
    privacy_goal = _privacy_goal(spec)
    return Rewrite(
        privacy_goal=privacy_goal,
        instructions=spec.instructions,
        risk_tolerance=spec.risk_tolerance,
        max_repair_iterations=spec.max_repair_iterations,
        strict_entity_protection=spec.strict_entity_protection,
    )


def _privacy_goal(spec: RewriteSpec) -> PrivacyGoal | None:
    if spec.protect is None and spec.preserve is None:
        return None
    return PrivacyGoal(
        protect=spec.protect or DEFAULT_PROTECT_TEXT,
        preserve=spec.preserve or DEFAULT_PRESERVE_TEXT,
    )


def combine_measurements(cases: list[BenchmarkCase], destination: Path) -> Path:
    with destination.open("w", encoding="utf-8") as output:
        for case in cases:
            if case.measurement_path is None:
                continue
            source = Path(case.measurement_path)
            if source.exists():
                output.write(source.read_text(encoding="utf-8"))
    return destination


def combine_detection_artifact_analysis(cases: list[BenchmarkCase], destination: Path) -> Path | None:
    chunks: list[str] = []
    for case in cases:
        if case.detection_artifact_path is None:
            continue
        source = Path(case.detection_artifact_path)
        if source.exists():
            chunks.append(_jsonl_chunk(source.read_text(encoding="utf-8")))
    if not chunks:
        return None
    destination.write_text("".join(chunks), encoding="utf-8")
    return destination


def _jsonl_chunk(text: str) -> str:
    if not text or text.endswith("\n"):
        return text
    return text + "\n"


def export_measurement_tables(measurement_path: Path, table_dir: Path) -> Path:
    dataframe = read_measurements(measurement_path)
    export_tables(
        dataframe, input_path=measurement_path, output_dir=table_dir, export_format=ExportFormat.parquet, overwrite=True
    )
    return table_dir


def snapshot_detection_artifacts(artifact_path: Path) -> dict[str, int]:
    if not artifact_path.exists():
        return {}
    return {
        str(parquet_file.relative_to(artifact_path)): parquet_file.stat().st_mtime_ns
        for parquet_file in iter_detection_parquet_files(artifact_path)
    }


def changed_detection_artifact_files(artifact_path: Path, snapshot: dict[str, int]) -> list[Path]:
    if not artifact_path.exists():
        return []
    changed: list[Path] = []
    for parquet_file in iter_detection_parquet_files(artifact_path):
        key = str(parquet_file.relative_to(artifact_path))
        if snapshot.get(key) != parquet_file.stat().st_mtime_ns:
            changed.append(parquet_file)
    return changed


def export_detection_artifact_analysis(
    artifact_path: Path,
    output_path: Path,
    *,
    artifact_snapshot: dict[str, int] | None = None,
) -> Path | None:
    if not artifact_path.exists():
        return None
    parquet_files = (
        changed_detection_artifact_files(artifact_path, artifact_snapshot) if artifact_snapshot is not None else None
    )
    analysis = analyze_artifacts(artifact_path, parquet_files=parquet_files)
    if not analysis.rows:
        return None
    write_detection_artifact_payloads([row.model_dump() for row in analysis.rows], output_path)
    return output_path


def export_case_detection_artifact_analysis(
    artifact_path: Path,
    output_path: Path,
    *,
    case: BenchmarkCase,
    artifact_snapshot: dict[str, int],
) -> Path | None:
    if not artifact_path.exists():
        return None
    parquet_files = changed_detection_artifact_files(artifact_path, artifact_snapshot)
    analysis = analyze_artifacts(artifact_path, parquet_files=parquet_files)
    if not analysis.rows:
        return None
    write_detection_artifact_payloads(
        [_with_case_metadata(row.model_dump(), case=case) for row in analysis.rows],
        output_path,
    )
    return output_path


def _with_case_metadata(row: dict[str, Any], *, case: BenchmarkCase) -> dict[str, Any]:
    return {
        "suite_id": case.suite_id,
        "workload_id": case.workload_id,
        "config_id": case.config_id,
        "repetition": case.repetition,
        "case_id": case.case_id,
        "run_id": case.case_id,
        **row,
    }


def write_detection_artifact_payloads(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(rows, sep=".").to_json(output_path, orient="records", lines=True)


def write_summary(result: BenchmarkResult) -> None:
    Path(result.summary_path).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def render_result(result: BenchmarkResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    completed = sum(case.status == CaseStatus.completed for case in result.cases)
    errored = sum(case.status == CaseStatus.error for case in result.cases)
    planned = sum(case.status == CaseStatus.planned for case in result.cases)
    if planned and completed == 0 and errored == 0:
        return f"Planned {planned} case(s); output={result.output_dir}"
    return f"Ran {completed}/{len(result.cases)} case(s); errors={errored}; output={result.output_dir}"


def _run_tags(case: BenchmarkCase, spec: BenchmarkSpec) -> dict[str, Any]:
    config = next(item for item in spec.configs if item.id == case.config_id)
    tags = {
        "suite_id": spec.suite_id,
        "workload_id": case.workload_id,
        "config_id": case.config_id,
        "repetition": case.repetition,
        "case_id": case.case_id,
        "experimental_detection_strategy": config.experimental_detection_strategy.value,
        "dd_parser_compat": spec.dd_parser_compat.value,
    }
    if config.experimental_detection_strategy in _NATIVE_RUNTIME_STRATEGIES:
        tags.update(_native_runtime_tags(_resolve_native_runtime_spec(spec, config)))
    return tags


def _native_runtime_tags(runtime: NativeRuntimeSpec) -> dict[str, Any]:
    return _present(
        {
            "native_runtime_id": runtime.runtime_id,
            "native_endpoint_env": runtime.endpoint_env,
            "native_model": runtime.model,
            "native_model_env": runtime.model_env,
            "native_provider": runtime.provider,
            "native_alias": runtime.alias,
            "native_max_tokens": runtime.max_tokens,
            "native_timeout_sec": runtime.timeout_sec,
            "native_max_workers": runtime.max_workers,
            "gliner_endpoint_env": runtime.gliner_endpoint_env,
            "gliner_model": runtime.gliner_model,
            "gliner_model_env": runtime.gliner_model_env,
            "gliner_provider": runtime.gliner_provider,
            "gliner_alias": runtime.gliner_alias,
            "gliner_api_key_env": runtime.gliner_api_key_env,
            "gliner_threshold": runtime.gliner_threshold,
        }
    )


def _present(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _get_item(items: dict[str, Any], item_id: str, item_type: str) -> Any:
    if item_id not in items:
        raise ValueError(f"unknown {item_type}: {item_id}")
    return items[item_id]


def _resolve_input_source(source: str, base_dir: Path) -> str | Path:
    if "://" in source:
        return source
    return _resolve_path(source, base_dir)


def _resolve_optional_path(raw: str | None, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    return _resolve_path(raw, base_dir)


def _resolve_config_source(raw: str | None, base_dir: Path) -> str | None:
    if raw is None or "\n" in raw:
        return raw
    candidate = Path(raw).expanduser()
    if candidate.suffix in {".yaml", ".yml"}:
        return str(_resolve_path(raw, base_dir))
    return raw


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else base_dir / path


def dry_run_result(
    spec: BenchmarkSpec,
    *,
    output_dir: Path,
    export: bool,
    dd_trace: DDTraceMode,
    trace_dir: Path | None,
    dd_task_trace: bool = False,
    task_trace_dir: Path | None = None,
) -> BenchmarkResult:
    cases = build_cases(spec)
    if dd_trace != DDTraceMode.none:
        resolved_trace_dir = trace_dir or output_dir / "traces"
        cases = [
            case.model_copy(update={"trace_path": str(resolved_trace_dir / f"{case.case_id}.jsonl")}) for case in cases
        ]
    if dd_task_trace:
        resolved_task_trace_dir = task_trace_dir or output_dir / "task-traces"
        cases = [
            case.model_copy(update={"task_trace_path": str(resolved_task_trace_dir / f"{case.case_id}.jsonl")})
            for case in cases
        ]
    return BenchmarkResult(
        suite_id=spec.suite_id,
        output_dir=str(output_dir),
        measurement_path=str(output_dir / "measurements.jsonl"),
        summary_path=str(output_dir / "summary.json"),
        table_dir=str(output_dir / "tables") if export else None,
        detection_artifact_analysis_path=str(output_dir / "detection-artifacts.jsonl") if export else None,
        cases=cases,
    )


@app.default
def main(
    spec: Path,
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    overwrite: Annotated[bool, cyclopts.Parameter("--overwrite")] = False,
    dry_run: Annotated[bool, cyclopts.Parameter("--dry-run")] = False,
    export: Annotated[bool, cyclopts.Parameter("--export")] = True,
    fail_fast: Annotated[bool, cyclopts.Parameter("--fail-fast")] = False,
    dd_trace: Annotated[DDTraceMode, cyclopts.Parameter("--dd-trace")] = DDTraceMode.none,
    trace_dir: Annotated[Path | None, cyclopts.Parameter("--trace-dir")] = None,
    dd_task_trace: Annotated[bool, cyclopts.Parameter("--dd-task-trace")] = False,
    task_trace_dir: Annotated[Path | None, cyclopts.Parameter("--task-trace-dir")] = None,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = run_or_plan(
            spec,
            output=output,
            overwrite=overwrite,
            dry_run=dry_run,
            export=export,
            fail_fast=fail_fast,
            dd_trace=dd_trace,
            trace_dir=trace_dir,
            dd_task_trace=dd_task_trace,
            task_trace_dir=task_trace_dir,
        )
    except (ValueError, ValidationError) as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")
    if any(case.status == CaseStatus.error for case in result.cases):
        raise SystemExit(1)


def run_or_plan(
    spec_path: Path,
    *,
    output: Path | None,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    dd_trace: DDTraceMode = DDTraceMode.none,
    trace_dir: Path | None = None,
    dd_task_trace: bool = False,
    task_trace_dir: Path | None = None,
) -> BenchmarkResult:
    benchmark_spec = load_spec(spec_path)
    output_dir = output or Path("benchmark-runs") / benchmark_spec.suite_id
    if trace_dir is not None and dd_trace == DDTraceMode.none:
        raise ValueError("--trace-dir requires --dd-trace")
    if task_trace_dir is not None and not dd_task_trace:
        raise ValueError("--task-trace-dir requires --dd-task-trace")
    preflight_suite(benchmark_spec, spec_path=spec_path)
    if dry_run:
        return dry_run_result(
            benchmark_spec,
            output_dir=output_dir,
            export=export,
            dd_trace=dd_trace,
            trace_dir=trace_dir,
            dd_task_trace=dd_task_trace,
            task_trace_dir=task_trace_dir,
        )
    prepare_output_dir(output_dir, overwrite=overwrite, dry_run=dry_run)
    return run_suite(
        benchmark_spec,
        spec_path=spec_path,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
        dd_task_trace=dd_task_trace,
        task_trace_dir=task_trace_dir,
    )


if __name__ == "__main__":
    app()
