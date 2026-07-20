#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Anonymizer benchmark suites and export measurement tables.

Usage:
    uv run python tools/measurement/run_benchmarks.py suite.yaml --output benchmark-runs/suite
    uv run python tools/measurement/run_benchmarks.py suite.yaml --dry-run --json
"""

import logging
import platform
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import pandas as pd
from analyze_detection_artifacts import (
    analyze_artifacts,
    iter_detection_parquet_files,
)
from export_measurements import export_tables, read_measurements
from measurement_tools.benchmark_inputs import (
    build_anonymizer_config,
    build_input,
    is_local_input_source,
    materialize_sliced_source,
    present,
    privacy_goal,
    read_local_input_dataframe,
    resolve_config_source,
    resolve_input_source,
    resolve_optional_path,
    resolve_path,
    safe_case_filename,
    slice_bounds,
    workload_has_row_slice,
    write_local_input_dataframe,
)
from measurement_tools.benchmark_inputs import (
    build_replace as build_replace,
)
from measurement_tools.benchmark_inputs import (
    build_rewrite as build_rewrite,
)
from measurement_tools.benchmark_models import (
    RESERVED_RUN_TAG_KEYS as RESERVED_RUN_TAG_KEYS,
)
from measurement_tools.benchmark_models import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSpec,
    CaseRunPaths,
    CaseStatus,
    ConfigSpec,
    DDTraceMode,
    ReplaceSpec,
    RewriteSpec,
    WorkloadSpec,
    duplicate_matrix_entries,
    duplicates,
)
from measurement_tools.benchmark_models import (
    MatrixEntry as MatrixEntry,
)
from measurement_tools.benchmark_models import (
    ReplaceKind as ReplaceKind,
)
from measurement_tools.benchmark_planning import dry_run_result as _canonical_dry_run_result
from measurement_tools.benchmark_planning import plan_suite as _canonical_plan_suite
from measurement_tools.benchmark_specs import (
    active_config_ids,
    build_cases,
    cross_product_matrix,
    input_columns,
    load_spec,
    preflight_config_errors,
    preflight_model_configs,
    preflight_model_providers,
    preflight_model_providers_with_errors,
    preflight_suite,
    preflight_workload,
    preflight_workload_errors,
    prepare_output_dir,
)
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input, summarize_validation_error
from measurement_tools.execution import benchmark_execution_metadata, stable_metadata_hash
from measurement_tools.tables import ExportFormat
from measurement_tools.wandb_setup import (
    WANDB_SANITIZER_VERSION,
    BenchmarkWandbFinalization,
    ResolvedWandbConfig,
    WandbMode,
    WandbRunMetadata,
    publish_benchmark_wandb_best_effort,
)
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import Rewrite as Rewrite
from anonymizer.config.anonymizer_config import (
    infer_input_source_suffix,
    is_remote_input_source,
)
from anonymizer.config.replace_strategies import Redact as Redact
from anonymizer.config.rewrite import RiskTolerance as RiskTolerance
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.measurement import (
    MEASUREMENT_SCHEMA_VERSION,
    MeasurementConfig,
    configured_measurement_session,
    record_evaluation_metrics,
)

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.benchmark")

BENCHMARK_SUITE_SCHEMA_VERSION = 1
WANDB_METADATA_SCHEMA_VERSION = 2

_CaseRunPaths = CaseRunPaths
_active_config_ids = active_config_ids
_cross_product_matrix = cross_product_matrix
_duplicate_matrix_entries = duplicate_matrix_entries
_duplicates = duplicates
_input_columns = input_columns
_is_local_input_source = is_local_input_source
_materialize_sliced_source = materialize_sliced_source
_preflight_config_errors = preflight_config_errors
_preflight_model_configs = preflight_model_configs
_preflight_model_providers = preflight_model_providers
_preflight_model_providers_with_errors = preflight_model_providers_with_errors
_preflight_workload = preflight_workload
_preflight_workload_errors = preflight_workload_errors
_present = present
_privacy_goal = privacy_goal
_read_local_input_dataframe = read_local_input_dataframe
_resolve_config_source = resolve_config_source
_resolve_input_source = resolve_input_source
_resolve_optional_path = resolve_optional_path
_resolve_path = resolve_path
_safe_case_filename = safe_case_filename
_slice_bounds = slice_bounds
_workload_has_row_slice = workload_has_row_slice
_write_local_input_dataframe = write_local_input_dataframe


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
        execution=_execution_metadata(
            output_dir=output_dir,
            export=export,
            fail_fast=fail_fast,
            dd_trace=dd_trace,
            dd_task_trace=dd_task_trace,
        ),
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
    execution: dict[str, Any] | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        suite_id=spec.suite_id,
        output_dir=str(output_dir),
        measurement_path=str(measurement_path),
        summary_path=str(output_dir / "summary.json"),
        table_dir=str(table_dir) if table_dir is not None else None,
        detection_artifact_analysis_path=str(artifact_analysis_path) if artifact_analysis_path is not None else None,
        cases=cases,
        execution=execution or {},
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
    _execute_case(
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
    )
    detection_artifact_path = _case_detection_artifact_path(
        contexts,
        paths,
        case=case,
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
) -> Path | None:
    detection_artifact_path = _export_case_detection_artifacts_if_requested(
        contexts,
        paths.artifact_output_path,
        case=case,
        artifact_snapshot=paths.artifact_snapshot,
    )
    if detection_artifact_path is not None or paths.artifact_snapshot is None:
        return detection_artifact_path
    return None


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
) -> None:
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
        result = anonymizer.run(
            config=anonymizer_config,
            data=input_data,
        )
        if config.evaluate:
            evaluated = anonymizer.evaluate(result)
            record_evaluation_metrics(
                evaluated.trace_dataframe,
                mode="replace",
                strategy=type(anonymizer_config.replace).__name__,
                text_column=evaluated.resolved_text_column,
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
    return {
        **spec.run_tags,
        "suite_id": spec.suite_id,
        "workload_id": case.workload_id,
        "config_id": case.config_id,
        "repetition": case.repetition,
        "case_id": case.case_id,
    }


def _get_item(items: dict[str, Any], item_id: str, item_type: str) -> Any:
    if item_id not in items:
        raise ValueError(f"unknown {item_type}: {item_id}")
    return items[item_id]


def dry_run_result(
    spec: BenchmarkSpec,
    *,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
    dd_trace: DDTraceMode,
    trace_dir: Path | None,
    dd_task_trace: bool = False,
    task_trace_dir: Path | None = None,
) -> BenchmarkResult:
    return _canonical_dry_run_result(
        spec,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
        dd_task_trace=dd_task_trace,
        task_trace_dir=task_trace_dir,
        execution_metadata=_execution_metadata,
    )


def plan_suite(
    spec: BenchmarkSpec,
    *,
    spec_path: Path,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
    dd_trace: DDTraceMode = DDTraceMode.none,
    trace_dir: Path | None = None,
    dd_task_trace: bool = False,
    task_trace_dir: Path | None = None,
) -> BenchmarkResult:
    return _canonical_plan_suite(
        spec,
        spec_path=spec_path,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
        dd_task_trace=dd_task_trace,
        task_trace_dir=task_trace_dir,
        execution_metadata=_execution_metadata,
    )


def build_wandb_metadata(
    spec: BenchmarkSpec,
    *,
    spec_path: Path,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
    dd_trace: DDTraceMode,
    dd_task_trace: bool,
) -> WandbRunMetadata:
    sweep = _sweep_metadata(spec.run_tags)
    metadata = {
        "run_kind": "sweep_arm" if sweep is not None else "native_suite",
        "benchmark": _benchmark_metadata(spec, spec_path=spec_path),
        "execution": _execution_metadata(
            output_dir=output_dir,
            export=export,
            fail_fast=fail_fast,
            dd_trace=dd_trace,
            dd_task_trace=dd_task_trace,
        ),
        "runtime": _runtime_metadata(),
        "git": _git_metadata(Path.cwd()),
        "model_sources": _model_source_metadata(spec),
        "workloads": [_workload_metadata(workload) for workload in spec.workloads],
        "configs": [_config_metadata(config) for config in spec.configs],
        "matrix": [entry.model_dump(mode="json") for entry in (spec.matrix or _cross_product_matrix(spec))],
        "sweep": sweep,
    }
    return WandbRunMetadata.model_validate(metadata, strict=True)


def _benchmark_metadata(spec: BenchmarkSpec, *, spec_path: Path) -> dict[str, Any]:
    matrix = spec.matrix or _cross_product_matrix(spec)
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
    if spec_hash := _file_hash(spec_path):
        metadata["suite_file_hash"] = spec_hash
    return metadata


def _execution_metadata(
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


def _runtime_metadata() -> dict[str, Any]:
    return {
        **_package_versions(),
        "python_version": platform.python_version(),
        "platform_machine": platform.machine(),
        "platform_system": platform.system(),
    }


def _package_versions() -> dict[str, str | None]:
    return {
        "anonymizer_version": _package_version("nemo-anonymizer"),
        "datadesigner_version": _package_version("data-designer"),
        "wandb_version": _package_version("wandb"),
    }


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _model_source_metadata(spec: BenchmarkSpec) -> dict[str, bool]:
    return {
        "has_model_configs": spec.model_configs is not None,
        "has_model_providers": spec.model_providers is not None,
        "has_artifact_path": spec.artifact_path is not None,
    }


def _workload_metadata(workload: WorkloadSpec) -> dict[str, Any]:
    return {
        "id": workload.id,
        "source": _source_metadata(workload.source),
        "text_column": workload.text_column,
        "has_id_column": workload.id_column is not None,
        "has_data_summary": workload.data_summary is not None,
        "row_limit": workload.row_limit,
        "row_offset": workload.row_offset,
    }


def _source_metadata(source: str) -> dict[str, str | None]:
    return {
        "kind": "remote_file" if is_remote_input_source(source) else "local_file",
        "suffix": _source_suffix(source),
    }


def _source_suffix(source: str) -> str | None:
    try:
        return infer_input_source_suffix(source)
    except ValueError:
        return None


def _config_metadata(config: ConfigSpec) -> dict[str, Any]:
    return {
        "id": config.id,
        "mode": "rewrite" if config.rewrite is not None else "replace",
        "detect": _detect_metadata(config.detect),
        "replace": _replace_metadata(config.replace),
        "rewrite": _rewrite_metadata(config.rewrite),
        "evaluate": config.evaluate,
        "emit_telemetry": config.emit_telemetry,
    }


def _sweep_metadata(run_tags: dict[str, Any]) -> dict[str, Any] | None:
    sweep = run_tags.get("wandb_sweep")
    return sweep if isinstance(sweep, dict) else None


def _detect_metadata(detect: dict[str, Any]) -> dict[str, Any]:
    entity_labels = detect.get("entity_labels")
    metadata = {
        "entity_label_source": "custom" if isinstance(entity_labels, list) else "default",
        "entity_label_count": len(entity_labels) if isinstance(entity_labels, list) else None,
    }
    if isinstance(entity_labels, list):
        metadata["entity_label_set_hash"] = _stable_hash(",".join(sorted(map(str, entity_labels))))
    for key in ("gliner_threshold", "validation_max_entities_per_call", "validation_excerpt_window_chars"):
        if key in detect:
            metadata[key] = detect[key]
    return metadata


def _replace_metadata(replace: str | ReplaceSpec | None) -> dict[str, Any] | None:
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


def _rewrite_metadata(rewrite: RewriteSpec | None) -> dict[str, Any] | None:
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


def _git_metadata(cwd: Path) -> dict[str, str | bool | None]:
    commit = _git_output(cwd, "rev-parse", "HEAD")
    branch = _git_output(cwd, "rev-parse", "--abbrev-ref", "HEAD")
    status = _git_output(cwd, "status", "--short")
    return {"commit": commit, "branch": branch, "dirty": bool(status) if status is not None else None}


def _git_output(cwd: Path, *args: str) -> str | None:
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


def _file_hash(path: Path) -> str | None:
    if not path.is_file():
        return None
    return _stable_hash(path.read_text(encoding="utf-8"))


def _stable_hash(value: str) -> str:
    return stable_metadata_hash(value)


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
    wandb_mode: Annotated[WandbMode | None, cyclopts.Parameter("--wandb-mode")] = None,
    wandb_entity: Annotated[str | None, cyclopts.Parameter("--wandb-entity")] = None,
    wandb_project: Annotated[str | None, cyclopts.Parameter("--wandb-project")] = None,
    wandb_base_url: Annotated[str | None, cyclopts.Parameter("--wandb-base-url")] = None,
    wandb_group: Annotated[str | None, cyclopts.Parameter("--wandb-group")] = None,
    wandb_job_type: Annotated[str | None, cyclopts.Parameter("--wandb-job-type")] = None,
    wandb_run_name: Annotated[str | None, cyclopts.Parameter("--wandb-run-name")] = None,
    wandb_tags: Annotated[str | None, cyclopts.Parameter("--wandb-tags")] = None,
    wandb_log_tables: Annotated[bool | None, cyclopts.Parameter("--wandb-log-tables")] = None,
) -> None:
    configure_logging(log_format)
    try:
        wandb_settings = resolve_wandb_settings(
            wandb_mode=wandb_mode,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
            wandb_base_url=wandb_base_url,
            wandb_group=wandb_group,
            wandb_job_type=wandb_job_type,
            wandb_run_name=wandb_run_name,
            wandb_tags=wandb_tags,
            wandb_log_tables=wandb_log_tables,
        )
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
            wandb_settings=wandb_settings,
        )
    except ValidationError as exc:
        log_bad_input(logger, summarize_validation_error(exc))
        raise SystemExit(125) from exc
    except ValueError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")
    if any(case.status == CaseStatus.error for case in result.cases):
        raise SystemExit(1)


def resolve_wandb_settings(
    *,
    wandb_mode: WandbMode | None = None,
    wandb_entity: str | None = None,
    wandb_project: str | None = None,
    wandb_base_url: str | None = None,
    wandb_group: str | None = None,
    wandb_job_type: str | None = None,
    wandb_run_name: str | None = None,
    wandb_tags: str | None = None,
    wandb_log_tables: bool | None = None,
) -> ResolvedWandbConfig:
    return ResolvedWandbConfig.from_env_and_overrides(
        wandb_mode=wandb_mode,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_base_url=wandb_base_url,
        wandb_group=wandb_group,
        wandb_job_type=wandb_job_type,
        wandb_run_name=wandb_run_name,
        wandb_tags=wandb_tags,
        wandb_log_tables=wandb_log_tables,
    )


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
    wandb_settings: ResolvedWandbConfig | None = None,
) -> BenchmarkResult:
    benchmark_spec = load_spec(spec_path)
    output_dir = output or Path("benchmark-runs") / benchmark_spec.suite_id
    resolved_wandb = wandb_settings or ResolvedWandbConfig()
    if trace_dir is not None and dd_trace == DDTraceMode.none:
        raise ValueError("--trace-dir requires --dd-trace")
    if task_trace_dir is not None and not dd_task_trace:
        raise ValueError("--task-trace-dir requires --dd-task-trace")
    if dry_run:
        return plan_suite(
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
    preflight_suite(benchmark_spec, spec_path=spec_path)
    prepare_output_dir(output_dir, overwrite=overwrite, dry_run=dry_run)
    result = run_suite(
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
    publish_benchmark_wandb_best_effort(
        resolved_wandb,
        suite_id=benchmark_spec.suite_id,
        output_dir=output_dir,
        finalization=BenchmarkWandbFinalization(
            measurement_path=Path(result.measurement_path),
            cases=result.cases,
        ),
        metadata_factory=lambda: build_wandb_metadata(
            benchmark_spec,
            spec_path=spec_path,
            output_dir=output_dir,
            export=export,
            fail_fast=fail_fast,
            dd_trace=dd_trace,
            dd_task_trace=dd_task_trace,
        ),
    )
    return result


if __name__ == "__main__":
    app()
