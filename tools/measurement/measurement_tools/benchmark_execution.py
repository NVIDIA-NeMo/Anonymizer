# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark suite and case execution orchestration."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.measurement import MeasurementConfig, configured_measurement_session, record_evaluation_metrics
from measurement_tools.benchmark_artifacts import (
    combine_detection_artifact_analysis,
    combine_measurements,
    export_case_detection_artifact_analysis,
    export_measurement_tables,
    snapshot_detection_artifacts,
    write_summary,
)
from measurement_tools.benchmark_inputs import (
    build_anonymizer_config,
    build_input,
    resolve_config_source,
    resolve_optional_path,
)
from measurement_tools.benchmark_models import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkSpec,
    CaseRunPaths,
    CaseStatus,
    ConfigSpec,
    DDTraceMode,
    WorkloadSpec,
)
from measurement_tools.benchmark_planning import plan_suite
from measurement_tools.benchmark_specs import build_cases, load_spec, preflight_suite, prepare_output_dir
from measurement_tools.benchmark_wandb_metadata import build_wandb_metadata, execution_metadata, run_tags
from measurement_tools.wandb_setup import (
    BenchmarkWandbFinalization,
    ResolvedWandbConfig,
    publish_benchmark_wandb_best_effort,
)

logger = logging.getLogger("measurement.benchmark")


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
    anonymizer_factory: Callable[..., Anonymizer] = Anonymizer,
    run_case_operation: Callable[..., BenchmarkCase] | None = None,
    combine_measurements_operation: Callable[[list[BenchmarkCase], Path], Path] = combine_measurements,
    export_tables_operation: Callable[[Path, Path], Path] = export_measurement_tables,
    combine_artifacts_operation: Callable[
        [list[BenchmarkCase], Path], Path | None
    ] = combine_detection_artifact_analysis,
    execution_metadata_builder: Callable[..., dict[str, Any]] = execution_metadata,
    write_summary_operation: Callable[[BenchmarkResult], None] = write_summary,
) -> BenchmarkResult:
    case_operation = run_case_operation or run_case
    contexts = build_contexts(
        spec,
        spec_path=spec_path,
        output_dir=output_dir,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
        dd_task_trace=dd_task_trace,
        task_trace_dir=task_trace_dir,
    )
    anonymizer = anonymizer_factory(**contexts["anonymizer_kwargs"])
    cases = run_cases(
        spec,
        contexts=contexts,
        anonymizer=anonymizer,
        fail_fast=fail_fast,
        export=export,
        run_case_operation=case_operation,
    )
    measurement_path = combine_measurements_operation(cases, output_dir / "measurements.jsonl")
    should_export = should_export_measurements(export=export, measurement_path=measurement_path)
    table_dir = export_suite_tables(
        measurement_path,
        output_dir=output_dir,
        should_export=should_export,
        export_tables_operation=export_tables_operation,
    )
    artifact_analysis_path = combine_suite_detection_artifacts(
        cases,
        output_dir=output_dir,
        should_export=should_export,
        combine_artifacts_operation=combine_artifacts_operation,
    )
    result = benchmark_result(
        spec,
        output_dir=output_dir,
        measurement_path=measurement_path,
        table_dir=table_dir,
        artifact_analysis_path=artifact_analysis_path,
        cases=cases,
        execution=execution_metadata_builder(
            output_dir=output_dir,
            export=export,
            fail_fast=fail_fast,
            dd_trace=dd_trace,
            dd_task_trace=dd_task_trace,
        ),
    )
    write_summary_operation(result)
    return result


def run_cases(
    spec: BenchmarkSpec,
    *,
    contexts: dict[str, Any],
    anonymizer: Anonymizer,
    fail_fast: bool,
    export: bool,
    run_case_operation: Callable[..., BenchmarkCase] | None = None,
) -> list[BenchmarkCase]:
    case_operation = run_case_operation or run_case
    return [
        case_operation(
            case,
            spec,
            contexts=contexts,
            anonymizer=anonymizer,
            fail_fast=fail_fast,
            export_detection_artifacts=export,
        )
        for case in build_cases(spec)
    ]


def should_export_measurements(*, export: bool, measurement_path: Path) -> bool:
    return export and measurement_path.stat().st_size > 0


def export_suite_tables(
    measurement_path: Path,
    *,
    output_dir: Path,
    should_export: bool,
    export_tables_operation: Callable[[Path, Path], Path] = export_measurement_tables,
) -> Path | None:
    if not should_export:
        return None
    return export_tables_operation(measurement_path, output_dir / "tables")


def combine_suite_detection_artifacts(
    cases: list[BenchmarkCase],
    *,
    output_dir: Path,
    should_export: bool,
    combine_artifacts_operation: Callable[
        [list[BenchmarkCase], Path], Path | None
    ] = combine_detection_artifact_analysis,
) -> Path | None:
    if not should_export:
        return None
    return combine_artifacts_operation(cases, output_dir / "detection-artifacts.jsonl")


def benchmark_result(
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


def build_contexts(
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
    artifact_path = resolve_optional_path(spec.artifact_path, base_dir) or output_dir / "artifacts"
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
            "model_configs": resolve_config_source(spec.model_configs, base_dir),
            "model_providers": resolve_config_source(spec.model_providers, base_dir),
            "artifact_path": artifact_path,
        },
    }


def run_case(
    case: BenchmarkCase,
    spec: BenchmarkSpec,
    *,
    contexts: dict[str, Any],
    anonymizer: Anonymizer,
    fail_fast: bool,
    export_detection_artifacts: bool,
    execute_case_operation: Callable[..., None] | None = None,
    run_case_error_operation: Callable[..., BenchmarkCase] | None = None,
) -> BenchmarkCase:
    execute_operation = execute_case_operation or execute_case
    error_operation = run_case_error_operation or run_case_error
    started = time.perf_counter()
    attempt_errors: list[str] = []
    max_attempts = 1 if fail_fast else spec.case_retries + 1
    for attempt_number in range(1, max_attempts + 1):
        paths = case_run_paths(case, contexts=contexts, export_detection_artifacts=export_detection_artifacts)
        try:
            return run_case_success(
                case,
                spec,
                contexts=contexts,
                anonymizer=anonymizer,
                paths=paths,
                started=started,
                attempt_count=attempt_number,
                attempt_errors=attempt_errors,
                execute_case_operation=execute_operation,
            )
        except Exception as exc:
            if fail_fast:
                raise
            attempt_errors.append(str(exc))
            if attempt_number >= max_attempts:
                return error_operation(
                    case,
                    contexts=contexts,
                    paths=paths,
                    started=started,
                    error=exc,
                    attempt_count=attempt_number,
                    attempt_errors=attempt_errors,
                )
            sleep_before_case_retry(spec, case=case, attempt_number=attempt_number, error=exc)

    raise RuntimeError("unreachable benchmark retry state")


def sleep_before_case_retry(
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


def case_run_paths(
    case: BenchmarkCase,
    *,
    contexts: dict[str, Any],
    export_detection_artifacts: bool,
) -> CaseRunPaths:
    return CaseRunPaths(
        raw_path=contexts["raw_dir"] / f"{case.case_id}.jsonl",
        artifact_output_path=contexts["raw_dir"] / f"{case.case_id}.detection-artifacts.jsonl",
        trace_path=case_trace_path(case, contexts=contexts),
        task_trace_path=case_task_trace_path(case, contexts=contexts),
        artifact_snapshot=snapshot_detection_artifacts(contexts["artifact_path"])
        if export_detection_artifacts
        else None,
        export_detection_artifacts=export_detection_artifacts,
    )


def run_case_success(
    case: BenchmarkCase,
    spec: BenchmarkSpec,
    *,
    contexts: dict[str, Any],
    anonymizer: Anonymizer,
    paths: CaseRunPaths,
    started: float,
    attempt_count: int,
    attempt_errors: list[str],
    execute_case_operation: Callable[..., None] | None = None,
) -> BenchmarkCase:
    execute_operation = execute_case_operation or execute_case
    workload = get_item(contexts["workloads"], case.workload_id, "workload")
    config = get_item(contexts["configs"], case.config_id, "config")
    execute_operation(
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
    detection_artifact_path = case_detection_artifact_path(contexts, paths, case=case)
    return case_with_result(
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


def run_case_error(
    case: BenchmarkCase,
    *,
    contexts: dict[str, Any],
    paths: CaseRunPaths,
    started: float,
    error: Exception,
    attempt_count: int,
    attempt_errors: list[str],
    artifact_export_operation: Callable[..., Path | None] = export_case_detection_artifact_analysis,
) -> BenchmarkCase:
    detection_artifact_path = export_case_detection_artifacts_if_requested(
        contexts,
        paths.artifact_output_path,
        case=case,
        artifact_snapshot=paths.artifact_snapshot,
        artifact_export_operation=artifact_export_operation,
    )
    return case_with_result(
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


def case_detection_artifact_path(
    contexts: dict[str, Any],
    paths: CaseRunPaths,
    *,
    case: BenchmarkCase,
) -> Path | None:
    detection_artifact_path = export_case_detection_artifacts_if_requested(
        contexts,
        paths.artifact_output_path,
        case=case,
        artifact_snapshot=paths.artifact_snapshot,
    )
    if detection_artifact_path is not None or paths.artifact_snapshot is None:
        return detection_artifact_path
    return None


def case_with_result(
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


def export_case_detection_artifacts_if_requested(
    contexts: dict[str, Any],
    output_path: Path,
    *,
    case: BenchmarkCase,
    artifact_snapshot: dict[str, int] | None,
    artifact_export_operation: Callable[..., Path | None] = export_case_detection_artifact_analysis,
) -> Path | None:
    if artifact_snapshot is None:
        return None
    return artifact_export_operation(
        contexts["artifact_path"],
        output_path,
        case=case,
        artifact_snapshot=artifact_snapshot,
    )


def case_trace_path(case: BenchmarkCase, *, contexts: dict[str, Any]) -> Path | None:
    if contexts["dd_trace"] == DDTraceMode.none:
        return None
    return contexts["trace_dir"] / f"{case.case_id}.jsonl"


def case_task_trace_path(case: BenchmarkCase, *, contexts: dict[str, Any]) -> Path | None:
    if not contexts["dd_task_trace"]:
        return None
    return contexts["task_trace_dir"] / f"{case.case_id}.jsonl"


def execute_case(
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
    build_config_operation: Callable[[ConfigSpec], Any] = build_anonymizer_config,
    build_input_operation: Callable[..., Any] = build_input,
    run_tags_operation: Callable[[BenchmarkCase, BenchmarkSpec], dict[str, Any]] = run_tags,
    measurement_session: Callable[..., Any] = configured_measurement_session,
    record_metrics_operation: Callable[..., None] = record_evaluation_metrics,
) -> None:
    anonymizer_config = build_config_operation(config)
    input_data = build_input_operation(
        workload,
        base_dir,
        slice_dir=raw_path.parent / "inputs",
        case_id=case.case_id,
    )
    measurement = MeasurementConfig(
        output_path=raw_path,
        run_id=case.case_id,
        run_tags=run_tags_operation(case, spec),
        streaming=True,
        keep_records=False,
        dd_trace=dd_trace.value,
        dd_trace_path=trace_path,
        dd_task_trace_path=task_trace_path,
        fail_on_write_error=True,
    )
    with measurement_session(measurement):
        result = anonymizer.run(config=anonymizer_config, data=input_data)
        if config.evaluate:
            evaluated = anonymizer.evaluate(result)
            record_metrics_operation(
                evaluated.trace_dataframe,
                mode="replace",
                strategy=type(anonymizer_config.replace).__name__,
                text_column=evaluated.resolved_text_column,
            )


def get_item(items: dict[str, Any], item_id: str, item_type: str) -> Any:
    if item_id not in items:
        raise ValueError(f"unknown {item_type}: {item_id}")
    return items[item_id]


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
    load_spec_operation: Callable[[Path], BenchmarkSpec] = load_spec,
    plan_suite_operation: Callable[..., BenchmarkResult] = plan_suite,
    preflight_operation: Callable[..., None] = preflight_suite,
    prepare_output_operation: Callable[..., None] = prepare_output_dir,
    run_suite_operation: Callable[..., BenchmarkResult] = run_suite,
    publisher_operation: Callable[..., Any] = publish_benchmark_wandb_best_effort,
    metadata_builder: Callable[..., Any] = build_wandb_metadata,
) -> BenchmarkResult:
    benchmark_spec = load_spec_operation(spec_path)
    output_dir = output or Path("benchmark-runs") / benchmark_spec.suite_id
    resolved_wandb = wandb_settings or ResolvedWandbConfig()
    if trace_dir is not None and dd_trace == DDTraceMode.none:
        raise ValueError("--trace-dir requires --dd-trace")
    if task_trace_dir is not None and not dd_task_trace:
        raise ValueError("--task-trace-dir requires --dd-task-trace")
    if dry_run:
        return plan_suite_operation(
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
    preflight_operation(benchmark_spec, spec_path=spec_path)
    prepare_output_operation(output_dir, overwrite=overwrite, dry_run=dry_run)
    result = run_suite_operation(
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
    publisher_operation(
        resolved_wandb,
        suite_id=benchmark_spec.suite_id,
        output_dir=output_dir,
        finalization=BenchmarkWandbFinalization(measurement_path=Path(result.measurement_path), cases=result.cases),
        metadata_factory=lambda: metadata_builder(
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


__all__ = [
    "benchmark_result",
    "build_contexts",
    "case_detection_artifact_path",
    "case_run_paths",
    "case_task_trace_path",
    "case_trace_path",
    "case_with_result",
    "combine_suite_detection_artifacts",
    "execute_case",
    "export_case_detection_artifacts_if_requested",
    "export_suite_tables",
    "get_item",
    "run_case",
    "run_case_error",
    "run_case_success",
    "run_cases",
    "run_or_plan",
    "run_suite",
    "should_export_measurements",
    "sleep_before_case_retry",
]
