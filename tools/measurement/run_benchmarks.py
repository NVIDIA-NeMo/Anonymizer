#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Anonymizer benchmark suites and export measurement tables.

Usage:
    uv run python tools/measurement/run_benchmarks.py suite.yaml --output benchmark-runs/suite
    uv run python tools/measurement/run_benchmarks.py suite.yaml --dry-run --json
"""

import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import cyclopts
from measurement_tools.benchmark_artifacts import (
    changed_detection_artifact_files as changed_detection_artifact_files,
)
from measurement_tools.benchmark_artifacts import (
    combine_detection_artifact_analysis,
    combine_measurements,
    export_case_detection_artifact_analysis,
    export_measurement_tables,
    jsonl_chunk,
    render_result,
    with_case_metadata,
    write_summary,
)
from measurement_tools.benchmark_artifacts import (
    export_detection_artifact_analysis as export_detection_artifact_analysis,
)
from measurement_tools.benchmark_artifacts import snapshot_detection_artifacts as snapshot_detection_artifacts
from measurement_tools.benchmark_artifacts import (
    write_detection_artifact_payloads as write_detection_artifact_payloads,
)
from measurement_tools.benchmark_execution import (
    benchmark_result,
    build_contexts,
    case_detection_artifact_path,
    case_run_paths,
    case_task_trace_path,
    case_trace_path,
    case_with_result,
    combine_suite_detection_artifacts,
    export_case_detection_artifacts_if_requested,
    export_suite_tables,
    get_item,
    run_case_success,
    run_cases,
    should_export_measurements,
    sleep_before_case_retry,
)
from measurement_tools.benchmark_execution import execute_case as _canonical_execute_case
from measurement_tools.benchmark_execution import run_case as _canonical_run_case
from measurement_tools.benchmark_execution import run_case_error as _canonical_run_case_error
from measurement_tools.benchmark_execution import run_or_plan as _canonical_run_or_plan
from measurement_tools.benchmark_execution import run_suite as _canonical_run_suite
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
from measurement_tools.benchmark_models import ReplaceSpec as ReplaceSpec
from measurement_tools.benchmark_models import RewriteSpec as RewriteSpec
from measurement_tools.benchmark_planning import dry_run_result as _canonical_dry_run_result
from measurement_tools.benchmark_planning import plan_suite as _canonical_plan_suite
from measurement_tools.benchmark_specs import (
    active_config_ids,
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
from measurement_tools.benchmark_specs import build_cases as build_cases
from measurement_tools.benchmark_wandb_metadata import (
    BENCHMARK_SUITE_SCHEMA_VERSION as BENCHMARK_SUITE_SCHEMA_VERSION,
)
from measurement_tools.benchmark_wandb_metadata import (
    WANDB_METADATA_SCHEMA_VERSION as WANDB_METADATA_SCHEMA_VERSION,
)
from measurement_tools.benchmark_wandb_metadata import (
    benchmark_metadata,
    config_metadata,
    detect_metadata,
    execution_metadata,
    file_hash,
    git_metadata,
    git_output,
    model_source_metadata,
    package_version,
    package_versions,
    replace_metadata,
    rewrite_metadata,
    run_tags,
    runtime_metadata,
    source_metadata,
    source_suffix,
    stable_hash,
    sweep_metadata,
)
from measurement_tools.benchmark_wandb_metadata import build_wandb_metadata as _canonical_build_wandb_metadata
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input, summarize_validation_error
from measurement_tools.wandb_setup import BenchmarkWandbFinalization as BenchmarkWandbFinalization
from measurement_tools.wandb_setup import (
    ResolvedWandbConfig,
    WandbMode,
    WandbRunMetadata,
    publish_benchmark_wandb_best_effort,
)
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import Rewrite as Rewrite
from anonymizer.config.anonymizer_config import is_remote_input_source as is_remote_input_source
from anonymizer.config.replace_strategies import Redact as Redact
from anonymizer.config.rewrite import RiskTolerance as RiskTolerance
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.measurement import (
    configured_measurement_session,
    record_evaluation_metrics,
)

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.benchmark")

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
_benchmark_metadata = benchmark_metadata
_config_metadata = config_metadata
_detect_metadata = detect_metadata
_execution_metadata = execution_metadata
_file_hash = file_hash
_git_metadata = git_metadata
_git_output = git_output
_jsonl_chunk = jsonl_chunk
_model_source_metadata = model_source_metadata
_package_version = package_version
_package_versions = package_versions
_replace_metadata = replace_metadata
_rewrite_metadata = rewrite_metadata
_run_tags = run_tags
_runtime_metadata = runtime_metadata
_source_metadata = source_metadata
_source_suffix = source_suffix
_stable_hash = stable_hash
_sweep_metadata = sweep_metadata
_with_case_metadata = with_case_metadata
_benchmark_result = benchmark_result
_build_contexts = build_contexts
_case_detection_artifact_path = case_detection_artifact_path
_case_run_paths = case_run_paths
_case_task_trace_path = case_task_trace_path
_case_trace_path = case_trace_path
_case_with_result = case_with_result
_combine_suite_detection_artifacts = combine_suite_detection_artifacts
_export_case_detection_artifacts_if_requested = export_case_detection_artifacts_if_requested
_export_suite_tables = export_suite_tables
_get_item = get_item
_run_case_success = run_case_success
_run_cases = run_cases
_should_export_measurements = should_export_measurements
_sleep_before_case_retry = sleep_before_case_retry


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
    return _canonical_run_suite(
        spec,
        spec_path=spec_path,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
        dd_task_trace=dd_task_trace,
        task_trace_dir=task_trace_dir,
        anonymizer_factory=Anonymizer,
        run_case_operation=_run_case,
        combine_measurements_operation=combine_measurements,
        export_tables_operation=export_measurement_tables,
        combine_artifacts_operation=combine_detection_artifact_analysis,
        execution_metadata_builder=_execution_metadata,
        write_summary_operation=write_summary,
    )


def _run_case(
    case: BenchmarkCase,
    spec: BenchmarkSpec,
    *,
    contexts: dict[str, Any],
    anonymizer: Anonymizer,
    fail_fast: bool,
    export_detection_artifacts: bool,
) -> BenchmarkCase:
    return _canonical_run_case(
        case,
        spec,
        contexts=contexts,
        anonymizer=anonymizer,
        fail_fast=fail_fast,
        export_detection_artifacts=export_detection_artifacts,
        execute_case_operation=_execute_case,
        run_case_error_operation=_run_case_error,
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
    return _canonical_run_case_error(
        case,
        contexts=contexts,
        paths=paths,
        started=started,
        error=error,
        attempt_count=attempt_count,
        attempt_errors=attempt_errors,
        artifact_export_operation=export_case_detection_artifact_analysis,
    )


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
    return _canonical_execute_case(
        anonymizer,
        workload,
        config,
        raw_path=raw_path,
        trace_path=trace_path,
        task_trace_path=task_trace_path,
        case=case,
        spec=spec,
        base_dir=base_dir,
        dd_trace=dd_trace,
        build_config_operation=build_anonymizer_config,
        build_input_operation=build_input,
        run_tags_operation=_run_tags,
        measurement_session=configured_measurement_session,
        record_metrics_operation=record_evaluation_metrics,
    )


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
    return _canonical_build_wandb_metadata(
        spec,
        spec_path=spec_path,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        dd_task_trace=dd_task_trace,
        execution_metadata_builder=_execution_metadata,
        git_metadata_builder=_git_metadata,
        package_versions_builder=_package_versions,
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
    return _canonical_run_or_plan(
        spec_path,
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
        load_spec_operation=load_spec,
        plan_suite_operation=plan_suite,
        preflight_operation=preflight_suite,
        prepare_output_operation=prepare_output_dir,
        run_suite_operation=run_suite,
        publisher_operation=publish_benchmark_wandb_best_effort,
        metadata_builder=build_wandb_metadata,
    )


if __name__ == "__main__":
    app()
