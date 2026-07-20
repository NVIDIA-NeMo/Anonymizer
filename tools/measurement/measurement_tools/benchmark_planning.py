# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark dry-run result construction and suite planning."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from measurement_tools.benchmark_models import BenchmarkResult, BenchmarkSpec, DDTraceMode
from measurement_tools.benchmark_specs import build_cases, preflight_suite


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
    execution_metadata: Callable[..., dict[str, Any]],
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
        execution=execution_metadata(
            output_dir=output_dir,
            export=export,
            fail_fast=fail_fast,
            dd_trace=dd_trace,
            dd_task_trace=dd_task_trace,
        ),
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
    execution_metadata: Callable[..., dict[str, Any]],
) -> BenchmarkResult:
    preflight_suite(spec, spec_path=spec_path)
    return dry_run_result(
        spec,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
        dd_task_trace=dd_task_trace,
        task_trace_dir=task_trace_dir,
        execution_metadata=execution_metadata,
    )


__all__ = ["dry_run_result", "plan_suite"]
