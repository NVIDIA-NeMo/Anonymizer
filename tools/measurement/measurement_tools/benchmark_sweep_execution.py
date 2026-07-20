# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plan and execute benchmark sweep arms and assemble their results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from measurement_tools.benchmark_models import BenchmarkResult, BenchmarkSpec, CaseStatus
from measurement_tools.benchmark_sweep_compiler import (
    arm_suite_payload,
    expand_sweep_arms,
    load_sweep_spec,
    materialize_arm_suite,
    resolve_path,
)
from measurement_tools.benchmark_sweep_models import SweepArm, SweepArmResult, SweepResult, SweepSpec
from measurement_tools.benchmark_sweep_views import (
    WandbViewCreationError,
    WandbViewOperation,
    create_view,
)
from measurement_tools.wandb_models import generated_wandb_tag
from measurement_tools.wandb_settings import ResolvedWandbConfig


def run_sweep(
    sweep_path: Path,
    *,
    output_root: Path | None,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    wandb_settings: ResolvedWandbConfig,
    create_report: bool,
    create_workspace: bool = False,
    run_or_plan_operation: Callable[..., BenchmarkResult],
    create_report_operation: Callable[..., Any],
    create_workspace_operation: Callable[..., Any],
    benchmark_spec_type: type[BenchmarkSpec],
    build_cases_operation: Callable[..., list[Any]],
    load_spec_operation: Callable[..., BenchmarkSpec],
    plan_suite_operation: Callable[..., BenchmarkResult],
) -> SweepResult:
    spec = load_sweep_spec(sweep_path)
    resolved_output_root = output_root or default_output_root(spec, sweep_path)
    arms = expand_sweep_arms(spec)
    results = run_sweep_arms(
        spec,
        arms,
        output_root=resolved_output_root,
        overwrite=overwrite,
        dry_run=dry_run,
        export=export,
        fail_fast=fail_fast,
        wandb_settings=wandb_settings,
        run_or_plan_operation=run_or_plan_operation,
        benchmark_spec_type=benchmark_spec_type,
        build_cases_operation=build_cases_operation,
        load_spec_operation=load_spec_operation,
        plan_suite_operation=plan_suite_operation,
    )
    report_url = None
    workspace_url = None
    report_error = None
    workspace_error = None
    for operation in (
        WandbViewOperation(create_report, "report", "report_url", create_report_operation),
        WandbViewOperation(create_workspace, "workspace", "workspace_url", create_workspace_operation),
    ):
        try:
            url = create_view(spec, wandb_settings=wandb_settings, dry_run=dry_run, operation=operation)
        except WandbViewCreationError as exc:
            if operation.label == "report":
                report_error = str(exc)
            else:
                workspace_error = str(exc)
        else:
            if operation.label == "report":
                report_url = url
            else:
                workspace_url = url
    return SweepResult(
        sweep_id=spec.sweep_id,
        output_root=str(resolved_output_root),
        arms=results,
        report_url=report_url,
        workspace_url=workspace_url,
        report_error=report_error,
        workspace_error=workspace_error,
    )


def run_sweep_arms(
    spec: SweepSpec,
    arms: list[SweepArm],
    *,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    wandb_settings: ResolvedWandbConfig,
    run_or_plan_operation: Callable[..., BenchmarkResult],
    benchmark_spec_type: type[BenchmarkSpec],
    build_cases_operation: Callable[..., list[Any]],
    load_spec_operation: Callable[..., BenchmarkSpec],
    plan_suite_operation: Callable[..., BenchmarkResult],
) -> list[SweepArmResult]:
    results: list[SweepArmResult] = []
    for arm in arms:
        result = run_arm(
            spec,
            arm,
            output_root=output_root,
            overwrite=overwrite,
            dry_run=dry_run,
            export=export,
            fail_fast=fail_fast,
            wandb_settings=wandb_settings,
            run_or_plan_operation=run_or_plan_operation,
            benchmark_spec_type=benchmark_spec_type,
            build_cases_operation=build_cases_operation,
            load_spec_operation=load_spec_operation,
            plan_suite_operation=plan_suite_operation,
        )
        results.append(result)
        if fail_fast and result.status == "error":
            break
    return results


def run_arm(
    spec: SweepSpec,
    arm: SweepArm,
    *,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    wandb_settings: ResolvedWandbConfig,
    run_or_plan_operation: Callable[..., BenchmarkResult],
    benchmark_spec_type: type[BenchmarkSpec],
    build_cases_operation: Callable[..., list[Any]],
    load_spec_operation: Callable[..., BenchmarkSpec],
    plan_suite_operation: Callable[..., BenchmarkResult],
) -> SweepArmResult:
    suite_path = output_root / arm.arm_id / "suite.yaml"
    output_dir = output_root / arm.arm_id / "output"
    run_name = f"{spec.sweep_id}-{arm.arm_id}"
    total_cases = 0
    try:
        arm_wandb_settings_value = arm_wandb_settings(wandb_settings, spec=spec, arm=arm, run_name=run_name)
        if dry_run:
            total_cases, result = plan_arm(
                spec,
                arm,
                suite_path=suite_path,
                output_dir=output_dir,
                export=export,
                fail_fast=fail_fast,
                benchmark_spec_type=benchmark_spec_type,
                build_cases_operation=build_cases_operation,
                plan_suite_operation=plan_suite_operation,
            )
        else:
            suite_path = materialize_arm_suite(spec, arm, output_root=output_root, overwrite=overwrite)
            total_cases = planned_case_count(
                suite_path,
                build_cases_operation=build_cases_operation,
                load_spec_operation=load_spec_operation,
            )
            result = run_or_plan_operation(
                suite_path,
                output=output_dir,
                overwrite=overwrite,
                dry_run=False,
                export=export,
                fail_fast=fail_fast,
                wandb_settings=arm_wandb_settings_value,
            )
    except Exception as exc:  # noqa: BLE001 -- keep sweeping other arms and report failure
        return arm_error(
            arm,
            suite_path=suite_path,
            output_dir=output_dir,
            run_name=run_name,
            total_cases=total_cases
            or planned_case_count_if_readable(
                suite_path,
                build_cases_operation=build_cases_operation,
                load_spec_operation=load_spec_operation,
            ),
            error=str(exc),
        )
    return arm_result(arm, suite_path=suite_path, output_dir=output_dir, run_name=run_name, result=result)


def plan_arm(
    spec: SweepSpec,
    arm: SweepArm,
    *,
    suite_path: Path,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
    benchmark_spec_type: type[BenchmarkSpec],
    build_cases_operation: Callable[..., list[Any]],
    plan_suite_operation: Callable[..., BenchmarkResult],
) -> tuple[int, BenchmarkResult]:
    benchmark_spec = benchmark_spec_type.model_validate(arm_suite_payload(spec, arm))
    total_cases = len(build_cases_operation(benchmark_spec))
    result = plan_suite_operation(
        benchmark_spec,
        spec_path=suite_path,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
    )
    return total_cases, result


def arm_wandb_settings(
    settings: ResolvedWandbConfig,
    *,
    spec: SweepSpec,
    arm: SweepArm,
    run_name: str,
) -> ResolvedWandbConfig:
    if not settings.enabled:
        return settings
    generated_tags = [
        tag
        for tag in (generated_wandb_tag("sweep", spec.sweep_id), generated_wandb_tag("arm", arm.arm_id))
        if tag is not None
    ]
    return settings.validated_update(
        wandb_group=settings.wandb_group or spec.sweep_id,
        wandb_job_type=settings.wandb_job_type or "benchmark-sweep",
        wandb_run_name=settings.wandb_run_name or run_name,
        wandb_tags=joined_tags(settings.effective_wandb_tags, ["sweep", *generated_tags]),
    )


def arm_result(
    arm: SweepArm,
    *,
    suite_path: Path,
    output_dir: Path,
    run_name: str,
    result: BenchmarkResult,
) -> SweepArmResult:
    errored = sum(1 for case in result.cases if case.status == CaseStatus.error)
    completed = sum(1 for case in result.cases if case.status == CaseStatus.completed)
    status = "error" if errored else "completed"
    return SweepArmResult(
        arm_id=arm.arm_id,
        parameters=arm.parameters,
        suite_path=str(suite_path),
        output_dir=str(output_dir),
        wandb_run_name=run_name,
        status=status,
        completed_cases=completed,
        errored_cases=errored,
        total_cases=len(result.cases),
    )


def arm_error(
    arm: SweepArm,
    *,
    suite_path: Path,
    output_dir: Path,
    run_name: str,
    total_cases: int,
    error: str,
) -> SweepArmResult:
    return SweepArmResult(
        arm_id=arm.arm_id,
        parameters=arm.parameters,
        suite_path=str(suite_path),
        output_dir=str(output_dir),
        wandb_run_name=run_name,
        status="error",
        total_cases=total_cases,
        error=error,
    )


def planned_case_count(
    suite_path: Path,
    *,
    build_cases_operation: Callable[..., list[Any]],
    load_spec_operation: Callable[..., BenchmarkSpec],
) -> int:
    return len(build_cases_operation(load_spec_operation(suite_path)))


def planned_case_count_if_readable(
    suite_path: Path,
    *,
    build_cases_operation: Callable[..., list[Any]],
    load_spec_operation: Callable[..., BenchmarkSpec],
) -> int:
    if not suite_path.exists():
        return 0
    try:
        return planned_case_count(
            suite_path,
            build_cases_operation=build_cases_operation,
            load_spec_operation=load_spec_operation,
        )
    except Exception:  # noqa: BLE001 -- keep the original per-arm failure as the reported error
        return 0


def joined_tags(first: list[str], second: list[str]) -> str:
    return ",".join(dict.fromkeys([*first, *second]))


def default_output_root(spec: SweepSpec, sweep_path: Path) -> Path:
    if spec.output_root:
        return resolve_path(spec.output_root, sweep_path.parent)
    return sweep_path.with_suffix("").with_name(spec.sweep_id)


__all__ = [
    "arm_error",
    "arm_result",
    "arm_wandb_settings",
    "default_output_root",
    "joined_tags",
    "plan_arm",
    "planned_case_count",
    "planned_case_count_if_readable",
    "run_arm",
    "run_sweep",
    "run_sweep_arms",
]
