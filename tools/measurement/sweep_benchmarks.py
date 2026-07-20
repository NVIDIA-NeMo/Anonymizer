#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a benchmark suite sweep as one W&B run per parameter arm.

Usage:
    uv run python tools/measurement/sweep_benchmarks.py sweep.yaml --dry-run
    uv run python tools/measurement/sweep_benchmarks.py sweep.yaml --wandb-mode online --create-report
    uv run python tools/measurement/sweep_benchmarks.py sweep.yaml --wandb-mode online --create-workspace
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable

import cyclopts
import run_benchmarks
from create_wandb_report import WandbProjectPath, create_benchmark_group_report, create_benchmark_workspace
from measurement_tools.benchmark_sweep_compiler import (
    arm_suite_payload,
    expand_sweep_arms,
    load_sweep_spec,
    materialize_arm_suite,
    rebase_suite_paths,
    resolve_path,
)
from measurement_tools.benchmark_sweep_models import SweepArm, SweepArmResult, SweepCliOptions, SweepResult, SweepSpec
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input, summarize_validation_error
from measurement_tools.wandb_models import generated_wandb_tag
from pydantic import ValidationError

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.sweep")


class WandbViewCreationError(RuntimeError):
    """A post-benchmark W&B presentation operation failed."""


@dataclass(frozen=True)
class _WandbViewOperation:
    enabled: bool
    label: str
    url_attribute: str
    create: Callable[..., Any]


_arm_suite_payload = arm_suite_payload
_rebase_suite_paths = rebase_suite_paths
_resolve_path = resolve_path


def run_sweep(
    sweep_path: Path,
    *,
    output_root: Path | None,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    wandb_settings: run_benchmarks.ResolvedWandbConfig,
    create_report: bool,
    create_workspace: bool = False,
) -> SweepResult:
    spec = load_sweep_spec(sweep_path)
    resolved_output_root = output_root or _default_output_root(spec, sweep_path)
    arms = expand_sweep_arms(spec)
    results = _run_sweep_arms(
        spec,
        arms,
        output_root=resolved_output_root,
        overwrite=overwrite,
        dry_run=dry_run,
        export=export,
        fail_fast=fail_fast,
        wandb_settings=wandb_settings,
    )
    report_url = None
    workspace_url = None
    report_error = None
    workspace_error = None
    for operation in (
        _WandbViewOperation(create_report, "report", "report_url", create_benchmark_group_report),
        _WandbViewOperation(create_workspace, "workspace", "workspace_url", create_benchmark_workspace),
    ):
        try:
            url = _maybe_create_view(spec, wandb_settings=wandb_settings, dry_run=dry_run, operation=operation)
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


def _run_sweep_arms(
    spec: SweepSpec,
    arms: list[SweepArm],
    *,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    wandb_settings: run_benchmarks.ResolvedWandbConfig,
) -> list[SweepArmResult]:
    results: list[SweepArmResult] = []
    for arm in arms:
        result = _run_arm(
            spec,
            arm,
            output_root=output_root,
            overwrite=overwrite,
            dry_run=dry_run,
            export=export,
            fail_fast=fail_fast,
            wandb_settings=wandb_settings,
        )
        results.append(result)
        if fail_fast and result.status == "error":
            break
    return results


def _run_arm(
    spec: SweepSpec,
    arm: SweepArm,
    *,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    wandb_settings: run_benchmarks.ResolvedWandbConfig,
) -> SweepArmResult:
    suite_path = output_root / arm.arm_id / "suite.yaml"
    output_dir = output_root / arm.arm_id / "output"
    run_name = f"{spec.sweep_id}-{arm.arm_id}"
    total_cases = 0
    try:
        arm_wandb_settings = _arm_wandb_settings(wandb_settings, spec=spec, arm=arm, run_name=run_name)
        if dry_run:
            total_cases, result = _plan_arm(
                spec,
                arm,
                suite_path=suite_path,
                output_dir=output_dir,
                export=export,
                fail_fast=fail_fast,
            )
        else:
            suite_path = materialize_arm_suite(spec, arm, output_root=output_root, overwrite=overwrite)
            total_cases = _planned_case_count(suite_path)
            result = run_benchmarks.run_or_plan(
                suite_path,
                output=output_dir,
                overwrite=overwrite,
                dry_run=False,
                export=export,
                fail_fast=fail_fast,
                wandb_settings=arm_wandb_settings,
            )
    except Exception as exc:  # noqa: BLE001 -- keep sweeping other arms and report failure
        return _arm_error(
            arm,
            suite_path=suite_path,
            output_dir=output_dir,
            run_name=run_name,
            total_cases=total_cases or _planned_case_count_if_readable(suite_path),
            error=str(exc),
        )
    return _arm_result(arm, suite_path=suite_path, output_dir=output_dir, run_name=run_name, result=result)


def _plan_arm(
    spec: SweepSpec,
    arm: SweepArm,
    *,
    suite_path: Path,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
) -> tuple[int, run_benchmarks.BenchmarkResult]:
    benchmark_spec = run_benchmarks.BenchmarkSpec.model_validate(_arm_suite_payload(spec, arm))
    total_cases = len(run_benchmarks.build_cases(benchmark_spec))
    result = run_benchmarks.plan_suite(
        benchmark_spec,
        spec_path=suite_path,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
    )
    return total_cases, result


def _arm_wandb_settings(
    settings: run_benchmarks.ResolvedWandbConfig,
    *,
    spec: SweepSpec,
    arm: SweepArm,
    run_name: str,
) -> run_benchmarks.ResolvedWandbConfig:
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
        wandb_tags=_joined_tags(settings.effective_wandb_tags, ["sweep", *generated_tags]),
    )


def _arm_result(
    arm: SweepArm,
    *,
    suite_path: Path,
    output_dir: Path,
    run_name: str,
    result: run_benchmarks.BenchmarkResult,
) -> SweepArmResult:
    errored = sum(1 for case in result.cases if case.status == run_benchmarks.CaseStatus.error)
    completed = sum(1 for case in result.cases if case.status == run_benchmarks.CaseStatus.completed)
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


def _arm_error(
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


def _planned_case_count(suite_path: Path) -> int:
    return len(run_benchmarks.build_cases(run_benchmarks.load_spec(suite_path)))


def _planned_case_count_if_readable(suite_path: Path) -> int:
    if not suite_path.exists():
        return 0
    try:
        return _planned_case_count(suite_path)
    except Exception:  # noqa: BLE001 -- keep the original per-arm failure as the reported error
        return 0


def _maybe_create_view(
    spec: SweepSpec,
    *,
    wandb_settings: run_benchmarks.ResolvedWandbConfig,
    dry_run: bool,
    operation: _WandbViewOperation,
) -> str | None:
    if not operation.enabled or dry_run or not wandb_settings.enabled:
        return None
    entity = wandb_settings.wandb_entity
    if not entity:
        return None
    try:
        project = WandbProjectPath(
            entity=entity,
            project=wandb_settings.effective_wandb_project,
        )
        result = operation.create(
            project,
            settings=wandb_settings,
            group=wandb_settings.wandb_group or spec.sweep_id,
            expected_run_kind="sweep_arm",
        )
    except Exception as exc:  # noqa: BLE001 -- remote SDK errors must not expose response contents
        raise WandbViewCreationError(f"W&B {operation.label} creation failed ({type(exc).__name__})") from None
    return getattr(result, operation.url_attribute)


def _joined_tags(first: list[str], second: list[str]) -> str:
    return ",".join(dict.fromkeys([*first, *second]))


def _default_output_root(spec: SweepSpec, sweep_path: Path) -> Path:
    if spec.output_root:
        return _resolve_path(spec.output_root, sweep_path.parent)
    return sweep_path.with_suffix("").with_name(spec.sweep_id)


def render_result(result: SweepResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [f"Ran {result.completed_arms}/{len(result.arms)} sweep arm(s); errors={result.errored_arms}"]
    lines.append(f"Output: {result.output_root}")
    if result.report_url:
        lines.append(f"Report: {result.report_url}")
    if result.workspace_url:
        lines.append(f"Workspace: {result.workspace_url}")
    for arm in result.arms:
        line = (
            f"- {arm.arm_id}: {arm.status}, cases={arm.completed_cases}/{arm.total_cases}, errors={arm.errored_cases}"
        )
        if arm.error:
            line = f"{line}, error={arm.error}"
        lines.append(line)
    return "\n".join(lines)


def _resolve_cli_wandb_settings(**overrides: Any) -> run_benchmarks.ResolvedWandbConfig:
    return run_benchmarks.ResolvedWandbConfig.from_env_and_overrides(**overrides)


@app.default
def main(
    spec: Path,
    *,
    output_root: Annotated[Path | None, cyclopts.Parameter("--output-root")] = None,
    overwrite: Annotated[bool, cyclopts.Parameter("--overwrite")] = False,
    dry_run: Annotated[bool, cyclopts.Parameter("--dry-run")] = False,
    export: Annotated[bool, cyclopts.Parameter("--export")] = True,
    fail_fast: Annotated[bool, cyclopts.Parameter("--fail-fast")] = False,
    create_report: Annotated[bool, cyclopts.Parameter("--create-report")] = False,
    create_workspace: Annotated[bool, cyclopts.Parameter("--create-workspace")] = False,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
    wandb_mode: Annotated[run_benchmarks.WandbMode | None, cyclopts.Parameter("--wandb-mode")] = None,
    wandb_entity: Annotated[str | None, cyclopts.Parameter("--wandb-entity")] = None,
    wandb_project: Annotated[str | None, cyclopts.Parameter("--wandb-project")] = None,
    wandb_base_url: Annotated[str | None, cyclopts.Parameter("--wandb-base-url")] = None,
    wandb_group: Annotated[str | None, cyclopts.Parameter("--wandb-group")] = None,
    wandb_job_type: Annotated[str | None, cyclopts.Parameter("--wandb-job-type")] = None,
    wandb_tags: Annotated[str | None, cyclopts.Parameter("--wandb-tags")] = None,
    wandb_log_tables: Annotated[bool | None, cyclopts.Parameter("--wandb-log-tables")] = None,
) -> None:
    configure_logging(log_format)
    options = SweepCliOptions(
        spec=spec,
        output_root=output_root,
        overwrite=overwrite,
        dry_run=dry_run,
        export=export,
        fail_fast=fail_fast,
        create_report=create_report,
        create_workspace=create_workspace,
        wandb_mode=wandb_mode,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_base_url=wandb_base_url,
        wandb_group=wandb_group,
        wandb_job_type=wandb_job_type,
        wandb_tags=wandb_tags,
        wandb_log_tables=wandb_log_tables,
    )
    try:
        result = _run_sweep_from_cli(options)
    except ValidationError as exc:
        log_bad_input(logger, summarize_validation_error(exc))
        raise SystemExit(125) from exc
    except ValueError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")
    if result.errored_arms or result.report_error is not None or result.workspace_error is not None:
        raise SystemExit(1)


def _run_sweep_from_cli(options: SweepCliOptions) -> SweepResult:
    wandb_settings = _resolve_cli_wandb_settings(
        wandb_mode=options.wandb_mode,
        wandb_entity=options.wandb_entity,
        wandb_project=options.wandb_project,
        wandb_base_url=options.wandb_base_url,
        wandb_group=options.wandb_group,
        wandb_job_type=options.wandb_job_type,
        wandb_tags=options.wandb_tags,
        wandb_log_tables=options.wandb_log_tables,
    )
    return run_sweep(
        options.spec,
        output_root=options.output_root,
        overwrite=options.overwrite,
        dry_run=options.dry_run,
        export=options.export,
        fail_fast=options.fail_fast,
        wandb_settings=wandb_settings,
        create_report=options.create_report,
        create_workspace=options.create_workspace,
    )


if __name__ == "__main__":
    app()
