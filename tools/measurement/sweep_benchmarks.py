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
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import run_benchmarks
from create_wandb_report import create_benchmark_group_report, create_benchmark_workspace
from measurement_tools import benchmark_sweep_execution
from measurement_tools.benchmark_sweep_compiler import (
    arm_suite_payload as arm_suite_payload,
)
from measurement_tools.benchmark_sweep_compiler import (
    expand_sweep_arms as expand_sweep_arms,
)
from measurement_tools.benchmark_sweep_compiler import (
    load_sweep_spec as load_sweep_spec,
)
from measurement_tools.benchmark_sweep_compiler import (
    materialize_arm_suite as materialize_arm_suite,
)
from measurement_tools.benchmark_sweep_compiler import (
    rebase_suite_paths,
    resolve_path,
)
from measurement_tools.benchmark_sweep_models import (
    SweepArm as SweepArm,
)
from measurement_tools.benchmark_sweep_models import (
    SweepArmResult as SweepArmResult,
)
from measurement_tools.benchmark_sweep_models import (
    SweepCliOptions,
    SweepResult,
)
from measurement_tools.benchmark_sweep_models import (
    SweepSpec as SweepSpec,
)
from measurement_tools.benchmark_sweep_views import (
    WandbViewCreationError as WandbViewCreationError,
)
from measurement_tools.benchmark_sweep_views import (
    WandbViewOperation,
)
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input, summarize_validation_error
from measurement_tools.wandb_report_models import WandbProjectPath as WandbProjectPath
from pydantic import ValidationError

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.sweep")


_arm_suite_payload = arm_suite_payload
_rebase_suite_paths = rebase_suite_paths
_resolve_path = resolve_path
_WandbViewOperation = WandbViewOperation
_arm_wandb_settings = benchmark_sweep_execution.arm_wandb_settings


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
    return benchmark_sweep_execution.run_sweep(
        sweep_path,
        output_root=output_root,
        overwrite=overwrite,
        dry_run=dry_run,
        export=export,
        fail_fast=fail_fast,
        wandb_settings=wandb_settings,
        create_report=create_report,
        create_workspace=create_workspace,
        run_or_plan_operation=run_benchmarks.run_or_plan,
        create_report_operation=create_benchmark_group_report,
        create_workspace_operation=create_benchmark_workspace,
        benchmark_spec_type=run_benchmarks.BenchmarkSpec,
        build_cases_operation=run_benchmarks.build_cases,
        load_spec_operation=run_benchmarks.load_spec,
        plan_suite_operation=run_benchmarks.plan_suite,
    )


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
