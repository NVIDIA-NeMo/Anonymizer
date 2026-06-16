#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a benchmark suite sweep as one W&B run per parameter arm.

Usage:
    uv run python tools/measurement/sweep_benchmarks.py sweep.yaml --dry-run
    uv run python tools/measurement/sweep_benchmarks.py sweep.yaml --wandb-mode online --create-report
"""

from __future__ import annotations

import copy
import itertools
import logging
import sys
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import run_benchmarks
import yaml
from create_wandb_report import WandbProjectPath, create_benchmark_group_report
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.sweep")


class SweepSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sweep_id: str
    base_suite: str
    output_root: str | None = None
    parameters: dict[str, list[Any]] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_parameters(self) -> "SweepSpec":
        empty = [name for name, values in self.parameters.items() if not values]
        if empty:
            raise ValueError(f"sweep parameter(s) must have at least one value: {', '.join(sorted(empty))}")
        return self


class SweepArm(BaseModel):
    arm_id: str
    parameters: dict[str, Any]


class SweepArmResult(BaseModel):
    arm_id: str
    parameters: dict[str, Any]
    suite_path: str
    output_dir: str
    wandb_run_name: str
    status: str
    completed_cases: int = 0
    errored_cases: int = 0
    error: str | None = None


class SweepResult(BaseModel):
    sweep_id: str
    output_root: str
    arms: list[SweepArmResult]
    report_url: str | None = None

    @property
    def completed_arms(self) -> int:
        return sum(1 for arm in self.arms if arm.status == "completed")

    @property
    def errored_arms(self) -> int:
        return sum(1 for arm in self.arms if arm.status == "error")


def load_sweep_spec(path: Path) -> SweepSpec:
    raw = _read_yaml_mapping(path)
    spec = SweepSpec.model_validate(raw)
    return spec.model_copy(update={"base_suite": str(_resolve_path(spec.base_suite, path.parent))})


def expand_sweep_arms(spec: SweepSpec) -> list[SweepArm]:
    names = list(spec.parameters)
    value_grid = itertools.product(*(spec.parameters[name] for name in names))
    return [
        SweepArm(arm_id=f"arm-{index:03d}", parameters=dict(zip(names, values, strict=True)))
        for index, values in enumerate(value_grid)
    ]


def materialize_arm_suite(spec: SweepSpec, arm: SweepArm, *, output_root: Path, overwrite: bool) -> Path:
    base_path = Path(spec.base_suite)
    suite = _rebase_suite_paths(_read_yaml_mapping(base_path), base_path.parent)
    patched = _patched_suite(suite, spec=spec, arm=arm)
    arm_dir = output_root / arm.arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    suite_path = arm_dir / "suite.yaml"
    if suite_path.exists() and not overwrite:
        raise ValueError(f"sweep arm suite already exists: {suite_path}")
    suite_path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")
    return suite_path


def _patched_suite(suite: dict[str, Any], *, spec: SweepSpec, arm: SweepArm) -> dict[str, Any]:
    patched = copy.deepcopy(suite)
    run_tags = dict(patched.get("run_tags") or {})
    run_tags.update(_sweep_run_tags(spec, arm))
    patched["run_tags"] = run_tags
    for path, value in arm.parameters.items():
        _apply_parameter(patched, path, value)
    return patched


def _sweep_run_tags(spec: SweepSpec, arm: SweepArm) -> dict[str, Any]:
    tags: dict[str, Any] = {"sweep_id": spec.sweep_id, "sweep_arm_id": arm.arm_id}
    tags.update({f"sweep_param_{_safe_param_name(path)}": value for path, value in arm.parameters.items()})
    return tags


def _apply_parameter(suite: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if len(parts) >= 3 and parts[0] == "configs":
        _apply_config_parameter(suite, parts[1], parts[2:], value)
        return
    _set_nested(suite, parts, value)


def _apply_config_parameter(suite: dict[str, Any], config_selector: str, parts: list[str], value: Any) -> None:
    configs = suite.get("configs")
    if not isinstance(configs, list):
        raise ValueError("base suite must define a configs list")
    matched = [config for config in configs if isinstance(config, dict) and _config_matches(config, config_selector)]
    if not matched:
        raise ValueError(f"sweep parameter references unknown config selector: {config_selector}")
    for config in matched:
        _set_nested(config, parts, value)


def _config_matches(config: dict[str, Any], selector: str) -> bool:
    return selector == "*" or config.get("id") == selector


def _set_nested(target: dict[str, Any], parts: list[str], value: Any) -> None:
    current = target
    for part in parts[:-1]:
        existing = current.get(part)
        if isinstance(existing, str):
            existing = {"strategy": existing}
            current[part] = existing
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"cannot set nested sweep parameter through non-mapping field: {part}")
        current = existing
    current[parts[-1]] = value


def run_sweep(
    sweep_path: Path,
    *,
    output_root: Path | None,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    wandb_settings: run_benchmarks.WandbSettings,
    create_report: bool,
) -> SweepResult:
    spec = load_sweep_spec(sweep_path)
    resolved_output_root = output_root or _default_output_root(spec, sweep_path)
    arms = expand_sweep_arms(spec)
    results: list[SweepArmResult] = []
    for arm in arms:
        result = _run_arm(
            spec,
            arm,
            output_root=resolved_output_root,
            overwrite=overwrite,
            dry_run=dry_run,
            export=export,
            fail_fast=fail_fast,
            wandb_settings=wandb_settings,
        )
        results.append(result)
        if fail_fast and result.status == "error":
            break
    report_url = _maybe_create_report(spec, wandb_settings=wandb_settings, create_report=create_report, dry_run=dry_run)
    return SweepResult(
        sweep_id=spec.sweep_id, output_root=str(resolved_output_root), arms=results, report_url=report_url
    )


def _run_arm(
    spec: SweepSpec,
    arm: SweepArm,
    *,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    export: bool,
    fail_fast: bool,
    wandb_settings: run_benchmarks.WandbSettings,
) -> SweepArmResult:
    suite_path = materialize_arm_suite(spec, arm, output_root=output_root, overwrite=overwrite)
    output_dir = output_root / arm.arm_id / "output"
    run_name = f"{spec.sweep_id}-{arm.arm_id}"
    try:
        result = run_benchmarks.run_or_plan(
            suite_path,
            output=output_dir,
            overwrite=overwrite,
            dry_run=dry_run,
            export=export,
            fail_fast=fail_fast,
            wandb_settings=_arm_wandb_settings(wandb_settings, spec=spec, arm=arm, run_name=run_name),
        )
    except Exception as exc:  # noqa: BLE001 -- keep sweeping other arms and report failure
        return _arm_error(arm, suite_path=suite_path, output_dir=output_dir, run_name=run_name, error=str(exc))
    return _arm_result(arm, suite_path=suite_path, output_dir=output_dir, run_name=run_name, result=result)


def _arm_wandb_settings(
    settings: run_benchmarks.WandbSettings,
    *,
    spec: SweepSpec,
    arm: SweepArm,
    run_name: str,
) -> run_benchmarks.WandbSettings:
    return settings.model_copy(
        update={
            "wandb_group": settings.wandb_group or spec.sweep_id,
            "wandb_job_type": settings.wandb_job_type or "benchmark-sweep",
            "wandb_run_name": settings.wandb_run_name or run_name,
            "wandb_tags": _joined_tags(
                settings.effective_wandb_tags, ["sweep", f"sweep:{spec.sweep_id}", f"arm:{arm.arm_id}"]
            ),
        }
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
    )


def _arm_error(arm: SweepArm, *, suite_path: Path, output_dir: Path, run_name: str, error: str) -> SweepArmResult:
    return SweepArmResult(
        arm_id=arm.arm_id,
        parameters=arm.parameters,
        suite_path=str(suite_path),
        output_dir=str(output_dir),
        wandb_run_name=run_name,
        status="error",
        error=error,
    )


def _maybe_create_report(
    spec: SweepSpec,
    *,
    wandb_settings: run_benchmarks.WandbSettings,
    create_report: bool,
    dry_run: bool,
) -> str | None:
    if not create_report or dry_run or not wandb_settings.enabled:
        return None
    project = WandbProjectPath(
        entity=wandb_settings.wandb_entity or "",
        project=wandb_settings.effective_wandb_project,
    )
    if not project.entity:
        return None
    result = create_benchmark_group_report(project, group=wandb_settings.wandb_group or spec.sweep_id)
    return result.report_url


def _joined_tags(first: list[str], second: list[str]) -> str:
    return ",".join(dict.fromkeys([*first, *second]))


def _default_output_root(spec: SweepSpec, sweep_path: Path) -> Path:
    if spec.output_root:
        return _resolve_path(spec.output_root, sweep_path.parent)
    return sweep_path.with_suffix("").with_name(spec.sweep_id)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return raw


def _safe_param_name(path: str) -> str:
    return path.replace("*", "all").replace(".", "_").replace("-", "_")


def _rebase_suite_paths(suite: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    rebased = copy.deepcopy(suite)
    for key in ("model_configs", "model_providers", "artifact_path"):
        if isinstance(rebased.get(key), str):
            rebased[key] = str(_resolve_path(rebased[key], base_dir))
    for workload in rebased.get("workloads", []):
        if isinstance(workload, dict) and isinstance(workload.get("source"), str):
            workload["source"] = _rebase_source(workload["source"], base_dir)
    return rebased


def _rebase_source(source: str, base_dir: Path) -> str:
    if run_benchmarks.is_remote_input_source(source):
        return source
    return str(_resolve_path(source, base_dir))


def render_result(result: SweepResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [f"Ran {result.completed_arms}/{len(result.arms)} sweep arm(s); errors={result.errored_arms}"]
    lines.append(f"Output: {result.output_root}")
    if result.report_url:
        lines.append(f"Report: {result.report_url}")
    for arm in result.arms:
        lines.append(
            f"- {arm.arm_id}: {arm.status}, cases={arm.completed_cases}/{arm.completed_cases + arm.errored_cases}"
        )
    return "\n".join(lines)


def _resolve_cli_wandb_settings(
    *,
    wandb_mode: run_benchmarks.WandbMode | None,
    wandb_entity: str | None,
    wandb_project: str | None,
    wandb_group: str | None,
    wandb_job_type: str | None,
    wandb_tags: str | None,
    wandb_log_tables: bool | None,
) -> run_benchmarks.WandbSettings:
    return run_benchmarks.resolve_wandb_settings(
        wandb_mode=wandb_mode,
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        wandb_job_type=wandb_job_type,
        wandb_tags=wandb_tags,
        wandb_log_tables=wandb_log_tables,
    )


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
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
    wandb_mode: Annotated[run_benchmarks.WandbMode | None, cyclopts.Parameter("--wandb-mode")] = None,
    wandb_entity: Annotated[str | None, cyclopts.Parameter("--wandb-entity")] = None,
    wandb_project: Annotated[str | None, cyclopts.Parameter("--wandb-project")] = None,
    wandb_group: Annotated[str | None, cyclopts.Parameter("--wandb-group")] = None,
    wandb_job_type: Annotated[str | None, cyclopts.Parameter("--wandb-job-type")] = None,
    wandb_tags: Annotated[str | None, cyclopts.Parameter("--wandb-tags")] = None,
    wandb_log_tables: Annotated[bool | None, cyclopts.Parameter("--wandb-log-tables")] = None,
) -> None:
    configure_logging(log_format)
    try:
        result = run_sweep(
            spec,
            output_root=output_root,
            overwrite=overwrite,
            dry_run=dry_run,
            export=export,
            fail_fast=fail_fast,
            wandb_settings=_resolve_cli_wandb_settings(
                wandb_mode=wandb_mode,
                wandb_entity=wandb_entity,
                wandb_project=wandb_project,
                wandb_group=wandb_group,
                wandb_job_type=wandb_job_type,
                wandb_tags=wandb_tags,
                wandb_log_tables=wandb_log_tables,
            ),
            create_report=create_report,
        )
    except (ValueError, ValidationError) as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")
    if result.errored_arms:
        raise SystemExit(1)


if __name__ == "__main__":
    app()
