# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compile benchmark sweep specifications into materialized benchmark suites."""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any, cast

import yaml

from anonymizer.config.anonymizer_config import is_remote_input_source
from measurement_tools.benchmark_sweep_models import SweepArm, SweepSpec
from measurement_tools.wandb_models import SweepMetadata


def load_sweep_spec(path: Path) -> SweepSpec:
    raw = read_yaml_mapping(path)
    spec = SweepSpec.model_validate(raw)
    values = spec.model_dump()
    values["base_suite"] = str(resolve_path(spec.base_suite, path.parent))
    return SweepSpec.model_validate(values)


def expand_sweep_arms(spec: SweepSpec) -> list[SweepArm]:
    names = list(spec.parameters)
    value_grid = itertools.product(*(spec.parameters[name] for name in names))
    return [
        SweepArm(arm_id=f"arm-{index:03d}", parameters=dict(zip(names, values, strict=True)))
        for index, values in enumerate(value_grid)
    ]


def materialize_arm_suite(spec: SweepSpec, arm: SweepArm, *, output_root: Path, overwrite: bool) -> Path:
    patched = arm_suite_payload(spec, arm)
    arm_dir = output_root / arm.arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    suite_path = arm_dir / "suite.yaml"
    if suite_path.exists() and not overwrite:
        raise ValueError(f"sweep arm suite already exists: {suite_path}")
    suite_path.write_text(yaml.safe_dump(patched, sort_keys=False), encoding="utf-8")
    return suite_path


def arm_suite_payload(spec: SweepSpec, arm: SweepArm) -> dict[str, Any]:
    base_path = Path(spec.base_suite)
    suite = rebase_suite_paths(read_yaml_mapping(base_path), base_path.parent)
    return patched_suite(suite, spec=spec, arm=arm)


def patched_suite(suite: dict[str, Any], *, spec: SweepSpec, arm: SweepArm) -> dict[str, Any]:
    patched = copy.deepcopy(suite)
    run_tags = dict(patched.get("run_tags") or {})
    run_tags.update(sweep_run_tags(spec, arm))
    patched["run_tags"] = run_tags
    for path, value in arm.parameters.items():
        apply_parameter(patched, path, value)
    return patched


def sweep_run_tags(spec: SweepSpec, arm: SweepArm) -> dict[str, Any]:
    sweep = SweepMetadata.from_arm(
        sweep_id=spec.sweep_id,
        arm_id=arm.arm_id,
        params={safe_param_name(path): value for path, value in arm.parameters.items()},
    )
    return {"wandb_sweep": sweep.model_dump(mode="json")}


def apply_parameter(suite: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if len(parts) >= 3 and parts[0] == "configs":
        apply_config_parameter(suite, parts[1], parts[2:], value)
        return
    set_nested(suite, parts, value)


def apply_config_parameter(suite: dict[str, Any], config_selector: str, parts: list[str], value: Any) -> None:
    configs = suite.get("configs")
    if not isinstance(configs, list):
        raise ValueError("base suite must define a configs list")
    matched = [config for config in configs if isinstance(config, dict) and config_matches(config, config_selector)]
    if not matched:
        raise ValueError(f"sweep parameter references unknown config selector: {config_selector}")
    for config in matched:
        set_nested(config, parts, value)


def config_matches(config: dict[str, Any], selector: str) -> bool:
    return selector == "*" or config.get("id") == selector


def set_nested(target: dict[str, Any], parts: list[str], value: Any) -> None:
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
        current = cast(dict[str, Any], existing)
    current[parts[-1]] = value


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def read_yaml_mapping(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return raw


def safe_param_name(path: str) -> str:
    return path.replace("*", "all").replace(".", "_").replace("-", "_")


def rebase_suite_paths(suite: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    rebased = copy.deepcopy(suite)
    for key in ("model_configs", "model_providers"):
        if isinstance(rebased.get(key), str) and is_yaml_file_reference(rebased[key]):
            rebased[key] = str(resolve_path(rebased[key], base_dir))
    if isinstance(rebased.get("artifact_path"), str):
        rebased["artifact_path"] = str(resolve_path(rebased["artifact_path"], base_dir))
    for workload in rebased.get("workloads", []):
        if isinstance(workload, dict) and isinstance(workload.get("source"), str):
            workload["source"] = rebase_source(workload["source"], base_dir)
    return rebased


def is_yaml_file_reference(value: str) -> bool:
    return "\n" not in value and Path(value).suffix.lower() in {".yaml", ".yml"}


def rebase_source(source: str, base_dir: Path) -> str:
    if is_remote_input_source(source):
        return source
    return str(resolve_path(source, base_dir))


__all__ = [
    "apply_config_parameter",
    "apply_parameter",
    "arm_suite_payload",
    "config_matches",
    "expand_sweep_arms",
    "is_yaml_file_reference",
    "load_sweep_spec",
    "materialize_arm_suite",
    "patched_suite",
    "read_yaml_mapping",
    "rebase_source",
    "rebase_suite_paths",
    "resolve_path",
    "safe_param_name",
    "set_nested",
    "sweep_run_tags",
]
