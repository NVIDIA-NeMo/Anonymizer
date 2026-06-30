# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed historical and current W&B run views for benchmark reports."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping
from typing import Annotated, Any, Literal
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, StrictStr, ValidationError, field_validator

from measurement_tools.wandb_models import (
    BenchmarkMetadata,
    ConfigMetadata,
    DetectMetadata,
    ExecutionMetadata,
    GitMetadata,
    ImportedRunMetadata,
    MatrixMetadata,
    ModelSourcesMetadata,
    ReplaceMetadata,
    RewriteMetadata,
    RuntimeMetadata,
    SlurmMetadata,
    SweepMetadata,
    WandbRunMetadata,
    WorkloadMetadata,
    WorkloadSourceMetadata,
)

RemoteIdentifier = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")]
VisibleText = Annotated[StrictStr, Field(min_length=1, max_length=512)]
MetricValue = StrictInt | FiniteFloat


class WandbRunPath(BaseModel):
    entity: RemoteIdentifier
    project: RemoteIdentifier
    run_id: RemoteIdentifier
    base_url: StrictStr = "https://wandb.ai"

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _validate_base_url(value)

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"

    @property
    def url(self) -> str:
        segments = (self.entity, self.project, "runs", self.run_id)
        return f"{self.base_url}/{'/'.join(quote(segment, safe='') for segment in segments)}"


class WandbProjectPath(BaseModel):
    entity: RemoteIdentifier
    project: RemoteIdentifier
    base_url: StrictStr = "https://wandb.ai"

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        return _validate_base_url(value)

    @property
    def path(self) -> str:
        return f"{self.entity}/{self.project}"


class NativeSuiteAxis(BaseModel):
    kind: Literal["native_suite"] = "native_suite"
    suite_id: VisibleText

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SweepArmAxis(BaseModel):
    kind: Literal["sweep_arm"] = "sweep_arm"
    sweep_id: VisibleText
    arm_id: VisibleText

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ImportedConfigAxis(BaseModel):
    kind: Literal["imported_case"] = "imported_case"
    config_id: VisibleText

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


ComparisonAxis = Annotated[NativeSuiteAxis | SweepArmAxis | ImportedConfigAxis, Field(discriminator="kind")]


class WandbSummaryView(BaseModel):
    metrics: dict[StrictStr, MetricValue]

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class WandbRunView(BaseModel):
    schema_version: Literal[1, 2]
    path: WandbRunPath
    name: VisibleText
    group: VisibleText | None = None
    job_type: VisibleText | None = None
    metadata: WandbRunMetadata
    summary: WandbSummaryView
    comparison: ComparisonAxis

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class GroupComparison(BaseModel):
    run_kind: Literal["native_suite", "sweep_arm", "imported_case"]
    config_key: Literal["benchmark_suite_id", "native_suite_id", "sweep_arm_id", "imported_config_id"]
    label: StrictStr

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def parse_wandb_run_view(
    run: Any,
    *,
    run_path: WandbRunPath,
    allowed_metrics: frozenset[str],
) -> WandbRunView:
    """Project one remote SDK run into a strict v1 or v2 report view."""
    raw_config = getattr(run, "config", None)
    raw_summary = getattr(run, "summary", None)
    if not isinstance(raw_config, Mapping) or not isinstance(raw_summary, Mapping):
        raise ValueError("W&B run config and summary must be mappings")
    config = dict(raw_config)
    summary = dict(raw_summary)
    remote_id = getattr(run, "id", None)
    if remote_id is not None and remote_id != run_path.run_id:
        raise ValueError("W&B run identity does not match requested path")
    try:
        schema_version, metadata = _parse_metadata(config)
        name = _strict_optional_text(getattr(run, "name", None), default="benchmark run")
        if name is None:
            raise ValueError("W&B run name could not be resolved")
        return WandbRunView(
            schema_version=schema_version,
            path=run_path,
            name=name,
            group=_strict_optional_text(getattr(run, "group", None)),
            job_type=_strict_optional_text(getattr(run, "job_type", None)),
            metadata=metadata,
            summary=WandbSummaryView(metrics=_project_summary(summary, allowed_metrics=allowed_metrics)),
            comparison=_comparison_axis(metadata),
        )
    except ValidationError as exc:
        locations = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        summary = ", ".join(location for location in locations[:8] if location) or "run"
        raise ValueError(f"invalid W&B report data: schema violation at {summary}") from None


def validate_sweep_group(views: list[WandbRunView]) -> str:
    """Require a nonempty, unambiguous set of arms from one sweep."""
    if not views:
        raise ValueError("W&B group contains no benchmark runs")
    kinds = {view.comparison.kind for view in views}
    if kinds != {"sweep_arm"}:
        raise ValueError("W&B group must contain only sweep-arm runs")
    sweep_ids = {view.comparison.sweep_id for view in views if isinstance(view.comparison, SweepArmAxis)}
    if len(sweep_ids) != 1:
        raise ValueError("W&B group contains ambiguous sweep identities")
    return next(iter(sweep_ids))


def group_comparison(
    views: list[WandbRunView],
    *,
    expected_run_kind: Literal["native_suite", "sweep_arm", "imported_case"] | None = None,
) -> GroupComparison:
    """Select one explicit comparison axis for a homogeneous run group."""
    if not views:
        raise ValueError("W&B group contains no benchmark runs")
    kinds = {view.comparison.kind for view in views}
    if len(kinds) != 1:
        raise ValueError("W&B group contains mixed run kinds and has no unambiguous comparison axis")
    run_kind = next(iter(kinds))
    if expected_run_kind is not None and run_kind != expected_run_kind:
        raise ValueError(f"W&B group must contain only {expected_run_kind} runs")
    if run_kind == "sweep_arm":
        validate_sweep_group(views)
        return GroupComparison(run_kind=run_kind, config_key="sweep_arm_id", label="Sweep Arm")
    if run_kind == "imported_case":
        return GroupComparison(run_kind=run_kind, config_key="imported_config_id", label="Imported Config")
    key = "benchmark_suite_id" if any(view.schema_version == 1 for view in views) else "native_suite_id"
    return GroupComparison(run_kind=run_kind, config_key=key, label="Native Suite")


def parse_wandb_run_path(value: str) -> WandbRunPath:
    stripped = value.strip()
    parsed = urlsplit(stripped)
    if parsed.scheme or parsed.netloc:
        base_url, parts = _validated_url_parts(parsed, expected_tail="runs")
        if len(parts) != 4 or parts[2] != "runs":
            raise ValueError("W&B run URL must look like https://host/{entity}/{project}/runs/{run_id}")
        return WandbRunPath(entity=parts[0], project=parts[1], run_id=parts[3], base_url=base_url)
    parts = stripped.split("/")
    if len(parts) != 3 or not all(parts):
        raise ValueError("W&B run path must look like {entity}/{project}/{run_id}")
    return WandbRunPath(entity=parts[0], project=parts[1], run_id=parts[2])


def parse_wandb_project_path(value: str) -> WandbProjectPath:
    stripped = value.strip()
    parsed = urlsplit(stripped)
    if parsed.scheme or parsed.netloc:
        base_url, parts = _validated_url_parts(parsed)
        if len(parts) != 2:
            raise ValueError("W&B project URL must look like https://host/{entity}/{project}")
        return WandbProjectPath(entity=parts[0], project=parts[1], base_url=base_url)
    parts = stripped.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("W&B project path must look like {entity}/{project}")
    return WandbProjectPath(entity=parts[0], project=parts[1])


def validate_wandb_returned_url(value: Any, *, expected_base_url: str) -> str:
    """Validate an SDK-returned URL before exposing it in terminal or JSON output."""
    if not isinstance(value, str) or any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        raise ValueError("W&B SDK returned an unsafe URL")
    parsed = urlsplit(value)
    expected = urlsplit(expected_base_url)
    try:
        allowed_hosts = {expected.hostname}
        if expected.hostname == "api.wandb.ai":
            allowed_hosts.add("wandb.ai")
        same_origin = parsed.scheme == expected.scheme and parsed.hostname in allowed_hosts and parsed.port == expected.port
    except ValueError as exc:
        raise ValueError("W&B SDK returned an invalid URL") from exc
    if (
        not same_origin
        or parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("W&B SDK returned a URL outside the configured origin")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("W&B SDK returned an insecure non-loopback URL")
    return value


def _validated_url_parts(parsed: Any, *, expected_tail: str | None = None) -> tuple[str, list[str]]:
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("W&B URL must be a credential-free HTTP(S) URL without query or fragment")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("W&B URL must use HTTPS unless it targets loopback")
    parts = [part for part in parsed.path.split("/") if part]
    if expected_tail is not None and expected_tail not in parts:
        raise ValueError("W&B URL has an unsupported path")
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{parsed.hostname}{port}", parts


def _validate_base_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("W&B base URL must be a credential-free HTTP(S) origin")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("W&B base URL must use HTTPS unless it targets loopback")
    return value.rstrip("/")


def _parse_metadata(config: dict[str, Any]) -> tuple[Literal[1, 2], WandbRunMetadata]:
    benchmark = config.get("benchmark")
    version = benchmark.get("metadata_schema_version") if isinstance(benchmark, dict) else None
    if version not in {None, 1, 2} or isinstance(version, bool):
        raise ValueError("unsupported W&B metadata schema version")
    if version == 2:
        values = _project_v2_metadata(config)
        return 2, WandbRunMetadata.model_validate(values, strict=True)
    values = _project_v1_metadata(config)
    return 1, WandbRunMetadata.model_validate(values, strict=True)


def _project_v2_metadata(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("run_kind") not in {"native_suite", "sweep_arm", "imported_case"}:
        raise ValueError("v2 W&B metadata requires a declared run kind")
    return _project_metadata_fields(config, run_kind=config["run_kind"], legacy=False)


def _project_v1_metadata(config: dict[str, Any]) -> dict[str, Any]:
    sweep = config.get("sweep")
    has_sweep = isinstance(sweep, dict) or (
        isinstance(config.get("sweep_id"), str) and isinstance(config.get("sweep_arm_id"), str)
    )
    return _project_metadata_fields(config, run_kind="sweep_arm" if has_sweep else "native_suite", legacy=True)


def _project_metadata_fields(config: dict[str, Any], *, run_kind: str, legacy: bool) -> dict[str, Any]:
    values: dict[str, Any] = {"run_kind": run_kind}
    for name, model in (
        ("benchmark", BenchmarkMetadata),
        ("runtime", RuntimeMetadata),
        ("git", GitMetadata),
        ("model_sources", ModelSourcesMetadata),
        ("imported", ImportedRunMetadata),
    ):
        raw = config.get(name)
        if isinstance(raw, dict):
            values[name] = _project_fields(raw, model)
    execution = config.get("execution")
    if isinstance(execution, dict):
        values["execution"] = _project_execution(execution)
    values["workloads"] = [
        _project_workload(item, legacy=legacy) for item in _strict_list(config.get("workloads"), "workloads")
    ]
    values["configs"] = [_project_config(item) for item in _strict_list(config.get("configs"), "configs")]
    values["matrix"] = [_project_fields(item, MatrixMetadata) for item in _strict_list(config.get("matrix"), "matrix")]
    sweep = _project_sweep(config)
    if sweep is not None:
        values["sweep"] = sweep
    return values


def _project_workload(value: dict[str, Any], *, legacy: bool) -> dict[str, Any]:
    projected = _project_fields(value, WorkloadMetadata, exclude={"source"})
    raw_source = value.get("source")
    if isinstance(raw_source, dict):
        projected["source"] = _project_fields(raw_source, WorkloadSourceMetadata)
    elif legacy:
        kind = value.get("source_kind")
        suffix = value.get("source_suffix")
        if kind == "file":
            kind = "local_file"
        projected["source"] = {"kind": kind, "suffix": suffix}
    return projected


def _project_execution(value: dict[str, Any]) -> dict[str, Any]:
    projected = _project_fields(value, ExecutionMetadata, exclude={"slurm"})
    slurm = value.get("slurm")
    if isinstance(slurm, dict):
        projected["slurm"] = _project_fields(slurm, SlurmMetadata)
    return projected


def _project_config(value: dict[str, Any]) -> dict[str, Any]:
    projected = _project_fields(value, ConfigMetadata, exclude={"detect", "replace", "rewrite"})
    for name, model in (("detect", DetectMetadata), ("replace", ReplaceMetadata), ("rewrite", RewriteMetadata)):
        raw = value.get(name)
        if isinstance(raw, dict):
            projected[name] = _project_fields(raw, model)
    return projected


def _project_sweep(config: dict[str, Any]) -> dict[str, Any] | None:
    raw = config.get("sweep")
    if isinstance(raw, dict):
        return _project_fields(raw, SweepMetadata)
    sweep_id = config.get("sweep_id")
    arm_id = config.get("sweep_arm_id")
    if not isinstance(sweep_id, str) or not isinstance(arm_id, str):
        return None
    params = {
        key.removeprefix("sweep_param_"): value for key, value in config.items() if key.startswith("sweep_param_")
    }
    return {"id": sweep_id, "arm_id": arm_id, "params": params}


def _project_fields(
    value: dict[str, Any], model: type[BaseModel], *, exclude: set[str] | None = None
) -> dict[str, Any]:
    excluded = exclude or set()
    return {name: value[name] for name in model.model_fields if name in value and name not in excluded}


def _strict_list(value: Any, name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"W&B {name} metadata must be a list of mappings")
    return value


def _project_summary(summary: dict[str, Any], *, allowed_metrics: frozenset[str]) -> dict[str, MetricValue]:
    projected: dict[str, MetricValue] = {}
    for key in allowed_metrics:
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"W&B summary metric has invalid type or range: {key}")
        if key == "benchmark/case_success_rate" and value > 1:
            raise ValueError("W&B case success rate exceeds one")
        projected[key] = value
    return projected


def _comparison_axis(metadata: WandbRunMetadata) -> ComparisonAxis:
    if metadata.run_kind == "sweep_arm":
        if metadata.sweep is None:
            raise ValueError("sweep run is missing sweep metadata")
        return SweepArmAxis(sweep_id=metadata.sweep.id, arm_id=metadata.sweep.arm_id)
    if metadata.run_kind == "imported_case":
        config_ids = [item.id for item in metadata.configs if item.id is not None]
        if len(config_ids) != 1:
            raise ValueError("imported run requires exactly one config identity")
        return ImportedConfigAxis(config_id=config_ids[0])
    suite_id = metadata.benchmark.suite_id if metadata.benchmark is not None else None
    if suite_id is None:
        raise ValueError("native run requires a suite identity")
    return NativeSuiteAxis(suite_id=suite_id)


def _strict_optional_text(value: Any, *, default: str | None = None) -> str | None:
    if value is None or value == "":
        return default
    if not isinstance(value, str):
        raise ValueError("W&B run text metadata must be strings")
    if any(
        unicodedata.category(character) == "Cf"
        or (unicodedata.category(character) == "Cc" and character not in "\t\n\r")
        for character in value
    ):
        raise ValueError("W&B run text metadata contains unsafe control characters")
    return value
