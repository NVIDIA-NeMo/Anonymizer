# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark input resolution, slicing, and Anonymizer configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    Detect,
    Rewrite,
    infer_input_source_suffix,
)
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT, DEFAULT_PROTECT_TEXT, PrivacyGoal
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from measurement_tools.benchmark_models import ConfigSpec, ReplaceKind, ReplaceSpec, RewriteSpec, WorkloadSpec


def build_input(
    workload: WorkloadSpec,
    base_dir: Path,
    *,
    slice_dir: Path | None = None,
    case_id: str | None = None,
) -> AnonymizerInput:
    resolved_source = resolve_input_source(workload.source, base_dir)
    source = (
        materialize_sliced_source(workload, resolved_source, slice_dir=slice_dir, case_id=case_id)
        if workload_has_row_slice(workload)
        else resolved_source
    )
    return AnonymizerInput(
        source=str(source),
        text_column=workload.text_column,
        id_column=workload.id_column,
        data_summary=workload.data_summary,
    )


def workload_has_row_slice(workload: WorkloadSpec) -> bool:
    return workload.row_limit is not None or workload.row_offset > 0


def is_local_input_source(source: str) -> bool:
    return "://" not in source


def materialize_sliced_source(
    workload: WorkloadSpec,
    source: str | Path,
    *,
    slice_dir: Path | None,
    case_id: str | None,
) -> Path:
    if not is_local_input_source(str(source)):
        raise ValueError(f"workload '{workload.id}' row slicing requires a local workload source")
    if slice_dir is None or case_id is None:
        raise ValueError("row slicing requires slice_dir and case_id")
    source_path = Path(source)
    suffix = infer_input_source_suffix(str(source_path))
    dataframe = read_local_input_dataframe(source_path, suffix=suffix)
    sliced = dataframe.iloc[slice_bounds(workload)]
    slice_dir.mkdir(parents=True, exist_ok=True)
    destination = slice_dir / f"{safe_case_filename(case_id)}{suffix}"
    write_local_input_dataframe(sliced, destination, suffix=suffix)
    return destination


def slice_bounds(workload: WorkloadSpec) -> slice:
    start = workload.row_offset
    stop = start + workload.row_limit if workload.row_limit is not None else None
    return slice(start, stop)


def read_local_input_dataframe(source: Path, *, suffix: str) -> pd.DataFrame:
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix == ".parquet":
        return pd.read_parquet(source)
    supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
    raise ValueError(f"Unsupported input format: {suffix}. Use {supported_formats}.")


def write_local_input_dataframe(dataframe: pd.DataFrame, destination: Path, *, suffix: str) -> None:
    if suffix == ".csv":
        dataframe.to_csv(destination, index=False)
        return
    if suffix == ".parquet":
        dataframe.to_parquet(destination, index=False)
        return
    supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
    raise ValueError(f"Unsupported input format: {suffix}. Use {supported_formats}.")


def safe_case_filename(case_id: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in case_id)


def build_anonymizer_config(config: ConfigSpec) -> AnonymizerConfig:
    detect = Detect.model_validate(config.detect)
    if config.replace is not None:
        return AnonymizerConfig(
            detect=detect, replace=build_replace(config.replace), emit_telemetry=config.emit_telemetry
        )
    return AnonymizerConfig(detect=detect, rewrite=build_rewrite(config.rewrite), emit_telemetry=config.emit_telemetry)


def build_replace(raw: str | ReplaceSpec) -> Redact | Hash | Annotate | Substitute:
    spec = ReplaceSpec(strategy=ReplaceKind(raw)) if isinstance(raw, str) else raw
    if spec.strategy == ReplaceKind.redact:
        return Redact(**present({"format_template": spec.format_template, "normalize_label": spec.normalize_label}))
    if spec.strategy == ReplaceKind.hash:
        return Hash(
            **present(
                {
                    "format_template": spec.format_template,
                    "algorithm": spec.algorithm,
                    "digest_length": spec.digest_length,
                }
            )
        )
    if spec.strategy == ReplaceKind.annotate:
        return Annotate(**present({"format_template": spec.format_template}))
    return Substitute(**present({"instructions": spec.instructions}))


def build_rewrite(spec: RewriteSpec | None) -> Rewrite:
    if spec is None:
        raise ValueError("rewrite config is missing")
    privacy_goal_value = privacy_goal(spec)
    return Rewrite(
        privacy_goal=privacy_goal_value,
        instructions=spec.instructions,
        risk_tolerance=spec.risk_tolerance,
        max_repair_iterations=spec.max_repair_iterations,
        strict_entity_protection=spec.strict_entity_protection,
    )


def privacy_goal(spec: RewriteSpec) -> PrivacyGoal | None:
    if spec.protect is None and spec.preserve is None:
        return None
    return PrivacyGoal(
        protect=spec.protect or DEFAULT_PROTECT_TEXT,
        preserve=spec.preserve or DEFAULT_PRESERVE_TEXT,
    )


def present(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def resolve_input_source(source: str, base_dir: Path) -> str | Path:
    if "://" in source:
        return source
    return resolve_path(source, base_dir)


def resolve_optional_path(raw: str | None, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    return resolve_path(raw, base_dir)


def resolve_config_source(raw: str | None, base_dir: Path) -> str | None:
    if raw is None or "\n" in raw:
        return raw
    candidate = Path(raw).expanduser()
    if candidate.suffix in {".yaml", ".yml"}:
        return str(resolve_path(raw, base_dir))
    return raw


def resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else base_dir / path


__all__ = [
    "build_anonymizer_config",
    "build_input",
    "build_replace",
    "build_rewrite",
    "is_local_input_source",
    "materialize_sliced_source",
    "present",
    "privacy_goal",
    "read_local_input_dataframe",
    "resolve_config_source",
    "resolve_input_source",
    "resolve_optional_path",
    "resolve_path",
    "safe_case_filename",
    "slice_bounds",
    "workload_has_row_slice",
    "write_local_input_dataframe",
]
