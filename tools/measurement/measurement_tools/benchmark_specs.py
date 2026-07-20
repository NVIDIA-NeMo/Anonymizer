# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark suite loading, expansion, output preparation, and preflight."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import yaml
from data_designer.config.models import ModelProvider
from data_designer.config.utils.io_helpers import load_config_file

from anonymizer.config.anonymizer_config import AnonymizerInput, infer_input_source_suffix, is_remote_input_source
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.engine.ndd.model_loader import parse_model_configs, validate_model_alias_references
from measurement_tools.benchmark_inputs import (
    build_anonymizer_config,
    is_local_input_source,
    resolve_config_source,
    resolve_input_source,
    workload_has_row_slice,
)
from measurement_tools.benchmark_models import BenchmarkCase, BenchmarkSpec, MatrixEntry, WorkloadSpec


def load_spec(path: Path) -> BenchmarkSpec:
    if not path.exists() or path.is_dir():
        raise ValueError(f"spec path is not a file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark spec must be a YAML mapping")
    return BenchmarkSpec.model_validate(raw)


def build_cases(spec: BenchmarkSpec) -> list[BenchmarkCase]:
    matrix = spec.matrix or cross_product_matrix(spec)
    return [
        BenchmarkCase(
            suite_id=spec.suite_id,
            workload_id=entry.workload,
            config_id=entry.config,
            repetition=repetition,
            case_id=f"{entry.workload}__{entry.config}__r{repetition:03d}",
        )
        for entry in matrix
        for repetition in range(entry.repetitions)
    ]


def cross_product_matrix(spec: BenchmarkSpec) -> list[MatrixEntry]:
    return [
        MatrixEntry(workload=workload.id, config=config.id, repetitions=1)
        for workload in spec.workloads
        for config in spec.configs
    ]


def prepare_output_dir(output_dir: Path, *, overwrite: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"output path exists and is not a directory: {output_dir}")
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        elif any(output_dir.iterdir()):
            raise ValueError(f"output directory is not empty: {output_dir}; pass --overwrite to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)


def preflight_suite(spec: BenchmarkSpec, *, spec_path: Path) -> None:
    """Validate cheap suite inputs before any benchmark case consumes model time."""
    base_dir = spec_path.parent
    errors: list[str] = []
    parsed_models = preflight_model_configs(spec, base_dir=base_dir, errors=errors)

    preflight_model_providers_with_errors(spec, base_dir=base_dir, errors=errors)
    errors.extend(preflight_workload_errors(spec, base_dir=base_dir))
    errors.extend(preflight_config_errors(spec, parsed_models=parsed_models))
    if errors:
        raise ValueError("Benchmark preflight failed:\n- " + "\n- ".join(errors))


def preflight_model_configs(spec: BenchmarkSpec, *, base_dir: Path, errors: list[str]) -> Any | None:
    try:
        return parse_model_configs(resolve_config_source(spec.model_configs, base_dir))
    except Exception as exc:
        errors.append(f"model_configs invalid: {exc}")
        return None


def preflight_model_providers_with_errors(
    spec: BenchmarkSpec,
    *,
    base_dir: Path,
    errors: list[str],
) -> None:
    try:
        preflight_model_providers(spec, base_dir=base_dir)
    except Exception as exc:
        errors.append(f"model_providers invalid: {exc}")


def preflight_workload_errors(spec: BenchmarkSpec, *, base_dir: Path) -> list[str]:
    errors: list[str] = []
    for workload in spec.workloads:
        try:
            preflight_workload(workload, base_dir=base_dir)
        except Exception as exc:
            errors.append(str(exc))
    return errors


def preflight_config_errors(spec: BenchmarkSpec, *, parsed_models: Any | None) -> list[str]:
    errors: list[str] = []
    active_ids = active_config_ids(spec)
    for config in spec.configs:
        if config.id not in active_ids:
            continue
        try:
            anonymizer_config = build_anonymizer_config(config)
        except Exception as exc:
            errors.append(f"config '{config.id}' invalid: {exc}")
            continue
        if parsed_models is None:
            continue
        try:
            validate_model_alias_references(
                parsed_models.model_configs,
                parsed_models.selected_models,
                check_substitute=isinstance(anonymizer_config.replace, Substitute)
                or anonymizer_config.rewrite is not None,
                check_rewrite=anonymizer_config.rewrite is not None,
                check_evaluate=config.evaluate,
            )
        except ValueError as exc:
            errors.append(f"config '{config.id}' model aliases invalid: {exc}")
    return errors


def active_config_ids(spec: BenchmarkSpec) -> set[str]:
    if spec.matrix is None:
        return {config.id for config in spec.configs}
    return {entry.config for entry in spec.matrix}


def preflight_model_providers(spec: BenchmarkSpec, *, base_dir: Path) -> None:
    raw = resolve_config_source(spec.model_providers, base_dir)
    if raw is None:
        return
    if "\n" in raw:
        config_dict = yaml.safe_load(raw)
    else:
        candidate = Path(raw.strip()).expanduser()
        if candidate.suffix in (".yaml", ".yml"):
            if not candidate.is_file():
                raise FileNotFoundError(f"Providers config file not found: {candidate}")
            config_dict = load_config_file(candidate)
        else:
            config_dict = yaml.safe_load(raw)
    raw_providers = config_dict.get("providers") if isinstance(config_dict, dict) else None
    if not isinstance(raw_providers, list):
        raise ValueError("model_providers YAML must contain a top-level 'providers' list.")
    for provider in raw_providers:
        ModelProvider.model_validate(provider)


def preflight_workload(workload: WorkloadSpec, *, base_dir: Path) -> None:
    resolved_source = resolve_input_source(workload.source, base_dir)
    if workload_has_row_slice(workload) and not is_local_input_source(str(resolved_source)):
        raise ValueError(f"workload '{workload.id}' row slicing requires a local workload source")
    input_data = AnonymizerInput(
        source=str(resolved_source),
        text_column=workload.text_column,
        id_column=workload.id_column,
        data_summary=workload.data_summary,
    )
    columns = input_columns(input_data.source)
    if columns is None:
        return
    if workload.text_column not in columns:
        raise ValueError(
            f"workload '{workload.id}' text_column '{workload.text_column}' not found in {input_data.source}; "
            f"available columns: {sorted(columns)}"
        )
    if workload.id_column is not None and workload.id_column not in columns:
        raise ValueError(
            f"workload '{workload.id}' id_column '{workload.id_column}' not found in {input_data.source}; "
            f"available columns: {sorted(columns)}"
        )


def input_columns(source: str) -> set[str] | None:
    suffix = infer_input_source_suffix(source)
    if suffix not in SUPPORTED_IO_FORMATS:
        supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
        raise ValueError(f"Unsupported input format: {suffix}. Use {supported_formats}.")
    if is_remote_input_source(source):
        return None
    if suffix == ".csv":
        return set(pd.read_csv(source, nrows=0).columns)
    return set(pq.ParquetFile(source).schema_arrow.names)


__all__ = [
    "active_config_ids",
    "build_cases",
    "cross_product_matrix",
    "input_columns",
    "load_spec",
    "preflight_config_errors",
    "preflight_model_configs",
    "preflight_model_providers",
    "preflight_model_providers_with_errors",
    "preflight_suite",
    "preflight_workload",
    "preflight_workload_errors",
    "prepare_output_dir",
]
