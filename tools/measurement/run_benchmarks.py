#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Anonymizer benchmark suites and export measurement tables.

Usage:
    uv run python tools/measurement/run_benchmarks.py suite.yaml --output benchmark-runs/suite
    uv run python tools/measurement/run_benchmarks.py suite.yaml --dry-run --json
"""

import json
import logging
import shutil
import sys
import time
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import pandas as pd
import pyarrow.parquet as pq
import yaml
from data_designer.config.models import ModelProvider
from data_designer.config.utils.io_helpers import load_config_file
from export_measurements import ExportFormat, export_tables, read_measurements
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    AnonymizerInput,
    Detect,
    Rewrite,
    infer_input_source_suffix,
    is_remote_input_source,
)
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT, DEFAULT_PROTECT_TEXT, PrivacyGoal, RiskTolerance
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.engine.ndd.model_loader import parse_model_configs, validate_model_alias_references
from anonymizer.interface.anonymizer import Anonymizer
from anonymizer.measurement import MeasurementConfig, configured_measurement_session

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.benchmark")


class LogFormat(StrEnum):
    plain = "plain"
    json = "json"


_log_format = LogFormat.plain


class CaseStatus(StrEnum):
    planned = "planned"
    completed = "completed"
    error = "error"


class DDTraceMode(StrEnum):
    none = "none"
    last_message = "last_message"
    all_messages = "all_messages"


class ReplaceKind(StrEnum):
    redact = "redact"
    hash = "hash"
    annotate = "annotate"
    substitute = "substitute"


class WorkloadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    source: str
    text_column: str = "text"
    id_column: str | None = None
    data_summary: str | None = None


class ReplaceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: ReplaceKind
    format_template: str | None = None
    normalize_label: bool | None = None
    algorithm: str | None = None
    digest_length: int | None = None
    instructions: str | None = None


class RewriteSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protect: str | None = None
    preserve: str | None = None
    instructions: str | None = None
    risk_tolerance: RiskTolerance = RiskTolerance.low
    max_repair_iterations: int = 3
    strict_entity_protection: bool = False


class ConfigSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    detect: dict[str, Any] = Field(default_factory=dict)
    replace: str | ReplaceSpec | None = None
    rewrite: RewriteSpec | None = None
    emit_telemetry: bool = False

    @model_validator(mode="after")
    def validate_mode(self) -> "ConfigSpec":
        if self.replace is None and self.rewrite is None:
            raise ValueError("config must define replace or rewrite")
        if self.replace is not None and self.rewrite is not None:
            raise ValueError("config cannot define both replace and rewrite")
        return self


class MatrixEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workload: str
    config: str
    repetitions: int = Field(default=1, ge=1)


def _duplicates(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


class BenchmarkSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    suite_id: str
    model_configs: str | None = None
    model_providers: str | None = None
    artifact_path: str | None = None
    workloads: list[WorkloadSpec] = Field(min_length=1)
    configs: list[ConfigSpec] = Field(min_length=1)
    matrix: list[MatrixEntry] | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_ids(self) -> "BenchmarkSpec":
        workload_ids = [workload.id for workload in self.workloads]
        config_ids = [config.id for config in self.configs]
        if duplicate_workloads := _duplicates(workload_ids):
            raise ValueError(f"duplicate workload id(s): {', '.join(duplicate_workloads)}")
        if duplicate_configs := _duplicates(config_ids):
            raise ValueError(f"duplicate config id(s): {', '.join(duplicate_configs)}")
        self._validate_matrix_references(set(workload_ids), set(config_ids))
        return self

    def _validate_matrix_references(self, workload_ids: set[str], config_ids: set[str]) -> None:
        if self.matrix is None:
            return
        missing_workloads = sorted({entry.workload for entry in self.matrix} - workload_ids)
        missing_configs = sorted({entry.config for entry in self.matrix} - config_ids)
        if missing_workloads:
            raise ValueError(f"matrix references unknown workload id(s): {', '.join(missing_workloads)}")
        if missing_configs:
            raise ValueError(f"matrix references unknown config id(s): {', '.join(missing_configs)}")


class BenchmarkCase(BaseModel):
    suite_id: str
    workload_id: str
    config_id: str
    repetition: int
    case_id: str
    status: CaseStatus = CaseStatus.planned
    elapsed_sec: float | None = None
    measurement_path: str | None = None
    trace_path: str | None = None
    error: str | None = None


class BenchmarkResult(BaseModel):
    suite_id: str
    output_dir: str
    measurement_path: str
    summary_path: str
    table_dir: str | None
    cases: list[BenchmarkCase]


def configure_logging(log_format: LogFormat) -> None:
    global _log_format

    _log_format = log_format
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def log_bad_input(error: str) -> None:
    if _log_format == LogFormat.json:
        payload = {"level": "error", "event": "bad_input", "error": error}
        sys.stderr.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        return
    logger.error("bad_input error=%s", error)


def load_spec(path: Path) -> BenchmarkSpec:
    if not path.exists() or path.is_dir():
        raise ValueError(f"spec path is not a file: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark spec must be a YAML mapping")
    return BenchmarkSpec.model_validate(raw)


def build_cases(spec: BenchmarkSpec) -> list[BenchmarkCase]:
    matrix = spec.matrix or _cross_product_matrix(spec)
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


def _cross_product_matrix(spec: BenchmarkSpec) -> list[MatrixEntry]:
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
    parsed_models = None

    try:
        parsed_models = parse_model_configs(_resolve_config_source(spec.model_configs, base_dir))
    except Exception as exc:
        errors.append(f"model_configs invalid: {exc}")

    try:
        _preflight_model_providers(spec, base_dir=base_dir)
    except Exception as exc:
        errors.append(f"model_providers invalid: {exc}")

    for workload in spec.workloads:
        try:
            _preflight_workload(workload, base_dir=base_dir)
        except Exception as exc:
            errors.append(str(exc))

    for config in spec.configs:
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
            )
        except ValueError as exc:
            errors.append(f"config '{config.id}' model aliases invalid: {exc}")

    if errors:
        raise ValueError("Benchmark preflight failed:\n- " + "\n- ".join(errors))


def _preflight_model_providers(spec: BenchmarkSpec, *, base_dir: Path) -> None:
    raw = _resolve_config_source(spec.model_providers, base_dir)
    if raw is None:
        return
    config_source: str | Path = raw
    if isinstance(raw, str) and "\n" not in raw:
        candidate = Path(raw.strip()).expanduser()
        if candidate.suffix in (".yaml", ".yml"):
            if not candidate.is_file():
                raise FileNotFoundError(f"Providers config file not found: {candidate}")
            config_source = candidate
    config_dict = load_config_file(config_source)
    raw_providers = config_dict.get("providers") if isinstance(config_dict, dict) else None
    if not isinstance(raw_providers, list):
        raise ValueError("model_providers YAML must contain a top-level 'providers' list.")
    for provider in raw_providers:
        ModelProvider.model_validate(provider)


def _preflight_workload(workload: WorkloadSpec, *, base_dir: Path) -> None:
    resolved_source = _resolve_input_source(workload.source, base_dir)
    input_data = AnonymizerInput(
        source=str(resolved_source),
        text_column=workload.text_column,
        id_column=workload.id_column,
        data_summary=workload.data_summary,
    )
    columns = _input_columns(input_data.source)
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


def _input_columns(source: str) -> set[str] | None:
    suffix = infer_input_source_suffix(source)
    if suffix not in SUPPORTED_IO_FORMATS:
        supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
        raise ValueError(f"Unsupported input format: {suffix}. Use {supported_formats}.")
    if is_remote_input_source(source):
        return None
    if suffix == ".csv":
        return set(pd.read_csv(source, nrows=0).columns)
    return set(pq.ParquetFile(source).schema_arrow.names)


def run_suite(
    spec: BenchmarkSpec,
    *,
    spec_path: Path,
    output_dir: Path,
    export: bool,
    fail_fast: bool,
    dd_trace: DDTraceMode,
    trace_dir: Path | None,
) -> BenchmarkResult:
    contexts = _build_contexts(
        spec,
        spec_path=spec_path,
        output_dir=output_dir,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
    )
    anonymizer = Anonymizer(**contexts["anonymizer_kwargs"])
    cases = [
        _run_case(case, spec, contexts=contexts, anonymizer=anonymizer, fail_fast=fail_fast)
        for case in build_cases(spec)
    ]
    measurement_path = combine_measurements(cases, output_dir / "measurements.jsonl")
    should_export = export and measurement_path.stat().st_size > 0
    table_dir = export_measurement_tables(measurement_path, output_dir / "tables") if should_export else None
    result = BenchmarkResult(
        suite_id=spec.suite_id,
        output_dir=str(output_dir),
        measurement_path=str(measurement_path),
        summary_path=str(output_dir / "summary.json"),
        table_dir=str(table_dir) if table_dir is not None else None,
        cases=cases,
    )
    write_summary(result)
    return result


def _build_contexts(
    spec: BenchmarkSpec,
    *,
    spec_path: Path,
    output_dir: Path,
    dd_trace: DDTraceMode,
    trace_dir: Path | None,
) -> dict[str, Any]:
    base_dir = spec_path.parent
    artifact_path = _resolve_optional_path(spec.artifact_path, base_dir) or output_dir / "artifacts"
    return {
        "base_dir": base_dir,
        "workloads": {workload.id: workload for workload in spec.workloads},
        "configs": {config.id: config for config in spec.configs},
        "raw_dir": output_dir / "raw",
        "dd_trace": dd_trace,
        "trace_dir": trace_dir or output_dir / "traces",
        "anonymizer_kwargs": {
            "model_configs": _resolve_config_source(spec.model_configs, base_dir),
            "model_providers": _resolve_config_source(spec.model_providers, base_dir),
            "artifact_path": artifact_path,
        },
    }


def _run_case(
    case: BenchmarkCase,
    spec: BenchmarkSpec,
    *,
    contexts: dict[str, Any],
    anonymizer: Anonymizer,
    fail_fast: bool,
) -> BenchmarkCase:
    raw_path = contexts["raw_dir"] / f"{case.case_id}.jsonl"
    trace_path = _case_trace_path(case, contexts=contexts)
    started = time.perf_counter()
    try:
        workload = _get_item(contexts["workloads"], case.workload_id, "workload")
        config = _get_item(contexts["configs"], case.config_id, "config")
        _execute_case(
            anonymizer,
            workload,
            config,
            raw_path=raw_path,
            trace_path=trace_path,
            case=case,
            spec=spec,
            base_dir=contexts["base_dir"],
            dd_trace=contexts["dd_trace"],
        )
        return case.model_copy(
            update={
                "status": CaseStatus.completed,
                "elapsed_sec": time.perf_counter() - started,
                "measurement_path": str(raw_path),
                "trace_path": str(trace_path) if trace_path is not None else None,
            }
        )
    except Exception as exc:
        if fail_fast:
            raise
        return case.model_copy(
            update={
                "status": CaseStatus.error,
                "elapsed_sec": time.perf_counter() - started,
                "measurement_path": str(raw_path),
                "trace_path": str(trace_path) if trace_path is not None else None,
                "error": str(exc),
            }
        )


def _case_trace_path(case: BenchmarkCase, *, contexts: dict[str, Any]) -> Path | None:
    if contexts["dd_trace"] == DDTraceMode.none:
        return None
    return contexts["trace_dir"] / f"{case.case_id}.jsonl"


def _execute_case(
    anonymizer: Anonymizer,
    workload: WorkloadSpec,
    config: ConfigSpec,
    *,
    raw_path: Path,
    trace_path: Path | None,
    case: BenchmarkCase,
    spec: BenchmarkSpec,
    base_dir: Path,
    dd_trace: DDTraceMode,
) -> None:
    measurement = MeasurementConfig(
        output_path=raw_path,
        run_id=case.case_id,
        run_tags=_run_tags(case, spec),
        streaming=True,
        keep_records=False,
        dd_trace=dd_trace.value,
        dd_trace_path=trace_path,
        fail_on_write_error=True,
    )
    with configured_measurement_session(measurement):
        anonymizer.run(config=build_anonymizer_config(config), data=build_input(workload, base_dir))


def build_input(workload: WorkloadSpec, base_dir: Path) -> AnonymizerInput:
    return AnonymizerInput(
        source=str(_resolve_input_source(workload.source, base_dir)),
        text_column=workload.text_column,
        id_column=workload.id_column,
        data_summary=workload.data_summary,
    )


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
        return Redact(**_present({"format_template": spec.format_template, "normalize_label": spec.normalize_label}))
    if spec.strategy == ReplaceKind.hash:
        return Hash(
            **_present(
                {
                    "format_template": spec.format_template,
                    "algorithm": spec.algorithm,
                    "digest_length": spec.digest_length,
                }
            )
        )
    if spec.strategy == ReplaceKind.annotate:
        return Annotate(**_present({"format_template": spec.format_template}))
    return Substitute(**_present({"instructions": spec.instructions}))


def build_rewrite(spec: RewriteSpec | None) -> Rewrite:
    if spec is None:
        raise ValueError("rewrite config is missing")
    privacy_goal = _privacy_goal(spec)
    return Rewrite(
        privacy_goal=privacy_goal,
        instructions=spec.instructions,
        risk_tolerance=spec.risk_tolerance,
        max_repair_iterations=spec.max_repair_iterations,
        strict_entity_protection=spec.strict_entity_protection,
    )


def _privacy_goal(spec: RewriteSpec) -> PrivacyGoal | None:
    if spec.protect is None and spec.preserve is None:
        return None
    return PrivacyGoal(
        protect=spec.protect or DEFAULT_PROTECT_TEXT,
        preserve=spec.preserve or DEFAULT_PRESERVE_TEXT,
    )


def combine_measurements(cases: list[BenchmarkCase], destination: Path) -> Path:
    with destination.open("w", encoding="utf-8") as output:
        for case in cases:
            if case.measurement_path is None:
                continue
            source = Path(case.measurement_path)
            if source.exists():
                output.write(source.read_text(encoding="utf-8"))
    return destination


def export_measurement_tables(measurement_path: Path, table_dir: Path) -> Path:
    dataframe = read_measurements(measurement_path)
    export_tables(
        dataframe, input_path=measurement_path, output_dir=table_dir, export_format=ExportFormat.parquet, overwrite=True
    )
    return table_dir


def write_summary(result: BenchmarkResult) -> None:
    Path(result.summary_path).write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def render_result(result: BenchmarkResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    completed = sum(case.status == CaseStatus.completed for case in result.cases)
    errored = sum(case.status == CaseStatus.error for case in result.cases)
    planned = sum(case.status == CaseStatus.planned for case in result.cases)
    if planned and completed == 0 and errored == 0:
        return f"Planned {planned} case(s); output={result.output_dir}"
    return f"Ran {completed}/{len(result.cases)} case(s); errors={errored}; output={result.output_dir}"


def _run_tags(case: BenchmarkCase, spec: BenchmarkSpec) -> dict[str, Any]:
    return {
        "suite_id": spec.suite_id,
        "workload_id": case.workload_id,
        "config_id": case.config_id,
        "repetition": case.repetition,
        "case_id": case.case_id,
    }


def _present(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _get_item(items: dict[str, Any], item_id: str, item_type: str) -> Any:
    if item_id not in items:
        raise ValueError(f"unknown {item_type}: {item_id}")
    return items[item_id]


def _resolve_input_source(source: str, base_dir: Path) -> str | Path:
    if "://" in source:
        return source
    return _resolve_path(source, base_dir)


def _resolve_optional_path(raw: str | None, base_dir: Path) -> Path | None:
    if raw is None:
        return None
    return _resolve_path(raw, base_dir)


def _resolve_config_source(raw: str | None, base_dir: Path) -> str | None:
    if raw is None or "\n" in raw:
        return raw
    candidate = Path(raw).expanduser()
    if candidate.suffix in {".yaml", ".yml"}:
        return str(_resolve_path(raw, base_dir))
    return raw


def _resolve_path(raw: str, base_dir: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else base_dir / path


def dry_run_result(
    spec: BenchmarkSpec,
    *,
    output_dir: Path,
    export: bool,
    dd_trace: DDTraceMode,
    trace_dir: Path | None,
) -> BenchmarkResult:
    cases = build_cases(spec)
    if dd_trace != DDTraceMode.none:
        resolved_trace_dir = trace_dir or output_dir / "traces"
        cases = [
            case.model_copy(update={"trace_path": str(resolved_trace_dir / f"{case.case_id}.jsonl")}) for case in cases
        ]
    return BenchmarkResult(
        suite_id=spec.suite_id,
        output_dir=str(output_dir),
        measurement_path=str(output_dir / "measurements.jsonl"),
        summary_path=str(output_dir / "summary.json"),
        table_dir=str(output_dir / "tables") if export else None,
        cases=cases,
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
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = run_or_plan(
            spec,
            output=output,
            overwrite=overwrite,
            dry_run=dry_run,
            export=export,
            fail_fast=fail_fast,
            dd_trace=dd_trace,
            trace_dir=trace_dir,
        )
    except (ValueError, ValidationError) as exc:
        log_bad_input(str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")
    if any(case.status == CaseStatus.error for case in result.cases):
        raise SystemExit(1)


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
) -> BenchmarkResult:
    benchmark_spec = load_spec(spec_path)
    output_dir = output or Path("benchmark-runs") / benchmark_spec.suite_id
    if trace_dir is not None and dd_trace == DDTraceMode.none:
        raise ValueError("--trace-dir requires --dd-trace")
    if dry_run:
        return dry_run_result(
            benchmark_spec,
            output_dir=output_dir,
            export=export,
            dd_trace=dd_trace,
            trace_dir=trace_dir,
        )
    preflight_suite(benchmark_spec, spec_path=spec_path)
    prepare_output_dir(output_dir, overwrite=overwrite, dry_run=dry_run)
    return run_suite(
        benchmark_spec,
        spec_path=spec_path,
        output_dir=output_dir,
        export=export,
        fail_fast=fail_fast,
        dd_trace=dd_trace,
        trace_dir=trace_dir,
    )


if __name__ == "__main__":
    app()
