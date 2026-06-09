#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze benchmark-only DD-free staged detection probe outputs.

Usage:
    uv run python tools/measurement/analyze_staged_detection_output.py /tmp/staged-probe
    uv run python tools/measurement/analyze_staged_detection_output.py /tmp/staged-probe/staged-detection-cases.jsonl
    uv run python tools/measurement/analyze_staged_detection_output.py /tmp/staged-probe --output analysis --format csv
    uv run python tools/measurement/analyze_staged_detection_output.py /tmp/staged-probe --json
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import pandas as pd
from pydantic import BaseModel, Field, computed_field

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.staged_detection_output")


class ExportFormat(StrEnum):
    parquet = "parquet"
    csv = "csv"
    jsonl = "jsonl"


class LogFormat(StrEnum):
    plain = "plain"
    json = "json"


_log_format = LogFormat.plain


class StagedCaseAnalysisRow(BaseModel):
    source_path: str
    case_id: str
    row_index: int | None = None
    seed_source: str | None = None
    status: str | None = None
    case_failed: bool = False
    elapsed_sec: float | None = None
    model_elapsed_sec: float | None = None
    model_phase_count: int = 0
    model_request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    seed_entity_count: int = 0
    validation_candidate_count: int = 0
    validation_decision_count: int = 0
    augmented_suggestion_count: int = 0
    final_entity_count: int = 0
    final_entity_signature_count: int = 0
    final_label_counts: dict[str, int] = Field(default_factory=dict)
    baseline_final_entity_signature_count: int | None = None
    shared_final_entity_signature_count: int | None = None
    baseline_only_final_entity_signature_count: int | None = None
    direct_only_final_entity_signature_count: int | None = None
    baseline_shared_signature_rate: float | None = None
    baseline_loss_signature_rate: float | None = None
    baseline_only_label_counts: dict[str, int] = Field(default_factory=dict)
    direct_only_label_counts: dict[str, int] = Field(default_factory=dict)
    error: str | None = None


class StagedGroupAnalysisRow(BaseModel):
    seed_source: str | None = None
    case_count: int
    completed_case_count: int = 0
    error_case_count: int = 0
    failed_case_rate: float | None = None
    elapsed_sec_sum: float | None = None
    elapsed_sec_mean: float | None = None
    model_elapsed_sec_sum: float | None = None
    model_elapsed_sec_mean: float | None = None
    model_phase_count_sum: int = 0
    model_request_count_sum: int = 0
    prompt_tokens_sum: int = 0
    completion_tokens_sum: int = 0
    total_tokens_sum: int = 0
    final_entity_count_sum: int = 0
    final_entity_signature_count_sum: int = 0
    baseline_final_entity_signature_count_sum: int = 0
    shared_final_entity_signature_count_sum: int = 0
    baseline_only_final_entity_signature_count_sum: int = 0
    direct_only_final_entity_signature_count_sum: int = 0
    baseline_shared_signature_rate: float | None = None
    baseline_loss_signature_rate: float | None = None


class LabelDeltaAnalysisRow(BaseModel):
    seed_source: str | None = None
    delta_type: str
    label: str
    count: int


class TableSummary(BaseModel):
    table: str
    rows: int
    path: str


class AnalysisExportResult(BaseModel):
    output_dir: str
    format: ExportFormat
    tables: list[TableSummary]
    manifest_path: str


class StagedDetectionOutputAnalysis(BaseModel):
    source_path: str
    cases: list[StagedCaseAnalysisRow] = Field(default_factory=list)
    groups: list[StagedGroupAnalysisRow] = Field(default_factory=list)
    label_deltas: list[LabelDeltaAnalysisRow] = Field(default_factory=list)

    @computed_field
    @property
    def case_count(self) -> int:
        return len(self.cases)

    @computed_field
    @property
    def group_count(self) -> int:
        return len(self.groups)

    @computed_field
    @property
    def label_delta_count(self) -> int:
        return len(self.label_deltas)


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


def analyze_staged_detection_output(input_path: Path) -> StagedDetectionOutputAnalysis:
    case_path = resolve_case_path(input_path)
    case_rows = [_build_case_row(row, source_path=case_path) for row in read_case_records(case_path)]
    return StagedDetectionOutputAnalysis(
        source_path=str(case_path),
        cases=case_rows,
        groups=build_group_rows(case_rows),
        label_deltas=build_label_delta_rows(case_rows),
    )


def resolve_case_path(input_path: Path) -> Path:
    if not input_path.exists():
        raise ValueError(f"input path does not exist: {input_path}")
    if input_path.is_dir():
        input_path = input_path / "staged-detection-cases.jsonl"
    if not input_path.exists():
        raise ValueError(f"staged detection case file does not exist: {input_path}")
    if input_path.is_dir():
        raise ValueError(f"input path is a directory: {input_path}")
    return input_path


def read_case_records(case_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line in case_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL row is not an object in {case_path}")
        if record.get("record_type") in (None, "staged_detection_case"):
            records.append(record)
    return records


def _build_case_row(record: dict[str, Any], *, source_path: Path) -> StagedCaseAnalysisRow:
    comparison = _dict_value(record.get("comparison"))
    baseline_count = _optional_int(comparison.get("baseline_final_entity_signature_count"))
    shared_count = _optional_int(comparison.get("shared_final_entity_signature_count"))
    baseline_only_count = _optional_int(comparison.get("baseline_only_final_entity_signature_count"))
    return StagedCaseAnalysisRow(
        source_path=str(source_path),
        case_id=str(record.get("case_id") or ""),
        row_index=_optional_int(record.get("row_index")),
        seed_source=_optional_str(record.get("seed_source")),
        status=_optional_str(record.get("status")),
        case_failed=str(record.get("status")).lower() == "error",
        elapsed_sec=_optional_float(record.get("elapsed_sec")),
        model_elapsed_sec=_optional_float(record.get("model_elapsed_sec")),
        model_phase_count=_int_value(record.get("model_phase_count")),
        model_request_count=_int_value(record.get("model_request_count")),
        **_usage_fields(_dict_value(record.get("total_usage"))),
        **_entity_count_fields(record),
        baseline_final_entity_signature_count=baseline_count,
        shared_final_entity_signature_count=shared_count,
        baseline_only_final_entity_signature_count=baseline_only_count,
        direct_only_final_entity_signature_count=_optional_int(
            comparison.get("direct_only_final_entity_signature_count")
        ),
        baseline_shared_signature_rate=_rate(shared_count, baseline_count),
        baseline_loss_signature_rate=_rate(baseline_only_count, baseline_count),
        baseline_only_label_counts=_counter_dict(comparison.get("baseline_only_label_counts")),
        direct_only_label_counts=_counter_dict(comparison.get("direct_only_label_counts")),
        error=_optional_str(record.get("error")),
    )


def _usage_fields(usage: dict[str, Any]) -> dict[str, int]:
    return {
        "prompt_tokens": _int_value(usage.get("prompt_tokens")),
        "completion_tokens": _int_value(usage.get("completion_tokens")),
        "total_tokens": _int_value(usage.get("total_tokens")),
    }


def _entity_count_fields(record: dict[str, Any]) -> dict[str, int | dict[str, int]]:
    return {
        "seed_entity_count": _int_value(record.get("seed_entity_count")),
        "validation_candidate_count": _int_value(record.get("validation_candidate_count")),
        "validation_decision_count": _int_value(record.get("validation_decision_count")),
        "augmented_suggestion_count": _int_value(record.get("augmented_suggestion_count")),
        "final_entity_count": _int_value(record.get("final_entity_count")),
        "final_entity_signature_count": _int_value(record.get("final_entity_signature_count")),
        "final_label_counts": _counter_dict(record.get("final_label_counts")),
    }


def build_group_rows(cases: list[StagedCaseAnalysisRow]) -> list[StagedGroupAnalysisRow]:
    groups: defaultdict[str | None, list[StagedCaseAnalysisRow]] = defaultdict(list)
    for case in cases:
        groups[case.seed_source].append(case)
    return [_build_group_row(seed_source, rows) for seed_source, rows in sorted(groups.items(), key=_group_sort_key)]


def _build_group_row(seed_source: str | None, rows: list[StagedCaseAnalysisRow]) -> StagedGroupAnalysisRow:
    case_count = len(rows)
    error_count = sum(1 for row in rows if row.case_failed)
    baseline_total = _sum_optional_int(rows, "baseline_final_entity_signature_count")
    shared_total = _sum_optional_int(rows, "shared_final_entity_signature_count")
    baseline_only_total = _sum_optional_int(rows, "baseline_only_final_entity_signature_count")
    model_request_count = sum(row.model_request_count for row in rows)
    return StagedGroupAnalysisRow(
        seed_source=seed_source,
        case_count=case_count,
        completed_case_count=case_count - error_count,
        error_case_count=error_count,
        failed_case_rate=_rate(error_count, case_count),
        elapsed_sec_sum=_sum_optional_float(rows, "elapsed_sec"),
        elapsed_sec_mean=_mean_optional_float(rows, "elapsed_sec"),
        model_elapsed_sec_sum=_sum_optional_float(rows, "model_elapsed_sec"),
        model_elapsed_sec_mean=_mean_optional_float(rows, "model_elapsed_sec"),
        model_phase_count_sum=sum(row.model_phase_count for row in rows),
        model_request_count_sum=model_request_count,
        prompt_tokens_sum=sum(row.prompt_tokens for row in rows),
        completion_tokens_sum=sum(row.completion_tokens for row in rows),
        total_tokens_sum=sum(row.total_tokens for row in rows),
        final_entity_count_sum=sum(row.final_entity_count for row in rows),
        final_entity_signature_count_sum=sum(row.final_entity_signature_count for row in rows),
        baseline_final_entity_signature_count_sum=baseline_total,
        shared_final_entity_signature_count_sum=shared_total,
        baseline_only_final_entity_signature_count_sum=baseline_only_total,
        direct_only_final_entity_signature_count_sum=_sum_optional_int(
            rows, "direct_only_final_entity_signature_count"
        ),
        baseline_shared_signature_rate=_rate(shared_total, baseline_total),
        baseline_loss_signature_rate=_rate(baseline_only_total, baseline_total),
    )


def _group_sort_key(item: tuple[str | None, list[StagedCaseAnalysisRow]]) -> str:
    return item[0] or ""


def build_label_delta_rows(cases: list[StagedCaseAnalysisRow]) -> list[LabelDeltaAnalysisRow]:
    counts: Counter[tuple[str | None, str, str]] = Counter()
    for case in cases:
        for label, count in case.baseline_only_label_counts.items():
            counts[(case.seed_source, "baseline_only", label)] += count
        for label, count in case.direct_only_label_counts.items():
            counts[(case.seed_source, "direct_only", label)] += count
    return [
        LabelDeltaAnalysisRow(seed_source=seed_source, delta_type=delta_type, label=label, count=count)
        for (seed_source, delta_type, label), count in sorted(counts.items(), key=_label_delta_sort_key)
    ]


def _label_delta_sort_key(item: tuple[tuple[str | None, str, str], int]) -> tuple[str, str, str]:
    (seed_source, delta_type, label), _ = item
    return (seed_source or "", delta_type, label)


def _sum_optional_float(rows: list[StagedCaseAnalysisRow], field_name: str) -> float | None:
    values = [value for value in (_optional_float(getattr(row, field_name)) for row in rows) if value is not None]
    return sum(values) if values else None


def _mean_optional_float(rows: list[StagedCaseAnalysisRow], field_name: str) -> float | None:
    values = [value for value in (_optional_float(getattr(row, field_name)) for row in rows) if value is not None]
    return sum(values) / len(values) if values else None


def _sum_optional_int(rows: list[StagedCaseAnalysisRow], field_name: str) -> int:
    return sum(value for value in (_optional_int(getattr(row, field_name)) for row in rows) if value is not None)


def _rate(numerator: object, denominator: object) -> float | None:
    numerator_value = _optional_float(numerator)
    denominator_value = _optional_float(denominator)
    if numerator_value is None or denominator_value is None or denominator_value <= 0:
        return None
    return numerator_value / denominator_value


def _dict_value(raw: object) -> dict[str, Any]:
    return raw if isinstance(raw, dict) else {}


def _counter_dict(raw: object) -> dict[str, int]:
    return {str(key): _int_value(value) for key, value in _dict_value(raw).items()}


def _optional_str(raw: object) -> str | None:
    return str(raw) if raw is not None else None


def _optional_float(raw: object) -> float | None:
    if raw is None:
        return None
    return float(raw)


def _optional_int(raw: object) -> int | None:
    if raw is None:
        return None
    return int(float(raw))


def _int_value(raw: object) -> int:
    return _optional_int(raw) or 0


def write_analysis_tables(
    result: StagedDetectionOutputAnalysis,
    output_dir: Path,
    export_format: ExportFormat,
) -> AnalysisExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        _write_model_rows(
            result.cases, output_dir / f"case_analysis.{export_format.value}", export_format, StagedCaseAnalysisRow
        ),
        _write_model_rows(
            result.groups,
            output_dir / f"group_analysis.{export_format.value}",
            export_format,
            StagedGroupAnalysisRow,
        ),
        _write_model_rows(
            result.label_deltas,
            output_dir / f"label_delta_analysis.{export_format.value}",
            export_format,
            LabelDeltaAnalysisRow,
        ),
    ]
    export_result = AnalysisExportResult(
        output_dir=str(output_dir),
        format=export_format,
        tables=tables,
        manifest_path=str(output_dir / "manifest.json"),
    )
    Path(export_result.manifest_path).write_text(export_result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return export_result


def _write_model_rows(
    rows: list[BaseModel],
    path: Path,
    export_format: ExportFormat,
    row_model: type[BaseModel],
) -> TableSummary:
    table = _rows_to_table(rows, row_model)
    if export_format == ExportFormat.parquet:
        table.to_parquet(path, index=False)
    elif export_format == ExportFormat.csv:
        table.to_csv(path, index=False)
    else:
        table.to_json(path, orient="records", lines=True)
    return TableSummary(table=path.stem, rows=len(table), path=str(path))


def _rows_to_table(rows: list[BaseModel], row_model: type[BaseModel]) -> pd.DataFrame:
    if rows:
        return pd.json_normalize([row.model_dump() for row in rows], sep=".")
    return pd.DataFrame(columns=list(row_model.model_fields))


def render_result(result: StagedDetectionOutputAnalysis, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [f"Analyzed {result.case_count} staged detection case(s) across {result.group_count} group(s)"]
    lines.extend(_render_group_line(group, result.label_deltas) for group in result.groups)
    return "\n".join(lines)


def _render_group_line(group: StagedGroupAnalysisRow, label_deltas: list[LabelDeltaAnalysisRow]) -> str:
    label = group.seed_source or "<unknown>"
    lost = _top_labels(label_deltas, seed_source=group.seed_source, delta_type="baseline_only")
    return (
        f"- {label}: cases={group.case_count}, errors={group.error_case_count}, "
        f"elapsed_sum={_fmt_float(group.elapsed_sec_sum)}s, "
        f"model_elapsed_sum={_fmt_float(group.model_elapsed_sec_sum)}s, "
        f"requests={group.model_request_count_sum}, tokens={group.total_tokens_sum}, "
        f"shared={group.shared_final_entity_signature_count_sum}/"
        f"{group.baseline_final_entity_signature_count_sum}, "
        f"baseline_only={group.baseline_only_final_entity_signature_count_sum}, "
        f"direct_only={group.direct_only_final_entity_signature_count_sum}, lost_labels={lost}"
    )


def _top_labels(label_deltas: list[LabelDeltaAnalysisRow], *, seed_source: str | None, delta_type: str) -> str:
    matches = [delta for delta in label_deltas if delta.seed_source == seed_source and delta.delta_type == delta_type]
    if not matches:
        return "{}"
    items = sorted(matches, key=lambda delta: (-delta.count, delta.label))[:8]
    return "{" + ", ".join(f"{item.label}:{item.count}" for item in items) + "}"


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


@app.default
def main(
    input_path: Path,
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.parquet,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = analyze_staged_detection_output(input_path)
        if output is not None:
            write_analysis_tables(result, output, format)
    except ValueError as exc:
        log_bad_input(str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
