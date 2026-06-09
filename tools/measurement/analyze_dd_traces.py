#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analyze DataDesigner message traces without emitting raw prompt or response text.

Usage:
    uv run python tools/measurement/analyze_dd_traces.py benchmark-runs/suite-id/traces
    uv run python tools/measurement/analyze_dd_traces.py benchmark-runs/suite-id/traces --output analysis --format csv
    uv run python tools/measurement/analyze_dd_traces.py benchmark-runs/suite-id/traces --json
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Annotated, Any

import cyclopts
import pandas as pd
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.stats import median_or_none as _median_or_none
from measurement_tools.stats import none_if_nan as _none_if_nan
from measurement_tools.stats import sum_int_or_zero as _sum_int_or_zero
from measurement_tools.tables import AnalysisExportResult, ExportFormat, ModelTableSpec
from measurement_tools.tables import write_analysis_tables as _write_analysis_table_specs
from pydantic import BaseModel, Field, computed_field

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.dd_traces")

JSON_FENCE_RE = re.compile(r"```\s*json\b", re.IGNORECASE)


class TraceAnalysisRow(BaseModel):
    trace_file: str
    line_number: int
    suite_id: str | None = None
    workload_id: str | None = None
    config_id: str | None = None
    case_id: str | None = None
    run_id: str
    workflow_name: str | None = None
    model_alias: str | None = None
    model_name: str | None = None
    model_provider_name: str | None = None
    status: str | None = None
    error_type: str | None = None
    elapsed_sec: float | None = None
    message_count: int = 0
    prompt_chars: int = 0
    response_shape: str
    response_chars: int = 0
    response_has_thinking: bool = False
    response_has_json_fence: bool = False
    response_has_embedded_json: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


class TraceGroupAnalysisRow(BaseModel):
    suite_id: str | None = None
    workload_id: str | None = None
    config_id: str | None = None
    workflow_name: str | None = None
    model_alias: str | None = None
    model_name: str | None = None
    model_provider_name: str | None = None
    status: str | None = None
    error_type: str | None = None
    response_shape: str
    trace_record_count: int
    error_count: int = 0
    thinking_count: int = 0
    json_fence_count: int = 0
    embedded_json_count: int = 0
    median_elapsed_sec: float | None = None
    median_prompt_chars: float | None = None
    median_response_chars: float | None = None
    sum_input_tokens: int = 0
    sum_output_tokens: int = 0
    sum_total_tokens: int = 0


class TraceAnalysis(BaseModel):
    trace_path: str
    rows: list[TraceAnalysisRow] = Field(default_factory=list)
    groups: list[TraceGroupAnalysisRow] = Field(default_factory=list)

    @computed_field
    @property
    def trace_record_count(self) -> int:
        return len(self.rows)

    @computed_field
    @property
    def group_count(self) -> int:
        return len(self.groups)


def analyze_trace_path(trace_path: Path) -> TraceAnalysis:
    rows = [
        _trace_row(record, trace_file=path, line_number=line_number)
        for path in _iter_trace_files(trace_path)
        for line_number, record in _iter_trace_records(path)
    ]
    return TraceAnalysis(
        trace_path=str(trace_path),
        rows=rows,
        groups=build_trace_group_rows(rows),
    )


def _iter_trace_files(trace_path: Path) -> list[Path]:
    if not trace_path.exists():
        raise ValueError(f"trace path does not exist: {trace_path}")
    if trace_path.is_file():
        return [trace_path]
    if not trace_path.is_dir():
        raise ValueError(f"trace path is not a file or directory: {trace_path}")
    return sorted(trace_path.rglob("*.jsonl"))


def _iter_trace_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict) and payload.get("record_type") == "dd_message_trace":
            records.append((line_number, payload))
    return records


def _trace_row(record: dict[str, Any], *, trace_file: Path, line_number: int) -> TraceAnalysisRow:
    response = record.get("response") if isinstance(record.get("response"), dict) else None
    response_content = _string_or_none(response.get("content")) if response is not None else None
    reasoning_content = _string_or_none(response.get("reasoning_content")) if response is not None else None
    run_tags = record.get("run_tags") if isinstance(record.get("run_tags"), dict) else {}
    usage = record.get("usage") if isinstance(record.get("usage"), dict) else {}
    return TraceAnalysisRow(
        trace_file=str(trace_file),
        line_number=line_number,
        suite_id=_string_or_none(run_tags.get("suite_id")),
        workload_id=_string_or_none(run_tags.get("workload_id")),
        config_id=_string_or_none(run_tags.get("config_id")),
        case_id=_string_or_none(run_tags.get("case_id")),
        run_id=str(record.get("run_id") or ""),
        workflow_name=_string_or_none(record.get("workflow_name")),
        model_alias=_string_or_none(record.get("model_alias")),
        model_name=_string_or_none(record.get("model_name")),
        model_provider_name=_string_or_none(record.get("model_provider_name")),
        status=_string_or_none(record.get("status")),
        error_type=_string_or_none(record.get("error_type")),
        elapsed_sec=_float_or_none(record.get("elapsed_sec")),
        message_count=_message_count(record.get("messages")),
        prompt_chars=_message_text_chars(record.get("messages")),
        response_shape=_response_shape(response_content),
        response_chars=len(response_content or ""),
        response_has_thinking=_has_thinking(response_content, reasoning_content),
        response_has_json_fence=_has_json_fence(response_content),
        response_has_embedded_json=_has_embedded_json(response_content),
        input_tokens=_int_or_none(usage.get("input_tokens")),
        output_tokens=_int_or_none(usage.get("output_tokens")),
        total_tokens=_int_or_none(usage.get("total_tokens")),
    )


def build_trace_group_rows(rows: list[TraceAnalysisRow]) -> list[TraceGroupAnalysisRow]:
    if not rows:
        return []
    table = pd.DataFrame([row.model_dump() for row in rows])
    group_columns = [
        "suite_id",
        "workload_id",
        "config_id",
        "workflow_name",
        "model_alias",
        "model_name",
        "model_provider_name",
        "status",
        "error_type",
        "response_shape",
    ]
    return [
        _build_trace_group_row(keys, group) for keys, group in table.groupby(group_columns, dropna=False, sort=True)
    ]


def _build_trace_group_row(keys: tuple[Any, ...], group: pd.DataFrame) -> TraceGroupAnalysisRow:
    (
        suite_id,
        workload_id,
        config_id,
        workflow_name,
        model_alias,
        model_name,
        provider_name,
        status,
        error_type,
        response_shape,
    ) = keys
    return TraceGroupAnalysisRow(
        suite_id=_none_if_nan(suite_id),
        workload_id=_none_if_nan(workload_id),
        config_id=_none_if_nan(config_id),
        workflow_name=_none_if_nan(workflow_name),
        model_alias=_none_if_nan(model_alias),
        model_name=_none_if_nan(model_name),
        model_provider_name=_none_if_nan(provider_name),
        status=_none_if_nan(status),
        error_type=_none_if_nan(error_type),
        response_shape=str(response_shape),
        trace_record_count=len(group),
        error_count=int((group["status"] == "error").sum()),
        thinking_count=int(group["response_has_thinking"].sum()),
        json_fence_count=int(group["response_has_json_fence"].sum()),
        embedded_json_count=int(group["response_has_embedded_json"].sum()),
        median_elapsed_sec=_median_or_none(group, "elapsed_sec"),
        median_prompt_chars=_median_or_none(group, "prompt_chars"),
        median_response_chars=_median_or_none(group, "response_chars"),
        sum_input_tokens=_sum_int_or_zero(group, "input_tokens"),
        sum_output_tokens=_sum_int_or_zero(group, "output_tokens"),
        sum_total_tokens=_sum_int_or_zero(group, "total_tokens"),
    )


def _response_shape(content: str | None) -> str:
    if not content:
        return "none"
    if _has_json_fence(content):
        return "fenced_json"
    stripped = content.strip()
    try:
        json.loads(stripped)
    except json.JSONDecodeError:
        return "embedded_json" if _has_embedded_json(content) else "text"
    return "raw_json"


def _has_thinking(content: str | None, reasoning_content: str | None) -> bool:
    return bool(reasoning_content) or "</think>" in (content or "")


def _has_json_fence(content: str | None) -> bool:
    return bool(content and JSON_FENCE_RE.search(content))


def _has_embedded_json(content: str | None) -> bool:
    if not content:
        return False
    decoder = json.JSONDecoder()
    for start, char in enumerate(content):
        if char not in "{[":
            continue
        try:
            parsed, _ = decoder.raw_decode(content, start)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict | list):
            return True
    return False


def _message_count(messages: object) -> int:
    return len(messages) if isinstance(messages, list) else 0


def _message_text_chars(messages: object) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(_message_content_chars(message) for message in messages)


def _message_content_chars(message: object) -> int:
    if not isinstance(message, dict):
        return 0
    return _content_chars(message.get("content"))


def _content_chars(content: object) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_item_chars(item) for item in content)
    return 0


def _content_item_chars(item: object) -> int:
    if isinstance(item, str):
        return len(item)
    if isinstance(item, dict):
        return len(str(item.get("text") or ""))
    return 0


def _string_or_none(value: object) -> str | None:
    return str(value) if value is not None and not pd.isna(value) else None


def _float_or_none(value: object) -> float | None:
    return float(value) if value is not None and not pd.isna(value) else None


def _int_or_none(value: object) -> int | None:
    return int(value) if value is not None and not pd.isna(value) else None


def write_analysis_tables(result: TraceAnalysis, output_dir: Path, export_format: ExportFormat) -> AnalysisExportResult:
    return _write_analysis_table_specs(
        output_dir,
        export_format,
        [
            ModelTableSpec("trace_analysis", result.rows, TraceAnalysisRow),
            ModelTableSpec("trace_group_analysis", result.groups, TraceGroupAnalysisRow),
        ],
    )


def render_result(result: TraceAnalysis, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [f"Analyzed {result.trace_record_count} trace record(s) across {result.group_count} group(s)"]
    for group in result.groups:
        model_label = group.model_alias or group.model_name
        label = f"{group.workload_id}/{group.config_id}/{group.workflow_name}/{model_label}/{group.response_shape}"
        lines.append(
            f"- {label}: traces={group.trace_record_count}, errors={group.error_count}, "
            f"thinking={group.thinking_count}, json_fence={group.json_fence_count}, "
            f"tokens={group.sum_total_tokens}"
        )
    return "\n".join(lines)


@app.default
def main(
    trace_path: Path,
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.parquet,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    try:
        result = analyze_trace_path(trace_path)
        if output is not None:
            write_analysis_tables(result, output, format)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
