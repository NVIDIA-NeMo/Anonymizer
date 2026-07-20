# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ingestion, assembly, and table export for benchmark output analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from measurement_tools.benchmark_analysis_models import (
    BenchmarkOutputAnalysis,
    CaseAnalysisRow,
    GroupAnalysisRow,
    ModelUsageAnalysisRow,
    ModelUsageGroupAnalysisRow,
)
from measurement_tools.benchmark_case_analysis import build_case_row
from measurement_tools.benchmark_group_analysis import build_group_rows
from measurement_tools.benchmark_model_usage import build_model_usage_group_rows, build_model_usage_rows
from measurement_tools.tables import AnalysisExportResult, ExportFormat, ModelTableSpec
from measurement_tools.tables import write_analysis_tables as _write_analysis_table_specs


def analyze_benchmark_output(
    benchmark_dir: Path,
    *,
    detection_artifacts: Path | None = None,
) -> BenchmarkOutputAnalysis:
    measurements = read_jsonl_table(benchmark_dir / "measurements.jsonl", required=True)
    artifacts_path = detection_artifacts or benchmark_dir / "detection-artifacts.jsonl"
    artifacts = read_jsonl_table(artifacts_path, required=detection_artifacts is not None)
    traces = read_trace_summary_table(benchmark_dir / "traces")
    cases = [
        build_case_row(case_id, measurements, artifacts, traces)
        for case_id in case_ids(measurements, artifacts, traces)
    ]
    model_usage = build_model_usage_rows(measurements)
    return BenchmarkOutputAnalysis(
        benchmark_dir=str(benchmark_dir),
        detection_artifacts_path=str(artifacts_path) if not artifacts.empty else None,
        cases=cases,
        groups=build_group_rows(cases),
        model_usage=model_usage,
        model_usage_groups=build_model_usage_group_rows(model_usage),
    )


def read_jsonl_table(path: Path, *, required: bool) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise ValueError(f"input path does not exist: {path}")
        return pd.DataFrame()
    if path.is_dir():
        raise ValueError(f"input path is a directory: {path}")
    raw = pd.read_json(path, lines=True)
    if raw.empty:
        return raw
    return pd.json_normalize(raw.to_dict("records"), sep=".")


def read_trace_summary_table(trace_path: Path) -> pd.DataFrame:
    """Read DD trace files into a sanitized table with no prompt/response text."""
    if not trace_path.exists():
        return pd.DataFrame()
    if trace_path.is_file():
        paths = [trace_path]
    elif trace_path.is_dir():
        paths = sorted(trace_path.rglob("*.jsonl"))
    else:
        raise ValueError(f"trace path is not a file or directory: {trace_path}")

    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict) or record.get("record_type") != "dd_message_trace":
                continue
            run_tags = record.get("run_tags") if isinstance(record.get("run_tags"), dict) else {}
            rows.append(
                {
                    "record_type": "dd_message_trace",
                    "run_id": record.get("run_id"),
                    "run_tags.case_id": run_tags.get("case_id"),
                    "run_tags.workload_id": run_tags.get("workload_id"),
                    "run_tags.config_id": run_tags.get("config_id"),
                    "run_tags.experimental_detection_strategy": run_tags.get("experimental_detection_strategy"),
                    "run_tags.experimental_replacement_strategy": run_tags.get("experimental_replacement_strategy"),
                    "run_tags.dd_parser_compat": run_tags.get("dd_parser_compat"),
                    "run_tags.repetition": run_tags.get("repetition"),
                    "workflow_name": record.get("workflow_name"),
                    "model_alias": record.get("model_alias"),
                    "status": record.get("status"),
                    "error_type": record.get("error_type"),
                    "is_async": record.get("is_async"),
                }
            )
    return pd.DataFrame(rows)


def case_ids(*frames: pd.DataFrame) -> list[str]:
    values: set[str] = set()
    for dataframe in frames:
        for column in ("run_tags.case_id", "case_id", "run_id"):
            if column in dataframe.columns:
                values.update(str(value) for value in dataframe[column].dropna().tolist())
    return sorted(values)


def write_analysis_tables(
    result: BenchmarkOutputAnalysis,
    output_dir: Path,
    export_format: ExportFormat,
) -> AnalysisExportResult:
    return _write_analysis_table_specs(
        output_dir,
        export_format,
        [
            ModelTableSpec("case_analysis", result.cases, CaseAnalysisRow),
            ModelTableSpec("group_analysis", result.groups, GroupAnalysisRow),
            ModelTableSpec("model_analysis", result.model_usage, ModelUsageAnalysisRow),
            ModelTableSpec("model_group_analysis", result.model_usage_groups, ModelUsageGroupAnalysisRow),
        ],
    )


__all__ = [
    "analyze_benchmark_output",
    "case_ids",
    "read_jsonl_table",
    "read_trace_summary_table",
    "write_analysis_tables",
]
