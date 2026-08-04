# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark measurement and detection-artifact exports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from analyze_detection_artifacts import analyze_artifacts, iter_detection_parquet_files
from export_measurements import export_tables, read_measurements

from measurement_tools.benchmark_models import BenchmarkCase, BenchmarkResult, CaseStatus
from measurement_tools.tables import ExportFormat


def combine_measurements(cases: list[BenchmarkCase], destination: Path) -> Path:
    with destination.open("w", encoding="utf-8") as output:
        for case in cases:
            if case.measurement_path is None:
                continue
            source = Path(case.measurement_path)
            if source.exists():
                output.write(source.read_text(encoding="utf-8"))
    return destination


def combine_detection_artifact_analysis(cases: list[BenchmarkCase], destination: Path) -> Path | None:
    chunks: list[str] = []
    for case in cases:
        if case.detection_artifact_path is None:
            continue
        source = Path(case.detection_artifact_path)
        if source.exists():
            chunks.append(jsonl_chunk(source.read_text(encoding="utf-8")))
    if not chunks:
        return None
    destination.write_text("".join(chunks), encoding="utf-8")
    return destination


def jsonl_chunk(text: str) -> str:
    if not text or text.endswith("\n"):
        return text
    return text + "\n"


def export_measurement_tables(measurement_path: Path, table_dir: Path) -> Path:
    dataframe = read_measurements(measurement_path)
    export_tables(
        dataframe, input_path=measurement_path, output_dir=table_dir, export_format=ExportFormat.parquet, overwrite=True
    )
    return table_dir


def snapshot_detection_artifacts(artifact_path: Path) -> dict[str, int]:
    if not artifact_path.exists():
        return {}
    return {
        str(parquet_file.relative_to(artifact_path)): parquet_file.stat().st_mtime_ns
        for parquet_file in iter_detection_parquet_files(artifact_path)
    }


def changed_detection_artifact_files(artifact_path: Path, snapshot: dict[str, int]) -> list[Path]:
    if not artifact_path.exists():
        return []
    changed: list[Path] = []
    for parquet_file in iter_detection_parquet_files(artifact_path):
        key = str(parquet_file.relative_to(artifact_path))
        if snapshot.get(key) != parquet_file.stat().st_mtime_ns:
            changed.append(parquet_file)
    return changed


def export_detection_artifact_analysis(
    artifact_path: Path,
    output_path: Path,
    *,
    artifact_snapshot: dict[str, int] | None = None,
) -> Path | None:
    if not artifact_path.exists():
        return None
    parquet_files = (
        changed_detection_artifact_files(artifact_path, artifact_snapshot) if artifact_snapshot is not None else None
    )
    analysis = analyze_artifacts(artifact_path, parquet_files=parquet_files)
    if not analysis.rows:
        return None
    write_detection_artifact_payloads([row.model_dump() for row in analysis.rows], output_path)
    return output_path


def export_case_detection_artifact_analysis(
    artifact_path: Path,
    output_path: Path,
    *,
    case: BenchmarkCase,
    artifact_snapshot: dict[str, int],
) -> Path | None:
    if not artifact_path.exists():
        return None
    parquet_files = changed_detection_artifact_files(artifact_path, artifact_snapshot)
    analysis = analyze_artifacts(artifact_path, parquet_files=parquet_files)
    if not analysis.rows:
        return None
    write_detection_artifact_payloads(
        [with_case_metadata(row.model_dump(), case=case) for row in analysis.rows],
        output_path,
    )
    return output_path


def with_case_metadata(row: dict[str, Any], *, case: BenchmarkCase) -> dict[str, Any]:
    return {
        "suite_id": case.suite_id,
        "workload_id": case.workload_id,
        "config_id": case.config_id,
        "repetition": case.repetition,
        "case_id": case.case_id,
        "run_id": case.case_id,
        **row,
    }


def write_detection_artifact_payloads(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.json_normalize(rows, sep=".").to_json(output_path, orient="records", lines=True)


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


__all__ = [
    "changed_detection_artifact_files",
    "combine_detection_artifact_analysis",
    "combine_measurements",
    "export_case_detection_artifact_analysis",
    "export_detection_artifact_analysis",
    "export_measurement_tables",
    "jsonl_chunk",
    "render_result",
    "snapshot_detection_artifacts",
    "with_case_metadata",
    "write_detection_artifact_payloads",
    "write_summary",
]
