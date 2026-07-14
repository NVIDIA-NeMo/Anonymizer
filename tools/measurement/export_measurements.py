#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export Anonymizer measurement JSONL into per-record-type tables.

Usage:
    uv run python tools/measurement/export_measurements.py measurements.jsonl --output tables
    uv run python tools/measurement/export_measurements.py measurements.jsonl -o tables --format csv
    uv run python tools/measurement/export_measurements.py measurements.jsonl -o tables --json
"""

import json
import logging
import sys
from pathlib import Path
from typing import Annotated, cast

import cyclopts
import pandas as pd
from measurement_tools.cli import LogFormat, configure_logging, log_bad_input
from measurement_tools.tables import ExportFormat, ensure_can_write, write_table
from pydantic import BaseModel, Field

app = cyclopts.App(help=__doc__)
logger = logging.getLogger("measurement.export")

MANIFEST_FILENAME = "manifest.json"


class TableSummary(BaseModel):
    record_type: str
    rows: int
    columns: int
    path: str


class ExportResult(BaseModel):
    input_path: str
    output_dir: str
    format: ExportFormat
    total_rows: int
    tables: list[TableSummary] = Field(default_factory=list)
    manifest_path: str


def read_measurements(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"input path does not exist: {path}")
    if path.is_dir():
        raise ValueError(f"input path is a directory: {path}")
    if path.suffix == ".json":
        dataframe = pd.read_json(path)
    else:
        dataframe = pd.read_json(path, lines=True)
    if "record_type" not in dataframe.columns:
        raise ValueError("measurement input must contain a record_type field")
    return dataframe


def normalize_table(rows: pd.DataFrame) -> pd.DataFrame:
    relevant_rows = rows.dropna(axis="columns", how="all")
    normalized = pd.json_normalize(relevant_rows.to_dict("records"), sep=".")
    for column in normalized.columns:
        if normalized[column].map(_is_nested_value).any():
            normalized[column] = normalized[column].map(_json_cell)
    return normalized


def _is_nested_value(value: object) -> bool:
    return isinstance(value, dict | list)


def _json_cell(value: object) -> object:
    if not _is_nested_value(value):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def export_tables(
    dataframe: pd.DataFrame,
    *,
    input_path: Path,
    output_dir: Path,
    export_format: ExportFormat,
    overwrite: bool,
) -> ExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        _export_one_table(
            cast(str, record_type),
            rows,
            output_dir=output_dir,
            export_format=export_format,
            overwrite=overwrite,
        )
        for record_type, rows in dataframe.groupby("record_type", sort=False)
    ]
    result = ExportResult(
        input_path=str(input_path),
        output_dir=str(output_dir),
        format=export_format,
        total_rows=len(dataframe),
        tables=tables,
        manifest_path=str(output_dir / MANIFEST_FILENAME),
    )
    write_manifest(result, output_dir / MANIFEST_FILENAME, overwrite=overwrite)
    return result


def _export_one_table(
    record_type: str,
    rows: pd.DataFrame,
    *,
    output_dir: Path,
    export_format: ExportFormat,
    overwrite: bool,
) -> TableSummary:
    table = normalize_table(rows)
    path = output_dir / f"{record_type}.{export_format.value}"
    ensure_can_write(path, overwrite=overwrite)
    write_table(table, path, export_format)
    return TableSummary(record_type=record_type, rows=len(table), columns=len(table.columns), path=str(path))


def write_manifest(result: ExportResult, path: Path, *, overwrite: bool) -> None:
    ensure_can_write(path, overwrite=overwrite)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")


def render_result(result: ExportResult, *, json_output: bool) -> str:
    if json_output:
        return result.model_dump_json(indent=2)
    lines = [f"Wrote {len(result.tables)} table(s) from {result.total_rows} measurement record(s)"]
    lines.append(f"Output: {result.output_dir}")
    for table in result.tables:
        lines.append(f"- {table.record_type}: {table.rows} rows, {table.columns} columns -> {table.path}")
    lines.append(f"Manifest: {result.manifest_path}")
    return "\n".join(lines)


@app.default
def main(
    input_path: Path,
    *,
    output: Annotated[Path | None, cyclopts.Parameter(("--output", "-o"))] = None,
    format: Annotated[ExportFormat, cyclopts.Parameter("--format")] = ExportFormat.parquet,
    overwrite: Annotated[bool, cyclopts.Parameter("--overwrite")] = False,
    json_output: Annotated[bool, cyclopts.Parameter("--json")] = False,
    log_format: Annotated[LogFormat, cyclopts.Parameter("--log-format")] = LogFormat.plain,
) -> None:
    configure_logging(log_format)
    output_dir = output or input_path.with_suffix("").with_name(f"{input_path.stem}-tables")
    try:
        dataframe = read_measurements(input_path)
        result = export_tables(
            dataframe,
            input_path=input_path,
            output_dir=output_dir,
            export_format=format,
            overwrite=overwrite,
        )
    except ValueError as exc:
        log_bad_input(logger, str(exc))
        raise SystemExit(125) from exc
    sys.stdout.write(render_result(result, json_output=json_output) + "\n")


if __name__ == "__main__":
    app()
