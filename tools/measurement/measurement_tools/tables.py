#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared table export helpers for measurement analysis tools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Sequence

import pandas as pd
from pydantic import BaseModel, Field


class ExportFormat(StrEnum):
    parquet = "parquet"
    csv = "csv"
    jsonl = "jsonl"


class TableSummary(BaseModel):
    table: str
    rows: int
    path: str


class AnalysisExportResult(BaseModel):
    output_dir: str
    format: ExportFormat
    tables: list[TableSummary] = Field(default_factory=list)
    manifest_path: str


@dataclass(frozen=True)
class ModelTableSpec:
    name: str
    rows: Sequence[BaseModel]
    row_model: type[BaseModel] | None = None


def write_analysis_tables(
    output_dir: Path,
    export_format: ExportFormat,
    specs: Sequence[ModelTableSpec],
) -> AnalysisExportResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = [
        write_model_rows(spec.rows, output_dir / f"{spec.name}.{export_format.value}", export_format, spec.row_model)
        for spec in specs
    ]
    export_result = AnalysisExportResult(
        output_dir=str(output_dir),
        format=export_format,
        tables=tables,
        manifest_path=str(output_dir / "manifest.json"),
    )
    Path(export_result.manifest_path).write_text(export_result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return export_result


def write_model_rows(
    rows: Sequence[BaseModel],
    path: Path,
    export_format: ExportFormat,
    row_model: type[BaseModel] | None = None,
) -> TableSummary:
    table = rows_to_table(rows, row_model)
    write_table(table, path, export_format)
    return TableSummary(table=path.stem, rows=len(table), path=str(path))


def rows_to_table(rows: Sequence[BaseModel], row_model: type[BaseModel] | None = None) -> pd.DataFrame:
    if rows:
        return pd.json_normalize([row.model_dump() for row in rows], sep=".")
    if row_model is None:
        return pd.DataFrame()
    return pd.DataFrame(columns=pd.Index(list(row_model.model_fields)))


def write_table(table: pd.DataFrame, path: Path, export_format: ExportFormat) -> None:
    if export_format == ExportFormat.parquet:
        table.to_parquet(path, index=False)
    elif export_format == ExportFormat.csv:
        table.to_csv(path, index=False)
    else:
        table.to_json(path, orient="records", lines=True)


def ensure_can_write(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise ValueError(f"output already exists: {path}; pass --overwrite to replace it")
