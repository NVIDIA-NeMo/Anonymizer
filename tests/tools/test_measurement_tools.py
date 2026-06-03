# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
from pydantic import ValidationError

from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def test_export_measurements_groups_records_by_type(tmp_path: Path) -> None:
    tool = load_tool("measurement_export_tool", REPO_ROOT / "tools/measurement/export_measurements.py")
    dataframe = pd.DataFrame(
        [
            {"record_type": "run", "run_id": "case-a", "run_tags": {"suite_id": "suite-a"}},
            {"record_type": "stage", "run_id": "case-a", "stage": "detect", "metrics": {"rows": 2}},
        ]
    )

    result = tool.export_tables(
        dataframe,
        input_path=tmp_path / "measurements.jsonl",
        output_dir=tmp_path / "tables",
        export_format=tool.ExportFormat.csv,
        overwrite=False,
    )

    assert result.total_rows == 2
    assert {table.record_type for table in result.tables} == {"run", "stage"}
    assert (tmp_path / "tables/run.csv").exists()
    assert (tmp_path / "tables/stage.csv").exists()
    assert (tmp_path / "tables/manifest.json").exists()


def test_benchmark_spec_validates_matrix_references(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: bad-suite
workloads:
  - id: biography
    source: input.csv
configs:
  - id: redact
    replace: redact
matrix:
  - workload: missing
    config: redact
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unknown workload"):
        tool.load_spec(spec_path)


def test_benchmark_partial_rewrite_goal_uses_public_defaults() -> None:
    tool = load_tool("measurement_benchmark_tool_defaults", REPO_ROOT / "tools/measurement/run_benchmarks.py")

    rewrite = tool.build_rewrite(tool.RewriteSpec(protect="Direct payroll identifiers"))

    assert rewrite.privacy_goal.protect == "Direct payroll identifiers"
    assert rewrite.privacy_goal.preserve == DEFAULT_PRESERVE_TEXT


def test_benchmark_output_dir_requires_overwrite_for_existing_files(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_output", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    output_dir = tmp_path / "benchmark-output"
    output_dir.mkdir()
    existing = output_dir / "summary.json"
    existing.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="not empty"):
        tool.prepare_output_dir(output_dir, overwrite=False, dry_run=False)

    tool.prepare_output_dir(output_dir, overwrite=True, dry_run=False)

    assert (output_dir / "raw").is_dir()
    assert not existing.exists()


def test_benchmark_dry_run_expands_cases_without_writing(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_dry_run", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: smoke-suite
workloads:
  - id: biography
    source: biographies.csv
configs:
  - id: redact
    replace: redact
matrix:
  - workload: biography
    config: redact
    repetitions: 2
""",
        encoding="utf-8",
    )
    output_dir = tmp_path / "dry-run-output"

    result = tool.run_or_plan(
        spec_path,
        output=output_dir,
        overwrite=False,
        dry_run=True,
        export=False,
        fail_fast=False,
    )

    assert len(result.cases) == 2
    assert result.table_dir is None
    assert {case.status for case in result.cases} == {tool.CaseStatus.planned}
    assert not output_dir.exists()
