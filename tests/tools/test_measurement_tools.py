# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

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


def test_benchmark_preflight_rejects_missing_text_column(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_preflight_input", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"body": ["Alice works at Acme"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: bad-input-suite
workloads:
  - id: biography
    source: input.csv
    text_column: text
configs:
  - id: redact
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="workload 'biography' text_column 'text' not found"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_rejects_bad_model_alias_references(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_preflight_models", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: bad-model-suite
model_configs: |
  selected_models:
    detection:
      entity_detector: detector
      entity_validator: [validator]
      entity_augmenter: augmenter
    replace:
      replacement_generator: missing-replacer
  model_configs:
    - alias: detector
      model: test/detector
    - alias: validator
      model: test/validator
    - alias: augmenter
      model: test/augmenter
workloads:
  - id: biography
    source: input.csv
configs:
  - id: substitute
    replace: substitute
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="missing-replacer"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_rejects_bad_provider_config(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_preflight_providers", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_path, index=False)
    provider_path = tmp_path / "providers.yaml"
    provider_path.write_text("not_providers: []\n", encoding="utf-8")
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: bad-provider-suite
model_providers: providers.yaml
workloads:
  - id: biography
    source: input.csv
configs:
  - id: redact
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="providers"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_accepts_provider_config_path(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_preflight_provider_path", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_path, index=False)
    provider_path = tmp_path / "providers.yaml"
    provider_path.write_text(
        """
providers:
  - name: test-provider
    endpoint: https://example.com/v1
    provider_type: openai
    api_key: TEST_API_KEY
""",
        encoding="utf-8",
    )
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: provider-path-suite
model_providers: providers.yaml
workloads:
  - id: biography
    source: input.csv
configs:
  - id: redact
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_case_passes_dd_trace_config_to_measurement_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool("measurement_benchmark_tool_dd_trace", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    captured: list[Any] = []

    @contextmanager
    def fake_measurement_session(config: Any) -> Iterator[None]:
        captured.append(config)
        yield None

    class FakeAnonymizer:
        def run(self, *, config: Any, data: Any) -> None:
            assert config.replace is not None
            assert data.text_column == "text"

    monkeypatch.setattr(tool, "configured_measurement_session", fake_measurement_session)

    spec = tool.BenchmarkSpec(
        suite_id="trace-suite",
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[tool.ConfigSpec(id="redact", replace="redact")],
    )
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(tmp_path / "input.csv", index=False)
    case = tool.BenchmarkCase(
        suite_id="trace-suite",
        workload_id="input",
        config_id="redact",
        repetition=0,
        case_id="input__redact__r000",
    )
    trace_path = tmp_path / "traces" / "input__redact__r000.jsonl"

    tool._execute_case(
        FakeAnonymizer(),
        spec.workloads[0],
        spec.configs[0],
        raw_path=tmp_path / "raw" / "input__redact__r000.jsonl",
        trace_path=trace_path,
        case=case,
        spec=spec,
        base_dir=tmp_path,
        dd_trace=tool.DDTraceMode.all_messages,
    )

    assert len(captured) == 1
    assert captured[0].dd_trace == "all_messages"
    assert captured[0].dd_trace_path == trace_path
    assert captured[0].streaming is True
    assert captured[0].keep_records is False
