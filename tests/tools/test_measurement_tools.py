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
import yaml
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


def _minimal_case_contexts(tool: ModuleType, spec: Any, tmp_path: Path) -> dict[str, Any]:
    return {
        "base_dir": tmp_path,
        "workloads": {workload.id: workload for workload in spec.workloads},
        "configs": {config.id: config for config in spec.configs},
        "raw_dir": tmp_path / "raw",
        "dd_trace": tool.DDTraceMode.none,
        "trace_dir": tmp_path / "traces",
        "dd_task_trace": False,
        "task_trace_dir": tmp_path / "task-traces",
        "artifact_path": tmp_path / "artifacts",
    }


def _copy_biography_data(tmp_path: Path, filename: str = "input.csv") -> Path:
    source = REPO_ROOT / "docs" / "data" / "NVIDIA_synthetic_biographies.csv"
    destination = tmp_path / filename
    destination.write_bytes(source.read_bytes())
    return destination


def test_benchmark_spec_rejects_duplicate_matrix_entries() -> None:
    tool = load_tool("measurement_benchmark_tool_duplicate_matrix", REPO_ROOT / "tools/measurement/run_benchmarks.py")

    with pytest.raises(ValidationError, match="duplicate matrix workload/config entry"):
        tool.BenchmarkSpec(
            suite_id="duplicate-suite",
            workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
            configs=[tool.ConfigSpec(id="redact", replace="redact")],
            matrix=[
                tool.MatrixEntry(workload="input", config="redact"),
                tool.MatrixEntry(workload="input", config="redact"),
            ],
        )


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


def test_benchmark_exports_detection_artifact_analysis(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_artifact_analysis", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    artifact_root = tmp_path / "artifacts"
    parquet_dir = artifact_root / "entity-detection" / "parquet-files"
    parquet_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "_seed_entities_json": '[{"value":"Alice","label":"first_name","start_position":0,"end_position":5}]',
                "_augmented_entities": '{"entities":[{"value":"Alice","label":"first_name"}]}',
                "_detected_entities": (
                    '{"entities":[{"value":"Alice","label":"first_name",'
                    '"start_position":0,"end_position":5,"source":"detector"}]}'
                ),
            }
        ]
    ).to_parquet(parquet_dir / "batch_00000.parquet", index=False)
    output_path = tmp_path / "detection-artifacts.jsonl"

    result_path = tool.export_detection_artifact_analysis(artifact_root, output_path)

    assert result_path == output_path
    rows = [pd.read_json(output_path, lines=True).iloc[0].to_dict()]
    assert rows[0]["augmented_duplicate_seed_value_count"] == 1
    assert "Alice" not in output_path.read_text(encoding="utf-8")


def test_benchmark_detection_artifact_analysis_ignores_stale_artifacts(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_artifact_delta", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    artifact_root = tmp_path / "artifacts"
    stale_dir = artifact_root / "entity-detection-old" / "parquet-files"
    stale_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "_seed_entities_json": "[]",
                "_augmented_entities": '{"entities":[]}',
                "_detected_entities": '{"entities":[]}',
            }
        ]
    ).to_parquet(stale_dir / "batch_00000.parquet", index=False)
    snapshot = tool.snapshot_detection_artifacts(artifact_root)
    fresh_dir = artifact_root / "entity-detection-new" / "parquet-files"
    fresh_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "_seed_entities_json": "[]",
                "_augmented_entities": '{"entities":[]}',
                "_detected_entities": (
                    '{"entities":[{"value":"sk-test-AAAAAAAAAAAAAAAAAAAAAAAA",'
                    '"label":"api_key","start_position":0,"end_position":32,"source":"augmenter"}]}'
                ),
            }
        ]
    ).to_parquet(fresh_dir / "batch_00000.parquet", index=False)
    output_path = tmp_path / "detection-artifacts.jsonl"

    result_path = tool.export_detection_artifact_analysis(
        artifact_root,
        output_path,
        artifact_snapshot=snapshot,
    )

    assert result_path == output_path
    rows = pd.read_json(output_path, lines=True)
    assert rows["workflow_name"].tolist() == ["entity-detection-new"]
    assert rows["final_entity_count"].tolist() == [1]


def test_benchmark_case_detection_artifact_analysis_adds_case_metadata(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_artifact_case", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    artifact_root = tmp_path / "artifacts"
    parquet_dir = artifact_root / "entity-detection" / "parquet-files"
    parquet_dir.mkdir(parents=True)
    snapshot = tool.snapshot_detection_artifacts(artifact_root)
    pd.DataFrame(
        [
            {
                "_seed_entities_json": "[]",
                "_augmented_entities": '{"entities":[]}',
                "_detected_entities": (
                    '{"entities":[{"value":"sk-test-AAAAAAAAAAAAAAAAAAAAAAAA",'
                    '"label":"api_key","start_position":0,"end_position":32,"source":"detector"}]}'
                ),
            }
        ]
    ).to_parquet(parquet_dir / "batch_00000.parquet", index=False)
    case = tool.BenchmarkCase(
        suite_id="suite-a",
        workload_id="shell",
        config_id="rules",
        repetition=2,
        case_id="shell__rules__r002",
    )
    output_path = tmp_path / "raw" / "shell__rules__r002.detection-artifacts.jsonl"

    result_path = tool.export_case_detection_artifact_analysis(
        artifact_root,
        output_path,
        case=case,
        artifact_snapshot=snapshot,
    )

    assert result_path == output_path
    rows = pd.read_json(output_path, lines=True)
    assert rows["suite_id"].tolist() == ["suite-a"]
    assert rows["workload_id"].tolist() == ["shell"]
    assert rows["config_id"].tolist() == ["rules"]
    assert rows["case_id"].tolist() == ["shell__rules__r002"]
    assert rows["run_id"].tolist() == ["shell__rules__r002"]
    assert rows["repetition"].tolist() == [2]
    assert "sk-test" not in output_path.read_text(encoding="utf-8")


def test_run_suite_records_detection_artifact_analysis_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool("measurement_benchmark_tool_run_suite_artifact", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    spec = tool.BenchmarkSpec(
        suite_id="artifact-suite",
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[tool.ConfigSpec(id="redact", replace="redact")],
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    artifact_path = output_dir / "artifacts"
    analysis_path = output_dir / "detection-artifacts.jsonl"

    class FakeAnonymizer:
        def __init__(self, **kwargs: Any) -> None:
            assert kwargs["artifact_path"] == artifact_path

    def fake_run_case(case: Any, *_args: Any, **_kwargs: Any) -> Any:
        assert _kwargs["export_detection_artifacts"] is True
        raw_path = output_dir / "raw" / f"{case.case_id}.jsonl"
        raw_path.parent.mkdir()
        raw_path.write_text('{"record_type":"run","run_id":"case"}\n', encoding="utf-8")
        artifact_output_path = output_dir / "raw" / f"{case.case_id}.detection-artifacts.jsonl"
        artifact_output_path.write_text(
            '{"case_id":"input__redact__r000","workflow_name":"entity-detection"}\n',
            encoding="utf-8",
        )
        return case.model_copy(
            update={
                "status": tool.CaseStatus.completed,
                "measurement_path": str(raw_path),
                "detection_artifact_path": str(artifact_output_path),
            }
        )

    monkeypatch.setattr(tool, "Anonymizer", FakeAnonymizer)
    monkeypatch.setattr(tool, "_run_case", fake_run_case)
    monkeypatch.setattr(tool, "export_measurement_tables", lambda *_args: output_dir / "tables")

    result = tool.run_suite(
        spec,
        spec_path=tmp_path / "suite.yaml",
        output_dir=output_dir,
        export=True,
        fail_fast=False,
        dd_trace=tool.DDTraceMode.none,
        trace_dir=None,
    )

    assert result.detection_artifact_analysis_path == str(analysis_path)
    assert analysis_path.read_text(encoding="utf-8") == (
        '{"case_id":"input__redact__r000","workflow_name":"entity-detection"}\n'
    )
    summary = (output_dir / "summary.json").read_text(encoding="utf-8")
    assert "detection_artifact_analysis_path" in summary
    assert "detection_artifact_path" in summary


def test_run_suite_skips_detection_artifact_analysis_when_export_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_run_suite_no_export_artifact", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    spec = tool.BenchmarkSpec(
        suite_id="artifact-suite",
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[tool.ConfigSpec(id="redact", replace="redact")],
    )
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    class FakeAnonymizer:
        def __init__(self, **_kwargs: Any) -> None:
            pass

    def fake_run_case(case: Any, *_args: Any, **_kwargs: Any) -> Any:
        assert _kwargs["export_detection_artifacts"] is False
        raw_path = output_dir / "raw" / f"{case.case_id}.jsonl"
        raw_path.parent.mkdir()
        raw_path.write_text('{"record_type":"run","run_id":"case"}\n', encoding="utf-8")
        return case.model_copy(update={"status": tool.CaseStatus.completed, "measurement_path": str(raw_path)})

    monkeypatch.setattr(tool, "Anonymizer", FakeAnonymizer)
    monkeypatch.setattr(tool, "_run_case", fake_run_case)
    monkeypatch.setattr(
        tool,
        "combine_detection_artifact_analysis",
        lambda *_args: pytest.fail("artifact analysis should not be combined"),
    )

    result = tool.run_suite(
        spec,
        spec_path=tmp_path / "suite.yaml",
        output_dir=output_dir,
        export=False,
        fail_fast=False,
        dd_trace=tool.DDTraceMode.none,
        trace_dir=None,
    )

    assert result.detection_artifact_analysis_path is None
    assert not (output_dir / "detection-artifacts.jsonl").exists()


def test_benchmark_case_retries_transient_errors_and_records_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool("measurement_benchmark_tool_case_retry_success", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    attempts: list[Path] = []
    spec = tool.BenchmarkSpec(
        suite_id="retry-suite",
        case_retries=1,
        case_retry_backoff_sec=0,
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[tool.ConfigSpec(id="redact", replace="redact")],
    )
    case = tool.BenchmarkCase(
        suite_id="retry-suite",
        workload_id="input",
        config_id="redact",
        repetition=0,
        case_id="input__redact__r000",
    )
    pd.DataFrame({"text": ["Alice"]}).to_csv(tmp_path / "input.csv", index=False)

    def fake_execute_case(*_args: Any, raw_path: Path, **_kwargs: Any) -> None:
        attempts.append(raw_path)
        if len(attempts) == 1:
            raise RuntimeError("transient provider health check failure")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text('{"record_type":"run"}\n', encoding="utf-8")

    monkeypatch.setattr(tool, "_execute_case", fake_execute_case)

    result = tool._run_case(
        case,
        spec,
        contexts=_minimal_case_contexts(tool, spec, tmp_path),
        anonymizer=object(),
        fail_fast=False,
        export_detection_artifacts=False,
    )

    assert result.status == tool.CaseStatus.completed
    assert result.attempt_count == 2
    assert result.attempt_errors == ["transient provider health check failure"]
    assert attempts == [tmp_path / "raw" / "input__redact__r000.jsonl"] * 2


def test_benchmark_case_records_persistent_retry_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool("measurement_benchmark_tool_case_retry_failure", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    spec = tool.BenchmarkSpec(
        suite_id="retry-suite",
        case_retries=1,
        case_retry_backoff_sec=0,
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[tool.ConfigSpec(id="redact", replace="redact")],
    )
    case = tool.BenchmarkCase(
        suite_id="retry-suite",
        workload_id="input",
        config_id="redact",
        repetition=0,
        case_id="input__redact__r000",
    )

    attempts = 0
    errors: list[str] = []

    def fake_execute_case(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError(f"provider unavailable #{attempts}")

    def capture_error(case: Any, *, error: Exception, **kwargs: Any) -> Any:
        errors.append(str(error))
        return original_run_case_error(case, error=error, **kwargs)

    original_run_case_error = tool._run_case_error
    monkeypatch.setattr(tool, "_execute_case", fake_execute_case)
    monkeypatch.setattr(tool, "_run_case_error", capture_error)

    result = tool._run_case(
        case,
        spec,
        contexts=_minimal_case_contexts(tool, spec, tmp_path),
        anonymizer=object(),
        fail_fast=False,
        export_detection_artifacts=False,
    )

    assert result.status == tool.CaseStatus.error
    assert result.error == "provider unavailable #2"
    assert result.attempt_count == 2
    assert result.attempt_errors == ["provider unavailable #1", "provider unavailable #2"]
    assert errors == ["provider unavailable #2"]
    assert attempts == 2


def test_benchmark_case_fail_fast_skips_retries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_case_retry_fail_fast", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    spec = tool.BenchmarkSpec(
        suite_id="retry-suite",
        case_retries=3,
        case_retry_backoff_sec=0,
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[tool.ConfigSpec(id="redact", replace="redact")],
    )
    case = tool.BenchmarkCase(
        suite_id="retry-suite",
        workload_id="input",
        config_id="redact",
        repetition=0,
        case_id="input__redact__r000",
    )
    attempts = 0

    def fake_execute_case(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("fail fast")

    monkeypatch.setattr(tool, "_execute_case", fake_execute_case)

    with pytest.raises(RuntimeError, match="fail fast"):
        tool._run_case(
            case,
            spec,
            contexts=_minimal_case_contexts(tool, spec, tmp_path),
            anonymizer=object(),
            fail_fast=True,
            export_detection_artifacts=False,
        )

    assert attempts == 1


def test_combine_detection_artifact_analysis_separates_jsonl_chunks(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_combine_artifact_newlines",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"case_id":"one"}', encoding="utf-8")
    second.write_text('{"case_id":"two"}\n', encoding="utf-8")
    destination = tmp_path / "combined.jsonl"
    cases = [
        tool.BenchmarkCase(
            suite_id="suite",
            workload_id="input",
            config_id="first",
            repetition=0,
            case_id="first",
            detection_artifact_path=str(first),
        ),
        tool.BenchmarkCase(
            suite_id="suite",
            workload_id="input",
            config_id="second",
            repetition=0,
            case_id="second",
            detection_artifact_path=str(second),
        ),
    ]

    result = tool.combine_detection_artifact_analysis(cases, destination)

    assert result == destination
    assert destination.read_text(encoding="utf-8") == '{"case_id":"one"}\n{"case_id":"two"}\n'


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
    text_column: biography
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
    _copy_biography_data(tmp_path, "biographies.csv")
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


def test_build_input_materializes_sliced_csv_workload(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_sliced_input", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"id": ["a", "b", "c", "d"], "text": ["row-a", "row-b", "row-c", "row-d"]}).to_csv(
        input_path, index=False
    )
    workload = tool.WorkloadSpec(
        id="slice",
        source="input.csv",
        text_column="text",
        id_column="id",
        row_offset=1,
        row_limit=2,
    )

    anonymizer_input = tool.build_input(
        workload,
        tmp_path,
        slice_dir=tmp_path / "slices",
        case_id="slice__redact__r000",
    )

    assert anonymizer_input.text_column == "text"
    assert anonymizer_input.id_column == "id"
    assert Path(anonymizer_input.source) != input_path
    sliced = pd.read_csv(anonymizer_input.source)
    assert sliced.to_dict("records") == [
        {"id": "b", "text": "row-b"},
        {"id": "c", "text": "row-c"},
    ]


def test_benchmark_preflight_rejects_sliced_remote_workload(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_sliced_remote", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: bad-remote-slice
workloads:
  - id: remote
    source: s3://bucket/input.csv
    row_limit: 2
configs:
  - id: redact
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="row slicing requires a local workload source"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_rejects_bad_model_alias_references(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_preflight_models", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    _copy_biography_data(tmp_path)
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
    text_column: biography
configs:
  - id: substitute
    replace: substitute
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="missing-replacer"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_example_suites_are_portable() -> None:
    example_paths = sorted((REPO_ROOT / "tools/measurement/examples").glob("*.yaml"))
    assert example_paths

    machine_specific_fragments = (
        "/root/",
        "/Users/",
        "/stable-cache/",
        "gpu-dev-pod",
        "serve-svc",
    )
    path_fields = {"source", "model_configs", "model_providers", "artifact_path"}

    def walk(value: Any) -> Iterator[tuple[str, Any]]:
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key), item
                yield from walk(item)
        elif isinstance(value, list):
            for item in value:
                yield from walk(item)

    for example_path in example_paths:
        payload = yaml.safe_load(example_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)

        for key, value in walk(payload):
            if isinstance(value, str):
                assert not any(fragment in value for fragment in machine_specific_fragments), (
                    f"{example_path} contains machine-specific value for {key}: {value}"
                )
                if key in path_fields:
                    assert not Path(value).is_absolute(), f"{example_path} uses absolute path for {key}: {value}"
                if key in {"endpoint", "gliner_endpoint"}:
                    raise AssertionError(f"{example_path} should use endpoint_env for {key}, not a literal endpoint")


def test_benchmark_preflight_rejects_bad_provider_config(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_preflight_providers", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    _copy_biography_data(tmp_path)
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
    text_column: biography
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
    _copy_biography_data(tmp_path)
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
    text_column: biography
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
    task_trace_path = tmp_path / "task-traces" / "input__redact__r000.jsonl"

    tool._execute_case(
        FakeAnonymizer(),
        spec.workloads[0],
        spec.configs[0],
        raw_path=tmp_path / "raw" / "input__redact__r000.jsonl",
        trace_path=trace_path,
        task_trace_path=task_trace_path,
        case=case,
        spec=spec,
        base_dir=tmp_path,
        dd_trace=tool.DDTraceMode.all_messages,
    )

    assert len(captured) == 1
    assert captured[0].dd_trace == "all_messages"
    assert captured[0].dd_trace_path == trace_path
    assert captured[0].dd_task_trace_path == task_trace_path
    assert captured[0].streaming is True
    assert captured[0].keep_records is False
