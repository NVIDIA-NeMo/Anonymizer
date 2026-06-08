# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
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
from anonymizer.engine.constants import COL_FINAL_ENTITIES

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
        "dd_parser_compat": spec.dd_parser_compat,
        "artifact_path": tmp_path / "artifacts",
    }


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


def test_benchmark_exports_rules_only_synthetic_detection_artifacts(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_only_synthetic_artifacts",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    secret = "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"
    pd.DataFrame({"text": [f"export API_KEY={secret}"]}).to_csv(input_path, index=False)
    config = tool.ConfigSpec(
        id="rules-only-redact",
        replace="redact",
        detect={"entity_labels": ["api_key", "email", "password", "url"]},
        experimental_detection_strategy="rules_only",
    )
    case = tool.BenchmarkCase(
        suite_id="rules-suite",
        workload_id="input",
        config_id="rules-only-redact",
        repetition=0,
        case_id="input__rules-only-redact__r000",
    )
    output_path = tmp_path / "raw" / "input__rules-only-redact__r000.detection-artifacts.jsonl"

    result = tool.export_rules_only_case_detection_artifacts(
        config,
        tool.AnonymizerInput(source=str(input_path), text_column="text"),
        output_path,
        case=case,
    )

    assert result == output_path
    text = output_path.read_text(encoding="utf-8")
    assert secret not in text
    row = json.loads(text)
    assert row["workflow_name"] == "entity-detection-rules-only"
    assert row["final_entity_count"] == 1
    assert row["final_entity_signature_count"] == 1
    assert row["final_label_counts.api_key"] == 1
    assert row["final_source_counts.rule"] == 1
    assert any(key.startswith("final_entity_signature_labels.") for key in row)


def test_benchmark_exports_rules_covered_or_default_synthetic_artifacts_for_structured_fast_lane_labels(
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_covered_synthetic_artifacts",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    secret = "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"
    pd.DataFrame({"text": [f"export API_KEY={secret}"]}).to_csv(input_path, index=False)
    config = tool.ConfigSpec(
        id="rules-covered-redact",
        replace="redact",
        detect={"entity_labels": ["api_key", "email", "password", "url"]},
        experimental_detection_strategy="rules_covered_or_default",
    )
    case = tool.BenchmarkCase(
        suite_id="rules-suite",
        workload_id="input",
        config_id="rules-covered-redact",
        repetition=0,
        case_id="input__rules-covered-redact__r000",
    )
    output_path = tmp_path / "raw" / "input__rules-covered-redact__r000.detection-artifacts.jsonl"

    result = tool.export_rules_only_case_detection_artifacts(
        config,
        tool.AnonymizerInput(source=str(input_path), text_column="text"),
        output_path,
        case=case,
    )

    assert result == output_path
    row = json.loads(output_path.read_text(encoding="utf-8"))
    assert row["workflow_name"] == "entity-detection-rules-only"
    assert row["final_entity_count"] == 1
    assert row["final_label_counts.api_key"] == 1


def test_benchmark_does_not_export_rules_covered_or_default_artifacts_for_contextual_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_covered_contextual_artifacts",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice has token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(input_path, index=False)
    config = tool.ConfigSpec(
        id="rules-covered-redact",
        replace="redact",
        detect={"entity_labels": ["api_key", "person"]},
        experimental_detection_strategy="rules_covered_or_default",
    )
    case = tool.BenchmarkCase(
        suite_id="rules-suite",
        workload_id="input",
        config_id="rules-covered-redact",
        repetition=0,
        case_id="input__rules-covered-redact__r000",
    )

    result = tool.export_rules_only_case_detection_artifacts(
        config,
        tool.AnonymizerInput(source=str(input_path), text_column="text"),
        tmp_path / "raw" / "input__rules-covered-redact__r000.detection-artifacts.jsonl",
        case=case,
    )

    assert result is None


def test_benchmark_does_not_export_rules_covered_artifacts_for_narrow_prose_rule_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_covered_prose_rule_artifacts",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Jordan worked at Acme Research Center and lived on Maple Street."]}).to_csv(
        input_path,
        index=False,
    )
    config = tool.ConfigSpec(
        id="rules-covered-redact",
        replace="redact",
        detect={"entity_labels": ["organization_name", "street_address"]},
        experimental_detection_strategy="rules_covered_or_default",
    )
    case = tool.BenchmarkCase(
        suite_id="rules-suite",
        workload_id="input",
        config_id="rules-covered-redact",
        repetition=0,
        case_id="input__rules-covered-redact__r000",
    )

    result = tool.export_rules_only_case_detection_artifacts(
        config,
        tool.AnonymizerInput(source=str(input_path), text_column="text"),
        tmp_path / "raw" / "input__rules-covered-redact__r000.detection-artifacts.jsonl",
        case=case,
    )

    assert result is None


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


def _stale_detection_artifact_payload() -> dict[str, Any]:
    return {
        "workflow_name": "entity-detection",
        "batch_file": "entity-detection/parquet-files/batch_00000.parquet",
        "row_index": 0,
        "seed_entity_count": 1,
        "seed_validation_candidate_count": 1,
        "merged_validation_candidate_count": 1,
        "augmented_entity_count": 0,
        "final_entity_count": 1,
        "augmented_duplicate_seed_value_count": 0,
        "augmented_new_value_count": 0,
        "augmented_new_final_value_count": 0,
        "weak_api_key_shape_count": 0,
        "final_entity_signature_count": 1,
        "final_entity_signature_hashes": ["stale"],
        "final_entity_signature_labels": {"stale": "person"},
        "weak_api_key_shape_label_counts": {},
        "final_label_counts": {"person": 1},
        "final_source_counts": {"detector": 1},
    }


def _final_trace_dataframe_with_rule_entity() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_FINAL_ENTITIES: [
                {
                    "entities": [
                        {
                            "value": "Alice",
                            "label": "person",
                            "start_position": 0,
                            "end_position": 5,
                            "source": "detector",
                        },
                        {
                            "value": "1990",
                            "label": "date_of_birth",
                            "start_position": 25,
                            "end_position": 29,
                            "source": "rule",
                        },
                    ]
                }
            ]
        }
    )


def test_benchmark_patches_detection_artifacts_from_final_trace_dataframe(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_patch_result_artifacts",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    output_path = tmp_path / "raw" / "case.detection-artifacts.jsonl"
    tool.write_detection_artifact_payloads([_stale_detection_artifact_payload()], output_path)

    result = tool.patch_case_detection_artifacts_from_trace_dataframe(
        output_path,
        _final_trace_dataframe_with_rule_entity(),
    )

    assert result == output_path
    text = output_path.read_text(encoding="utf-8")
    assert "Alice" not in text
    assert "1990" not in text
    row = json.loads(text)
    assert row["final_entity_count"] == 2
    assert row["final_entity_signature_count"] == 2
    assert row["final_label_counts.person"] == 1
    assert row["final_label_counts.date_of_birth"] == 1
    assert row["final_source_counts.detector"] == 1
    assert row["final_source_counts.rule"] == 1
    assert any(key.startswith("final_entity_signature_details.") for key in row)
    assert "final_entity_signature_labels.stale" not in row


def test_rules_covered_or_default_detection_artifacts_use_final_trace_dataframe(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_covered_trace_artifacts",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    case = tool.BenchmarkCase(
        suite_id="suite-a",
        workload_id="input",
        config_id="rules-covered",
        repetition=0,
        case_id="input__rules-covered__r000",
    )
    config = tool.ConfigSpec(
        id="rules-covered",
        replace="redact",
        detect={"entity_labels": ["api_key", "password"]},
        experimental_detection_strategy="rules_covered_or_default",
    )
    trace_dataframe = pd.DataFrame(
        {
            COL_FINAL_ENTITIES: [
                {
                    "entities": [
                        {
                            "value": "sk-test-AAAAAAAAAAAAAAAAAAAAAAAA",
                            "label": "api_key",
                            "start_position": 6,
                            "end_position": 38,
                            "source": "rule",
                        }
                    ]
                },
                {
                    "entities": [
                        {
                            "value": "SecretNoRule123!",
                            "label": "password",
                            "start_position": 14,
                            "end_position": 30,
                            "source": "detector",
                        }
                    ]
                },
            ]
        }
    )
    paths = tool._CaseRunPaths(
        raw_path=tmp_path / "raw" / "case.jsonl",
        artifact_output_path=tmp_path / "raw" / "case.detection-artifacts.jsonl",
        trace_path=None,
        artifact_snapshot={},
    )
    tool.write_detection_artifact_payloads([_stale_detection_artifact_payload()], paths.artifact_output_path)
    contexts = {"artifact_path": tmp_path / "artifacts"}
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(input_path, index=False)
    execution = tool._CaseExecution(
        input_data=tool.AnonymizerInput(source=str(input_path), text_column="text"),
        trace_dataframe=trace_dataframe,
    )

    result = tool._case_detection_artifact_path(
        contexts,
        paths,
        case=case,
        config=config,
        execution=execution,
    )

    assert result == paths.artifact_output_path
    rows = [json.loads(line) for line in paths.artifact_output_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert [row["workflow_name"] for row in rows] == [
        "entity-detection-final-trace",
        "entity-detection-final-trace",
    ]
    assert [row["row_index"] for row in rows] == [0, 1]
    assert [row["final_source_counts.rule"] for row in rows] == [1.0, None]
    assert [row["final_source_counts.detector"] for row in rows] == [None, 1.0]
    assert "sk-test" not in paths.artifact_output_path.read_text(encoding="utf-8")
    assert "SecretNoRule123!" not in paths.artifact_output_path.read_text(encoding="utf-8")


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

    def fake_execute_case(*_args: Any, raw_path: Path, **_kwargs: Any) -> Any:
        attempts.append(raw_path)
        if len(attempts) == 1:
            raise RuntimeError("transient provider health check failure")
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text('{"record_type":"run"}\n', encoding="utf-8")
        return tool._CaseExecution(
            input_data=tool.AnonymizerInput(source=str(tmp_path / "input.csv"), text_column="text")
        )

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
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(tmp_path / "biographies.csv", index=False)
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


def test_benchmark_preflight_rejects_local_structured_substitute_for_contextual_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_local_substitute_contextual_labels",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice has token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: local-substitute-suite
workloads:
  - id: input
    source: input.csv
configs:
  - id: local-substitute
    detect:
      entity_labels: [api_key, person]
    replace: substitute
    experimental_replacement_strategy: local_structured_substitute
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="unsupported local structured substitute labels: person"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_accepts_local_structured_substitute_supported_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_local_substitute_supported_labels",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: local-substitute-suite
workloads:
  - id: input
    source: input.csv
configs:
  - id: local-substitute
    detect:
      entity_labels: [api_key, email, http_cookie, password, pin, unique_id, url, user_name]
    replace: substitute
    experimental_replacement_strategy: local_structured_substitute
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

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


def test_benchmark_preflight_rejects_native_strategy_without_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_native_runtime_required",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    monkeypatch.delenv("ANONYMIZER_BENCH_NATIVE_ENDPOINT", raising=False)
    monkeypatch.delenv("ANONYMIZER_BENCH_NATIVE_MODEL", raising=False)
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: native-runtime-suite
workloads:
  - id: input
    source: input.csv
configs:
  - id: native-single-pass
    experimental_detection_strategy: native_single_pass
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="native_runtime.runtime_id"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_native_runtime_config_override_merges_suite_defaults() -> None:
    tool = load_tool(
        "measurement_benchmark_tool_native_runtime_merge",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    config = tool.ConfigSpec(
        id="native-single-pass",
        experimental_detection_strategy="native_single_pass",
        native_runtime=tool.NativeRuntimeSpec(model="config-model", max_workers=2),
        replace="redact",
    )
    spec = tool.BenchmarkSpec(
        suite_id="native-runtime-suite",
        native_runtime=tool.NativeRuntimeSpec(
            runtime_id="suite-runtime",
            endpoint="http://suite-endpoint/v1",
            model="suite-model",
            provider="suite-provider",
            max_workers=4,
        ),
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[config],
    )

    runtime = tool._native_detection_runtime(spec, config)

    assert runtime.endpoint == "http://suite-endpoint/v1"
    assert runtime.model == "config-model"
    assert runtime.provider == "suite-provider"
    assert runtime.max_workers == 2


def test_benchmark_preflight_rejects_native_strategy_without_endpoint_or_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_native_runtime_endpoint_required",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    monkeypatch.delenv("ANONYMIZER_BENCH_NATIVE_ENDPOINT", raising=False)
    monkeypatch.delenv("ANONYMIZER_BENCH_NATIVE_MODEL", raising=False)
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: native-runtime-suite
native_runtime:
  runtime_id: native-runtime
workloads:
  - id: input
    source: input.csv
configs:
  - id: native-single-pass
    experimental_detection_strategy: native_single_pass
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="native_runtime.endpoint"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_native_runtime_resolves_endpoint_and_model_from_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_native_runtime_env",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    monkeypatch.setenv("ANONYMIZER_BENCH_NATIVE_ENDPOINT", "http://runtime-from-env/v1")
    monkeypatch.setenv("ANONYMIZER_BENCH_NATIVE_MODEL", "env-model")
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: native-runtime-suite
native_runtime:
  runtime_id: env-runtime
workloads:
  - id: input
    source: input.csv
configs:
  - id: native-single-pass
    experimental_detection_strategy: native_single_pass
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    tool.preflight_suite(spec, spec_path=spec_path)
    runtime = tool._native_detection_runtime(spec, spec.configs[0])
    tags = tool._run_tags(
        tool.BenchmarkCase(
            suite_id="native-runtime-suite",
            workload_id="input",
            config_id="native-single-pass",
            repetition=0,
            case_id="input__native-single-pass__r000",
        ),
        spec,
    )

    assert runtime.endpoint == "http://runtime-from-env/v1"
    assert runtime.model == "env-model"
    assert tags["native_runtime_id"] == "env-runtime"
    assert "native_endpoint" not in tags
    assert tags["native_endpoint_env"] == "ANONYMIZER_BENCH_NATIVE_ENDPOINT"
    assert tags["native_model"] == "env-model"


def test_benchmark_preflight_skips_inactive_native_configs(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_inactive_native_runtime",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice works at Acme"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: inactive-native-suite
workloads:
  - id: input
    source: input.csv
configs:
  - id: redact
    replace: redact
  - id: inactive-native
    experimental_detection_strategy: native_single_pass
    replace: redact
matrix:
  - workload: input
    config: redact
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
        dd_parser_compat=tool.DDParserCompatMode.none,
    )

    assert len(captured) == 1
    assert captured[0].dd_trace == "all_messages"
    assert captured[0].dd_trace_path == trace_path
    assert captured[0].streaming is True
    assert captured[0].keep_records is False


def test_benchmark_config_accepts_experimental_detection_strategy() -> None:
    tool = load_tool(
        "measurement_benchmark_tool_detection_strategy_config", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )

    config = tool.ConfigSpec(
        id="rules-only",
        replace="redact",
        experimental_detection_strategy="rules_only",
    )

    assert config.experimental_detection_strategy == tool.ExperimentalDetectionStrategy.rules_only
    anonymizer_config = tool.build_anonymizer_config(config)
    assert not hasattr(anonymizer_config.detect, "experimental_detection_strategy")

    detector_only = tool.ConfigSpec(
        id="detector-only",
        replace="redact",
        experimental_detection_strategy="detector_only",
    )

    assert detector_only.experimental_detection_strategy == tool.ExperimentalDetectionStrategy.detector_only

    rules_covered = tool.ConfigSpec(
        id="rules-covered",
        replace="redact",
        experimental_detection_strategy="rules_covered_or_default",
    )

    assert rules_covered.experimental_detection_strategy == tool.ExperimentalDetectionStrategy.rules_covered_or_default

    native_rules_router = tool.ConfigSpec(
        id="native-rules-router",
        replace="redact",
        experimental_detection_strategy="native_rules_router",
    )

    assert native_rules_router.experimental_detection_strategy == tool.ExperimentalDetectionStrategy.native_rules_router

    native_candidate_validate = tool.ConfigSpec(
        id="native-candidate-validate",
        replace="redact",
        experimental_detection_strategy="native_candidate_validate_no_augment",
    )

    assert (
        native_candidate_validate.experimental_detection_strategy
        == tool.ExperimentalDetectionStrategy.native_candidate_validate_no_augment
    )

    detector_native_validate = tool.ConfigSpec(
        id="detector-native-validate",
        replace="redact",
        experimental_detection_strategy="detector_native_validate_no_augment",
    )

    assert (
        detector_native_validate.experimental_detection_strategy
        == tool.ExperimentalDetectionStrategy.detector_native_validate_no_augment
    )

    detector_native_augment = tool.ConfigSpec(
        id="detector-native-augment",
        replace="redact",
        experimental_detection_strategy="detector_native_validate_native_augment",
    )

    assert (
        detector_native_augment.experimental_detection_strategy
        == tool.ExperimentalDetectionStrategy.detector_native_validate_native_augment
    )

    gliner_native_validate = tool.ConfigSpec(
        id="gliner-native-validate",
        replace="redact",
        experimental_detection_strategy="gliner_native_validate_no_augment",
    )

    assert (
        gliner_native_validate.experimental_detection_strategy
        == tool.ExperimentalDetectionStrategy.gliner_native_validate_no_augment
    )

    gliner_native_augment = tool.ConfigSpec(
        id="gliner-native-augment",
        replace="redact",
        experimental_detection_strategy="gliner_native_validate_native_augment",
    )

    assert (
        gliner_native_augment.experimental_detection_strategy
        == tool.ExperimentalDetectionStrategy.gliner_native_validate_native_augment
    )

    native_single_pass = tool.ConfigSpec(
        id="native-single-pass",
        replace="redact",
        experimental_detection_strategy="native_single_pass",
    )

    assert native_single_pass.experimental_detection_strategy == tool.ExperimentalDetectionStrategy.native_single_pass

    native_single_pass_recall = tool.ConfigSpec(
        id="native-single-pass-recall",
        replace="redact",
        experimental_detection_strategy="native_single_pass_recall",
    )

    assert (
        native_single_pass_recall.experimental_detection_strategy
        == tool.ExperimentalDetectionStrategy.native_single_pass_recall
    )

    native_single_pass_values = tool.ConfigSpec(
        id="native-single-pass-values",
        replace="redact",
        experimental_detection_strategy="native_single_pass_values",
    )

    assert (
        native_single_pass_values.experimental_detection_strategy
        == tool.ExperimentalDetectionStrategy.native_single_pass_values
    )

    native_single_pass_values_recall = tool.ConfigSpec(
        id="native-single-pass-values-recall",
        replace="redact",
        experimental_detection_strategy="native_single_pass_values_recall",
    )

    assert (
        native_single_pass_values_recall.experimental_detection_strategy
        == tool.ExperimentalDetectionStrategy.native_single_pass_values_recall
    )


def test_benchmark_config_accepts_experimental_rule_labels() -> None:
    tool = load_tool("measurement_benchmark_tool_rule_labels_config", REPO_ROOT / "tools/measurement/run_benchmarks.py")

    config = tool.ConfigSpec(
        id="rules-guardrail",
        replace="redact",
        experimental_detection_strategy="rules_guardrail",
        experimental_rule_labels=["street_address"],
    )

    assert config.experimental_rule_labels == ["street_address"]
    anonymizer_config = tool.build_anonymizer_config(config)
    assert not hasattr(anonymizer_config.detect, "experimental_rule_labels")

    detector_only = tool.ConfigSpec(
        id="rules-guardrail-detector-only",
        replace="redact",
        experimental_detection_strategy="rules_guardrail_detector_only",
        experimental_rule_labels=["api_key"],
    )

    assert detector_only.experimental_rule_labels == ["api_key"]


def test_benchmark_spec_accepts_dd_parser_compat() -> None:
    tool = load_tool(
        "measurement_benchmark_tool_dd_parser_compat_config", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )

    spec = tool.BenchmarkSpec(
        suite_id="raw-json-suite",
        dd_parser_compat="raw_json",
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[tool.ConfigSpec(id="redact", replace="redact")],
    )

    assert spec.dd_parser_compat == tool.DDParserCompatMode.raw_json


def test_benchmark_preflight_rejects_rules_only_without_explicit_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_only_without_labels", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: rules-only-no-labels
workloads:
  - id: input
    source: input.csv
configs:
  - id: rules-only-redact
    experimental_detection_strategy: rules_only
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="requires explicit detect.entity_labels"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_rejects_rules_only_unsupported_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_only_unsupported_labels", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: rules-only-unsupported-labels
workloads:
  - id: input
    source: input.csv
configs:
  - id: rules-only-redact
    experimental_detection_strategy: rules_only
    detect:
      entity_labels: [api_key, person]
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="unsupported high-confidence rule labels.*person"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_accepts_rules_only_supported_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_only_supported_labels", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: rules-only-supported-labels
workloads:
  - id: input
    source: input.csv
configs:
  - id: rules-only-redact
    experimental_detection_strategy: rules_only
    detect:
      entity_labels: [api_key, email, http_cookie, password, pin, unique_id, url, user_name]
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_accepts_rules_covered_or_default_contextual_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rules_covered_contextual_labels",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice has token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: rules-covered-contextual-labels
workloads:
  - id: input
    source: input.csv
configs:
  - id: rules-covered-redact
    experimental_detection_strategy: rules_covered_or_default
    detect:
      entity_labels: [api_key, person]
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_rejects_experimental_rule_labels_for_non_rule_strategy(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rule_labels_non_rule_strategy",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: rule-labels-non-rule-strategy
workloads:
  - id: input
    source: input.csv
configs:
  - id: redact
    experimental_detection_strategy: prose_augment_focus
    experimental_rule_labels: [street_address]
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="experimental_rule_labels requires a rule-backed strategy"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_accepts_experimental_rule_labels_for_compact_rule_guardrail(
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rule_labels_compact_rule_guardrail",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice lives on West Roberts Drive."]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: rule-labels-compact-rule-guardrail
workloads:
  - id: input
    source: input.csv
configs:
  - id: redact
    experimental_detection_strategy: rules_guardrail_compact_validation
    experimental_rule_labels: [street_address]
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_preflight_rejects_unsupported_experimental_rule_labels(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_rule_labels_unsupported",
        REPO_ROOT / "tools/measurement/run_benchmarks.py",
    )
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Alice"]}).to_csv(input_path, index=False)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: rule-labels-unsupported
workloads:
  - id: input
    source: input.csv
configs:
  - id: redact
    experimental_detection_strategy: rules_guardrail
    experimental_rule_labels: [person]
    replace: redact
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="unsupported experimental_rule_labels.*person"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_case_enters_experimental_detection_strategy_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_detection_strategy_case", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    captured_measurements: list[Any] = []
    captured_parser_compat: list[Any] = []
    captured_strategies: list[Any] = []
    captured_rule_labels: list[Any] = []

    @contextmanager
    def fake_measurement_session(config: Any) -> Iterator[None]:
        captured_measurements.append(config)
        yield None

    @contextmanager
    def fake_detection_strategy_context(strategy: Any, *, rule_labels: list[str] | None = None) -> Iterator[None]:
        captured_strategies.append(strategy)
        captured_rule_labels.append(rule_labels)
        yield None

    @contextmanager
    def fake_dd_parser_compat_context(mode: Any) -> Iterator[None]:
        captured_parser_compat.append(mode)
        yield None

    class FakeAnonymizer:
        def run(self, *, config: Any, data: Any) -> None:
            assert config.replace is not None
            assert data.text_column == "text"

    monkeypatch.setattr(tool, "configured_measurement_session", fake_measurement_session)
    monkeypatch.setattr(tool, "dd_parser_compat_context", fake_dd_parser_compat_context)
    monkeypatch.setattr(tool, "experimental_detection_strategy_context", fake_detection_strategy_context)

    spec = tool.BenchmarkSpec(
        suite_id="rules-suite",
        dd_parser_compat="raw_json",
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=[
            tool.ConfigSpec(
                id="rules-only-redact",
                replace="redact",
                experimental_detection_strategy="rules_only",
                experimental_rule_labels=["api_key"],
            )
        ],
    )
    pd.DataFrame({"text": ["token=sk-test-AAAAAAAAAAAAAAAAAAAAAAAA"]}).to_csv(tmp_path / "input.csv", index=False)
    case = tool.BenchmarkCase(
        suite_id="rules-suite",
        workload_id="input",
        config_id="rules-only-redact",
        repetition=0,
        case_id="input__rules-only-redact__r000",
    )

    tool._execute_case(
        FakeAnonymizer(),
        spec.workloads[0],
        spec.configs[0],
        raw_path=tmp_path / "raw" / "input__rules-only-redact__r000.jsonl",
        trace_path=None,
        case=case,
        spec=spec,
        base_dir=tmp_path,
        dd_trace=tool.DDTraceMode.none,
        dd_parser_compat=spec.dd_parser_compat,
    )

    assert captured_parser_compat == [tool.DDParserCompatMode.raw_json]
    assert captured_strategies == [tool.ExperimentalDetectionStrategy.rules_only]
    assert captured_rule_labels == [["api_key"]]
    assert captured_measurements[0].run_tags["experimental_detection_strategy"] == "rules_only"
    assert captured_measurements[0].run_tags["experimental_rule_labels"] == ["api_key"]
    assert captured_measurements[0].run_tags["dd_parser_compat"] == "raw_json"
