# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from anonymizer.config.rewrite import DEFAULT_PRESERVE_TEXT

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_FIXTURES = REPO_ROOT / "tests/fixtures/measurement"
RUN_BENCHMARKS_PATH = REPO_ROOT / "tools/measurement/run_benchmarks.py"
SWEEP_BENCHMARKS_PATH = REPO_ROOT / "tools/measurement/sweep_benchmarks.py"
EXECUTION_PATH = REPO_ROOT / "tools/measurement/measurement_tools/execution.py"
WANDB_LOGGING_PATH = REPO_ROOT / "tools/measurement/measurement_tools/wandb_logging.py"
WANDB_SETUP_PATH = REPO_ROOT / "tools/measurement/measurement_tools/wandb_setup.py"
WANDB_REPORT_PATH = REPO_ROOT / "tools/measurement/create_wandb_report.py"


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def _load_measurement_fixture(name: str) -> dict[str, Any]:
    return json.loads((MEASUREMENT_FIXTURES / name).read_text(encoding="utf-8"))


def _anonymizer_result_from_fixture(result_cls: type[Any], payload: dict[str, Any]) -> Any:
    return result_cls(
        dataframe=pd.DataFrame(payload["dataframe"]),
        trace_dataframe=pd.DataFrame(payload["trace_dataframe"]),
        resolved_text_column="text",
        failed_records=[],
        replace_method=None,
    )


class _FakeEvaluatingAnonymizer:
    def __init__(self, run_result: Any, evaluated_result: Any) -> None:
        self.run_result = run_result
        self.evaluated_result = evaluated_result

    def run(self, *, config: Any, data: Any) -> Any:
        return self.run_result

    def evaluate(self, result: Any) -> Any:
        assert result is self.run_result
        return self.evaluated_result


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


def _minimal_benchmark_spec(
    tool: ModuleType,
    *,
    suite_id: str = "suite",
    configs: list[Any] | None = None,
    case_retries: int = 0,
    case_retry_backoff_sec: float = 0.0,
    run_tags: dict[str, Any] | None = None,
) -> Any:
    return tool.BenchmarkSpec(
        suite_id=suite_id,
        case_retries=case_retries,
        case_retry_backoff_sec=case_retry_backoff_sec,
        run_tags=run_tags or {},
        workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
        configs=configs or [tool.ConfigSpec(id="redact", replace="redact")],
    )


def _minimal_benchmark_case(
    tool: ModuleType,
    *,
    suite_id: str = "suite",
    workload_id: str = "input",
    config_id: str = "redact",
    repetition: int = 0,
) -> Any:
    return tool.BenchmarkCase(
        suite_id=suite_id,
        workload_id=workload_id,
        config_id=config_id,
        repetition=repetition,
        case_id=f"{workload_id}__{config_id}__r{repetition:03d}",
    )


def _write_text_input(tmp_path: Path, text: str = "Alice works at Acme") -> Path:
    input_path = tmp_path / "input.csv"
    pd.DataFrame({"text": [text]}).to_csv(input_path, index=False)
    return input_path


def _copy_biography_data(tmp_path: Path, filename: str = "input.csv") -> Path:
    source = REPO_ROOT / "docs" / "data" / "NVIDIA_synthetic_biographies.csv"
    destination = tmp_path / filename
    destination.write_bytes(source.read_bytes())
    return destination


def _fixture_module_name(prefix: str, request: pytest.FixtureRequest) -> str:
    safe_name = "".join(char if char.isalnum() else "_" for char in request.node.name)
    return f"{prefix}_{safe_name}"


@pytest.fixture
def run_benchmarks_tool(request: pytest.FixtureRequest) -> ModuleType:
    return load_tool(_fixture_module_name("measurement_run_benchmarks", request), RUN_BENCHMARKS_PATH)


@pytest.fixture
def sweep_tool(request: pytest.FixtureRequest) -> ModuleType:
    return load_tool(_fixture_module_name("measurement_sweep", request), SWEEP_BENCHMARKS_PATH)


@pytest.fixture
def execution_tool(request: pytest.FixtureRequest) -> ModuleType:
    return load_tool(_fixture_module_name("measurement_execution", request), EXECUTION_PATH)


@pytest.fixture
def wandb_logging_tool(request: pytest.FixtureRequest) -> ModuleType:
    load_tool(_fixture_module_name("measurement_wandb_logging_prereq", request), RUN_BENCHMARKS_PATH)
    return load_tool(_fixture_module_name("measurement_wandb_logging", request), WANDB_LOGGING_PATH)


@pytest.fixture
def wandb_setup_tool(request: pytest.FixtureRequest) -> ModuleType:
    return load_tool(_fixture_module_name("measurement_wandb_setup", request), WANDB_SETUP_PATH)


@pytest.fixture
def wandb_report_tool(request: pytest.FixtureRequest) -> ModuleType:
    return load_tool(_fixture_module_name("measurement_wandb_report", request), WANDB_REPORT_PATH)


def _write_yaml(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _simple_suite_payload(*, suite_id: str = "base-suite", include_model_paths: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "suite_id": suite_id,
        "workloads": [{"id": "input", "source": "input.csv"}],
        "configs": [{"id": "redact", "replace": "redact"}],
    }
    if include_model_paths:
        payload.update({"model_configs": "./models.yaml", "model_providers": "./providers.yaml"})
    return payload


def _write_simple_suite(tmp_path: Path, *, suite_id: str = "base-suite") -> Path:
    return _write_yaml(tmp_path / "suite.yaml", _simple_suite_payload(suite_id=suite_id))


def _write_threshold_sweep(
    tmp_path: Path,
    *,
    base_suite: Path,
    sweep_id: str = "threshold",
    parameters: dict[str, list[Any]] | None = None,
) -> Path:
    return _write_yaml(
        tmp_path / "sweep.yaml",
        {
            "sweep_id": sweep_id,
            "base_suite": base_suite.name,
            "parameters": parameters or {"configs.*.detect.gliner_threshold": [0.2, 0.4]},
        },
    )


def _fake_run() -> SimpleNamespace:
    return SimpleNamespace(id="fake-run-id")


def _fake_wandb_module(state: SimpleNamespace, *, active_run: bool = False) -> SimpleNamespace:
    module = SimpleNamespace(run=_fake_run() if active_run else None)

    def init(**kwargs: Any) -> SimpleNamespace:
        state.init_kwargs.update(kwargs)
        module.run = _fake_run()
        return module.run

    def finish() -> None:
        state.finished.append("finish")
        module.run = None

    module.config = SimpleNamespace(
        update=lambda payload, *, allow_val_change: _record_config_update(state, payload, allow_val_change)
    )
    module.summary = SimpleNamespace(update=lambda payload: state.summary_updates.append(payload))
    module.Settings = lambda **kwargs: kwargs
    module.init = init
    module.define_metric = lambda *args, **kwargs: state.defined_metrics.append((args, kwargs))
    module.log = lambda payload: state.logged.append(payload)
    module.Table = lambda *, dataframe: dataframe
    module.finish = finish
    return module


def _record_config_update(state: SimpleNamespace, payload: dict[str, Any], allow_val_change: bool) -> None:
    assert allow_val_change is True
    state.config_updates.append(payload)


def _wandb_state() -> SimpleNamespace:
    return SimpleNamespace(
        init_kwargs={},
        config_updates=[],
        defined_metrics=[],
        logged=[],
        summary_updates=[],
        finished=[],
    )


def _benchmark_result(tool: ModuleType, *, suite_id: str, output_dir: Path, cases: list[Any] | None = None) -> Any:
    return tool.BenchmarkResult(
        suite_id=suite_id,
        output_dir=str(output_dir),
        measurement_path=str(output_dir / "measurements.jsonl"),
        summary_path=str(output_dir / "summary.json"),
        table_dir=None,
        cases=cases or [],
    )


def _wandb_metadata_spec(tool: ModuleType) -> Any:
    return tool.BenchmarkSpec(
        suite_id="suite-a",
        model_configs="models.yaml",
        model_providers="providers.yaml",
        artifact_path="artifacts",
        run_tags={"ci_job": "123", "api_key": "sk-secret-token"},
        case_retries=2,
        case_retry_backoff_sec=3.5,
        workloads=[
            tool.WorkloadSpec(
                id="workload-a",
                source="/private/path/input.csv",
                text_column="body",
                id_column="id",
                data_summary="contains Alice and raw secret",
                row_limit=5,
                row_offset=10,
            )
        ],
        configs=[_hash_config(tool), _rewrite_config(tool)],
        matrix=[tool.MatrixEntry(workload="workload-a", config="hash-a", repetitions=2)],
    )


def _hash_config(tool: ModuleType) -> Any:
    return tool.ConfigSpec(
        id="hash-a",
        detect={"entity_labels": ["person", "api_key"], "gliner_threshold": 0.4},
        replace=tool.ReplaceSpec(strategy=tool.ReplaceKind.hash, digest_length=12, instructions="raw secret"),
        evaluate=True,
    )


def _rewrite_config(tool: ModuleType) -> Any:
    return tool.ConfigSpec(
        id="rewrite-a",
        rewrite=tool.RewriteSpec(
            protect="protect Alice",
            preserve="preserve raw secret",
            instructions="raw prompt",
            risk_tolerance=tool.RiskTolerance.minimal,
            max_repair_iterations=1,
            strict_entity_protection=True,
        ),
    )


def _patch_stable_wandb_metadata(monkeypatch: pytest.MonkeyPatch, tool: ModuleType) -> None:
    monkeypatch.setattr(tool, "_git_metadata", lambda _cwd: {"commit": "abc123", "branch": "main", "dirty": False})
    monkeypatch.setattr(
        tool,
        "_package_versions",
        lambda: {"anonymizer_version": "1.2.3", "datadesigner_version": "4.5.6", "wandb_version": "7.8.9"},
    )


def _write_case_measurements(tool: ModuleType, output_dir: Path, case: Any) -> Any:
    raw_path = output_dir / "raw" / f"{case.case_id}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "record_type": "run",
                        "run_id": case.case_id,
                        "mode": "replace",
                        "strategy": "Redact",
                        "input_row_count": 1,
                    }
                ),
                json.dumps(
                    {
                        "record_type": "record",
                        "run_id": case.case_id,
                        "final_entity_count": 1,
                        "text": "Alice",
                        "utility_score": 0.95,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return case.model_copy(
        update={"status": tool.CaseStatus.completed, "measurement_path": str(raw_path), "elapsed_sec": 1.5}
    )


class _FakeAnonymizer:
    def __init__(self, **_kwargs: Any) -> None:
        pass


def _set_wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "WANDB_MODE": "online",
        "WANDB_ENTITY": "global-entity",
        "WANDB_PROJECT": "global-project",
        "WANDB_GROUP": "global-group",
        "WANDB_JOB_TYPE": "global-job",
        "WANDB_NAME": "global-run",
        "WANDB_TAGS": "global",
        "ANONYMIZER_MEASUREMENT_WANDB_MODE": "offline",
        "ANONYMIZER_MEASUREMENT_WANDB_ENTITY": "env-entity",
        "ANONYMIZER_MEASUREMENT_WANDB_PROJECT": "env-project",
        "ANONYMIZER_MEASUREMENT_WANDB_GROUP": "env-group",
        "ANONYMIZER_MEASUREMENT_WANDB_JOB_TYPE": "env-job",
        "ANONYMIZER_MEASUREMENT_WANDB_RUN_NAME": "env-run",
        "ANONYMIZER_MEASUREMENT_WANDB_TAGS": "env-a, env-b",
        "ANONYMIZER_MEASUREMENT_WANDB_LOG_TABLES": "true",
    }.items():
        monkeypatch.setenv(name, value)


def _wandb_report_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {
            "benchmark/case_completed": 2,
            "benchmark/case_total": 2,
            "benchmark/case_errored": 0,
            "benchmark/case_success_rate": 1.0,
            "benchmark/case_elapsed_sec_mean": 12.34,
            "measurement/record/final_entity_count": 10,
            "measurement/record/original_value_leak_count": 0,
            "measurement/ndd_workflow/observed_total_requests": 4,
            "measurement/ndd_workflow/observed_failed_requests": 0,
            "measurement/ndd_workflow/observed_total_tokens": 100,
        },
        {
            "benchmark": {"suite_id": "suite", "suite_file_hash": "hash"},
            "runtime": {"anonymizer_version": "1", "datadesigner_version": "2", "wandb_version": "3"},
            "git": {"branch": "feature", "commit": "abc", "dirty": True},
            "workloads": [_report_workload()],
            "configs": [_report_config()],
        },
    )


def _report_workload() -> dict[str, Any]:
    return {
        "id": "workload",
        "source_kind": "file",
        "source_suffix": ".csv",
        "source_path": "/secret/path.csv",
        "text_column": "text",
        "row_limit": 5,
    }


def _report_config() -> dict[str, Any]:
    return {
        "id": "redact",
        "detect": {"entity_label_count": 4, "gliner_threshold": 0.3},
        "replace": {"strategy": "redact"},
        "model_providers": "sk-secret-token",
    }


def _sweep_materialization_suite_payload() -> dict[str, Any]:
    return {
        **_simple_suite_payload(include_model_paths=True),
        "configs": [
            {"id": "redact", "replace": "redact"},
            {
                "id": "hash",
                "detect": {"entity_labels": ["person"]},
                "replace": {"strategy": "hash", "digest_length": 12},
            },
        ],
        "matrix": [{"workload": "input", "config": "redact"}, {"workload": "input", "config": "hash"}],
    }


def _assert_wandb_init_state(state: SimpleNamespace) -> None:
    assert state.init_kwargs["entity"] == "entity-a"
    assert state.init_kwargs["project"] == "project-a"
    assert state.init_kwargs["group"] == "group-a"
    assert state.init_kwargs["job_type"] == "job-a"
    assert state.init_kwargs["name"] == "suite-a main@abc1234"
    assert state.init_kwargs["tags"] == ["tag-a", "tag-b", "suite:suite-a", "branch:main", "clean"]
    assert state.defined_metrics == [(("benchmark/*",), {"summary": "last"}), (("measurement/*",), {"summary": "last"})]
    assert state.config_updates == [
        {
            "suite_id": "suite-a",
            "wandb_mode": "offline",
            "wandb_log_tables": False,
            "benchmark_suite_id": "suite-a",
            "benchmark_case_count": 2,
            "benchmark_workload_ids": "workload-a",
            "benchmark_workload_row_limits": 5,
            "benchmark_workload_source_kinds": "local_file",
            "benchmark_workload_source_suffixes": ".csv",
            "benchmark_config_ids": "redact",
            "benchmark_modes": "replace",
            "benchmark_strategies": "redact",
            "benchmark_gliner_thresholds": 0.3,
            "benchmark_entity_label_counts": 4,
            "git": {"commit": "abc123456789", "branch": "main", "dirty": False},
            "benchmark": {"suite_id": "suite-a", "case_count": 2},
            "workloads": [
                {
                    "id": "workload-a",
                    "row_limit": 5,
                    "source": {"kind": "local_file", "suffix": ".csv"},
                }
            ],
            "configs": [
                {
                    "id": "redact",
                    "mode": "replace",
                    "detect": {"gliner_threshold": 0.3, "entity_label_count": 4},
                    "replace": {"strategy": "redact"},
                }
            ],
        }
    ]


def _patch_fake_benchmark_run(
    monkeypatch: pytest.MonkeyPatch,
    tool: ModuleType,
    *,
    output_dir: Path,
    fake_wandb: SimpleNamespace,
) -> None:
    monkeypatch.setattr(tool, "Anonymizer", _FakeAnonymizer)
    monkeypatch.setattr(
        tool, "_run_case", lambda case, *_args, **_kwargs: _write_case_measurements(tool, output_dir, case)
    )
    monkeypatch.setattr(tool, "export_measurement_tables", lambda *_args: output_dir / "tables")
    monkeypatch.setattr(tool, "require_wandb", lambda: fake_wandb)


def _patch_imported_wandb_finish(monkeypatch: pytest.MonkeyPatch, tool: ModuleType, finish: Any) -> None:
    setup_module = sys.modules[tool.finalize_benchmark_wandb_run.__module__]
    monkeypatch.setattr(setup_module, "finish_benchmark_wandb_run", finish)


def _patch_sweep_runner(
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
    calls: list[tuple[Path, Path, Any]],
    *,
    status: Any,
) -> None:
    class FakeCase:
        pass

    FakeCase.status = status
    result = SimpleNamespace(suite_id="base-suite", cases=[FakeCase()])

    def fake_run_or_plan(spec_path: Path, **kwargs: Any) -> Any:
        calls.append((spec_path, kwargs["output"], kwargs["wandb_settings"]))
        return result

    monkeypatch.setattr(sweep_tool.run_benchmarks, "run_or_plan", fake_run_or_plan)


def _assert_sweep_arm_run(calls: list[tuple[Path, Path, Any]], result: Any, tmp_path: Path) -> None:
    assert result.completed_arms == 2
    assert len(calls) == 2
    assert calls[0][1] == tmp_path / "runs" / "arm-000" / "output"
    assert calls[0][2].wandb_group == "threshold"
    assert calls[0][2].wandb_run_name == "threshold-arm-000"
    assert calls[0][2].effective_wandb_tags == ["base", "sweep", "sweep:threshold", "arm:arm-000"]
    assert result.arms[0].wandb_run_name == "threshold-arm-000"


def _assert_wandb_metadata(metadata: dict[str, Any]) -> None:
    assert metadata["benchmark"]["suite_id"] == "suite-a"
    assert metadata["benchmark"]["case_count"] == 2
    assert metadata["git"] == {"commit": "abc123", "branch": "main", "dirty": False}
    assert metadata["runtime"]["anonymizer_version"] == "1.2.3"
    assert metadata["model_sources"] == {
        "has_model_configs": True,
        "has_model_providers": True,
        "has_artifact_path": True,
    }
    assert metadata["workloads"] == [_expected_wandb_workload(metadata)]
    assert metadata["configs"][0]["replace"] == {
        "strategy": "hash",
        "digest_length": 12,
        "has_instructions": True,
    }
    assert metadata["configs"][1]["rewrite"] == {
        "risk_tolerance": "minimal",
        "max_repair_iterations": 1,
        "strict_entity_protection": True,
        "has_privacy_goal": True,
        "has_protect": True,
        "has_preserve": True,
        "has_instructions": True,
    }


def _assert_evaluation_record_matches_fixture(row: dict[str, Any], fixture: dict[str, Any]) -> None:
    assert fixture["expected_evaluation_fields"].items() <= row.items()
    assert set(fixture["forbidden_fields"]).isdisjoint(row)


def _assert_raw_fixture_values_absent(serialized: str, fixture: dict[str, Any]) -> None:
    for raw_value in fixture["dangerous_values"]:
        assert raw_value not in serialized


def _execute_substitute_evaluation_case(tool: ModuleType, tmp_path: Path, anonymizer: Any, input_text: str) -> Path:
    spec = _minimal_benchmark_spec(
        tool,
        suite_id="evaluate-suite",
        configs=[
            tool.ConfigSpec(
                id="substitute",
                replace=tool.ReplaceSpec(strategy=tool.ReplaceKind.substitute),
                evaluate=True,
            )
        ],
    )
    _write_text_input(tmp_path, input_text)
    case = _minimal_benchmark_case(tool, suite_id="evaluate-suite", config_id="substitute")
    measurement_path = tmp_path / "raw" / "input__substitute__r000.jsonl"

    tool._execute_case(
        anonymizer,
        spec.workloads[0],
        spec.configs[0],
        raw_path=measurement_path,
        trace_path=None,
        task_trace_path=None,
        case=case,
        spec=spec,
        base_dir=tmp_path,
        dd_trace=tool.DDTraceMode.none,
    )
    return measurement_path


def _expected_wandb_workload(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "workload-a",
        "source": {
            "kind": "local_file",
            "suffix": ".csv",
            "source_hash": metadata["workloads"][0]["source"]["source_hash"],
        },
        "text_column": "body",
        "has_id_column": True,
        "has_data_summary": True,
        "row_limit": 5,
        "row_offset": 10,
    }


def _assert_logged_wandb_payload(state: SimpleNamespace) -> None:
    assert state.finished == ["finish"]
    assert state.logged
    payload = state.logged[0]
    assert payload["benchmark/case_completed"] == 1
    assert payload["measurement/record/final_entity_count"] == 1.0
    assert payload["measurement/record/utility_score_mean"] == pytest.approx(0.95)
    assert state.summary_updates == [payload]
    assert "Alice" not in json.dumps(payload)


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


def test_benchmark_spec_rejects_reserved_run_tags() -> None:
    tool = load_tool("measurement_benchmark_tool_reserved_tags", REPO_ROOT / "tools/measurement/run_benchmarks.py")

    with pytest.raises(ValidationError, match="reserved benchmark tag"):
        tool.BenchmarkSpec(
            suite_id="tag-suite",
            run_tags={"pipeline_id": "1234", "case_id": "manual"},
            workloads=[tool.WorkloadSpec(id="input", source="input.csv")],
            configs=[tool.ConfigSpec(id="redact", replace="redact")],
        )


def test_benchmark_config_rejects_evaluate_on_rewrite() -> None:
    tool = load_tool("measurement_benchmark_tool_evaluate_rewrite", REPO_ROOT / "tools/measurement/run_benchmarks.py")

    with pytest.raises(ValidationError, match="evaluate is only supported for replace configs"):
        tool.ConfigSpec(
            id="rewrite-evaluate",
            rewrite=tool.RewriteSpec(),
            evaluate=True,
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
    spec = _minimal_benchmark_spec(tool, suite_id="artifact-suite")
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
    assert result.execution["backend"] == "local"
    assert result.execution["export"] is True
    assert result.execution["fail_fast"] is False
    assert analysis_path.read_text(encoding="utf-8") == (
        '{"case_id":"input__redact__r000","workflow_name":"entity-detection"}\n'
    )
    summary = (output_dir / "summary.json").read_text(encoding="utf-8")
    summary_payload = json.loads(summary)
    assert summary_payload["execution"] == result.execution
    assert "detection_artifact_analysis_path" in summary
    assert "detection_artifact_path" in summary


def test_run_suite_skips_detection_artifact_analysis_when_export_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_run_suite_no_export_artifact", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    spec = _minimal_benchmark_spec(tool, suite_id="artifact-suite")
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
    spec = _minimal_benchmark_spec(tool, suite_id="retry-suite", case_retries=1)
    case = _minimal_benchmark_case(tool, suite_id="retry-suite")
    _write_text_input(tmp_path, text="Alice")

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
    spec = _minimal_benchmark_spec(tool, suite_id="retry-suite", case_retries=1)
    case = _minimal_benchmark_case(tool, suite_id="retry-suite")

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
    spec = _minimal_benchmark_spec(tool, suite_id="retry-suite", case_retries=3)
    case = _minimal_benchmark_case(tool, suite_id="retry-suite")
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
    assert result.execution["backend"] == "local"
    assert result.execution["export"] is False
    assert result.execution["fail_fast"] is False
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
      provider: stub
    - alias: validator
      model: test/validator
      provider: stub
    - alias: augmenter
      model: test/augmenter
      provider: stub
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


def test_benchmark_preflight_rejects_missing_evaluate_model_alias(tmp_path: Path) -> None:
    tool = load_tool(
        "measurement_benchmark_tool_preflight_evaluate_models", REPO_ROOT / "tools/measurement/run_benchmarks.py"
    )
    _copy_biography_data(tmp_path)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text(
        """
suite_id: bad-evaluate-model-suite
model_configs: |
  selected_models:
    detection:
      entity_detector: detector
      entity_validator: [validator]
      entity_augmenter: augmenter
    evaluate:
      detection_validity_judge: missing-evaluator
  model_configs:
    - alias: detector
      model: test/detector
      provider: stub
    - alias: validator
      model: test/validator
      provider: stub
    - alias: augmenter
      model: test/augmenter
      provider: stub
workloads:
  - id: biography
    source: input.csv
    text_column: biography
configs:
  - id: redact-evaluate
    replace: redact
    evaluate: true
""",
        encoding="utf-8",
    )
    spec = tool.load_spec(spec_path)

    with pytest.raises(ValueError, match="evaluate.detection_validity_judge='missing-evaluator'"):
        tool.preflight_suite(spec, spec_path=spec_path)


def test_benchmark_example_suites_are_portable() -> None:
    example_paths = sorted((REPO_ROOT / "tools/measurement/examples").glob("*.yaml"))
    assert example_paths

    allowed_public_endpoints = {"https://integrate.api.nvidia.com/v1"}
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
                    assert value in allowed_public_endpoints, (
                        f"{example_path} should use an approved portable endpoint for {key}: {value}"
                    )


def test_benchmark_ci_dd_trace_options_match_runner_enum() -> None:
    tool = load_tool("measurement_benchmark_tool_ci", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/benchmark-ci.yml").read_text(encoding="utf-8"))
    on_section = workflow.get("on", workflow.get(True))

    options = on_section["workflow_dispatch"]["inputs"]["dd_trace"]["options"]

    assert options == [mode.value for mode in tool.DDTraceMode]


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

    spec = _minimal_benchmark_spec(
        tool,
        suite_id="trace-suite",
        run_tags={"commit_sha": "abc123", "pipeline_id": "456"},
    )
    _write_text_input(tmp_path)
    case = _minimal_benchmark_case(tool, suite_id="trace-suite")
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
    assert captured[0].run_tags == {
        "suite_id": "trace-suite",
        "workload_id": "input",
        "config_id": "redact",
        "repetition": 0,
        "case_id": "input__redact__r000",
        "commit_sha": "abc123",
        "pipeline_id": "456",
    }


def test_benchmark_case_can_run_optional_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tool = load_tool("measurement_benchmark_tool_evaluate", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    from anonymizer.interface.results import AnonymizerResult

    calls: list[Any] = []
    run_result = AnonymizerResult(
        dataframe=pd.DataFrame({"text": ["Alice works at Acme"]}),
        trace_dataframe=pd.DataFrame({"text": ["Alice works at Acme"]}),
        resolved_text_column="text",
        failed_records=[],
        replace_method=None,
    )

    @contextmanager
    def fake_measurement_session(_config: Any) -> Iterator[None]:
        yield None

    class FakeAnonymizer:
        def run(self, *, config: Any, data: Any) -> object:
            calls.append(("run", config.replace, data.text_column))
            return run_result

        def evaluate(self, result: object) -> object:
            calls.append(("evaluate", result))
            return result

    monkeypatch.setattr(tool, "configured_measurement_session", fake_measurement_session)

    spec = _minimal_benchmark_spec(
        tool,
        suite_id="evaluate-suite",
        configs=[tool.ConfigSpec(id="redact", replace="redact", evaluate=True)],
    )
    _write_text_input(tmp_path)
    case = _minimal_benchmark_case(tool, suite_id="evaluate-suite")

    tool._execute_case(
        FakeAnonymizer(),
        spec.workloads[0],
        spec.configs[0],
        raw_path=tmp_path / "raw" / "input__redact__r000.jsonl",
        trace_path=None,
        task_trace_path=None,
        case=case,
        spec=spec,
        base_dir=tmp_path,
        dd_trace=tool.DDTraceMode.none,
    )

    assert calls == [("run", tool.Redact(), "text"), ("evaluate", run_result)]


def test_benchmark_optional_evaluation_records_sanitized_judge_metrics(tmp_path: Path) -> None:
    tool = load_tool("measurement_benchmark_tool_evaluate_metrics", REPO_ROOT / "tools/measurement/run_benchmarks.py")
    from anonymizer.interface.results import AnonymizerResult

    fixture = _load_measurement_fixture("evaluation-sanitization.json")
    run_result = _anonymizer_result_from_fixture(AnonymizerResult, fixture["run_result"])
    evaluated_result = _anonymizer_result_from_fixture(AnonymizerResult, fixture["evaluated_result"])
    measurement_path = _execute_substitute_evaluation_case(
        tool,
        tmp_path,
        _FakeEvaluatingAnonymizer(run_result, evaluated_result),
        fixture["input_text"],
    )

    serialized = measurement_path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in serialized.splitlines()]
    evaluation_rows = [row for row in rows if row["record_type"] == "evaluation_record"]

    assert len(evaluation_rows) == 1
    _assert_evaluation_record_matches_fixture(evaluation_rows[0], fixture)
    _assert_raw_fixture_values_absent(serialized, fixture)

    table_dir = tmp_path / "tables"
    tool.export_measurement_tables(measurement_path, table_dir)
    exported = pd.read_parquet(table_dir / "evaluation_record.parquet")
    exported_text = str(exported.to_json(orient="records"))

    assert set(fixture["forbidden_fields"]).isdisjoint(exported.columns)
    _assert_raw_fixture_values_absent(exported_text, fixture)


def test_wandb_settings_default_disabled(run_benchmarks_tool: ModuleType) -> None:
    settings = run_benchmarks_tool.WandbSettings()

    assert settings.wandb_mode == run_benchmarks_tool.WandbMode.disabled
    assert settings.enabled is False
    assert settings.wandb_log_tables is False


def test_execution_metadata_defaults_to_local_without_slurm_env(tmp_path: Path, execution_tool: ModuleType) -> None:
    metadata = execution_tool.benchmark_execution_metadata(
        output_dir=tmp_path / "benchmark-output",
        export=True,
        fail_fast=False,
        dd_trace="none",
        dd_task_trace=False,
        env={},
    )

    assert metadata == {
        "backend": "local",
        "output_dir_hash": execution_tool.stable_metadata_hash(str(tmp_path / "benchmark-output")),
        "export": True,
        "fail_fast": False,
        "dd_trace": "none",
        "dd_task_trace": False,
    }


def test_execution_metadata_detects_slurm_without_raw_paths_or_nodes(
    tmp_path: Path,
    execution_tool: ModuleType,
) -> None:
    dangerous_values = [
        "private-node",
        "secret-account",
        "private-partition",
        "paid-qos",
        "secret-cluster",
        "sensitive-job",
        "private-submit-host",
        "private-stdout-path",
        "private-stderr-path",
    ]
    metadata = execution_tool.benchmark_execution_metadata(
        output_dir=tmp_path / "private" / "benchmark-output",
        export=False,
        fail_fast=True,
        dd_trace="last_message",
        dd_task_trace=True,
        env={
            "SLURM_JOB_ID": "12345",
            "SLURM_ARRAY_JOB_ID": "12300",
            "SLURM_ARRAY_TASK_ID": "7",
            "SLURM_ARRAY_TASK_COUNT": "32",
            "SLURM_RESTART_COUNT": "1",
            "SLURM_NTASKS": "8",
            "SLURM_JOB_NUM_NODES": "2",
            "SLURM_JOB_NODELIST": "private-node-[1-2]",
            "SLURM_JOB_ACCOUNT": "secret-account",
            "SLURM_JOB_PARTITION": "private-partition",
            "SLURM_JOB_QOS": "paid-qos",
            "SLURM_CLUSTER_NAME": "secret-cluster",
            "SLURM_JOB_NAME": "sensitive-job",
            "SLURM_SUBMIT_HOST": "private-submit-host",
            "SLURM_SUBMIT_DIR": str(tmp_path / "private"),
            "SLURM_STDOUT": str(tmp_path / "private-stdout-path"),
            "SLURM_STDERR": str(tmp_path / "private-stderr-path"),
        },
    )
    serialized = json.dumps(metadata, sort_keys=True)

    assert metadata["backend"] == "slurm"
    assert metadata["slurm"] == {
        "job_id": "12345",
        "array_job_id": "12300",
        "array_task_id": "7",
        "array_task_count": 32,
        "restart_count": 1,
        "ntasks": 8,
        "job_num_nodes": 2,
    }
    assert str(tmp_path) not in serialized
    for raw_value in dangerous_values:
        assert raw_value not in serialized


def test_wandb_settings_accepts_field_name_overrides(run_benchmarks_tool: ModuleType) -> None:
    settings = run_benchmarks_tool.WandbSettings(
        wandb_mode=run_benchmarks_tool.WandbMode.online,
        wandb_entity="field-entity",
        wandb_project="field-project",
        wandb_group="field-group",
        wandb_job_type="field-job",
        wandb_run_name="field-run",
        wandb_tags="alpha, beta",
        wandb_log_tables=True,
    )

    assert settings.wandb_mode == run_benchmarks_tool.WandbMode.online
    assert settings.wandb_entity == "field-entity"
    assert settings.wandb_project == "field-project"
    assert settings.wandb_group == "field-group"
    assert settings.wandb_job_type == "field-job"
    assert settings.wandb_run_name == "field-run"
    assert settings.effective_wandb_tags == ["alpha", "beta"]
    assert settings.wandb_log_tables is True


def test_resolve_wandb_settings_honors_env_when_cli_flags_are_omitted(
    monkeypatch: pytest.MonkeyPatch,
    run_benchmarks_tool: ModuleType,
) -> None:
    _set_wandb_env(monkeypatch)
    settings = run_benchmarks_tool.resolve_wandb_settings()
    overridden = run_benchmarks_tool.resolve_wandb_settings(
        wandb_mode=run_benchmarks_tool.WandbMode.disabled,
        wandb_entity="cli-entity",
        wandb_group="cli-group",
        wandb_job_type="cli-job",
        wandb_run_name="cli-run",
        wandb_tags="cli-a",
        wandb_log_tables=False,
    )

    assert settings.wandb_mode == run_benchmarks_tool.WandbMode.offline
    assert settings.wandb_entity == "env-entity"
    assert settings.wandb_project == "env-project"
    assert settings.wandb_group == "env-group"
    assert settings.wandb_job_type == "env-job"
    assert settings.wandb_run_name == "env-run"
    assert settings.effective_wandb_tags == ["env-a", "env-b"]
    assert settings.wandb_log_tables is True
    assert overridden.wandb_mode == run_benchmarks_tool.WandbMode.disabled
    assert overridden.wandb_entity == "cli-entity"
    assert overridden.wandb_project == "env-project"
    assert overridden.wandb_group == "cli-group"
    assert overridden.wandb_job_type == "cli-job"
    assert overridden.wandb_run_name == "cli-run"
    assert overridden.effective_wandb_tags == ["cli-a"]
    assert overridden.wandb_log_tables is False


def test_wandb_require_wandb_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    run_benchmarks_tool: ModuleType,
) -> None:
    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with pytest.raises(ImportError, match="wandb package is not installed"):
        run_benchmarks_tool.require_wandb()


def test_wandb_report_run_path_accepts_path_and_url(wandb_report_tool: ModuleType) -> None:
    path = wandb_report_tool.parse_wandb_run_path("entity/project/abc123")
    url = wandb_report_tool.parse_wandb_run_path("https://wandb.ai/entity/project/runs/abc123")

    assert path.entity == "entity"
    assert path.project == "project"
    assert path.run_id == "abc123"
    assert path.path == "entity/project/abc123"
    assert url == path


def test_wandb_report_project_path_accepts_entity_project(wandb_report_tool: ModuleType) -> None:
    project = wandb_report_tool.parse_wandb_project_path("entity/project")

    assert project.entity == "entity"
    assert project.project == "project"
    assert project.path == "entity/project"


def test_wandb_report_sweep_columns_include_benchmark_metric_groups(wandb_report_tool: ModuleType) -> None:
    columns = wandb_report_tool._sweep_visible_columns()

    expected = {
        "config:sweep_id",
        "config:sweep_arm_id",
        "summary:benchmark/case_elapsed_sec_sum",
        "summary:measurement/ndd_workflow/elapsed_sec",
        "summary:measurement/ndd_workflow/observed_input_tokens",
        "summary:measurement/ndd_workflow/observed_output_tokens",
        "summary:measurement/ndd_workflow/observed_tokens_per_sec_mean",
        "summary:measurement/ndd_workflow/observed_failed_request_rate_mean",
        "summary:measurement/record/replacement_duplicate_value_count",
        "summary:measurement/record/replacement_missing_final_value_count",
        "summary:measurement/stage/input_rows_per_sec_mean",
    }
    assert expected.issubset(set(columns))


def test_wandb_report_sweep_panels_cover_metric_sections(wandb_report_tool: ModuleType) -> None:
    class FakePanel:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    class FakeConfig:
        def __init__(self, name: str) -> None:
            self.name = name

    class FakeWr:
        BarPlot = FakePanel
        MediaBrowser = FakePanel
        Config = FakeConfig

    panels = wandb_report_tool._sweep_panels(FakeWr)
    titles = {panel.title for panel in panels}
    metrics = {metric for panel in panels for metric in getattr(panel, "metrics", [])}
    media_keys = {key for panel in panels for key in getattr(panel, "media_keys", [])}

    assert {
        "Case Health",
        "Case Latency",
        "NDD Row Flow",
        "NDD Request Health",
        "NDD Token Usage",
        "NDD Throughput",
        "Record Privacy",
        "Replacement Quality",
        "Stage Throughput",
        "Sanitized Measurement Tables",
    }.issubset(titles)
    assert "measurement/ndd_workflow/observed_tokens_per_successful_request_mean" in metrics
    assert "measurement/record/replacement_synthetic_original_collision_value_count" in metrics
    assert "measurement_table/record" in media_keys


@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_wandb_workspace_builds_manual_benchmark_sections(wandb_report_tool: ModuleType) -> None:
    workspace = wandb_report_tool.build_benchmark_workspace(
        wandb_report_tool.WandbProjectPath(entity="entity", project="project"),
        group="threshold",
        title="Benchmark Workspace",
    )

    sections = {section.name: section for section in workspace.sections}
    comparison_panel_types = {type(panel).__name__ for panel in sections["Sweep Comparison"].panels}
    summary_metrics = {
        getattr(panel, "metric", None)
        for panel in sections["Benchmark Summary"].panels
        if type(panel).__name__ == "ScalarChart"
    }

    assert workspace.name == "Benchmark Workspace"
    assert workspace.auto_generate_panels is False
    assert "benchmark-sweep" in str(workspace.runset_settings.filters)
    assert "Metric(\"Group\") == 'threshold'" in str(workspace.runset_settings.filters)
    assert {
        "Benchmark Summary",
        "Privacy",
        "Utility",
        "Cost/Throughput",
        "Sweep Comparison",
        "Tables",
    } == set(sections)
    assert {"RunComparer", "ParallelCoordinatesPlot", "ParameterImportancePlot"}.issubset(comparison_panel_types)
    assert "benchmark/case_success_rate" in summary_metrics
    assert sections["Sweep Comparison"].panels[0].diff_only is True


@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_wandb_sweep_report_uses_043_grouping_accessors(wandb_report_tool: ModuleType) -> None:
    import wandb_workspaces.reports.v2 as wr

    report = wandb_report_tool.build_benchmark_group_report(
        wandb_report_tool.WandbProjectPath(entity="entity", project="project"),
        group="threshold",
        title="Sweep",
        description="Offline serialization test",
    )
    panel_grid = report.blocks[-1]
    grouped_panels = [panel for panel in panel_grid.panels if type(panel).__name__ == "BarPlot"]

    assert panel_grid.runsets[0].groupby == ["config.sweep_arm_id"]
    assert grouped_panels
    assert all(isinstance(panel.groupby, wr.Config) for panel in grouped_panels)
    assert all(panel.groupby.name == "sweep_arm_id" for panel in grouped_panels)
    for panel in grouped_panels:
        panel._to_model()


@pytest.mark.filterwarnings("ignore::pydantic.warnings.PydanticDeprecatedSince20")
def test_wandb_parallel_coordinates_use_typed_accessors(wandb_report_tool: ModuleType) -> None:
    import wandb_workspaces.reports.v2 as wr

    panels = wandb_report_tool._comparison_workspace_panels(wr)
    parallel_coordinates = next(panel for panel in panels if isinstance(panel, wr.ParallelCoordinatesPlot))

    assert [type(column.metric) for column in parallel_coordinates.columns] == [
        wr.Config,
        wr.Config,
        wr.Config,
        wr.Config,
        wr.SummaryMetric,
        wr.SummaryMetric,
        wr.SummaryMetric,
        wr.SummaryMetric,
    ]
    parallel_coordinates._to_model()


def test_wandb_report_markdown_uses_sanitized_fields(wandb_report_tool: ModuleType) -> None:
    run = SimpleNamespace(name="run-a", url="https://wandb.ai/entity/project/runs/run-a")
    summary, config = _wandb_report_fixture()

    markdown = wandb_report_tool.build_report_markdown(run, summary=summary, config=config)

    assert "2/2 cases completed" in markdown
    assert "Final entities" in markdown
    assert "source_suffix=`.csv`" in markdown
    assert "/secret/path.csv" not in markdown
    assert "sk-secret-token" not in markdown


def test_sweep_expands_parameter_grid_and_materializes_arm_suite(
    tmp_path: Path,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _sweep_materialization_suite_payload())
    sweep_path = _write_threshold_sweep(
        tmp_path,
        base_suite=base_suite,
        sweep_id="threshold-digest",
        parameters={
            "configs.*.detect.gliner_threshold": [0.2, 0.4],
            "configs.hash.replace.digest_length": [8],
        },
    )

    spec = sweep_tool.load_sweep_spec(sweep_path)
    arms = sweep_tool.expand_sweep_arms(spec)
    arm_path = sweep_tool.materialize_arm_suite(spec, arms[0], output_root=tmp_path / "out", overwrite=False)
    materialized = yaml.safe_load(arm_path.read_text(encoding="utf-8"))

    assert [arm.arm_id for arm in arms] == ["arm-000", "arm-001"]
    assert arms[0].parameters == {
        "configs.*.detect.gliner_threshold": 0.2,
        "configs.hash.replace.digest_length": 8,
    }
    assert materialized["run_tags"]["sweep_id"] == "threshold-digest"
    assert materialized["run_tags"]["sweep_arm_id"] == "arm-000"
    assert materialized["run_tags"]["sweep_param_configs_all_detect_gliner_threshold"] == 0.2
    assert materialized["model_configs"] == str(tmp_path / "models.yaml")
    assert materialized["model_providers"] == str(tmp_path / "providers.yaml")
    assert materialized["workloads"][0]["source"] == str(tmp_path / "input.csv")
    assert materialized["configs"][0]["detect"]["gliner_threshold"] == 0.2
    assert materialized["configs"][1]["detect"]["gliner_threshold"] == 0.2
    assert materialized["configs"][1]["replace"]["digest_length"] == 8


@pytest.mark.parametrize(
    "inline_key,file_key", [("model_configs", "model_providers"), ("model_providers", "model_configs")]
)
def test_sweep_rebase_preserves_inline_model_yaml(
    tmp_path: Path,
    sweep_tool: ModuleType,
    inline_key: str,
    file_key: str,
) -> None:
    inline_yaml = "selected_models:\n  detection:\n    entity_detector: detector\n"
    suite = {
        inline_key: inline_yaml,
        file_key: "models.yml",
        "artifact_path": "artifacts",
        "workloads": [],
    }

    rebased = sweep_tool._rebase_suite_paths(suite, tmp_path)

    assert rebased[inline_key] == inline_yaml
    assert rebased[file_key] == str(tmp_path / "models.yml")
    assert rebased["artifact_path"] == str(tmp_path / "artifacts")


def test_sweep_run_uses_one_wandb_run_per_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite)
    calls: list[tuple[Path, Path, Any]] = []
    _patch_sweep_runner(monkeypatch, sweep_tool, calls, status=sweep_tool.run_benchmarks.CaseStatus.completed)

    result = sweep_tool.run_sweep(
        sweep_path,
        output_root=tmp_path / "runs",
        overwrite=True,
        dry_run=False,
        export=True,
        fail_fast=False,
        wandb_settings=sweep_tool.run_benchmarks.resolve_wandb_settings(
            wandb_mode=sweep_tool.run_benchmarks.WandbMode.offline,
            wandb_project="project",
            wandb_tags="base",
        ),
        create_report=False,
    )

    _assert_sweep_arm_run(calls, result, tmp_path)


def test_sweep_run_can_create_wandb_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite)
    calls: list[tuple[Path, Path, Any]] = []
    workspace_calls: list[tuple[Any, str]] = []
    _patch_sweep_runner(monkeypatch, sweep_tool, calls, status=sweep_tool.run_benchmarks.CaseStatus.completed)

    def fake_create_workspace(project: Any, *, group: str) -> SimpleNamespace:
        workspace_calls.append((project, group))
        return SimpleNamespace(workspace_url="https://wandb.ai/entity/project/workspace")

    monkeypatch.setattr(sweep_tool, "create_benchmark_group_workspace", fake_create_workspace)

    result = sweep_tool.run_sweep(
        sweep_path,
        output_root=tmp_path / "runs",
        overwrite=True,
        dry_run=False,
        export=True,
        fail_fast=False,
        wandb_settings=sweep_tool.run_benchmarks.resolve_wandb_settings(
            wandb_mode=sweep_tool.run_benchmarks.WandbMode.offline,
            wandb_entity="entity",
            wandb_project="project",
        ),
        create_report=False,
        create_workspace=True,
    )

    assert workspace_calls[0][0].path == "entity/project"
    assert workspace_calls[0][1] == "threshold"
    assert result.workspace_url == "https://wandb.ai/entity/project/workspace"


def test_sweep_fail_fast_stops_after_first_errored_arm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite)
    calls: list[Path] = []

    class FakeCase:
        status = sweep_tool.run_benchmarks.CaseStatus.error

    class FakeResult:
        suite_id = "base-suite"
        cases = [FakeCase()]

    def fake_run_or_plan(spec_path: Path, **_kwargs: Any) -> FakeResult:
        calls.append(spec_path)
        return FakeResult()

    monkeypatch.setattr(sweep_tool.run_benchmarks, "run_or_plan", fake_run_or_plan)

    result = sweep_tool.run_sweep(
        sweep_path,
        output_root=tmp_path / "runs",
        overwrite=True,
        dry_run=False,
        export=True,
        fail_fast=True,
        wandb_settings=sweep_tool.run_benchmarks.resolve_wandb_settings(),
        create_report=False,
    )

    assert len(calls) == 1
    assert len(result.arms) == 1
    assert result.errored_arms == 1


def test_sweep_result_shows_planned_cases_for_pre_case_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(
        tmp_path,
        base_suite=base_suite,
        parameters={"configs.*.detect.gliner_threshold": [0.2]},
    )

    def fail_run_or_plan(_spec_path: Path, **_kwargs: Any) -> None:
        raise RuntimeError("endpoint unavailable")

    monkeypatch.setattr(sweep_tool.run_benchmarks, "run_or_plan", fail_run_or_plan)

    result = sweep_tool.run_sweep(
        sweep_path,
        output_root=tmp_path / "runs",
        overwrite=True,
        dry_run=False,
        export=True,
        fail_fast=False,
        wandb_settings=sweep_tool.run_benchmarks.resolve_wandb_settings(),
        create_report=False,
    )
    rendered = sweep_tool.render_result(result, json_output=False)

    assert result.arms[0].total_cases == 1
    assert "- arm-000: error, cases=0/1, errors=0, error=endpoint unavailable" in rendered


def test_sweep_materialization_failure_is_reported_per_arm(
    tmp_path: Path,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(
        tmp_path,
        base_suite=base_suite,
        parameters={"configs.*.detect.gliner_threshold": [0.2]},
    )
    output_root = tmp_path / "runs"
    spec = sweep_tool.load_sweep_spec(sweep_path)
    arm = sweep_tool.expand_sweep_arms(spec)[0]
    sweep_tool.materialize_arm_suite(spec, arm, output_root=output_root, overwrite=False)

    result = sweep_tool.run_sweep(
        sweep_path,
        output_root=output_root,
        overwrite=False,
        dry_run=False,
        export=True,
        fail_fast=False,
        wandb_settings=sweep_tool.run_benchmarks.resolve_wandb_settings(),
        create_report=False,
    )
    rendered = sweep_tool.render_result(result, json_output=False)

    assert len(result.arms) == 1
    assert result.arms[0].status == "error"
    assert result.arms[0].total_cases == 1
    assert "- arm-000: error, cases=0/1, errors=0, error=sweep arm suite already exists:" in rendered


def test_build_wandb_metadata_projects_sweep_run_tags(tmp_path: Path, run_benchmarks_tool: ModuleType) -> None:
    spec = _minimal_benchmark_spec(
        run_benchmarks_tool,
        run_tags={
            "sweep_id": "threshold",
            "sweep_arm_id": "arm-000",
            "sweep_param_configs_all_detect_gliner_threshold": 0.2,
        },
    )
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text("suite_id: suite\n", encoding="utf-8")

    metadata = run_benchmarks_tool.build_wandb_metadata(
        spec,
        spec_path=spec_path,
        output_dir=tmp_path / "out",
        export=True,
        fail_fast=False,
        dd_trace=run_benchmarks_tool.DDTraceMode.none,
        dd_task_trace=False,
    )

    assert metadata["sweep_id"] == "threshold"
    assert metadata["sweep_arm_id"] == "arm-000"
    assert metadata["sweep"]["id"] == "threshold"
    assert metadata["sweep"]["arm_id"] == "arm-000"
    assert metadata["sweep_params"]["configs_all_detect_gliner_threshold"] == 0.2
    assert metadata["sweep_param_configs_all_detect_gliner_threshold"] == 0.2


def test_build_wandb_metadata_includes_slurm_execution_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    monkeypatch.setenv("SLURM_JOB_ID", "98765")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "3")
    monkeypatch.setenv("SLURM_JOB_NODELIST", "private-node")
    spec = _minimal_benchmark_spec(run_benchmarks_tool)
    spec_path = tmp_path / "suite.yaml"
    spec_path.write_text("suite_id: suite\n", encoding="utf-8")

    metadata = run_benchmarks_tool.build_wandb_metadata(
        spec,
        spec_path=spec_path,
        output_dir=tmp_path / "private-output",
        export=True,
        fail_fast=True,
        dd_trace=run_benchmarks_tool.DDTraceMode.last_message,
        dd_task_trace=True,
    )
    serialized = json.dumps(metadata, sort_keys=True)

    assert metadata["execution"]["backend"] == "slurm"
    assert metadata["execution"]["dd_trace"] == "last_message"
    assert metadata["execution"]["slurm"]["job_id"] == "98765"
    assert metadata["execution"]["slurm"]["array_task_id"] == "3"
    assert str(tmp_path) not in serialized
    assert "private-node" not in serialized


def test_wandb_extract_scalar_metrics_excludes_sensitive_fields(wandb_logging_tool: ModuleType) -> None:
    dangerous_values = [
        "alice@example.com",
        "sk-secret-token",
        "raw judge prompt",
    ]
    record = {
        "record_type": "record",
        "run_id": "case-a",
        "text": "Alice works at Acme",
        "text_replaced": "Avery works at Acme",
        "final_entities": [{"value": "alice@example.com"}],
        "run_tags.api_key": "sk-secret-token",
        "run_tags.text": "Alice",
        "measurement_path": "/tmp/private.jsonl",
        "notes": "raw judge prompt",
        "status": "completed",
        "final_entity_count": 2,
        "utility_score": 0.9,
        "mode": "replace",
    }

    metrics = wandb_logging_tool.extract_scalar_metrics(record)
    serialized = json.dumps(metrics)

    assert metrics["measurement/record/final_entity_count"] == 2
    assert metrics["measurement/record/utility_score"] == 0.9
    assert metrics["measurement/record/status"] == "completed"
    assert "text" not in metrics
    assert "measurement/record/measurement_path" not in metrics
    assert "measurement/record/run_tags.api_key" not in metrics
    assert "measurement/record/run_tags.text" not in metrics
    assert "measurement/record/notes" not in metrics
    for raw_value in dangerous_values:
        assert raw_value not in serialized


def test_wandb_table_logging_drops_sensitive_dotted_columns(wandb_logging_tool: ModuleType) -> None:
    table = pd.DataFrame(
        {
            "final_entity_count": [2],
            "run_tags.api_key": ["sk-secret-token"],
            "run_tags.api-key": ["sk-secret-token"],
            "run_tags.access_token": ["sk-secret-token"],
            "run_tags.secret_tag": ["raw secret"],
            "run_tags.text": ["Alice"],
            "measurement_path": ["/tmp/private.jsonl"],
        }
    )

    sanitized = wandb_logging_tool._project_table(table, record_type="record")

    assert list(sanitized.columns) == ["final_entity_count"]


def test_wandb_table_logging_drops_all_user_run_tags(wandb_logging_tool: ModuleType) -> None:
    table = pd.DataFrame(
        {
            "final_entity_count": [2],
            "run_tags.customer_label": ["alice@example.com"],
            "run_tags.account_number": [123456789],
            "run_tags.benign_name": ["raw PII value"],
        }
    )

    sanitized = wandb_logging_tool._project_table(table, record_type="record")

    assert list(sanitized.columns) == ["final_entity_count"]


def test_wandb_table_logging_projects_allowlisted_columns(wandb_logging_tool: ModuleType) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state, active_run=True)
    records = [
        {
            "record_type": "stage",
            "stage": "EntityDetectionWorkflow.run",
            "status": "completed",
            "elapsed_sec": 1.5,
            "input_row_count": 2,
            "customer_email": "alice@example.com",
            "payload": {"note": "SSN 123-45-6789"},
            "model_usage": {"provider": {"response": "raw model response"}},
        }
    ]

    wandb_logging_tool._log_sanitized_tables(fake_wandb, records)

    assert len(state.logged) == 1
    table = state.logged[0]["measurement_table/stage"]
    assert table.to_dict(orient="records") == [
        {
            "record_type": "stage",
            "stage": "EntityDetectionWorkflow.run",
            "status": "completed",
            "elapsed_sec": 1.5,
            "input_row_count": 2,
        }
    ]


def test_wandb_record_tables_use_length_buckets(wandb_logging_tool: ModuleType) -> None:
    table = pd.DataFrame(
        {
            "record_type": ["record"],
            "text_length_chars": [137],
            "text_length_tokens": [31],
            "text_length_chars_bucket": ["128-511"],
            "text_length_tokens_bucket": ["1-127"],
        }
    )

    projected = wandb_logging_tool._project_table(table, record_type="record")

    assert projected.to_dict(orient="records") == [
        {
            "record_type": "record",
            "text_length_chars_bucket": "128-511",
            "text_length_tokens_bucket": "1-127",
        }
    ]


def test_wandb_table_logging_rejects_unsafe_dimension_values(wandb_logging_tool: ModuleType) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state, active_run=True)

    wandb_logging_tool._log_sanitized_tables(
        fake_wandb,
        [{"record_type": "stage", "stage": "alice@example.com", "status": "completed", "elapsed_sec": 1.0}],
    )

    table = state.logged[0]["measurement_table/stage"]
    assert "alice@example.com" not in table.to_json()


def test_wandb_table_logging_builds_all_tables_before_upload(wandb_logging_tool: ModuleType) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state, active_run=True)
    table_calls = 0

    def table(*, dataframe: pd.DataFrame) -> pd.DataFrame:
        nonlocal table_calls
        table_calls += 1
        if table_calls == 2:
            raise ValueError("invalid second table")
        return dataframe

    fake_wandb.Table = table

    with pytest.raises(ValueError, match="invalid second table"):
        wandb_logging_tool._log_sanitized_tables(
            fake_wandb,
            [
                {"record_type": "stage", "stage": "Anonymizer._run_internal", "elapsed_sec": 1.0},
                {"record_type": "record", "mode": "replace", "final_entity_count": 1},
            ],
        )

    assert state.logged == []


def test_wandb_blocks_engine_internal_columns_by_default(wandb_logging_tool: ModuleType) -> None:
    from anonymizer.engine import constants as engine_constants

    internal_columns = {
        value
        for name, value in vars(engine_constants).items()
        if name.startswith("COL_") and isinstance(value, str) and value.startswith("_")
    }

    assert internal_columns <= wandb_logging_tool.PRIVACY_BLOCKED_FIELD_NAMES


def test_wandb_extract_scalar_metrics_drops_nan_values(wandb_logging_tool: ModuleType) -> None:
    metrics = wandb_logging_tool.extract_scalar_metrics(
        {
            "record_type": "record",
            "final_entity_count": 1,
            "utility_score": float("nan"),
            "weighted_leakage_rate": float("nan"),
        }
    )

    assert metrics == {"measurement/record/final_entity_count": 1}


def test_wandb_extract_scalar_metrics_rejects_unsafe_dimensions(wandb_logging_tool: ModuleType) -> None:
    metrics = wandb_logging_tool.extract_scalar_metrics(
        {
            "record_type": "stage",
            "stage": "alice@example.com",
            "status": "SSN 123-45-6789",
            "mode": "customer-a",
            "strategy": "secret strategy",
            "elapsed_sec": 1.0,
        }
    )

    assert metrics == {"measurement/stage/elapsed_sec": 1.0}


def test_wandb_aggregates_with_explicit_metric_policy(wandb_logging_tool: ModuleType) -> None:
    aggregated = wandb_logging_tool.aggregate_measurement_scalars(
        [
            {
                "record_type": "record",
                "schema_version": 1,
                "timestamp_unix_sec": 1_700_000_000.0,
                "input_has_id_column": True,
                "final_entity_count": 2,
                "utility_score": 0.8,
                "leakage_mass": 0.4,
                "weighted_leakage_rate": 0.1,
                "entity_precision": 0.5,
                "entity_recall": 0.25,
                "entity_f1": 1 / 3,
                "input_rows_per_sec": 10.0,
            },
            {
                "record_type": "record",
                "schema_version": 1,
                "timestamp_unix_sec": 1_700_000_001.0,
                "final_entity_count": 1,
                "utility_score": 1.0,
                "leakage_mass": 0.2,
                "weighted_leakage_rate": 0.3,
                "entity_precision": 1.0,
                "entity_recall": 0.75,
                "entity_f1": 6 / 7,
                "input_rows_per_sec": 20.0,
            },
        ]
    )

    assert aggregated["measurement/record/final_entity_count"] == 3.0
    assert aggregated["measurement/record/utility_score_mean"] == pytest.approx(0.9)
    assert aggregated["measurement/record/leakage_mass_mean"] == pytest.approx(0.3)
    assert aggregated["measurement/record/weighted_leakage_rate_mean"] == pytest.approx(0.2)
    assert aggregated["measurement/record/entity_precision_mean"] == pytest.approx(0.75)
    assert aggregated["measurement/record/entity_recall_mean"] == pytest.approx(0.5)
    assert aggregated["measurement/record/entity_f1_mean"] == pytest.approx(25 / 42)
    assert aggregated["measurement/record/input_rows_per_sec_mean"] == pytest.approx(15.0)
    assert not any("schema_version" in key for key in aggregated)
    assert not any("timestamp_unix_sec" in key for key in aggregated)
    assert not any("input_has_id_column" in key for key in aggregated)


def test_wandb_report_metric_names_match_aggregated_names(
    wandb_logging_tool: ModuleType,
    wandb_report_tool: ModuleType,
) -> None:
    aggregated = wandb_logging_tool.aggregate_measurement_scalars(
        [
            {
                "record_type": "record",
                "entity_precision": 0.9,
                "entity_recall": 0.8,
                "entity_f1": 0.85,
                "leakage_mass": 0.1,
                "repair_iterations": 2,
            },
            {
                "record_type": "ndd_workflow",
                "observed_tokens_per_successful_request": 42.0,
            },
        ]
    )

    expected = {
        "measurement/record/entity_precision_mean",
        "measurement/record/entity_recall_mean",
        "measurement/record/entity_f1_mean",
        "measurement/record/leakage_mass_mean",
        "measurement/record/repair_iterations_mean",
        "measurement/ndd_workflow/observed_tokens_per_successful_request_mean",
    }
    assert expected <= set(aggregated)
    assert expected <= set(wandb_report_tool._all_report_metrics())


def test_wandb_config_excludes_suite_run_tags(wandb_setup_tool: ModuleType) -> None:
    config = wandb_setup_tool._benchmark_wandb_config(
        wandb_setup_tool.WandbSettings(wandb_mode=wandb_setup_tool.WandbMode.offline),
        suite_id="suite-a",
        run_tags={
            "commit_sha": "abc123",
            "owner_email": "alice@example.com",
            "customer_id": 123456,
            "relative_path": "home/alice/input.csv",
            "opaque_access": "opaque-live-value-9Z",
        },
        metadata=None,
    )

    assert "run_tags" not in config


def test_wandb_config_projects_only_declared_metadata(wandb_setup_tool: ModuleType) -> None:
    config = wandb_setup_tool._benchmark_wandb_config(
        wandb_setup_tool.WandbSettings(wandb_mode=wandb_setup_tool.WandbMode.offline),
        suite_id="suite-a",
        run_tags=None,
        metadata={
            "benchmark": {
                "suite_id": "suite-a",
                "case_count": 2,
                "suite_file_hash": "content-hash",
                "owner_email": "alice@example.com",
            },
            "execution": {
                "backend": "slurm",
                "output_dir_hash": "path-hash",
                "export": True,
                "slurm": {"job_id": "123", "cluster_name": "secret-cluster"},
            },
            "runtime": {"anonymizer_version": "1.2.3", "hostname": "private-node"},
            "git": {"commit": "abc123", "branch": "main", "dirty": False, "remote": "ssh://secret"},
            "model_sources": {"has_model_configs": True, "provider_payload": "secret"},
            "workloads": [
                {
                    "id": "workload-a",
                    "row_limit": 5,
                    "source": {"kind": "local_file", "suffix": ".csv", "source_hash": "path-hash"},
                    "customer_note": "Alice Smith",
                }
            ],
            "configs": [
                {
                    "id": "redact",
                    "mode": "replace",
                    "detect": {"gliner_threshold": 0.3, "entity_label_count": 4, "prompt": "raw prompt"},
                    "replace": {"strategy": "redact", "instructions": "raw instructions"},
                    "unknown": object(),
                }
            ],
            "matrix": [{"workload": "workload-a", "config": "redact", "repetitions": 1, "note": "Alice"}],
            "sweep_id": "threshold-sweep",
            "sweep_arm_id": "arm-001",
            "sweep_params": {
                "configs_all_detect_gliner_threshold": 0.3,
                "configs_all_replace_instructions": "replace Alice using raw instructions",
                "configs_all_replace_instructions_algorithm": "Alice Smith",
                "configs_all_replace_algorithm": "Alice Smith",
            },
            "unknown_top_level": {"pii": "alice@example.com"},
        },
    )

    serialized = json.dumps(config, sort_keys=True)
    assert config["benchmark"] == {"suite_id": "suite-a", "case_count": 2, "suite_file_hash": "content-hash"}
    assert config["execution"] == {
        "backend": "slurm",
        "export": True,
        "slurm": {"job_id": "123"},
    }
    assert config["workloads"][0]["source"] == {"kind": "local_file", "suffix": ".csv"}
    assert config["configs"][0]["detect"] == {"gliner_threshold": 0.3, "entity_label_count": 4}
    assert config["configs"][0]["replace"] == {"strategy": "redact"}
    assert config["matrix"] == [{"workload": "workload-a", "config": "redact", "repetitions": 1}]
    assert config["sweep_params"] == {"configs_all_detect_gliner_threshold": 0.3}
    for forbidden in (
        "alice@example.com",
        "Alice Smith",
        "private-node",
        "secret-cluster",
        "path-hash",
        "raw prompt",
        "raw instructions",
        "unknown_top_level",
    ):
        assert forbidden not in serialized


def test_wandb_run_tags_filter_sensitive_tag_values(wandb_setup_tool: ModuleType) -> None:
    tags = wandb_setup_tool._effective_wandb_tags(
        wandb_setup_tool.WandbSettings(
            wandb_mode=wandb_setup_tool.WandbMode.offline,
            wandb_tags="tag-a, sk-secret-token, release/api-key, branch:feature/token-fix, tag-b",
        ),
        suite_id="suite-a",
        metadata={"git": {"branch": "main", "dirty": False}},
    )

    assert tags == ["tag-a", "tag-b", "suite:suite-a", "branch:main", "clean"]


def test_initialize_wandb_run_adds_routing_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: fake_wandb)

    created = wandb_setup_tool.initialize_benchmark_wandb_run(
        wandb_setup_tool.WandbSettings(
            wandb_mode=wandb_setup_tool.WandbMode.offline,
            wandb_entity="entity-a",
            wandb_project="project-a",
            wandb_group="group-a",
            wandb_job_type="job-a",
            wandb_tags="tag-a,tag-b",
        ),
        suite_id="suite-a",
        output_dir=tmp_path,
        run_tags={"commit_sha": "abc123", "secret_tag": "raw secret", "pipeline_id": "sk-secret-token"},
        metadata={
            "git": {"commit": "abc123456789", "branch": "main", "dirty": False},
            "benchmark": {"suite_id": "suite-a", "case_count": 2},
            "workloads": [
                {
                    "id": "workload-a",
                    "row_limit": 5,
                    "source": {"kind": "local_file", "suffix": ".csv"},
                }
            ],
            "configs": [
                {
                    "id": "redact",
                    "mode": "replace",
                    "detect": {"gliner_threshold": 0.3, "entity_label_count": 4},
                    "replace": {"strategy": "redact"},
                }
            ],
        },
    )

    assert created is True
    _assert_wandb_init_state(state)


def test_initialize_wandb_run_rejects_ambient_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state, active_run=True)
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: fake_wandb)

    with pytest.raises(ValueError, match="active W&B run"):
        wandb_setup_tool.initialize_benchmark_wandb_run(
            wandb_setup_tool.WandbSettings(wandb_mode=wandb_setup_tool.WandbMode.offline),
            suite_id="suite-a",
            output_dir=tmp_path,
        )

    assert state.init_kwargs == {}


def test_initialize_wandb_run_disables_implicit_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: fake_wandb)

    wandb_setup_tool.initialize_benchmark_wandb_run(
        wandb_setup_tool.WandbSettings(wandb_mode=wandb_setup_tool.WandbMode.offline),
        suite_id="suite-a",
        output_dir=tmp_path,
    )

    assert state.init_kwargs["settings"] == {
        "console": "off",
        "disable_code": True,
        "disable_git": True,
        "host": "redacted",
        "save_code": False,
        "x_disable_machine_info": True,
        "x_disable_meta": True,
        "x_disable_stats": True,
        "x_save_requirements": False,
    }


def test_initialize_wandb_run_uses_private_staging_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: fake_wandb)

    wandb_setup_tool.initialize_benchmark_wandb_run(
        wandb_setup_tool.WandbSettings(wandb_mode=wandb_setup_tool.WandbMode.offline),
        suite_id="suite-a",
        output_dir=tmp_path,
    )

    staging_dir = tmp_path / ".wandb-private"
    assert state.init_kwargs["dir"] == str(staging_dir)
    assert stat.S_IMODE(staging_dir.stat().st_mode) == 0o700


def test_initialize_wandb_run_rejects_symlinked_staging_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: fake_wandb)
    shared_dir = tmp_path / "shared"
    shared_dir.mkdir()
    (tmp_path / ".wandb-private").symlink_to(shared_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot be a symlink"):
        wandb_setup_tool.initialize_benchmark_wandb_run(
            wandb_setup_tool.WandbSettings(wandb_mode=wandb_setup_tool.WandbMode.offline),
            suite_id="suite-a",
            output_dir=tmp_path,
        )

    assert state.init_kwargs == {}
    assert stat.S_IMODE(shared_dir.stat().st_mode) != 0o700


def test_initialize_wandb_run_uses_explicit_run_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: fake_wandb)

    wandb_setup_tool.initialize_benchmark_wandb_run(
        wandb_setup_tool.WandbSettings(
            wandb_mode=wandb_setup_tool.WandbMode.offline,
            wandb_run_name="operator-name",
        ),
        suite_id="suite-a",
        output_dir=tmp_path,
        metadata={"git": {"commit": "abc123456789", "branch": "main", "dirty": False}},
    )

    assert state.init_kwargs["name"] == "operator-name"


def test_initialize_wandb_run_finishes_created_run_when_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: fake_wandb)

    def fail_config_update(_payload: dict[str, Any], *, allow_val_change: bool) -> None:
        assert allow_val_change is True
        raise RuntimeError("config update failed")

    fake_wandb.config.update = fail_config_update

    with pytest.raises(RuntimeError, match="config update failed"):
        wandb_setup_tool.initialize_benchmark_wandb_run(
            wandb_setup_tool.WandbSettings(wandb_mode=wandb_setup_tool.WandbMode.offline),
            suite_id="suite-a",
            output_dir=tmp_path,
        )

    assert state.finished == ["finish"]
    assert fake_wandb.run is None


def test_build_wandb_metadata_includes_sanitized_configs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    spec = _wandb_metadata_spec(run_benchmarks_tool)
    _patch_stable_wandb_metadata(monkeypatch, run_benchmarks_tool)

    metadata = run_benchmarks_tool.build_wandb_metadata(
        spec,
        spec_path=tmp_path / "suite.yaml",
        output_dir=tmp_path / "output",
        export=True,
        fail_fast=True,
        dd_trace=run_benchmarks_tool.DDTraceMode.none,
        dd_task_trace=False,
    )
    serialized = json.dumps(metadata, sort_keys=True)

    _assert_wandb_metadata(metadata)
    for forbidden in ("Alice", "raw secret", "raw prompt", "/private/path", "models.yaml", "providers.yaml"):
        assert forbidden not in serialized


def test_wandb_extract_scalar_metrics_rejects_trace_record_types(wandb_logging_tool: ModuleType) -> None:
    metrics = wandb_logging_tool.extract_scalar_metrics(
        {
            "record_type": "dd_message_trace",
            "workflow_name": "entity-detection",
            "messages": [{"role": "user", "content": "secret prompt"}],
        }
    )

    assert metrics == {}


def test_run_or_plan_skips_wandb_on_dry_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    spec_path = _write_simple_suite(tmp_path, suite_id="wandb-dry-run")
    _write_text_input(tmp_path)
    init_calls: list[Any] = []
    monkeypatch.setattr(
        run_benchmarks_tool,
        "initialize_benchmark_wandb_run",
        lambda *args, **kwargs: init_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        run_benchmarks_tool,
        "require_wandb",
        lambda: pytest.fail("require_wandb should not run on dry-run"),
    )

    run_benchmarks_tool.run_or_plan(
        spec_path,
        output=tmp_path / "output",
        overwrite=False,
        dry_run=True,
        export=False,
        fail_fast=False,
        wandb_settings=run_benchmarks_tool.resolve_wandb_settings(wandb_mode=run_benchmarks_tool.WandbMode.offline),
    )

    assert init_calls == []


def test_run_or_plan_does_not_finish_wandb_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    spec_path = _write_simple_suite(tmp_path, suite_id="wandb-disabled")
    _write_text_input(tmp_path)

    def fake_run_suite(
        benchmark_spec: Any,
        *,
        output_dir: Path,
        **_kwargs: Any,
    ) -> Any:
        return _benchmark_result(run_benchmarks_tool, suite_id=benchmark_spec.suite_id, output_dir=output_dir)

    monkeypatch.setattr(run_benchmarks_tool, "run_suite", fake_run_suite)
    monkeypatch.setattr(
        run_benchmarks_tool,
        "finish_benchmark_wandb_run",
        lambda: pytest.fail("disabled W&B must not finish runs"),
    )

    result = run_benchmarks_tool.run_or_plan(
        spec_path,
        output=tmp_path / "output",
        overwrite=False,
        dry_run=False,
        export=False,
        fail_fast=False,
        wandb_settings=run_benchmarks_tool.resolve_wandb_settings(wandb_mode=run_benchmarks_tool.WandbMode.disabled),
    )

    assert result.suite_id == "wandb-disabled"


def test_run_or_plan_logs_benchmark_with_fake_wandb(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    spec_path = _write_simple_suite(tmp_path, suite_id="wandb-suite")
    _write_text_input(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state, active_run=True)

    _patch_fake_benchmark_run(monkeypatch, run_benchmarks_tool, output_dir=output_dir, fake_wandb=fake_wandb)
    init_calls: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        run_benchmarks_tool,
        "initialize_benchmark_wandb_run",
        lambda settings, **kwargs: init_calls.append((settings, kwargs)) or True,
    )
    _patch_imported_wandb_finish(monkeypatch, run_benchmarks_tool, lambda: state.finished.append("finish"))

    result = run_benchmarks_tool.run_or_plan(
        spec_path,
        output=output_dir,
        overwrite=True,
        dry_run=False,
        export=True,
        fail_fast=False,
        wandb_settings=run_benchmarks_tool.resolve_wandb_settings(wandb_mode=run_benchmarks_tool.WandbMode.offline),
    )

    assert result.suite_id == "wandb-suite"
    assert len(init_calls) == 1
    assert init_calls[0][0].wandb_mode == run_benchmarks_tool.WandbMode.offline
    _assert_logged_wandb_payload(state)


def test_finalize_benchmark_wandb_run_logs_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state, active_run=True)
    _patch_imported_wandb_finish(monkeypatch, run_benchmarks_tool, lambda: state.finished.append("finish"))

    run_benchmarks_tool.finalize_benchmark_wandb_run(
        run_benchmarks_tool.WandbSettings(wandb_mode=run_benchmarks_tool.WandbMode.offline),
        finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
            measurement_path=Path(case.measurement_path),
            cases=[case],
            table_dir=None,
        ),
        run_created=True,
        wandb=fake_wandb,
    )

    _assert_logged_wandb_payload(state)
