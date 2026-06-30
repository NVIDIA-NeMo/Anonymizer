# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import logging
import os
import socket
import subprocess
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
WANDB_INGRESS_PATH = REPO_ROOT / "tools/measurement/measurement_tools/wandb_ingress.py"
WANDB_COMPLETION_PATH = REPO_ROOT / "tools/measurement/measurement_tools/wandb_completion.py"
WANDB_IMPORT_PATH = REPO_ROOT / "tools/measurement/import_wandb_run.py"
WANDB_REPORT_PATH = REPO_ROOT / "tools/measurement/create_wandb_report.py"


def load_tool(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    sys.path.insert(0, str(REPO_ROOT / "tools/measurement"))
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
def wandb_ingress_tool(request: pytest.FixtureRequest) -> ModuleType:
    return load_tool(_fixture_module_name("measurement_wandb_ingress", request), WANDB_INGRESS_PATH)


@pytest.fixture
def wandb_completion_tool(request: pytest.FixtureRequest) -> ModuleType:
    return load_tool(_fixture_module_name("measurement_wandb_completion", request), WANDB_COMPLETION_PATH)


@pytest.fixture
def wandb_import_tool(request: pytest.FixtureRequest) -> ModuleType:
    return load_tool(_fixture_module_name("measurement_wandb_import", request), WANDB_IMPORT_PATH)


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


def _fake_wandb_module(state: SimpleNamespace, *, active_run: bool = False) -> SimpleNamespace:
    def update_summary(payload: dict[str, Any]) -> None:
        state.summary_updates.append(payload)
        state.remote_summary.update(payload)

    def log(payload: dict[str, Any], **kwargs: Any) -> None:
        state.logged.append(payload)
        state.log_kwargs.append(kwargs)

    run = SimpleNamespace(
        id=None,
        config=SimpleNamespace(
            update=lambda payload, *, allow_val_change: _record_config_update(state, payload, allow_val_change),
            get=state.remote_config.get,
        ),
        summary=SimpleNamespace(update=update_summary, get=state.remote_summary.get),
        log=log,
        define_metric=lambda *args, **kwargs: state.defined_metrics.append((args, kwargs)),
        finish=None,
    )
    module = SimpleNamespace(run=run if active_run else None)

    def finish() -> None:
        state.finished.append("finish")
        module.run = None

    run.finish = finish

    def init(**kwargs: Any) -> SimpleNamespace:
        state.init_kwargs.update(kwargs)
        run.id = kwargs["id"]
        module.run = run
        return run

    module.Settings = lambda **kwargs: kwargs
    module.init = init
    module.log = lambda payload: state.logged.append(payload)
    module.Table = lambda **kwargs: kwargs.get("dataframe", kwargs)
    return module


def _record_config_update(state: SimpleNamespace, payload: dict[str, Any], allow_val_change: bool) -> None:
    assert allow_val_change is True
    state.config_updates.append(payload)
    state.remote_config.update(payload)


def _wandb_state() -> SimpleNamespace:
    return SimpleNamespace(
        init_kwargs={},
        config_updates=[],
        remote_config={},
        defined_metrics=[],
        logged=[],
        log_kwargs=[],
        summary_updates=[],
        remote_summary={},
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
                        "schema_version": 1,
                        "record_type": "run",
                        "run_id": case.case_id,
                        "run_tags": {},
                        "timestamp_unix_sec": 1.0,
                        "mode": "replace",
                        "strategy": "Redact",
                        "input_row_count": 1,
                        "source_hash": "a" * 64,
                        "input_source": {"kind": "local_file"},
                        "input_text_column": "text",
                        "input_has_id_column": False,
                        "input_has_data_summary": False,
                        "detect": {},
                        "replace": {"strategy": "redact"},
                        "rewrite": {},
                        "models": [],
                        "runtime": {},
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "record_type": "record",
                        "run_id": case.case_id,
                        "run_tags": {},
                        "timestamp_unix_sec": 2.0,
                        "mode": "replace",
                        "strategy": "Redact",
                        "row_index": 0,
                        "record_hash": "b" * 64,
                        "text_length_chars": 5,
                        "text_length_chars_bucket": "1-127",
                        "text_length_tokens": 1,
                        "text_length_tokens_bucket": "1-127",
                        "final_entity_count": 1,
                        "final_entity_label_counts": {"person": 1},
                        "utility_score": 0.95,
                    }
                ),
                json.dumps(
                    {
                        "schema_version": 1,
                        "record_type": "stage",
                        "run_id": case.case_id,
                        "run_tags": {},
                        "timestamp_unix_sec": 3.0,
                        "stage": "Anonymizer._run_internal",
                        "status": "completed",
                        "elapsed_sec": 1.5,
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


def _strict_record_payload(*, run_id: str = "run-a", **updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "record_type": "record",
        "run_id": run_id,
        "run_tags": {},
        "timestamp_unix_sec": 1.0,
        "mode": "replace",
        "strategy": "Redact",
        "row_index": 0,
        "record_hash": "a" * 64,
        "text_length_chars": 5,
        "text_length_chars_bucket": "1-127",
        "text_length_tokens": 1,
        "text_length_tokens_bucket": "1-127",
        "final_entity_count": 1,
        "final_entity_label_counts": {"person": 1},
    }
    payload.update(updates)
    return payload


def _set_wandb_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name, value in {
        "WANDB_MODE": "online",
        "WANDB_ENTITY": "global-entity",
        "WANDB_PROJECT": "global-project",
        "WANDB_BASE_URL": "https://ambient.invalid",
        "WANDB_GROUP": "global-group",
        "WANDB_JOB_TYPE": "global-job",
        "WANDB_NAME": "global-run",
        "WANDB_TAGS": "global",
        "ANONYMIZER_MEASUREMENT_WANDB_MODE": "offline",
        "ANONYMIZER_MEASUREMENT_WANDB_ENTITY": "env-entity",
        "ANONYMIZER_MEASUREMENT_WANDB_PROJECT": "env-project",
        "ANONYMIZER_MEASUREMENT_WANDB_BASE_URL": "https://env.example",
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
    settings = run_benchmarks_tool.ResolvedWandbConfig()

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
    settings = run_benchmarks_tool.ResolvedWandbConfig(
        wandb_mode=run_benchmarks_tool.WandbMode.online,
        wandb_entity="field-entity",
        wandb_project="field-project",
        wandb_base_url="https://field.example",
        wandb_group="field-group",
        wandb_job_type="field-job",
        wandb_run_name="field-run",
        wandb_tags="alpha, beta",
        wandb_log_tables=True,
    )

    assert settings.wandb_mode == run_benchmarks_tool.WandbMode.online
    assert settings.wandb_entity == "field-entity"
    assert settings.wandb_project == "field-project"
    assert settings.wandb_base_url == "https://field.example"
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
        wandb_base_url="https://cli.example",
        wandb_group="cli-group",
        wandb_job_type="cli-job",
        wandb_run_name="cli-run",
        wandb_tags="cli-a",
        wandb_log_tables=False,
    )

    assert settings.wandb_mode == run_benchmarks_tool.WandbMode.offline
    assert settings.wandb_entity == "env-entity"
    assert settings.wandb_project == "env-project"
    assert settings.wandb_base_url == "https://env.example"
    assert settings.wandb_group == "env-group"
    assert settings.wandb_job_type == "env-job"
    assert settings.wandb_run_name == "env-run"
    assert settings.effective_wandb_tags == ["env-a", "env-b"]
    assert settings.wandb_log_tables is True
    assert overridden.wandb_mode == run_benchmarks_tool.WandbMode.disabled
    assert overridden.wandb_entity == "cli-entity"
    assert overridden.wandb_project == "env-project"
    assert overridden.wandb_base_url == "https://cli.example"
    assert overridden.wandb_group == "cli-group"
    assert overridden.wandb_job_type == "cli-job"
    assert overridden.wandb_run_name == "cli-run"
    assert overridden.effective_wandb_tags == ["cli-a"]
    assert overridden.wandb_log_tables is False


def test_resolve_wandb_settings_ignores_generic_routing_env(
    monkeypatch: pytest.MonkeyPatch,
    run_benchmarks_tool: ModuleType,
) -> None:
    monkeypatch.setenv("WANDB_MODE", "online")
    monkeypatch.setenv("WANDB_PROJECT", "ambient-project")
    monkeypatch.setenv("WANDB_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("WANDB_GROUP", "ambient-group")

    settings = run_benchmarks_tool.resolve_wandb_settings()

    assert settings.wandb_mode == run_benchmarks_tool.WandbMode.disabled
    assert settings.wandb_project == "nemo-anonymizer-benchmarks"
    assert settings.wandb_base_url is None
    assert settings.wandb_group is None


def test_resolve_wandb_settings_rejects_invalid_override(run_benchmarks_tool: ModuleType) -> None:
    with pytest.raises(ValidationError):
        run_benchmarks_tool.resolve_wandb_settings(wandb_mode="invalid")
    with pytest.raises(ValidationError):
        run_benchmarks_tool.ResolvedWandbConfig(wandb_mode=run_benchmarks_tool.WandbMode.online, wandb_project=" ")
    with pytest.raises(ValidationError):
        run_benchmarks_tool.ResolvedWandbConfig(wandb_mode="offline")
    with pytest.raises(ValidationError):
        run_benchmarks_tool.ResolvedWandbConfig(wandb_log_tables="false")
    with pytest.raises(ValidationError, match="safe identifiers"):
        run_benchmarks_tool.ResolvedWandbConfig(wandb_tags="x" * 65)
    with pytest.raises(ValidationError, match="safe identifiers"):
        run_benchmarks_tool.ResolvedWandbConfig(wandb_tags="release,sk-secret-token")
    with pytest.raises(ValidationError, match="credential-free HTTP"):
        run_benchmarks_tool.ResolvedWandbConfig(
            wandb_mode=run_benchmarks_tool.WandbMode.online,
            wandb_base_url="https://user:secret@wandb.example.com?token=secret",
        )
    with pytest.raises(ValidationError, match="HTTPS unless it targets loopback"):
        run_benchmarks_tool.ResolvedWandbConfig(
            wandb_mode=run_benchmarks_tool.WandbMode.online,
            wandb_base_url="http://wandb.internal.example",
        )
    assert (
        run_benchmarks_tool.ResolvedWandbConfig(
            wandb_mode=run_benchmarks_tool.WandbMode.online,
            wandb_base_url="http://127.0.0.1:8080",
        ).wandb_base_url
        == "http://127.0.0.1:8080"
    )


def test_wandb_require_wandb_reports_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
    wandb_setup_tool: ModuleType,
) -> None:
    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            raise ImportError("No module named 'wandb'")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    with (
        wandb_setup_tool.WandbSdkEnvironment(
            wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline)
        ),
        pytest.raises(ImportError, match="wandb package is not installed"),
    ):
        wandb_setup_tool.require_wandb()


def test_wandb_report_run_path_accepts_path_and_url(wandb_report_tool: ModuleType) -> None:
    path = wandb_report_tool.parse_wandb_run_path("entity/project/abc123")
    url = wandb_report_tool.parse_wandb_run_path("https://wandb.ai/entity/project/runs/abc123")

    assert path.entity == "entity"
    assert path.project == "project"
    assert path.run_id == "abc123"
    assert path.path == "entity/project/abc123"
    assert url == path
    with pytest.raises(ValueError, match="credential-free"):
        wandb_report_tool.parse_wandb_run_path("https://user:secret@wandb.example/entity/project/runs/abc123")
    with pytest.raises(ValidationError, match=r"HTTP\(S\) origin"):
        wandb_report_tool.WandbRunPath(
            entity="entity",
            project="project",
            run_id="abc123",
            base_url="javascript:alert(1)",
        )


def test_wandb_report_project_path_accepts_entity_project(wandb_report_tool: ModuleType) -> None:
    project = wandb_report_tool.parse_wandb_project_path("entity/project")

    assert project.entity == "entity"
    assert project.project == "project"
    assert project.path == "entity/project"


def test_wandb_report_sweep_columns_include_benchmark_metric_groups(wandb_report_tool: ModuleType) -> None:
    comparison = wandb_report_tool.GroupComparison(
        run_kind="sweep_arm",
        config_key="sweep_arm_id",
        label="Sweep Arm",
    )
    columns = wandb_report_tool._group_visible_columns(comparison)

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

    panels = wandb_report_tool._benchmark_panels(FakeWr, groupby=FakeConfig("sweep_arm_id"))
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
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    workspace = wandb_report_tool.build_benchmark_workspace(
        wandb_report_tool.WandbProjectPath(entity="entity", project="project"),
        group="threshold",
        title="Benchmark Workspace",
        ws=ws,
        wr=wr,
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
    from wandb_workspaces import expr

    report = wandb_report_tool.build_benchmark_group_report(
        wandb_report_tool.WandbProjectPath(entity="entity", project="project"),
        group="threshold",
        title="Sweep",
        description="Offline serialization test",
        wr=wr,
        expr=expr,
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
    summary, config = _wandb_report_fixture()
    run_path = wandb_report_tool.WandbRunPath(entity="entity", project="project", run_id="run-a")
    view = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(id="run-a", name="run-a", group=None, job_type="benchmark", summary=summary, config=config),
        run_path=run_path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )

    markdown = wandb_report_tool.build_report_markdown(view)

    assert "2/2 cases completed" in markdown
    assert "Final entities" in markdown
    assert "source_suffix=`.csv`" in markdown
    assert "/secret/path.csv" not in markdown
    assert "sk-secret-token" not in markdown


def test_wandb_report_views_parse_v1_and_v2_with_explicit_axes(wandb_report_tool: ModuleType) -> None:
    summary, v1_config = _wandb_report_fixture()
    path = wandb_report_tool.WandbRunPath(entity="entity", project="project", run_id="run-a")
    v1 = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(id="run-a", name="v1", group=None, job_type="benchmark", summary=summary, config=v1_config),
        run_path=path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )
    v2_config = {
        "run_kind": "imported_case",
        "benchmark": {"metadata_schema_version": 2, "suite_id": "case-a", "case_count": 1},
        "configs": [{"id": "config-a"}],
        "imported": {
            "completion_seal_schema_version": 1,
            "completion_seal_sha256": "a" * 64,
            "producer_repository": "anonymizer-experiments",
            "producer_commit": "b" * 40,
            "phase": "baseline",
            "case_id": "case-a",
        },
    }
    v2 = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(
            id="run-a", name="v2", group=None, job_type="benchmark-import", summary=summary, config=v2_config
        ),
        run_path=path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )

    assert v1.schema_version == 1
    assert v1.comparison.kind == "native_suite"
    assert v2.schema_version == 2
    assert v2.comparison.kind == "imported_case"
    assert v2.comparison.config_id == "config-a"
    assert wandb_report_tool.group_comparison([v2]).config_key == "imported_config_id"


@pytest.mark.parametrize(
    "fixture_name,schema_version,run_kind",
    [
        ("v1_native", 1, "native_suite"),
        ("v1_sweep", 1, "sweep_arm"),
        ("v2_native", 2, "native_suite"),
        ("v2_sweep", 2, "sweep_arm"),
        ("v2_import", 2, "imported_case"),
    ],
)
def test_wandb_report_acceptance_fixtures_render(
    wandb_report_tool: ModuleType,
    fixture_name: str,
    schema_version: int,
    run_kind: str,
) -> None:
    fixture = _load_measurement_fixture("wandb-report-runs.json")[fixture_name]
    run_path = wandb_report_tool.WandbRunPath(
        entity="entity",
        project="project",
        run_id=fixture["id"],
    )
    view = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(**fixture),
        run_path=run_path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )
    rendered = wandb_report_tool.build_report_markdown(view)

    assert view.schema_version == schema_version
    assert view.comparison.kind == run_kind
    assert "### Run Summary" in rendered
    assert run_path.url in rendered


@pytest.mark.parametrize(
    "run_kind,extra,expected_axis",
    [
        ("native_suite", {}, "native_suite_id"),
        (
            "sweep_arm",
            {
                "sweep": {
                    "id": "sweep-a",
                    "arm_id": "arm-000",
                    "params": {"configs_all_detect_gliner_threshold": 0.3},
                }
            },
            "sweep_arm_id",
        ),
    ],
)
def test_wandb_report_v2_native_and_sweep_axes(
    wandb_report_tool: ModuleType,
    run_kind: str,
    extra: dict[str, Any],
    expected_axis: str,
) -> None:
    summary, _config = _wandb_report_fixture()
    config = {
        "run_kind": run_kind,
        "benchmark": {"metadata_schema_version": 2, "suite_id": "suite-a", "case_count": 1},
        **extra,
    }
    path = wandb_report_tool.WandbRunPath(entity="entity", project="project", run_id="run-a")
    view = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(id="run-a", name="run", group="group", job_type="benchmark", summary=summary, config=config),
        run_path=path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )

    assert wandb_report_tool.group_comparison([view]).config_key == expected_axis


def test_wandb_report_preserves_typed_execution_metadata(wandb_report_tool: ModuleType) -> None:
    summary, _config = _wandb_report_fixture()
    config = {
        "run_kind": "native_suite",
        "benchmark": {"metadata_schema_version": 2, "suite_id": "suite-a", "case_count": 1},
        "execution": {
            "backend": "slurm",
            "export": True,
            "slurm": {"job_id": "123", "array_task_id": "4"},
        },
    }
    path = wandb_report_tool.WandbRunPath(entity="entity", project="project", run_id="run-a")

    view = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(id="run-a", name="run", group=None, job_type="benchmark", summary=summary, config=config),
        run_path=path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )

    assert view.metadata.execution is not None
    assert view.metadata.execution.backend == "slurm"
    assert view.metadata.execution.slurm is not None
    assert view.metadata.execution.slurm.array_task_id == "4"


def test_wandb_report_markdown_escapes_hostile_remote_values(wandb_report_tool: ModuleType) -> None:
    summary, config = _wandb_report_fixture()
    config["benchmark"]["suite_id"] = "suite|`x`<script>"
    config["workloads"][0]["id"] = "row|\n# injected"
    path = wandb_report_tool.WandbRunPath(entity="entity", project="project", run_id="safe-id")
    view = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(
            id="safe-id",
            name="run](https://evil.example)\n# injected",
            group=None,
            job_type="benchmark",
            url="https://evil.example/stolen",
            summary=summary,
            config=config,
        ),
        run_path=path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )

    markdown = wandb_report_tool.build_report_markdown(view)

    assert "](https://wandb.ai/entity/project/runs/safe-id)" in markdown
    assert "https://evil.example/stolen" not in markdown
    assert "<script>" not in markdown
    assert "\n# injected" not in markdown
    assert "&#124;" in markdown


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "https://evil.example/report",
        "https://wandb.ai/report\x1b]8;;https://evil.example\x07",
    ],
)
def test_wandb_report_rejects_unsafe_sdk_urls(wandb_report_tool: ModuleType, url: str) -> None:
    with pytest.raises(ValueError, match="unsafe URL|configured origin"):
        wandb_report_tool.validate_wandb_returned_url(url, expected_base_url="https://wandb.ai")


def test_wandb_report_rejects_mixed_or_ambiguous_groups(wandb_report_tool: ModuleType) -> None:
    summary, config = _wandb_report_fixture()
    path = wandb_report_tool.WandbRunPath(entity="entity", project="project", run_id="run-a")
    native = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(id="run-a", name="native", group="group", job_type="benchmark", summary=summary, config=config),
        run_path=path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )
    sweep_config = {
        **config,
        "sweep_id": "sweep-a",
        "sweep_arm_id": "arm-000",
        "sweep_param_configs_all_detect_gliner_threshold": 0.3,
    }
    sweep = wandb_report_tool.parse_wandb_run_view(
        SimpleNamespace(
            id="run-a", name="sweep", group="group", job_type="benchmark-sweep", summary=summary, config=sweep_config
        ),
        run_path=path,
        allowed_metrics=frozenset(wandb_report_tool._all_report_metrics()),
    )

    with pytest.raises(ValueError, match="mixed run kinds"):
        wandb_report_tool.group_comparison([native, sweep])
    assert wandb_report_tool.group_comparison([sweep]).config_key == "sweep_arm_id"


def test_wandb_report_cli_uses_typed_report_path(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    wandb_report_tool: ModuleType,
) -> None:
    calls: list[Any] = []

    def fake_create(run_path: Any, **kwargs: Any) -> Any:
        calls.append((run_path, kwargs))
        return wandb_report_tool.WandbReportResult(
            run_path=run_path.path,
            run_url=run_path.url,
            project_path=f"{run_path.entity}/{run_path.project}",
            report_url="https://wandb.ai/report",
            draft=True,
            title="Report",
        )

    monkeypatch.setattr(wandb_report_tool, "create_benchmark_report", fake_create)
    wandb_report_tool.main("entity/project/run")

    assert calls[0][0].run_id == "run"
    assert "Created draft W&B report" in capsys.readouterr().out


def test_wandb_report_cli_redacts_sdk_errors(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    wandb_report_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"
    monkeypatch.setattr(
        wandb_report_tool,
        "create_benchmark_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(canary)),
    )

    with caplog.at_level(logging.ERROR, logger="measurement.wandb_report"), pytest.raises(SystemExit) as raised:
        wandb_report_tool.main("entity/project/run")

    assert raised.value.code == 1
    assert canary not in caplog.text
    assert "ValueError" in caplog.text


def test_wandb_report_api_read_uses_guarded_resolved_environment(
    monkeypatch: pytest.MonkeyPatch,
    wandb_report_tool: ModuleType,
) -> None:
    summary, config = _wandb_report_fixture()
    config["run_kind"] = "native_suite"
    config["benchmark"]["metadata_schema_version"] = 2
    run = SimpleNamespace(
        id="run-a",
        name="run-a",
        group=None,
        job_type="benchmark",
        summary=summary,
        config=config,
    )
    observed_environment: list[dict[str, str]] = []

    class FakeApi:
        def __init__(self, *, timeout: int) -> None:
            assert timeout == 60

        def run(self, path: str) -> Any:
            observed_environment.append(dict(os.environ))
            assert path == "entity/project/run-a"
            return run

    fake_wandb = SimpleNamespace(Api=FakeApi)
    monkeypatch.setattr(wandb_report_tool, "require_wandb_report_sdk", lambda: (fake_wandb, None, None))
    monkeypatch.setattr(
        wandb_report_tool,
        "build_benchmark_report",
        lambda *_args, **_kwargs: SimpleNamespace(url="https://resolved.example/report"),
    )
    monkeypatch.setattr(wandb_report_tool, "_save_report", lambda *_args, **_kwargs: None)
    monkeypatch.setenv("WANDB_NOTES", "ambient-note")
    monkeypatch.setenv("WANDB_BASE_URL", "https://ambient.invalid")
    before = dict(os.environ)
    settings = wandb_report_tool.ResolvedWandbConfig(
        wandb_mode=wandb_report_tool.WandbMode.online,
        wandb_project="project",
        wandb_entity="entity",
        wandb_base_url="https://resolved.example",
    )

    result = wandb_report_tool.create_benchmark_report(
        wandb_report_tool.WandbRunPath(entity="entity", project="project", run_id="run-a"),
        settings=settings,
    )

    assert result.run_url == "https://resolved.example/entity/project/runs/run-a"
    assert observed_environment[0]["WANDB_BASE_URL"] == "https://resolved.example"
    assert observed_environment[0]["WANDB_ERROR_REPORTING"] == "false"
    assert "WANDB_NOTES" not in observed_environment[0]
    assert dict(os.environ) == before


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
    assert materialized["run_tags"]["wandb_sweep"] == {
        "id": "threshold-digest",
        "arm_id": "arm-000",
        "params": {
            "configs_all_detect_gliner_threshold": 0.2,
            "configs_hash_replace_digest_length": 8,
        },
    }
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


def test_sweep_run_creates_workspace_after_typed_report_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite)
    calls: list[tuple[Path, Path, Any]] = []
    workspace_calls: list[tuple[Any, Any, str]] = []
    _patch_sweep_runner(monkeypatch, sweep_tool, calls, status=sweep_tool.run_benchmarks.CaseStatus.completed)

    def fake_workspace(project: Any, *, settings: Any, group: str, expected_run_kind: str) -> Any:
        workspace_calls.append((project, settings, group))
        assert expected_run_kind == "sweep_arm"
        return SimpleNamespace(workspace_url="https://wandb.ai/workspace")

    monkeypatch.setattr(sweep_tool, "create_benchmark_group_workspace", fake_workspace)
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

    assert len(calls) == 2
    assert workspace_calls[0][0].path == "entity/project"
    assert workspace_calls[0][2] == "threshold"
    assert result.workspace_url == "https://wandb.ai/workspace"


def test_sweep_preserves_arm_results_when_report_creation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite)
    calls: list[tuple[Path, Path, Any]] = []
    _patch_sweep_runner(monkeypatch, sweep_tool, calls, status=sweep_tool.run_benchmarks.CaseStatus.completed)
    monkeypatch.setattr(
        sweep_tool,
        "create_benchmark_group_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("remote secret response")),
    )

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
        create_report=True,
    )

    assert result.completed_arms == 2
    assert result.report_url is None
    assert result.report_error == "W&B report creation failed (RuntimeError)"
    assert "remote secret response" not in result.model_dump_json()


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
            "wandb_sweep": {
                "id": "threshold",
                "arm_id": "arm-000",
                "params": {"configs_all_detect_gliner_threshold": 0.2},
            },
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

    assert metadata.sweep is not None
    assert metadata.sweep.id == "threshold"
    assert metadata.sweep.arm_id == "arm-000"
    assert metadata.sweep.params["configs_all_detect_gliner_threshold"] == 0.2


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
    serialized = metadata.model_dump_json()

    assert metadata.execution is not None
    assert metadata.execution.backend == "slurm"
    assert metadata.execution.dd_trace == "last_message"
    assert metadata.execution.slurm is not None
    assert metadata.execution.slurm.job_id == "98765"
    assert metadata.execution.slurm.array_task_id == "3"
    assert str(tmp_path) not in serialized
    assert "private-node" not in serialized


def test_wandb_typed_metrics_and_tables_exclude_local_fields(
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"
    record = wandb_ingress_tool.RecordMeasurement.model_validate(
        _strict_record_payload(run_tags={"customer": canary}),
        strict=True,
    )

    metrics = wandb_logging_tool.extract_scalar_metrics(record)
    tables = wandb_logging_tool._typed_tables((record,))
    serialized = json.dumps(
        {"metrics": metrics, "tables": [table.model_dump(mode="json") for table in tables]},
        sort_keys=True,
    )

    assert metrics["measurement/record/final_entity_count"] == 1
    assert "text_length_chars" not in tables[0].columns
    assert "text_length_chars_bucket" in tables[0].columns
    assert canary not in serialized


@pytest.mark.parametrize("field", ["strategy", "row_index"])
def test_wandb_typed_ingress_rejects_arbitrary_table_strings(
    field: str,
    wandb_ingress_tool: ModuleType,
) -> None:
    with pytest.raises(ValidationError):
        wandb_ingress_tool.RecordMeasurement.model_validate(
            _strict_record_payload(**{field: "alice@example.com/private/input"}),
            strict=True,
        )


def test_wandb_aggregates_with_explicit_metric_policy(
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
) -> None:
    records = tuple(
        wandb_ingress_tool.RecordMeasurement.model_validate(payload, strict=True)
        for payload in (
            _strict_record_payload(
                final_entity_count=2,
                utility_score=0.8,
                leakage_mass=0.4,
                weighted_leakage_rate=0.1,
                entity_precision=0.5,
                entity_recall=0.25,
                entity_f1=1 / 3,
            ),
            _strict_record_payload(
                timestamp_unix_sec=2.0,
                final_entity_count=1,
                utility_score=1.0,
                leakage_mass=0.2,
                weighted_leakage_rate=0.3,
                entity_precision=1.0,
                entity_recall=0.75,
                entity_f1=6 / 7,
            ),
        )
    )
    aggregated = wandb_logging_tool.aggregate_measurement_scalars(records)

    assert aggregated["measurement/record/final_entity_count"] == 3.0
    assert aggregated["measurement/record/utility_score_mean"] == pytest.approx(0.9)
    assert aggregated["measurement/record/leakage_mass_mean"] == pytest.approx(0.3)
    assert aggregated["measurement/record/weighted_leakage_rate_mean"] == pytest.approx(0.2)
    assert aggregated["measurement/record/entity_precision_mean"] == pytest.approx(0.75)
    assert aggregated["measurement/record/entity_recall_mean"] == pytest.approx(0.5)
    assert aggregated["measurement/record/entity_f1_mean"] == pytest.approx(25 / 42)
    assert not any("schema_version" in key for key in aggregated)
    assert not any("timestamp_unix_sec" in key for key in aggregated)
    assert not any("input_has_id_column" in key for key in aggregated)


def test_wandb_report_metric_names_match_aggregated_names(
    wandb_ingress_tool: ModuleType,
    wandb_logging_tool: ModuleType,
    wandb_report_tool: ModuleType,
) -> None:
    record = wandb_ingress_tool.RecordMeasurement.model_validate(
        _strict_record_payload(
            entity_precision=0.9,
            entity_recall=0.8,
            entity_f1=0.85,
            leakage_mass=0.1,
            repair_iterations=2,
        ),
        strict=True,
    )
    aggregated = wandb_logging_tool.aggregate_measurement_scalars((record,))

    expected = {
        "measurement/record/entity_precision_mean",
        "measurement/record/entity_recall_mean",
        "measurement/record/entity_f1_mean",
        "measurement/record/leakage_mass_mean",
        "measurement/record/repair_iterations_mean",
    }
    assert expected <= set(aggregated)
    assert expected <= set(wandb_report_tool._all_report_metrics())


def test_wandb_config_excludes_suite_run_tags(wandb_setup_tool: ModuleType) -> None:
    config = wandb_setup_tool.WandbConfigPayload.from_run_metadata(
        wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
        suite_id="suite-a",
        metadata=wandb_setup_tool.WandbRunMetadata(benchmark=wandb_setup_tool.BenchmarkMetadata(suite_id="suite-a")),
    )

    assert "run_tags" not in config.sdk_values()


def test_wandb_config_projects_only_declared_metadata(wandb_setup_tool: ModuleType) -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        wandb_setup_tool.WandbRunMetadata.model_validate(
            {"benchmark": {"suite_id": "suite-a", "owner_email": "alice@example.com"}},
            strict=True,
        )

    metadata = wandb_setup_tool.WandbRunMetadata.model_validate(
        {
            "run_kind": "sweep_arm",
            "benchmark": {"suite_id": "suite-a", "case_count": 2, "suite_file_hash": "content-hash"},
            "execution": {"backend": "slurm", "export": True, "slurm": {"job_id": "123"}},
            "workloads": [{"id": "workload-a", "row_limit": 5, "source": {"kind": "local_file", "suffix": ".csv"}}],
            "configs": [
                {
                    "id": "redact",
                    "mode": "replace",
                    "detect": {"gliner_threshold": 0.3, "entity_label_count": 4},
                    "replace": {"strategy": "redact"},
                }
            ],
            "matrix": [{"workload": "workload-a", "config": "redact", "repetitions": 1}],
            "sweep": {
                "id": "threshold-sweep",
                "arm_id": "arm-001",
                "params": {"configs_all_detect_gliner_threshold": 0.3},
            },
        },
        strict=True,
    )
    config = wandb_setup_tool.WandbConfigPayload.from_run_metadata(
        wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
        suite_id="suite-a",
        metadata=metadata,
    )

    assert config.benchmark == metadata.benchmark
    assert config.execution == metadata.execution
    assert config.sweep_params == {"configs_all_detect_gliner_threshold": 0.3}
    assert config.sdk_values()["sweep_param_configs_all_detect_gliner_threshold"] == 0.3


def test_wandb_run_tags_filter_sensitive_generated_values(wandb_setup_tool: ModuleType) -> None:
    metadata = wandb_setup_tool.WandbRunMetadata.model_validate(
        {
            "benchmark": {"suite_id": "suite-a"},
            "git": {"branch": "feature/token-fix", "dirty": False},
        },
        strict=True,
    )
    tags = wandb_setup_tool._effective_wandb_tags(
        wandb_setup_tool.ResolvedWandbConfig(
            wandb_mode=wandb_setup_tool.WandbMode.offline,
            wandb_tags="tag-a,tag-b",
        ),
        suite_id="suite-a",
        metadata=metadata,
    )

    assert tags == ["tag-a", "tag-b", "suite:suite-a", "clean"]


def test_wandb_run_tags_bound_long_slurm_case_identity(wandb_setup_tool: ModuleType) -> None:
    case_id = "rat_bench-diff1__val-gpt-oss-120b-low__aug-gpt-oss-120b-medium__rep-gpt-oss-120b-low__r000"
    metadata = wandb_setup_tool.WandbRunMetadata.model_validate(
        {
            "run_kind": "imported_case",
            "benchmark": {"suite_id": case_id},
            "configs": [{"id": "config-a"}],
            "imported": {
                "completion_seal_schema_version": 1,
                "completion_seal_sha256": "a" * 64,
                "producer_repository": "anonymizer-experiments",
                "producer_commit": "b" * 40,
                "phase": "baseline",
                "case_id": case_id,
            },
        },
        strict=True,
    )

    tags = wandb_setup_tool._effective_wandb_tags(
        wandb_setup_tool.ResolvedWandbConfig(
            wandb_mode=wandb_setup_tool.WandbMode.offline,
            wandb_tags="release",
        ),
        suite_id=case_id,
        metadata=metadata,
    )

    assert tags == ["release", "suite:rat_bench-diff1__val-gpt-oss-120b-low__aug-gp-cc0ea432483a"]
    assert all(1 <= len(tag) <= 64 for tag in tags)


def test_wandb_init_payload_rejects_sdk_invalid_tag_length(
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    with pytest.raises(ValidationError, match="tags.0"):
        wandb_setup_tool.WandbInitPayload(
            run_id="run-a",
            project="project-a",
            name="run-a",
            mode=wandb_setup_tool.WandbMode.offline,
            directory=tmp_path,
            group="group-a",
            job_type="benchmark",
            tags=("x" * 65,),
        )


def test_wandb_environment_isolates_routing_and_restores_exactly(
    monkeypatch: pytest.MonkeyPatch,
    wandb_setup_tool: ModuleType,
) -> None:
    monkeypatch.setenv("WANDB_GROUP", "ambient-group")
    monkeypatch.setenv("WANDB_API_KEY", "auth-token")
    monkeypatch.setenv("WANDB_ERROR_REPORTING", "true")
    monkeypatch.setenv("UNRELATED", "unchanged")
    before = dict(os.environ)
    settings = wandb_setup_tool.ResolvedWandbConfig(
        wandb_mode=wandb_setup_tool.WandbMode.offline,
        wandb_project="resolved-project",
        wandb_group="resolved-group",
    )

    with wandb_setup_tool.WandbSdkEnvironment(settings):
        assert os.environ["WANDB_GROUP"] == "resolved-group"
        assert os.environ["WANDB_PROJECT"] == "resolved-project"
        assert "WANDB_API_KEY" not in os.environ
        assert os.environ["WANDB_ERROR_REPORTING"] == "false"
        assert "UNRELATED" not in os.environ
        with pytest.raises(RuntimeError, match="nested or concurrent"):
            with wandb_setup_tool.WandbSdkEnvironment(settings):
                pass

    assert dict(os.environ) == before


def test_wandb_environment_uses_namespaced_base_url(
    monkeypatch: pytest.MonkeyPatch,
    wandb_setup_tool: ModuleType,
) -> None:
    monkeypatch.setenv("WANDB_BASE_URL", "https://ambient.invalid")
    monkeypatch.setenv("ANONYMIZER_MEASUREMENT_WANDB_BASE_URL", "https://resolved.example")
    settings = wandb_setup_tool.ResolvedWandbConfig.from_env_and_overrides()

    with wandb_setup_tool.WandbSdkEnvironment(settings):
        assert os.environ["WANDB_BASE_URL"] == "https://resolved.example"


def test_wandb_settings_validation_does_not_echo_credential_url(wandb_setup_tool: ModuleType) -> None:
    canary = "alice:super-secret"

    with pytest.raises(ValidationError) as raised:
        wandb_setup_tool.ResolvedWandbConfig(wandb_base_url=f"https://{canary}@example.com")

    assert canary not in str(raised.value)


def test_wandb_cli_validation_does_not_log_credential_url(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    run_benchmarks_tool: ModuleType,
) -> None:
    canary = "alice:super-secret"

    with caplog.at_level(logging.ERROR, logger="measurement.benchmark"), pytest.raises(SystemExit) as raised:
        run_benchmarks_tool.main(
            tmp_path / "unused-suite.yaml",
            wandb_base_url=f"https://{canary}@example.com",
        )

    assert raised.value.code == 125
    assert canary not in caplog.text
    assert "wandb_base_url:value_error" in caplog.text


def test_wandb_native_failure_log_does_not_echo_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"

    def fail_publish(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError(canary)

    monkeypatch.setattr(wandb_setup_tool.WandbPublisher, "publish", fail_publish)
    with caplog.at_level(logging.WARNING, logger="measurement.wandb"):
        result = wandb_setup_tool.publish_benchmark_wandb_best_effort(
            wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=wandb_setup_tool.BenchmarkWandbFinalization(
                measurement_path=tmp_path / "missing.jsonl",
                cases=[],
            ),
        )

    assert result.published is False
    assert canary not in caplog.text
    assert "ValueError" in caplog.text


def test_wandb_outbound_models_structurally_exclude_sensitive_fields(wandb_setup_tool: ModuleType) -> None:
    models = sys.modules[wandb_setup_tool.WandbPublishPayload.__module__]

    with pytest.raises(ValidationError, match="Extra inputs"):
        models.WandbHistoryPayload(metrics={}, text="Alice")
    with pytest.raises(ValidationError, match="no aggregate exposure policy"):
        models.WandbHistoryPayload(metrics={"measurement/record/text": "Alice"})
    with pytest.raises(ValidationError, match="Extra inputs"):
        models.WandbConfigPayload(
            suite_id="suite",
            wandb_mode=models.WandbMode.offline,
            wandb_log_tables=False,
            benchmark={"prompt": "Alice"},
        )
    assert all(
        policy.exposure != models.Exposure.never
        for policies in models.OUTBOUND_FIELD_POLICIES.values()
        for policy in policies.values()
    )
    assert set(models._METRIC_TABLE_POLICY_FIELDS) == set(models._MetricTableRow.model_fields)
    with pytest.raises(ValidationError):
        models.BenchmarkMetadata(case_retry_backoff_sec=-1.0)
    with pytest.raises(ValidationError):
        models.DetectMetadata(gliner_threshold=1.1)
    with pytest.raises(ValidationError):
        models.SweepMetadata(
            id="sweep",
            arm_id="arm",
            params={"configs_all_detect_gliner_threshold": -0.1},
        )


def test_wandb_publisher_validates_before_import(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    measurements = tmp_path / "measurements.jsonl"
    measurements.write_text('{"record_type":"record","unknown":"secret"}\n', encoding="utf-8")
    monkeypatch.setattr(wandb_setup_tool, "require_wandb", lambda: pytest.fail("SDK import must be post-validation"))

    with pytest.raises(ValueError, match="invalid measurement record"):
        wandb_setup_tool.WandbPublisher().publish(
            wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=wandb_setup_tool.BenchmarkWandbFinalization(
                measurement_path=measurements,
                cases=[],
            ),
        )


def test_wandb_snapshot_uses_one_descriptor_and_enforces_limits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    record = _strict_record_payload()
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    reads = 0
    real_read = wandb_ingress_tool.os.read

    def counted_read(descriptor: int, count: int) -> bytes:
        nonlocal reads
        reads += 1
        return real_read(descriptor, count)

    monkeypatch.setattr(wandb_ingress_tool.os, "read", counted_read)
    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)

    assert reads >= 1
    assert len(snapshot.records) == 1
    with pytest.raises(ValueError, match="byte limit"):
        wandb_ingress_tool.read_measurement_snapshot(path, max_bytes=1)
    with pytest.raises(ValueError, match="record limit"):
        wandb_ingress_tool.read_measurement_snapshot(path, max_records=0)


def test_wandb_snapshot_rejects_symlink_and_special_file(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    target = tmp_path / "target.jsonl"
    target.write_text("", encoding="utf-8")
    symlink = tmp_path / "link.jsonl"
    symlink.symlink_to(target)

    with pytest.raises(ValueError, match="symlink"):
        wandb_ingress_tool.read_measurement_snapshot(symlink)
    with pytest.raises(ValueError, match="regular file"):
        wandb_ingress_tool.read_measurement_snapshot(tmp_path)


def test_wandb_snapshot_rejects_fifo_socket_and_device(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    fifo = tmp_path / "measurements.fifo"
    os.mkfifo(fifo)
    unix_socket = tmp_path / "measurements.sock"
    listener = socket.socket(socket.AF_UNIX)
    listener.bind(str(unix_socket))
    try:
        for path in (fifo, unix_socket, Path("/dev/null")):
            with pytest.raises(ValueError, match="regular file|safely open"):
                wandb_ingress_tool.read_measurement_snapshot(path)
    finally:
        listener.close()


def test_wandb_snapshot_rejects_parent_symlink_and_hard_link(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    source = real_dir / "measurements.jsonl"
    source.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    redirected = tmp_path / "redirected"
    redirected.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        wandb_ingress_tool.read_measurement_snapshot(redirected / source.name)

    hard_link = tmp_path / "hard-link.jsonl"
    os.link(source, hard_link)
    with pytest.raises(ValueError, match="hard links"):
        wandb_ingress_tool.read_measurement_snapshot(source)


def test_wandb_snapshot_uses_pinned_parent_descriptor_during_path_swap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = source_dir / "measurements.jsonl"
    source.write_text(json.dumps(_strict_record_payload(run_id="trusted")) + "\n", encoding="utf-8")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    (replacement / source.name).write_text(
        json.dumps(_strict_record_payload(run_id="redirected")) + "\n",
        encoding="utf-8",
    )
    moved = tmp_path / "source-moved"
    real_open = wandb_ingress_tool.os.open
    swapped = False

    def swapping_open(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags, mode, dir_fd=dir_fd)
        if path == source_dir.name and dir_fd is not None and not swapped:
            source_dir.rename(moved)
            source_dir.symlink_to(replacement, target_is_directory=True)
            swapped = True
        return descriptor

    monkeypatch.setattr(wandb_ingress_tool.os, "open", swapping_open)

    snapshot = wandb_ingress_tool.read_measurement_snapshot(source)

    assert snapshot.records[0].run_id == "trusted"


def test_wandb_staging_rejects_parent_symlink(tmp_path: Path, wandb_setup_tool: ModuleType) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    redirected = tmp_path / "redirected"
    redirected.symlink_to(real_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="cannot contain symlinks"):
        wandb_setup_tool.prepare_wandb_staging_dir(redirected)


def test_wandb_staging_rejects_untrusted_writable_output(tmp_path: Path, wandb_setup_tool: ModuleType) -> None:
    output = tmp_path / "world-writable"
    output.mkdir(mode=0o777)
    output.chmod(0o777)

    with pytest.raises(ValueError, match="untrusted directories"):
        wandb_setup_tool.prepare_wandb_staging_dir(output)


def test_wandb_snapshot_rejects_line_nesting_negative_and_wrong_shape(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    payload = _strict_record_payload(run_tags={"nested": {"deeper": {"value": 1}}})
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="line byte limit"):
        wandb_ingress_tool.read_measurement_snapshot(path, max_line_bytes=8)
    with pytest.raises(ValueError, match="nesting limit"):
        wandb_ingress_tool.read_measurement_snapshot(path, max_nesting=2)

    path.write_text(json.dumps(_strict_record_payload(final_entity_count=-1)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="record.final_entity_count"):
        wandb_ingress_tool.read_measurement_snapshot(path)

    path.write_text(json.dumps(_strict_record_payload(stage="RewriteWorkflow.run")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="schema violation"):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_snapshot_schema_error_does_not_echo_rejected_value(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(_strict_record_payload(unknown=canary)) + "\n", encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        wandb_ingress_tool.read_measurement_snapshot(path)

    assert canary not in str(raised.value)
    assert "record.unknown" in str(raised.value)


def test_wandb_snapshot_rejects_source_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    real_read = wandb_ingress_tool.os.read

    def mutate_after_read(descriptor: int, count: int) -> bytes:
        payload = real_read(descriptor, count)
        path.write_bytes(payload + b" ")
        return payload

    monkeypatch.setattr(wandb_ingress_tool.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="changed while being read"):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_snapshot_rejects_source_truncation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    real_read = wandb_ingress_tool.os.read
    truncated = False

    def truncate_after_read(descriptor: int, count: int) -> bytes:
        nonlocal truncated
        payload = real_read(descriptor, count)
        if payload and not truncated:
            path.write_bytes(b"")
            truncated = True
        return payload

    monkeypatch.setattr(wandb_ingress_tool.os, "read", truncate_after_read)
    with pytest.raises(ValueError, match="changed while being read"):
        wandb_ingress_tool.read_measurement_snapshot(path)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (
            _strict_record_payload(unknown="secret"),
            "schema violation",
        ),
        (
            {
                "schema_version": 1,
                "record_type": "dd_message_trace",
                "run_id": "run-a",
                "run_tags": {},
                "timestamp_unix_sec": 1.0,
            },
            "record_type",
        ),
        (
            _strict_record_payload(final_entity_count="1"),
            "schema violation",
        ),
    ],
)
def test_wandb_snapshot_rejects_unknown_and_trace_records(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
    payload: dict[str, Any],
    match: str,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_snapshot_rejects_nonfinite_and_mixed_runs(
    tmp_path: Path,
    wandb_ingress_tool: ModuleType,
) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(
        json.dumps(_strict_record_payload(utility_score=float("nan"))) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite"):
        wandb_ingress_tool.read_measurement_snapshot(path)

    path.write_text(
        json.dumps(_strict_record_payload(run_id="run-a"))
        + "\n"
        + json.dumps(_strict_record_payload(run_id="run-b"))
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mixed run identities"):
        wandb_ingress_tool.read_measurement_snapshot(path)


def test_wandb_snapshot_rejects_case_status_mismatch(tmp_path: Path, wandb_ingress_tool: ModuleType) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "stage",
                "run_id": "run-a",
                "run_tags": {},
                "timestamp_unix_sec": 1.0,
                "stage": "Anonymizer._run_internal",
                "status": "error",
                "elapsed_sec": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="status does not match"):
        wandb_ingress_tool.read_measurement_snapshot(path, expected_statuses={"run-a": "completed"})


def test_wandb_snapshot_requires_exactly_one_terminal_stage(tmp_path: Path, wandb_ingress_tool: ModuleType) -> None:
    path = tmp_path / "measurements.jsonl"
    path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly one terminal stage"):
        wandb_ingress_tool.read_measurement_snapshot(path, expected_statuses={"run-a": "completed"})


def test_completion_seal_round_trip_and_digest_verification(
    tmp_path: Path,
    wandb_completion_tool: ModuleType,
) -> None:
    measurement_path = tmp_path / "measurements.jsonl"
    measurement_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "stage",
                "run_id": "dataset__config__r000",
                "run_tags": {
                    "case_id": "dataset__config__r000",
                    "workload_id": "dataset-split",
                    "config_id": "config",
                    "repetition": 0,
                },
                "timestamp_unix_sec": 1.0,
                "stage": "Anonymizer._run_internal",
                "status": "completed",
                "elapsed_sec": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    seal = wandb_completion_tool.build_completion_seal(
        snapshot,
        case=wandb_completion_tool.ImportedCaseIdentity(
            case_id="dataset__config__r000",
            workload_id="dataset-split",
            config_id="config",
            repetition=0,
        ),
        slurm=wandb_completion_tool.SlurmCaseProvenance(
            phase="baseline",
            case_index=3,
            job_id="12345",
        ),
        producer=wandb_completion_tool.CompletionSealProducer(
            repository="anonymizer-experiments",
            commit="a" * 40,
        ),
    )
    seal_path = tmp_path / wandb_completion_tool.COMPLETION_SEAL_FILENAME

    wandb_completion_tool.write_completion_seal(seal_path, seal)
    first_bytes = seal_path.read_bytes()
    captured_seal = wandb_completion_tool.read_completion_seal(seal_path)
    wandb_completion_tool.verify_completion_seal(snapshot, captured_seal.seal)
    wandb_completion_tool.write_completion_seal(seal_path, seal)

    assert seal_path.read_bytes() == first_bytes
    assert captured_seal.seal.case.config_id == "config"
    assert not list(tmp_path.glob(".*.tmp"))

    real_parent = tmp_path / "real-parent"
    case_dir = real_parent / "case"
    case_dir.mkdir(parents=True)
    redirected_parent = tmp_path / "redirected-parent"
    redirected_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="safely open completion seal directory"):
        wandb_completion_tool.write_completion_seal(
            redirected_parent / "case" / wandb_completion_tool.COMPLETION_SEAL_FILENAME,
            seal,
        )

    changed_record = json.loads(measurement_path.read_text(encoding="utf-8"))
    changed_record["elapsed_sec"] = 0.75
    measurement_path.write_text(json.dumps(changed_record) + "\n", encoding="utf-8")
    changed_snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    with pytest.raises(ValueError, match="does not match"):
        wandb_completion_tool.verify_completion_seal(changed_snapshot, captured_seal.seal)


def test_completion_seal_rejects_failed_or_missing_terminal_stage(
    tmp_path: Path,
    wandb_completion_tool: ModuleType,
) -> None:
    measurement_path = tmp_path / "measurements.jsonl"
    measurement_path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)

    with pytest.raises(ValueError, match="terminal stage record"):
        wandb_completion_tool.build_completion_seal(
            snapshot,
            case=wandb_completion_tool.ImportedCaseIdentity(
                case_id="case",
                workload_id="workload",
                config_id="config",
                repetition=0,
            ),
            slurm=wandb_completion_tool.SlurmCaseProvenance(phase="phase", case_index=0),
            producer=wandb_completion_tool.CompletionSealProducer(
                repository="anonymizer-experiments",
                commit="b" * 40,
            ),
        )


def test_completion_seal_rejects_case_identity_mismatch_and_hidden_workflow_error(
    tmp_path: Path,
    wandb_completion_tool: ModuleType,
) -> None:
    case = wandb_completion_tool.ImportedCaseIdentity(
        case_id="case",
        workload_id="workload",
        config_id="config",
        repetition=0,
    )
    tags = {
        "case_id": "case",
        "workload_id": "workload",
        "config_id": "config",
        "repetition": 0,
    }
    records = [
        {
            "schema_version": 1,
            "record_type": "stage",
            "run_id": "case",
            "run_tags": tags,
            "timestamp_unix_sec": 1.0,
            "stage": "EntityDetectionWorkflow.run",
            "status": "error",
            "elapsed_sec": 0.25,
        },
        {
            "schema_version": 1,
            "record_type": "stage",
            "run_id": "case",
            "run_tags": tags,
            "timestamp_unix_sec": 2.0,
            "stage": "Anonymizer._run_internal",
            "status": "completed",
            "elapsed_sec": 0.5,
        },
    ]
    measurement_path = tmp_path / "measurements.jsonl"
    measurement_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    seal_kwargs = {
        "slurm": wandb_completion_tool.SlurmCaseProvenance(phase="phase", case_index=0),
        "producer": wandb_completion_tool.CompletionSealProducer(
            repository="anonymizer-experiments",
            commit="b" * 40,
        ),
    }

    with pytest.raises(ValueError, match="status does not match"):
        wandb_completion_tool.build_completion_seal(snapshot, case=case, **seal_kwargs)

    records.pop(0)
    records[0]["run_tags"] = {**tags, "config_id": "other"}
    measurement_path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    mismatched_snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    with pytest.raises(ValueError, match="case identity"):
        wandb_completion_tool.build_completion_seal(mismatched_snapshot, case=case, **seal_kwargs)

    failed_terminal = {
        "schema_version": 1,
        "record_type": "stage",
        "run_id": "case",
        "run_tags": {
            "case_id": "case",
            "workload_id": "workload",
            "config_id": "config",
            "repetition": 0,
        },
        "timestamp_unix_sec": 1.0,
        "stage": "Anonymizer._run_internal",
        "status": "error",
        "elapsed_sec": 0.5,
    }
    measurement_path.write_text(json.dumps(failed_terminal) + "\n", encoding="utf-8")
    failed_snapshot = wandb_completion_tool.read_measurement_snapshot(measurement_path)
    with pytest.raises(ValueError, match="terminal status does not match"):
        wandb_completion_tool.build_completion_seal(
            failed_snapshot,
            case=wandb_completion_tool.ImportedCaseIdentity(
                case_id="case",
                workload_id="workload",
                config_id="config",
                repetition=0,
            ),
            slurm=wandb_completion_tool.SlurmCaseProvenance(phase="phase", case_index=0),
            producer=wandb_completion_tool.CompletionSealProducer(
                repository="anonymizer-experiments",
                commit="b" * 40,
            ),
        )


def _write_sealed_import_case(tool: ModuleType, root: Path) -> tuple[Path, Path]:
    completion = sys.modules[tool.read_completion_seal.__module__]
    measurement_path = root / "measurements.jsonl"
    measurement_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "stage",
                "run_id": "dataset__config__r000",
                "run_tags": {
                    "case_id": "dataset__config__r000",
                    "workload_id": "dataset-split",
                    "config_id": "config-a",
                    "repetition": 0,
                },
                "timestamp_unix_sec": 1.0,
                "stage": "Anonymizer._run_internal",
                "status": "completed",
                "elapsed_sec": 2.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = completion.read_measurement_snapshot(measurement_path)
    seal = completion.build_completion_seal(
        snapshot,
        case=completion.ImportedCaseIdentity(
            case_id="dataset__config__r000",
            workload_id="dataset-split",
            config_id="config-a",
            repetition=0,
        ),
        slurm=completion.SlurmCaseProvenance(
            phase="baseline",
            case_index=7,
            job_id="12345",
        ),
        producer=completion.CompletionSealProducer(
            repository="anonymizer-experiments",
            commit="c" * 40,
        ),
    )
    seal_path = root / completion.COMPLETION_SEAL_FILENAME
    completion.write_completion_seal(seal_path, seal)
    return measurement_path, seal_path


def test_strict_import_builds_stable_typed_payload_without_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wandb_import_tool: ModuleType,
) -> None:
    measurement_path, seal_path = _write_sealed_import_case(wandb_import_tool, tmp_path)
    settings = wandb_import_tool.ResolvedWandbConfig(
        wandb_mode=wandb_import_tool.WandbMode.offline,
        wandb_project="project",
        wandb_entity="entity",
    )
    calls: list[Any] = []

    def capture_publish(
        _publisher: Any,
        _settings: Any,
        *,
        payload: Any,
        measurement_sha256: str,
        record_count: int,
    ) -> Any:
        calls.append((payload, measurement_sha256, record_count))
        return wandb_import_tool.WandbPublishResult(
            published=True,
            run_id=payload.init.run_id,
            measurement_sha256=measurement_sha256,
            record_count=record_count,
        )

    monkeypatch.setattr(wandb_import_tool.WandbPublisher, "publish_payload", capture_publish)

    first = wandb_import_tool.import_sealed_run(measurement_path, seal_path=seal_path, settings=settings)
    second = wandb_import_tool.import_sealed_run(measurement_path, seal_path=seal_path, settings=settings)

    payload = calls[0][0]
    serialized = payload.model_dump_json()
    assert first.run_id == second.run_id
    assert len(first.run_id) == 32
    assert payload.init.resume == "allow"
    assert payload.config.run_kind == "imported_case"
    assert payload.config.imported_config_id == "config-a"
    assert payload.config.imported is not None
    assert payload.config.imported.completion_seal_sha256
    assert str(measurement_path) not in serialized
    assert calls[0][2] == 1


def test_strict_import_retry_is_a_remote_publication_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wandb_import_tool: ModuleType,
) -> None:
    measurement_path, seal_path = _write_sealed_import_case(wandb_import_tool, tmp_path)
    settings = wandb_import_tool.ResolvedWandbConfig(
        wandb_mode=wandb_import_tool.WandbMode.offline,
        wandb_project="project",
    )
    prepared = wandb_import_tool.prepare_sealed_import(
        measurement_path,
        seal_path=seal_path,
        settings=settings,
    )
    setup = sys.modules[wandb_import_tool.WandbPublisher.__module__]
    state = _wandb_state()
    monkeypatch.setattr(setup, "require_wandb", lambda: _fake_wandb_module(state))
    publisher = wandb_import_tool.WandbPublisher()

    first = publisher.publish_payload(
        settings,
        payload=prepared.payload,
        measurement_sha256=prepared.measurement_sha256,
        record_count=prepared.record_count,
    )
    second = publisher.publish_payload(
        settings,
        payload=prepared.payload,
        measurement_sha256=prepared.measurement_sha256,
        record_count=prepared.record_count,
    )

    assert first.run_id == second.run_id
    assert len(state.config_updates) == 1
    assert len(state.logged) == 1
    assert state.log_kwargs == [{"step": 0}]
    assert len(state.summary_updates) == 1
    assert state.finished == ["finish", "finish"]

    state.remote_summary["publication/completion_seal_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="different sealed content"):
        publisher.publish_payload(
            settings,
            payload=prepared.payload,
            measurement_sha256=prepared.measurement_sha256,
            record_count=prepared.record_count,
        )


def test_strict_import_identity_is_destination_scoped_and_rejects_mismatch(
    tmp_path: Path,
    wandb_import_tool: ModuleType,
) -> None:
    measurement_path, seal_path = _write_sealed_import_case(wandb_import_tool, tmp_path)
    seal_snapshot = wandb_import_tool.read_completion_seal(seal_path)
    first = wandb_import_tool.ResolvedWandbConfig(
        wandb_mode=wandb_import_tool.WandbMode.offline,
        wandb_project="project-a",
    )
    second = first.validated_update(wandb_project="project-b")
    original_run_id = wandb_import_tool.stable_import_run_id(first, seal_snapshot=seal_snapshot)

    assert original_run_id != wandb_import_tool.stable_import_run_id(second, seal_snapshot=seal_snapshot)

    changed = json.loads(measurement_path.read_text(encoding="utf-8"))
    changed["elapsed_sec"] = 3.0
    measurement_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        wandb_import_tool.import_sealed_run(measurement_path, seal_path=seal_path, settings=first)

    completion = sys.modules[wandb_import_tool.read_completion_seal.__module__]
    changed_snapshot = completion.read_measurement_snapshot(measurement_path)
    changed_seal = completion.build_completion_seal(
        changed_snapshot,
        case=seal_snapshot.seal.case,
        slurm=seal_snapshot.seal.slurm,
        producer=seal_snapshot.seal.producer,
    )
    completion.write_completion_seal(seal_path, changed_seal)
    changed_seal_snapshot = completion.read_completion_seal(seal_path)

    assert original_run_id != wandb_import_tool.stable_import_run_id(first, seal_snapshot=changed_seal_snapshot)


def test_strict_import_does_not_suppress_publisher_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    wandb_import_tool: ModuleType,
) -> None:
    measurement_path, seal_path = _write_sealed_import_case(wandb_import_tool, tmp_path)
    settings = wandb_import_tool.ResolvedWandbConfig(wandb_mode=wandb_import_tool.WandbMode.offline)
    monkeypatch.setattr(
        wandb_import_tool.WandbPublisher,
        "publish_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("finish failed")),
    )

    with pytest.raises(RuntimeError, match="finish failed"):
        wandb_import_tool.import_sealed_run(measurement_path, seal_path=seal_path, settings=settings)


def test_strict_import_rejects_missing_completion_seal(
    tmp_path: Path,
    wandb_import_tool: ModuleType,
) -> None:
    measurement_path = tmp_path / "measurements.jsonl"
    measurement_path.write_text(json.dumps(_strict_record_payload()) + "\n", encoding="utf-8")
    settings = wandb_import_tool.ResolvedWandbConfig(wandb_mode=wandb_import_tool.WandbMode.offline)

    with pytest.raises(wandb_import_tool.ImportInputError, match="safely open"):
        wandb_import_tool.import_sealed_run(
            measurement_path,
            seal_path=tmp_path / "missing-completion-seal.json",
            settings=settings,
        )


def test_strict_import_command_default_does_not_override_namespaced_mode(
    monkeypatch: pytest.MonkeyPatch,
    wandb_import_tool: ModuleType,
) -> None:
    monkeypatch.setenv("ANONYMIZER_MEASUREMENT_WANDB_MODE", "offline")

    settings = wandb_import_tool.ResolvedWandbConfig.from_env_and_overrides(
        defaults={"wandb_mode": wandb_import_tool.WandbMode.online},
        wandb_mode=None,
    )

    assert settings.wandb_mode == wandb_import_tool.WandbMode.offline


def test_strict_import_cli_redacts_sdk_value_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    wandb_import_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"
    prepared = wandb_import_tool.PreparedWandbImport(
        payload=SimpleNamespace(),
        measurement_sha256="a" * 64,
        record_count=1,
    )
    monkeypatch.setattr(wandb_import_tool, "prepare_sealed_import", lambda *_args, **_kwargs: prepared)
    monkeypatch.setattr(
        wandb_import_tool.WandbPublisher,
        "publish_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError(canary)),
    )

    with caplog.at_level(logging.ERROR, logger="measurement.wandb_import"), pytest.raises(SystemExit) as raised:
        wandb_import_tool.main(tmp_path / "measurements.jsonl", wandb_mode=wandb_import_tool.WandbMode.offline)

    assert raised.value.code == 1
    assert canary not in caplog.text
    assert "ValueError" in caplog.text


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
    serialized = metadata.model_dump_json()

    _assert_wandb_metadata(metadata.model_dump(mode="json", exclude_none=True))
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
    publish_calls: list[Any] = []
    monkeypatch.setattr(
        run_benchmarks_tool,
        "publish_benchmark_wandb_best_effort",
        lambda *args, **kwargs: publish_calls.append((args, kwargs)),
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

    assert publish_calls == []


def test_run_or_plan_publishes_only_after_suite_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    spec_path = _write_simple_suite(tmp_path, suite_id="wandb-disabled")
    _write_text_input(tmp_path)

    events: list[str] = []

    def fake_run_suite(
        benchmark_spec: Any,
        *,
        output_dir: Path,
        **_kwargs: Any,
    ) -> Any:
        events.append("run-suite")
        return _benchmark_result(run_benchmarks_tool, suite_id=benchmark_spec.suite_id, output_dir=output_dir)

    monkeypatch.setattr(run_benchmarks_tool, "run_suite", fake_run_suite)
    monkeypatch.setattr(
        run_benchmarks_tool,
        "publish_benchmark_wandb_best_effort",
        lambda *_args, **_kwargs: events.append("publish"),
    )

    result = run_benchmarks_tool.run_or_plan(
        spec_path,
        output=tmp_path / "output",
        overwrite=False,
        dry_run=False,
        export=False,
        fail_fast=False,
        wandb_settings=run_benchmarks_tool.resolve_wandb_settings(wandb_mode=run_benchmarks_tool.WandbMode.offline),
    )

    assert result.suite_id == "wandb-disabled"
    assert events == ["run-suite", "publish"]


def test_wandb_publisher_uses_explicit_handle_and_finishes_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    setup_module = sys.modules[run_benchmarks_tool.publish_benchmark_wandb_best_effort.__module__]
    monkeypatch.setattr(setup_module, "require_wandb", lambda: fake_wandb)

    result = setup_module.WandbPublisher().publish(
        run_benchmarks_tool.ResolvedWandbConfig(wandb_mode=run_benchmarks_tool.WandbMode.offline),
        suite_id="suite",
        output_dir=tmp_path,
        finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
            measurement_path=Path(case.measurement_path), cases=[case]
        ),
    )

    assert result.published is True
    assert len(state.init_kwargs["id"]) == 32
    assert state.init_kwargs["resume"] == "never"
    _assert_logged_wandb_payload(state)


def test_wandb_real_sdk_supports_sequential_offline_publication(tmp_path: Path) -> None:
    if importlib.util.find_spec("wandb") is None:
        pytest.skip("wandb is not installed")
    script = r"""
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path("tools/measurement").resolve()))
from measurement_tools.wandb_models import ResolvedWandbConfig, WandbMode
from measurement_tools.wandb_setup import BenchmarkWandbFinalization, WandbPublisher

root = Path(sys.argv[1])
canary = "wandb-ambient-canary-value"
os.environ["WANDB_NOTES"] = canary
os.environ["WANDB_RUN_ID"] = canary
before = dict(os.environ)
run_ids = []
for index in range(2):
    output_dir = root / f"run-{index}"
    output_dir.mkdir()
    measurement_path = output_dir / "measurements.jsonl"
    case_id = f"case-{index}"
    measurement_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "record_type": "stage",
                "run_id": case_id,
                "run_tags": {},
                "timestamp_unix_sec": 1.0,
                "stage": "Anonymizer._run_internal",
                "status": "completed",
                "elapsed_sec": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    case = SimpleNamespace(
        case_id=case_id,
        status=SimpleNamespace(value="completed"),
        elapsed_sec=0.1,
    )
    result = WandbPublisher().publish(
        ResolvedWandbConfig(wandb_mode=WandbMode.offline),
        suite_id="real-sdk-probe",
        output_dir=output_dir,
        finalization=BenchmarkWandbFinalization(measurement_path=measurement_path, cases=[case]),
    )
    assert result.published
    assert dict(os.environ) == before
    run_ids.append(result.run_id)

leaks = []
for path in root.rglob("*"):
    if path.is_file() and canary.encode() in path.read_bytes():
        leaks.append(str(path))
print(json.dumps({"run_ids": run_ids, "leaks": leaks}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "tools/measurement"), environment.get("PYTHONPATH")])
    )

    completed = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    result = json.loads(completed.stdout.strip().splitlines()[-1])

    assert len(set(result["run_ids"])) == 2
    assert all(len(run_id) == 32 for run_id in result["run_ids"])
    assert result["leaks"] == []


def test_wandb_table_opt_in_changes_only_table_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    setup_module = sys.modules[run_benchmarks_tool.publish_benchmark_wandb_best_effort.__module__]
    states: list[SimpleNamespace] = []

    for log_tables in (False, True):
        state = _wandb_state()
        states.append(state)
        monkeypatch.setattr(setup_module, "require_wandb", lambda state=state: _fake_wandb_module(state))
        setup_module.WandbPublisher().publish(
            run_benchmarks_tool.ResolvedWandbConfig(
                wandb_mode=run_benchmarks_tool.WandbMode.offline,
                wandb_log_tables=log_tables,
            ),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
                measurement_path=Path(case.measurement_path), cases=[case]
            ),
        )

    without_tables, with_tables = states
    assert without_tables.summary_updates == with_tables.summary_updates
    assert without_tables.logged[0] == with_tables.summary_updates[0]
    table_keys = {key for key in with_tables.logged[0] if key.startswith("measurement_table/")}
    assert table_keys == {"measurement_table/run", "measurement_table/record", "measurement_table/stage"}
    assert {
        key: value for key, value in with_tables.logged[0].items() if key not in table_keys
    } == without_tables.logged[0]
    assert without_tables.config_updates[0] | {"wandb_log_tables": True} == with_tables.config_updates[0]


@pytest.mark.parametrize("failure_stage", ["init", "config", "log", "summary"])
def test_wandb_publisher_surfaces_each_strict_lifecycle_failure(
    failure_stage: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    original_init = fake_wandb.init

    def failing_init(**kwargs: Any) -> Any:
        if failure_stage == "init":
            raise RuntimeError("init failed")
        run = original_init(**kwargs)
        if failure_stage == "config":
            run.config.update = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("config failed"))
        elif failure_stage == "log":
            run.log = lambda _payload: (_ for _ in ()).throw(RuntimeError("log failed"))
        elif failure_stage == "summary":
            run.summary.update = lambda _payload: (_ for _ in ()).throw(RuntimeError("summary failed"))
        return run

    fake_wandb.init = failing_init
    setup_module = sys.modules[run_benchmarks_tool.publish_benchmark_wandb_best_effort.__module__]
    monkeypatch.setattr(setup_module, "require_wandb", lambda: fake_wandb)

    with pytest.raises(RuntimeError, match=f"{failure_stage} failed"):
        setup_module.WandbPublisher().publish(
            run_benchmarks_tool.ResolvedWandbConfig(wandb_mode=run_benchmarks_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
                measurement_path=Path(case.measurement_path), cases=[case]
            ),
        )

    assert state.finished == ([] if failure_stage == "init" else ["finish"])


def test_wandb_publisher_preserves_publish_and_finish_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    original_init = fake_wandb.init

    def init_with_two_failures(**kwargs: Any) -> Any:
        run = original_init(**kwargs)
        run.log = lambda _payload: (_ for _ in ()).throw(RuntimeError("publish failed"))
        run.finish = lambda: (_ for _ in ()).throw(RuntimeError("finish failed"))
        return run

    fake_wandb.init = init_with_two_failures
    setup_module = sys.modules[run_benchmarks_tool.publish_benchmark_wandb_best_effort.__module__]
    monkeypatch.setattr(setup_module, "require_wandb", lambda: fake_wandb)

    with pytest.raises(ExceptionGroup) as raised:
        setup_module.WandbPublisher().publish(
            run_benchmarks_tool.ResolvedWandbConfig(wandb_mode=run_benchmarks_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
                measurement_path=Path(case.measurement_path), cases=[case]
            ),
        )

    assert [str(error) for error in raised.value.exceptions] == ["publish failed", "finish failed"]


def test_wandb_publisher_surfaces_finish_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    original_init = fake_wandb.init

    def init_with_bad_finish(**kwargs: Any) -> Any:
        run = original_init(**kwargs)
        run.finish = lambda: (_ for _ in ()).throw(RuntimeError("finish failed"))
        return run

    fake_wandb.init = init_with_bad_finish
    setup_module = sys.modules[run_benchmarks_tool.publish_benchmark_wandb_best_effort.__module__]
    monkeypatch.setattr(setup_module, "require_wandb", lambda: fake_wandb)

    with pytest.raises(RuntimeError, match="finish failed"):
        setup_module.WandbPublisher().publish(
            run_benchmarks_tool.ResolvedWandbConfig(wandb_mode=run_benchmarks_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
                measurement_path=Path(case.measurement_path), cases=[case]
            ),
        )


def test_wandb_best_effort_contains_metadata_failures_and_skips_disabled(
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
    wandb_setup_tool: ModuleType,
) -> None:
    canary = "alice@example.com/private/input"

    def fail_metadata() -> Any:
        raise ValueError(canary)

    disabled = wandb_setup_tool.publish_benchmark_wandb_best_effort(
        wandb_setup_tool.ResolvedWandbConfig(),
        suite_id="suite",
        output_dir=tmp_path,
        finalization=wandb_setup_tool.BenchmarkWandbFinalization(
            measurement_path=tmp_path / "missing.jsonl",
            cases=[],
        ),
        metadata_factory=lambda: pytest.fail("disabled W&B must not build metadata"),
    )
    with caplog.at_level(logging.WARNING, logger="measurement.wandb"):
        failed = wandb_setup_tool.publish_benchmark_wandb_best_effort(
            wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=wandb_setup_tool.BenchmarkWandbFinalization(
                measurement_path=tmp_path / "missing.jsonl",
                cases=[],
            ),
            metadata_factory=fail_metadata,
        )

    assert disabled.published is False
    assert failed.published is False
    assert canary not in caplog.text


def test_wandb_publisher_rejects_changed_run_identity_and_finishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    original_init = fake_wandb.init

    def init_with_changed_id(**kwargs: Any) -> Any:
        run = original_init(**kwargs)
        run.id = "different-id"
        return run

    fake_wandb.init = init_with_changed_id
    setup_module = sys.modules[run_benchmarks_tool.publish_benchmark_wandb_best_effort.__module__]
    monkeypatch.setattr(setup_module, "require_wandb", lambda: fake_wandb)

    with pytest.raises(RuntimeError, match="different run identity"):
        setup_module.WandbPublisher().publish(
            run_benchmarks_tool.ResolvedWandbConfig(wandb_mode=run_benchmarks_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
                measurement_path=Path(case.measurement_path), cases=[case]
            ),
        )

    assert state.finished == ["finish"]


def test_wandb_publisher_rejects_ambient_active_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state, active_run=True)
    setup_module = sys.modules[run_benchmarks_tool.publish_benchmark_wandb_best_effort.__module__]
    monkeypatch.setattr(setup_module, "require_wandb", lambda: fake_wandb)

    with pytest.raises(RuntimeError, match="ambient W&B run"):
        setup_module.WandbPublisher().publish(
            run_benchmarks_tool.ResolvedWandbConfig(wandb_mode=run_benchmarks_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
                measurement_path=Path(case.measurement_path), cases=[case]
            ),
        )

    assert state.init_kwargs == {}
    assert state.finished == []


def test_wandb_publisher_finishes_after_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    run_benchmarks_tool: ModuleType,
) -> None:
    case = _write_case_measurements(run_benchmarks_tool, tmp_path, _minimal_benchmark_case(run_benchmarks_tool))
    state = _wandb_state()
    fake_wandb = _fake_wandb_module(state)
    original_init = fake_wandb.init

    def init_with_interrupt(**kwargs: Any) -> Any:
        run = original_init(**kwargs)
        run.log = lambda _payload: (_ for _ in ()).throw(KeyboardInterrupt())
        return run

    fake_wandb.init = init_with_interrupt
    setup_module = sys.modules[run_benchmarks_tool.publish_benchmark_wandb_best_effort.__module__]
    monkeypatch.setattr(setup_module, "require_wandb", lambda: fake_wandb)

    with pytest.raises(KeyboardInterrupt):
        setup_module.WandbPublisher().publish(
            run_benchmarks_tool.ResolvedWandbConfig(wandb_mode=run_benchmarks_tool.WandbMode.offline),
            suite_id="suite",
            output_dir=tmp_path,
            finalization=run_benchmarks_tool.BenchmarkWandbFinalization(
                measurement_path=Path(case.measurement_path), cases=[case]
            ),
        )

    assert state.finished == ["finish"]


def test_wandb_publisher_rejects_preloaded_sdk(monkeypatch: pytest.MonkeyPatch, wandb_setup_tool: ModuleType) -> None:
    monkeypatch.setattr(wandb_setup_tool, "_PUBLISHER_WANDB_MODULE", None)
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace())

    with (
        wandb_setup_tool.WandbSdkEnvironment(
            wandb_setup_tool.ResolvedWandbConfig(wandb_mode=wandb_setup_tool.WandbMode.offline)
        ),
        pytest.raises(RuntimeError, match="must not be imported"),
    ):
        wandb_setup_tool.require_wandb()


def test_wandb_parser_rejects_deep_input_before_json_load(
    monkeypatch: pytest.MonkeyPatch,
    wandb_ingress_tool: ModuleType,
) -> None:
    payload = b"[" * 5_000 + b"0" + b"]" * 5_000
    monkeypatch.setattr(wandb_ingress_tool.json, "loads", lambda *_args, **_kwargs: pytest.fail("parser called"))

    with pytest.raises(ValueError, match="nesting limit"):
        wandb_ingress_tool._parse_records(payload, max_records=1, max_line_bytes=len(payload), max_nesting=16)


def test_wandb_case_elapsed_metrics_reject_negative_and_nonfinite(wandb_logging_tool: ModuleType) -> None:
    status = SimpleNamespace(value="completed")
    for elapsed_sec in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            wandb_logging_tool.summarize_benchmark_cases([SimpleNamespace(status=status, elapsed_sec=elapsed_sec)])


def test_wandb_ingress_accepts_collector_model_workflow(tmp_path: Path, wandb_ingress_tool: ModuleType) -> None:
    from anonymizer.measurement import (
        MeasurementCollector,
        MeasurementConfig,
        measurement_session,
        record_model_workflow,
    )

    path = tmp_path / "measurements.jsonl"
    canary = "alice@example.com/private/input"
    collector = MeasurementCollector(run_id="collector-run", record_hash_key="test-key")
    with measurement_session(collector):
        record_model_workflow(
            workflow_name="entity-detection-native-rules-router",
            model_aliases=["native-direct"],
            input_row_count=1,
            output_row_count=1,
            failed_record_count=0,
            elapsed_sec=0.25,
            extra_fields={"operator_note": canary},
        )
    MeasurementConfig(output_path=path).write_collector(collector)

    snapshot = wandb_ingress_tool.read_measurement_snapshot(path)

    assert len(snapshot.records) == 1
    assert snapshot.records[0].record_type == "model_workflow"
    source_record = json.loads(path.read_text(encoding="utf-8"))
    assert source_record["local_fields"]["operator_note"] == canary
    assert canary not in snapshot.records[0].model_dump_json()

    source_record["operator_note"] = source_record["local_fields"].pop("operator_note")
    path.write_text(json.dumps(source_record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="operator_note"):
        wandb_ingress_tool.read_measurement_snapshot(path)
