# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from tests.tools.measurement_test_support import (
    _minimal_benchmark_case,
    _simple_suite_payload,
    _strict_record_payload,
    _write_text_input,
    _write_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_simple_suite(tmp_path: Path, *, suite_id: str = "base-suite") -> Path:
    return _write_yaml(tmp_path / "suite.yaml", _simple_suite_payload(suite_id=suite_id))


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


def _assert_logged_wandb_payload(state: SimpleNamespace) -> None:
    assert state.finished == ["finish"]
    assert state.logged
    payload = state.logged[0]
    assert payload["benchmark/case_completed"] == 1
    assert payload["measurement/record/final_entity_count"] == 1.0
    assert payload["measurement/record/utility_score_mean"] == pytest.approx(0.95)
    assert state.summary_updates == [payload]
    assert "Alice" not in json.dumps(payload)


def _write_sealed_import_case(
    tool: ModuleType,
    root: Path,
    *,
    jobs: tuple[tuple[str, str], ...] = (),
) -> tuple[Path, Path]:
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
            jobs=tuple(completion.SlurmJobProvenance(role=role, job_id=job_id) for role, job_id in jobs),
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


def test_strict_import_exposes_multi_job_slurm_metadata(
    tmp_path: Path,
    wandb_import_tool: ModuleType,
) -> None:
    measurement_path, seal_path = _write_sealed_import_case(
        wandb_import_tool,
        tmp_path,
        jobs=(("detect", "123"), ("replace", "456")),
    )
    prepared = wandb_import_tool.prepare_sealed_import(
        measurement_path,
        seal_path=seal_path,
        settings=wandb_import_tool.ResolvedWandbConfig(wandb_mode=wandb_import_tool.WandbMode.offline),
    )

    assert prepared.payload.config.execution is not None
    assert prepared.payload.config.execution.slurm is not None
    assert [(job.role, job.job_id) for job in prepared.payload.config.execution.slurm.jobs] == [
        ("detect", "123"),
        ("replace", "456"),
    ]


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
