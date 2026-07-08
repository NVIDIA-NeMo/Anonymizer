# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import yaml

from tests.tools.measurement_test_support import (
    _simple_suite_payload,
    _write_text_input,
    _write_yaml,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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


@pytest.mark.parametrize("sweep_id", ["token-ablation", "latency-" + "x" * 80])
def test_sweep_disabled_wandb_accepts_generated_identifiers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
    sweep_id: str,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite, sweep_id=sweep_id)
    calls: list[tuple[Path, Path, Any]] = []
    _patch_sweep_runner(monkeypatch, sweep_tool, calls, status=sweep_tool.run_benchmarks.CaseStatus.completed)

    result = sweep_tool.run_sweep(
        sweep_path,
        output_root=tmp_path / "runs",
        overwrite=False,
        dry_run=False,
        export=False,
        fail_fast=False,
        wandb_settings=sweep_tool.run_benchmarks.resolve_wandb_settings(),
        create_report=False,
    )

    assert result.completed_arms == 2
    assert len(calls) == 2
    assert all(not settings.enabled for _, _, settings in calls)
    assert all(settings.effective_wandb_tags == [] for _, _, settings in calls)


def test_sweep_enabled_wandb_bounds_generated_tags(tmp_path: Path, sweep_tool: ModuleType) -> None:
    sweep_id = "latency-" + "x" * 80
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite, sweep_id=sweep_id)
    spec = sweep_tool.load_sweep_spec(sweep_path)
    arm = sweep_tool.expand_sweep_arms(spec)[0]

    settings = sweep_tool._arm_wandb_settings(
        sweep_tool.run_benchmarks.resolve_wandb_settings(
            wandb_mode=sweep_tool.run_benchmarks.WandbMode.offline,
        ),
        spec=spec,
        arm=arm,
        run_name=f"{sweep_id}-{arm.arm_id}",
    )

    generated = [tag for tag in settings.effective_wandb_tags if tag.startswith(("sweep:", "arm:"))]
    assert {tag.split(":", 1)[0] for tag in generated} == {"sweep", "arm"}
    assert all(len(tag) <= 64 for tag in generated)


def test_sweep_dry_run_is_repeatable_without_writing(tmp_path: Path, sweep_tool: ModuleType) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite)
    _write_text_input(tmp_path)
    output_root = tmp_path / "runs"
    settings = sweep_tool.run_benchmarks.resolve_wandb_settings()

    first = sweep_tool.run_sweep(
        sweep_path,
        output_root=output_root,
        overwrite=False,
        dry_run=True,
        export=False,
        fail_fast=False,
        wandb_settings=settings,
        create_report=False,
    )
    second = sweep_tool.run_sweep(
        sweep_path,
        output_root=output_root,
        overwrite=False,
        dry_run=True,
        export=False,
        fail_fast=False,
        wandb_settings=settings,
        create_report=False,
    )

    assert first.errored_arms == second.errored_arms == 0
    assert [arm.total_cases for arm in first.arms] == [1, 1]
    assert [arm.total_cases for arm in second.arms] == [1, 1]
    assert not output_root.exists()


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

    monkeypatch.setattr(sweep_tool, "create_benchmark_workspace", fake_workspace, raising=False)
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


def test_sweep_report_without_entity_preserves_arm_results(
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
        lambda *_args, **_kwargs: pytest.fail("report creation called without an entity"),
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
            wandb_project="project",
        ),
        create_report=True,
    )

    assert result.completed_arms == 2
    assert result.report_url is None
    assert result.report_error is None


def test_sweep_workspace_without_entity_preserves_arm_results(
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
        "create_benchmark_workspace",
        lambda *_args, **_kwargs: pytest.fail("workspace creation called without an entity"),
        raising=False,
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
            wandb_project="project",
        ),
        create_report=False,
        create_workspace=True,
    )

    assert result.completed_arms == 2
    assert result.workspace_url is None
    assert result.workspace_error is None


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


def test_sweep_keeps_report_and_workspace_failures_separate(
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
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("report secret")),
    )
    monkeypatch.setattr(
        sweep_tool,
        "create_benchmark_workspace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(LookupError("workspace secret")),
        raising=False,
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
        create_workspace=True,
    )

    assert result.report_error == "W&B report creation failed (RuntimeError)"
    assert result.workspace_error == "W&B workspace creation failed (LookupError)"
    assert "secret" not in result.model_dump_json()


@pytest.mark.parametrize("view_kind", ["report", "workspace"])
def test_sweep_preserves_arm_results_when_project_validation_fails_redacted(
    view_kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sweep_tool: ModuleType,
) -> None:
    base_suite = _write_yaml(tmp_path / "base.yaml", _simple_suite_payload())
    sweep_path = _write_threshold_sweep(tmp_path, base_suite=base_suite)
    calls: list[tuple[Path, Path, Any]] = []
    _patch_sweep_runner(monkeypatch, sweep_tool, calls, status=sweep_tool.run_benchmarks.CaseStatus.completed)
    secret_entity = "private/entity"

    result = sweep_tool.run_sweep(
        sweep_path,
        output_root=tmp_path / "runs",
        overwrite=True,
        dry_run=False,
        export=True,
        fail_fast=False,
        wandb_settings=sweep_tool.run_benchmarks.resolve_wandb_settings(
            wandb_mode=sweep_tool.run_benchmarks.WandbMode.offline,
            wandb_entity=secret_entity,
            wandb_project="project",
        ),
        create_report=view_kind == "report",
        create_workspace=view_kind == "workspace",
    )

    error = result.report_error if view_kind == "report" else result.workspace_error
    assert result.completed_arms == 2
    assert error == f"W&B {view_kind} creation failed (ValidationError)"
    assert secret_entity not in result.model_dump_json()


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
