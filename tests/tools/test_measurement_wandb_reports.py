# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from tests.tools.measurement_test_support import (
    _load_measurement_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


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


def test_wandb_report_paths_rebind_validated_base_url(wandb_report_tool: ModuleType) -> None:
    project = wandb_report_tool.WandbProjectPath(entity="entity", project="project").with_base_url(
        "https://reports.example"
    )
    run = wandb_report_tool.WandbRunPath(entity="entity", project="project", run_id="run").with_base_url(
        "https://reports.example"
    )

    assert project.model_dump() == {
        "entity": "entity",
        "project": "project",
        "base_url": "https://reports.example",
    }
    assert list(project.model_json_schema()["properties"]) == ["entity", "project", "base_url"]
    assert list(run.model_json_schema()["properties"]) == ["entity", "project", "run_id", "base_url"]
    assert run.url == "https://reports.example/entity/project/runs/run"
    with pytest.raises(ValidationError, match="credential-free"):
        project.with_base_url("https://user:secret@reports.example")
    with pytest.raises(ValidationError, match="HTTPS unless it targets loopback"):
        run.with_base_url("http://reports.example")


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
            "slurm": {
                "job_id": "123",
                "array_task_id": "4",
                "jobs": [{"role": "detect", "job_id": "234"}, {"role": "replace", "job_id": "345"}],
            },
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
    assert [(job.role, job.job_id) for job in view.metadata.execution.slurm.jobs] == [
        ("detect", "234"),
        ("replace", "345"),
    ]


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
