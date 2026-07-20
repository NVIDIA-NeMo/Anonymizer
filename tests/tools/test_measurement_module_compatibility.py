# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import os
import pickle
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
MEASUREMENT_ROOT = REPO_ROOT / "tools/measurement"


@pytest.fixture(autouse=True)
def _measurement_import_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(MEASUREMENT_ROOT))


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


@pytest.mark.parametrize(
    ("relative_path", "defined_class", "canonical_import", "canonical_module"),
    [
        (
            "measurement_tools/wandb_models.py",
            "WandbMode",
            "StrictFrozenModel",
            "measurement_tools.wandb_settings",
        ),
        (
            "measurement_tools/wandb_setup.py",
            "WandbPublisher",
            "ResolvedWandbConfig",
            "measurement_tools.wandb_publisher",
        ),
        (
            "create_wandb_report.py",
            "WandbReportResult",
            "WandbProjectPath",
            "measurement_tools.wandb_report_contracts",
        ),
        (
            "run_benchmarks.py",
            "BenchmarkSpec",
            "Anonymizer",
            "measurement_tools.benchmark_models",
        ),
        (
            "sweep_benchmarks.py",
            "SweepSpec",
            "WandbProjectPath",
            "measurement_tools.benchmark_sweep_models",
        ),
        (
            "analyze_benchmark_output.py",
            "BenchmarkOutputAnalysis",
            "AnalysisExportResult",
            "measurement_tools.benchmark_analysis_models",
        ),
    ],
)
def test_exact_tool_paths_support_repeated_dynamic_loading(
    relative_path: str,
    defined_class: str,
    canonical_import: str,
    canonical_module: str | None,
) -> None:
    path = MEASUREMENT_ROOT / relative_path
    module_names = (f"compat_first_{path.stem}", f"compat_second_{path.stem}")
    first = _load_module(path, module_names[0])
    second = _load_module(path, module_names[1])
    try:
        first_class = getattr(first, defined_class)
        second_class = getattr(second, defined_class)
        if canonical_module is None:
            assert first_class is not second_class
            assert first_class.__module__ == module_names[0]
            assert second_class.__module__ == module_names[1]
        else:
            assert first_class is second_class
            assert first_class.__module__ == canonical_module
        assert getattr(first, canonical_import) is getattr(second, canonical_import)
    finally:
        for module_name in module_names:
            sys.modules.pop(module_name, None)


def test_wandb_facades_preserve_canonical_identity_and_reconstruction() -> None:
    models = importlib.import_module("measurement_tools.wandb_models")
    setup = importlib.import_module("measurement_tools.wandb_setup")
    report_models = importlib.import_module("measurement_tools.wandb_report_models")
    report = _load_module(MEASUREMENT_ROOT / "create_wandb_report.py", "compat_report_facade")
    runner = _load_module(MEASUREMENT_ROOT / "run_benchmarks.py", "compat_runner_facade")
    sweep = _load_module(MEASUREMENT_ROOT / "sweep_benchmarks.py", "compat_sweep_facade")
    try:
        assert setup.WandbMode is models.WandbMode
        assert setup.ResolvedWandbConfig is models.ResolvedWandbConfig
        assert report.WandbProjectPath is report_models.WandbProjectPath
        assert report.WandbRunPath is report_models.WandbRunPath
        assert runner.WandbMode is models.WandbMode
        assert runner.ResolvedWandbConfig is models.ResolvedWandbConfig
        assert sweep.WandbProjectPath is report_models.WandbProjectPath
        assert sweep.run_benchmarks.is_remote_input_source is runner.is_remote_input_source

        assert models.WandbMode.__module__ == "measurement_tools.wandb_settings"
        assert models.ResolvedWandbConfig.__module__ == "measurement_tools.wandb_settings"
        settings = models.ResolvedWandbConfig()
        reconstructed = models.ResolvedWandbConfig.model_validate(settings.model_dump())
        assert reconstructed == settings
        assert pickle.loads(pickle.dumps(models.WandbMode.offline)) is models.WandbMode.offline
    finally:
        for module_name in ("compat_report_facade", "compat_runner_facade", "compat_sweep_facade"):
            sys.modules.pop(module_name, None)


def test_wandb_setup_facade_preserves_staging_identity_and_payload_contracts() -> None:
    setup = importlib.import_module("measurement_tools.wandb_setup")
    payload = importlib.import_module("measurement_tools.wandb_payload")
    run_identity = importlib.import_module("measurement_tools.wandb_run_identity")
    staging = importlib.import_module("measurement_tools.wandb_staging")

    assert payload.__all__ == ["BenchmarkWandbFinalization", "build_publish_payload"]
    assert run_identity.__all__ == ["default_run_name", "effective_wandb_tags"]
    assert staging.__all__ == [
        "open_directory_no_follow",
        "prepare_wandb_staging_dir",
        "validate_directory_metadata",
    ]
    assert setup.BenchmarkWandbFinalization is payload.BenchmarkWandbFinalization
    assert setup._default_run_name is run_identity.default_run_name
    assert setup._effective_wandb_tags is run_identity.effective_wandb_tags
    assert setup.BenchmarkWandbFinalization.__module__ == "measurement_tools.wandb_payload"


def test_wandb_setup_facade_preserves_sdk_environment_and_publisher_contracts() -> None:
    setup = importlib.import_module("measurement_tools.wandb_setup")
    publisher = importlib.import_module("measurement_tools.wandb_publisher")
    sdk_environment = importlib.import_module("measurement_tools.wandb_sdk_environment")

    assert publisher.__all__ == [
        "WandbPublisher",
        "define_benchmark_metrics",
        "publication_already_complete",
        "publication_state",
        "raise_lifecycle_failures",
        "sdk_init_kwargs",
        "wandb_run_url",
    ]
    assert sdk_environment.__all__ == ["WandbSdkEnvironment", "publisher_environment", "require_wandb"]
    assert setup.WandbPublisher is publisher.WandbPublisher
    assert setup.WandbSdkEnvironment is sdk_environment.WandbSdkEnvironment
    assert setup._publisher_environment is sdk_environment.publisher_environment
    assert setup.WandbPublisher.__module__ == "measurement_tools.wandb_publisher"


def test_wandb_report_facade_preserves_leaf_contracts() -> None:
    catalog = importlib.import_module("measurement_tools.wandb_report_catalog")
    contracts = importlib.import_module("measurement_tools.wandb_report_contracts")
    sdk = importlib.import_module("measurement_tools.wandb_report_sdk")
    text = importlib.import_module("measurement_tools.wandb_report_text")
    report = _load_module(MEASUREMENT_ROOT / "create_wandb_report.py", "compat_report_leaf_facade")
    try:
        assert report.WandbReportResult is contracts.WandbReportResult
        assert report.WandbWorkspaceResult is contracts.WandbWorkspaceResult
        assert report._all_report_metrics is catalog.all_report_metrics
        assert report._group_visible_columns is catalog.group_visible_columns
        assert report._read_group_views is sdk.read_group_views
        assert report._report_settings is sdk.report_settings
        assert report._plain_text is text.plain_text
        assert report._validate_output_url is text.validate_output_url
    finally:
        sys.modules.pop("compat_report_leaf_facade", None)


def test_wandb_report_facade_preserves_report_construction_contracts() -> None:
    reports = importlib.import_module("measurement_tools.wandb_reports")
    report = _load_module(MEASUREMENT_ROOT / "create_wandb_report.py", "compat_report_construction_facade")
    try:
        assert reports.__all__ == [
            "build_benchmark_group_report",
            "build_benchmark_report",
            "build_group_report_markdown",
            "build_report_markdown",
            "config_line",
            "create_benchmark_group_report",
            "create_benchmark_report",
            "default_report_title",
            "int_metric",
            "metric",
            "number_metric",
            "save_report",
            "workload_line",
        ]
        assert report.build_benchmark_report is reports.build_benchmark_report
        assert report.build_benchmark_group_report is reports.build_benchmark_group_report
        assert report.build_report_markdown is reports.build_report_markdown
        assert report._save_report is reports.save_report
        assert report._default_report_title is reports.default_report_title
        assert report.create_benchmark_report is not reports.create_benchmark_report
        assert report.create_benchmark_group_report is not reports.create_benchmark_group_report
        assert report.create_benchmark_report.__module__ == "compat_report_construction_facade"
    finally:
        sys.modules.pop("compat_report_construction_facade", None)


def test_wandb_report_facade_preserves_workspace_construction_contracts() -> None:
    workspaces = importlib.import_module("measurement_tools.wandb_workspaces")
    report = _load_module(MEASUREMENT_ROOT / "create_wandb_report.py", "compat_workspace_construction_facade")
    try:
        assert workspaces.__all__ == [
            "benchmark_run_filters",
            "benchmark_run_groupby",
            "benchmark_runset_settings",
            "benchmark_workspace_sections",
            "benchmark_workspace_settings",
            "build_benchmark_workspace",
            "comparison_workspace_panels",
            "create_benchmark_workspace",
            "default_workspace_title",
            "save_workspace",
            "table_workspace_panels",
            "workspace_bar_panels",
            "workspace_metric_accessor",
        ]
        assert report.build_benchmark_workspace is workspaces.build_benchmark_workspace
        assert report._benchmark_workspace_settings is workspaces.benchmark_workspace_settings
        assert report._benchmark_runset_settings is workspaces.benchmark_runset_settings
        assert report._benchmark_run_filters is workspaces.benchmark_run_filters
        assert report._benchmark_run_groupby is workspaces.benchmark_run_groupby
        assert report._benchmark_workspace_sections is workspaces.benchmark_workspace_sections
        assert report._workspace_bar_panels is workspaces.workspace_bar_panels
        assert report._comparison_workspace_panels is workspaces.comparison_workspace_panels
        assert report._table_workspace_panels is workspaces.table_workspace_panels
        assert report._workspace_metric_accessor is workspaces.workspace_metric_accessor
        assert report._save_workspace is workspaces.save_workspace
        assert report._default_workspace_title is workspaces.default_workspace_title
        assert report.create_benchmark_workspace is not workspaces.create_benchmark_workspace
        assert report.create_benchmark_workspace.__module__ == "compat_workspace_construction_facade"
    finally:
        sys.modules.pop("compat_workspace_construction_facade", None)


def test_benchmark_facade_preserves_models_specs_inputs_and_planning_contracts() -> None:
    inputs = importlib.import_module("measurement_tools.benchmark_inputs")
    models = importlib.import_module("measurement_tools.benchmark_models")
    planning = importlib.import_module("measurement_tools.benchmark_planning")
    specs = importlib.import_module("measurement_tools.benchmark_specs")
    runner = _load_module(MEASUREMENT_ROOT / "run_benchmarks.py", "compat_benchmark_b1_facade")
    try:
        assert models.__all__ == [
            "BenchmarkCase",
            "BenchmarkResult",
            "BenchmarkSpec",
            "CaseRunPaths",
            "CaseStatus",
            "ConfigSpec",
            "DDTraceMode",
            "MatrixEntry",
            "ReplaceKind",
            "ReplaceSpec",
            "RESERVED_RUN_TAG_KEYS",
            "RewriteSpec",
            "WorkloadSpec",
            "duplicate_matrix_entries",
            "duplicates",
        ]
        assert specs.__all__ == [
            "active_config_ids",
            "build_cases",
            "cross_product_matrix",
            "input_columns",
            "load_spec",
            "preflight_config_errors",
            "preflight_model_configs",
            "preflight_model_providers",
            "preflight_model_providers_with_errors",
            "preflight_suite",
            "preflight_workload",
            "preflight_workload_errors",
            "prepare_output_dir",
        ]
        assert inputs.__all__ == [
            "build_anonymizer_config",
            "build_input",
            "build_replace",
            "build_rewrite",
            "is_local_input_source",
            "materialize_sliced_source",
            "present",
            "privacy_goal",
            "read_local_input_dataframe",
            "resolve_config_source",
            "resolve_input_source",
            "resolve_optional_path",
            "resolve_path",
            "safe_case_filename",
            "slice_bounds",
            "workload_has_row_slice",
            "write_local_input_dataframe",
        ]
        assert planning.__all__ == ["dry_run_result", "plan_suite"]
        assert runner.BenchmarkSpec is models.BenchmarkSpec
        assert runner._CaseRunPaths is models.CaseRunPaths
        assert runner.load_spec is specs.load_spec
        assert runner.build_cases is specs.build_cases
        assert runner.build_input is inputs.build_input
        assert runner.build_anonymizer_config is inputs.build_anonymizer_config
        assert runner._resolve_input_source is inputs.resolve_input_source
        assert runner.dry_run_result is not planning.dry_run_result
        assert runner.plan_suite is not planning.plan_suite
        assert runner.plan_suite.__module__ == "compat_benchmark_b1_facade"
    finally:
        sys.modules.pop("compat_benchmark_b1_facade", None)


def test_benchmark_facade_preserves_artifact_and_metadata_contracts() -> None:
    artifacts = importlib.import_module("measurement_tools.benchmark_artifacts")
    metadata = importlib.import_module("measurement_tools.benchmark_wandb_metadata")
    runner = _load_module(MEASUREMENT_ROOT / "run_benchmarks.py", "compat_benchmark_b2_facade")
    try:
        assert artifacts.__all__ == [
            "changed_detection_artifact_files",
            "combine_detection_artifact_analysis",
            "combine_measurements",
            "export_case_detection_artifact_analysis",
            "export_detection_artifact_analysis",
            "export_measurement_tables",
            "jsonl_chunk",
            "render_result",
            "snapshot_detection_artifacts",
            "with_case_metadata",
            "write_detection_artifact_payloads",
            "write_summary",
        ]
        assert metadata.__all__ == [
            "BENCHMARK_SUITE_SCHEMA_VERSION",
            "WANDB_METADATA_SCHEMA_VERSION",
            "benchmark_metadata",
            "build_wandb_metadata",
            "config_metadata",
            "detect_metadata",
            "execution_metadata",
            "file_hash",
            "git_metadata",
            "git_output",
            "model_source_metadata",
            "package_version",
            "package_versions",
            "replace_metadata",
            "rewrite_metadata",
            "run_tags",
            "runtime_metadata",
            "source_metadata",
            "source_suffix",
            "stable_hash",
            "sweep_metadata",
            "workload_metadata",
        ]
        assert runner.combine_measurements is artifacts.combine_measurements
        assert runner.export_measurement_tables is artifacts.export_measurement_tables
        assert runner._with_case_metadata is artifacts.with_case_metadata
        assert runner._git_metadata is metadata.git_metadata
        assert runner._package_versions is metadata.package_versions
        assert runner._execution_metadata is metadata.execution_metadata
        assert runner.build_wandb_metadata is not metadata.build_wandb_metadata
        assert runner.build_wandb_metadata.__module__ == "compat_benchmark_b2_facade"
    finally:
        sys.modules.pop("compat_benchmark_b2_facade", None)


def test_benchmark_facade_preserves_execution_contracts() -> None:
    execution = importlib.import_module("measurement_tools.benchmark_execution")
    runner = _load_module(MEASUREMENT_ROOT / "run_benchmarks.py", "compat_benchmark_b3_facade")
    try:
        assert execution.__all__ == [
            "benchmark_result",
            "build_contexts",
            "case_detection_artifact_path",
            "case_run_paths",
            "case_task_trace_path",
            "case_trace_path",
            "case_with_result",
            "combine_suite_detection_artifacts",
            "execute_case",
            "export_case_detection_artifacts_if_requested",
            "export_suite_tables",
            "get_item",
            "run_case",
            "run_case_error",
            "run_case_success",
            "run_cases",
            "run_or_plan",
            "run_suite",
            "should_export_measurements",
            "sleep_before_case_retry",
        ]
        assert runner._build_contexts is execution.build_contexts
        assert runner._run_cases is execution.run_cases
        assert runner._run_case_success is execution.run_case_success
        assert runner._case_run_paths is execution.case_run_paths
        assert runner.run_suite is not execution.run_suite
        assert runner._run_case is not execution.run_case
        assert runner._run_case_error is not execution.run_case_error
        assert runner._execute_case is not execution.execute_case
        assert runner.run_or_plan is not execution.run_or_plan
        assert runner.run_or_plan.__module__ == "compat_benchmark_b3_facade"
    finally:
        sys.modules.pop("compat_benchmark_b3_facade", None)


@pytest.mark.parametrize(
    "canonical_module",
    [
        "measurement_tools.wandb_policy",
        "measurement_tools.wandb_settings",
        "measurement_tools.wandb_metadata",
        "measurement_tools.wandb_metrics",
        "measurement_tools.wandb_publication",
        "measurement_tools.wandb_field_policies",
    ],
)
def test_wandb_model_facade_reexports_every_canonical_symbol(canonical_module: str) -> None:
    models = importlib.import_module("measurement_tools.wandb_models")
    canonical = importlib.import_module(canonical_module)

    for name in canonical.__all__:
        assert getattr(models, name) is getattr(canonical, name)


@pytest.mark.parametrize(
    ("script", "help_text"),
    [
        ("create_wandb_report.py", "Create a W&B benchmark report"),
        ("run_benchmarks.py", "Run Anonymizer benchmark suites"),
        ("sweep_benchmarks.py", "Run a benchmark suite sweep"),
        ("analyze_benchmark_output.py", "Analyze joined benchmark measurements"),
    ],
)
def test_executable_tool_paths_preserve_help_and_main_behavior(script: str, help_text: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(MEASUREMENT_ROOT), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, str(MEASUREMENT_ROOT / script), "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert help_text in result.stdout
    assert "Usage:" in result.stdout


def test_observed_monkeypatch_namespaces_remain_available() -> None:
    setup = importlib.import_module("measurement_tools.wandb_setup")
    report = _load_module(MEASUREMENT_ROOT / "create_wandb_report.py", "compat_report_patches")
    runner = _load_module(MEASUREMENT_ROOT / "run_benchmarks.py", "compat_runner_patches")
    sweep = _load_module(MEASUREMENT_ROOT / "sweep_benchmarks.py", "compat_sweep_patches")
    surfaces: list[tuple[Any, tuple[str, ...]]] = [
        (
            setup,
            (
                "_PUBLISHER_WANDB_MODULE",
                "prepare_wandb_staging_dir",
                "_effective_wandb_tags",
                "publish_benchmark_wandb_best_effort",
            ),
        ),
        (
            setup.os,
            ("open", "close", "fstat"),
        ),
        (
            report,
            (
                "require_wandb_report_sdk",
                "require_wandb_workspace_sdk",
                "build_benchmark_report",
                "build_benchmark_group_report",
                "build_benchmark_workspace",
                "_save_report",
                "_save_workspace",
                "create_benchmark_report",
                "create_benchmark_group_report",
                "create_benchmark_workspace",
            ),
        ),
        (
            runner,
            (
                "Anonymizer",
                "_run_case",
                "_execute_case",
                "_run_case_error",
                "configured_measurement_session",
                "export_measurement_tables",
                "combine_detection_artifact_analysis",
                "_git_metadata",
                "_package_versions",
                "publish_benchmark_wandb_best_effort",
                "run_suite",
                "run_or_plan",
            ),
        ),
        (
            sweep,
            (
                "create_benchmark_workspace",
                "create_benchmark_group_report",
                "run_benchmarks",
                "run_sweep",
            ),
        ),
    ]
    try:
        for namespace, names in surfaces:
            for name in names:
                assert hasattr(namespace, name), f"{namespace.__name__}.{name}"
    finally:
        for module_name in ("compat_report_patches", "compat_runner_patches", "compat_sweep_patches"):
            sys.modules.pop(module_name, None)


def _inherited_policy_names(
    model: type[BaseModel],
    registry: dict[type[BaseModel], dict[str, Any]],
) -> Iterator[str]:
    for parent in reversed(model.mro()):
        if issubclass(parent, BaseModel):
            yield from registry.get(parent, {})


def test_outbound_field_policy_validation_is_complete_at_import() -> None:
    models = importlib.import_module("measurement_tools.wandb_models")

    models.validate_outbound_field_policies()
    for model, _policies in models.OUTBOUND_FIELD_POLICIES.items():
        assert set(model.model_fields) == set(_inherited_policy_names(model, models.OUTBOUND_FIELD_POLICIES))


@pytest.mark.parametrize(
    ("relative_path", "anchor"),
    [
        ("tools/measurement/import_wandb_run.py", "from measurement_tools.wandb_models import"),
        ("tools/measurement/measurement_tools/wandb_logging.py", "from measurement_tools.wandb_models import"),
        ("tools/measurement/measurement_tools/wandb_report_models.py", "from measurement_tools.wandb_models import"),
        ("tools/measurement/sweep_benchmarks.py", "import run_benchmarks"),
        ("tests/tools/test_measurement_strict_import_publisher.py", "WandbPublisher.__module__"),
        (".github/workflows/benchmark-ci.yml", "uv run python tools/measurement/run_benchmarks.py"),
        ("tools/measurement/examples/run-repo-data-smoke-with-dd-traces.sh", "tools/measurement/run_benchmarks.py"),
        ("docs/development/observability.md", "tools/measurement/create_wandb_report.py"),
    ],
)
def test_direct_consumer_anchors_remain_at_observed_paths(relative_path: str, anchor: str) -> None:
    assert anchor in (REPO_ROOT / relative_path).read_text(encoding="utf-8")
