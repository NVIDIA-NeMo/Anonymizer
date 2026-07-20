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
        ("run_benchmarks.py", "BenchmarkSpec", "Anonymizer", None),
        ("sweep_benchmarks.py", "SweepSpec", "WandbProjectPath", None),
        ("analyze_benchmark_output.py", "BenchmarkOutputAnalysis", "AnalysisExportResult", None),
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
