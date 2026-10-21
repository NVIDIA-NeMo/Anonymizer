# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The source-frame binding checks DataDesigner's private hooks, not its exact version.

Every case runs against fake DataDesigner modules and classes, so it pins the binding's behavior on a
release shaped differently from the installed one.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest

from anonymizer.engine.ndd import _graph_requests
from anonymizer.engine.ndd._graph_requests import SUPPORTED_DATA_DESIGNER_RANGE
from anonymizer.interface.errors import AnonymizerWorkflowError

_MISSING = object()


@pytest.fixture
def fake_dd(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Install a fake DataDesigner artifact-storage module and report installed version 0.9.9."""
    storage = types.ModuleType("fake_artifact_storage")
    setattr(storage, "BatchStage", SimpleNamespace(PARTIAL_RESULT="partial", FINAL_RESULT="final"))
    setattr(storage, "SDG_CONFIG_FILENAME", "sdg.json")
    monkeypatch.setitem(sys.modules, _graph_requests._ARTIFACT_STORAGE_MODULE, storage)
    monkeypatch.setattr(_graph_requests, "version", lambda _name: "0.9.9")
    return types.SimpleNamespace(storage=storage)


def _assert_capability_message(error: BaseException, attribute: str) -> None:
    assert str(error) == (
        f"Anonymizer's DataDesigner binding requires DataDesigner's private `{attribute}`, which the installed "
        f"data-designer 0.9.9 does not provide (supported: data-designer {SUPPORTED_DATA_DESIGNER_RANGE})"
    )


def test_pyproject_range_matches_supported_range() -> None:
    import tomllib
    from pathlib import Path

    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    declared = [dep for dep in pyproject["project"]["dependencies"] if dep.startswith("data-designer")]
    assert declared == [f"data-designer{SUPPORTED_DATA_DESIGNER_RANGE}"]


def test_bridge_modules_import_no_data_designer_internals_at_module_scope() -> None:
    """Importing ``anonymizer`` must not fail on a release that moves a hook the binding checks before it runs."""
    import ast
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "src" / "anonymizer"
    lazy = {"engine/ndd/_graph_requests.py", "engine/harness/kernel.py", "engine/detection/_graph_profile.py"}
    offenders = []
    for path in sorted(src.rglob("*.py")):
        relative = path.relative_to(src).as_posix()
        for node in ast.parse(path.read_text()).body:
            if (
                not isinstance(node, ast.ImportFrom)
                or node.module is None
                or not node.module.startswith("data_designer")
            ):
                continue
            names = [alias.name for alias in node.names]
            if any(name.startswith("_") for name in names) or (
                relative in lazy and node.module.startswith("data_designer.engine")
            ):
                offenders.append(f"{relative}: from {node.module} import {', '.join(names)}")
    assert offenders == []


class _BuilderDesigner:
    def _create_dataset_builder(self) -> None:
        pass


def test_source_frame_hooks_resolve(fake_dd: types.SimpleNamespace) -> None:
    _graph_requests.require_source_frame_capabilities(_BuilderDesigner())


def test_designer_without_dataset_builder_hook_fails_before_any_work(fake_dd: types.SimpleNamespace) -> None:
    with pytest.raises(AnonymizerWorkflowError) as raised:
        _graph_requests.require_source_frame_capabilities(SimpleNamespace())
    _assert_capability_message(raised.value, "DataDesigner._create_dataset_builder")


@pytest.mark.parametrize(
    ("change", "attribute"),
    [
        ("stage", "BatchStage.FINAL_RESULT"),
        ("filename", f"{_graph_requests._ARTIFACT_STORAGE_MODULE}.SDG_CONFIG_FILENAME"),
    ],
)
def test_artifact_storage_module_hooks_fail_before_any_work(
    fake_dd: types.SimpleNamespace, change: str, attribute: str
) -> None:
    if change == "stage":
        setattr(fake_dd.storage, "BatchStage", SimpleNamespace(PARTIAL_RESULT="partial"))
    else:
        delattr(fake_dd.storage, "SDG_CONFIG_FILENAME")
    with pytest.raises(AnonymizerWorkflowError) as raised:
        _graph_requests.require_source_frame_capabilities(_BuilderDesigner())
    _assert_capability_message(raised.value, attribute)


def _builder(**overrides: Any) -> SimpleNamespace:
    storage = {
        "write_parquet_file": lambda *args, **kwargs: None,
        "load_dataset": lambda *args, **kwargs: None,
        "write_metadata": lambda *args, **kwargs: None,
        "mkdir_if_needed": lambda *args, **kwargs: None,
        "base_dataset_path": "base",
        "final_dataset_path": "final",
    }
    builder = {
        "_initialize_generators_and_graph": lambda: ([], None),
        "_write_builder_config": lambda: None,
        "_data_designer_config": object(),
    }
    for key, value in overrides.items():
        target = storage if key in storage else builder
        if value is _MISSING:
            del target[key]
        else:
            target[key] = value
    return SimpleNamespace(**builder, artifact_storage=SimpleNamespace(**storage))


def test_complete_dataset_builder_passes(fake_dd: types.SimpleNamespace) -> None:
    builder = _builder()
    assert _graph_requests.require_dataset_builder_capabilities(builder) is builder.artifact_storage


@pytest.mark.parametrize(
    ("missing", "attribute"),
    [
        ("_initialize_generators_and_graph", "DatasetBuilder._initialize_generators_and_graph"),
        ("_write_builder_config", "DatasetBuilder._write_builder_config"),
        ("_data_designer_config", "DatasetBuilder._data_designer_config"),
        ("write_parquet_file", "ArtifactStorage.write_parquet_file"),
        ("load_dataset", "ArtifactStorage.load_dataset"),
        ("write_metadata", "ArtifactStorage.write_metadata"),
        ("mkdir_if_needed", "ArtifactStorage.mkdir_if_needed"),
        ("final_dataset_path", "ArtifactStorage.final_dataset_path"),
    ],
)
def test_dataset_builder_without_hook_fails_at_creation(
    fake_dd: types.SimpleNamespace, missing: str, attribute: str
) -> None:
    with pytest.raises(AnonymizerWorkflowError) as raised:
        _graph_requests.require_dataset_builder_capabilities(_builder(**{missing: _MISSING}))
    _assert_capability_message(raised.value, attribute)
