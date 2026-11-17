# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import weakref
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.interface.data_designer import DataDesigner

from anonymizer.engine.ndd.adapter import NddAdapter


def test_run_workflow_frees_its_dataset_builder_without_a_gc_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live builder from an earlier run must not outlast the run: reference counting alone frees it."""
    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[
            ModelProvider(name="stub", endpoint="http://127.0.0.1:9/v1", provider_type="openai", api_key="EMPTY")
        ],
        auto_configure_logging=False,
    )
    created: list[weakref.ref[Any]] = []
    original = designer._create_dataset_builder

    def spy(*args: Any, **kwargs: Any) -> Any:
        builder = original(*args, **kwargs)
        created.append(weakref.ref(builder))
        return builder

    monkeypatch.setattr(designer, "_create_dataset_builder", spy)
    gc.disable()
    try:
        NddAdapter(data_designer=designer).run_workflow(
            pd.DataFrame({"text": ["a", "b"]}),
            model_configs=[ModelConfig(alias="known", model="stub-model", provider="stub", skip_health_check=True)],
            columns=[ExpressionColumnConfig(name="upper", expr="{{ text }}")],
            workflow_name="free-builder",
        )
        assert created
        assert all(ref() is None for ref in created)
    finally:
        gc.enable()


def _designer(tmp_path: Path) -> DataDesigner:
    return DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[
            ModelProvider(name="stub", endpoint="http://127.0.0.1:9/v1", provider_type="openai", api_key="EMPTY")
        ],
        auto_configure_logging=False,
    )


def _run(designer: DataDesigner) -> None:
    NddAdapter(data_designer=designer).run_workflow(
        pd.DataFrame({"text": ["a", "b"]}),
        model_configs=[ModelConfig(alias="known", model="stub-model", provider="stub", skip_health_check=True)],
        columns=[ExpressionColumnConfig(name="upper", expr="{{ text }}")],
        workflow_name="free-designer",
    )


def test_run_workflow_leaves_no_override_that_keeps_the_designer_alive(tmp_path: Path) -> None:
    """Undoing the run's builder and resource-provider overrides must not leave the designer referring to itself."""
    designer = _designer(tmp_path)
    ref = weakref.ref(designer)
    gc.disable()
    try:
        _run(designer)
        assert "_create_dataset_builder" not in vars(designer)
        assert "_create_resource_provider" not in vars(designer)
        del designer
        assert ref() is None
    finally:
        gc.enable()


def test_run_workflow_frees_its_artifact_storage_without_a_gc_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Undoing the run's storage overrides must not leave the storage referring to itself."""
    designer = _designer(tmp_path)
    storages: list[weakref.ref[Any]] = []
    original = DataDesigner._create_dataset_builder

    def spy(self: DataDesigner, *args: Any, **kwargs: Any) -> Any:
        builder = original(self, *args, **kwargs)
        storages.append(weakref.ref(builder.artifact_storage))
        return builder

    # Patched on the class, so the designer instance gains no attribute of its own.
    monkeypatch.setattr(DataDesigner, "_create_dataset_builder", spy)
    gc.disable()
    try:
        _run(designer)
        assert storages
        assert all(ref() is None for ref in storages)
    finally:
        gc.enable()
