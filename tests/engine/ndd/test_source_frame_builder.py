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
