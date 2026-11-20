# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gc
import threading
import weakref
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


def _concurrently(monkeypatch: pytest.MonkeyPatch, *runs: Callable[[], Any]) -> list[Any]:
    """Run ``runs`` on threads that must all be inside DataDesigner's builder at once; a run that cannot get there
    while another is in flight (runs serialized) breaks the barrier and fails the test."""
    from data_designer.engine.models.registry import ModelRegistry

    async def no_check(self: ModelRegistry, model_aliases: list[str]) -> None:
        return None

    monkeypatch.setattr(ModelRegistry, "run_health_check", lambda self, model_aliases: None)
    monkeypatch.setattr(ModelRegistry, "arun_health_check", no_check)
    barrier = threading.Barrier(len(runs), timeout=20)
    original = DataDesigner._create_dataset_builder

    def paused(self: DataDesigner, *args: Any, **kwargs: Any) -> Any:
        barrier.wait()
        return original(self, *args, **kwargs)

    monkeypatch.setattr(DataDesigner, "_create_dataset_builder", paused)
    with ThreadPoolExecutor(len(runs)) as pool:
        return [future.result() for future in [pool.submit(run) for run in runs]]


def test_runs_sharing_a_designer_run_concurrently_on_their_own_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two adapters on one designer, both inside DataDesigner at once: each run's hooks see only its own calls."""
    designer = _designer(tmp_path)

    def run(tag: str) -> list[str]:
        return list(
            NddAdapter(data_designer=designer)
            .run_workflow(
                pd.DataFrame({"text": [f"{tag}-x", f"{tag}-y"]}),
                model_configs=[ModelConfig(alias="known", model="stub-model", provider="stub", skip_health_check=True)],
                columns=[ExpressionColumnConfig(name="upper", expr="{{ text }}!")],
                workflow_name=f"shared-{tag}",
            )
            .dataframe["upper"]
        )

    assert _concurrently(monkeypatch, lambda: run("one"), lambda: run("two")) == [
        ["one-x!", "one-y!"],
        ["two-x!", "two-y!"],
    ]
    assert "_create_dataset_builder" not in vars(designer)
    assert "_create_resource_provider" not in vars(designer)
    assert designer.run_config.async_trace is False


def test_facades_sharing_a_designer_run_concurrently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The public form: two ``Anonymizer`` facades given one designer, regex detection and Redact, no model calls."""
    from anonymizer import Anonymizer
    from anonymizer.config.anonymizer_config import AnonymizerConfig, Detect, TextRecord, TextRecordsInput
    from anonymizer.config.regex import RegexRule
    from anonymizer.config.replace_strategies import Redact

    designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[
            ModelProvider(name=name, endpoint="http://127.0.0.1:9/v1", provider_type="openai", api_key="EMPTY")
            for name in ("openrouter", "local-gliner2")
        ],
        auto_configure_logging=False,
    )
    rule = RegexRule(
        label="ticket", pattern=r"TKT-[0-9]+", validate_matches_with_llm=False, detect_additional_matches=False
    )
    config = AnonymizerConfig(detect=Detect(entity_labels=["ticket"], regex_rules=[rule]), replace=Redact())

    def run(tag: str) -> list[str]:
        data = TextRecordsInput(records=[TextRecord(id=tag, text=f"Escalate TKT-1234 for {tag}.")])
        return list(Anonymizer(data_designer=designer).run(config=config, data=data).dataframe["text_replaced"])

    assert _concurrently(monkeypatch, lambda: run("one"), lambda: run("two")) == [
        ["Escalate [REDACTED_TICKET] for one."],
        ["Escalate [REDACTED_TICKET] for two."],
    ]
