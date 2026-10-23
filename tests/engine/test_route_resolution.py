# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native vs opaque routing per runner shape, observed at each entry's executor boundary.

Each row pins how the entry classifies its runner (``opaque`` or native) and that every entry is hosted
by the kernel, whatever the route: a caller's code, or a caller's
adapter, runs opaque; builtin code on a builtin adapter is lowered natively. Where each body actually
runs is pinned end to end in ``tests/interface/test_thread_placement.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd._route import native_adapter
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow


def _any(value: object) -> Any:
    return cast(Any, value)


class _Stop(Exception):
    pass


class _OverriddenAdapter(NddAdapter):
    def run_workflow(self, *args: Any, **kwargs: Any) -> Any:
        return super().run_workflow(*args, **kwargs)


class _SubclassDetection(EntityDetectionWorkflow):
    def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
        return super().run(dataframe, **kwargs)


class _MutateThenDelegate(EntityDetectionWorkflow):
    def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
        self.mutated = True
        return super().run(dataframe, **kwargs)


class _DuckDetection:
    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
        return dataframe


class _Signature040Detection:
    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def run(self, dataframe, *, model_configs, selected_models, gliner_detection_threshold):  # noqa: ANN001, ANN201
        return dataframe


def _native() -> NddAdapter:
    return NddAdapter(_any(SimpleNamespace()))


def _overridden() -> NddAdapter:
    return _OverriddenAdapter(_any(SimpleNamespace()))


def _patched_instance() -> NddAdapter:
    adapter = _native()
    original = adapter.run_workflow
    cast(Any, adapter).run_workflow = lambda *args, **kwargs: original(*args, **kwargs)
    return adapter


_SOURCE = pd.DataFrame({COL_TEXT: ["Alice met Bob."]})


def _observe(monkeypatch: pytest.MonkeyPatch, module: Any, name: str) -> dict[str, Any]:
    """Run an entry's kernel operation; stop at the executor and keep what it received."""
    from anonymizer.engine.harness import kernel

    seen: dict[str, Any] = {}

    def run_sync_profile(operation: Any) -> Any:
        seen["hosted"] = True
        try:
            asyncio.run(operation(None))
        except _Stop:
            pass
        raise _Stop

    async def executor(*args: Any, **kwargs: Any) -> Any:
        seen["executor"] = name
        seen["args"], seen["kwargs"] = args, kwargs
        raise _Stop

    monkeypatch.setattr(kernel, "_run_sync_profile", run_sync_profile)
    if hasattr(module, "_run_sync_profile"):
        monkeypatch.setattr(module, "_run_sync_profile", run_sync_profile)
    monkeypatch.setattr(module, name, executor)
    return seen


@pytest.mark.parametrize(
    ("adapter", "opaque"),
    [(_native, False), (_overridden, True), (_patched_instance, True)],
    ids=["native", "overridden-adapter", "patched-instance-adapter"],
)
def test_builtin_detection_entry(monkeypatch: pytest.MonkeyPatch, adapter: Any, opaque: bool) -> None:
    from anonymizer.engine.detection import _semantic_detection

    seen = _observe(monkeypatch, _semantic_detection, "execute_detection")
    with pytest.raises(_Stop):
        _semantic_detection.run_direct_detection(EntityDetectionWorkflow(adapter()), _SOURCE, {})
    assert native_adapter(adapter()) is not opaque
    assert seen["hosted"]


_DETECTORS = [
    (lambda: EntityDetectionWorkflow(_native()), True),
    (lambda: EntityDetectionWorkflow(_overridden()), True),
    (lambda: _SubclassDetection(_native()), False),
    (lambda: _MutateThenDelegate(_native()), False),
    (lambda: _DuckDetection(_native()), False),
    (lambda: _Signature040Detection(_native()), False),
]
_DETECTOR_IDS = ["builtin", "builtin-overridden-adapter", "subclass", "mutate-then-delegate", "duck", "sig040"]


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
def test_rewrite_runner_on_adapter_without_run_workflow_raises_its_attribute_error(
    tmp_path: Any, preview: bool
) -> None:
    from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite
    from tests.interface.test_rewrite_caller_thread import _close, _designer, _model_configs

    class Duck:
        pass

    designer, clients = _designer(tmp_path)
    anonymizer = Anonymizer(
        data_designer=designer, model_configs=_model_configs(), rewrite_runner=RewriteWorkflow(_any(Duck()))
    )
    path = tmp_path / "input.csv"
    pd.DataFrame({"biography": ["Alice travels.", "Alice stays."]}).to_csv(path, index=False)
    config = AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False)
    data = AnonymizerInput(source=str(path), text_column="biography")
    try:
        with pytest.raises(AttributeError, match="^'Duck' object has no attribute 'run_workflow'$"):
            if preview:
                anonymizer.preview(config=config, data=data, num_records=2)
            else:
                anonymizer.run(config=config, data=data)
    finally:
        _close(clients)
