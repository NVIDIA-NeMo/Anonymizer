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

from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd._route import native_adapter
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
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


class _SubclassMap(LlmReplaceWorkflow):
    def generate_map_only(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
        return super().generate_map_only(dataframe, **kwargs)


class _SubclassRewrite(RewriteWorkflow):
    def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
        return super().run(dataframe, **kwargs)


class _DuckRewrite:
    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
        return dataframe


class _SubclassReplacement(ReplacementWorkflow):
    def run(self, *args: Any, **kwargs: Any) -> Any:
        return super().run(*args, **kwargs)


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


_ARGUMENTS: Any = {"evaluation": SimpleNamespace(max_repair_iterations=1), "preview_num_records": None}


@pytest.mark.parametrize(
    ("runner", "opaque"),
    [
        (lambda: RewriteWorkflow(_native()), False),
        (lambda: CombinedRewriteWorkflow(_native()), False),
        (lambda: RewriteWorkflow(_overridden()), True),
        (lambda: RewriteWorkflow(_patched_instance()), True),
    ],
    ids=["builtin", "combined", "overridden-adapter", "patched-instance-adapter"],
)
def test_builtin_rewrite_entry(monkeypatch: pytest.MonkeyPatch, runner: Any, opaque: bool) -> None:
    from anonymizer.engine.rewrite import _semantic_profile

    seen = _observe(monkeypatch, _semantic_profile, "execute_initial_evaluation")
    with pytest.raises(_Stop):
        _semantic_profile.run_builtin_rewrite(runner(), _SOURCE, _ARGUMENTS)
    assert seen["hosted"]
    assert seen["kwargs"]["opaque"] is opaque


@pytest.mark.parametrize(
    ("adapter", "opaque"),
    [(_native, False), (_overridden, True), (_patched_instance, True)],
    ids=["native", "overridden-adapter", "patched-instance-adapter"],
)
def test_semantic_rewrite_entry(monkeypatch: pytest.MonkeyPatch, adapter: Any, opaque: bool) -> None:
    from anonymizer.engine.rewrite import _semantic_profile

    seen = _observe(monkeypatch, _semantic_profile, "execute_initial_evaluation")
    owned = adapter()
    with pytest.raises(_Stop):
        _semantic_profile.run_semantic_rewrite(
            _SOURCE,
            detection=EntityDetectionWorkflow(owned),
            adapter=owned,
            combined=False,
            arguments=_ARGUMENTS,
            detection_arguments={},
        )
    assert seen["hosted"]
    assert seen["kwargs"]["opaque"] is opaque


_DETECTORS = [
    (lambda: EntityDetectionWorkflow(_native()), True),
    (lambda: EntityDetectionWorkflow(_overridden()), True),
    (lambda: _SubclassDetection(_native()), False),
    (lambda: _MutateThenDelegate(_native()), False),
    (lambda: _DuckDetection(_native()), False),
    (lambda: _Signature040Detection(_native()), False),
]
_DETECTOR_IDS = ["builtin", "builtin-overridden-adapter", "subclass", "mutate-then-delegate", "duck", "sig040"]


class _Signature040Rewrite:
    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def run(self, dataframe, *, model_configs, selected_models, privacy_goal, evaluation):  # noqa: ANN001, ANN201
        return dataframe


def _rewrite_runner_route(runner: object) -> bool:
    """The facade's Rewrite routing check, called with whichever signature the tree under test has."""
    import inspect

    from anonymizer.interface import anonymizer as facade

    check: Any = facade.builtin_rewrite_runner
    if "combined" in inspect.signature(check).parameters:
        return check(runner, combined=False, supplied=False)
    return check(runner)


@pytest.mark.parametrize(
    ("runner", "builtin"),
    [
        (lambda: RewriteWorkflow(_native()), True),
        (lambda: CombinedRewriteWorkflow(_native()), True),
        (lambda: RewriteWorkflow(_overridden()), True),
        (lambda: _SubclassRewrite(_native()), False),
        (lambda: _DuckRewrite(_native()), False),
        (lambda: _Signature040Rewrite(_native()), False),
        (lambda: SimpleNamespace(run=RewriteWorkflow(_native()).run, _adapter=_native()), False),
    ],
    ids=["builtin", "combined", "overridden-adapter", "subclass", "duck", "sig040", "bound-method-duck"],
)
def test_facade_rewrite_runner_route(runner: Any, builtin: bool) -> None:
    assert _rewrite_runner_route(runner()) is builtin


def _delegating_generate_map_only(self: Any, *args: Any, **kwargs: Any) -> Any:
    return _GENERATE_MAP_ONLY(self, *args, **kwargs)


class _BorrowedMap:
    """A duck that borrows the builtin function as its own method."""

    generate_map_only = LlmReplaceWorkflow.generate_map_only

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter


class _Signature040Map:
    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def generate_map_only(self, dataframe, *, model_configs, selected_models):  # noqa: ANN001, ANN201
        return dataframe


class _SubclassColumns(LlmReplaceWorkflow):
    def _generate_columns(self, *args: Any, **kwargs: Any) -> Any:
        return super()._generate_columns(*args, **kwargs)


_GENERATE_MAP_ONLY = LlmReplaceWorkflow.generate_map_only


class _GateSeen(Exception):
    pass


_BINDING_SITES = {
    ("_semantic_detection.py", "lower"): "detection",
    ("_semantic_map.py", "lower"): "map",
    ("_semantic_profile.py", "prepare_maps"): "rewrite-map",
}


def _trace_binding_routes(run: Any) -> dict[str, list[bool]]:
    """Run end to end, recording the ``native`` each binding site decided when its frame returned."""
    import sys
    import threading

    seen: dict[str, list[bool]] = {}

    def local(frame: Any, event: str, arg: Any) -> Any:
        if event == "return" and "native" in frame.f_locals:
            name = frame.f_code.co_filename.rsplit("/", 1)[-1]
            seen.setdefault(_BINDING_SITES[(name, frame.f_code.co_name)], []).append(frame.f_locals["native"])
        return local

    def tracer(frame: Any, event: str, arg: Any) -> Any:
        if event == "call" and (frame.f_code.co_filename.rsplit("/", 1)[-1], frame.f_code.co_name) in _BINDING_SITES:
            return local
        return None

    threading.settrace(tracer)
    sys.settrace(tracer)
    try:
        run()
    finally:
        sys.settrace(None)
        threading.settrace(None)  # type: ignore[arg-type]
    return {key: sorted(set(values)) for key, values in seen.items()}


@pytest.mark.parametrize(
    ("mode", "detector", "expected"),
    [
        ("rewrite", "builtin", {"detection": [True], "map": [True], "rewrite-map": [True]}),
        ("rewrite", "overridden-adapter", {"detection": [False], "map": [True], "rewrite-map": [True]}),
        ("rewrite", "mutate-then-delegate", {"detection": [False], "map": [True], "rewrite-map": [True]}),
    ],
)
def test_binding_sites_route_end_to_end(
    tmp_path: Any, mode: str, detector: str, expected: dict[str, list[bool]]
) -> None:
    from anonymizer import Anonymizer, AnonymizerConfig, AnonymizerInput, Rewrite
    from tests.interface.test_rewrite_caller_thread import _close, _designer, _model_configs

    designer, clients = _designer(tmp_path)

    class MutateThenDelegate(EntityDetectionWorkflow):
        def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> Any:
            original = self._adapter.run_workflow
            cast(Any, self._adapter).run_workflow = lambda *args, **kw: original(*args, **kw)
            return super().run(dataframe, **kwargs)

    extra: dict[str, Any] = {}
    if detector == "overridden-adapter":
        extra["detection_workflow"] = EntityDetectionWorkflow(_OverriddenAdapter(designer))
    elif detector == "mutate-then-delegate":
        extra["detection_workflow"] = MutateThenDelegate(NddAdapter(designer))
    anonymizer = Anonymizer(data_designer=designer, model_configs=_model_configs(), **extra)
    path = tmp_path / "input.csv"
    pd.DataFrame({"biography": ["Alice travels.", "Alice stays."]}).to_csv(path, index=False)
    config = (
        AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False)
        if mode == "rewrite"
        else AnonymizerConfig(replace=Substitute(), emit_telemetry=False)
    )
    data = AnonymizerInput(source=str(path), text_column="biography")
    try:
        seen = _trace_binding_routes(lambda: anonymizer.run(config=config, data=data))
    finally:
        _close(clients)
    assert seen == expected


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
