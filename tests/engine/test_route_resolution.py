# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native vs opaque routing per runner shape, observed at each entry's executor boundary.

Each row pins how the entry classifies its runner (``opaque`` or native) and that every entry hands
its synchronous body to the caller's thread, whatever the route: a caller's code, or a caller's
adapter, runs opaque; builtin code on a builtin adapter is lowered natively. Where each body actually
runs is pinned end to end in ``tests/interface/test_thread_placement.py``.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.ndd._route import builtin_detection, native_adapter, native_map
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow

_HANDOFF = object()


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
    """Run an entry's kernel operation with a marked handoff; stop at the executor and keep what it received."""
    from anonymizer.engine.harness import kernel

    seen: dict[str, Any] = {}

    def run_sync_profile(operation: Any, *, caller_callbacks: bool) -> Any:
        seen["caller_callbacks"] = caller_callbacks
        try:
            asyncio.run(operation(None, _HANDOFF))
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
    assert seen["caller_callbacks"] is True
    assert seen["args"][5] is _HANDOFF


@pytest.mark.parametrize(
    ("receiver", "opaque"),
    [
        (lambda: LlmReplaceWorkflow(_native()), False),
        (lambda: LlmReplaceWorkflow(_overridden()), True),
        (lambda: _SubclassMap(_native()), True),
    ],
    ids=["builtin", "overridden-adapter", "subclass-map"],
)
def test_direct_map_entry(monkeypatch: pytest.MonkeyPatch, receiver: Any, opaque: bool) -> None:
    from anonymizer.engine.replace import _semantic_map_profile

    seen: dict[str, Any] = {}

    def run_sync_profile(operation: Any, *, caller_callbacks: bool) -> Any:
        seen["caller_callbacks"] = caller_callbacks
        raise _Stop

    monkeypatch.setattr(_semantic_map_profile, "_run_sync_profile", run_sync_profile)
    monkeypatch.setattr(_semantic_map_profile, "map_configuration", lambda arguments: None)
    with pytest.raises(_Stop):
        _semantic_map_profile.run_direct_map(receiver(), _SOURCE, {})
    assert native_map(receiver()) is not opaque
    assert seen["caller_callbacks"] is True


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
    assert seen["caller_callbacks"] is True
    assert seen["kwargs"]["opaque"] is opaque
    assert seen["kwargs"]["handoff"] is _HANDOFF


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
    assert seen["caller_callbacks"] is True
    assert seen["kwargs"]["opaque"] is opaque
    assert seen["kwargs"]["handoff"] is _HANDOFF


_DETECTORS = [
    (lambda: EntityDetectionWorkflow(_native()), True),
    (lambda: EntityDetectionWorkflow(_overridden()), True),
    (lambda: _SubclassDetection(_native()), False),
    (lambda: _MutateThenDelegate(_native()), False),
    (lambda: _DuckDetection(_native()), False),
    (lambda: _Signature040Detection(_native()), False),
]
_DETECTOR_IDS = ["builtin", "builtin-overridden-adapter", "subclass", "mutate-then-delegate", "duck", "sig040"]


@pytest.mark.parametrize(("detector", "builtin"), _DETECTORS, ids=_DETECTOR_IDS)
@pytest.mark.parametrize("method", [Redact(), Hash(), Annotate()], ids=["redact", "hash", "annotate"])
def test_detection_redact_entry_opaque(
    monkeypatch: pytest.MonkeyPatch, detector: Any, builtin: bool, method: Any
) -> None:
    from anonymizer.engine.detection import _graph_profile, _opaque_redact

    seen = _observe(monkeypatch, _opaque_redact, "execute_injected_redact")
    with pytest.raises(_Stop):
        _graph_profile.run_detection_redact(
            _SOURCE,
            detection=detector(),
            model_configs=[],
            selected_models=_any(None),
            detection_options={},
            replace_method=method,
            opaque=True,
        )
    assert builtin_detection(detector()) is builtin
    assert seen["caller_callbacks"] is True
    assert seen["args"][0].handoff is _HANDOFF


@pytest.mark.parametrize(("detector", "builtin"), _DETECTORS, ids=_DETECTOR_IDS)
def test_detection_redact_entry_native(monkeypatch: pytest.MonkeyPatch, detector: Any, builtin: bool) -> None:
    from anonymizer.engine.detection import _graph_profile

    seen = _observe(monkeypatch, _graph_profile, "execute_detection_redact")
    with pytest.raises(_Stop):
        _graph_profile.run_detection_redact(
            _SOURCE,
            detection=detector(),
            model_configs=[],
            selected_models=_any(None),
            detection_options={},
        )
    assert builtin_detection(detector()) is builtin
    assert seen["caller_callbacks"] is True
    assert "handoff" not in seen["kwargs"]


@pytest.mark.parametrize(("detector", "builtin"), _DETECTORS, ids=_DETECTOR_IDS)
@pytest.mark.parametrize(
    "replacement", [ReplacementWorkflow, _SubclassReplacement], ids=["builtin-replacement", "subclass-replacement"]
)
def test_detection_substitute_entry(
    monkeypatch: pytest.MonkeyPatch, detector: Any, builtin: bool, replacement: Any
) -> None:
    from anonymizer.engine.detection import _graph_profile, _opaque_redact

    seen = _observe(monkeypatch, _opaque_redact, "execute_injected_redact")
    with pytest.raises(_Stop):
        _graph_profile.run_detection_redact(
            _SOURCE,
            detection=detector(),
            model_configs=[],
            selected_models=_any(None),
            detection_options={},
            replace_method=Substitute(),
            replacement=replacement(llm_workflow=LlmReplaceWorkflow(_native())),
            replacement_models=_any(SimpleNamespace()),
        )
    assert seen["caller_callbacks"] is True
    assert seen["args"][0].handoff is _HANDOFF


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


@pytest.mark.parametrize(
    ("receiver", "patch_class", "opaque"),
    [
        (lambda: LlmReplaceWorkflow(_native()), True, False),
        (lambda: _BorrowedMap(_native()), False, False),
        (lambda: _Signature040Map(_native()), False, True),
        (lambda: LlmReplaceWorkflow(_patched_instance()), False, True),
    ],
    ids=["class-patched-delegating", "borrowed-function", "sig040", "patched-instance-adapter"],
)
def test_direct_map_entry_method_shapes(
    monkeypatch: pytest.MonkeyPatch, receiver: Any, patch_class: bool, opaque: bool
) -> None:
    from anonymizer.engine.replace import _semantic_map_profile

    seen: dict[str, Any] = {}

    def run_sync_profile(operation: Any, *, caller_callbacks: bool) -> Any:
        seen["caller_callbacks"] = caller_callbacks
        raise _Stop

    if patch_class:
        monkeypatch.setattr(LlmReplaceWorkflow, "generate_map_only", _delegating_generate_map_only)
    monkeypatch.setattr(_semantic_map_profile, "_run_sync_profile", run_sync_profile)
    monkeypatch.setattr(_semantic_map_profile, "map_configuration", lambda arguments: None)
    with pytest.raises(_Stop):
        _semantic_map_profile.run_direct_map(receiver(), _SOURCE, {})
    assert native_map(receiver()) is not opaque
    assert seen["caller_callbacks"] is True


def _designer_stub() -> Any:
    run_config = SimpleNamespace(max_conversation_correction_steps=0, max_conversation_restarts=0)
    return _any(SimpleNamespace(run_config=run_config))


@pytest.mark.parametrize(
    ("receiver", "native"),
    [
        (lambda: LlmReplaceWorkflow(NddAdapter(_designer_stub())), True),
        (lambda: _SubclassMap(NddAdapter(_designer_stub())), True),
        (lambda: LlmReplaceWorkflow(_OverriddenAdapter(_designer_stub())), False),
        (lambda: _SubclassColumns(NddAdapter(_designer_stub())), False),
        (lambda: SimpleNamespace(_generate_columns=None), False),
    ],
    ids=["builtin", "subclass-map-only", "overridden-adapter", "subclass-columns", "adapterless-duck"],
)
def test_native_map_columns_gate(receiver: Any, native: bool) -> None:
    from anonymizer.config.models import ReplaceModelSelection
    from anonymizer.engine.replace._native_map_requests import native_map_capacity

    arguments = {"selected_models": ReplaceModelSelection(replacement_generator="m"), "model_configs": []}
    assert (native_map_capacity(receiver(), arguments) is not None) is native


class _GateSeen(Exception):
    pass


@pytest.mark.parametrize(
    ("detector", "replacement", "patch_class", "semantic_map"),
    [
        (lambda: EntityDetectionWorkflow(_native()), ReplacementWorkflow, False, True),
        (lambda: EntityDetectionWorkflow(_native()), _SubclassReplacement, False, False),
        (lambda: _SubclassDetection(_native()), ReplacementWorkflow, False, False),
        (lambda: EntityDetectionWorkflow(_native()), ReplacementWorkflow, True, False),
    ],
    ids=["builtin", "subclass-replacement", "subclass-detection", "class-patched-map"],
)
def test_injected_redact_map_gate(
    monkeypatch: pytest.MonkeyPatch, detector: Any, replacement: Any, patch_class: bool, semantic_map: bool
) -> None:
    import sys

    from anonymizer.engine.detection import _graph_profile, _opaque_redact
    from anonymizer.engine.harness import kernel

    seen: dict[str, Any] = {}

    class _Graph:
        @staticmethod
        def new() -> Any:
            seen["semantic_map"] = sys._getframe(1).f_locals["semantic_map"]
            raise _GateSeen

    def run_sync_profile(operation: Any, *, caller_callbacks: bool) -> Any:
        asyncio.run(operation(None, _HANDOFF))

    if patch_class:
        monkeypatch.setattr(LlmReplaceWorkflow, "generate_map_only", _delegating_generate_map_only)
    monkeypatch.setattr(kernel, "_run_sync_profile", run_sync_profile)
    monkeypatch.setattr(_opaque_redact, "DataGraph", _Graph)
    with pytest.raises(_GateSeen):
        _graph_profile.run_detection_redact(
            _SOURCE,
            detection=detector(),
            model_configs=[],
            selected_models=_any(None),
            detection_options={},
            replace_method=Substitute(),
            replacement=replacement(llm_workflow=LlmReplaceWorkflow(_native())),
            replacement_models=_any(SimpleNamespace()),
        )
    assert seen["semantic_map"] is semantic_map


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
        ("substitute", "builtin", {"detection": [True], "map": [True]}),
        ("substitute", "overridden-adapter", {"detection": [False], "map": [True]}),
        ("substitute", "mutate-then-delegate", {"detection": [False], "map": [True]}),
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
