# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every Anonymizer body runs on the thread that called the public entry point, as in the release.

The released facade ran detection, replacement, every DataDesigner call and every caller override on
the caller's thread, including builtin code reached through ``super()`` from a caller override. These
cases use a real DataDesigner whose model clients answer from a scripted transport, so the adapter is
native, and record the thread of each DataDesigner call and each caller override.
"""

from __future__ import annotations

import asyncio
import signal
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.interface.test_rewrite_caller_thread import _close, _designer, _model_configs

pytestmark = pytest.mark.compat

_METHODS = ["redact", "annotate", "hash", "substitute", "rewrite"]


def _config(method: str) -> Any:
    from anonymizer import AnonymizerConfig, Rewrite
    from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute

    if method == "rewrite":
        return AnonymizerConfig(rewrite=Rewrite(), emit_telemetry=False)
    replace = {"redact": Redact(), "annotate": Annotate(), "hash": Hash(), "substitute": Substitute()}[method]
    return AnonymizerConfig(replace=replace, emit_telemetry=False)


def _run(
    tmp_path: Path, *, method: str, detector: str = "builtin", replacer: str = "builtin", preview: bool = False
) -> tuple[int, list[tuple[str, int, bool]]]:
    from anonymizer import Anonymizer, AnonymizerInput
    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.ndd.adapter import NddAdapter
    from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
    from anonymizer.engine.replace.replace_runner import ReplacementWorkflow

    seen: list[tuple[str, int, bool]] = []

    def probe(label: str) -> None:
        # Caller code may install a signal handler exactly when it runs on the main thread.
        try:
            signal.signal(signal.SIGUSR1, signal.getsignal(signal.SIGUSR1))
            ok = True
        except ValueError:
            ok = False
        seen.append((label, threading.get_ident(), ok))

    designer, clients = _designer(tmp_path)
    for name in ("create", "preview"):
        original = getattr(designer, name)

        def wrapped(*args: Any, _original: Any = original, _name: str = name, **kwargs: Any) -> Any:
            probe(f"data_designer.{_name}")
            return _original(*args, **kwargs)

        setattr(designer, name, wrapped)

    class Delegating(EntityDetectionWorkflow):
        def run(self, *args: Any, **kwargs: Any) -> Any:
            probe("detector.run")
            return super().run(*args, **kwargs)

    class DelegatingWithStage(Delegating):
        def detect_and_validate_entities(self, *args: Any, **kwargs: Any) -> Any:
            probe("detector.detect_and_validate_entities")
            return super().detect_and_validate_entities(*args, **kwargs)

    class Replacer(ReplacementWorkflow):
        def run(self, *args: Any, **kwargs: Any) -> Any:
            probe("replacer.run")
            return super().run(*args, **kwargs)

    extra: dict[str, Any] = {}
    if detector == "delegating":
        extra["detection_workflow"] = Delegating(NddAdapter(designer))
    elif detector == "delegating-stage":
        extra["detection_workflow"] = DelegatingWithStage(NddAdapter(designer))
    if replacer == "delegating":
        adapter = NddAdapter(designer)
        extra["replace_runner"] = Replacer(llm_workflow=LlmReplaceWorkflow(adapter=adapter), adapter=adapter)
    anonymizer = Anonymizer(data_designer=designer, model_configs=_model_configs(), **extra)
    path = tmp_path / "input.csv"
    records = ["Alice travels.", "Alice stays."]
    pd.DataFrame({"biography": records}).to_csv(path, index=False)
    data = AnonymizerInput(source=str(path), text_column="biography")
    try:
        if preview:
            result = anonymizer.preview(config=_config(method), data=data, num_records=len(records))
        else:
            result = anonymizer.run(config=_config(method), data=data)
    finally:
        _close(clients)
    assert len(result.dataframe) == len(records)
    return threading.get_ident(), seen


def _off_caller(caller: int, seen: list[tuple[str, int, bool]]) -> list[str]:
    return [label for label, ident, _ in seen if ident != caller]


@pytest.mark.parametrize("preview", [False, True], ids=["run", "preview"])
@pytest.mark.parametrize("method", ["redact", "annotate", "hash"])
def test_builtin_local_replacement_runs_data_designer_on_caller_thread(
    tmp_path: Path, method: str, preview: bool
) -> None:
    caller, seen = _run(tmp_path, method=method, preview=preview)

    assert {label for label, _, _ in seen} == {"data_designer.preview" if preview else "data_designer.create"}
    assert _off_caller(caller, seen) == []
    assert all(ok for _, _, ok in seen)


@pytest.mark.parametrize("detector", ["delegating", "delegating-stage"])
@pytest.mark.parametrize("method", _METHODS)
def test_builtin_detection_under_a_delegating_detector_runs_on_caller_thread(
    tmp_path: Path, method: str, detector: str
) -> None:
    caller, seen = _run(tmp_path, method=method, detector=detector)

    labels = {label for label, _, _ in seen}
    assert {"detector.run", "data_designer.create"} <= labels
    if detector == "delegating-stage":
        assert "detector.detect_and_validate_entities" in labels
    assert _off_caller(caller, seen) == []
    assert all(ok for _, _, ok in seen)


@pytest.mark.parametrize("detector", ["builtin", "delegating"])
def test_builtin_substitute_under_a_delegating_replace_runner_runs_on_caller_thread(
    tmp_path: Path, detector: str
) -> None:
    caller, seen = _run(tmp_path, method="substitute", detector=detector, replacer="delegating")

    assert {"replacer.run", "data_designer.create"} <= {label for label, _, _ in seen}
    assert _off_caller(caller, seen) == []


@pytest.mark.parametrize("method", ["redact", "substitute", "rewrite"])
def test_placement_follows_a_non_main_caller_thread(tmp_path: Path, method: str) -> None:
    outcome: list[Any] = []

    def serve() -> None:
        try:
            outcome.append(_run(tmp_path, method=method, detector="delegating"))
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=serve, name="service-worker-1")
    thread.start()
    thread.join(timeout=300)
    assert not thread.is_alive()
    assert not isinstance(outcome[0], BaseException), outcome[0]
    caller, seen = outcome[0]
    assert caller == thread.ident
    assert _off_caller(caller, seen) == []
    # Off the main thread, caller code cannot install signal handlers, as in the release.
    assert not any(ok for _, _, ok in seen)


@pytest.mark.parametrize("method", ["redact", "substitute", "rewrite"])
def test_placement_from_inside_a_running_event_loop(tmp_path: Path, method: str) -> None:
    async def handler() -> tuple[tuple[int, list[tuple[str, int, bool]]], bool]:
        ran = _run(tmp_path, method=method, detector="delegating")
        return ran, asyncio.get_running_loop().is_running()

    (caller, seen), still_running = asyncio.run(handler())

    assert still_running
    assert seen
    assert _off_caller(caller, seen) == []


def test_threads_started_per_run(tmp_path: Path) -> None:
    started: list[str] = []
    original = threading.Thread.start

    def start(self: threading.Thread) -> None:
        started.append(self.name)
        original(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(threading.Thread, "start", start)
        _run(tmp_path, method="redact")

    # DataDesigner may start its own threads; Anonymizer starts none: no body worker and no loop host.
    assert [name for name in started if name.startswith("anonymizer-graph")] == []
