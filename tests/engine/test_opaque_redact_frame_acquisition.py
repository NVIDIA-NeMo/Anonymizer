# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Acquisition-wait harness around ``_execute_frame_plan``'s native transport lease.

``_execute_frame_plan`` (in ``_opaque_redact.py``) schedules a ``create_resource``
coroutine on DataDesigner's engine loop and waits for it under ``asyncio.shield``.
These tests exercise that wait loop directly: a genuinely cancelled acquisition must
not busy-spin, and an outer-cancelled caller must preserve the *first* interruption
across either a later successful acquisition or a later non-cancel acquisition
failure, matching the guarded twin in ``_graph_profile.py``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

import pytest
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

from anonymizer.engine.detection import _opaque_redact
from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport


class _FakeTransport:
    def __init__(self) -> None:
        self.closed = False
        self._claimed = False

    async def close(self) -> None:
        self.closed = True


class _SlowClosingPhysical:
    """Stands in for ``NddPhysicalTransport``; its ``close`` suspends indefinitely.

    Used only to give the engine-loop ``create_resource`` task a real await point
    to be cancelled at (the factory failure path awaits ``value.close()``).
    """

    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        await asyncio.sleep(3600)


def _patch_owned_and_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "anonymizer.engine.replace._native_map_requests.bind_native_map_policy",
        lambda prepared, policies, nodes: policies,
    )

    async def owned(**kwargs: Any) -> tuple[str, Any]:
        return "owned-ran", kwargs["transport"]

    monkeypatch.setattr(_opaque_redact, "_execute_frame_plan_owned", owned)


def _call_frame_plan() -> Any:
    return _opaque_redact._execute_frame_plan(
        prepared=cast(Any, None),
        capabilities=(),
        policies=(),
        callback=cast(Any, None),
        pool=cast(Any, None),
        borrowed_loop=cast(Any, None),
        limits=cast(Any, None),
        ingress_limits=cast(Any, None),
        assessment_limits=cast(Any, None),
        native_request_nodes=cast(Any, {"probe": object()}),
    )


def test_cancelled_acquisition_raises_promptly_without_busy_spin(monkeypatch: pytest.MonkeyPatch) -> None:
    """A genuinely cancelled acquisition must be observed and re-raised, not retried forever."""
    _patch_owned_and_policy(monkeypatch)
    monkeypatch.setattr("anonymizer.engine.ndd._graph_requests.NddPhysicalTransport", _SlowClosingPhysical)

    def failing_create(cls: type[ForeignLoopTransport], **kwargs: Any) -> ForeignLoopTransport:
        raise RuntimeError("forced create failure")

    monkeypatch.setattr(ForeignLoopTransport, "create", classmethod(failing_create))

    engine_loop = ensure_async_engine_loop()
    outcome: dict[str, Any] = {}

    def runner() -> None:
        async def main() -> None:
            task = asyncio.ensure_future(_call_frame_plan())
            # Let create_resource start on the engine loop, fail in ForeignLoopTransport.create,
            # and suspend inside `await value.close()`.
            await asyncio.sleep(0.2)

            found = threading.Event()

            def cancel_when_present() -> None:
                for engine_task in asyncio.all_tasks(engine_loop):
                    coro = engine_task.get_coro()
                    code = getattr(coro, "cr_code", None)
                    if getattr(code, "co_name", "") == "create_resource":
                        engine_task.cancel()
                        found.set()
                        return
                engine_loop.call_later(0.02, cancel_when_present)

            engine_loop.call_soon_threadsafe(cancel_when_present)
            if not found.wait(timeout=5):
                outcome["result"] = "engine-task-not-found"
                return
            try:
                await task
                outcome["result"] = "no-error"
            except asyncio.CancelledError:
                outcome["result"] = "cancelled"

        asyncio.run(main())

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive(), (
        "caller did not return promptly after the acquisition was cancelled (busy-spin regression: engine loop starved)"
    )
    assert outcome.get("result") == "cancelled"


def test_outer_cancel_then_success_closes_transport_and_reraises_first(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller cancelled twice while pending; acquisition then succeeds -> close + first error.

    The acquisition's success is gated on a ``threading.Event`` set only after both outer
    cancels have been delivered and observed (each is followed by a yielding sleep so the
    cancellation has reached the awaiting coroutine). This removes the fixed-delay race
    between the two cancels and ``ForeignLoopTransport.create`` completing: without the
    gate, a slow CI host could let the acquisition resolve before either cancel lands.
    """
    _patch_owned_and_policy(monkeypatch)
    created: list[_FakeTransport] = []
    cancels_delivered = threading.Event()

    def slow_create(cls: type[ForeignLoopTransport], **kwargs: Any) -> _FakeTransport:
        assert cancels_delivered.wait(timeout=10), "outer cancels were not delivered before the gate timeout"
        transport = _FakeTransport()
        created.append(transport)
        return transport

    monkeypatch.setattr(ForeignLoopTransport, "create", classmethod(slow_create))
    ensure_async_engine_loop()

    async def main() -> asyncio.CancelledError:
        task = asyncio.ensure_future(_call_frame_plan())
        await asyncio.sleep(0.05)
        task.cancel("first-cancel")
        await asyncio.sleep(0.05)  # let the first cancellation reach the awaiting coroutine
        task.cancel("second-cancel")
        await asyncio.sleep(0.05)  # let the second cancellation reach the awaiting coroutine
        cancels_delivered.set()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await task
        return excinfo.value

    error = asyncio.run(main())
    assert created, "ForeignLoopTransport.create was never invoked"
    assert created[0].closed is True
    assert error.args and error.args[0] == "first-cancel"


def test_outer_cancel_then_factory_failure_chains_first_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Caller cancelled once while pending; acquisition then fails with a non-cancel error.

    As above, the factory failure is gated on a ``threading.Event`` set only after the
    outer cancel has been delivered and observed, so the ordering is deterministic rather
    than relying on a fixed delay racing the cancel.
    """
    _patch_owned_and_policy(monkeypatch)
    cancel_delivered = threading.Event()

    def failing_slow_create(cls: type[ForeignLoopTransport], **kwargs: Any) -> ForeignLoopTransport:
        assert cancel_delivered.wait(timeout=10), "outer cancel was not delivered before the gate timeout"
        raise RuntimeError("factory boom")

    monkeypatch.setattr(ForeignLoopTransport, "create", classmethod(failing_slow_create))
    ensure_async_engine_loop()

    async def main() -> asyncio.CancelledError:
        task = asyncio.ensure_future(_call_frame_plan())
        await asyncio.sleep(0.05)
        task.cancel("first-cancel")
        await asyncio.sleep(0.05)  # let the cancellation reach the awaiting coroutine
        cancel_delivered.set()
        with pytest.raises(asyncio.CancelledError) as excinfo:
            await task
        return excinfo.value

    error = asyncio.run(main())
    assert error.args and error.args[0] == "first-cancel"
    assert isinstance(error.__cause__, RuntimeError)
    assert "factory boom" in str(error.__cause__)
