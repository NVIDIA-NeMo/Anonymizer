# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stop-hook completion cannot abandon an already initiated physical call."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from anonymizer.engine.graph_sdk._request_driver import drive_request
from anonymizer.engine.graph_sdk.requests import ExactUsage, ExternalSettlement, StopUnknown, TransportSuccess
from tests.graph_sdk.test_blocking import until
from tests.graph_sdk.test_delegated_worker import _loop_and_pool
from tests.graph_sdk.test_foreign_entry import Harness

TABLE = json.loads((Path(__file__).parent / "reference/foreign_entry/stop_hook.json").read_text())


@pytest.mark.parametrize("path", TABLE["paths"])
@pytest.mark.parametrize("hook_result", TABLE["hook_results"])
def test_completed_stop_hook_retains_original_call(path, hook_result):
    async def run():
        async with _loop_and_pool() as (loop, _):
            held = {}

            async def stop(request):
                held["hook"] = asyncio.current_task()
                if hook_result == "raise_cancelled":
                    raise asyncio.CancelledError
                if hook_result == "self_cancel":
                    hook_task = asyncio.current_task()
                    assert hook_task is not None
                    hook_task.cancel()
                    await asyncio.sleep(0)
                    raise AssertionError("self-cancellation must interrupt this hook")
                if hook_result == "runtime_error":
                    raise RuntimeError("stop unconfirmed")
                return StopUnknown()

            h = Harness(loop, hold=True, stop_hook=stop)
            release = asyncio.Event()

            async def invoke():
                held["call"] = asyncio.current_task()
                h.counts[0] += 1
                h.entered.set()
                await release.wait()
                return TransportSuccess(
                    results=(),
                    settlement=ExternalSettlement(
                        request=h.request,
                        disposition="completed",
                        remote_stopped=True,
                        usage=ExactUsage(input_units=1, output_units=1),
                    ),
                )

            task = asyncio.create_task(
                drive_request(
                    h.authority, h.request, h.control, invoke, stop, prepare_dispatch=h.prepare, drain_on_cancel=True
                )
                if path == "ordinary_control"
                else h.run()
            )
            try:
                await until(h.entered.is_set)
                if path == "foreign_observer":
                    task.cancel()
                else:
                    h.control.cancelled = True
                await until(lambda: bool(h.authority.state.terminals) or task.done())
                before = TABLE["before_release"]
                assert not task.done(), "a completed stop hook must not abandon the live call"
                calls = h.call_tasks if path != "ordinary_control" else [held["call"]]
                assert len(calls) == 1 and not calls[0].done()
                assert [h.counts[0], len(h.authority.state.dispatches), h.counts[1]] == [
                    before["inner_entries"],
                    before["dispatches"],
                    before["start_commits"],
                ]
                first = h.authority.state.terminals
                assert first[0].category == before["first_terminal"]
                assert len(h.authority.state.settlements) == before["settlements"]
                if path != "ordinary_control":
                    for _ in range(TABLE["bounds"]["observer_cancellations"]):
                        task.cancel()
                        await asyncio.sleep(0)
                        assert not task.done()
                release.set()
                h.unblock()
                completion = await asyncio.wait_for(task, 3)
                h.finish(completion)
                assert all(call.done() for call in calls)
                assert h.authority.state.terminals == first
                assert len(h.authority.state.settlements) == TABLE["after_release"]["settlements"]
                assert not h.authority.state.remote_outstanding
                assert held["hook"].done()
                if "call" in held:
                    assert held["call"].done()
            finally:
                release.set()
                h.unblock()
                await asyncio.gather(task, return_exceptions=True)
                await until(lambda: all(call.done() for call in h.call_tasks))
                if "call" in held:
                    await until(held["call"].done)

    asyncio.run(run())


@pytest.mark.parametrize("foreign", [False, True])
def test_observer_cancellation_does_not_abandon_active_stop_hook(foreign):
    async def run():
        async with _loop_and_pool() as (loop, _):
            held = {}
            hook_release, call_release = asyncio.Event(), asyncio.Event()

            async def stop(request):
                held["hook"] = asyncio.current_task()
                await hook_release.wait()
                return StopUnknown()

            h = Harness(loop, hold=True, stop_hook=stop)

            async def invoke():
                held["call"] = asyncio.current_task()
                h.counts[0] += 1
                h.entered.set()
                await call_release.wait()
                return TransportSuccess(
                    results=(),
                    settlement=ExternalSettlement(
                        request=h.request,
                        disposition="completed",
                        remote_stopped=True,
                        usage=ExactUsage(input_units=1, output_units=1),
                    ),
                )

            task = asyncio.create_task(
                h.run()
                if foreign
                else drive_request(
                    h.authority,
                    h.request,
                    h.control,
                    invoke,
                    stop,
                    prepare_dispatch=h.prepare,
                    drain_on_cancel=True,
                )
            )
            try:
                await until(h.entered.is_set)
                h.control.cancelled = True
                await until(lambda: "hook" in held)
                for _ in range(TABLE["bounds"]["observer_cancellations"]):
                    task.cancel()
                    await asyncio.sleep(0)
                    assert not task.done() and not held["hook"].done()
                assert not h.authority.state.terminals
                calls = h.call_tasks if foreign else [held["call"]]
                assert len(calls) == 1 and not calls[0].done()
                (loop.call_soon_threadsafe(hook_release.set) if foreign else hook_release.set())
                await until(lambda: bool(h.authority.state.terminals))
                first = h.authority.state.terminals
                assert first[0].category == "lost" and not task.done()
                h.unblock()
                call_release.set()
                h.finish(await asyncio.wait_for(task, 3))
                assert all(call.done() for call in calls)
                assert h.authority.state.terminals == first
                assert len(h.authority.state.settlements) == 1
                assert held["hook"].done()
            finally:
                loop.call_soon_threadsafe(hook_release.set) if foreign else hook_release.set()
                h.unblock()
                call_release.set()
                await asyncio.gather(task, return_exceptions=True)
                await until(lambda: all(call.done() for call in h.call_tasks))
                if "call" in held:
                    await until(held["call"].done)

    asyncio.run(run())
