# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission observers cannot abandon an initiated registered child."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from anonymizer.engine.graph_sdk._execution_delegated import _DelegatedOwner
from anonymizer.engine.graph_sdk.executor import LocalFailure, start_execution
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    ExactUsage,
    ExternalSettlement,
    StopConfirmed,
    StopUnknown,
    TransportFailure,
)
from anonymizer.graph import EffectRejected
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_blocking import until
from tests.graph_sdk.test_delegated_execution import _execute, _fixture

TABLE = json.loads((Path(__file__).parent / "reference/delegated_execution/cancellation.json").read_text())


@pytest.mark.parametrize("caught", TABLE["live_variants"]["parent_catches"])
@pytest.mark.parametrize("confirmed, hook_failure", [(False, None), (True, None), (False, "raise"), (False, "self")])
@pytest.mark.parametrize("cancel_owner", TABLE["live_variants"]["cancel_owner"])
def test_cancelled_submitter_drains_live_child(monkeypatch, caught, confirmed, cancel_owner, hook_failure):
    async def run():
        execution, services, submit, _ = _fixture(simple=True, parent_io=True)
        holder = {}
        release, stopping = asyncio.Event(), asyncio.Event()
        original = _DelegatedOwner.run

        async def capture(owner, handle):
            holder["owner"], holder["owner_task"] = owner, asyncio.current_task()
            return await original(owner, handle)

        monkeypatch.setattr(_DelegatedOwner, "run", capture)

        class Parent:
            execution = "same_loop"

            async def submit(self, inputs, requests):
                holder["parent"] = asyncio.current_task()
                try:
                    await requests.request(submit.slots[0])
                except asyncio.CancelledError:
                    if not caught:
                        raise
                    return LocalFailure(failure="implementation_exception")
                raise AssertionError("submission cancellation must reach its observer")

        class Transport:
            calls = 0
            cancels = 0

            async def dispatch(self, envelope):
                self.calls += 1
                holder["call"] = asyncio.current_task()
                holder["parent"].cancel()
                await release.wait()
                return TransportFailure(
                    failure="permanent",
                    settlement=ExternalSettlement(
                        request=envelope.request,
                        disposition="stopped" if confirmed else "completed",
                        usage=ExactUsage(input_units=1, output_units=1),
                        remote_stopped=True,
                    ),
                )

            async def cancel(self, request):
                self.cancels += 1
                stopping.set()
                if hook_failure == "raise":
                    raise asyncio.CancelledError
                if hook_failure == "self":
                    hook_task = asyncio.current_task()
                    assert hook_task is not None
                    hook_task.cancel()
                    await asyncio.sleep(0)
                return StopConfirmed(usage=ExactUsage(input_units=1, output_units=1)) if confirmed else StopUnknown()

        transport = Transport()
        services = replace(
            services,
            handles=tuple(
                replace(handle, delegated=Parent())
                if handle.delegated
                else replace(handle, transport=transport)
                if handle.transport
                else handle
                for handle in services.handles
            ),
        )
        running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
        waiter = asyncio.create_task(running.wait())
        try:
            await asyncio.wait_for(stopping.wait(), 2)
            await until(lambda: holder["parent"].done())
            await until(lambda: bool(holder["owner"].runtime.request_authority.state.terminals))
            first = tuple(holder["owner"].runtime.request_authority.state.terminals)
            kind = "confirmed" if confirmed else "unknown"
            assert [item.category for item in first] == [TABLE["first_terminal"][kind]]
            assert not waiter.done() and not holder["call"].done()
            assert len(holder["owner"].runtime.request_authority.state.remote_outstanding) == int(not confirmed)
            for _ in range(2):
                observer = asyncio.create_task(running.wait())
                await asyncio.sleep(0)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
                if cancel_owner:
                    holder["owner_task"].cancel()
                    await asyncio.sleep(0)
                    assert not holder["owner_task"].done()
            assert not waiter.done() and not holder["call"].done()
        finally:
            release.set()
        result = await asyncio.wait_for(waiter, 2)
        expected = TABLE["live_expected"]
        assert transport.calls == expected["calls"]
        assert transport.cancels == expected["cancel_hooks"]
        assert len(result.requests.dispatches) == expected["dispatches"]
        assert tuple(result.requests.terminals) == first
        assert len(result.requests.terminals) == expected["request_terminals"]
        assert len(result.requests.local_in_flight) == expected["local_outstanding"]
        assert len(result.requests.remote_outstanding) == expected["remote_after_settlement"]
        assert any(item.disposition == TABLE["late_settlement"][kind] for item in result.requests.settlements)
        assert not result.requests.defects
        assert all(state.complete for state in result.states)
        admitted, current, submissions = _inputs(execution, result)
        qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not qualified.qualified and qualified.targets[0].withholding
        assert all(holder[key].done() for key in ("parent", "call", "owner_task"))
        child = result.requests.dispatches[0].child
        assert child is not None
        leaf = child.leaf
        assert (
            next(item.category for item in result.record.terminals if item.attempt == leaf)
            == TABLE["first_terminal"][kind]
        )
        assert not any(
            getattr(task.get_coro(), "__name__", None) == "call" for task in asyncio.all_tasks() if not task.done()
        )

    asyncio.run(run())


@pytest.mark.parametrize("caught", [False, True])
@pytest.mark.parametrize("pre_entry", [False, True])
def test_cancelled_submitter_preserves_entry_and_completion_boundaries(caught, pre_entry):
    async def run():
        execution, services, submit, transports = _fixture(simple=True)
        holder = {}
        original = transports[0].dispatch

        class Parent:
            execution = "same_loop"

            async def submit(self, inputs, requests):
                parent = asyncio.current_task()
                assert parent is not None
                holder["parent"] = parent
                if pre_entry:
                    holder["parent"].cancel()
                try:
                    await requests.request(submit.slots[0])
                except asyncio.CancelledError:
                    if not caught:
                        raise
                    return LocalFailure(failure="implementation_exception")
                raise AssertionError("expected submission cancellation")

        class Transport:
            cancels = 0

            async def dispatch(self, envelope):
                holder["parent"].cancel()
                return await original(envelope)

            async def cancel(self, request):
                self.cancels += 1
                return StopUnknown()

        transport = Transport()
        services = replace(
            services,
            handles=tuple(
                replace(handle, delegated=Parent())
                if handle.delegated
                else replace(handle, transport=transport)
                if handle.transport
                else handle
                for handle in services.handles
            ),
        )
        result = await _execute(execution, services)
        expected = TABLE["pre_entry"] if pre_entry else TABLE["completion_wins"]
        assert transports[0].calls == len(result.requests.dispatches) == expected["calls"]
        assert transport.cancels == expected["cancel_hooks"]
        assert all(state.complete for state in result.states)
        assert not result.requests.local_in_flight and not result.requests.remote_outstanding
        if not pre_entry:
            assert [item.category for item in result.requests.terminals] == [expected["first_terminal"]]
        assert not result.requests.defects

    asyncio.run(run())


def test_retention_does_not_admit_other_task_requests():
    async def run():
        execution, services, submit, transports = _fixture(simple=True)
        holder = {}

        class Parent:
            execution = "same_loop"

            async def submit(self, inputs, requests):
                holder["request"] = asyncio.create_task(requests.request(submit.slots[0]))
                return LocalFailure(failure="implementation_exception")

        services = replace(
            services,
            handles=tuple(
                replace(handle, delegated=Parent()) if handle.delegated else handle for handle in services.handles
            ),
        )
        result = await _execute(execution, services)
        with pytest.raises(EffectRejected, match=TABLE["other_task"]["rejection"]):
            await holder["request"]
        assert len(result.requests.dispatches) == transports[0].calls == 0
        assert all(state.complete for state in result.states)

    asyncio.run(run())


async def _confirmed_stop_without_settlement(outcome):
    execution, services, submit, _ = _fixture(simple=True, parent_io=True)
    holder = {}
    release, stopping = asyncio.Event(), asyncio.Event()

    class Parent:
        execution = "same_loop"

        async def submit(self, inputs, requests):
            holder["parent"] = asyncio.current_task()
            try:
                await requests.request(submit.slots[0])
            except asyncio.CancelledError:
                return LocalFailure(failure="implementation_exception")
            raise AssertionError("expected submission cancellation")

    class Transport:
        async def dispatch(self, envelope):
            holder["parent"].cancel()
            await release.wait()
            if outcome == "raise":
                raise RuntimeError("connection reset")
            return None

        async def cancel(self, request):
            stopping.set()
            return StopConfirmed(usage=ExactUsage(input_units=1, output_units=1))

    services = replace(
        services,
        handles=tuple(
            replace(handle, delegated=Parent())
            if handle.delegated
            else replace(handle, transport=Transport())
            if handle.transport
            else handle
            for handle in services.handles
        ),
    )
    running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
    waiter = asyncio.create_task(running.wait())
    await asyncio.wait_for(stopping.wait(), 2)
    release.set()
    return execution, await asyncio.wait_for(waiter, 2)


@pytest.mark.parametrize("outcome", ["raise", "none"])
def test_confirmed_stop_without_settlement_withholds(outcome):
    execution, result = asyncio.run(_confirmed_stop_without_settlement(outcome))
    assert [item.category for item in result.requests.terminals] == ["cancelled"]
    assert not result.requests.settlements and not result.requests.remote_outstanding
    assert not result.requests.defects
    admitted, current, submissions = _inputs(execution, result)
    qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not qualified.qualified
    assert qualified.record.statuses[0].qualification == "unknown"
    assert "request_accounting" in qualified.targets[0].withholding


def test_cancelled_child_still_outstanding_without_settlement_rejects():
    execution, result = asyncio.run(_confirmed_stop_without_settlement("none"))
    request = result.requests.dispatches[0].request
    admitted, current, submissions = _inputs(execution, result)
    object.__setattr__(result.requests, "remote_outstanding", frozenset({request}))
    with pytest.raises(EffectRejected, match="contradictory"):
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
