# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent entry counts with one registered graph activation, not qualification."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from threading import Event, get_ident

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport, _EntryGate
from anonymizer.engine.graph_sdk._request_driver import _RequestAuthority, drive_request
from anonymizer.engine.graph_sdk.requests import (
    AcceptResult,
    AssociationInput,
    AssociationResult,
    Dispatch,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    InvocationRequestScope,
    PhysicalRequestId,
    RequestPolicyBinding,
    Reserve,
    StopConfirmed,
    StopUnknown,
    TransportSuccess,
    initialize_requests,
)
from anonymizer.graph.activation import CloseUnstarted, ObserveTerminal, Select, Start, advance_activation
from tests.graph_sdk.test_activation import _initialized
from tests.graph_sdk.test_blocking import Control, until
from tests.graph_sdk.test_delegated_worker import _loop_and_pool
from tests.graph_sdk.test_requests import _association, _policy

TABLE = json.loads((Path(__file__).parent / "reference/foreign_entry/manifest.json").read_text())


class Harness:
    def __init__(self, loop, *, hold=False, confirmed=False, timeout_ns=1_000_000_000, stop_hook=None):
        self.state, self.seed = _initialized()
        self.state = advance_activation(state=self.state, event=Select(seeds=frozenset({self.seed})))
        self.host_thread = get_ident()
        self.loop = loop
        self.control = Control()
        scope = InvocationRequestScope(invocation=self.state.invocation)
        policy = _policy(attempts=1)
        self.association = _association(self.state.invocation, 0)
        self.authority = _RequestAuthority(initialize_requests(scope=scope, hard_limit=1, policies=frozenset({policy})))
        self.authority.bind(RequestPolicyBinding.create(association=self.association, policies=frozenset({policy})))
        self.request = PhysicalRequestId.new(scope=scope)
        self.authority.apply(
            Reserve(request=self.request, purpose="initial", associations=frozenset({self.association}), policy=policy)
        )
        operation = next(iter(self.state.workflow.workflow.nodes)).operation
        envelope = DispatchEnvelope(
            request=self.request,
            purpose="initial",
            operation=operation,
            associations=(AssociationInput(association=self.association, inputs=()),),
        )
        self.call_tasks = []
        self.counts = [0, 0]  # Actual inner entries, pure Start commits; never a request oracle.
        self.entered, self.stopping = Event(), Event()
        self.release = asyncio.Event()
        self.hold, self.confirmed = hold, confirmed
        owner = self

        class Inner:
            async def dispatch(self, request):
                assert asyncio.get_running_loop() is loop and get_ident() != owner.host_thread
                owner.call_tasks.append(asyncio.current_task())
                owner.counts[0] += 1
                owner.entered.set()
                if hold:
                    await owner.release.wait()
                return TransportSuccess(
                    results=(
                        AssociationResult(
                            association=owner.association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                        ),
                    ),
                    settlement=ExternalSettlement(
                        request=request.request,
                        disposition="completed",
                        remote_stopped=True,
                        usage=ExactUsage(input_units=1, output_units=1),
                    ),
                )

            async def cancel(self, request):
                assert asyncio.get_running_loop() is loop
                owner.stopping.set()
                if stop_hook is not None:
                    return await stop_hook(request)
                return StopConfirmed(usage=ExactUsage(input_units=1, output_units=1)) if confirmed else StopUnknown()

        self.adapter = ForeignLoopTransport(inner=Inner(), loop=loop, gate_timeout_ns=timeout_ns)
        self.foreign = self.adapter._invocation(envelope)

    def prepare(self):
        assert get_ident() == self.host_thread
        next_state = advance_activation(state=self.state, event=Start(activation=self.seed.activation))

        def commit():
            assert self.request in self.authority.state.dispatched
            self.state = next_state
            self.counts[1] += 1

        return commit

    async def run(self):
        async def forbidden():
            raise AssertionError("the registered inner transport is the only forwarding target")

        return await drive_request(
            self.authority,
            self.request,
            self.control,
            forbidden,
            self.adapter.cancel,
            foreign=self.foreign,
            prepare_dispatch=self.prepare,
        )

    def unblock(self):
        self.loop.call_soon_threadsafe(self.release.set)

    def finish(self, completion):
        if not completion.cancelled and isinstance(completion.value, TransportSuccess):
            self.authority.apply(AcceptResult(request=self.request, results=completion.value.results))
        if self.counts[1]:
            terminal = self.authority.state.terminals[0]
            self.state = advance_activation(
                state=self.state,
                event=ObserveTerminal(
                    activation=self.seed.activation,
                    outcome="ok" if terminal.category == "success" else None,
                    category=terminal.category,
                ),
            )
        else:
            self.state = advance_activation(
                state=self.state, event=CloseUnstarted(activation=self.seed.activation, category="blocked")
            )
        assert self.state.complete

    def assert_counts(self, case):
        assert [self.counts[0], len(self.authority.state.dispatches), self.counts[1]] == TABLE["counts"][case]


@pytest.mark.parametrize("cancelled", [False, True])
def test_foreign_success_and_pre_entry_control(cancelled):
    async def run():
        async with _loop_and_pool() as (loop, _):
            h = Harness(loop)
            h.control.cancelled = cancelled
            completion = await h.run()
            h.finish(completion)
            h.assert_counts("cancel_before_entry" if cancelled else "success")
            assert not h.authority.state.reserved

    asyncio.run(run())


@pytest.mark.parametrize("rejection", ["preflight", "dispatch"])
def test_foreign_rejection_cannot_commit_start(monkeypatch, rejection):
    async def run():
        async with _loop_and_pool() as (loop, _):
            h = Harness(loop)
            if rejection == "preflight":

                def reject():
                    raise EffectRejected(EffectCode.REQUEST_CAUSALITY)

                monkeypatch.setattr(h, "prepare", reject)
            else:
                original = _RequestAuthority.apply

                def apply(owner, event):
                    if owner is h.authority and isinstance(event, Dispatch):
                        raise EffectRejected(EffectCode.REQUEST_CAUSALITY)
                    return original(owner, event)

                monkeypatch.setattr(_RequestAuthority, "apply", apply)
            completion = await h.run()
            h.finish(completion)
            h.assert_counts("rejected_preflight")
            assert not h.authority.state.reserved and not h.stopping.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("observer_cancel", [False, True])
def test_delayed_gate_timeout_invalidates_queued_callback(monkeypatch, observer_cancel):
    async def run():
        async with _loop_and_pool() as (loop, _):
            h = Harness(loop, timeout_ns=1_000_000_000 if observer_cancel else 10_000_000)
            host = asyncio.get_running_loop()
            schedule = host.call_soon_threadsafe
            delayed = []

            def queue(callback, *args, **kwargs):
                if getattr(callback, "__self__", None) is h.foreign._gate:
                    delayed.append((callback, args))
                    return None
                return schedule(callback, *args, **kwargs)

            monkeypatch.setattr(host, "call_soon_threadsafe", queue)
            task = asyncio.create_task(h.run())
            if observer_cancel:
                await until(lambda: bool(delayed))
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
            heartbeats = 0
            while not task.done():
                heartbeats += 1
                await asyncio.sleep(0)
            completion = task.result()
            assert delayed and (observer_cancel or heartbeats > 1)
            for callback, args in delayed:
                callback(*args)
            h.finish(completion)
            h.assert_counts("queued_gate_timeout")
            assert not h.stopping.is_set() and not h.authority.state.reserved

    asyncio.run(run())


def test_timeout_losing_to_acceptance_still_forwards(monkeypatch):
    async def run():
        async with _loop_and_pool() as (loop, _):
            h = Harness(loop)
            result = h.foreign._gate._ack.result

            def timed_out_after_acceptance(timeout=None):
                value = result(timeout)
                if timeout is not None:
                    raise TimeoutError
                return value

            monkeypatch.setattr(h.foreign._gate._ack, "result", timed_out_after_acceptance)
            completion = await h.run()
            h.finish(completion)
            h.assert_counts("acceptance_wins_timeout")

    asyncio.run(run())


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("boundary", ["after_accept_before_inner", "after_inner"])
def test_retained_forwarding_and_late_settlement(monkeypatch, confirmed, boundary):
    async def run():
        async with _loop_and_pool() as (loop, _):
            h = Harness(loop, hold=True, confirmed=confirmed)
            reached, proceed = Event(), Event()
            original = _EntryGate.wait

            def barrier(gate, timeout):
                value = original(gate, timeout)
                if gate is h.foreign._gate and value and boundary == "after_accept_before_inner":
                    reached.set()
                    assert proceed.wait(3)
                return value

            monkeypatch.setattr(_EntryGate, "wait", barrier)
            task = asyncio.create_task(h.run())
            try:
                await until(reached.is_set if boundary == "after_accept_before_inner" else h.entered.is_set)
                if boundary == "after_accept_before_inner":
                    assert h.counts == [0, 1] and len(h.authority.state.dispatches) == 1
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
                assert not task.done()
                proceed.set()
                await until(h.entered.is_set)
                await until(h.stopping.is_set)
                await until(lambda: bool(h.authority.state.terminals))
                first = h.authority.state.terminals
                assert first[0].category == ("cancelled" if confirmed else "lost")
                assert not task.done()
                h.unblock()
                completion = await asyncio.wait_for(task, 3)
                assert h.authority.state.terminals == first
                assert len(h.authority.state.settlements) == 1
                assert not h.authority.state.remote_outstanding
                h.finish(completion)
                h.assert_counts("cancel_" + boundary)
            finally:
                proceed.set()
                h.unblock()
                if not task.done():
                    await asyncio.wait_for(task, 3)

    asyncio.run(run())


def test_foreign_graph_service_admission_stays_closed(monkeypatch):
    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.graph._values import InvocationId
    from tests.graph_sdk.test_delegated_execution import _fixture

    async def run():
        async with _loop_and_pool() as (loop, _):
            execution, services, _, _ = _fixture(simple=True)
            handle = next(item for item in services.handles if item.transport is not None)
            wrapped = ForeignLoopTransport(inner=handle.transport, loop=loop, gate_timeout_ns=1_000_000)
            services = replace(
                services,
                handles=tuple(
                    replace(item, transport=wrapped) if item is handle else item for item in services.handles
                ),
            )

            def forbidden(*args, **kwargs):
                raise AssertionError("InvocationId must not be created for the closed X2 service")

            monkeypatch.setattr(InvocationId, "new", forbidden)
            with pytest.raises(EffectRejected):
                await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)

    asyncio.run(run())


@pytest.mark.parametrize("boundary", ["before_entry", "after_entry", "after_accept"])
def test_borrowed_loop_stop_requires_same_loop_resume(monkeypatch, boundary):
    from threading import Thread

    async def run():
        async with _loop_and_pool() as (loop, _):
            h = Harness(loop, hold=True)
            resumed = None
            schedule = asyncio.run_coroutine_threadsafe
            if boundary == "before_entry":

                def stop_before(coro, target):
                    if getattr(coro, "__name__", None) == "enter":
                        loop.call_soon_threadsafe(loop.stop)
                    return schedule(coro, target)

                monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", stop_before)
            if boundary == "after_accept":
                original = _EntryGate.wait

                def stop_after_accept(gate, timeout):
                    accepted = original(gate, timeout)
                    if gate is h.foreign._gate and accepted:
                        loop.call_soon(loop.stop)
                    return accepted

                monkeypatch.setattr(_EntryGate, "wait", stop_after_accept)
            task = asyncio.create_task(h.run())
            try:
                if boundary == "after_entry":
                    await until(h.entered.is_set)
                    loop.call_soon_threadsafe(loop.stop)
                await until(lambda: not loop.is_running())
                for _ in range(2):
                    task.cancel()
                    await asyncio.sleep(0)
                assert not task.done()
                h.assert_counts(
                    "loop_stopped_before_entry" if boundary == "before_entry" else "loop_stopped_after_entry"
                )
                # A stopped loop is not a confirmed stop or a complete drain.
                if boundary != "before_entry":
                    assert h.authority.state.remote_outstanding
                resumed = Thread(target=loop.run_forever, daemon=True)
                resumed.start()
                if boundary != "before_entry":
                    await until(h.stopping.is_set)
                    await until(lambda: bool(h.authority.state.terminals))
                    assert h.authority.state.terminals[0].category == "lost"
                    assert not task.done()
                h.unblock()
                completion = await asyncio.wait_for(task, 3)
                h.finish(completion)
                h.assert_counts(
                    "loop_stopped_before_entry" if boundary == "before_entry" else "loop_stopped_after_entry"
                )
            finally:
                if not loop.is_running():
                    resumed = Thread(target=loop.run_forever, daemon=True)
                    resumed.start()
                h.unblock()
                if not task.done():
                    await asyncio.wait_for(task, 3)
                loop.call_soon_threadsafe(loop.stop)
                if resumed is not None:
                    await asyncio.to_thread(resumed.join)

    asyncio.run(run())


def test_cancelled_foreign_observers_retain_host_driver():
    async def run():
        async with _loop_and_pool() as (loop, _):
            h = Harness(loop, hold=True)
            host = asyncio.get_running_loop()
            task = asyncio.create_task(h.run())
            observed = []

            async def host_observation():
                return await asyncio.shield(task)

            async def foreign_observer(ready):
                future = asyncio.run_coroutine_threadsafe(host_observation(), host)
                observed.append(future)
                ready.set()
                try:
                    return await asyncio.shield(asyncio.wrap_future(future))
                except asyncio.CancelledError:
                    host.call_soon_threadsafe(task.cancel)
                    raise

            try:
                await until(h.entered.is_set)
                for _ in range(2):
                    ready = Event()
                    watcher = asyncio.run_coroutine_threadsafe(foreign_observer(ready), loop)
                    await until(ready.is_set)
                    watcher.cancel()
                    await asyncio.sleep(0)
                await until(h.stopping.is_set)
                await until(lambda: bool(h.authority.state.terminals))
                first = h.authority.state.terminals
                assert first[0].category == "lost" and not task.done()
                h.unblock()
                completion = await asyncio.wait_for(task, 3)
                await until(lambda: all(future.done() for future in observed))
                assert h.authority.state.terminals == first
                h.finish(completion)
                h.assert_counts("cancel_after_inner")
            finally:
                h.unblock()
                if not task.done():
                    await asyncio.wait_for(task, 3)

    asyncio.run(run())


@pytest.mark.parametrize("profile", ["same_loop", "stopped_loop", "closed_loop"])
def test_unavailable_loop_profiles_have_no_effect(profile):
    async def run():
        loop = asyncio.get_running_loop() if profile == "same_loop" else asyncio.new_event_loop()
        if profile == "closed_loop":
            loop.close()
        try:
            h = Harness(loop)
            completion = await h.run()
            h.finish(completion)
            h.assert_counts("cancel_before_entry")
            assert not h.stopping.is_set() and not h.authority.state.reserved
        finally:
            if profile == "stopped_loop":
                loop.close()

    asyncio.run(run())


def test_bound_transport_and_cancel_are_not_replaced_after_issue():
    async def run():
        async with _loop_and_pool() as (loop, _):
            h = Harness(loop, hold=True)

            class Replacement:
                async def dispatch(self, request):
                    raise AssertionError("replaced dispatch")

                async def cancel(self, request):
                    raise AssertionError("replaced cancellation")

            h.adapter._inner = Replacement()
            task = asyncio.create_task(h.run())
            await until(h.entered.is_set)
            task.cancel()
            await until(h.stopping.is_set)
            await until(lambda: bool(h.authority.state.terminals))
            assert h.authority.state.terminals[0].category == "lost"
            h.unblock()
            h.finish(await task)
            h.assert_counts("cancel_after_inner")

    asyncio.run(run())
