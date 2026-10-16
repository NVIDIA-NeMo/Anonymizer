# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixed worker expectations; physical transport remains on the invocation loop."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from contextvars import Context
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread, get_ident

import pytest

from anonymizer.engine.graph_sdk import _execution_delegated
from anonymizer.engine.graph_sdk.blocking import BlockingAffinity, BlockingPool
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import LocalFailure, admit_execution_plan, start_execution
from anonymizer.engine.graph_sdk.preparation import prepare
from anonymizer.engine.graph_sdk.requests import ExactUsage, StopConfirmed, StopUnknown, TextArtifactValue
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph import EffectRejected
from tests.graph_sdk.implementation_fixtures import _selection
from tests.graph_sdk.test_blocking import until
from tests.graph_sdk.test_delegated_execution import _execute, _fixture, _Transport
from tests.graph_sdk.test_delegated_readiness import _qualify, _ready_fixture

TABLE = json.loads((Path(__file__).parent / "reference/delegated_worker/manifest.json").read_text())


def test_worker_readiness_round_trip(monkeypatch):
    async def run():
        loop = asyncio.new_event_loop()
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        pool = BlockingPool(max_workers=1)
        execution, services, submit, generation, readiness, probe = _ready_fixture()
        parent = next(handle.delegated for handle in services.handles if handle.delegated)
        invocation_loop = asyncio.get_running_loop()
        locations = []
        threads = []
        host_thread = get_ident()
        transport_loops = []
        original_dispatch = _Transport.dispatch

        async def dispatch(transport, envelope):
            transport_loops.append(asyncio.get_running_loop())
            return await original_dispatch(transport, envelope)

        monkeypatch.setattr(_Transport, "dispatch", dispatch)

        async def foreign(inputs, borrowed):
            locations.append(asyncio.get_running_loop())

            # to_thread copies the issued context; run_coroutine_threadsafe copies it back.
            def copied_worker():
                threads.append(get_ident())
                return asyncio.run_coroutine_threadsafe(parent.submit(inputs, borrowed), loop).result()

            return await asyncio.to_thread(copied_worker)

        def callback(inputs, borrowed):
            threads.append(get_ident())
            return asyncio.run_coroutine_threadsafe(foreign(inputs, borrowed), loop).result()

        execution, services, adapter = _worker_services(execution, services, loop, pool, callback)
        try:
            result = await _execute(execution, services)
            expected = TABLE["positive"]
            assert submit.calls == expected["submissions"]
            assert readiness.calls == generation.calls == probe.calls == 1
            assert len(result.requests.dispatches) == expected["physical_requests"]
            assert len(result.record.terminals) == expected["terminals"]
            assert sum(row.attempt is not None for row in result.record.terminals) == expected["starts"]
            assert [row.child.role for row in result.requests.dispatches] == expected["roles"]
            assert dict(result.artifacts)[result.final_outputs[0].candidate.artifact] == TextArtifactValue(
                text=expected["candidate"]
            )
            assert len(_qualify(execution, result).qualified) == expected["qualified"]
            assert locations == [loop] and loop is not invocation_loop
            assert transport_loops == [invocation_loop, invocation_loop]
            assert len(set(threads)) == 2 and host_thread not in threads
            assert pool.occupied == adapter.local_users == 0
            assert len(result.cleanup) == 1
            assert result.cleanup[0].owner == "caller" and result.cleanup[0].disposition == "left_open"
            assert result.cleanup_associations[0].purpose == "accounting"
            assert not pool.closed and loop.is_running()

            class Control:
                cancelled = False

            assert await pool._run(lambda: 7, lambda: True, Control()) == 7

            async def still_live():
                return 11

            assert await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(still_live(), loop)) == 11
        finally:
            await pool.close()
            loop.call_soon_threadsafe(loop.stop)
            await asyncio.to_thread(thread.join)
            loop.close()

    asyncio.run(run())


@asynccontextmanager
async def _loop_and_pool():
    loop = asyncio.new_event_loop()
    ready = Event()
    loop.call_soon(ready.set)
    thread = Thread(target=loop.run_forever, daemon=True)
    thread.start()
    assert ready.wait(2)
    pool = BlockingPool(max_workers=1)
    try:
        yield loop, pool
    finally:
        await pool.close()
        loop.call_soon_threadsafe(loop.stop)
        await asyncio.to_thread(thread.join)
        loop.close()


def _worker_services(execution, services, loop, pool, callback):
    previous = execution.context.prepared
    owner = next(item for item in execution.policies if item.delegated)
    selected = owner.implementations[0].capability
    capability = replace(selected, resource_lifetime="caller_owned")
    capabilities = tuple(capability if item == selected else item for item in execution.capabilities)
    prepared = prepare(
        data=previous.data,
        workflow=previous.workflow,
        activation_limits=previous.activation_limits,
        bound_inputs=tuple(previous.bound_inputs),
        configuration=previous.configuration,
        state=previous.state,
        selections=tuple(_selection(item.node, item.capability) for item in previous.implementations),
        capabilities=capabilities,
        limits=previous.limits,
    )
    assert prepared.reservation_recipe == previous.reservation_recipe
    policies = tuple(
        replace(item, implementations=tuple(replace(impl, capability=capability) for impl in item.implementations))
        if item.node == owner.node
        else item
        for item in execution.policies
    )
    execution = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=policies,
        decisions=(),
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
    )
    adapter = _execution_delegated.WorkerDelegatedSubmitter(
        callback=callback,
        pool=pool,
        affinity=BlockingAffinity(cross_thread=True, concurrency_safe=False, close_on_event_loop=False),
        borrowed_loop=loop,
    )
    lease = ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=adapter)
    return (
        execution,
        replace(
            services,
            handles=tuple(
                replace(h, delegated=adapter, resource=lease) if h.delegated else h for h in services.handles
            ),
        ),
        adapter,
    )


@pytest.mark.parametrize("caught", [False, True])
@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("cancel_owner", [False, True])
def test_worker_cancellation_retains_live_child(monkeypatch, caught, confirmed, cancel_owner):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, generation, readiness, _ = _ready_fixture()
            started, release, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
            holder = {}
            original = _execution_delegated._DelegatedOwner.run

            async def capture(owner, handle):
                holder["owner"] = asyncio.current_task()
                holder["authority"] = owner.runtime.request_authority
                return await original(owner, handle)

            monkeypatch.setattr(_execution_delegated._DelegatedOwner, "run", capture)
            host = asyncio.get_running_loop()

            class Transport:
                async def dispatch(self, envelope):
                    assert asyncio.get_running_loop() is host
                    holder["transport"] = asyncio.current_task()
                    started.set()
                    await release.wait()
                    return await readiness.dispatch(envelope)

                async def cancel(self, request):
                    stopping.set()
                    return (
                        StopConfirmed(usage=ExactUsage(input_units=1, output_units=1)) if confirmed else StopUnknown()
                    )

            async def foreign(borrowed):
                holder["foreign"] = asyncio.current_task()
                try:
                    await borrowed.readiness()
                except asyncio.CancelledError:
                    if not caught:
                        raise
                return LocalFailure(failure="implementation_exception")

            def callback(inputs, borrowed):
                try:
                    return asyncio.run_coroutine_threadsafe(foreign(borrowed), loop).result()
                finally:
                    holder["worker_finished"] = True

            execution, services, adapter = _worker_services(execution, services, loop, pool, callback)
            services = replace(
                services,
                handles=tuple(
                    replace(h, transport=Transport()) if h.transport is readiness else h for h in services.handles
                ),
            )
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            waiter = asyncio.create_task(running.wait())
            try:
                await asyncio.wait_for(started.wait(), 2)
                if cancel_owner:
                    holder["owner"].cancel()
                else:
                    loop.call_soon_threadsafe(holder["foreign"].cancel)
                await asyncio.wait_for(stopping.wait(), 2)
                await until(lambda: bool(holder["authority"].state.terminals))
                first = holder["authority"].state.terminals
                for _ in range(2):
                    holder["owner"].cancel()
                    waiter.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await waiter
                    waiter = asyncio.create_task(running.wait())
                    await asyncio.sleep(0)
                assert not waiter.done() and not holder["transport"].done()
                release.set()
                result = await asyncio.wait_for(waiter, 2)
                assert holder["worker_finished"]
                assert holder["foreign"].done() and holder["transport"].done()
                assert not any(
                    getattr(task.get_coro(), "__name__", None) == "_worker_call"
                    for task in asyncio.all_tasks()
                    if not task.done()
                )
                assert pool.occupied == adapter.local_users == 0
                assert generation.calls == 0 and readiness.calls == 1
                assert len(result.requests.dispatches) == 1
                assert len(result.requests.terminals) == len(result.requests.settlements) == 1
                assert result.requests.terminals == first
                assert result.requests.terminals[0].category == ("cancelled" if confirmed else "lost")
                assert not result.requests.remote_outstanding and not result.requests.local_in_flight
                assert all(state.complete for state in result.states)
                assert not _qualify(execution, result).qualified
            finally:
                release.set()
                await asyncio.wait_for(running.wait(), 2)

    asyncio.run(run())


def test_worker_borrow_rejects_context_loop_expiry_and_concurrency():
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, next_submit, generation, readiness, _ = _ready_fixture()
            parent = next(h.delegated for h in services.handles if h.delegated)
            holder = {}

            async def foreign(inputs, borrowed):
                holder["borrow"] = borrowed
                # A capability object without its issued context is insufficient.
                task = asyncio.create_task(borrowed.readiness(), context=Context())
                with pytest.raises(EffectRejected):
                    await task
                # Copying the context to an undeclared loop does not authorize it.
                with pytest.raises(EffectRejected):
                    await asyncio.to_thread(lambda: asyncio.run(borrowed.readiness()))
                return await parent.submit(inputs, borrowed)

            def callback(inputs, borrowed):
                return asyncio.run_coroutine_threadsafe(foreign(inputs, borrowed), loop).result()

            execution, services, adapter = _worker_services(execution, services, loop, pool, callback)
            result = await _execute(execution, services)
            assert len(_qualify(execution, result).qualified) == 1
            assert readiness.calls == generation.calls == 1
            # No issued context survives here and the submission is expired.
            future = asyncio.run_coroutine_threadsafe(holder["borrow"].readiness(), loop)
            with pytest.raises(EffectRejected):
                await asyncio.wrap_future(future)
            assert pool.occupied == adapter.local_users == 0

    asyncio.run(run())


@pytest.mark.parametrize("bad", ["simultaneous", "foreign_owner", "expired"])
def test_worker_exact_owner_and_single_callback(bad):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, next_submit, generation, readiness, _ = _ready_fixture()
            parent = next(h.delegated for h in services.handles if h.delegated)
            host = asyncio.get_running_loop()
            holder = {}
            release = asyncio.Event()
            late_tasks: list[asyncio.Task[None]] = []

            async def foreign(inputs, borrowed):
                if bad == "simultaneous":
                    first = asyncio.create_task(borrowed.readiness())
                    while not holder.get("entered"):
                        await asyncio.sleep(0)
                    with pytest.raises(EffectRejected):
                        await borrowed.readiness()
                    host.call_soon_threadsafe(release.set)
                    await first
                    await borrowed.probe()
                    return await next_submit.submit(inputs, borrowed)
                if bad == "foreign_owner":
                    # Copying even the genuine context does not authenticate a different issued object.
                    forged = _execution_delegated._WorkerBorrow(borrowed.owner, borrowed.adapter)
                    token = _execution_delegated._WORKER_BORROW.set(forged)
                    try:
                        with pytest.raises(EffectRejected):
                            await forged.readiness()
                    finally:
                        _execution_delegated._WORKER_BORROW.reset(token)
                if bad == "expired":

                    async def late():
                        while not borrowed.expired:
                            await asyncio.sleep(0)
                        with pytest.raises(EffectRejected):
                            await borrowed.readiness()

                    late_tasks.append(asyncio.create_task(late()))
                return await parent.submit(inputs, borrowed)

            def callback(inputs, borrowed):
                return asyncio.run_coroutine_threadsafe(foreign(inputs, borrowed), loop).result()

            if bad == "simultaneous":

                class Transport:
                    async def dispatch(self, envelope):
                        assert asyncio.get_running_loop() is host
                        holder["entered"] = True
                        await release.wait()
                        return await readiness.dispatch(envelope)

                    async def cancel(self, request):
                        return StopUnknown()

                services = replace(
                    services,
                    handles=tuple(
                        replace(h, transport=Transport()) if h.transport is readiness else h for h in services.handles
                    ),
                )
            execution, services, _ = _worker_services(execution, services, loop, pool, callback)
            result = await _execute(execution, services)
            assert len(result.requests.dispatches) == 2
            assert readiness.calls == generation.calls == 1
            assert len(_qualify(execution, result).qualified) == 1
            if bad == "expired":

                async def wait_late():
                    await late_tasks[0]

                await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(wait_late(), loop))

    asyncio.run(run())


@pytest.mark.parametrize("profile", ["unregistered", "host_loop", "stopped_loop", "blocking_transport", "closed_pool"])
def test_worker_unsupported_services_reject_before_submission(profile):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, *_ = _ready_fixture()
            calls = []

            def callback(inputs, borrowed):
                calls.append(True)
                return LocalFailure(failure="permanent")

            execution, services, adapter = _worker_services(execution, services, loop, pool, callback)
            stopped = asyncio.new_event_loop()
            try:
                if profile == "unregistered":

                    class Pretender:
                        execution = "worker"

                        async def submit(self, inputs, requests):
                            calls.append(True)

                    services = replace(
                        services,
                        handles=tuple(
                            replace(h, delegated=Pretender()) if h.delegated else h for h in services.handles
                        ),
                    )
                elif profile in {"host_loop", "stopped_loop"}:
                    wrong = _execution_delegated.WorkerDelegatedSubmitter(
                        callback=callback,
                        pool=pool,
                        affinity=adapter.affinity,
                        borrowed_loop=asyncio.get_running_loop() if profile == "host_loop" else stopped,
                    )
                    services = replace(
                        services,
                        handles=tuple(
                            replace(
                                h,
                                delegated=wrong,
                                resource=ResourceLease.create(
                                    owner="caller", safe_detachment="forbidden", handle=wrong
                                ),
                            )
                            if h.delegated
                            else h
                            for h in services.handles
                        ),
                    )
                elif profile == "closed_pool":
                    await pool.close()
                else:
                    from anonymizer.engine.graph_sdk.blocking import BlockingTransport

                    transport = BlockingTransport(callback=lambda request: None, pool=pool, affinity=adapter.affinity)
                    services = replace(
                        services,
                        handles=tuple(replace(h, transport=transport) if h.transport else h for h in services.handles),
                    )
                with pytest.raises(EffectRejected):
                    await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
                assert not calls and pool.occupied == 0
            finally:
                stopped.close()

    asyncio.run(run())


def test_cancel_queued_worker_submits_nothing(monkeypatch):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            from anonymizer.engine.graph_sdk.blocking import _BlockingInvocation

            execution, services, _, generation, readiness, probe = _ready_fixture()
            entered, release = Event(), Event()
            holder = {}
            calls = []
            original = _execution_delegated._DelegatedOwner.run

            async def capture(owner, handle):
                holder["owner"] = asyncio.current_task()
                return await original(owner, handle)

            monkeypatch.setattr(_execution_delegated._DelegatedOwner, "run", capture)

            def callback(inputs, borrow):
                calls.append(True)
                return LocalFailure(failure="permanent")

            execution, services, adapter = _worker_services(execution, services, loop, pool, callback)

            # Occupy the existing bounded pool; no additional executor is introduced.
            def block():
                entered.set()
                release.wait(2)

            class Control:
                cancelled = False

            occupant = asyncio.create_task(_BlockingInvocation(adapter, block).run(lambda: True, Control()))
            await until(entered.is_set)
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            waiter = asyncio.create_task(running.wait())
            try:
                await until(lambda: "owner" in holder)
                holder["owner"].cancel()
                result = await asyncio.wait_for(waiter, 2)
                assert not calls
                assert pool.occupied == adapter.local_users == 1  # Unrelated work is still live.
                assert not result.requests.dispatches
                assert readiness.calls == generation.calls == probe.calls == 0
                assert sum(row.attempt is not None for row in result.record.terminals) == 1
                assert all(state.complete for state in result.states)
            finally:
                release.set()
                await occupant

    asyncio.run(run())


@pytest.mark.parametrize("simple", [False, True])
def test_worker_generation_uses_existing_correction_and_publication(simple):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, submit, transports = _fixture(simple=simple, parent_io=True)

            def callback(inputs, borrowed):
                return asyncio.run_coroutine_threadsafe(submit.submit(inputs, borrowed), loop).result()

            execution, services, _ = _worker_services(execution, services, loop, pool, callback)
            result = await _execute(execution, services)
            assert submit.calls == 1
            assert sum(item.calls for item in transports) == len(result.requests.dispatches) == (1 if simple else 2)
            assert [item.category for item in result.requests.terminals] == (
                ["success"] if simple else ["failure", "success"]
            )
            assert len(_qualify(execution, result).qualified) == 1
            assert dict(result.artifacts)[result.final_outputs[0].candidate.artifact] == TextArtifactValue(
                text="candidate:target-0"
            )

    asyncio.run(run())


def test_completed_borrow_wins_observer_cancellation(monkeypatch):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, _, readiness, _ = _ready_fixture()
            parent = next(h.delegated for h in services.handles if h.delegated)
            entered, release = asyncio.Event(), asyncio.Event()
            blocked, unblock = Event(), Event()
            holder = {}
            original = _execution_delegated._DelegatedOwner.run

            async def capture(owner, handle):
                holder["owner"] = owner
                return await original(owner, handle)

            monkeypatch.setattr(_execution_delegated._DelegatedOwner, "run", capture)

            class Transport:
                async def dispatch(self, envelope):
                    entered.set()
                    await release.wait()
                    return await readiness.dispatch(envelope)

                async def cancel(self, request):
                    raise AssertionError("a completed borrow requires no transport cancellation")

            async def foreign(inputs, borrowed):
                holder["observer"] = asyncio.current_task()
                return await parent.submit(inputs, borrowed)

            def callback(inputs, borrowed):
                return asyncio.run_coroutine_threadsafe(foreign(inputs, borrowed), loop).result()

            def barrier():
                blocked.set()
                unblock.wait(2)

            execution, services, _ = _worker_services(execution, services, loop, pool, callback)
            services = replace(
                services,
                handles=tuple(
                    replace(h, transport=Transport()) if h.transport is readiness else h for h in services.handles
                ),
            )
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            waiter = asyncio.create_task(running.wait())
            try:
                await asyncio.wait_for(entered.wait(), 2)
                loop.call_soon_threadsafe(barrier)
                await until(blocked.is_set)
                # Queue observer cancellation first, but complete the source future
                # before the foreign loop can deliver either notification.
                loop.call_soon_threadsafe(holder["observer"].cancel)
                release.set()
                await until(lambda: holder["owner"].worker_call is None)
                await asyncio.sleep(0)
                unblock.set()
                result = await asyncio.wait_for(waiter, 2)
                assert not holder["owner"].cancel_requested
                assert len(result.requests.dispatches) == 2
                assert all(item.category == "success" for item in result.requests.terminals)
                assert len(_qualify(execution, result).qualified) == 1
            finally:
                unblock.set()
                release.set()
                await asyncio.wait_for(running.wait(), 2)

    asyncio.run(run())


@pytest.mark.parametrize("bad", ["missing", "mismatched", "sdk", "closed", "stateless_parent"])
def test_worker_parent_requires_exact_caller_owned_lease(bad):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            base, services, *_ = _ready_fixture()
            calls = []

            def callback(inputs, borrowed):
                calls.append(True)
                return LocalFailure(failure="permanent")

            execution, services, adapter = _worker_services(base, services, loop, pool, callback)
            if bad == "missing":
                services = replace(
                    services, handles=tuple(replace(h, resource=None) if h.delegated else h for h in services.handles)
                )
            elif bad in {"mismatched", "sdk"}:
                lease = ResourceLease.create(
                    owner="sdk" if bad == "sdk" else "caller",
                    safe_detachment="forbidden",
                    handle=adapter if bad == "sdk" else pool,
                )
                services = replace(
                    services, handles=tuple(replace(h, resource=lease) if h.delegated else h for h in services.handles)
                )
            elif bad == "closed":
                await adapter.close()
            else:
                execution = base
            with pytest.raises(EffectRejected):
                await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            assert not calls and pool.occupied == 0
            assert not pool.closed and loop.is_running()

    asyncio.run(run())
