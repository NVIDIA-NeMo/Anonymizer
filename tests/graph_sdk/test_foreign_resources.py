# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public foreign worker execution and actual client lifetime witnesses."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from threading import Event

import pytest

from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import admit_execution_plan, start_execution
from anonymizer.engine.graph_sdk.preparation import prepare
from anonymizer.engine.graph_sdk.requests import ExactUsage, StopConfirmed, StopUnknown, TextArtifactValue
from tests.graph_sdk.implementation_fixtures import _selection
from tests.graph_sdk.test_blocking import until
from tests.graph_sdk.test_delegated_execution import _execute
from tests.graph_sdk.test_delegated_readiness import _qualify, _ready_fixture
from tests.graph_sdk.test_delegated_worker import _loop_and_pool, _worker_services

TABLE = json.loads((Path(__file__).parent / "reference/foreign_resources/manifest.json").read_text())


async def fixture(
    loop,
    pool,
    *,
    owner="sdk",
    base=None,
    held_transport=None,
    close_result="success",
    hold=False,
    hold_stop=False,
    stop_self_cancel=False,
    confirmed=False,
    lifetime=None,
    detachment="forbidden",
    client_factory=None,
):
    if base is None:
        execution, services, submit, generation, _, _ = _ready_fixture()
    else:
        execution, services, submit = base
        generation = held_transport
    parent = next(h.delegated for h in services.handles if h.delegated)

    async def foreign(inputs, borrowed):
        return await asyncio.to_thread(
            lambda: asyncio.run_coroutine_threadsafe(parent.submit(inputs, borrowed), loop).result()
        )

    def callback(inputs, borrowed):
        return asyncio.run_coroutine_threadsafe(foreign(inputs, borrowed), loop).result()

    execution, services, worker = _worker_services(execution, services, loop, pool, callback)
    clients, adapters = [], {}

    class Client:
        def __init__(self, transport):
            self.loop = asyncio.get_running_loop()
            assert self.loop is loop
            self.transport = transport
            self.closes = 0
            self.close_result = close_result
            self.closed = False
            self.entered, self.stopping = Event(), Event()
            self.release = asyncio.Event()
            self.close_entered = Event()
            self.close_release = asyncio.Event()
            self.stop_release = asyncio.Event()
            self.hold_close = False
            self.calls = []
            self.pings = 0
            clients.append(self)

        async def ping(self):
            assert asyncio.get_running_loop() is self.loop and not self.closed
            self.pings += 1
            return self.pings

        async def dispatch(self, request):
            assert asyncio.get_running_loop() is self.loop and not self.closed
            self.calls.append(asyncio.current_task())
            self.entered.set()
            if hold and self.transport is generation:
                await self.release.wait()
            return await self.transport.dispatch(request)

        async def cancel(self, request):
            assert asyncio.get_running_loop() is self.loop
            self.stopping.set()
            if hold_stop:
                await self.stop_release.wait()
            if stop_self_cancel == "task":
                task = asyncio.current_task()
                assert task is not None
                task.cancel()
                await asyncio.sleep(0)
            if stop_self_cancel:
                raise asyncio.CancelledError
            return StopConfirmed(usage=ExactUsage(input_units=1, output_units=1)) if confirmed else StopUnknown()

        async def close(self):
            assert asyncio.get_running_loop() is self.loop
            assert worker.local_users == 0 and all(task.done() for task in self.calls)
            assert all(a.local_users == 0 for a in adapters.values())
            self.closes += 1
            self.close_entered.set()
            if self.hold_close:
                await self.close_release.wait()
            if self.close_result == "exception":
                raise RuntimeError("close failed")
            if self.close_result == "self_cancel":
                raise asyncio.CancelledError
            self.closed = True

    async def construct(transport):
        def create():
            if client_factory is None:
                return Client(transport)
            client = client_factory(transport)
            clients.append(client)
            return client

        return ForeignLoopTransport.create(factory=create, owner=owner, gate_timeout_ns=1_000_000_000, exclusive=True)

    handles, registered = [], {}
    for h in services.handles:
        if h.transport is None:
            handles.append(h)
            continue
        adapter = registered.get(id(h.transport))
        if adapter is None:
            adapter = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(construct(h.transport), loop))
            registered[id(h.transport)] = adapter
        adapters[h.implementation] = adapter
        handles.append(replace(h, transport=adapter, resource=adapter.lease))
    previous = execution.context.prepared
    capabilities = tuple(
        replace(c, resource_lifetime=lifetime or ("executor_owned" if owner == "sdk" else "caller_owned"))
        if c.implementation in adapters
        else c
        for c in execution.capabilities
    )
    by_impl = {c.implementation: c for c in capabilities}
    prepared = prepare(
        data=previous.data,
        workflow=previous.workflow,
        activation_limits=previous.activation_limits,
        bound_inputs=tuple(previous.bound_inputs),
        configuration=previous.configuration,
        state=previous.state,
        selections=tuple(_selection(i.node, i.capability) for i in previous.implementations),
        capabilities=capabilities,
        limits=previous.limits,
    )
    execution = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=tuple(
            replace(
                p,
                implementations=tuple(replace(i, capability=by_impl[i.implementation]) for i in p.implementations),
                safe_detachment=detachment
                if any(i.implementation in adapters for i in p.implementations)
                else p.safe_detachment,
            )
            for p in execution.policies
        ),
        decisions=(),
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
    )
    return execution, replace(services, handles=tuple(handles)), submit, clients, worker, adapters


@pytest.mark.parametrize("owner", ["sdk", "caller"])
def test_public_foreign_worker_protected_and_cleanup(owner):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, submit, clients, worker, adapters = await fixture(loop, pool, owner=owner)
            result = await _execute(execution, services)
            expected = TABLE["positive"]
            assert submit.calls == expected["submissions"]
            assert sum(len(c.calls) for c in clients) == len(result.requests.dispatches) == expected["sends"]
            assert len(result.record.terminals) == expected["terminals"]
            assert sum(t.attempt is not None for t in result.record.terminals) == expected["starts"]
            assert dict(result.artifacts)[result.final_outputs[0].candidate.artifact] == TextArtifactValue(
                text=expected["candidate"]
            )
            assert len(_qualify(execution, result).qualified) == expected["qualified"]
            assert sum(c.closes for c in clients) == expected[owner + "_closes"]
            assert all(a.local_users == 0 for a in adapters.values()) and worker.local_users == 0
            assert {f.disposition for f in result.cleanup if f.owner == "sdk"} == (
                {"closed"} if owner == "sdk" else set()
            )
            assert not pool.closed and loop.is_running()
            if owner == "caller":
                assert all(not c.closed for c in clients)

                async def usable():
                    return [await c.ping() == 1 for c in clients]

                assert all(await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(usable(), loop)))

    asyncio.run(run())


@pytest.mark.parametrize("close_result", ["exception", "self_cancel"])
def test_close_failure_withholds_qualification(close_result):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, clients, _, adapters = await fixture(loop, pool, close_result=close_result)
            result = await _execute(execution, services)
            assert all(c.closes == 1 and not c.closed for c in clients)
            assert {f.disposition for f in result.cleanup if f.owner == "sdk"} == {TABLE["cleanup"][close_result]}
            assert not _qualify(execution, result).qualified
            assert all(s.complete for s in result.states)
            assert all(a.local_users == 0 for a in adapters.values())

    asyncio.run(run())


def test_close_is_retained_once_through_repeated_observers():
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, clients, worker, adapters = await fixture(loop, pool)
            for client in clients:
                client.hold_close = True
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            try:
                await until(lambda: any(c.close_entered.is_set() for c in clients))
                assert worker.local_users == 0 and all(a.local_users == 0 for a in adapters.values())
                for _ in range(2):
                    running._task.cancel()
                    observer = asyncio.create_task(running.wait())
                    await asyncio.sleep(0)
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
                    assert not running._task.done()
                for c in clients:
                    loop.call_soon_threadsafe(c.close_release.set)
                result = await asyncio.wait_for(running.wait(), 3)
                assert len(_qualify(execution, result).qualified) == 1
                assert all(c.closes == 1 and c.closed for c in clients)
                for adapter in adapters.values():
                    await adapter.close()
                assert all(c.closes == 1 for c in clients)
            finally:
                for c in clients:
                    loop.call_soon_threadsafe(c.close_release.set)
                await running.wait()

    asyncio.run(run())


@pytest.mark.parametrize("stop_self_cancel, confirmed", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("pause_loop", [False, True])
def test_foreign_public_cancel_drains_before_close(monkeypatch, stop_self_cancel, confirmed, pause_loop):
    from threading import Thread

    from anonymizer.engine.graph_sdk._execution_delegated import _DelegatedOwner

    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, clients, worker, adapters = await fixture(
                loop, pool, hold=True, stop_self_cancel=stop_self_cancel, confirmed=confirmed
            )
            holder = {}
            original = _DelegatedOwner.run

            async def capture(owner, handle):
                holder["owner"] = owner
                return await original(owner, handle)

            monkeypatch.setattr(_DelegatedOwner, "run", capture)
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            resumed = None
            try:
                await until(lambda: sum(len(c.calls) for c in clients) == 2)
                if pause_loop:
                    loop.call_soon_threadsafe(loop.stop)
                    await until(lambda: not loop.is_running())
                running.request_cancel()
                for _ in range(2):
                    observer = asyncio.create_task(running.wait())
                    await asyncio.sleep(0)
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
                assert not running._task.done() and sum(c.closes for c in clients) == 0
                if pause_loop:
                    resumed = Thread(target=loop.run_forever, daemon=True)
                    resumed.start()
                authority = holder["owner"].runtime.request_authority
                await until(lambda: len(authority.state.terminals) == 2)
                first = authority.state.terminals
                assert first[-1].category == ("cancelled" if confirmed else TABLE["cancel"]["first_terminal"])
                from anonymizer.engine.graph_sdk.resources import close_resource

                live = next(a for a in adapters.values() if a.local_users)
                assert not authority.state.local_in_flight
                if confirmed:
                    assert not authority.state.remote_outstanding
                assert (await close_resource(live.lease)).disposition == "left_open"
                assert not running._task.done() and sum(c.closes for c in clients) == 0
                assert worker.local_users > 0 and sum(a.local_users for a in adapters.values()) > 0
                for client in clients:
                    loop.call_soon_threadsafe(client.release.set)
                result = await asyncio.wait_for(running.wait(), 3)
                assert result.requests.terminals == first
                assert len(result.requests.dispatches) == len(result.requests.settlements) == 2
                assert not result.requests.remote_outstanding and not result.requests.local_in_flight
                assert all(state.complete for state in result.states)
                assert all(c.closes == 1 and c.closed for c in clients)
                assert all(a.local_users == 0 for a in adapters.values()) and worker.local_users == 0
                assert not _qualify(execution, result).qualified
            finally:
                if not loop.is_running():
                    resumed = Thread(target=loop.run_forever, daemon=True)
                    resumed.start()
                for client in clients:
                    loop.call_soon_threadsafe(client.release.set)
                await running.wait()
                if resumed is not None:
                    loop.call_soon_threadsafe(loop.stop)
                    await asyncio.to_thread(resumed.join)

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad",
    [
        "unregistered",
        "wrong_loop",
        "wrong_client",
        "missing_lease",
        "foreign_lease",
        "wrong_owner",
        "closed",
        "stateless",
        "independent_detachment",
    ],
)
@pytest.mark.parametrize("optional", [False, True])
def test_foreign_resource_rejection_precedes_invocation(monkeypatch, bad, optional):
    from anonymizer.engine.graph_sdk.resources import ResourceLease
    from anonymizer.graph import EffectRejected
    from anonymizer.graph._values import InvocationId

    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, submit, clients, _, adapters = await fixture(
                loop,
                pool,
                owner="caller" if bad == "wrong_owner" else "sdk",
                base=_optional_base() if optional else None,
                lifetime="executor_owned" if bad == "wrong_owner" else "stateless" if bad == "stateless" else None,
                detachment="independent_after_dispatch" if bad == "independent_detachment" else "forbidden",
            )
            h = next(h for h in services.handles if h.transport is not None)
            adapter = h.transport
            if bad == "unregistered":
                h = replace(
                    h, transport=ForeignLoopTransport(inner=adapter._inner, loop=loop, gate_timeout_ns=1_000_000)
                )
            elif bad == "wrong_loop":
                adapter._loop = asyncio.get_running_loop()
            elif bad == "wrong_client":
                adapter._inner = next(a._inner for a in adapters.values() if a is not adapter)
            elif bad == "missing_lease":
                h = replace(h, resource=None)
            elif bad == "foreign_lease":
                h = replace(h, resource=ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=adapter))
            elif bad == "closed":
                await adapter.close()
            services = replace(
                services,
                handles=tuple(h if old.implementation == h.implementation else old for old in services.handles),
            )

            def forbidden(*args, **kwargs):
                raise AssertionError("rejected resource profile allocated InvocationId")

            monkeypatch.setattr(InvocationId, "new", forbidden)
            try:
                with pytest.raises(EffectRejected):
                    await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
                assert submit.calls == 0 and all(not c.calls for c in clients)
                assert all(not a._claimed for a in adapters.values())
            finally:
                adapter._loop = loop
                for a in adapters.values():
                    await a.close()

    asyncio.run(run())


@pytest.mark.parametrize("optional", [False, True])
def test_claim_is_atomic_deduplicated_and_concurrent_admission_is_single_use(monkeypatch, optional):
    from anonymizer.graph import EffectRejected
    from anonymizer.graph._values import InvocationId

    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, submit, clients, _, adapters = await fixture(
                loop, pool, owner="caller", base=_optional_base() if optional else None
            )
            free, occupied, *_ = tuple(adapters.values())
            ForeignLoopTransport.claim_many((occupied, occupied))
            with pytest.raises(EffectRejected):
                ForeignLoopTransport.claim_many((free, occupied))
            assert not free._claimed and occupied._claimed

            async def alias_client():
                return ForeignLoopTransport.create(
                    factory=lambda: free._inner, owner="caller", gate_timeout_ns=1_000_000, exclusive=True
                )

            alias = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(alias_client(), loop))
            with pytest.raises(EffectRejected):
                ForeignLoopTransport.claim_many((free, alias))
            assert not free._claimed and not alias._claimed
            for a in adapters.values():
                await a.close()
            execution, services, submit, clients, _, adapters = await fixture(
                loop, pool, owner="caller", base=_optional_base() if optional else None
            )
            outcomes = await asyncio.gather(
                *(
                    start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
                    for _ in range(2)
                ),
                return_exceptions=True,
            )
            assert sum(isinstance(x, EffectRejected) for x in outcomes) == 1
            running = next(x for x in outcomes if not isinstance(x, BaseException))
            result = await running.wait()
            assert len(_qualify(execution, result).qualified) == 1 and submit.calls == 1
            assert sum(len(c.calls) for c in clients) == 2

            def forbidden(*args, **kwargs):
                raise AssertionError("reused resource allocated InvocationId")

            monkeypatch.setattr(InvocationId, "new", forbidden)
            with pytest.raises(EffectRejected):
                await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            assert all(c.closes == 0 and not c.closed for c in clients)
            for a in adapters.values():
                await a.close()

    asyncio.run(run())


def test_pre_accept_cancellation_has_no_physical_start_or_charge(monkeypatch):
    from anonymizer.engine.graph_sdk._foreign_transport import _EntryGate

    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, clients, _, adapters = await fixture(loop, pool)
            host = asyncio.get_running_loop()
            schedule = host.call_soon_threadsafe
            delayed = []

            def queue(callback, *args, **kwargs):
                if isinstance(getattr(callback, "__self__", None), _EntryGate):
                    delayed.append((callback, args))
                    return None
                return schedule(callback, *args, **kwargs)

            monkeypatch.setattr(host, "call_soon_threadsafe", queue)
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            await until(lambda: bool(delayed))
            running.request_cancel()
            result = await asyncio.wait_for(running.wait(), 3)
            for callback, args in delayed:
                callback(*args)
            assert not result.requests.dispatches and all(not c.calls for c in clients)
            leaves = {
                execution.context.prepared.reservation_recipe[index].template
                for p in execution.policies
                if p.delegated
                for index in (*p.delegated.child_slots, p.delegated.readiness.request_slot)
            }
            leaf_keys = {
                entry.activation for state in result.states for entry in state.entries if entry.template in leaves
            }
            assert leaf_keys and all(t.attempt is None for t in result.record.terminals if t.activation in leaf_keys)
            assert all(c.closes == 1 for c in clients) and all(a.local_users == 0 for a in adapters.values())
            assert all(s.complete for s in result.states) and not _qualify(execution, result).qualified

    asyncio.run(run())


@pytest.mark.parametrize("optional", [False, True])
def test_caller_pool_unrelated_work_is_not_drained(optional):
    from anonymizer.engine.graph_sdk.blocking import BlockingPool

    async def run():
        async with _loop_and_pool() as (loop, _):
            pool = BlockingPool(max_workers=2)
            release, entered = Event(), Event()

            def unrelated():
                entered.set()
                assert release.wait(5)
                return 7

            class Control:
                cancelled = False

            pending = asyncio.create_task(pool._run(unrelated, lambda: True, Control()))
            try:
                await until(entered.is_set)
                execution, services, _, clients, worker, adapters = await fixture(
                    loop, pool, owner="caller", base=_optional_base() if optional else None
                )
                result = await _execute(execution, services)
                assert len(_qualify(execution, result).qualified) == 1
                assert not pending.done() and pool.occupied == 1 and worker.local_users == 0
                assert all(a.local_users == 0 for a in adapters.values())
                assert all(c.closes == 0 for c in clients)
                release.set()
                assert await pending == 7
                assert await pool._run(lambda: 9, lambda: True, Control()) == 9
            finally:
                release.set()
                await pending
                await pool.close()

    asyncio.run(run())


def test_unqualified_factory_profile_has_no_creation_effect():
    from anonymizer.graph import EffectRejected

    async def run():
        def forbidden():
            raise AssertionError("unsupported client sharing cannot invoke factory")

        with pytest.raises(EffectRejected):
            ForeignLoopTransport.create(factory=forbidden, owner="sdk", gate_timeout_ns=1, exclusive=False)

    asyncio.run(run())


def test_live_stop_hook_is_a_resource_user_after_send_completes():
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, clients, worker, adapters = await fixture(loop, pool, hold=True, hold_stop=True)
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            try:
                await until(lambda: sum(len(c.calls) for c in clients) == 2)
                running.request_cancel()
                await until(lambda: any(c.stopping.is_set() for c in clients))
                for c in clients:
                    loop.call_soon_threadsafe(c.release.set)
                await until(lambda: all(task.done() for c in clients for task in c.calls))
                assert not running._task.done() and worker.local_users > 0
                assert sum(a.local_users for a in adapters.values()) > 0
                assert sum(c.closes for c in clients) == 0
                for _ in range(2):
                    observer = asyncio.create_task(running.wait())
                    await asyncio.sleep(0)
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
                    assert not running._task.done()
                for c in clients:
                    loop.call_soon_threadsafe(c.stop_release.set)
                result = await asyncio.wait_for(running.wait(), 3)
                assert all(c.closes == 1 for c in clients)
                assert all(a.local_users == 0 for a in adapters.values())
                assert len(result.requests.settlements) == 2 and all(s.complete for s in result.states)
            finally:
                for c in clients:
                    loop.call_soon_threadsafe(c.stop_release.set)
                    loop.call_soon_threadsafe(c.release.set)
                await running.wait()

    asyncio.run(run())


def _optional_base():
    from tests.graph_sdk.test_optional_tail import optional_fixture

    return optional_fixture(readiness="available")
