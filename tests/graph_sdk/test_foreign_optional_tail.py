# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public optional foreign traces against literals frozen before comparison."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk._execution_delegated import _DelegatedOwner
from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport, _EntryGate
from anonymizer.engine.graph_sdk.executor import start_execution
from anonymizer.engine.graph_sdk.requests import TextArtifactValue
from anonymizer.graph._values import InvocationId
from tests.graph_sdk.test_blocking import until
from tests.graph_sdk.test_delegated_execution import _execute
from tests.graph_sdk.test_delegated_readiness import _Probe, _qualify
from tests.graph_sdk.test_delegated_worker import _loop_and_pool
from tests.graph_sdk.test_foreign_resources import fixture
from tests.graph_sdk.test_optional_tail import optional_fixture

TABLE = json.loads((Path(__file__).parent / "reference/foreign_resources/optional_tail.json").read_text())


def tail_transport(execution, services):
    owner = next(row for row in execution.policies if row.delegated)
    node = execution.context.prepared.reservation_recipe[owner.delegated.child_slots[1]].template
    impl = next(row.implementations[0].implementation for row in execution.policies if row.node == node)
    return next(row.transport for row in services.handles if row.implementation == impl)


async def foreign_optional(loop, pool, *, correction=False, probe="available", base=None, **options):
    base = base or optional_fixture(correction=correction, readiness=probe)
    q = tail_transport(*base[:2])
    values = await fixture(loop, pool, base=base, held_transport=q, **options)
    return (*values, next(client for client in values[3] if client.transport is q))


def assert_trace(result, clients, expected):
    assert sum(len(c.calls) for c in clients) == len(result.requests.dispatches) == expected["sends"]
    assert sum(t.attempt is not None for t in result.record.terminals) == expected["starts"]
    assert len(result.record.terminals) == expected["terminals"]
    assert len(result.requests.settlements) == expected["settlements"]
    assert sum(t.unused_retry is not None for t in result.record.terminals) == expected["unused"]
    if "request_terminals" in expected:
        assert [t.category for t in result.requests.terminals] == expected["request_terminals"]
    assert all(state.complete for state in result.states)


@pytest.mark.parametrize("owner", ["sdk", "caller"])
@pytest.mark.parametrize("probe", ["available", "async_required"])
@pytest.mark.parametrize("correction", [False, True])
def test_public_optional_foreign_candidate_and_all_registered_resources(owner, probe, correction):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, submit, clients, worker, adapters, q = await foreign_optional(
                loop, pool, owner=owner, correction=correction, probe=probe
            )
            result = await _execute(execution, services)
            expected = TABLE["correction" if correction else "unused"]
            assert_trace(result, clients, expected)
            assert submit.calls == expected["submissions"]
            assert sum(h.local.calls for h in services.handles if isinstance(h.local, _Probe)) == expected["probes"]
            assert len(result.requests.chains) == expected["chains"]
            assert len(q.calls) == int(correction)
            chain = next(c for c in result.requests.chains if c.role == "generation")
            owner_attempt = next(iter(chain.associations)).task
            assert sum(d.child.leaf == owner_attempt for d in result.requests.dispatches) == expected["parent_charges"]
            assert len({d.child.leaf for d in result.requests.dispatches}) == expected["sends"]
            unused = [t for t in result.record.terminals if t.unused_retry]
            if unused:
                assert unused[0].unused_retry.owner == owner_attempt and unused[0].attempt is None
                assert unused[0].unused_retry.after == result.requests.dispatches[-1].child.leaf
            candidate = result.final_outputs[0].candidate.artifact
            assert dict(result.artifacts)[candidate] == TextArtifactValue(text=TABLE["candidate"])
            assert result.assessments[0].evidence_artifact != candidate
            assert len(_qualify(execution, result).qualified) == expected["protected"]
            assert sum(c.closes for c in clients) == expected[owner + "_closes"]
            assert len(result.cleanup) == len(result.cleanup_associations) == expected["cleanup_facts"]
            assert all(a.purpose == "accounting" for a in result.cleanup_associations)
            assert all(a.local_users == 0 for a in adapters.values()) and worker.local_users == 0
            assert not pool.closed and loop.is_running()
            if owner == "caller":

                async def ping():
                    return [await c.ping() for c in clients]

                assert await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(ping(), loop)) == [1, 1, 1]
                control = type("Control", (), {"cancelled": False})()
                assert await pool._run(lambda: 7, lambda: True, control) == 7
            else:
                assert all(c.closed and c.closes == 1 for c in clients)

    asyncio.run(run())


@pytest.mark.parametrize("close", ["exception", "self_cancel", "held"])
def test_unsent_q_cleanup_is_owned_and_scoped(close):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, clients, _, adapters, q = await foreign_optional(loop, pool)
            q.close_result = close
            q.hold_close = close == "held"
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            try:
                if close == "held":
                    await until(q.close_entered.is_set)
                    for _ in range(2):
                        running._task.cancel()
                        observer = asyncio.create_task(running.wait())
                        await asyncio.sleep(0)
                        observer.cancel()
                        with pytest.raises(asyncio.CancelledError):
                            await observer
                        assert not running._task.done() and q.closes == 1
                    loop.call_soon_threadsafe(q.close_release.set)
                result = await asyncio.wait_for(running.wait(), 3)
                assert_trace(result, clients, TABLE["unused"])
                q_adapter = next(a for a in adapters.values() if a._inner is q)
                fact = next(f for f in result.cleanup if f.resource == q_adapter.lease.resource)
                assert fact.disposition == TABLE["q_close"][close]
                assert all(c.closes == 1 for c in clients) and not q.calls
                qualified = _qualify(execution, result)
                assert len(qualified.qualified) == int(close == "held")
                if close != "held":
                    assert "cleanup_accounting" in qualified.targets[0].withholding
            finally:
                loop.call_soon_threadsafe(q.close_release.set)
                await running.wait()

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["queued", "accepted", "live", "paused", "hook"])
@pytest.mark.parametrize("stop", ["unknown", "confirmed", "direct", "task"])
def test_q_cancellation_retains_entry_hook_settlement_and_cleanup(monkeypatch, stage, stop):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, _, clients, worker, adapters, q = await foreign_optional(
                loop,
                pool,
                correction=True,
                hold=True,
                hold_stop=stage == "hook",
                stop_self_cancel=stop if stop in {"direct", "task"} else False,
                confirmed=stop == "confirmed",
            )
            holder, delayed = {}, []
            original_run = _DelegatedOwner.run

            async def capture(owner, handle):
                holder["owner"] = owner
                return await original_run(owner, handle)

            monkeypatch.setattr(_DelegatedOwner, "run", capture)
            host = asyncio.get_running_loop()
            schedule = host.call_soon_threadsafe
            gate_count = 0

            def queue(callback, *args, **kwargs):
                nonlocal gate_count
                if isinstance(getattr(callback, "__self__", None), _EntryGate):
                    gate_count += 1
                    if stage == "queued" and gate_count == 3:
                        delayed.append((callback, args))
                        return None
                return schedule(callback, *args, **kwargs)

            monkeypatch.setattr(host, "call_soon_threadsafe", queue)
            reached, proceed = Event(), Event()
            original_wait = _EntryGate.wait
            waits = 0

            def barrier(gate, timeout):
                nonlocal waits
                accepted = original_wait(gate, timeout)
                waits += 1
                if stage == "accepted" and waits == 3 and accepted:
                    reached.set()
                    assert proceed.wait(5)
                return accepted

            monkeypatch.setattr(_EntryGate, "wait", barrier)
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            resumed = None
            try:
                await until(
                    lambda: (
                        bool(delayed)
                        if stage == "queued"
                        else reached.is_set()
                        if stage == "accepted"
                        else q.entered.is_set()
                    )
                )
                if stage == "accepted":
                    assert not q.calls and len(holder["owner"].runtime.request_authority.state.dispatches) == 3
                if stage == "paused":
                    loop.call_soon_threadsafe(loop.stop)
                    await until(lambda: not loop.is_running())
                running.request_cancel()
                for _ in range(2):
                    observer = asyncio.create_task(running.wait())
                    await asyncio.sleep(0)
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
                if stage != "queued":
                    assert not running._task.done() and not any(c.closes for c in clients)
                proceed.set()
                if stage == "paused":
                    resumed = Thread(target=loop.run_forever, daemon=True)
                    resumed.start()
                authority = holder["owner"].runtime.request_authority
                if stage == "hook":
                    await until(q.stopping.is_set)
                    loop.call_soon_threadsafe(q.release.set)
                    await until(lambda: all(task.done() for task in q.calls))
                    assert not running._task.done() and worker.local_users > 0
                    assert any(a.local_users for a in adapters.values()) and not any(c.closes for c in clients)
                    loop.call_soon_threadsafe(q.stop_release.set)
                await until(lambda: len(authority.state.terminals) == 3)
                first = authority.state.terminals
                assert first[-1].category == ("cancelled" if stage == "queued" or stop == "confirmed" else "lost")
                if stage != "queued":
                    loop.call_soon_threadsafe(q.release.set)
                result = await asyncio.wait_for(running.wait(), 3)
                for callback, args in delayed:
                    callback(*args)
                expected = TABLE["queued_q_cancel" if stage == "queued" else "entered_q_cancel"]
                assert_trace(result, clients, expected)
                assert result.requests.terminals == first
                assert len(q.calls) == int(stage != "queued")
                assert q.stopping.is_set() == (stage != "queued")
                assert all(c.closes == 1 for c in clients)
                assert worker.local_users == 0 and all(a.local_users == 0 for a in adapters.values())
                assert not result.requests.remote_outstanding and not result.requests.local_in_flight
                assert not _qualify(execution, result).qualified
            finally:
                proceed.set()
                if not loop.is_running():
                    resumed = Thread(target=loop.run_forever, daemon=True)
                    resumed.start()
                loop.call_soon_threadsafe(q.release.set)
                loop.call_soon_threadsafe(q.stop_release.set)
                await running.wait()
                if resumed:
                    loop.call_soon_threadsafe(loop.stop)
                    await asyncio.to_thread(resumed.join)

    asyncio.run(run())


def test_mandatory_foreign_two_slot_shape_still_rejects_before_claim(monkeypatch):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            base = optional_fixture(readiness="available", declared=False)
            execution, services, submit, clients, _, adapters, _ = await foreign_optional(loop, pool, base=base)

            def forbidden(*args, **kwargs):
                raise AssertionError("unsupported mandatory shape reached identity or claim")

            monkeypatch.setattr(InvocationId, "new", forbidden)
            monkeypatch.setattr(ForeignLoopTransport, "claim_many", forbidden)
            try:
                with pytest.raises(EffectRejected) as caught:
                    await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
                assert caught.value.code == EffectCode.UNSUPPORTED
                assert submit.calls == 0 and all(not c.calls for c in clients)
                assert all(not a._claimed for a in adapters.values())
            finally:
                for adapter in adapters.values():
                    await adapter.close()

    asyncio.run(run())


@pytest.mark.parametrize("correction", [False, True])
def test_same_registered_adapter_serves_sequential_leaves_and_closes_once(correction):
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, submit = optional_fixture(readiness="available", correction=correction)
            q = tail_transport(execution, services)
            routes = [
                (h.operation, "correction" if h.transport is q else "initial", h.transport)
                for h in services.handles
                if h.transport
            ]

            class Shared:
                async def dispatch(self, request):
                    transport = next(
                        transport
                        for op, purpose, transport in routes
                        if op == request.operation and purpose == request.purpose
                    )
                    return await transport.dispatch(request)

                async def cancel(self, request):
                    raise AssertionError("successful shared client received cancellation")

            shared = Shared()
            services = replace(
                services, handles=tuple(replace(h, transport=shared) if h.transport else h for h in services.handles)
            )
            execution, services, _, clients, worker, adapters = await fixture(
                loop, pool, base=(execution, services, submit)
            )
            result = await _execute(execution, services)
            expected = TABLE["correction" if correction else "unused"]
            assert_trace(result, clients, expected)
            assert len(clients) == 1 and len({id(a) for a in adapters.values()}) == 1
            assert clients[0].closes == 1 and clients[0].closed
            assert len({d.request for d in result.requests.dispatches}) == expected["sends"]
            assert len({d.child.leaf for d in result.requests.dispatches}) == expected["sends"]
            assert len(result.cleanup) == len(result.cleanup_associations) == 2
            assert worker.local_users == 0 and all(a.local_users == 0 for a in adapters.values())
            assert len(_qualify(execution, result).qualified) == 1

    asyncio.run(run())


@pytest.mark.parametrize("stage", ["rollback", "after_publication"])
def test_foreign_resources_do_not_override_publication_boundary(monkeypatch, stage):
    from anonymizer.engine.graph_sdk import _execution_runtime
    from anonymizer.engine.graph_sdk.executor import LocalFailure

    async def run():
        async with _loop_and_pool() as (loop, pool):
            base = optional_fixture(readiness="available")
            execution, services, submit = base
            owner_node = next(row.node for row in execution.policies if row.delegated)
            if stage == "rollback":
                original = _execution_runtime._materialize_subgraph_outputs

                def rollback(admitted, target, state, *args):
                    if any(row.template == owner_node and row.status == "success" for row in state.entries):
                        raise EffectRejected(EffectCode.CONTRADICTORY)
                    return original(admitted, target, state, *args)

                monkeypatch.setattr(_execution_runtime, "_materialize_subgraph_outputs", rollback)
            else:

                class FailedAssessment:
                    async def run(self, inputs):
                        return LocalFailure(failure="permanent")

                services = replace(
                    services,
                    handles=tuple(
                        replace(h, local=FailedAssessment()) if h.local and not isinstance(h.local, _Probe) else h
                        for h in services.handles
                    ),
                )
            execution, services, _, clients, _, _, q = await foreign_optional(
                loop, pool, base=(execution, services, submit)
            )
            result = await _execute(execution, services)
            assert len(result.requests.dispatches) == 2 and not q.calls
            assert all(c.closes == 1 and c.closed for c in clients)
            assert all(state.complete for state in result.states)
            assert sum(t.unused_retry is not None for t in result.record.terminals) == int(stage == "after_publication")
            if stage == "rollback":
                assert not any(p.node == owner_node and p.port == "context" for p in result.ports)
            try:
                assert not _qualify(execution, result).qualified
            except EffectRejected as error:
                assert stage == "rollback" and error.code == EffectCode.CONTRADICTORY

    asyncio.run(run())


def test_completed_q_and_cancelled_waiters_preserve_success():
    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, submit = optional_fixture(readiness="available", correction=True)
            parent = next(h.delegated for h in services.handles if h.delegated)
            completed, release = Event(), Event()

            class Parent:
                execution = "same_loop"

                async def submit(self, inputs, requests):
                    value = await parent.submit(inputs, requests)
                    completed.set()
                    await asyncio.to_thread(release.wait)
                    return value

            services = replace(
                services, handles=tuple(replace(h, delegated=Parent()) if h.delegated else h for h in services.handles)
            )
            execution, services, _, clients, _, _, _ = await foreign_optional(
                loop, pool, base=(execution, services, submit)
            )
            running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            try:
                await until(completed.is_set)
                assert sum(len(c.calls) for c in clients) == 3 and all(t.done() for c in clients for t in c.calls)
                for _ in range(2):
                    observer = asyncio.create_task(running.wait())
                    await asyncio.sleep(0)
                    observer.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await observer
                release.set()
                result = await asyncio.wait_for(running.wait(), 3)
                assert_trace(result, clients, TABLE["correction"])
                assert all(not c.stopping.is_set() and c.closes == 1 for c in clients)
                assert len(_qualify(execution, result).qualified) == 1
            finally:
                release.set()
                await running.wait()

    asyncio.run(run())
