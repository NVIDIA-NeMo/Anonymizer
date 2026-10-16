# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional capacity compared with a literal reference frozen before implementation."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest

from anonymizer.engine.graph_sdk.executor import LocalFailure
from anonymizer.engine.graph_sdk.requests import TextArtifactValue
from anonymizer.graph import EffectCode, EffectRejected
from anonymizer.graph._values import ContractViolation, InvocationId, TaskAttemptId
from tests.graph_sdk.test_delegated_execution import _execute, _fixture
from tests.graph_sdk.test_delegated_readiness import _qualify, _ready_fixture
from tests.graph_sdk.test_delegated_worker import _loop_and_pool, _worker_services
from tests.graph_sdk.test_request_child_qualification import readmit

TABLE = json.loads((Path(__file__).parent / "reference/optional_tail/manifest.json").read_text())


def optional_fixture(*, correction=False, readiness=None, declared=True):
    if readiness is None:
        execution, services, submit, _ = _fixture(parent_io=True, usage="exact")
    else:
        execution, services, submit, *_ = _ready_fixture(readiness, simple=False, usage="exact")
    owner = next(row for row in execution.policies if row.delegated)
    if declared:
        owner = replace(owner, delegated=replace(owner.delegated, optional_tail=owner.delegated.child_slots[1:]))
        execution = readmit(execution, owner)
    generation = execution.context.prepared.reservation_recipe[owner.delegated.child_slots[0]].template
    implementation = next(row.implementations[0].implementation for row in execution.policies if row.node == generation)
    next(
        handle.transport for handle in services.handles if handle.implementation == implementation
    ).malformed = correction
    if not correction:
        submit.slots = submit.slots[:1]
    return execution, services, submit


@pytest.mark.parametrize("case", TABLE["cases"], ids=lambda row: row["name"])
@pytest.mark.parametrize("worker", [False, True])
def test_literal_optional_tail_public_protection(case, worker):
    async def run():
        readiness = (
            "available"
            if "available" in case["name"]
            else "async_required"
            if "async_required" in case["name"]
            else None
        )
        execution, services, submit = optional_fixture(correction="malformed" in case["name"], readiness=readiness)
        async with _loop_and_pool() as (loop, pool):
            if worker:
                parent = next(handle.delegated for handle in services.handles if handle.delegated)

                async def foreign(inputs, borrowed):
                    def copied():
                        return asyncio.run_coroutine_threadsafe(parent.submit(inputs, borrowed), loop).result()

                    return await asyncio.to_thread(copied)

                def callback(inputs, borrowed):
                    return asyncio.run_coroutine_threadsafe(foreign(inputs, borrowed), loop).result()

                execution, services, adapter = _worker_services(execution, services, loop, pool, callback)
            result = await _execute(execution, services)
            assert submit.calls == case["submits"]
            assert sum(row.attempt is not None for row in result.record.terminals) == case["starts"]
            assert len(result.record.terminals) == case["terminals"]
            assert sum(state.events_applied for state in result.states) == 1 + 2 * case["starts"] + case["unused"]
            assert len(result.requests.chains) == case["chains"]
            assert sum(handle.transport.calls for handle in services.handles if handle.transport) == case["sends"]
            assert len(result.requests.dispatches) == case["sends"]
            assert len(result.requests.terminals) == case["request_terminals"]
            assert len(result.requests.settlements) == case["settlements"]
            assert [row.category for row in result.requests.terminals] == case["physical_categories"]
            unused = [row for row in result.record.terminals if row.unused_retry]
            assert len(unused) == case["unused"]
            assert all(row.attempt is None and row.category == "blocked" for row in unused)
            generation_chain = next(row for row in result.requests.chains if row.role == "generation")
            owner_attempt = next(iter(generation_chain.associations)).task
            assert sum(row.child.leaf == owner_attempt for row in result.requests.dispatches) == case["parent_charges"]
            assert dict(result.artifacts)[result.final_outputs[0].candidate.artifact] == TextArtifactValue(
                text=TABLE["candidate"]
            )
            assert result.assessments[0].evidence_artifact != result.final_outputs[0].candidate.artifact
            assert len(_qualify(execution, result).qualified) == case["protected"]
            assert all(state.complete for state in result.states)
            if worker:
                assert adapter.local_users == pool.occupied == 0
                assert not pool.closed and loop.is_running()

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad", ["mandatory", "malformed", "unknown_outcome", "rollback", "artifact_limit", "after_publication"]
)
def test_optional_publication_boundary(monkeypatch, bad):
    from anonymizer.engine.graph_sdk import _execution_runtime

    execution, services, _ = optional_fixture(declared=bad != "mandatory")
    parent = next(handle.delegated for handle in services.handles if handle.delegated)
    owner_node = next(row.node for row in execution.policies if row.delegated)

    class Parent:
        execution = "same_loop"

        async def submit(self, inputs, requests):
            result = await parent.submit(inputs, requests)
            if bad == "malformed":
                return object()
            if bad == "unknown_outcome":
                return replace(result, results=(replace(result.results[0], outcome="unknown"),))
            return result

    services = replace(
        services, handles=tuple(replace(row, delegated=Parent()) if row.delegated else row for row in services.handles)
    )
    if bad in {"rollback", "artifact_limit"}:
        original = _execution_runtime._materialize_structure

        def rollback(admitted, target, state, *args):
            if any(row.template == owner_node and row.status == "success" for row in state.entries):
                raise EffectRejected(EffectCode.LIMIT_EXCEEDED if bad == "artifact_limit" else EffectCode.CONTRADICTORY)
            return original(admitted, target, state, *args)

        monkeypatch.setattr(_execution_runtime, "_materialize_structure", rollback)
    if bad == "after_publication":

        class FailedAssessment:
            async def run(self, inputs):
                return LocalFailure(failure="permanent")

        services = replace(
            services,
            handles=tuple(replace(row, local=FailedAssessment()) if row.local else row for row in services.handles),
        )
    result = asyncio.run(_execute(execution, services))
    assert len(result.requests.dispatches) == 1
    assert sum(row.unused_retry is not None for row in result.record.terminals) == (bad == "after_publication")
    assert all(state.complete for state in result.states)
    if bad != "after_publication":
        assert not any(row.node == owner_node and row.port == "context" for row in result.ports)
    if bad == "mandatory":
        owner_key = next(
            row.activation for state in result.states for row in state.entries if row.template == owner_node
        )
        owner_terminal = next(row for row in result.record.terminals if row.activation == owner_key)
        q = next(row.delegated.child_slots[1] for row in execution.policies if row.delegated)
        tail = next(row for row in result.record.terminals if row.activation.occurrence == q)
        assert owner_terminal.category == "failure" and owner_terminal.reasons == {"execution_failed"}
        assert tail.category == "blocked" and tail.attempt is None and tail.reasons == {"prerequisite"}
    try:
        assert not _qualify(execution, result).qualified
    except EffectRejected as error:
        assert error.code == EffectCode.CONTRADICTORY


@pytest.mark.parametrize(
    "bad, expected", json.loads((Path(__file__).parent / "reference/optional_tail/rejections.json").read_text()).items()
)
def test_retained_unused_proof_rejects_forgery(bad, expected):
    from anonymizer.engine.graph_sdk.requests import RequestDenialFact
    from tests.graph_sdk.child_qualification_fixtures import result_with

    execution, services, _ = optional_fixture()
    result = asyncio.run(_execute(execution, services))
    unused = next(row for row in result.record.terminals if row.unused_retry)
    cause = unused.unused_retry
    with pytest.raises((ContractViolation, EffectRejected)) as caught:
        if bad == "cause_type":
            changed = replace(unused, unused_retry="untrusted")
        elif bad == "foreign_owner":
            foreign = TaskAttemptId.new(
                activation=replace(cause.owner.activation, invocation=InvocationId.new(plan=result.record.plan))
            )
            changed = replace(unused, unused_retry=replace(cause, owner=foreign))
        elif bad == "reason":
            changed = replace(unused, unused_retry=None)
        elif bad == "owner":
            changed = replace(unused, unused_retry=replace(cause, owner=cause.after))
        elif bad == "after":
            changed = replace(
                unused, unused_retry=replace(cause, after=TaskAttemptId.new(activation=cause.after.activation))
            )
        elif bad == "attempt":
            changed = replace(unused, attempt=TaskAttemptId.new(activation=unused.activation))
        elif bad == "ordinary_reason":
            changed = replace(unused, unused_retry=None, reasons=frozenset({"omitted_optional"}))
        else:
            changed = unused
            receipt = result.requests
            if bad == "missing_publication":
                from anonymizer.engine.graph_sdk.executor import OperationOutputKey

                result = result_with(
                    result,
                    provenance=tuple(
                        row
                        for row in result.provenance
                        if not (
                            isinstance(row.key, OperationOutputKey) and row.key.activation == cause.owner.activation
                        )
                    ),
                )
            elif bad == "extra_dispatch":
                receipt = replace(
                    receipt,
                    dispatches=(*receipt.dispatches, replace(receipt.dispatches[0], child=None)),
                    dispatched_count=2,
                )
            elif bad == "denial":
                receipt = replace(
                    receipt,
                    denials=(
                        RequestDenialFact(
                            request=receipt.dispatches[0].request,
                            associations=receipt.dispatches[0].associations,
                            category="request_limit_stopped",
                        ),
                    ),
                )
            else:
                receipt = replace(
                    receipt, settlements=(replace(receipt.settlements[0], disposition="unknown", remote_stopped=None),)
                )
            result = result_with(result, requests=receipt)
        result = result_with(
            result,
            record=replace(
                result.record, terminals=tuple(changed if row == unused else row for row in result.record.terminals)
            ),
        )
        assert not _qualify(execution, result).qualified
    assert caught.value.code.value == expected


@pytest.mark.parametrize("stage", ["before_entry", "live", "before_publication"])
def test_optional_tail_cancellation_never_stamps(stage):
    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.engine.graph_sdk.requests import StopUnknown

    async def run():
        execution, services, submit = optional_fixture()
        parent = next(row.delegated for row in services.handles if row.delegated)
        slot = next(row.delegated.child_slots[0] for row in execution.policies if row.delegated)
        node = execution.context.prepared.reservation_recipe[slot].template
        implementation = next(row.implementations[0].implementation for row in execution.policies if row.node == node)
        inner = next(row.transport for row in services.handles if row.implementation == implementation)
        reached, release, stopping = asyncio.Event(), asyncio.Event(), asyncio.Event()
        calls = []

        class Parent:
            execution = "same_loop"

            async def submit(self, inputs, requests):
                if stage == "before_entry":
                    reached.set()
                    await release.wait()
                value = await parent.submit(inputs, requests)
                if stage == "before_publication":
                    reached.set()
                    await release.wait()
                return value

        class Transport:
            async def dispatch(self, envelope):
                calls.append(envelope.request)
                if stage == "live":
                    reached.set()
                    await release.wait()
                return await inner.dispatch(envelope)

            async def cancel(self, request):
                stopping.set()
                return StopUnknown()

        services = replace(
            services,
            handles=tuple(
                replace(row, delegated=Parent())
                if row.delegated
                else replace(row, transport=Transport())
                if row.implementation == implementation
                else row
                for row in services.handles
            ),
        )
        running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
        waiter = asyncio.create_task(running.wait())
        await asyncio.wait_for(reached.wait(), 2)
        running.request_cancel()
        if stage == "live":
            await asyncio.wait_for(stopping.wait(), 2)
        for _ in range(2):
            observer = asyncio.create_task(running.wait())
            await asyncio.sleep(0)
            observer.cancel()
            with pytest.raises(asyncio.CancelledError):
                await observer
        if stage == "live":
            assert not waiter.done()
        release.set()
        result = await asyncio.wait_for(waiter, 2)
        assert len(calls) == len(result.requests.dispatches) == int(stage != "before_entry")
        assert all(row.unused_retry is None for row in result.record.terminals)
        assert all(state.complete for state in result.states)
        if stage == "live":
            assert [row.category for row in result.requests.terminals] == ["lost"]
            assert result.requests.settlements[0].disposition == "completed"
            assert not result.requests.local_in_flight and not result.requests.remote_outstanding
        try:
            assert not _qualify(execution, result).qualified
        except EffectRejected:
            pass

    asyncio.run(run())


@pytest.mark.parametrize("bad", ["first_optional", "duplicate", "outside", "wrong_initial", "missing_evidence"])
def test_optional_shape_and_position_rejection(bad):
    execution, services, submit = optional_fixture()
    policy = next(row for row in execution.policies if row.delegated)
    g, q = policy.delegated.child_slots
    if bad in {"first_optional", "duplicate", "outside"}:
        with pytest.raises(EffectRejected) as caught:
            readmit(
                execution,
                replace(
                    policy,
                    delegated=replace(
                        policy.delegated,
                        optional_tail={"first_optional": (g,), "duplicate": (q, q), "outside": (999,)}[bad],
                    ),
                ),
            )
        assert caught.value.code == (EffectCode.DUPLICATE if bad == "duplicate" else EffectCode.UNSUPPORTED)
        return

    class Parent:
        execution = "same_loop"

        async def submit(self, inputs, requests):
            with pytest.raises(EffectRejected) as caught:
                await requests.request(q, source_kind="initial" if bad == "wrong_initial" else "correction")
            assert caught.value.code == (
                EffectCode.REQUEST_CAUSALITY if bad == "wrong_initial" else EffectCode.MISSING_PREDECESSOR
            )
            return LocalFailure(failure="implementation_exception")

    services = replace(
        services, handles=tuple(replace(row, delegated=Parent()) if row.delegated else row for row in services.handles)
    )
    result = asyncio.run(_execute(execution, services))
    assert not result.requests.dispatches and all(row.unused_retry is None for row in result.record.terminals)


def test_unregistered_foreign_optional_tail_rejects_before_identity(monkeypatch):
    from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.graph._values import InvocationId

    async def run():
        execution, services, submit = optional_fixture(readiness="available")
        async with _loop_and_pool() as (loop, _):
            services = replace(
                services,
                handles=tuple(
                    replace(
                        row, transport=ForeignLoopTransport(inner=row.transport, loop=loop, gate_timeout_ns=1_000_000)
                    )
                    if row.transport
                    else row
                    for row in services.handles
                ),
            )

            def forbidden(*args, **kwargs):
                raise AssertionError("unregistered foreign admission reached invocation/resource claims")

            monkeypatch.setattr(InvocationId, "new", forbidden)
            monkeypatch.setattr(ForeignLoopTransport, "claim_many", forbidden)
            with pytest.raises(EffectRejected) as caught:
                await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
            assert caught.value.code == EffectCode.UNSUPPORTED and submit.calls == 0

    asyncio.run(run())


def test_caught_optional_request_rejection_cannot_stamp():
    execution, services, _ = optional_fixture()
    parent = next(row.delegated for row in services.handles if row.delegated)
    q = next(row.delegated.child_slots[1] for row in execution.policies if row.delegated)

    class Parent:
        execution = "same_loop"

        async def submit(self, inputs, requests):
            result = await parent.submit(inputs, requests)
            with pytest.raises(EffectRejected) as caught:
                await requests.request(q, source_kind="initial")
            assert caught.value.code == EffectCode.REQUEST_CAUSALITY
            return result

    services = replace(
        services, handles=tuple(replace(row, delegated=Parent()) if row.delegated else row for row in services.handles)
    )
    result = asyncio.run(_execute(execution, services))
    assert len(result.requests.dispatches) == 1
    assert all(row.unused_retry is None for row in result.record.terminals)
    assert not any(
        row.port == "context" and row.node == next(p.node for p in execution.policies if p.delegated)
        for row in result.ports
    )


@pytest.mark.parametrize("usage", ["exact", "unknown_allowed", "failed_correction"])
def test_optional_usage_policy_and_failed_used_tail(usage):
    from anonymizer.engine.graph_sdk.requests import UnknownUsage

    expected = json.loads((Path(__file__).parent / "reference/optional_tail/usage.json").read_text())[usage]
    execution, services, _ = optional_fixture(correction=usage == "failed_correction")
    owner = next(row for row in execution.policies if row.delegated)
    if usage == "unknown_allowed":
        execution = readmit(execution, replace(owner, delegated=replace(owner.delegated, usage_requirement=usage)))
    if usage == "failed_correction":
        for handle in services.handles:
            if handle.transport:
                handle.transport.malformed = True
    else:
        node = execution.context.prepared.reservation_recipe[owner.delegated.child_slots[0]].template
        implementation = next(row.implementations[0].implementation for row in execution.policies if row.node == node)
        inner = next(row.transport for row in services.handles if row.implementation == implementation)

        class Unknown:
            async def dispatch(self, envelope):
                value = await inner.dispatch(envelope)
                return replace(value, settlement=replace(value.settlement, usage=UnknownUsage()))

            async def cancel(self, request):
                return await inner.cancel(request)

        services = replace(
            services,
            handles=tuple(
                replace(row, transport=Unknown()) if row.implementation == implementation else row
                for row in services.handles
            ),
        )
    result = asyncio.run(_execute(execution, services))
    assert len(result.requests.dispatches) == expected["sends"]
    assert sum(row.unused_retry is not None for row in result.record.terminals) == expected["unused"]
    try:
        assert len(_qualify(execution, result).qualified) == expected["protected"]
    except EffectRejected:
        assert expected["protected"] == 0
