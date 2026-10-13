# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real readiness/probe execution compared with an independently fixed trace."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.executor import (
    ImplementationHandle,
    LocalCompleted,
    LocalFailure,
    LocalResult,
    admit_execution_plan,
)
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    TextArtifactValue,
    TransportSuccess,
)
from tests.graph_sdk.delegated_execution_fixtures import with_parent_publication
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_delegated_execution import _execute, _fixture, _Transport

TABLE = json.loads((Path(__file__).parent / "reference/delegated_readiness/manifest.json").read_text())


@dataclass
class _Probe:
    outcome: str = "async_required"
    failure: bool = False
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalResult:
        self.calls += 1
        if self.failure:
            return LocalFailure(failure="permanent")
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome=self.outcome,
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            )
        )


class _ReadinessTransport(_Transport):
    async def dispatch(self, request):
        result = await super().dispatch(request)
        if isinstance(result, TransportSuccess):
            return replace(result, results=tuple(replace(row, outcome="ready") for row in result.results))
        return result


def _ready_fixture(outcome="async_required", failure=None, early=False, *, simple=True, usage="unknown_allowed"):
    base, services, submit, transports = _fixture(simple=simple, usage=usage)
    execution = with_parent_publication(base, readiness=True)
    owner = next(item for item in execution.policies if item.delegated)
    assert owner.delegated is not None and owner.delegated.readiness is not None
    descriptor = owner.delegated.readiness
    recipe = execution.context.prepared.reservation_recipe
    h, r, p = (recipe[index].template for index in descriptor.slots)
    probe = _Probe(outcome, failure == "probe_failure")
    readiness = _ReadinessTransport(malformed=failure == "readiness_failure")
    submit.slots = owner.delegated.child_slots

    class Parent:
        execution = "same_loop"

        async def submit(self, inputs, requests):
            if early:
                with pytest.raises(EffectRejected):
                    await requests.request(submit.slots[0])
            await requests.readiness()
            await requests.probe()
            return await submit.submit(inputs, requests)

    handles = []
    for policy in execution.policies:
        if policy.node == h:
            continue
        impl = policy.implementations[0]
        previous = next((item for item in services.handles if item.implementation == impl.implementation), None)
        if previous is not None:
            handle = replace(previous, operation=impl.capability.operation)
            if handle.delegated:
                handle = replace(handle, delegated=Parent())
            if handle.local:
                handle.local.evidence_port = "assessment"
        else:
            handle = ImplementationHandle(
                implementation=impl.implementation,
                operation=impl.capability.operation,
                configuration=impl.configuration,
                local=probe if policy.node == p else None,
                transport=readiness if policy.node == r else None,
                resource=None,
            )
        handles.append(handle)
    return execution, replace(services, handles=tuple(handles)), submit, transports[0], readiness, probe


def _qualify(execution, result):
    admitted, current, submissions = _inputs(execution, result)
    return qualify(admitted=admitted, result=result, current=current, submissions=submissions)


@pytest.mark.parametrize("outcome", TABLE["positive"])
def test_readiness_and_probe_qualify_actual_candidate(outcome):
    execution, services, submit, generation, readiness, probe = _ready_fixture(outcome, early=True)
    result = asyncio.run(_execute(execution, services))
    expected = TABLE["expected"]
    assert submit.calls == expected["submissions"] and probe.calls == expected["probe_calls"]
    assert readiness.calls == generation.calls == 1
    assert sum(state.events_applied for state in result.states) == 1 + 2 * expected["starts"]
    assert len(result.record.terminals) == expected["canonical_terminals"]
    assert sum(row.attempt is not None for row in result.record.terminals) == expected["attempts"]
    assert all(row.category == "success" for row in result.record.terminals)
    assert len(result.requests.chains) == expected["chains"]
    assert len(result.requests.dispatches) == expected["dispatches"]
    assert len(result.requests.terminals) == expected["request_terminals"]
    assert len(result.requests.settlements) == expected["settlements"]
    assert [row.child.role for row in result.requests.dispatches] == expected["roles"]
    assert [row.results[0].outcome for row in result.requests.terminals] == expected["physical_outcomes"]
    for dispatch in result.requests.dispatches:
        semantic = next(iter(dispatch.associations)).task
        assert semantic != dispatch.child.leaf
        assert any(row.attempt == semantic and row.category == "success" for row in result.record.terminals)
        assert any(row.attempt == dispatch.child.leaf and row.category == "success" for row in result.record.terminals)
    candidate = result.final_outputs[0].candidate.artifact
    assert dict(result.artifacts)[candidate] == TextArtifactValue(text=expected["candidate"])
    assert result.assessments[0].evidence_artifact != candidate
    qualified = _qualify(execution, result)
    assert len(qualified.qualified) == expected["qualified"] and not qualified.targets[0].withholding
    assert all(state.complete for state in result.states)


@pytest.mark.parametrize("failure", TABLE["stopped"])
def test_failed_readiness_or_probe_stops_generation(failure):
    execution, services, _, generation, readiness, probe = _ready_fixture(failure=failure)
    result = asyncio.run(_execute(execution, services))
    expected = TABLE["stopped"][failure]
    assert readiness.calls == expected["sends"] and probe.calls == expected["probes"]
    assert generation.calls == expected["generation"]
    assert all(state.complete for state in result.states)
    qualified = _qualify(execution, result)
    assert not qualified.qualified and qualified.targets[0].withholding


@pytest.mark.parametrize("bad", ["semantic", "physical", "probe", "outcome", "route"])
def test_readiness_descriptor_rejects_foreign_roles(bad):
    execution, *_ = _ready_fixture()
    owner = next(item for item in execution.policies if item.delegated)
    descriptor = owner.delegated.readiness
    changes = {
        "semantic": {"owner_slot": owner.delegated.child_slots[0]},
        "physical": {"request_slot": 999},
        "probe": {"probe_slot": owner.delegated.child_slots[0]},
        "outcome": {"ready_outcome": "NOT_DECLARED"},
        "route": {"route": replace(descriptor.route, parse_attempts=2)},
    }[bad]
    with pytest.raises(EffectRejected):
        changed = replace(owner, delegated=replace(owner.delegated, readiness=replace(descriptor, **changes)))
        admit_execution_plan(
            context=execution.context,
            capabilities=execution.capabilities,
            policies=tuple(changed if item.node == owner.node else item for item in execution.policies),
            decisions=(),
            assessment_productions=execution.assessment_productions,
            assessment_limits=execution.assessment_limits,
        )


@pytest.mark.parametrize("caught", TABLE["cancellation"]["parent_catches"])
@pytest.mark.parametrize("stage", TABLE["cancellation"]["stage"])
def test_readiness_and_probe_retain_cancelled_owned_work(monkeypatch, caught, stage):
    from anonymizer.engine.graph_sdk._execution_delegated import _DelegatedOwner
    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.engine.graph_sdk.requests import StopUnknown
    from tests.graph_sdk.test_blocking import until

    async def run():
        execution, services, submit, generation, readiness, probe = _ready_fixture()
        holder = {}
        started, stopping, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        original = _DelegatedOwner.run

        async def capture(owner, handle):
            holder["owner_task"] = asyncio.current_task()
            return await original(owner, handle)

        monkeypatch.setattr(_DelegatedOwner, "run", capture)

        class Parent:
            execution = "same_loop"

            async def submit(self, inputs, requests):
                holder["parent"] = asyncio.current_task()
                try:
                    await requests.readiness()
                    await requests.probe()
                    return await submit.submit(inputs, requests)
                except asyncio.CancelledError:
                    if not caught:
                        raise
                    return LocalFailure(failure="implementation_exception")

        class Transport:
            async def dispatch(self, envelope):
                holder["work"] = asyncio.current_task()
                started.set()
                await release.wait()
                return await readiness.dispatch(envelope)

            async def cancel(self, request):
                stopping.set()
                return StopUnknown()

        class Probe:
            async def run(self, inputs):
                holder["work"] = asyncio.current_task()
                started.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    stopping.set()
                    await release.wait()
                return await probe.run(inputs)

        descriptor = next(item.delegated.readiness for item in execution.policies if item.delegated)
        node = execution.context.prepared.reservation_recipe[descriptor.probe_slot].template
        probe_implementation = next(
            item.implementations[0].implementation for item in execution.policies if item.node == node
        )
        services = replace(
            services,
            handles=tuple(
                replace(handle, delegated=Parent())
                if handle.delegated
                else replace(handle, transport=Transport())
                if stage == "readiness" and handle.transport is readiness
                else replace(handle, local=Probe())
                if stage == "probe" and handle.implementation == probe_implementation
                else handle
                for handle in services.handles
            ),
        )
        running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
        waiter = asyncio.create_task(running.wait())
        try:
            await asyncio.wait_for(started.wait(), 2)
            holder["parent"].cancel()
            await asyncio.wait_for(stopping.wait(), 2)
            await until(lambda: holder["parent"].done())
            for _ in range(2):
                holder["owner_task"].cancel()
                await asyncio.sleep(0)
                assert not waiter.done() and not holder["work"].done()
        finally:
            release.set()
        result = await asyncio.wait_for(waiter, 2)
        expected = TABLE["cancellation"]["after_release"]
        assert generation.calls == expected["generation"]
        assert len(result.requests.dispatches) == expected["dispatches"]
        assert len(result.requests.local_in_flight) == expected["local_outstanding"]
        assert len(result.requests.remote_outstanding) == expected["remote_outstanding"]
        assert all(state.complete for state in result.states)
        assert all(holder[key].done() for key in ("work", "owner_task", "parent"))
        assert len(_qualify(execution, result).qualified) == expected["qualified"]

    asyncio.run(run())


@pytest.mark.parametrize(
    "bad",
    [
        "readiness_role",
        "semantic_owner",
        "missing_leaf",
        "unstarted_h",
        "undeclared_outcome",
        "extra_h",
        "extra_r",
        "extra_p",
    ],
)
def test_qualifier_rejects_forged_readiness_receipts(bad):
    from anonymizer.engine.graph_sdk.requests import PhysicalRequestId, RequestPolicyBinding, SemanticAssociation
    from tests.graph_sdk.child_qualification_fixtures import result_with

    execution, services, *_ = _ready_fixture()
    result = asyncio.run(_execute(execution, services))
    receipt = result.requests
    first = receipt.dispatches[0]
    chain = next(row for row in receipt.chains if row.role == "readiness")
    semantic = next(iter(first.associations)).task
    changes = {}
    if bad == "readiness_role":
        receipt = replace(
            receipt, chains=tuple(replace(row, role="generation") if row == chain else row for row in receipt.chains)
        )
    elif bad == "semantic_owner":
        other = receipt.dispatches[1].associations
        receipt = replace(
            receipt, chains=tuple(replace(row, associations=other) if row == chain else row for row in receipt.chains)
        )
    elif bad == "missing_leaf":
        receipt = replace(
            receipt, dispatches=(replace(first, child=replace(first.child, leaf=None)), *receipt.dispatches[1:])
        )
    elif bad == "unstarted_h":
        changes["record"] = replace(
            result.record,
            terminals=tuple(
                replace(row, attempt=None, category="blocked", reasons=frozenset({"prerequisite"}))
                if row.attempt == semantic
                else row
                for row in result.record.terminals
            ),
        )
    elif bad == "undeclared_outcome":
        terminal = receipt.terminals[0]
        receipt = replace(
            receipt,
            terminals=(
                replace(terminal, results=tuple(replace(row, outcome="NOT_DECLARED") for row in terminal.results)),
                *receipt.terminals[1:],
            ),
        )
    else:
        attempt = semantic if bad == "extra_h" else first.child.leaf
        if bad == "extra_p":
            descriptor = next(item.delegated.readiness for item in execution.policies if item.delegated)
            attempt = next(
                row.attempt for row in result.record.terminals if row.activation.occurrence == descriptor.probe_slot
            )
        request = PhysicalRequestId.new(scope=receipt.scope)
        extra = replace(first, request=request, child=None, associations=frozenset({SemanticAssociation(task=attempt)}))
        association = next(iter(extra.associations))
        terminal = replace(
            receipt.terminals[0],
            request=request,
            results=tuple(replace(row, association=association) for row in receipt.terminals[0].results),
        )
        receipt = replace(
            receipt,
            dispatched_count=3,
            dispatches=(*receipt.dispatches, extra),
            terminals=(*receipt.terminals, terminal),
            settlements=(*receipt.settlements, replace(receipt.settlements[0], request=request)),
            bindings=receipt.bindings
            | {RequestPolicyBinding.create(association=association, policies=frozenset({first.policy}))},
        )
    forged = result_with(result, requests=receipt, **changes)
    with pytest.raises(EffectRejected):
        _qualify(execution, forged)


def test_probe_self_cancellation_closes_canonical_attempt():
    class Probe:
        async def run(self, inputs):
            raise asyncio.CancelledError

    execution, services, _, generation, _, probe = _ready_fixture()
    services = replace(
        services,
        handles=tuple(
            replace(handle, local=Probe()) if handle.local is probe else handle for handle in services.handles
        ),
    )
    result = asyncio.run(_execute(execution, services))
    assert generation.calls == 0
    assert all(state.complete for state in result.states)
    assert not _qualify(execution, result).qualified


@pytest.mark.parametrize("bad", ["extra_h_handle", "missing_probe", "synchronous_probe"])
def test_unsupported_readiness_services_reject_before_identity(monkeypatch, bad):
    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.graph._values import InvocationId

    execution, services, submit, generation, readiness, probe = _ready_fixture()
    descriptor = next(item.delegated.readiness for item in execution.policies if item.delegated)
    if bad == "extra_h_handle":
        node = execution.context.prepared.reservation_recipe[descriptor.owner_slot].template
        impl = next(item.implementations[0] for item in execution.policies if item.node == node)
        services = replace(
            services,
            handles=(
                *services.handles,
                ImplementationHandle(
                    implementation=impl.implementation,
                    operation=impl.capability.operation,
                    configuration=impl.configuration,
                    transport=readiness,
                    local=None,
                    resource=None,
                ),
            ),
        )
    elif bad == "missing_probe":
        services = replace(services, handles=tuple(handle for handle in services.handles if handle.local is not probe))
    else:

        class SyncProbe:
            def run(self, inputs):
                raise AssertionError("must not execute")

        services = replace(
            services,
            handles=tuple(
                replace(handle, local=SyncProbe()) if handle.local is probe else handle for handle in services.handles
            ),
        )
    monkeypatch.setattr(
        InvocationId, "new", staticmethod(lambda **kwargs: pytest.fail("identity allocated before service rejection"))
    )
    with pytest.raises(EffectRejected):
        asyncio.run(start_execution(admitted=execution, capabilities=execution.capabilities, services=services))
    assert submit.calls == generation.calls == readiness.calls == probe.calls == 0


def test_probe_success_outside_declared_continuations_stops_generation():
    execution, services, _, generation, _, probe = _ready_fixture("available")
    owner = next(item for item in execution.policies if item.delegated)
    changed = replace(
        owner,
        delegated=replace(
            owner.delegated, readiness=replace(owner.delegated.readiness, probe_outcomes=frozenset({"async_required"}))
        ),
    )
    execution = admit_execution_plan(
        context=execution.context,
        capabilities=execution.capabilities,
        policies=tuple(changed if item.node == owner.node else item for item in execution.policies),
        decisions=(),
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
    )
    result = asyncio.run(_execute(execution, services))
    assert probe.calls == 1 and generation.calls == 0
    assert all(state.complete for state in result.states)
    assert not _qualify(execution, result).qualified


@pytest.mark.parametrize("role, protected", [("readiness", False), ("generation", True)])
def test_unknown_usage_is_allowed_only_by_the_matching_admitted_role(role, protected):
    from anonymizer.engine.graph_sdk.requests import UnknownUsage
    from tests.graph_sdk.child_qualification_fixtures import result_with

    execution, services, *_ = _ready_fixture()
    result = asyncio.run(_execute(execution, services))
    request = next(row.request for row in result.requests.dispatches if row.child.role == role)
    receipt = replace(
        result.requests,
        settlements=tuple(
            replace(row, usage=UnknownUsage()) if row.request == request else row for row in result.requests.settlements
        ),
    )
    qualified = _qualify(execution, result_with(result, requests=receipt))
    assert bool(qualified.qualified) is protected
