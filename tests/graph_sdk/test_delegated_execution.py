# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual owner execution against frozen, independent finite expectations."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Literal

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.executor import (
    AssessmentFinding,
    DecisionLimits,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    PortArtifact,
    StopConfirmed,
    TextArtifactValue,
    TransportFailure,
    TransportSuccess,
)
from tests.graph_sdk.child_qualification_fixtures import child_fixture
from tests.graph_sdk.effects_admission_fixtures import _ZeroClock
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_adaptive_executor import _AssessingLocal

MANIFEST = json.loads((Path(__file__).parent / "reference/delegated_execution/manifest.json").read_text())


@dataclass
class _Transport:
    malformed: bool = False
    calls: int = 0

    async def dispatch(self, request: DispatchEnvelope) -> TransportFailure | TransportSuccess:
        self.calls += 1
        settlement = ExternalSettlement(
            request=request.request,
            disposition="completed",
            remote_stopped=True,
            usage=ExactUsage(input_units=1, output_units=1),
        )
        if self.malformed:
            return TransportFailure(failure="malformed_response", settlement=settlement)
        return TransportSuccess(
            results=(
                AssociationResult(
                    association=request.associations[0].association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            ),
            settlement=settlement,
        )

    async def cancel(self, request: object) -> StopConfirmed:
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


@dataclass
class _Submit:
    execution: ClassVar[Literal["same_loop"]] = "same_loop"
    slots: tuple[int, ...]
    calls: int = 0
    borrowed: DelegatedRequests | None = None
    bad: str | None = None
    rejected: EffectCode | None = None

    async def submit(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        self.calls += 1
        self.borrowed = requests
        if self.bad == "cross_loop":
            with pytest.raises(EffectRejected) as caught:
                await asyncio.to_thread(lambda: asyncio.run(requests.request(self.slots[0])))
            self.rejected = caught.value.code
        if self.bad == "foreign_slot":
            with pytest.raises(EffectRejected) as caught:
                await requests.request(999)
            self.rejected = caught.value.code
        if self.bad == "unready_slot":
            with pytest.raises(EffectRejected) as caught:
                await requests.request(self.slots[0])
            self.rejected = caught.value.code
            return LocalCompleted(
                results=(
                    AssociationResult(
                        association=inputs[0].association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                    ),
                )
            )
        first = await requests.request(self.slots[0])
        if self.bad in {"duplicate_slot", "exhausted_slots"}:
            with pytest.raises(EffectRejected) as caught:
                await requests.request(self.slots[0])
            self.rejected = caught.value.code
        if len(self.slots) == 2:
            evidence = requests.parser_rejected(first)
            await requests.request(self.slots[1], source_kind="correction", predecessor=first, evidence=evidence)
        outputs = ()
        if inputs[0].inputs:
            value = inputs[0].inputs[0]
            assert isinstance(value.value, TextArtifactValue)
            outputs = (
                PortArtifact(
                    port="context",
                    artifact_type=value.artifact_type,
                    artifact=None,
                    value=TextArtifactValue(text="candidate:" + value.value.text),
                ),
            )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=inputs[0].association, outcome="ok", outputs=outputs, consumed_context_ports=frozenset()
                ),
            )
        )


def _fixture(
    *,
    simple: bool = False,
    bad: str | None = None,
    parent_io: bool = False,
    usage: Literal["exact", "unknown_allowed"] = "unknown_allowed",
):
    execution, _ = child_fixture(simple=simple, stateless=True, unready=bad == "unready_slot", usage=usage)
    owner = next(policy for policy in execution.policies if policy.delegated is not None)
    assert owner.delegated is not None
    owner = replace(owner, delegated=replace(owner.delegated, followups=frozenset({"correction"})))
    execution = admit_execution_plan(
        context=execution.context,
        capabilities=execution.capabilities,
        policies=tuple(owner if item.node == owner.node else item for item in execution.policies),
        decisions=(),
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
    )
    if parent_io:
        from tests.graph_sdk.delegated_execution_fixtures import with_parent_publication

        execution = with_parent_publication(execution)
        owner = next(item for item in execution.policies if item.delegated is not None)
    assert owner.delegated is not None
    submit = _Submit(owner.delegated.child_slots, bad=bad)
    transports = [_Transport(malformed=not simple and index == 0) for index in range(len(submit.slots))]
    handles = []
    physical = {
        execution.context.prepared.reservation_recipe[slot].template: transport
        for slot, transport in zip(submit.slots, transports, strict=True)
    }
    for policy in execution.policies:
        implementation = policy.implementations[0]
        handles.append(
            ImplementationHandle(
                implementation=implementation.implementation,
                operation=implementation.capability.operation,
                configuration=implementation.configuration,
                local=_AssessingLocal(
                    finding=AssessmentFinding(status="satisfied", code="observed"),
                    evidence_port="assessment" if parent_io else "context",
                    alias_evidence=False,
                )
                if policy.kind == "local"
                else None,
                transport=physical[policy.node] if policy.kind == "external" else None,
                delegated=submit if policy.kind == "delegated" else None,
                resource=None,
            )
        )
    services = ExecutionServices(
        handles=tuple(handles),
        context_resources=(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1,
            max_runtime_artifacts=8,
            max_runtime_artifact_bytes=100,
            max_collection_items=0,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_ZeroClock(),
    )
    return execution, services, submit, transports


async def _execute(execution, services):
    running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
    return await running.wait()


@pytest.mark.parametrize("case", MANIFEST["traces"], ids=lambda case: case["name"])
def test_registered_delegation_qualifies_without_parent_charge(case) -> None:
    execution, services, submit, transports = _fixture(simple=case["name"] == "success")
    result = asyncio.run(_execute(execution, services))
    assert submit.calls == case["submissions"]
    assert sum(transport.calls for transport in transports) == len(result.requests.dispatches) == case["dispatches"]
    assert [item.category for item in result.requests.terminals] == case["physical_terminals"]
    leaves = [item.child.leaf for item in result.requests.dispatches]
    assert len(set(leaves)) == len(leaves)
    assert all(leaf is not None and any(item.attempt == leaf for item in result.record.terminals) for leaf in leaves)
    assert len(result.record.terminals) == case["leaves"] + 2
    assert all(state.complete for state in result.states)
    assert not result.requests.defects and not result.requests.remote_outstanding
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.record.statuses[0].qualification == "met"
    assert not output.targets[0].withholding
    assert submit.calls == 1 and sum(item.calls for item in transports) == case["dispatches"]


@pytest.mark.parametrize("bad", ["duplicate_slot", "foreign_slot", "exhausted_slots"])
def test_borrowed_slots_reject_illegal_use_without_extra_dispatch(bad: str) -> None:
    execution, services, submit, transports = _fixture(simple=True, bad=bad)
    result = asyncio.run(_execute(execution, services))
    assert submit.rejected == (EffectCode.FOREIGN_OWNER if bad == "foreign_slot" else EffectCode.DUPLICATE)
    assert len(result.requests.dispatches) == transports[0].calls == 1
    assert len(result.record.terminals) == 3


def test_borrowed_owner_cannot_escape_its_loop_or_lifetime() -> None:
    execution, services, submit, _ = _fixture(simple=True)
    asyncio.run(_execute(execution, services))
    assert submit.borrowed is not None
    with pytest.raises(EffectRejected):
        asyncio.run(submit.borrowed.request(submit.slots[0]))


@pytest.mark.parametrize("boundary", ["before_driver", "before_task_entry", "reject_dispatch", "after_dispatch"])
def test_actual_owner_cancel_or_rejection_has_no_phantom_leaf(monkeypatch, boundary: str) -> None:
    from anonymizer.engine.graph_sdk._request_driver import _RequestAuthority
    from anonymizer.engine.graph_sdk.requests import Dispatch

    execution, services, submit, transports = _fixture(simple=True)
    running = []
    apply = _RequestAuthority.apply
    create_task = asyncio.create_task

    def apply_at_boundary(authority, event):
        if isinstance(event, Dispatch) and boundary == "reject_dispatch":
            raise EffectRejected(EffectCode.CONTRADICTORY)
        state = apply(authority, event)
        if isinstance(event, Dispatch) and boundary == "after_dispatch":
            running[0].request_cancel()
        return state

    def schedule(coro, *args, **kwargs):
        task = create_task(coro, *args, **kwargs)
        if (
            getattr(coro, "cr_code", None) is not None
            and coro.cr_code.co_name == "call"
            and boundary == "before_task_entry"
        ):
            running[0].request_cancel()
        return task

    monkeypatch.setattr(_RequestAuthority, "apply", apply_at_boundary)
    monkeypatch.setattr(asyncio, "create_task", schedule)

    async def run():
        value = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
        running.append(value)
        if boundary == "before_driver":
            value.request_cancel()
        return await value.wait()

    result = asyncio.run(run())
    sent = int(boundary == "after_dispatch")
    assert len(result.requests.dispatches) == transports[0].calls == sent
    physical_nodes = {execution.context.prepared.reservation_recipe[index].template for index in submit.slots}
    physical_keys = {
        entry.activation for state in result.states for entry in state.entries if entry.template in physical_nodes
    }
    terminals = [item for item in result.record.terminals if item.activation in physical_keys]
    assert len(terminals) == 1
    assert sum(item.attempt is not None for item in terminals) == sent
    assert all(state.complete for state in result.states)
    assert not result.requests.local_in_flight


@pytest.mark.parametrize(
    "defect", ["missing_handle", "blocking_transport", "followup_route", "owned_resource", "cross_loop_profile"]
)
def test_unsupported_delegation_rejects_before_invocation(monkeypatch, defect: str) -> None:
    from anonymizer.engine.graph_sdk.blocking import BlockingAffinity, BlockingPool, BlockingTransport
    from anonymizer.engine.graph_sdk.resources import ResourceLease
    from anonymizer.graph._values import InvocationId

    execution, services, _, _ = _fixture(simple=True)
    if defect == "cross_loop_profile":
        delegate = next(item.delegated for item in services.handles if item.delegated is not None)
        monkeypatch.setattr(delegate, "execution", "cross_loop")
    elif defect == "missing_handle":
        services = replace(services, handles=tuple(item for item in services.handles if item.delegated is None))
    elif defect == "followup_route":
        owner = next(item for item in execution.policies if item.delegated is not None)
        owner = replace(owner, delegated=replace(owner.delegated, followups=frozenset({"failover"})))
        execution = admit_execution_plan(
            context=execution.context,
            capabilities=execution.capabilities,
            policies=tuple(owner if item.node == owner.node else item for item in execution.policies),
            decisions=(),
            assessment_productions=execution.assessment_productions,
            assessment_limits=execution.assessment_limits,
        )
    else:
        handle = next(item for item in services.handles if item.transport is not None)
        if defect == "blocking_transport":
            pool = BlockingPool(max_workers=1)
            transport = BlockingTransport(
                callback=lambda _: None,
                pool=pool,
                affinity=BlockingAffinity(cross_thread=True, concurrency_safe=True, close_on_event_loop=False),
            )
            changed = replace(handle, transport=transport)
        else:
            changed = replace(
                handle,
                resource=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=handle.transport),
            )
        services = replace(services, handles=tuple(changed if item is handle else item for item in services.handles))

    def forbidden(*args, **kwargs):
        raise AssertionError("invocation created before service rejection")

    monkeypatch.setattr(InvocationId, "new", forbidden)
    with pytest.raises(EffectRejected):
        asyncio.run(_execute(execution, services))
    if defect == "blocking_transport":
        asyncio.run(pool.close())


def test_unready_declared_slot_cannot_start_or_charge() -> None:
    execution, services, submit, transports = _fixture(simple=True, bad="unready_slot")
    result = asyncio.run(_execute(execution, services))
    assert submit.rejected == EffectCode.REQUEST_CAUSALITY
    assert not result.requests.dispatches and transports[0].calls == 0
    assert all(state.complete for state in result.states)
    assert [item.attempt for item in result.record.terminals].count(None) == 1


@pytest.mark.parametrize("reason", ["scope_cancel", "stale_parser_proof"])
def test_authoritative_dispatch_rechecks_after_start_preflight(monkeypatch, reason: str) -> None:
    from anonymizer.engine.graph_sdk._execution_delegated import _DelegatedOwner
    from anonymizer.engine.graph_sdk.requests import AcceptResult, ScopeCancel

    execution, services, _, transports = _fixture(simple=reason == "scope_cancel")
    prepare = _DelegatedOwner._prepare
    preflights = []

    def invalidate(owner, leaf, attempt):
        commit = prepare(owner, leaf, attempt)
        authority = owner.runtime.request_authority
        if reason == "scope_cancel":
            authority.apply(ScopeCancel())
        elif authority.state.dispatches:
            first = authority.state.dispatches[0]
            authority.apply(
                AcceptResult(
                    request=first.request,
                    results=(
                        AssociationResult(
                            association=owner._source_job().association,
                            outcome="ok",
                            outputs=(),
                            consumed_context_ports=frozenset(),
                        ),
                    ),
                )
            )
        preflights.append(leaf.activation)
        return commit

    monkeypatch.setattr(_DelegatedOwner, "_prepare", invalidate)
    result = asyncio.run(_execute(execution, services))
    expected = int(reason == "stale_parser_proof")
    assert sum(item.calls for item in transports) == len(result.requests.dispatches) == expected
    rejected = preflights[-1]
    terminal = next(item for item in result.record.terminals if item.activation == rejected)
    assert terminal.attempt is None and terminal.category == "blocked"
    assert all(state.complete for state in result.states)


def test_live_borrowed_owner_rejects_cross_loop_before_dispatch() -> None:
    execution, services, submit, transports = _fixture(simple=True, bad="cross_loop")
    result = asyncio.run(_execute(execution, services))
    assert submit.rejected == EffectCode.UNSUPPORTED
    assert len(result.requests.dispatches) == transports[0].calls == 1


@pytest.mark.parametrize("simple", [True, False])
def test_parent_publishes_candidate_then_local_assessment_qualifies(simple: bool) -> None:
    table = json.loads((Path(__file__).parent / "reference/delegated_execution/publication.json").read_text())
    execution, services, submit, transports = _fixture(simple=simple, parent_io=True)
    result = asyncio.run(_execute(execution, services))
    owner = next(item.node for item in execution.policies if item.kind == "delegated")
    owner_outputs = [port for port in result.ports if port.node == owner and port.port == "context"]
    assert len(owner_outputs) == 1
    candidate = owner_outputs[0].artifact
    assert dict(result.artifacts)[candidate] == TextArtifactValue(text=table["parent_output"])
    assert result.final_outputs[0].candidate.artifact == candidate
    assert result.assessments[0].evidence_artifact != candidate
    admitted, current, submissions = _inputs(execution, result)
    qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(qualified.qualified) == table["qualified_count"]
    assert not qualified.targets[0].withholding
    assert submit.calls == 1
    assert sum(item.calls for item in transports) == len(result.requests.dispatches) == (1 if simple else 2)


def test_missing_parent_input_closes_owned_slots_without_submission(monkeypatch) -> None:
    from anonymizer.engine.graph_sdk import _execution_runtime
    from anonymizer.engine.graph_sdk._effect_values import reject

    execution, services, submit, transports = _fixture(simple=True, parent_io=True)
    owner = next(item.node for item in execution.policies if item.kind == "delegated")
    resolve = _execution_runtime._operation_inputs

    def missing(admitted, target, node, *args):
        if node == owner:
            reject(EffectCode.MISSING)
        return resolve(admitted, target, node, *args)

    monkeypatch.setattr(_execution_runtime, "_operation_inputs", missing)
    result = asyncio.run(_execute(execution, services))
    assert submit.calls == sum(item.calls for item in transports) == len(result.requests.dispatches) == 0
    assert all(state.complete for state in result.states)
    assert all(item.attempt is None for item in result.record.terminals)


def test_parser_rejection_cannot_reclassify_successful_physical_terminal() -> None:
    table = json.loads((Path(__file__).parent / "reference/delegated_execution/parser.json").read_text())
    execution, services, submit, transports = _fixture(parent_io=True)
    transports[0].malformed = False
    result = asyncio.run(_execute(execution, services))
    assert [item.category for item in result.requests.terminals] == table["physical_terminals"]
    assert submit.calls == 1 and sum(item.calls for item in transports) == len(result.requests.dispatches) == 1
    assert all(state.complete for state in result.states)
    assert not result.requests.defects
    owner = next(item.node for item in execution.policies if item.kind == "delegated")
    owner_key = next(item.activation for state in result.states for item in state.entries if item.template == owner)
    assert next(item.category for item in result.record.terminals if item.activation == owner_key) == "failure"
