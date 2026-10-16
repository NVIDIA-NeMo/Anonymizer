# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent protected qualification controls for retained child recipes."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from anonymizer.engine.graph_sdk.executor import (
    ExecutionLimits,
    ExecutionServices,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    ExactUsage,
    PhysicalRequestId,
    RequestChainId,
    RequestPolicyBinding,
    SemanticAssociation,
    UnknownUsage,
)
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import ActivationKey, ContractViolation, InvocationId, TaskAttemptId
from tests.graph_sdk.child_qualification_fixtures import child_fixture, result_with
from tests.graph_sdk.effects_admission_fixtures import _ZeroClock
from tests.graph_sdk.qualification_fixtures import _inputs

ORACLE = Path(__file__).parent / "reference/request_child_receipts/cases.json"
CASES = json.loads(ORACLE.read_text())


def test_oracle_frozen_before_comparison() -> None:
    digest = hashlib.sha256(ORACLE.read_bytes()).hexdigest()
    assert digest in (ORACLE.parent / "README.md").read_text()
    assert len(CASES) == 15


@pytest.mark.parametrize("case", CASES)
def test_full_protected_child_receipt_oracle(case: str) -> None:
    execution, result = child_fixture(
        failover=case == "exhaustion_failover",
        usage="exact" if case in {"unknown_under_exact", "receipt_permission", "exact_usage"} else "unknown_allowed",
        exact=case == "exact_usage",
    )
    receipt = result.requests
    first, last = receipt.dispatches[0], receipt.dispatches[-1]
    assert first.child is not None and last.child is not None and first.child.leaf is not None
    assert execution.context.prepared.configuration.purpose == "protection"
    assert all(state.complete for state in result.states)
    assert result.assessments[0].finding.status == "satisfied"
    try:
        if case == "foreign_leaf":
            foreign = TaskAttemptId.new(
                activation=ActivationKey(
                    invocation=result.record.invocation, occurrence=999, parent=None, iteration=None
                )
            )
            receipt = replace(receipt, dispatches=(replace(first, child=replace(first.child, leaf=foreign)), last))
        elif case == "missing_leaf":
            receipt = replace(receipt, dispatches=(replace(first, child=replace(first.child, leaf=None)), last))
        elif case == "duplicate_leaf":
            receipt = replace(
                receipt, dispatches=(first, replace(last, child=replace(last.child, leaf=first.child.leaf)))
            )
        elif case == "unstarted_leaf":
            state = result.states[0]
            result = result_with(
                result,
                states=(
                    replace(
                        state,
                        entries=frozenset(
                            replace(entry, status="unstarted", outcome=None)
                            if entry.activation == first.child.leaf.activation
                            else entry
                            for entry in state.entries
                        ),
                    ),
                ),
            )
        elif case == "wrong_predecessor":
            receipt = replace(
                receipt,
                dispatches=(first, replace(last, child=replace(last.child, predecessor=last.request, evidence=None))),
            )
        elif case == "forged_evidence":
            receipt = replace(receipt, child_evidence=())
        elif case == "cross_chain_recovery":
            chain = replace(receipt.chains[0], chain=RequestChainId.new(scope=receipt.scope))
            receipt = replace(
                receipt,
                chains=(*receipt.chains, chain),
                dispatches=(
                    first,
                    replace(
                        last,
                        purpose="initial",
                        child=replace(
                            last.child, chain=chain.chain, source_kind="initial", predecessor=None, evidence=None
                        ),
                    ),
                ),
            )
        elif case == "remote_unknown":
            receipt = replace(
                receipt,
                settlements=(
                    replace(receipt.settlements[0], disposition="unknown", remote_stopped=None),
                    *receipt.settlements[1:],
                ),
            )
        elif case == "missing_terminal":
            result = result_with(
                result,
                record=replace(
                    result.record,
                    terminals=tuple(item for item in result.record.terminals if item.attempt != first.child.leaf),
                ),
            )
        elif case == "leaf_without_dispatch":
            receipt = replace(
                receipt,
                dispatches=(last,),
                terminals=receipt.terminals[1:],
                settlements=receipt.settlements[1:],
                dispatched_count=1,
            )
        result = result_with(result, requests=receipt)
        admitted, current, submissions = _inputs(execution, result)
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert bool(output.qualified) == CASES[case]
        if CASES[case]:
            assert len(output.qualified) == 1
            assert output.record.statuses[0].qualification == "met"
            assert not output.targets[0].withholding
            assert len(submissions) == 1
            if case != "exact_usage":
                assert all(isinstance(item.usage, UnknownUsage) for item in result.requests.settlements)
    except (EffectRejected, ContractViolation):
        assert not CASES[case]


def readmit(execution, policy):
    return admit_execution_plan(
        context=execution.context,
        capabilities=execution.capabilities,
        policies=tuple(policy if item.node == policy.node else item for item in execution.policies),
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
    )


@pytest.mark.parametrize(
    "change", ["opaque", "wrong_slot", "owner_slot", "missing_slot", "ordinary_retry", "receipt_routes"]
)
def test_delegated_structural_admission_is_closed(change: str) -> None:
    execution, _ = child_fixture()
    policy = next(item for item in execution.policies if item.kind == "delegated")
    assert policy.delegated is not None
    contract = policy.delegated
    with pytest.raises(EffectRejected):
        if change == "opaque":
            policy = replace(policy, delegated=replace(contract, mode="opaque"))
        elif change == "wrong_slot":
            policy = replace(policy, delegated=replace(contract, child_slots=(99, 100)))
        elif change == "owner_slot":
            owner_slot = next(
                slot.index for slot in execution.context.prepared.reservation_recipe if slot.template == policy.node
            )
            policy = replace(policy, delegated=replace(contract, child_slots=(owner_slot, contract.child_slots[0])))
        elif change == "missing_slot":
            policy = replace(policy, delegated=replace(contract, child_slots=(2,)))
        elif change == "ordinary_retry":
            policy = replace(policy, kind="external", delegated=None)
        else:
            policy = replace(
                policy, delegated=replace(contract, routes=(replace(contract.routes[0], parse_attempts=3),))
            )
        readmit(execution, policy)


def test_delegated_service_gate_precedes_identity_and_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    execution, _ = child_fixture()

    def forbidden(*args: object, **kwargs: object):
        raise AssertionError("identity or task allocated before delegated rejection")

    services = ExecutionServices(
        handles=(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1,
            max_runtime_artifacts=8,
            max_runtime_artifact_bytes=100,
            max_collection_items=0,
        ),
        clock=_ZeroClock(),
    )

    async def attempt():
        monkeypatch.setattr(InvocationId, "new", forbidden)
        monkeypatch.setattr(asyncio, "create_task", forbidden)
        with pytest.raises(EffectRejected, match="unsupported"):
            await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)

    asyncio.run(attempt())


def test_child_recovery_does_not_hide_provenance_defect() -> None:
    execution, result = child_fixture()
    result = result_with(result, provenance=result.provenance[:1])
    with pytest.raises(EffectRejected):
        admitted, current, submissions = _inputs(execution, result)
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)


def test_child_receipt_bound_precedes_indexes() -> None:
    execution, result = child_fixture()
    result = result_with(result, requests=replace(result.requests, chains=result.requests.chains * 100))
    with pytest.raises(EffectRejected, match="limit_exceeded"):
        admitted, current, submissions = _inputs(execution, result)
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)


def test_interleaved_chains_use_their_explicit_predecessors() -> None:
    execution, result = child_fixture(interleaved=True)
    a, b, c, d = result.requests.dispatches
    assert a.child is not None and b.child is not None and c.child is not None and d.child is not None
    assert c.child.predecessor == a.request and d.child.predecessor == b.request
    assert a.child.chain != b.child.chain
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1 and not output.targets[0].withholding


@pytest.mark.parametrize("outcome", ["NOT_DECLARED", "alternate"])
def test_child_success_requires_declared_and_canonical_outcome(outcome: str) -> None:
    execution, result = child_fixture(alternate_outcome=True)
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    owner = next(policy for policy in execution.policies if policy.kind == "delegated")
    assert (outcome in owner.result_outcomes) == (outcome == "alternate")
    terminal = result.requests.terminals[-1]
    changed = replace(terminal, results=(replace(terminal.results[0], outcome=outcome),))
    result = result_with(
        result, requests=replace(result.requests, terminals=(*result.requests.terminals[:-1], changed))
    )
    admitted, current, submissions = _inputs(execution, result)
    with pytest.raises(EffectRejected, match="contradictory"):
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)


@pytest.mark.parametrize("owned_slot", [True, False])
@pytest.mark.parametrize("prepend", [True, False])
def test_ordinary_dispatch_cannot_reuse_delegated_physical_slot(owned_slot: bool, prepend: bool) -> None:
    execution, result = child_fixture(exact=True, ordinary_slot=True)
    receipt = result.requests
    last = receipt.dispatches[-1]
    assert last.child is not None and last.child.leaf is not None
    declared = next(policy.delegated for policy in execution.policies if policy.kind == "delegated")
    assert declared is not None
    entries = {entry.activation: entry for state in result.states for entry in state.entries}
    task = (
        last.child.leaf
        if owned_slot
        else next(
            terminal.attempt
            for terminal in result.record.terminals
            if terminal.attempt is not None
            and terminal.activation.occurrence not in declared.child_slots
            and next(
                policy for policy in execution.policies if policy.node == entries[terminal.activation].template
            ).kind
            == "external"
        )
    )
    policy = next(policy.request for policy in execution.policies if policy.node == entries[task.activation].template)
    assert policy is not None
    association = SemanticAssociation(task=task)
    request = PhysicalRequestId.new(scope=receipt.scope)
    extra = replace(
        last, request=request, purpose="initial", associations=frozenset({association}), policy=policy, child=None
    )
    terminal = replace(
        receipt.terminals[-1],
        request=request,
        results=(replace(receipt.terminals[-1].results[0], association=association),),
    )
    settlement = replace(receipt.settlements[-1], request=request, usage=ExactUsage(input_units=1, output_units=1))
    receipt = replace(
        receipt,
        dispatches=(extra, *receipt.dispatches) if prepend else (*receipt.dispatches, extra),
        terminals=(*receipt.terminals, terminal),
        settlements=(*receipt.settlements, settlement),
        dispatched_count=3,
        policies=receipt.policies | {policy},
        bindings=receipt.bindings
        | {RequestPolicyBinding.create(association=association, policies=frozenset({policy}))},
    )
    result = result_with(result, requests=receipt)
    admitted, current, submissions = _inputs(execution, result)
    if owned_slot:
        with pytest.raises(EffectRejected, match="contradictory"):
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    else:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert len(output.qualified) == 1 and not output.targets[0].withholding
        assert len(result.requests.dispatches) == 3
