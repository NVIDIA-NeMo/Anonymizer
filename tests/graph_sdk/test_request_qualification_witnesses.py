# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical recovery and retained-receipt qualification discriminators."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource, _inputs, _RecoveringTransport


def test_real_failover_to_distinct_implementation_qualifies() -> None:
    transport = _RecoveringTransport(failure="permanent")
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease), external_failover=True))
    policy = next(item for item in execution.policies if item.kind == "external")
    assert len({item.implementation for item in policy.implementations}) == 2
    initial, failover = result.requests.dispatches
    assert (initial.purpose, failover.purpose) == ("initial", "failover")
    assert initial.request != failover.request
    assert initial.associations == failover.associations and initial.policy == failover.policy
    assert [item.category for item in result.requests.terminals] == ["failure", "success"]
    assert result.requests.terminals[0].failure == "permanent"
    assert len(result.requests.settlements) == 2
    assert not result.requests.remote_outstanding and not result.requests.local_in_flight
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == len(submissions) == 1
    assert output.record.statuses[0].qualification == "met"
    assert not output.targets[0].withholding
    assert transport.calls == 2 and transport.closes == 1
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    assert transport.calls == 2 and transport.closes == 1


@pytest.mark.parametrize(
    "mutation", ["changed-policy", "permanent-retry", "malformed-retry", "retryable-failover", "latest-failure"]
)
def test_retained_recovery_receipt_cannot_hide_illegal_or_failed_followup(mutation: str) -> None:
    transport = _RecoveringTransport(failure="retryable")
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    receipt = result.requests
    dispatch = receipt.dispatches[1]
    terminal = receipt.terminals[1] if mutation == "latest-failure" else receipt.terminals[0]
    changed_policy = replace(dispatch.policy, max_attempts=3)
    changes = (
        (
            (dispatch, "policy", changed_policy),
            (receipt, "policies", receipt.policies | {changed_policy}),
            *((binding, "policies", binding.policies | {changed_policy}) for binding in receipt.bindings),
        )
        if mutation == "changed-policy"
        else ((dispatch, "purpose", "failover"),)
        if mutation == "retryable-failover"
        else ((terminal, "category", "failure"), (terminal, "failure", "permanent"), (terminal, "results", ()))
        if mutation == "latest-failure"
        else ((terminal, "failure", "permanent" if mutation == "permanent-retry" else "malformed_response"),)
    )
    originals = tuple((owner, name, getattr(owner, name)) for owner, name, _ in changes)
    try:
        for owner, name, value in changes:
            object.__setattr__(owner, name, value)
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        assert output.record.statuses[0].qualification == "unmet"
        assert output.targets[0].withholding == frozenset({"request_accounting"})
        assert result.assessments[0].finding.status == "satisfied"
        assert result.final_outputs
    finally:
        for owner, name, value in originals:
            object.__setattr__(owner, name, value)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
    assert transport.calls == 2 and transport.closes == 1


def test_retained_retry_cannot_change_to_another_executed_target_association() -> None:
    transport = _RecoveringTransport(failure="retryable")
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease), target_count=2))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    initial, retry, other = result.requests.dispatches
    assert initial.associations == retry.associations != other.associations
    original = retry.associations
    object.__setattr__(retry, "associations", other.associations)
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        assert all(item.withholding == frozenset({"request_accounting"}) for item in output.targets)
        assert all(item.qualification == "unmet" for item in output.record.statuses)
    finally:
        object.__setattr__(retry, "associations", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
    assert transport.calls == 3 and transport.closes == 1


@pytest.mark.parametrize("purpose", ["initial_binding", "repair"])
def test_fresh_purpose_receipt_classification_without_claiming_physical_selector(purpose: str) -> None:
    # This is a retained-receipt reconciliation witness, not execution of these purposes.
    transport = _RecoveringTransport()
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    (dispatch,) = result.requests.dispatches
    assert dispatch.purpose == "initial"
    object.__setattr__(dispatch, "purpose", purpose)
    try:
        assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
        assert len(baseline.qualified) == 1
    finally:
        object.__setattr__(dispatch, "purpose", "initial")
    assert transport.calls == transport.closes == 1


def test_corrupted_three_attempt_history_cannot_reuse_an_earlier_success() -> None:
    # Actual A retry succeeds; actual B fails. Reattribute B's retained failed attempt
    # as an illegal later retry of A, reproducing the legacy success-then-failure shape.
    transport = _RecoveringTransport(failures=("retryable", None, "permanent"))
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease), target_count=2))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    released = baseline.qualified[0].target
    first, recovered, last = result.requests.dispatches
    assert first.associations == recovered.associations != last.associations
    assert [item.category for item in result.requests.terminals] == ["failure", "success", "failure"]
    assert len(result.requests.settlements) == 3
    original = last.associations, last.purpose
    object.__setattr__(last, "associations", first.associations)
    object.__setattr__(last, "purpose", "retry")
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        target = next(item for item in output.targets if item.target == released)
        assert target.withholding == frozenset({"request_accounting"})
        assert next(item for item in output.record.statuses if item.target == released).qualification == "unmet"
        assert result.assessments[0].finding.status == "satisfied"
        assert any(item.target == released for item in result.final_outputs)
    finally:
        object.__setattr__(last, "associations", original[0])
        object.__setattr__(last, "purpose", original[1])
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
    assert transport.calls == 3 and transport.closes == 1


def test_real_fresh_initial_binding_request_qualifies_with_exact_accounting() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="fresh-binding", revision=1))
    execution, result = asyncio.run(_execute_assessment(initial_resources=(_initial_resource(provider),)))
    bound = execution.context.bound_context
    assert bound is not None
    (dispatch,) = bound.receipt.requests.dispatches
    assert dispatch.purpose == "initial_binding"
    assert len(bound.receipt.requests.terminals) == len(bound.receipt.requests.settlements) == 1
    assert bound.receipt.requests.terminals[0].category == "success"
    assert not result.requests.dispatches
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.record.statuses[0].qualification == "met"
    assert not output.targets[0].withholding
    assert provider.calls == provider.closes == 1


def test_real_success_with_unknown_usage_remains_unknown_despite_satisfied_assessment() -> None:
    transport = _RecoveringTransport(unknown_usage=True)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease)))
    assert result.requests.terminals[0].category == "success"
    assert result.assessments[0].finding.status == "satisfied"
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert output.record.statuses[0].qualification == "unknown"
    assert output.targets[0].withholding == frozenset({"request_accounting"})
    assert transport.calls == transport.closes == 1
