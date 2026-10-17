# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Physical recovery and retained-receipt qualification discriminators."""

from __future__ import annotations

import asyncio

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
