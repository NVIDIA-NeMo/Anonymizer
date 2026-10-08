# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource lifetime eligibility against frozen request histories."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    ExecutionImplementation,
    ImplementationHandle,
    OperationExecutionPolicy,
    _cleanup_execution,
    admit_execution_plan,
)
from anonymizer.engine.graph_sdk.requests import (
    InvocationRequestScope,
    PhysicalRequestId,
    SemanticAssociation,
    initialize_requests,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph._values import ActivationKey, InvocationId, PlanId, TaskAttemptId
from tests.graph_sdk.test_effects_production_conformance import (
    CORPUS,
    _apply,
    _assessment_limits,
    _normalize,
    _policy,
    _valid_runtime_rows,
)
from tests.graph_sdk.test_preparation import _capability, _data, _prepare, _workflow

CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if case["case_id"]
    in {
        "resources/sdk_remote_waits",
        "resources/sdk_safe_detach",
        "resources/local_inflight_waits",
        "resources/trusted_stop_then_close",
    }
)


@dataclass
class _Resource:
    closes: int = 0

    async def close(self) -> None:
        self.closes += 1


@pytest.mark.parametrize("case", CASES, ids=lambda case: cast(str, case["case_id"]))
def test_frozen_resource_eligibility(case: dict[str, Any]) -> None:
    asyncio.run(_assert_resource_eligibility(case))


async def _assert_resource_eligibility(case: dict[str, Any]) -> None:
    declaration = case["declaration"]
    policies = {name: _policy(value) for name, value in declaration["policies"].items()}
    workflow, node, _ = _workflow(requests=policies["P0"].max_attempts)
    capability = replace(
        _capability(workflow, external=True), max_physical_requests_per_activation=policies["P0"].max_attempts
    )
    prepared = _prepare(data=_data(1), workflow=workflow, capability=capability)
    context = admit_context_plan(prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=())
    resource_event = case["events"][0]
    resource = _Resource()
    lease = ResourceLease.create(
        owner=resource_event["owner"], safe_detachment=resource_event["safe_detachment"], handle=resource
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=policies["P0"],
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=node,
                kind="external",
                request=policies["P0"],
                safe_detachment=lease.safe_detachment,
                implementations=(implementation,),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_valid_runtime_rows("external", frozenset({"ok"})),
            ),
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    handle = ImplementationHandle(
        implementation=capability.implementation,
        operation=capability.operation,
        configuration=capability.configuration,
        local=None,
        transport=None,
        resource=lease,
    )
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    tasks = {
        "T0": SemanticAssociation(
            task=TaskAttemptId.new(
                activation=ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
            )
        )
    }
    requests = {"R0": PhysicalRequestId.new(scope=scope)}
    state = initialize_requests(
        scope=scope, hard_limit=declaration["hard_limit"], policies=frozenset(policies.values())
    )
    for event in case["events"][1:-1]:
        state = _apply(state, event, tasks, requests, policies)
    expected = case["expected"]["state"]
    for key, value in _normalize(state, tasks, requests, policies).items():
        assert value == expected[key], key
    cleanup, associations = await _cleanup_execution(admitted, (handle,), {}, state, {requests["R0"]: lease.resource})
    # The frozen resource model records completed close attempts; the executor
    # additionally reports left_open when eligibility retains an SDK lease.
    completed = {"Q0": fact.disposition for fact in cleanup if fact.disposition != "left_open"}
    assert completed == expected["cleanup"]
    assert resource.closes == len(expected["cleanup"])
    assert len(cleanup) == len(associations) == expected["resource_count"] == 1
    assert cleanup[0].resource == associations[0].resource == lease.resource
    assert cleanup[0].owner == expected["resources"]["Q0"]["owner"]
    assert lease.safe_detachment == expected["resources"]["Q0"]["safe_detachment"]
    assert associations[0].purpose == "accounting"
    assert associations[0].targets == prepared.data.targets
