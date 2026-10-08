# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen shared-request cases exercised through the real executor."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import PreparationConfiguration
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    DispatchEnvelope,
    PhysicalRequestPolicy,
    SemanticAssociation,
    StopConfirmed,
    TransportFailure,
    TransportSuccess,
)
from anonymizer.graph._values import ActivationKey, InvocationId, PlanId, TaskAttemptId
from tests.graph_sdk.test_effects_production_conformance import (
    CORPUS,
    _assessment_limits,
    _normalize,
    _valid_runtime_rows,
    _ZeroClock,
)
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow

CASES = tuple(
    case
    for case in json.loads(CORPUS.read_bytes())
    if case["case_id"].startswith("bridges/shared_request_")
    or case["case_id"]
    in {
        "bridges/retry_uses_latest_physical_request",
        "bridges/inconsistent_request_cannot_emit_success",
        "bridges/valid_request_cannot_emit_inconsistent",
    }
)


@dataclass
class _SharedTransport:
    events: list[dict[str, Any]]
    target_count: int
    tasks: dict[str, SemanticAssociation] = field(default_factory=dict)
    envelopes: list[DispatchEnvelope] = field(default_factory=list)

    async def dispatch(self, request: DispatchEnvelope) -> TransportSuccess | TransportFailure:
        event = self.events[len(self.envelopes)]
        self.envelopes.append(request)
        assert len(request.associations) == self.target_count
        for index, item in enumerate(request.associations):
            assert isinstance(item.association, SemanticAssociation)
            self.tasks[f"T{index}"] = item.association
        original = self.tasks["T0"].task.activation
        self.tasks["T2"] = SemanticAssociation(task=TaskAttemptId.new(activation=original))
        foreign = ActivationKey(
            invocation=InvocationId.new(plan=PlanId.new()), occurrence=0, parent=None, iteration=None
        )
        self.tasks["X0"] = SemanticAssociation(task=TaskAttemptId.new(activation=foreign))
        if event["kind"] == "failure":
            return TransportFailure(failure=event["failure"], settlement=None)
        return TransportSuccess(
            results=tuple(
                AssociationResult(
                    association=self.tasks[name],
                    outcome=event["outcomes"].get(name, "ok"),
                    outputs=(),
                    consumed_context_ports=frozenset(),
                )
                for name in event["returned"]
            ),
            settlement=None,
        )

    async def cancel(self, request: object) -> StopConfirmed:
        raise AssertionError("shared result cases do not cancel")

    async def close(self) -> None:
        raise AssertionError("stateless transport has no owned resource")


@pytest.mark.parametrize("case", CASES, ids=lambda case: cast(str, case["case_id"]))
def test_frozen_shared_case_through_executor(case: dict[str, Any]) -> None:
    asyncio.run(_assert_shared_case(case))


async def _assert_shared_case(case: dict[str, Any]) -> None:
    declaration = case["declaration"]
    target_count = len(next(event["associations"] for event in case["events"] if event["kind"] == "reserve"))
    policies = {
        name: PhysicalRequestPolicy(
            visibility="dispatch_and_settlement",
            pre_dispatch_control="executor",
            retry_owner=value["retry_owner"],
            replay=value["replay"],
            max_attempts=value["max_attempts"],
        )
        for name, value in declaration["policies"].items()
    }
    request_policy = policies["P0"]
    workflow, node, _ = _workflow(requests=request_policy.max_attempts)
    capability = replace(
        _capability(workflow, external=True),
        attribution="keyed_shared_request",
        resource_lifetime="stateless",
        max_physical_requests_per_activation=request_policy.max_attempts,
    )
    prepared = _prepare(
        data=_data(target_count),
        workflow=workflow,
        capability=capability,
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=declaration["hard_limit"],
        ),
        limits=_limits(),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=request_policy,
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=node,
                kind="external",
                request=request_policy,
                safe_detachment="forbidden",
                implementations=(implementation,),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_valid_runtime_rows("external", frozenset({"ok"})),
            ),
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    transport = _SharedTransport(
        events=[event for event in case["events"] if event["kind"] in {"result", "failure"}], target_count=target_count
    )
    result = await (
        await start_execution(
            admitted=admitted,
            capabilities=(capability,),
            services=ExecutionServices(
                handles=(
                    ImplementationHandle(
                        implementation=capability.implementation,
                        operation=capability.operation,
                        configuration=capability.configuration,
                        local=None,
                        transport=transport,
                        resource=None,
                    ),
                ),
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=0,
                    max_remote_outstanding=1,
                    max_runtime_artifacts=0,
                    max_runtime_artifact_bytes=0,
                    max_collection_items=0,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_ZeroClock(),
            ),
        )
    ).wait()
    assert len(transport.envelopes) == len(transport.events)
    assert [envelope.purpose for envelope in transport.envelopes] == [
        event["purpose"] for event in case["events"] if event["kind"] == "reserve"
    ]
    actual = _normalize(
        result.requests,
        transport.tasks,
        {f"R{index}": envelope.request for index, envelope in enumerate(transport.envelopes)},
        policies,
    )
    if case["expected"]["status"] == "rejected":
        # The public executor derives the bridge condition. There is no API to
        # inject the contradictory bridge_condition used by the reducer case.
        assert case["expected"]["code"] == "request_causality"
        inconsistent = case["case_id"] == "bridges/inconsistent_request_cannot_emit_success"
        category = "inconsistent" if inconsistent else "success"
        assert actual["terminals"] == {"R0": category}
        assert len(result.record.terminals) == 2
        assert all(terminal.category == category for terminal in result.record.terminals)
        assert all(state.complete for state in result.states)
        assert not result.artifacts and not result.ports and not result.final_outputs
        return
    expected = case["expected"]["state"]
    for key, value in actual.items():
        assert value == expected[key], key
    names = {association.task: name for name, association in transport.tasks.items()}
    assert {
        names[item.attempt]: item.category for item in result.record.terminals if item.attempt is not None
    } == expected["tasks"]
    assert all(state.complete for state in result.states)
    # Task/request joins are derived from the dispatched semantic keys, not result order.
    task_requests = {
        names[item.attempt]: f"R{index}"
        for index, envelope in enumerate(transport.envelopes)
        for item in result.record.terminals
        if item.attempt is not None
        and any(
            row.association.task == item.attempt
            for row in envelope.associations
            if isinstance(row.association, SemanticAssociation)
        )
    }
    assert task_requests == expected["task_requests"]
    assert not result.artifacts and expected["artifacts"] == []
    assert not result.ports and not result.final_outputs and not result.assessments and not result.provenance
    assert not result.cleanup and expected["cleanup"] == {}
    assert not result.cleanup_associations and expected["resources"] == {} and expected["resource_count"] == 0
    assert not result.pending_decisions and expected["decisions"] == {}
    assert context.bound_context is None
    assert expected["binding_declarations"] == expected["binding_sources"] == {}
    assert expected["binding_terminal"] is None
    assert len(result.record.terminals) == target_count and all(
        item.attempt is not None for item in result.record.terminals
    )
    assert expected["closed_unstarted"] == {}
    assert {
        name: {
            "retry_owner": policy.retry_owner,
            "replay": policy.replay,
            "max_attempts": policy.max_attempts,
            "failover_failures": ["permanent", "implementation_exception"],
        }
        for name, policy in policies.items()
    } == expected["policies"]

    assert set(expected) == set(actual) | {
        "tasks",
        "task_requests",
        "artifacts",
        "cleanup",
        "resources",
        "resource_count",
        "decisions",
        "binding_declarations",
        "binding_sources",
        "binding_terminal",
        "closed_unstarted",
        "policies",
    }
    emitted = {event["task"]: event for event in case["events"] if event["kind"] == "bridge_emit"}
    for terminal in result.record.terminals:
        assert terminal.attempt is not None
        entry = next(
            entry
            for state in result.states
            for entry in state.entries
            if entry.activation == terminal.attempt.activation
        )
        assert entry.outcome == emitted[names[terminal.attempt]]["outcome"]
