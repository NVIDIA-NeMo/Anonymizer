# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-free integration tests for the local graph executor."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionDeclaration,
    DecisionLimits,
    DecisionOutcome,
    DecisionResponse,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    LocalDecisionWait,
    OperationExecutionPolicy,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, SemanticAssociation
from tests.graph_sdk.test_preparation import _capability, _data, _prepare, _workflow


@dataclass
class _Clock:
    value: int = 0

    def now_ns(self) -> int:
        return self.value


@dataclass
class _Uppercase:
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        assert len(request) == 1
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            )
        )


@dataclass
class _Decision:
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalDecisionWait:
        self.calls += 1
        artifact = request[0].inputs[0].artifact
        assert artifact is not None
        assert isinstance(request[0].association, SemanticAssociation)
        return LocalDecisionWait(association=request[0].association, artifact=artifact)


def _runtime_rows() -> tuple[RuntimeOutcome, ...]:
    rows = [RuntimeOutcome(condition="result", reported_outcome="ok", failure=None, outcome="ok", category="success")]
    rows.extend(
        RuntimeOutcome(
            condition="failure",
            reported_outcome=None,
            failure=failure,
            outcome=None,
            category="failure",
        )
        for failure in (
            "rejected_before_acceptance",
            "retryable",
            "malformed_response",
            "permanent",
            "transport_unknown",
            "implementation_exception",
        )
    )
    rows.extend(
        RuntimeOutcome(
            condition=condition,
            reported_outcome=None,
            failure=None,
            outcome=None,
            category=category,
        )
        for condition, category in (
            ("cancel_before_start", "blocked"),
            ("cancel_after_start", "cancelled"),
            ("artifact_limit_exhausted", "blocked"),
            ("deadline_exhausted", "blocked"),
        )
    )
    return tuple(rows)


def test_local_execution_uses_real_callback_and_p3_state() -> None:
    asyncio.run(_assert_local_execution())


async def _assert_local_execution() -> None:
    prepared = _prepare(data=None)
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    selected = next(iter(prepared.implementations))
    implementation = ExecutionImplementation(
        implementation=selected.capability.implementation,
        configuration=selected.capability.configuration,
        capability=selected.capability,
        request=None,
    )
    policy = OperationExecutionPolicy(
        node=selected.node,
        kind="local",
        request=None,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_runtime_rows(),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(selected.capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=10,
            max_provenance_edges=10,
        ),
    )
    callback = _Uppercase()
    services = ExecutionServices(
        handles=(
            ImplementationHandle(
                implementation=implementation.implementation,
                operation=implementation.capability.operation,
                configuration=implementation.configuration,
                local=callback,
                transport=None,
                resource=None,
            ),
        ),
        context_resources=(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=0,
            max_runtime_artifacts=10,
            max_runtime_artifact_bytes=100,
            max_collection_items=10,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_Clock(),
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=(selected.capability,),
        services=services,
    )
    result = await running.wait()
    assert callback.calls == 2
    assert len({state.invocation for state in result.states}) == 1
    assert all(state.complete for state in result.states)
    assert all(item.category == "success" for item in result.record.terminals)
    assert all(item.qualification == "not_assessed" for item in result.record.statuses)


def test_decision_execution_exposes_exact_wait_and_resumes_one_activation() -> None:
    asyncio.run(_assert_decision_execution())


async def _assert_decision_execution() -> None:
    workflow, node, artifact_type = _workflow(with_input=True)
    data = _data(1)
    target = next(iter(data.targets))
    capability = _capability(workflow)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact_type),),
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
        request=None,
    )
    rows = tuple(
        RuntimeOutcome(
            condition=condition,
            reported_outcome=None,
            failure=failure,
            outcome=None,
            category=category,
        )
        for condition, failure, category in (
            ("failure", "permanent", "failure"),
            ("failure", "implementation_exception", "failure"),
            ("cancel_before_start", None, "blocked"),
            ("cancel_after_start", None, "cancelled"),
            ("artifact_limit_exhausted", None, "blocked"),
            ("deadline_exhausted", None, "blocked"),
        )
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="decision",
        request=None,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset(),
        runtime_outcomes=rows,
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(policy,),
        decisions=(
            DecisionDeclaration(
                node=node,
                artifact_port="input",
                outcomes=(DecisionOutcome(decision="approve", outcome="ok"),),
                max_lifetime_ns=10,
            ),
        ),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=1,
            max_provenance_edges=0,
        ),
    )
    callback = _Decision()
    services = ExecutionServices(
        handles=(
            ImplementationHandle(
                implementation=capability.implementation,
                operation=capability.operation,
                configuration=capability.configuration,
                local=callback,
                transport=None,
                resource=None,
            ),
        ),
        context_resources=(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=0,
            max_runtime_artifacts=2,
            max_runtime_artifact_bytes=100,
            max_collection_items=1,
        ),
        decision_limits=DecisionLimits(max_pending=1, max_lifetime_ns=10),
        clock=_Clock(),
    )
    running = await start_execution(admitted=admitted, capabilities=(capability,), services=services)
    while not running.pending_decisions():
        await asyncio.sleep(0)
    wait = running.pending_decisions()[0]
    running.submit_decision(
        DecisionResponse(wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve")
    )
    result = await running.wait()
    assert callback.calls == 1
    assert result.states[0].complete
    assert result.record.terminals[0].category == "success"
    assert not result.pending_decisions
