# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime causes of frozen terminal bridge conditions."""

from __future__ import annotations

import asyncio
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
    PortArtifact,
    StopConfirmed,
    TextArtifactValue,
    TransportLost,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.graph.workflow import (
    DynamicScope,
    NodeOutputRef,
    OperationNode,
    OutputBinding,
    OutputDependency,
    OutputPort,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_effects_production_conformance import (
    _assert_decision_submission,
    _assessment_limits,
    _valid_runtime_rows,
    _ZeroClock,
)
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow

CASES = tuple(
    case
    for case in load_cases("effects")
    if case["case_id"]
    in {
        "bridges/cancel_after_dispatch",
        "bridges/lost",
        "bridges/request_inconsistent",
        "bridges/budget_exhausted",
        "bridges/request_limit_exhausted",
        "bridges/artifact_limit_exhausted",
    }
)


@dataclass
class _Transport:
    condition: str
    calls: int = 0
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def dispatch(self, request: DispatchEnvelope) -> TransportLost | TransportSuccess:
        self.calls += 1
        self.started.set()
        if self.condition == "cancel_after_dispatch":
            await asyncio.Event().wait()
        if self.condition == "lost":
            return TransportLost(settlement=None)
        if self.condition == "request_inconsistent":
            return TransportSuccess(results=(), settlement=None)
        assert self.condition == "artifact_limit_exhausted"
        return TransportSuccess(
            results=(
                AssociationResult(
                    association=request.associations[0].association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port="output",
                            artifact_type=request.operation.outputs[0].artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text="x"),
                        ),
                    ),
                    consumed_context_ports=frozenset(),
                ),
            ),
            settlement=None,
        )

    async def cancel(self, request: object) -> StopConfirmed:
        return StopConfirmed(usage=UnknownUsage())

    async def close(self) -> None:
        raise AssertionError("stateless")


@pytest.mark.parametrize("case", CASES, ids=lambda case: cast(str, case["case_id"]))
def test_runtime_terminal_bridge(case: dict[str, Any]) -> None:
    asyncio.run(_assert_terminal(case))


async def _assert_terminal(case: dict[str, Any]) -> None:
    condition = case["case_id"].split("/")[1]
    workflow, node, artifact_type = _workflow(requests=1)
    if condition == "artifact_limit_exhausted":
        static = workflow.workflow
        operation = replace(
            static.interface,
            outputs=(OutputPort(name="output", artifact_type=artifact_type),),
            output_dependencies=(OutputDependency(output="output", inputs=frozenset(), identity_input=None),),
            outcomes=(replace(static.interface.outcomes[0], produced_ports=frozenset({"output"})),),
        )
        rebuilt = admit_static_workflow(
            workflow=static.workflow,
            interface=operation,
            nodes=(OperationNode(id=node, operation=operation),),
            input_bindings=(),
            output_bindings=(
                OutputBinding(
                    source=NodeOutputRef(node=node, port="output"), destination=WorkflowOutputRef(port="output")
                ),
            ),
            outcome_bindings=tuple(static.outcome_bindings),
            sequence=(),
            choices=(),
            protection=(),
            limits=replace(static.limits, max_bindings=2),
        )
        workflow = admit_activation_workflow(
            workflow=rebuilt,
            scopes=(DynamicScope(workflow=rebuilt, maps=(), joins=(), loops=()),),
            limits=workflow.limits,
        )
    capability = replace(_capability(workflow, external=True), resource_lifetime="stateless")
    prepared = _prepare(
        data=_data(1),
        workflow=workflow,
        capability=capability,
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=0 if condition == "budget_exhausted" else 1,
        ),
        limits=_limits(),
    )
    context = admit_context_plan(prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=())
    request_policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
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
    transport = _Transport(condition=condition)
    running = await start_execution(
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
                max_remote_outstanding=0 if condition == "request_limit_exhausted" else 1,
                max_runtime_artifacts=0,
                max_runtime_artifact_bytes=0,
                max_collection_items=0,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_ZeroClock(),
        ),
    )
    if condition == "cancel_after_dispatch":
        await transport.started.wait()
        running.request_cancel()
    result = await running.wait()
    expected = case["expected"]["state"]
    assert len(result.record.terminals) == 1
    terminal = result.record.terminals[0]
    assert terminal.attempt is not None
    assert terminal.category == expected["tasks"]["T0"]
    assert all(state.complete for state in result.states)
    assert (
        not result.artifacts
        and not result.ports
        and not result.final_outputs
        and not result.assessments
        and not result.provenance
    )
    assert not result.pending_decisions and not result.cleanup
    assert transport.calls == (0 if condition in {"budget_exhausted", "request_limit_exhausted"} else 1)
    assert result.requests.dispatched_count == transport.calls
    if condition == "budget_exhausted":
        assert len(result.requests.denials) == 1
        assert result.requests.denials[0].category == "budget_stopped"
    elif condition == "request_limit_exhausted":
        assert not result.requests.dispatches
    else:
        assert len(result.requests.terminals) == 1
        request_terminal = result.requests.terminals[0]
        if condition == "artifact_limit_exhausted":
            assert request_terminal.failure == "malformed_response"
        else:
            assert request_terminal.category == terminal.category


def test_deadline_bridge_uses_real_decision_deadline() -> None:
    corpus = load_cases("effects")
    bridge = next(case for case in corpus if case["case_id"] == "bridges/deadline_exhausted")
    decision = next(case for case in corpus if case["case_id"] == "decisions/deadline")
    assert bridge["expected"]["state"]["tasks"] == decision["expected"]["state"]["tasks"]
    asyncio.run(_assert_decision_submission("decisions/deadline", ""))
