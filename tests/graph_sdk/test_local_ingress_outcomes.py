# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named graph outcomes distinguish callback exceptions from malformed returns."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import cast

import pytest

from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult
from anonymizer.graph.workflow import NodeOutcomeRef, OperationNode, OutcomeBinding, WorkflowOutcomeRef
from tests.graph_sdk.implementation_fixtures import _execution_implementation
from tests.graph_sdk.test_local_executor import _Clock, _runtime_rows
from tests.graph_sdk.test_preparation import _capability, _data, _prepare, _workflow
from tests.graph_sdk.workflow_construction import DynamicScope, admit_activation_workflow, admit_static_workflow


@dataclass
class _Callback:
    mode: str
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        if self.mode == "raise":
            raise RuntimeError("private callback detail")
        if self.mode == "outer":
            return cast(LocalCompleted, object())
        if self.mode == "container":
            return LocalCompleted(results=cast(tuple[AssociationResult, ...], []))
        if self.mode == "member":
            return LocalCompleted(results=cast(tuple[AssociationResult, ...], (object(),)))
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome="undeclared",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            )
        )


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("raise", "callback_failed"),
        ("outer", "callback_failed"),
        ("container", "malformed"),
        ("member", "malformed"),
        ("outcome", "malformed"),
    ],
)
def test_local_named_ingress_outcomes(mode: str, expected: str) -> None:
    asyncio.run(_execute(mode, expected))


async def _execute(mode: str, expected: str) -> None:
    workflow, node, _ = _workflow()
    old = workflow
    operation = replace(
        old.interface,
        outcomes=(
            old.interface.outcomes[0],
            replace(old.interface.outcomes[0], name="callback_failed", category="failure"),
            replace(old.interface.outcomes[0], name="malformed", category="failure"),
        ),
    )
    static = admit_static_workflow(
        workflow=old.workflow,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node, outcome=item.name), destination=WorkflowOutcomeRef(outcome=item.name)
            )
            for item in operation.outcomes
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=replace(old.limits, max_bindings=3, max_choice_states=3),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
        limits=workflow.dynamic_limits,
    )
    capability = _capability(workflow)
    prepared = _prepare(data=_data(1), workflow=workflow, capability=capability)
    context = admit_context_plan(prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=())
    implementation = _execution_implementation(capability, request=None)
    rows = tuple(
        replace(
            row, outcome={"implementation_exception": "callback_failed", "malformed_response": "malformed"}[row.failure]
        )
        if row.failure in {"implementation_exception", "malformed_response"}
        else row
        for row in _runtime_rows()
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(implementation,),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=rows,
            ),
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=0,
            max_provenance_edges=0,
        ),
    )
    callback = _Callback(mode)
    running = await start_execution(
        admitted=admitted,
        capabilities=(capability,),
        services=ExecutionServices(
            handles=(
                ImplementationHandle(
                    implementation=capability.implementation,
                    operation=operation,
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
                max_runtime_artifacts=0,
                max_runtime_artifact_bytes=0,
                max_collection_items=0,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_Clock(),
        ),
    )
    result = await running.wait()
    assert callback.calls == 1 and len(result.states) == 1
    assert next(iter(result.states[0].entries)).outcome == expected
    assert result.record.terminals[0].category == "failure"
    assert not result.artifacts and result.requests.dispatched_count == 0
