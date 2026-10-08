# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decision waits release execution capacity while retaining their own bound."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace

import pytest

from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
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
    RootInputKey,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, SemanticAssociation
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    DynamicScope,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    OperationNode,
    OutcomeBinding,
    SequenceEdge,
    WorkflowInputRef,
    WorkflowOutcomeRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_effects_production_conformance import (
    CORPUS,
    _assessment_limits,
    _normalize,
    _valid_runtime_rows,
    _ZeroClock,
)
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _workflow


@dataclass
class _Callback:
    decision: bool
    calls: int = 0
    association: SemanticAssociation | None = None
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted | LocalDecisionWait:
        self.calls += 1
        self.started.set()
        association = request[0].association
        assert isinstance(association, SemanticAssociation)
        self.association = association
        if self.decision:
            artifact = request[0].inputs[0].artifact
            assert artifact is not None
            return LocalDecisionWait(association=association, artifact=artifact)
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            )
        )


@pytest.mark.parametrize(
    "case_id",
    [
        "decisions/matching_resume_unrelated_advances",
        "decisions/pending_exact",
        "decisions/pending_one_over",
    ],
)
@pytest.mark.parametrize("local_capacity", [1, 2])
def test_decision_scheduling(case_id: str, local_capacity: int) -> None:
    asyncio.run(_assert_scheduling(case_id, local_capacity))


async def _assert_scheduling(case_id: str, local_capacity: int) -> None:
    case = next(item for item in json.loads(CORPUS.read_bytes()) if item["case_id"] == case_id)
    both_decisions = case_id != "decisions/matching_resume_unrelated_advances"
    pending_limit = case["declaration"].get("max_pending", 2)
    original, _, artifact_type = _workflow(with_input=True)
    static = original.workflow
    nodes = tuple(NodeId.new(workflow=static.workflow) for _ in range(3))
    operations = tuple(
        replace(static.interface, name=f"operation-{index}", inputs=() if index == 2 else static.interface.inputs)
        for index in range(3)
    )
    interface = replace(
        static.interface,
        inputs=tuple(InputPort(name=name, artifact_type=artifact_type) for name in ("left", "right")),
        outcomes=tuple(
            replace(outcome, ceiling=replace(outcome.ceiling, max_activations=3))
            for outcome in static.interface.outcomes
        ),
    )
    rebuilt = admit_static_workflow(
        workflow=static.workflow,
        interface=interface,
        nodes=tuple(
            OperationNode(id=node, operation=operation) for node, operation in zip(nodes, operations, strict=True)
        ),
        input_bindings=tuple(
            InputBinding(source=WorkflowInputRef(port=name), destination=NodeInputRef(node=node, port="input"))
            for name, node in zip(("left", "right"), nodes[:2], strict=True)
        ),
        output_bindings=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=nodes[2], outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=tuple(SequenceEdge(before=node, after=nodes[2]) for node in nodes[:2]),
        choices=(),
        protection=(),
        limits=replace(static.limits, max_nodes=3, max_bindings=3, max_sequence_edges=2),
    )
    workflow = admit_activation_workflow(
        workflow=rebuilt,
        scopes=(DynamicScope(workflow=rebuilt, maps=(), joins=(), loops=()),),
        limits=replace(original.limits, max_activation_occurrences=3),
    )
    capabilities = tuple(
        replace(
            _capability(workflow),
            operation=operation,
            implementation=ImplementationRef(name=f"implementation-{index}", revision=1),
        )
        for index, operation in enumerate(operations)
    )
    data = _data(1)
    target = next(iter(data.targets))
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=9, max_entries=3, max_parent_depth=1),
        bound_inputs=tuple(
            BoundInput(target=target, source=target, port=name, artifact_type=artifact_type)
            for name in ("left", "right")
        ),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=tuple(
            ImplementationSelection(
                node=node, implementation=capability.implementation, configuration=capability.configuration
            )
            for node, capability in zip(nodes, capabilities, strict=True)
        ),
        capabilities=capabilities,
        limits=_limits(capabilities=3, slots=3),
    )
    context = admit_context_plan(prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=())
    callbacks = (_Callback(decision=True), _Callback(decision=both_decisions), _Callback(decision=False))
    policies = tuple(
        OperationExecutionPolicy(
            node=node,
            kind="decision" if callback.decision else "local",
            request=None,
            safe_detachment="forbidden",
            implementations=(
                ExecutionImplementation(
                    implementation=capability.implementation,
                    configuration=capability.configuration,
                    capability=capability,
                    request=None,
                ),
            ),
            result_outcomes=frozenset() if callback.decision else frozenset({"ok"}),
            runtime_outcomes=_valid_runtime_rows(
                "decision" if callback.decision else "local", frozenset() if callback.decision else frozenset({"ok"})
            ),
        )
        for node, capability, callback in zip(nodes, capabilities, callbacks, strict=True)
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=capabilities,
        policies=policies,
        decisions=tuple(
            DecisionDeclaration(
                node=node,
                artifact_port="input",
                outcomes=tuple(
                    DecisionOutcome(decision=name, outcome="ok")
                    for name in (("approve", "reject") if index == 0 else ("approve",))
                ),
                max_lifetime_ns=10,
            )
            for index, (node, callback) in enumerate(zip(nodes, callbacks, strict=True))
            if callback.decision
        ),
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=capabilities,
        services=ExecutionServices(
            handles=tuple(
                ImplementationHandle(
                    implementation=capability.implementation,
                    operation=capability.operation,
                    configuration=capability.configuration,
                    local=callback,
                    transport=None,
                    resource=None,
                )
                for capability, callback in zip(capabilities, callbacks, strict=True)
            ),
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=local_capacity,
                max_remote_outstanding=0,
                max_runtime_artifacts=2,
                max_runtime_artifact_bytes=100,
                max_collection_items=0,
            ),
            decision_limits=DecisionLimits(max_pending=pending_limit, max_lifetime_ns=10),
            clock=_ZeroClock(),
        ),
    )

    async def wait_for_count(count: int) -> None:
        while len(running.pending_decisions()) != count:
            await asyncio.sleep(0)

    try:
        await asyncio.wait_for(wait_for_count(1 if pending_limit == 1 or not both_decisions else 2), 1)
        waits = running.pending_decisions()
        opened = [event for event in case["events"] if event["kind"] == "decision_open"]
        for index, callback in enumerate(callbacks[:2]):
            if callback.association is None or not callback.decision:
                continue
            wait = next(wait for wait in waits if wait.activation == callback.association.task.activation)
            event = next(event for event in opened if event["task"] == f"T{index}")
            assert wait.allowed_decisions == frozenset(event["allowed"])
            assert wait.workflow == rebuilt.workflow
        if case_id == "decisions/pending_exact":
            assert len(waits) == len(case["expected"]["state"]["decisions"]) == 2
        elif case_id == "decisions/pending_one_over":
            assert case["expected"] == {"status": "rejected", "code": "pending_limit"}
        if both_decisions and pending_limit == 2:
            assert {wait.allowed_decisions for wait in waits} == {
                frozenset({"approve", "reject"}),
                frozenset({"approve"}),
            }
            assert len({wait.artifact for wait in waits}) == 2
            assert [callback.calls for callback in callbacks] == [1, 1, 0]
        elif both_decisions:
            # Public equivalent of the reducer's rejected second decision_open:
            # the scheduler leaves it unstarted until a slot is available.
            assert sum(callback.calls for callback in callbacks) == 1
        else:
            await asyncio.wait_for(callbacks[1].started.wait(), 1)
            assert len(running.pending_decisions()) == 1
        for wait in waits:
            running.submit_decision(
                DecisionResponse(wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve")
            )
        if both_decisions and pending_limit == 1:
            while not running.pending_decisions() or running.pending_decisions()[0].wait == waits[0].wait:
                await asyncio.sleep(0)
            wait = running.pending_decisions()[0]
            assert wait.artifact != waits[0].artifact
            assert sum(callback.calls for callback in callbacks) == 2
            running.submit_decision(
                DecisionResponse(wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve")
            )
        result = await asyncio.wait_for(running.wait(), 1)
        assert all(state.complete for state in result.states)
        assert len(result.record.terminals) == 3
        assert callbacks[2].calls == 1  # required static sink runs only after both peers finish
        assert all(item.category == "success" for item in result.record.terminals)
        assert result.requests.dispatched_count == 0
        assert not result.pending_decisions
        assert len(result.artifacts) == len(result.provenance) == len(result.ports) == 2
        assert all(
            isinstance(fact.key, RootInputKey) and not fact.parents and not fact.decision for fact in result.provenance
        )
        assert not result.final_outputs and not result.assessments and not result.cleanup
        if case_id == "decisions/matching_resume_unrelated_advances":
            expected = case["expected"]["state"]
            tasks = {}
            for index, callback in enumerate(callbacks[:2]):
                assert callback.association is not None
                terminal = next(item for item in result.record.terminals if item.attempt == callback.association.task)
                tasks[f"T{index}"] = terminal.category
            assert tasks == expected["tasks"]
            for key, value in _normalize(result.requests, {}, {}, {}).items():
                assert value == expected[key], key
    finally:
        running.request_cancel()
        await asyncio.wait_for(running.wait(), 1)
