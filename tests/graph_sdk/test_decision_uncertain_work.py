# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A real decision can pause and resume beside uncertain external work."""

from __future__ import annotations

import asyncio
from dataclasses import replace

from anonymizer.engine.graph_sdk import executor as ex
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.requests import PhysicalRequestPolicy
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph import workflow as wf
from anonymizer.graph.activation import ActivationLimits
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows
from tests.graph_sdk.test_external_executor import _LostTransport
from tests.graph_sdk.test_local_executor import _Clock, _Decision, _Immediate
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _workflow


def test_decision_pause_and_resume_preserves_uncertain_remote_work() -> None:
    asyncio.run(_execute())


async def _execute() -> None:
    workflow, decision_node, artifact = _workflow(with_input=True)
    old = workflow.workflow
    remote_node = wf.NodeId.new(workflow=old.workflow)
    join_node = wf.NodeId.new(workflow=old.workflow)
    decision_operation = old.interface
    outcome = decision_operation.outcomes[0]
    remote_operation = replace(
        decision_operation,
        name="uncertain",
        inputs=(),
        outcomes=(replace(outcome, ceiling=replace(outcome.ceiling, max_model_requests=1)),),
    )
    join_operation = replace(decision_operation, name="join", inputs=())
    interface = replace(
        decision_operation,
        name="decision-and-uncertain",
        outcomes=(replace(outcome, ceiling=replace(outcome.ceiling, max_activations=3, max_model_requests=1)),),
    )
    static = wf.admit_static_workflow(
        workflow=old.workflow,
        interface=interface,
        nodes=(
            wf.OperationNode(id=decision_node, operation=decision_operation),
            wf.OperationNode(id=remote_node, operation=remote_operation),
            wf.OperationNode(id=join_node, operation=join_operation),
        ),
        input_bindings=tuple(old.input_bindings),
        output_bindings=(),
        outcome_bindings=(
            wf.OutcomeBinding(
                source=wf.NodeOutcomeRef(node=join_node, outcome="ok"), destination=wf.WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=(
            wf.SequenceEdge(before=decision_node, after=join_node),
            wf.SequenceEdge(before=remote_node, after=join_node),
        ),
        choices=(),
        protection=(),
        limits=replace(old.limits, max_nodes=3, max_sequence_edges=2),
    )
    workflow = wf.admit_activation_workflow(
        workflow=static,
        scopes=(wf.DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
        limits=replace(workflow.limits, max_activation_occurrences=3),
    )
    capabilities = (
        replace(
            _capability(workflow),
            operation=decision_operation,
            implementation=ImplementationRef(name="decision", revision=1),
        ),
        replace(
            _capability(workflow, external=True),
            operation=remote_operation,
            implementation=ImplementationRef(name="uncertain", revision=1),
        ),
    )
    capabilities += (
        replace(
            _capability(workflow), operation=join_operation, implementation=ImplementationRef(name="join", revision=1)
        ),
    )
    nodes = (decision_node, remote_node, join_node)
    data = _data(1)
    target = next(iter(data.targets))
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=9, max_entries=3, max_parent_depth=1),
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact),),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=1
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=tuple(
            ImplementationSelection(
                node=node, implementation=capability.implementation, configuration=capability.configuration
            )
            for node, capability in zip(nodes, capabilities, strict=True)
        ),
        capabilities=capabilities,
        limits=_limits(capabilities=3),
    )
    request = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    implementations = tuple(
        ex.ExecutionImplementation(
            implementation=capability.implementation,
            configuration=capability.configuration,
            capability=capability,
            request=request if index == 1 else None,
        )
        for index, capability in enumerate(capabilities)
    )
    admitted = ex.admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=tuple(
            ex.OperationExecutionPolicy(
                node=node,
                kind="external" if index == 1 else "local" if index == 2 else "decision",
                request=request if index == 1 else None,
                safe_detachment="forbidden",
                implementations=(implementations[index],),
                result_outcomes=frozenset({"ok"}) if index else frozenset(),
                runtime_outcomes=_valid_runtime_rows(
                    "external" if index == 1 else "local" if index == 2 else "decision",
                    frozenset({"ok"}) if index else frozenset(),
                ),
            )
            for index, node in enumerate(nodes)
        ),
        decisions=(
            ex.DecisionDeclaration(
                node=decision_node,
                artifact_port="input",
                outcomes=(ex.DecisionOutcome(decision="approve", outcome="ok"),),
                max_lifetime_ns=10,
            ),
        ),
        assessment_productions=(),
        assessment_limits=ex.AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=2,
            max_provenance_edges=0,
        ),
    )
    decision = _Decision()
    remote = _LostTransport()
    join = _Immediate()
    running = await ex.start_execution(
        admitted=admitted,
        capabilities=capabilities,
        services=ex.ExecutionServices(
            handles=tuple(
                ex.ImplementationHandle(
                    implementation=capability.implementation,
                    operation=capability.operation,
                    configuration=capability.configuration,
                    local=None if index == 1 else join if index == 2 else decision,
                    transport=remote if index == 1 else None,
                    resource=ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=remote)
                    if index == 1
                    else None,
                )
                for index, capability in enumerate(capabilities)
            ),
            context_resources=(),
            limits=ex.ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=1,
                max_runtime_artifacts=1,
                max_runtime_artifact_bytes=100,
                max_collection_items=0,
            ),
            decision_limits=ex.DecisionLimits(max_pending=1, max_lifetime_ns=10),
            clock=_Clock(),
        ),
    )
    waiter = asyncio.create_task(running.wait())
    async with asyncio.timeout(2):
        while not running.pending_decisions() or remote.calls != 1:
            await asyncio.sleep(0)
        assert not waiter.done() and not remote.closed
        wait = running.pending_decisions()[0]
        running.submit_decision(
            ex.DecisionResponse(wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve")
        )
        result = await waiter
    assert decision.calls == remote.calls == 1 and not remote.closed
    assert not result.pending_decisions and result.states[0].complete
    # Sequence edges order terminal completion; the join consumes no remote value.
    entries = {entry.template: entry for entry in result.states[0].entries}
    assert entries[decision_node].status == entries[join_node].status == "success"
    assert entries[remote_node].status == "lost" and join.calls == 1
    assert result.requests.dispatched_count == 1 and result.requests.terminals[0].category == "lost"
    assert len(result.requests.remote_outstanding) == 1 and not result.requests.settlements
