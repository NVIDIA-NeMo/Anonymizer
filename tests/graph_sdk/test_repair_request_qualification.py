# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An admitted repair consumes prior candidate/findings and publishes a new candidate."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Literal, cast

import pytest

from anonymizer.engine.graph_sdk.capabilities import ImplementationRef
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentFinding,
    AssessmentLimits,
    EvidenceProductionDecl,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, prepare
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    FailureClass,
    PhysicalRequestPolicy,
    PortArtifact,
    StopConfirmed,
    TextArtifactValue,
    TransportFailure,
    TransportSuccess,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph import (
    ActivationLimits,
    EffectCode,
    EffectRejected,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ProtectionRequirement,
    SequenceEdge,
    WorkflowInputRef,
    WorkflowOutputRef,
)
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.implementation_fixtures import _execution_implementation, _selection
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_assessment_executor import _AssessingLocal, _single_output_workflow
from tests.graph_sdk.test_preparation import _capability, _limits
from tests.graph_sdk.workflow_construction import DynamicScope, admit_activation_workflow, admit_static_workflow


@dataclass
class _RepairTransport:
    failure: FailureClass | None = None
    calls: int = 0
    closes: int = 0

    async def dispatch(self, request: DispatchEnvelope) -> TransportSuccess | TransportFailure:
        self.calls += 1
        (item,) = request.associations
        values = {port.port: port.value for port in item.inputs}
        assert values == {
            "candidate": TextArtifactValue(text="original"),
            "findings": TextArtifactValue(text="needs repair"),
        }
        if self.calls == 1 and self.failure is not None:
            return TransportFailure(
                failure=self.failure,
                settlement=ExternalSettlement(
                    request=request.request,
                    disposition="completed",
                    usage=ExactUsage(input_units=2, output_units=0),
                    remote_stopped=True,
                ),
            )
        return TransportSuccess(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        PortArtifact(
                            port="repaired",
                            artifact_type=item.inputs[0].artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text="repaired"),
                        ),
                    ),
                ),
            ),
            settlement=ExternalSettlement(
                request=request.request,
                disposition="completed",
                usage=ExactUsage(input_units=2, output_units=1),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: object) -> StopConfirmed:
        raise AssertionError("completed repair is not cancelled")

    async def close(self) -> None:
        self.closes += 1


class _ProduceRepairInputs:
    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        (item,) = request
        (source,) = item.inputs
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        PortArtifact(
                            port="prior", artifact_type=source.artifact_type, artifact=None, value=source.value
                        ),
                        PortArtifact(
                            port="findings",
                            artifact_type=source.artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text="needs repair"),
                        ),
                    ),
                ),
            )
        )


async def _execute_repair(
    transport: _RepairTransport,
    purpose: Literal["initial", "repair"],
    *,
    kind: Literal["external", "local"] = "external",
):
    base, assessor, artifact = _single_output_workflow(assessment=True, distinct_subject=True, requests=0)
    raw = base
    repair = NodeId.new(workflow=raw.workflow)
    producer = NodeId.new(workflow=raw.workflow)
    assessment = replace(
        raw.interface,
        output_dependencies=(OutputDependency(output="assessment", inputs=frozenset(), identity_input=None),),
        outcomes=tuple(
            replace(
                outcome,
                evidence=frozenset(replace(promise, consumed_ports=frozenset()) for promise in outcome.evidence),
            )
            for outcome in raw.interface.outcomes
        ),
    )
    repair_operation = replace(
        assessment,
        name="repair",
        inputs=tuple(InputPort(name=name, artifact_type=artifact) for name in ("candidate", "findings")),
        outputs=(OutputPort(name="repaired", artifact_type=artifact),),
        output_dependencies=(
            OutputDependency(output="repaired", inputs=frozenset({"candidate", "findings"}), identity_input=None),
        ),
        outcomes=tuple(
            replace(
                outcome,
                evidence=frozenset(),
                produced_ports=frozenset({"repaired"}),
                ceiling=replace(outcome.ceiling, max_model_requests=2),
            )
            for outcome in assessment.outcomes
        ),
    )
    producer_operation = replace(
        repair_operation,
        name="evaluate-prior",
        inputs=(repair_operation.inputs[0],),
        outputs=tuple(OutputPort(name=name, artifact_type=artifact) for name in ("prior", "findings")),
        output_dependencies=(
            OutputDependency(output="prior", inputs=frozenset({"candidate"}), identity_input="candidate"),
            OutputDependency(output="findings", inputs=frozenset({"candidate"}), identity_input=None),
        ),
        outcomes=tuple(
            replace(
                outcome,
                produced_ports=frozenset({"prior", "findings"}),
                ceiling=replace(outcome.ceiling, max_model_requests=0),
            )
            for outcome in repair_operation.outcomes
        ),
    )
    interface = replace(
        repair_operation,
        inputs=(repair_operation.inputs[0],),
        outputs=(OutputPort(name="final", artifact_type=artifact),),
        output_dependencies=(
            replace(repair_operation.output_dependencies[0], output="final", inputs=frozenset({"candidate"})),
        ),
        outcomes=tuple(
            replace(
                outcome,
                produced_ports=frozenset({"final"}),
                evidence=frozenset(
                    replace(promise, subject_port="final") for promise in assessment.outcomes[0].evidence
                ),
                ceiling=replace(outcome.ceiling, max_activations=3, max_input_bytes=300, max_output_bytes=300),
            )
            for outcome in repair_operation.outcomes
        ),
    )
    nodes = (
        OperationNode(id=repair, operation=repair_operation),
        OperationNode(id=assessor, operation=assessment),
        OperationNode(id=producer, operation=producer_operation),
    )
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=interface,
        nodes=nodes,
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="candidate"), destination=NodeInputRef(node=producer, port="candidate")
            ),
            InputBinding(
                source=NodeOutputRef(node=producer, port="prior"),
                destination=NodeInputRef(node=repair, port="candidate"),
            ),
            InputBinding(
                source=NodeOutputRef(node=producer, port="findings"),
                destination=NodeInputRef(node=repair, port="findings"),
            ),
            InputBinding(
                source=NodeOutputRef(node=repair, port="repaired"),
                destination=NodeInputRef(node=assessor, port="input"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=repair, port="repaired"), destination=WorkflowOutputRef(port="final")
            ),
        ),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(SequenceEdge(before=producer, after=repair), SequenceEdge(before=repair, after=assessor)),
        choices=(),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning="test assessment",
                subject_port="final",
                consumed_ports=frozenset(),
                coverage=frozenset(),
            ),
        ),
        limits=replace(raw.limits, max_nodes=3, max_bindings=6, max_sequence_edges=2),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),),
        limits=replace(base.dynamic_limits, max_activation_occurrences=3),
    )
    graph = DataGraph.new()
    graph, target = graph.add_text("original")
    data = graph.validate(
        targets=(target,),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(max_datums=1, max_targets=1, max_text_bytes=32, max_declarations=0, max_group_members=0),
    )
    capabilities = tuple(
        replace(
            _capability(workflow, external=index == 0),
            operation=node.operation,
            implementation=ImplementationRef(name=f"repair-{index}", revision=1),
            max_physical_requests_per_activation=2 if index == 0 else 0,
            resource_lifetime="executor_owned" if index == 0 else "stateless",
        )
        for index, node in enumerate(nodes)
    )
    prepared = prepare(
        data=data,
        workflow=workflow,
        capabilities=capabilities,
        activation_limits=ActivationLimits(max_events=9, max_entries=3, max_parent_depth=1),
        selections=tuple(_selection(node.id, cap) for node, cap in zip(nodes, capabilities, strict=True)),
        limits=_limits(capabilities=3),
        configuration=PreparationConfiguration(
            purpose="protection", required_protection_outcomes=frozenset({"ok"}), hard_request_limit=2
        ),
        bound_inputs=(BoundInput(target=target, source=target, port="candidate", artifact_type=artifact),),
    )
    request = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=2,
    )
    finding = AssessmentFinding(status="satisfied", code="observed")
    policies = tuple(
        OperationExecutionPolicy(
            node=node.id,
            kind=kind if index == 0 else "local",
            request=request if index == 0 else None,
            initial_request_purpose=purpose if index == 0 else "initial",
            safe_detachment="forbidden",
            implementations=(_execution_implementation(cap, request=request if index == 0 else None),),
            result_outcomes=frozenset({"ok"}),
            runtime_outcomes=_valid_runtime_rows("external" if index == 0 else "local", frozenset({"ok"})),
        )
        for index, (node, cap) in enumerate(zip(nodes, capabilities, strict=True))
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(prepared=prepared, bound_context=None),
        capabilities=capabilities,
        policies=policies,
        assessment_productions=(
            EvidenceProductionDecl(
                node=assessor,
                outcome="ok",
                promise="checked",
                evidence_port="assessment",
                supported_findings=frozenset({finding}),
            ),
        ),
        assessment_limits=AssessmentLimits(
            max_productions=1,
            max_findings_per_production=1,
            max_finding_code_bytes=16,
            max_assessment_facts=1,
            max_port_facts=8,
            max_provenance_edges=4,
        ),
    )
    services = ExecutionServices(
        handles=tuple(
            ImplementationHandle(
                implementation=cap.implementation,
                operation=node.operation,
                configuration=cap.configuration,
                local=None
                if index == 0
                else _AssessingLocal(finding, "assessment", False)
                if index == 1
                else _ProduceRepairInputs(),
                transport=transport if index == 0 else None,
                resource=ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
                if index == 0
                else None,
            )
            for index, (node, cap) in enumerate(zip(nodes, capabilities, strict=True))
        ),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1,
            max_runtime_artifacts=4,
            max_runtime_artifact_bytes=100,
            max_collection_items=0,
        ),
        clock=_ZeroClock(),
    )
    return admitted, await (
        await start_execution(admitted=admitted, capabilities=capabilities, services=services)
    ).wait()


@pytest.mark.parametrize("purpose", ["initial", "repair"])
def test_declared_fresh_repair_publishes_new_candidate_and_qualifies(purpose: Literal["initial", "repair"]) -> None:
    transport = _RepairTransport()
    execution, result = asyncio.run(_execute_repair(transport, purpose))
    policy = next(policy for policy in execution.policies if policy.kind == "external")
    assert policy != replace(policy, initial_request_purpose="initial" if purpose == "repair" else "repair")
    (dispatch,) = result.requests.dispatches
    assert dispatch.purpose == purpose
    assert result.requests.terminals[0].category == "success" and len(result.requests.settlements) == 1
    assert len(result.record.terminals) == 3
    (final,) = result.final_outputs
    value = dict(result.artifacts)[final.candidate.artifact]
    assert value == TextArtifactValue(text="repaired")
    source = next(port for port in result.ports if port.port == "prior")
    assert source.artifact != final.candidate.artifact
    repaired = next(fact for fact in result.provenance if fact.key == final.producer)
    assert len(repaired.parents) == 2
    parents = [fact for fact in result.provenance if fact.key in repaired.parents]
    assert {fact.key.port for fact in parents} == {"prior", "findings"}
    assert len({fact.key.activation for fact in parents}) == 1
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1 and output.record.statuses[0].qualification == "met"
    assert not output.targets[0].withholding
    assert transport.calls == transport.closes == 1
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    assert transport.calls == transport.closes == 1


@pytest.mark.parametrize("purpose", ["retry", "correction", "failover", "unknown"])
def test_invalid_initial_request_purpose_rejects_before_effects(purpose: str) -> None:
    transport = _RepairTransport()
    with pytest.raises(EffectRejected) as error:
        asyncio.run(_execute_repair(transport, cast(Literal["initial", "repair"], purpose)))
    assert error.value.code is EffectCode.INVALID_VALUE
    assert transport.calls == transport.closes == 0


def test_repair_requires_an_external_policy_before_effects() -> None:
    transport = _RepairTransport()
    with pytest.raises(EffectRejected) as error:
        asyncio.run(_execute_repair(transport, "repair", kind="local"))
    assert error.value.code is EffectCode.CONTRADICTORY
    assert transport.calls == transport.closes == 0


@pytest.mark.parametrize(("failure", "followup"), [("retryable", "retry"), ("malformed_response", "correction")])
def test_repair_followups_keep_cumulative_request_accounting(failure: FailureClass, followup: str) -> None:
    transport = _RepairTransport(failure=failure)
    execution, result = asyncio.run(_execute_repair(transport, "repair"))
    first, second = result.requests.dispatches
    assert (first.purpose, second.purpose) == ("repair", followup)
    assert first.request != second.request and first.associations == second.associations
    assert first.policy == second.policy and first.policy.max_attempts == 2
    assert result.requests.dispatched_count == result.requests.budget_limit == len(result.requests.settlements) == 2
    assert len(result.record.terminals) == 3
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1 and output.record.statuses[0].qualification == "met"
    assert transport.calls == 2 and transport.closes == 1
