# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real adaptive-context execution through the shared request authority."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    ContextSourceRef,
    RetrievalBounds,
    SourceItem,
    SourceResponse,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.executor import (
    AssessmentFinding,
    AssessmentLimits,
    DecisionLimits,
    EvidenceProductionDecl,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalAssessmentResult,
    LocalCompleted,
    OperationExecutionPolicy,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    StateRevisionView,
    prepare,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    RequestAssociation,
    SemanticAssociation,
    StopConfirmed,
    TextArtifactValue,
    TransportFailure,
    TransportResult,
    TransportSuccess,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    DynamicLimits,
    DynamicScope,
    EvidencePromise,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_local_executor import _Clock, _runtime_rows
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow

SOURCE = ContextSourceRef(name="adaptive", revision=1)


@dataclass
class _Provider:
    calls: int = 0

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse:
        self.calls += 1
        assert selector.fields[0].value == "target-0"
        assert bounds.max_items == 1
        return SourceResponse(
            source=SOURCE,
            items=(SourceItem(association=association, key=4, version=1, text="retrieved"),),
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                usage=ExactUsage(input_units=1, output_units=1),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


@dataclass
class _UnusedTransport:
    async def dispatch(self, request: DispatchEnvelope) -> TransportResult:
        del request
        raise AssertionError("adaptive retrieval must use its context provider")

    async def cancel(self, request: object) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))

    async def close(self) -> None:
        return None


@dataclass
class _RetryingTransport:
    calls: int = 0

    async def dispatch(self, request: DispatchEnvelope) -> TransportResult:
        self.calls += 1
        settlement = ExternalSettlement(
            request=request.request,
            disposition="completed",
            usage=ExactUsage(input_units=1, output_units=1),
            remote_stopped=True,
        )
        if self.calls == 1:
            return TransportFailure(failure="retryable", settlement=settlement)
        association = request.associations[0].association
        return TransportSuccess(
            results=(
                AssociationResult(
                    association=association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port="context",
                            artifact_type=request.operation.outputs[0].artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text="retried"),
                        ),
                    ),
                    consumed_context_ports=frozenset(),
                ),
            ),
            settlement=settlement,
        )

    async def cancel(self, request: object) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))

    async def close(self) -> None:
        return None


def _adaptive_rows() -> tuple[RuntimeOutcome, ...]:
    rows = [RuntimeOutcome(condition="result", reported_outcome="ok", failure=None, outcome="ok", category="success")]
    rows.extend(
        RuntimeOutcome(condition="failure", reported_outcome=None, failure=failure, outcome=None, category="failure")
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
        RuntimeOutcome(condition=value, reported_outcome=None, failure=None, outcome=None, category=category)
        for value, category in (
            ("cancel_before_start", "blocked"),
            ("cancel_after_start", "cancelled"),
            ("cancel_after_dispatch", "lost"),
            ("lost", "lost"),
            ("request_inconsistent", "inconsistent"),
            ("budget_exhausted", "blocked"),
            ("request_limit_exhausted", "blocked"),
            ("artifact_limit_exhausted", "blocked"),
            ("deadline_exhausted", "blocked"),
        )
    )
    return tuple(rows)


def _adaptive_workflow(
    *,
    assessment: bool = False,
    distinct_subject: bool = False,
    alias_evidence: bool = False,
    requests: int = 1,
):
    base, node, artifact = _workflow(requests=requests, with_input=True)
    static = base.workflow
    raw = next(item for item in static.nodes if isinstance(item, OperationNode))
    output_port = "assessment" if distinct_subject else "context"
    outcome = replace(
        raw.operation.outcomes[0],
        produced_ports=frozenset({output_port}),
        evidence=(
            frozenset(
                {
                    EvidencePromise(
                        name="checked",
                        meaning="test assessment",
                        subject_port="input" if distinct_subject else output_port,
                        consumed_ports=frozenset({"input"}),
                        coverage=frozenset(),
                    )
                }
            )
            if assessment
            else frozenset()
        ),
        ceiling=ResourceCeiling(
            max_activations=1,
            max_model_requests=requests,
            max_input_bytes=100,
            max_output_bytes=100,
        ),
    )
    operation = replace(
        raw.operation,
        outputs=(OutputPort(name=output_port, artifact_type=artifact),),
        output_dependencies=(
            OutputDependency(
                output=output_port,
                inputs=frozenset({"input"}),
                identity_input="input" if alias_evidence else None,
            ),
        ),
        outcomes=(outcome,),
    )
    admitted = admit_static_workflow(
        workflow=static.workflow,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=tuple(static.input_bindings),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=node, port=output_port),
                destination=WorkflowOutputRef(port=output_port),
            ),
        ),
        outcome_bindings=tuple(static.outcome_bindings),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=1,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    return (
        admit_activation_workflow(
            workflow=admitted,
            scopes=(DynamicScope(workflow=admitted, maps=(), joins=(), loops=()),),
            limits=DynamicLimits(
                max_maps=0,
                max_joins=0,
                max_loops=0,
                max_children_per_map=0,
                max_iterations_per_loop=0,
                max_dynamic_depth=1,
                max_activation_occurrences=1,
            ),
        ),
        node,
        artifact,
    )


def test_adaptive_retrieval_materializes_provider_result_once() -> None:
    asyncio.run(_assert_adaptive_retrieval())


async def _assert_adaptive_retrieval() -> None:
    workflow, node, artifact = _adaptive_workflow()
    data = _data(1)
    target = next(iter(data.targets))
    capability = _capability(workflow, external=True)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=1
        ),
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact),),
        limits=_limits(capabilities=1),
    )
    request_policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    source_capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=artifact,
        uses=frozenset({"adaptive_retrieval"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=request_policy,
        safe_detachment="forbidden",
    )
    declaration = AdaptiveRetrievalDecl(
        node=node,
        source=SOURCE,
        selector_ports=("input",),
        output_port="context",
        bounds=RetrievalBounds(max_items=1, max_bytes=20, max_requests=1),
        materialization=ContextMaterialization(kind="single", item_type=artifact),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(declaration,),
        context_capabilities=(source_capability,),
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=request_policy,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="external",
        request=request_policy,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_adaptive_rows(),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=2,
            max_provenance_edges=1,
        ),
    )
    provider = _Provider()
    transport = _UnusedTransport()
    services = ExecutionServices(
        handles=(
            ImplementationHandle(
                implementation=capability.implementation,
                operation=capability.operation,
                configuration=capability.configuration,
                local=None,
                transport=transport,
                resource=ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport),
            ),
        ),
        context_resources=(
            ContextResource(
                source=SOURCE,
                capability=source_capability,
                lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                factory=None,
            ),
        ),
        limits=ExecutionLimits(
            max_local_in_flight=0,
            max_remote_outstanding=1,
            max_runtime_artifacts=3,
            max_runtime_artifact_bytes=100,
            max_collection_items=1,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_Clock(),
    )
    running = await start_execution(admitted=admitted, capabilities=(capability,), services=services)
    result = await running.wait()
    assert provider.calls == 1
    assert result.requests.dispatched_count == 1
    assert result.states[0].complete
    assert any(value == TextArtifactValue(text="retrieved") for _, value in result.artifacts)
    assert result.final_outputs[0].candidate.target == target


def test_external_execution_retries_with_new_charged_request() -> None:
    asyncio.run(_assert_external_retry())


async def _assert_external_retry() -> None:
    workflow, node, artifact = _adaptive_workflow(requests=2)
    data = _data(1)
    target = next(iter(data.targets))
    capability = replace(_capability(workflow, external=True), max_physical_requests_per_activation=2)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=2
        ),
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact),),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    request_policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=2,
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=request_policy,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="external",
        request=request_policy,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_adaptive_rows(),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=2,
            max_provenance_edges=1,
        ),
    )
    transport = _RetryingTransport()
    services = ExecutionServices(
        handles=(
            ImplementationHandle(
                implementation=capability.implementation,
                operation=capability.operation,
                configuration=capability.configuration,
                local=None,
                transport=transport,
                resource=ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport),
            ),
        ),
        context_resources=(),
        limits=ExecutionLimits(
            max_local_in_flight=0,
            max_remote_outstanding=1,
            max_runtime_artifacts=2,
            max_runtime_artifact_bytes=100,
            max_collection_items=1,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_Clock(),
    )
    running = await start_execution(admitted=admitted, capabilities=(capability,), services=services)
    result = await running.wait()
    assert transport.calls == 2
    assert [item.purpose for item in result.requests.dispatches] == ["initial", "retry"]
    assert result.requests.dispatched_count == 2
    assert result.states[0].complete
    assert any(value == TextArtifactValue(text="retried") for _, value in result.artifacts)


@dataclass
class _AssessingLocal:
    finding: AssessmentFinding
    evidence_port: str
    alias_evidence: bool

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        association = request[0].association
        assert isinstance(association, SemanticAssociation)
        input_value = request[0].inputs[0].value
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port=self.evidence_port,
                            artifact_type=request[0].inputs[0].artifact_type,
                            artifact=None,
                            value=input_value if self.alias_evidence else TextArtifactValue(text="evidence"),
                        ),
                    ),
                    consumed_context_ports=frozenset(),
                ),
            ),
            assessments=(
                LocalAssessmentResult(
                    association=association,
                    promise="checked",
                    evidence_port=self.evidence_port,
                    finding=self.finding,
                ),
            ),
        )


@pytest.mark.parametrize(("distinct_subject", "alias_evidence"), [(False, False), (True, False), (True, True)])
def test_local_assessment_retains_actual_callback_fact_and_environment(
    distinct_subject: bool, alias_evidence: bool
) -> None:
    asyncio.run(_assert_local_assessment(distinct_subject=distinct_subject, alias_evidence=alias_evidence))


async def _assert_local_assessment(*, distinct_subject: bool, alias_evidence: bool) -> None:
    evidence_port = "assessment" if distinct_subject else "context"
    workflow, node, artifact = _adaptive_workflow(
        assessment=True,
        distinct_subject=distinct_subject,
        alias_evidence=alias_evidence,
    )
    data = _data(1)
    target = next(iter(data.targets))
    capability = _capability(workflow)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact),),
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
    policy = OperationExecutionPolicy(
        node=node,
        kind="local",
        request=None,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_runtime_rows(),
    )
    finding = AssessmentFinding(status="satisfied", code="observed")
    limits = AssessmentLimits(
        max_productions=1,
        max_findings_per_production=1,
        max_finding_code_bytes=20,
        max_absence_queries=0,
        max_assessment_facts=1,
        max_port_facts=3,
        max_provenance_edges=2,
    )
    if distinct_subject:
        with pytest.raises(EffectRejected) as rejected:
            admit_execution_plan(
                context=context,
                capabilities=(capability,),
                policies=(policy,),
                decisions=(),
                assessment_productions=(
                    EvidenceProductionDecl(
                        node=node,
                        outcome="ok",
                        promise="checked",
                        evidence_port="input",
                        absence_queries=frozenset(),
                        supported_findings=frozenset({finding}),
                    ),
                ),
                assessment_limits=limits,
            )
        assert rejected.value.code.value == "unsupported"
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(
            EvidenceProductionDecl(
                node=node,
                outcome="ok",
                promise="checked",
                evidence_port=evidence_port,
                absence_queries=frozenset(),
                supported_findings=frozenset({finding}),
            ),
        ),
        assessment_limits=limits,
    )
    callback = _AssessingLocal(
        finding=finding,
        evidence_port=evidence_port,
        alias_evidence=alias_evidence,
    )
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
            max_runtime_artifacts=3,
            max_runtime_artifact_bytes=100,
            max_collection_items=1,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_Clock(),
    )
    running = await start_execution(admitted=admitted, capabilities=(capability,), services=services)
    result = await running.wait()
    assert len(result.assessments) == 1
    fact = result.assessments[0]
    assert fact.finding == finding
    assert fact.environment.configuration == capability.configuration
    assert fact.evidence_artifact == result.final_outputs[0].candidate.artifact
    if distinct_subject:
        subject = next(item for item in result.ports if item.port == "input")
        evidence = next(item for item in result.ports if item.port == evidence_port)
        assert subject.role == "candidate"
        assert evidence.role == "evidence"
        assert (subject.artifact == result.final_outputs[0].candidate.artifact) is alias_evidence
    else:
        assert any(item.role == "candidate" and item.port == "context" for item in result.ports)


def test_nested_assessment_marks_exact_inner_subject_occurrence() -> None:
    asyncio.run(_assert_nested_assessment())


async def _assert_nested_assessment() -> None:
    artifact = ArtifactType(name="text", revision=1)
    body_owner = WorkflowId.new()
    inner = NodeId.new(workflow=body_owner)
    outcome = OutcomeSpec(
        name="ok",
        category="success",
        produced_ports=frozenset({"assessment"}),
        context=frozenset(),
        evidence=frozenset(
            {
                EvidencePromise(
                    name="checked",
                    meaning="nested assessment",
                    subject_port="input",
                    consumed_ports=frozenset({"input"}),
                    coverage=frozenset(),
                )
            }
        ),
        state_effects=frozenset(),
        model_requirements=frozenset(),
        ceiling=ResourceCeiling(
            max_activations=1,
            max_model_requests=0,
            max_input_bytes=100,
            max_output_bytes=100,
        ),
    )
    operation = OperationSpec(
        name="nested-assessor",
        inputs=(InputPort(name="input", artifact_type=artifact),),
        outputs=(OutputPort(name="assessment", artifact_type=artifact),),
        output_dependencies=(OutputDependency(output="assessment", inputs=frozenset({"input"}), identity_input=None),),
        outcomes=(outcome,),
    )
    body = admit_static_workflow(
        workflow=body_owner,
        interface=operation,
        nodes=(OperationNode(id=inner, operation=operation),),
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="input"),
                destination=NodeInputRef(node=inner, port="input"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=inner, port="assessment"),
                destination=WorkflowOutputRef(port="assessment"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=inner, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=2,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    root_owner = WorkflowId.new()
    container = NodeId.new(workflow=root_owner)
    root_interface = replace(
        operation,
        outcomes=(
            replace(
                outcome,
                ceiling=replace(outcome.ceiling, max_activations=2),
            ),
        ),
    )
    root = admit_static_workflow(
        workflow=root_owner,
        interface=root_interface,
        nodes=(SubgraphNode(id=container, operation=operation, body=body),),
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="input"),
                destination=NodeInputRef(node=container, port="input"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=container, port="assessment"),
                destination=WorkflowOutputRef(port="assessment"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=container, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=2,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
        workflow=root,
        scopes=(
            DynamicScope(workflow=root, maps=(), joins=(), loops=()),
            DynamicScope(workflow=body, maps=(), joins=(), loops=()),
        ),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=2,
            max_activation_occurrences=2,
        ),
    )
    data = _data(1)
    target = next(iter(data.targets))
    template_workflow, _, _ = _workflow(with_input=True)
    capability = replace(_capability(template_workflow), operation=operation)
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=6, max_entries=2, max_parent_depth=2),
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact),),
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=(
            ImplementationSelection(
                node=inner,
                implementation=capability.implementation,
                configuration=capability.configuration,
            ),
        ),
        capabilities=(capability,),
        limits=_limits(capabilities=1, slots=2),
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
    policy = OperationExecutionPolicy(
        node=inner,
        kind="local",
        request=None,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_runtime_rows(),
    )
    finding = AssessmentFinding(status="satisfied", code="nested")
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(
            EvidenceProductionDecl(
                node=inner,
                outcome="ok",
                promise="checked",
                evidence_port="assessment",
                absence_queries=frozenset(),
                supported_findings=frozenset({finding}),
            ),
        ),
        assessment_limits=AssessmentLimits(
            max_productions=1,
            max_findings_per_production=1,
            max_finding_code_bytes=20,
            max_absence_queries=0,
            max_assessment_facts=1,
            max_port_facts=5,
            max_provenance_edges=5,
        ),
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
                        local=_AssessingLocal(
                            finding=finding,
                            evidence_port="assessment",
                            alias_evidence=False,
                        ),
                        transport=None,
                        resource=None,
                    ),
                ),
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=4,
                    max_runtime_artifact_bytes=100,
                    max_collection_items=1,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    subject = next(item for item in result.ports if item.node == inner and item.port == "input")
    evidence = next(item for item in result.ports if item.node == inner and item.port == "assessment")
    assert subject.role == "candidate"
    assert evidence.role == "evidence"
    assert result.final_outputs[0].candidate.artifact == evidence.artifact
    assert subject.artifact != result.final_outputs[0].candidate.artifact
