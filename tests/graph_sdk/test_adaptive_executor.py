# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real adaptive-context execution through the shared request authority."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

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
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    EvidenceProductionDecl,
    LocalAssessmentResult,
    LocalCompleted,
    OperationExecutionPolicy,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    ExactUsage,
    DispatchEnvelope,
    ExternalSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    RequestAssociation,
    SemanticAssociation,
    StopConfirmed,
    TextArtifactValue,
    TransportResult,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.workflow import (
    DynamicLimits,
    DynamicScope,
    NodeOutputRef,
    OperationNode,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    EvidencePromise,
    WorkflowOutputRef,
    WorkflowLimits,
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


def _adaptive_workflow(*, assessment: bool = False):
    base, node, artifact = _workflow(requests=1, with_input=True)
    static = base.workflow
    raw = next(item for item in static.nodes if isinstance(item, OperationNode))
    outcome = replace(
        raw.operation.outcomes[0],
        produced_ports=frozenset({"context"}),
        evidence=(
            frozenset(
                {
                    EvidencePromise(
                        name="checked",
                        meaning="test assessment",
                        subject_port="context",
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
            max_model_requests=1,
            max_input_bytes=100,
            max_output_bytes=100,
        ),
    )
    operation = replace(
        raw.operation,
        outputs=(OutputPort(name="context", artifact_type=artifact),),
        output_dependencies=(OutputDependency(output="context", inputs=frozenset({"input"}), identity_input=None),),
        outcomes=(outcome,),
    )
    admitted = admit_static_workflow(
        workflow=static.workflow,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=tuple(static.input_bindings),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=node, port="context"),
                destination=WorkflowOutputRef(port="context"),
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


@dataclass
class _AssessingLocal:
    finding: AssessmentFinding

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        association = request[0].association
        assert isinstance(association, SemanticAssociation)
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port="context",
                            artifact_type=request[0].inputs[0].artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text="evidence"),
                        ),
                    ),
                    consumed_context_ports=frozenset(),
                ),
            ),
            assessments=(
                LocalAssessmentResult(
                    association=association,
                    promise="checked",
                    evidence_port="context",
                    finding=self.finding,
                ),
            ),
        )


def test_local_assessment_retains_actual_callback_fact_and_environment() -> None:
    asyncio.run(_assert_local_assessment())


async def _assert_local_assessment() -> None:
    workflow, node, artifact = _adaptive_workflow(assessment=True)
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
                evidence_port="context",
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
            max_port_facts=3,
            max_provenance_edges=2,
        ),
    )
    callback = _AssessingLocal(finding=finding)
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
    assert any(item.role == "evidence" and item.port == "context" for item in result.ports)
