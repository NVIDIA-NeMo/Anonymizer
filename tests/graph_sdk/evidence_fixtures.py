# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real assessment execution fixtures for evidence and qualification tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields, replace

from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, ImplementationSelection
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    InitialContextDecl,
    InitialVersionSelection,
    RetrievalBounds,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.engine.graph_sdk.evidence import (
    QualificationLimits,
)
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentFinding,
    AssessmentLimits,
    DecisionDeclaration,
    DecisionLimits,
    DecisionOutcome,
    DecisionResponse,
    EvidenceProductionDecl,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    LocalAssessmentResult,
    LocalCompleted,
    OperationExecutionPolicy,
    RequestTransport,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    StateRevision,
    StateRevisionView,
    prepare,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    PhysicalRequestPolicy,
    PortArtifact,
    TextArtifactValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ContextInputRef,
    ContextUse,
    CoverageAtom,
    DynamicScope,
    InputBinding,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OutcomeBinding,
    OutputBinding,
    ProtectionRequirement,
    SequenceEdge,
    StateEffect,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.test_adaptive_executor import _adaptive_workflow, _AssessingLocal
from tests.graph_sdk.test_decision_scheduling import _Callback
from tests.graph_sdk.test_preparation import _capability, _data, _limits


def _qualification_limits(**changes: int) -> QualificationLimits:
    return QualificationLimits(**{item.name: changes.get(item.name, 16) for item in fields(QualificationLimits)})


@dataclass
class _AssessmentWithAuxiliaryOutput:
    assessor: _AssessingLocal
    partial_assessment: bool = False

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        completed = await self.assessor.run(request)
        result = completed.results[0]
        return replace(
            completed,
            assessments=(
                *completed.assessments,
                LocalAssessmentResult(
                    association=completed.assessments[0].association,
                    promise="partial",
                    evidence_port="metadata",
                    finding=self.assessor.finding,
                ),
            )
            if self.partial_assessment
            else completed.assessments,
            results=(
                replace(
                    result,
                    outputs=(
                        *result.outputs,
                        PortArtifact(
                            port="metadata",
                            artifact_type=request[0].inputs[0].artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text="auxiliary"),
                        ),
                    ),
                ),
            ),
        )


@dataclass
class _AssessmentWithContext:
    assessor: _AssessingLocal

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        completed = await self.assessor.run(request)
        return replace(
            completed,
            results=tuple(replace(item, consumed_context_ports=frozenset({"input"})) for item in completed.results),
        )


async def _execute_assessment(
    *,
    environment: bool = False,
    candidate_input: bool = False,
    alias_output: bool = False,
    auxiliary_output: bool = False,
    partial_assessment: bool = False,
    root_passthrough: bool = False,
    decision_input: bool = False,
    coverage: frozenset[CoverageAtom] = frozenset(),
    target_count: int = 1,
    nested: bool = False,
    nested_depth: int = 0,
    rename_ports: bool = False,
    data: ValidatedDataGraph | None = None,
    execution_only: bool = False,
    finding: AssessmentFinding | None = None,
    resource: ResourceLease | None = None,
    external: tuple[RequestTransport, ResourceLease] | None = None,
    initial_resources: tuple[ContextResource, ...] = (),
    initial_item_limit: int = 1,
    initial_version_selection: InitialVersionSelection = "exact_one",
    execution_limits: ExecutionLimits | None = None,
    assessment_edge_limit: int | None = None,
) -> tuple[AdmittedExecutionPlan, ExecutionResult]:
    auxiliary_output = auxiliary_output or partial_assessment
    depth = nested_depth or int(nested)
    nested = depth > 0
    assert not root_passthrough or candidate_input
    assert not (decision_input and external is not None)
    assert not (auxiliary_output and (rename_ports or nested or decision_input or external is not None))
    assert not initial_resources or not (nested or rename_ports or decision_input or external or root_passthrough)
    predecessor = decision_input or external is not None
    request_policy = (
        PhysicalRequestPolicy(
            visibility="dispatch_and_settlement",
            pre_dispatch_control="executor",
            retry_owner="executor",
            replay="idempotent",
            max_attempts=2,
        )
        if external is not None
        else None
    )
    evidence_port = "assessment" if candidate_input else "context"
    base, node, artifact = _adaptive_workflow(
        assessment=True, requests=0, distinct_subject=candidate_input, alias_evidence=candidate_input or alias_output
    )
    raw = base.workflow
    read = StateEffect(kind="read", name="assessment-policy")
    operation = replace(
        raw.interface,
        outcomes=tuple(
            replace(
                outcome,
                state_effects=frozenset({read}) if environment else frozenset(),
                context=frozenset({ContextUse(port="input", meaning="retrieved", capture="whole_artifact")})
                if initial_resources
                else outcome.context,
                evidence=frozenset(replace(promise, coverage=coverage) for promise in outcome.evidence),
            )
            for outcome in raw.interface.outcomes
        ),
    )
    if auxiliary_output:
        operation = replace(
            operation,
            outputs=(*operation.outputs, replace(operation.outputs[0], name="metadata")),
            output_dependencies=(
                *operation.output_dependencies,
                replace(operation.output_dependencies[0], output="metadata", identity_input=None),
            ),
            outcomes=tuple(
                replace(
                    outcome,
                    produced_ports=outcome.produced_ports | {"metadata"},
                    evidence=outcome.evidence
                    | frozenset(replace(promise, name="partial", coverage=frozenset()) for promise in outcome.evidence)
                    if partial_assessment
                    else outcome.evidence,
                )
                for outcome in operation.outcomes
            ),
        )
    decision_node = NodeId.new(workflow=raw.workflow)
    nodes = (OperationNode(id=node, operation=operation),)
    bindings = tuple(
        replace(item, source=ContextInputRef(port="input")) if initial_resources else item
        for item in raw.input_bindings
    )
    if predecessor:
        decision_operation = replace(
            operation,
            name="approve-input",
            outcomes=tuple(
                replace(
                    outcome,
                    evidence=frozenset(),
                    state_effects=frozenset(),
                    ceiling=replace(outcome.ceiling, max_model_requests=2 if external is not None else 0),
                )
                for outcome in operation.outcomes
            ),
            output_dependencies=tuple(
                replace(dependency, identity_input="input") for dependency in operation.output_dependencies
            ),
        )
        nodes = (OperationNode(id=decision_node, operation=decision_operation), *nodes)
        bindings = (
            InputBinding(
                source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=decision_node, port="input")
            ),
            InputBinding(
                source=NodeOutputRef(node=decision_node, port=evidence_port),
                destination=NodeInputRef(node=node, port="input"),
            ),
        )
    interface = replace(
        operation,
        outcomes=tuple(
            replace(
                outcome,
                ceiling=replace(
                    outcome.ceiling,
                    max_activations=len(nodes),
                    max_model_requests=2 if external is not None else 0,
                    max_input_bytes=100 * len(nodes),
                    max_output_bytes=100 * len(nodes),
                ),
            )
            for outcome in operation.outcomes
        ),
    )
    root_input = "document" if rename_ports else "input"
    root_output = "protected" if rename_ports else evidence_port
    if rename_ports:
        interface = replace(
            interface,
            inputs=tuple(replace(port, name=root_input) for port in interface.inputs),
            outputs=tuple(replace(port, name=root_output) for port in interface.outputs),
            output_dependencies=tuple(
                replace(
                    item,
                    output=root_output,
                    inputs=frozenset({root_input}),
                    identity_input=root_input if item.identity_input is not None else None,
                )
                for item in interface.output_dependencies
            ),
            outcomes=tuple(
                replace(
                    outcome,
                    produced_ports=frozenset({root_output}),
                    evidence=frozenset(
                        replace(
                            promise,
                            subject_port=root_input if candidate_input else root_output,
                            consumed_ports=frozenset({root_input}),
                        )
                        for promise in outcome.evidence
                    ),
                )
                for outcome in interface.outcomes
            ),
        )
        bindings = tuple(
            replace(binding, source=WorkflowInputRef(port=root_input))
            if isinstance(binding.source, WorkflowInputRef)
            else binding
            for binding in bindings
        )
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=interface,
        nodes=nodes,
        input_bindings=bindings,
        output_bindings=tuple(
            replace(
                binding,
                source=WorkflowInputRef(port=root_input) if root_passthrough else binding.source,
                destination=WorkflowOutputRef(port=root_output),
            )
            for binding in raw.output_bindings
        )
        + (
            (
                OutputBinding(
                    source=NodeOutputRef(node=node, port="metadata"), destination=WorkflowOutputRef(port="metadata")
                ),
            )
            if auxiliary_output
            else ()
        ),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(SequenceEdge(before=decision_node, after=node),) if predecessor else (),
        choices=tuple(raw.choices),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning="test assessment",
                subject_port=root_input if candidate_input else root_output,
                consumed_ports=frozenset({root_input}),
                coverage=coverage,
            ),
        ),
        limits=replace(raw.limits, max_nodes=len(nodes), max_bindings=4, max_sequence_edges=1),
    )
    scopes = (DynamicScope(workflow=static, maps=(), loops=(), joins=()),)
    for level in range(depth):
        body = static
        owner = WorkflowId.new()
        container = NodeId.new(workflow=owner)
        outer_interface = replace(
            body.interface,
            outcomes=tuple(
                replace(outcome, ceiling=replace(outcome.ceiling, max_activations=outcome.ceiling.max_activations + 1))
                for outcome in body.interface.outcomes
            ),
        )
        static = admit_static_workflow(
            workflow=owner,
            interface=outer_interface,
            nodes=(SubgraphNode(id=container, operation=body.interface, body=body),),
            input_bindings=(
                InputBinding(
                    source=WorkflowInputRef(port=root_input), destination=NodeInputRef(node=container, port=root_input)
                ),
            ),
            output_bindings=(
                OutputBinding(
                    source=NodeOutputRef(node=container, port=root_output),
                    destination=WorkflowOutputRef(port=root_output),
                ),
            ),
            outcome_bindings=tuple(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=container, outcome=item.name),
                    destination=WorkflowOutcomeRef(outcome=item.name),
                )
                for item in body.interface.outcomes
            ),
            sequence=(),
            choices=(),
            protection=tuple(body.protection_requirements),
            limits=replace(body.limits, max_nodes=len(nodes) + level + 1, max_subgraph_depth=level + 2),
        )
        scopes = (*scopes, DynamicScope(workflow=static, maps=(), loops=(), joins=()))
    count = len(nodes) + depth
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=scopes,
        limits=replace(base.limits, max_activation_occurrences=count, max_dynamic_depth=depth + 1),
    )
    data = data if data is not None else _data(target_count)
    target_count = len(data.targets)
    bound_context = None
    if initial_resources:
        binding = await (
            await start_initial_binding(
                data=data,
                workflow=workflow,
                declarations=tuple(
                    InitialContextDecl(
                        target=target,
                        node=node,
                        port="input",
                        artifact_type=artifact,
                        source=initial_resources[index % len(initial_resources)].source,
                        selector=ContextSelector(fields=()),
                        requirement="required",
                        bounds=RetrievalBounds(max_items=initial_item_limit, max_bytes=20, max_requests=1),
                        materialization=ContextMaterialization(kind="single", item_type=artifact),
                        version_selection=initial_version_selection,
                    )
                    for index, target in enumerate(data.targets)
                ),
                capabilities=tuple(item.capability for item in initial_resources),
                resources=initial_resources,
                limits=BindingLimits(
                    max_declarations=target_count,
                    max_sources=len(initial_resources),
                    max_capabilities=len(initial_resources),
                    max_selector_fields=0,
                    max_selector_bytes=0,
                    max_items=initial_item_limit * target_count,
                    max_bytes=20 * target_count,
                    max_requests=target_count,
                    max_resources=len(initial_resources),
                ),
            )
        ).wait()
        assert binding.receipt.terminal == "success"
        assert binding.context is not None
        bound_context = binding.context
    capabilities = tuple(
        replace(
            _capability(workflow, external=external is not None and item.id == decision_node),
            operation=item.operation,
            max_physical_requests_per_activation=2 if external is not None and item.id == decision_node else 0,
            resource_lifetime="executor_owned"
            if external is not None and item.id == decision_node
            else "stateless"
            if resource is None
            else "caller_owned"
            if resource.owner == "caller"
            else "executor_owned",
            implementation=ImplementationRef(name=f"evidence-fixture-{index}", revision=1),
        )
        for index, item in enumerate(nodes)
    )
    prepared = prepare(
        data=data,
        workflow=workflow,
        capabilities=capabilities,
        activation_limits=ActivationLimits(max_events=3 * count, max_entries=count, max_parent_depth=depth + 1),
        selections=tuple(
            ImplementationSelection(
                node=item.id, implementation=capability.implementation, configuration=capability.configuration
            )
            for item, capability in zip(nodes, capabilities, strict=True)
        ),
        limits=_limits(capabilities=len(nodes)),
        configuration=PreparationConfiguration(
            purpose="execution_only" if execution_only else "protection",
            required_protection_outcomes=frozenset() if execution_only else frozenset({"ok"}),
            hard_request_limit=2 * target_count if external is not None else None,
        ),
        bound_inputs=()
        if initial_resources
        else tuple(
            BoundInput(target=target, source=target, port=root_input, artifact_type=artifact) for target in data.targets
        ),
        state=StateRevisionView(
            revisions=frozenset({StateRevision(effect=read, revision=3)}) if environment else frozenset()
        ),
    )
    finding = finding if finding is not None else AssessmentFinding(status="satisfied", code="observed")
    declaration = EvidenceProductionDecl(
        node=node,
        outcome="ok",
        promise="checked",
        evidence_port=evidence_port,
        absence_queries=frozenset({7}) if environment else frozenset(),
        supported_findings=frozenset({finding}),
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=bound_context, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=tuple(
            OperationExecutionPolicy(
                node=item.id,
                kind=("external" if external is not None else "decision") if item.id == decision_node else "local",
                request=request_policy if item.id == decision_node else None,
                safe_detachment="forbidden",
                implementations=(
                    ExecutionImplementation(
                        implementation=capability.implementation,
                        configuration=capability.configuration,
                        capability=capability,
                        request=request_policy if item.id == decision_node else None,
                    ),
                ),
                result_outcomes=frozenset() if item.id == decision_node and decision_input else frozenset({"ok"}),
                runtime_outcomes=_valid_runtime_rows(
                    ("external" if external is not None else "decision") if item.id == decision_node else "local",
                    frozenset() if item.id == decision_node and decision_input else frozenset({"ok"}),
                ),
            )
            for item, capability in zip(nodes, capabilities, strict=True)
        ),
        decisions=(
            DecisionDeclaration(
                node=decision_node,
                artifact_port="input",
                outcomes=(DecisionOutcome(decision="approve", outcome="ok"),),
                max_lifetime_ns=10,
            ),
        )
        if decision_input
        else (),
        assessment_productions=(declaration, replace(declaration, promise="partial", evidence_port="metadata"))
        if partial_assessment
        else (declaration,),
        assessment_limits=AssessmentLimits(
            max_productions=1 + int(partial_assessment),
            max_findings_per_production=1,
            max_finding_code_bytes=20,
            max_absence_queries=1 if environment else 0,
            max_assessment_facts=(1 + int(partial_assessment)) * target_count,
            max_port_facts=(3 + 2 * int(predecessor) + depth + int(auxiliary_output)) * target_count,
            max_provenance_edges=(2 + int(predecessor) + depth + int(auxiliary_output)) * target_count
            if assessment_edge_limit is None
            else assessment_edge_limit,
        ),
    )
    services = ExecutionServices(
        handles=tuple(
            ImplementationHandle(
                implementation=capability.implementation,
                operation=capability.operation,
                configuration=capability.configuration,
                local=None
                if external is not None and item.id == decision_node
                else _Callback(decision=True)
                if item.id == decision_node
                else _AssessmentWithContext(
                    _AssessingLocal(
                        finding=finding, evidence_port=evidence_port, alias_evidence=candidate_input or alias_output
                    )
                )
                if initial_resources
                else _AssessmentWithAuxiliaryOutput(
                    _AssessingLocal(
                        finding=finding, evidence_port=evidence_port, alias_evidence=candidate_input or alias_output
                    ),
                    partial_assessment=partial_assessment,
                )
                if auxiliary_output
                else _AssessingLocal(
                    finding=finding, evidence_port=evidence_port, alias_evidence=candidate_input or alias_output
                ),
                transport=external[0] if external is not None and item.id == decision_node else None,
                resource=external[1] if external is not None and item.id == decision_node else resource,
            )
            for item, capability in zip(nodes, capabilities, strict=True)
        ),
        context_resources=(),
        limits=execution_limits
        or ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1 if external is not None else 0,
            max_runtime_artifacts=(1 + initial_item_limit + int(auxiliary_output)) * target_count,
            max_runtime_artifact_bytes=100,
            max_collection_items=0,
        ),
        decision_limits=DecisionLimits(
            max_pending=target_count if decision_input else 0, max_lifetime_ns=10 if decision_input else 0
        ),
        clock=_ZeroClock(),
        absence_revisions=((7, 4),) if environment else (),
    )
    running = await start_execution(admitted=admitted, capabilities=capabilities, services=services)
    if decision_input:
        async with asyncio.timeout(2):
            submitted = set()
            while len(submitted) < target_count:
                for wait in running.pending_decisions():
                    if wait.wait not in submitted:
                        running.submit_decision(
                            DecisionResponse(
                                wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve"
                            )
                        )
                        submitted.add(wait.wait)
                await asyncio.sleep(0)
    return admitted, await asyncio.wait_for(running.wait(), 2)
