# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real assessment execution fixtures for evidence and qualification tests."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, fields, replace

from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef
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
    EvidenceProductionDecl,
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
    prepare,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PhysicalRequestPolicy,
    PortArtifact,
    TextArtifactValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph import (
    ActivationLimits,
    ContextInputRef,
    ContextUse,
    CoverageAtom,
    DatumId,
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
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
)
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.implementation_fixtures import _execution_implementation, _selection
from tests.graph_sdk.test_assessment_executor import _AssessingLocal, _single_output_workflow
from tests.graph_sdk.test_preparation import _capability, _data, _limits
from tests.graph_sdk.workflow_construction import DynamicScope, admit_activation_workflow, admit_static_workflow

# Shared hang-guard bound for test-side waits on normally-fast in-memory async
# operations (events, tasks, futures). This is not a measurement of expected
# runtime -- it only bounds how long a test will wait before failing if the
# code under test hangs. Keep it generous so CI hosts under load (parallel
# test workers, noisy neighbors) don't trip a timeout that would never fire
# in a healthy run; it should never be tightened to assert timing behavior.
HANG_GUARD_TIMEOUT = 30.0


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


@dataclass
class _IdentityHop:
    port: str

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        item = request[0]
        source = item.inputs[0]
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        PortArtifact(
                            port=self.port, artifact_type=source.artifact_type, artifact=None, value=source.value
                        ),
                    ),
                ),
            )
        )


async def _execute_assessment(
    *,
    candidate_input: bool = False,
    alias_output: bool = False,
    auxiliary_output: bool = False,
    partial_assessment: bool = False,
    root_passthrough: bool = False,
    predecessor_input: bool = False,
    predecessor_hop: bool = False,
    external_failover: bool = False,
    external_shared: bool = False,
    coverage: frozenset[CoverageAtom] = frozenset(),
    target_count: int = 1,
    nested: bool = False,
    nested_depth: int = 0,
    rename_ports: bool = False,
    data: ValidatedDataGraph | None = None,
    bound_input_source: DatumId | None = None,
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
    assert not (predecessor_input and external is not None)
    assert not (auxiliary_output and (rename_ports or nested or predecessor_input or external is not None))
    assert not initial_resources or not (nested or rename_ports or predecessor_input or external or root_passthrough)
    assert not predecessor_hop or predecessor_input
    assert not external_shared or external is not None
    assert not external_failover or external is not None
    predecessor = predecessor_input or external is not None
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
    base, node, artifact = _single_output_workflow(
        assessment=True, requests=0, distinct_subject=candidate_input, alias_evidence=candidate_input or alias_output
    )
    raw = base
    operation = replace(
        raw.interface,
        outcomes=tuple(
            replace(
                outcome,
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
    predecessor_node = NodeId.new(workflow=raw.workflow)
    nodes = (OperationNode(id=node, operation=operation),)
    bindings = tuple(
        replace(item, source=ContextInputRef(port="input")) if initial_resources else item
        for item in raw.input_bindings
    )
    if predecessor:
        predecessor_operation = replace(
            operation,
            name="approve-input",
            outcomes=tuple(
                replace(
                    outcome,
                    evidence=frozenset(),
                    ceiling=replace(outcome.ceiling, max_model_requests=2 if external is not None else 0),
                )
                for outcome in operation.outcomes
            ),
            output_dependencies=tuple(
                replace(dependency, identity_input="input") for dependency in operation.output_dependencies
            ),
        )
        nodes = (OperationNode(id=predecessor_node, operation=predecessor_operation), *nodes)
        bindings = (
            InputBinding(
                source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=predecessor_node, port="input")
            ),
            InputBinding(
                source=NodeOutputRef(node=predecessor_node, port=evidence_port),
                destination=NodeInputRef(node=node, port="input"),
            ),
        )
    hop = NodeId.new(workflow=raw.workflow)
    if predecessor_hop:
        nodes = (nodes[0], OperationNode(id=hop, operation=predecessor_operation), *nodes[1:])
        bindings = (
            bindings[0],
            replace(bindings[1], destination=NodeInputRef(node=hop, port="input")),
            InputBinding(
                source=NodeOutputRef(node=hop, port=evidence_port), destination=NodeInputRef(node=node, port="input")
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
        sequence=(SequenceEdge(before=predecessor_node, after=hop), SequenceEdge(before=hop, after=node))
        if predecessor_hop
        else (SequenceEdge(before=predecessor_node, after=node),)
        if predecessor
        else (),
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
        limits=replace(
            raw.limits,
            max_nodes=len(nodes),
            max_bindings=4 + int(predecessor_hop),
            max_sequence_edges=1 + int(predecessor_hop),
        ),
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
        limits=replace(base.dynamic_limits, max_activation_occurrences=count, max_dynamic_depth=depth + 1),
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
            _capability(workflow, external=external is not None and item.id == predecessor_node),
            operation=item.operation,
            max_physical_requests_per_activation=2 if external is not None and item.id == predecessor_node else 0,
            resource_lifetime=("caller_owned" if external[1].owner == "caller" else "executor_owned")
            if external is not None and item.id == predecessor_node
            else "stateless"
            if resource is None
            else "caller_owned"
            if resource.owner == "caller"
            else "executor_owned",
            implementation=ImplementationRef(name=f"evidence-fixture-{index}", revision=1),
        )
        for index, item in enumerate(nodes)
    )
    if external_shared:
        capabilities = (replace(capabilities[0], attribution="keyed_shared_request"), *capabilities[1:])
    fallback = replace(capabilities[0], implementation=ImplementationRef(name="evidence-fixture-fallback", revision=1))
    all_capabilities = (*capabilities, fallback) if external_failover else capabilities
    prepared = prepare(
        data=data,
        workflow=workflow,
        capabilities=all_capabilities,
        activation_limits=ActivationLimits(max_events=3 * count, max_entries=count, max_parent_depth=depth + 1),
        selections=tuple(_selection(item.id, capability) for item, capability in zip(nodes, capabilities, strict=True)),
        limits=_limits(capabilities=len(all_capabilities)),
        configuration=PreparationConfiguration(
            purpose="execution_only" if execution_only else "protection",
            required_protection_outcomes=frozenset() if execution_only else frozenset({"ok"}),
            hard_request_limit=2 * target_count if external is not None else None,
        ),
        bound_inputs=()
        if initial_resources
        else tuple(
            BoundInput(
                target=target,
                source=target if bound_input_source is None else bound_input_source,
                port=root_input,
                artifact_type=artifact,
            )
            for target in data.targets
        ),
    )
    finding = finding if finding is not None else AssessmentFinding(status="satisfied", code="observed")
    declaration = EvidenceProductionDecl(
        node=node,
        outcome="ok",
        promise="checked",
        evidence_port=evidence_port,
        supported_findings=frozenset({finding}),
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(prepared=prepared, bound_context=bound_context),
        capabilities=all_capabilities,
        policies=tuple(
            OperationExecutionPolicy(
                node=item.id,
                kind="external" if external is not None and item.id == predecessor_node else "local",
                request=request_policy if item.id == predecessor_node else None,
                safe_detachment="forbidden",
                implementations=(
                    _execution_implementation(
                        capability, request=request_policy if item.id == predecessor_node else None
                    ),
                )
                + (
                    (_execution_implementation(fallback, request=request_policy),)
                    if external_failover and item.id == predecessor_node
                    else ()
                ),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_valid_runtime_rows(
                    "external" if external is not None and item.id == predecessor_node else "local",
                    frozenset({"ok"}),
                ),
            )
            for item, capability in zip(nodes, capabilities, strict=True)
        ),
        assessment_productions=(declaration, replace(declaration, promise="partial", evidence_port="metadata"))
        if partial_assessment
        else (declaration,),
        assessment_limits=AssessmentLimits(
            max_productions=1 + int(partial_assessment),
            max_findings_per_production=1,
            max_finding_code_bytes=20,
            max_assessment_facts=(1 + int(partial_assessment)) * target_count,
            max_port_facts=(3 + 2 * int(predecessor) + 2 * int(predecessor_hop) + depth + int(auxiliary_output))
            * target_count,
            max_provenance_edges=(2 + int(predecessor) + int(predecessor_hop) + depth + int(auxiliary_output))
            * target_count
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
                if external is not None and item.id == predecessor_node
                else _IdentityHop(evidence_port)
                if item.id == predecessor_node or predecessor_hop and item.id == hop
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
                transport=external[0] if external is not None and item.id == predecessor_node else None,
                resource=external[1] if external is not None and item.id == predecessor_node else resource,
            )
            for item, capability in zip(nodes, capabilities, strict=True)
        ),
        limits=execution_limits
        or ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1 if external is not None else 0,
            max_runtime_artifacts=(1 + initial_item_limit + int(auxiliary_output)) * target_count,
            max_runtime_artifact_bytes=100,
            max_collection_items=0,
        ),
        clock=_ZeroClock(),
    )
    if external_failover:
        services = replace(
            services,
            handles=(
                *services.handles,
                replace(
                    services.handles[0],
                    implementation=fallback.implementation,
                ),
            ),
        )
    running = await start_execution(admitted=admitted, capabilities=all_capabilities, services=services)
    return admitted, await asyncio.wait_for(running.wait(), HANG_GUARD_TIMEOUT)
