# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real qualification with independent subject and consumed-context ports."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest

from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    ContextMaterialization,
    ContextSelector,
    ContextSourceRef,
    InitialContextDecl,
    RetrievalBounds,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.data import AtomicGroup, DataGraph, DataLimits, DatumDependency
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission, admit_qualification, evidence_revision_view
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
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    prepare,
)
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.graph import EffectCode, EffectRejected
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    ContextInputRef,
    ContextUse,
    CoverageAtom,
    EvidencePromise,
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
from tests.graph_sdk.implementation_fixtures import _execution_implementation, _selection
from tests.graph_sdk.test_effects_production_conformance import _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.test_evidence import _qualification_limits
from tests.graph_sdk.test_preparation import _capability, _limits, _workflow
from tests.graph_sdk.workflow_construction import DynamicScope, admit_activation_workflow, admit_static_workflow


@dataclass
class _SeparateAssessment:
    finding: AssessmentFinding | None
    target_findings: tuple[tuple[str, AssessmentFinding], ...] = ()
    returned_evidence_port: str = "evidence"
    additional_evidence: bool = False
    calls: int = 0
    consumed_context_ports: frozenset[str] = frozenset()

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        (item,) = request
        assert isinstance(item.association, SemanticAssociation)
        subject = next(port for port in item.inputs if port.port == "subject")
        finding = self.finding
        if self.target_findings:
            assert isinstance(subject.value, TextArtifactValue)
            finding = dict(self.target_findings)[subject.value.text]
        outputs = (
            PortArtifact(port="result", artifact_type=subject.artifact_type, artifact=None, value=subject.value),
        )
        if finding is not None:
            outputs += (
                PortArtifact(
                    port="evidence",
                    artifact_type=subject.artifact_type,
                    artifact=None,
                    value=TextArtifactValue(text="E"),
                ),
            )
            if self.additional_evidence:
                outputs += (
                    PortArtifact(
                        port="evidence2",
                        artifact_type=subject.artifact_type,
                        artifact=None,
                        value=TextArtifactValue(text="E2"),
                    ),
                )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    outputs=outputs,
                    consumed_context_ports=self.consumed_context_ports,
                ),
            ),
            assessments=()
            if finding is None
            else (
                LocalAssessmentResult(
                    association=item.association,
                    promise="checked",
                    evidence_port=self.returned_evidence_port,
                    finding=finding,
                ),
            )
            + (
                (
                    LocalAssessmentResult(
                        association=item.association, promise="complete", evidence_port="evidence2", finding=finding
                    ),
                )
                if self.additional_evidence
                else ()
            ),
        )


class _ContextIdentity:
    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        (item,) = request
        (source,) = item.inputs
        assert source.port == "b" and source.artifact is not None
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port="approved", artifact_type=source.artifact_type, artifact=None, value=source.value
                        ),
                    ),
                    consumed_context_ports=frozenset({"b"}),
                ),
            )
        )


async def _execute_separate_subject_context(
    *,
    extra_evidence_dependency: bool = False,
    coverage: frozenset[CoverageAtom] | None = None,
    requirement_coverage: frozenset[CoverageAtom] | None = None,
    finding: AssessmentFinding | None = None,
    execution_only: bool = False,
    expose_evidence: bool = True,
    target_labels: tuple[str, ...] = ("A",),
    target_findings: tuple[AssessmentFinding, ...] = (),
    dependencies: tuple[tuple[int, int], ...] = (),
    atomic_groups: tuple[tuple[int, ...], ...] = (),
    returned_evidence_port: str = "evidence",
    provenance_edge_limit: int | None = None,
    additional_coverage: frozenset[CoverageAtom] | None = None,
    versioned_context_dependencies: frozenset[str] | None = None,
    versioned_context_ports: tuple[str, ...] = ("a", "b"),
    ordinary_context_hop: bool = False,
    context_from_subject: bool = False,
    independent_context_sources: bool = False,
    context_artifact_type: ArtifactType | None = None,
    initial_subject: bool = False,
    initial_collection: bool = False,
) -> tuple[AdmittedExecutionPlan, ExecutionResult]:
    base, node, artifact = _workflow(with_input=True)
    raw = base
    coverage = coverage if coverage is not None else frozenset({CoverageAtom(kind="field", name="text")})
    assert versioned_context_ports in {("a",), ("b",), ("a", "b")}
    assert not initial_subject or versioned_context_dependencies is None
    assert not initial_collection or versioned_context_dependencies is not None
    if initial_collection:
        context_artifact_type = ArtifactType(name="text_collection", revision=1)
    context_ports = ("context",) if versioned_context_dependencies is None else versioned_context_ports
    consumed = frozenset(context_ports) if versioned_context_dependencies is None else versioned_context_dependencies
    assert consumed <= frozenset(context_ports)
    assert not context_from_subject or len(target_labels) == 1
    assert context_artifact_type is None or versioned_context_dependencies is None or initial_collection
    assert not ordinary_context_hop or (versioned_context_dependencies is not None and "b" in context_ports)
    assert versioned_context_dependencies is None or additional_coverage is None
    initial_ports = (
        (("subject",) if initial_subject else context_ports)
        if (initial_subject or versioned_context_dependencies is not None)
        else ()
    )
    operation = replace(
        raw.interface,
        inputs=tuple(
            InputPort(
                name=name,
                artifact_type=context_artifact_type
                if name != "subject" and context_artifact_type is not None
                else artifact,
            )
            for name in ("subject", *context_ports)
        ),
        outputs=tuple(OutputPort(name=name, artifact_type=artifact) for name in ("result", "evidence")),
        output_dependencies=(
            OutputDependency(
                output="result",
                inputs=frozenset({"subject", *context_ports}),
                identity_input="subject",
            ),
            OutputDependency(
                output="evidence",
                inputs=consumed | {"subject"} if extra_evidence_dependency else consumed,
                identity_input=None,
            ),
        ),
        outcomes=(
            replace(
                raw.interface.outcomes[0],
                produced_ports=frozenset({"result", "evidence"}),
                context=frozenset(
                    ContextUse(port=port, meaning="source", capture="whole_artifact") for port in initial_ports
                )
                if initial_ports
                else frozenset(),
                evidence=frozenset(
                    {
                        EvidencePromise(
                            name="checked",
                            meaning="test assessment",
                            subject_port="subject",
                            consumed_ports=consumed,
                            coverage=coverage,
                        )
                    }
                ),
                ceiling=replace(raw.interface.outcomes[0].ceiling, max_input_bytes=32, max_output_bytes=32),
            ),
        ),
    )
    if additional_coverage is not None:
        assert not execution_only and not expose_evidence
        operation = replace(
            operation,
            outputs=(*operation.outputs, OutputPort(name="evidence2", artifact_type=artifact)),
            output_dependencies=(
                *operation.output_dependencies,
                OutputDependency(output="evidence2", inputs=frozenset({"context"}), identity_input=None),
            ),
            outcomes=tuple(
                replace(
                    outcome,
                    produced_ports=outcome.produced_ports | {"evidence2"},
                    evidence=outcome.evidence
                    | {
                        EvidencePromise(
                            name="complete",
                            meaning="test assessment",
                            subject_port="subject",
                            consumed_ports=frozenset({"context"}),
                            coverage=additional_coverage,
                        )
                    },
                )
                for outcome in operation.outcomes
            ),
        )
    if execution_only:
        expose_evidence = False
        operation = replace(
            operation,
            outputs=tuple(port for port in operation.outputs if port.name == "result"),
            output_dependencies=tuple(dep for dep in operation.output_dependencies if dep.output == "result"),
            outcomes=tuple(
                replace(outcome, produced_ports=frozenset({"result"}), evidence=frozenset())
                for outcome in operation.outcomes
            ),
        )
    hop_node = NodeId.new(workflow=raw.workflow)
    hop_operation = replace(
        operation,
        inputs=(InputPort(name="b", artifact_type=artifact),),
        outputs=(OutputPort(name="approved", artifact_type=artifact),),
        output_dependencies=(OutputDependency(output="approved", inputs=frozenset({"b"}), identity_input="b"),),
        outcomes=tuple(
            replace(
                outcome,
                produced_ports=frozenset({"approved"}),
                evidence=frozenset(),
                context=frozenset({ContextUse(port="b", meaning="source", capture="whole_artifact")}),
            )
            for outcome in operation.outcomes
        ),
    )
    nodes = (OperationNode(id=node, operation=operation),)
    if ordinary_context_hop:
        nodes = (OperationNode(id=hop_node, operation=hop_operation), *nodes)
    interface = replace(
        operation,
        outcomes=tuple(
            replace(
                outcome,
                ceiling=replace(
                    outcome.ceiling,
                    max_activations=len(nodes),
                    max_input_bytes=32 * len(nodes),
                    max_output_bytes=32 * len(nodes),
                ),
            )
            for outcome in operation.outcomes
        ),
    )
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=interface
        if expose_evidence
        else replace(
            interface,
            outputs=tuple(port for port in operation.outputs if port.name == "result"),
            output_dependencies=tuple(dep for dep in operation.output_dependencies if dep.output == "result"),
            outcomes=tuple(replace(outcome, produced_ports=frozenset({"result"})) for outcome in interface.outcomes),
        ),
        nodes=nodes,
        input_bindings=tuple(
            InputBinding(
                source=NodeOutputRef(node=hop_node, port="approved")
                if ordinary_context_hop and name == "b"
                else ContextInputRef(port=name)
                if name in initial_ports
                else WorkflowInputRef(port=name),
                destination=NodeInputRef(node=node, port=name),
            )
            for name in ("subject", *context_ports)
        )
        + (
            (InputBinding(source=ContextInputRef(port="b"), destination=NodeInputRef(node=hop_node, port="b")),)
            if ordinary_context_hop
            else ()
        ),
        output_bindings=tuple(
            OutputBinding(source=NodeOutputRef(node=node, port=name), destination=WorkflowOutputRef(port=name))
            for name in (("result", "evidence") if expose_evidence else ("result",))
        ),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(SequenceEdge(before=hop_node, after=node),) if ordinary_context_hop else (),
        choices=(),
        protection=()
        if execution_only
        else (
            ProtectionRequirement(
                outcome="ok",
                meaning="test assessment",
                subject_port="subject",
                consumed_ports=consumed,
                coverage=coverage if requirement_coverage is None else requirement_coverage,
            ),
        ),
        limits=replace(
            raw.limits,
            max_bindings=4 + len(context_ports) + int(ordinary_context_hop),
            max_nodes=len(nodes),
            max_sequence_edges=int(ordinary_context_hop),
        ),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),),
        limits=replace(base.dynamic_limits, max_activation_occurrences=len(nodes)),
    )
    capabilities = tuple(
        replace(
            _capability(workflow),
            operation=next(declared.operation for declared in workflow.operations if declared.id == item.id),
            implementation=ImplementationRef(name=f"separate-context-{index}", revision=1),
        )
        for index, item in enumerate(nodes)
    )
    capability = capabilities[-1]
    graph = DataGraph.new()
    targets = []
    for label in target_labels:
        graph, target = graph.add_text(label)
        targets.append(target)
    context_sources = {}
    for index, target in enumerate(targets):
        if index == 0 or independent_context_sources:
            graph, context_source = graph.add_text("X")
        context_sources[target] = context_source
    context_count = len(set(context_sources.values()))
    target_count = len(targets)
    data = graph.validate(
        targets=tuple(targets),
        source_relations=(),
        contexts=(),
        dependencies=tuple(DatumDependency(prerequisite=targets[a], dependent=targets[b]) for a, b in dependencies),
        coherence=(),
        atomic=tuple(AtomicGroup(members=tuple(targets[index] for index in group)) for group in atomic_groups),
        output_regions=(),
        limits=DataLimits(
            max_datums=target_count + context_count,
            max_targets=target_count,
            max_text_bytes=sum(len(label.encode()) for label in target_labels) + context_count,
            max_declarations=len(dependencies) + len(atomic_groups),
            max_group_members=target_count,
        ),
    )
    bound_context = None
    if initial_ports:
        from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource

        providers = tuple(
            _BindingProvider(ContextSourceRef(name=port, revision=1), versions=(1, 2)) for port in initial_ports
        )
        resources = tuple(_initial_resource(provider) for provider in providers)
        binding = await (
            await start_initial_binding(
                data=data,
                workflow=workflow,
                declarations=tuple(
                    InitialContextDecl(
                        target=target,
                        node=hop_node if ordinary_context_hop and port == "b" else node,
                        port=port,
                        artifact_type=context_artifact_type
                        if context_artifact_type is not None and initial_collection
                        else artifact,
                        source=resource.source,
                        selector=ContextSelector(fields=()),
                        requirement="required",
                        bounds=RetrievalBounds(max_items=2, max_bytes=2, max_requests=1),
                        materialization=ContextMaterialization(
                            kind="collection" if initial_collection else "single", item_type=artifact
                        ),
                        version_selection="exact_one" if initial_collection else "latest",
                    )
                    for target in targets
                    for port, resource in zip(initial_ports, resources, strict=True)
                ),
                capabilities=tuple(resource.capability for resource in resources),
                resources=resources,
                limits=BindingLimits(
                    max_declarations=2 * target_count,
                    max_sources=2,
                    max_capabilities=2,
                    max_selector_fields=0,
                    max_selector_bytes=0,
                    max_items=4 * target_count,
                    max_bytes=4 * target_count,
                    max_requests=2 * target_count,
                    max_resources=2,
                ),
            )
        ).wait()
        assert binding.receipt.terminal == "success" and binding.context is not None
        assert all(provider.calls == target_count and provider.closes == 1 for provider in providers)
        bound_context = binding.context
    prepared = prepare(
        workflow=workflow,
        capabilities=capabilities,
        selections=tuple(_selection(item.id, cap) for item, cap in zip(nodes, capabilities, strict=True)),
        limits=_limits(capabilities=len(capabilities)),
        activation_limits=ActivationLimits(max_events=3 * len(nodes), max_entries=len(nodes), max_parent_depth=1),
        data=data,
        configuration=PreparationConfiguration(
            purpose="execution_only" if execution_only else "protection",
            required_protection_outcomes=frozenset() if execution_only else frozenset({"ok"}),
            hard_request_limit=None,
        ),
        bound_inputs=tuple(
            BoundInput(
                target=target,
                source=target if context_from_subject else source,
                port=name,
                artifact_type=context_artifact_type
                if name != "subject" and context_artifact_type is not None
                else artifact,
            )
            for target in targets
            for name, source in (("subject", target), ("context", context_sources[target]))
            if name not in initial_ports and (versioned_context_dependencies is None or name == "subject")
        ),
    )
    finding = finding if finding is not None else AssessmentFinding(status="satisfied", code="observed")
    assert not target_findings or len(target_findings) == target_count
    supported_findings = frozenset(target_findings) if target_findings else frozenset({finding})
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=bound_context, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=(
            OperationExecutionPolicy(
                node=node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(_execution_implementation(capability, request=None),),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_valid_runtime_rows("local", frozenset({"ok"})),
            ),
        )
        + (
            (
                OperationExecutionPolicy(
                    node=hop_node,
                    kind="local",
                    request=None,
                    safe_detachment="forbidden",
                    implementations=(_execution_implementation(capabilities[0], request=None),),
                    result_outcomes=frozenset({"ok"}),
                    runtime_outcomes=_valid_runtime_rows("local", frozenset({"ok"})),
                ),
            )
            if ordinary_context_hop
            else ()
        ),
        assessment_productions=()
        if execution_only
        else tuple(
            EvidenceProductionDecl(
                node=node,
                outcome="ok",
                promise=promise,
                evidence_port=port,
                supported_findings=supported_findings,
            )
            for promise, port in (
                ("checked", "evidence"),
                *((("complete", "evidence2"),) if additional_coverage is not None else ()),
            )
        ),
        assessment_limits=AssessmentLimits(
            max_productions=1 + (additional_coverage is not None),
            max_findings_per_production=len(supported_findings),
            max_finding_code_bytes=16,
            max_assessment_facts=(1 + (additional_coverage is not None)) * target_count,
            max_port_facts=(3 + len(context_ports) + 2 * int(ordinary_context_hop) + (additional_coverage is not None))
            * target_count,
            max_provenance_edges=(
                1
                + len(context_ports)
                + len(consumed)
                + int(ordinary_context_hop)
                + 2 * int(initial_collection)
                + (additional_coverage is not None)
            )
            * target_count
            if provenance_edge_limit is None
            else provenance_edge_limit,
        ),
    )
    callback = _SeparateAssessment(
        None if execution_only else finding,
        tuple(zip(target_labels, target_findings, strict=True)) if target_findings else (),
        returned_evidence_port,
        additional_coverage is not None,
        consumed_context_ports=frozenset(initial_ports),
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=capabilities,
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
            )
            + (
                (
                    ImplementationHandle(
                        implementation=capabilities[0].implementation,
                        operation=hop_operation,
                        configuration=capabilities[0].configuration,
                        local=_ContextIdentity(),
                        transport=None,
                        resource=None,
                    ),
                )
                if ordinary_context_hop
                else ()
            ),
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=0,
                max_runtime_artifacts=(3 + (additional_coverage is not None) + (3 if initial_ports else 0))
                * target_count,
                max_runtime_artifact_bytes=32 * target_count,
                max_collection_items=2 if initial_collection else 0,
            ),
            clock=_ZeroClock(),
        ),
    )
    result = await running.wait()
    assert callback.calls == target_count
    return admitted, result


@pytest.mark.parametrize("missing", [None, "subject", "context", "evidence"])
def test_assessment_subject_and_consumed_context_remain_distinct(missing: str | None) -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    ports = {port.port: port.artifact for port in result.ports}
    assert ports["result"] == ports["subject"]
    assert len({ports["subject"], ports["context"], ports["evidence"]}) == 3
    values = dict(result.artifacts)
    assert values[ports["subject"]] == TextArtifactValue(text="A")
    assert values[ports["context"]] == TextArtifactValue(text="X")
    assert values[ports["evidence"]] == TextArtifactValue(text="E")
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts if missing is None or ref != ports[missing]),
        configurations=((result.assessments[0].node, result.assessments[0].environment.configuration),),
    )
    output = qualify(
        admitted=admitted,
        result=result,
        current=current,
        submissions=(AssessmentSubmission(fact=result.assessments[0]),),
    )
    assert bool(output.qualified) == (missing is None)
    (verified,) = output.verified
    assert verified.subject.artifact == ports["subject"]
    assert verified.reference.consumed == frozenset({ports["context"]})
    assert verified.reference.artifact == ports["evidence"]


def test_evidence_dependency_must_equal_the_promised_consumed_ports() -> None:
    with pytest.raises(EffectRejected) as error:
        asyncio.run(_execute_separate_subject_context(extra_evidence_dependency=True))
    assert error.value.code is EffectCode.CONTRADICTORY


def test_execution_only_has_no_assessment_production_or_callback_result() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context(execution_only=True, expose_evidence=False))
    assert execution.assessment_productions == ()
    assert result.assessments == ()
    admitted = admit_qualification(execution=execution, productions=(), limits=_qualification_limits())
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=(),
        configurations=(),
    )
    output = qualify(admitted=admitted, result=result, current=current, submissions=())
    assert output.qualified == output.verified == ()
    assert output.targets[0].candidate is None
    assert output.targets[0].withholding == frozenset({"execution_only"})
    assert output.record.statuses[0].qualification == "not_assessed"
    assert output.record.statuses[0].artifact_available
    assert not output.record.statuses[0].protection_available


def test_wrong_callback_evidence_port_fails_before_retaining_assessment() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context(returned_evidence_port="wrong"))
    assert len(execution.assessment_productions) == 1
    assert result.assessments == ()
    assert result.final_outputs == ()
    assert result.record.terminals[0].category == "failure"
    assert len(result.artifacts) == 2
    assert all(value != TextArtifactValue(text="E") for _, value in result.artifacts)
