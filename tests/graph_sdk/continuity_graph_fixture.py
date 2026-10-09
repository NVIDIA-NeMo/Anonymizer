# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real shared-decision graph and retained evidence for continuity witnesses."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, replace

from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import AtomicGroup, DataGraph, DataLimits, DatumDependency, ValidatedDataGraph
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    EvidenceRevisionView,
    admit_qualification,
    evidence_revision_view,
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
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.graph._values import DatumId
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    CoverageAtom,
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
    ProtectionRequirement,
    ResourceCeiling,
    SequenceEdge,
    StateEffect,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.evidence_fixtures import _qualification_limits
from tests.graph_sdk.test_decision_scheduling import _Callback
from tests.graph_sdk.test_effects_production_conformance import _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.test_preparation import _capability, _limits


@dataclass(frozen=True)
class GraphCase:
    requirements: tuple[tuple[str, tuple[str, ...]], ...]
    support_targets: tuple[tuple[str, str], ...]
    dependencies: tuple[tuple[str, str], ...] = ()
    atomic: tuple[tuple[str, ...], ...] = ()
    candidate_dependencies: tuple[tuple[str, tuple[str, ...]], ...] = ()


CASES = {
    "independent": GraphCase((("A", ("x",)), ("B", ("x", "y"))), (("x", "A"), ("y", "B"))),
    "sole_b": GraphCase((("A", ("x",)), ("B", ("x", "y"))), (("x", "B"), ("y", "B"))),
    "sole_b_without_y": GraphCase(
        (("A", ("x",)), ("B", ("x",)), ("Y", ("y",))), (("x", "B"), ("y", "Y")), candidate_dependencies=(("B", ("y",)),)
    ),
    "consumes_y": GraphCase((("A", ("x", "y")), ("B", ("x", "y"))), (("x", "A"), ("y", "B"))),
    "atomic": GraphCase((("A", ("x",)), ("B", ("x", "y"))), (("x", "A"), ("y", "B")), atomic=(("A", "B"),)),
    "alternating": GraphCase(
        (("A", ("x",)), ("B", ("x", "y")), ("D", ("x", "z")), ("C", ("z",)), ("E", ()), ("F", ()), ("U", ())),
        (("x", "A"), ("y", "B"), ("z", "D")),
        dependencies=(("B", "D"), ("E", "F")),
        atomic=(("C", "E"),),
    ),
}


@dataclass
class _Verifier:
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        (item,) = request
        assert isinstance(item.association, SemanticAssociation)
        subject = next(port for port in item.inputs if port.port == "subject")
        assert isinstance(subject.value, TextArtifactValue)
        name = subject.value.text
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome=name,
                    outputs=(
                        PortArtifact(
                            port=f"result_{name}",
                            artifact_type=subject.artifact_type,
                            artifact=None,
                            value=subject.value,
                        ),
                        PortArtifact(
                            port=f"evidence_{name}",
                            artifact_type=subject.artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text="checked"),
                        ),
                    ),
                    consumed_context_ports=frozenset(),
                ),
            ),
            assessments=(
                LocalAssessmentResult(
                    association=item.association,
                    promise=f"support_{name}",
                    evidence_port=f"evidence_{name}",
                    finding=AssessmentFinding(status="satisfied", code="checked"),
                ),
            ),
        )


@dataclass(frozen=True)
class GraphWitness:
    case: GraphCase
    targets: tuple[tuple[str, DatumId], ...]
    decision_nodes: tuple[tuple[str, NodeId], ...]
    assessor: NodeId
    coverage: frozenset[CoverageAtom]
    execution: AdmittedExecutionPlan
    result: ExecutionResult
    qualified: QualificationResult
    current: EvidenceRevisionView


async def execute_graph(
    case_name: str,
    *,
    namespace: str = "r4",
    read_revision: int = 1,
    before_prepare: Callable[[ValidatedDataGraph], int] | None = None,
    decision_values: dict[str, str] | None = None,
) -> GraphWitness:
    case = CASES[case_name]
    keys = tuple(sorted({key for _, required in case.requirements for key in required}))
    artifact = ArtifactType(name="text", revision=1)
    owner = WorkflowId.new()
    decision_nodes = {key: NodeId.new(workflow=owner) for key in keys}
    assessor = NodeId.new(workflow=owner)
    read = StateEffect(kind="read", name=namespace)
    write = StateEffect(kind="write", name=namespace)
    coverage = frozenset({CoverageAtom(kind="field", name="assignment")})
    ceiling = ResourceCeiling(max_activations=1, max_model_requests=0, max_input_bytes=128, max_output_bytes=128)
    decision = OperationSpec(
        name="decide",
        inputs=(InputPort(name="input", artifact_type=artifact),),
        outputs=(OutputPort(name="approved", artifact_type=artifact),),
        output_dependencies=(OutputDependency(output="approved", inputs=frozenset({"input"}), identity_input="input"),),
        outcomes=(
            OutcomeSpec(
                name="ok",
                category="success",
                produced_ports=frozenset({"approved"}),
                context=frozenset(),
                evidence=frozenset(),
                state_effects=frozenset({read, write}),
                model_requirements=frozenset(),
                ceiling=ceiling,
            ),
        ),
    )
    verifier = OperationSpec(
        name="verify",
        inputs=tuple(InputPort(name=name, artifact_type=artifact) for name in ("subject", *keys)),
        outputs=(
            *(OutputPort(name=f"result_{name}", artifact_type=artifact) for name, _ in case.requirements),
            *(OutputPort(name=f"evidence_{name}", artifact_type=artifact) for name, _ in case.requirements),
        ),
        output_dependencies=(
            *(
                OutputDependency(
                    output=f"result_{name}",
                    inputs=frozenset({"subject", *dict(case.candidate_dependencies).get(name, ())}),
                    identity_input="subject",
                )
                for name, _ in case.requirements
            ),
            *(
                OutputDependency(output=f"evidence_{name}", inputs=frozenset(required), identity_input=None)
                for name, required in case.requirements
            ),
        ),
        outcomes=tuple(
            OutcomeSpec(
                name=name,
                category="success",
                produced_ports=frozenset({f"result_{name}", f"evidence_{name}"}),
                context=frozenset(),
                evidence=frozenset(
                    {
                        EvidencePromise(
                            name=f"support_{name}",
                            meaning="assignment supported",
                            subject_port="subject",
                            consumed_ports=frozenset(required),
                            coverage=coverage,
                        )
                    }
                ),
                state_effects=frozenset({read}),
                model_requirements=frozenset(),
                ceiling=ceiling,
            )
            for name, required in case.requirements
        ),
    )
    nodes = (
        *(OperationNode(id=decision_nodes[key], operation=decision) for key in keys),
        OperationNode(id=assessor, operation=verifier),
    )
    interface = replace(
        verifier,
        outputs=tuple(port for port in verifier.outputs if port.name.startswith("result_")),
        output_dependencies=tuple(dep for dep in verifier.output_dependencies if dep.output.startswith("result_")),
        outcomes=tuple(
            replace(
                outcome,
                produced_ports=frozenset({f"result_{outcome.name}"}),
                state_effects=frozenset({read, write}),
                ceiling=replace(
                    ceiling,
                    max_activations=len(nodes),
                    max_input_bytes=128 * len(nodes),
                    max_output_bytes=128 * len(nodes),
                ),
            )
            for outcome in verifier.outcomes
        ),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=nodes,
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="subject"), destination=NodeInputRef(node=assessor, port="subject")
            ),
            *(
                InputBinding(
                    source=WorkflowInputRef(port=key), destination=NodeInputRef(node=decision_nodes[key], port="input")
                )
                for key in keys
            ),
            *(
                InputBinding(
                    source=NodeOutputRef(node=decision_nodes[key], port="approved"),
                    destination=NodeInputRef(node=assessor, port=key),
                )
                for key in keys
            ),
        ),
        output_bindings=tuple(
            OutputBinding(
                source=NodeOutputRef(node=assessor, port=f"result_{name}"),
                destination=WorkflowOutputRef(port=f"result_{name}"),
            )
            for name, _ in case.requirements
        ),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=assessor, outcome=name), destination=WorkflowOutcomeRef(outcome=name)
            )
            for name, _ in case.requirements
        ),
        sequence=tuple(SequenceEdge(before=decision_nodes[key], after=assessor) for key in keys),
        choices=(),
        protection=tuple(
            ProtectionRequirement(
                outcome=name,
                meaning="assignment supported",
                subject_port="subject",
                consumed_ports=frozenset(required),
                coverage=coverage,
            )
            for name, required in case.requirements
        ),
        limits=WorkflowLimits(
            max_nodes=len(nodes),
            max_bindings=1 + 2 * len(keys) + 2 * len(case.requirements),
            max_sequence_edges=len(keys),
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=32,
        ),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=1,
            max_activation_occurrences=len(nodes),
        ),
    )
    graph = DataGraph.new()
    targets = {}
    sources = {}
    for name, _ in case.requirements:
        graph, target = graph.add_text(name)
        targets[name] = target
    for key in keys:
        graph, source = graph.add_text(key if decision_values is None else decision_values.get(key, key))
        sources[key] = source
    data = graph.validate(
        targets=tuple(targets.values()),
        source_relations=(),
        contexts=(),
        dependencies=tuple(
            DatumDependency(prerequisite=targets[a], dependent=targets[b]) for a, b in case.dependencies
        ),
        coherence=(),
        atomic=tuple(AtomicGroup(members=tuple(targets[name] for name in group)) for group in case.atomic),
        output_regions=(),
        limits=DataLimits(
            max_datums=len(targets) + len(keys),
            max_targets=len(targets),
            max_text_bytes=128,
            max_declarations=len(case.dependencies) + len(case.atomic),
            max_group_members=len(targets),
        ),
    )
    if before_prepare is not None:
        read_revision = before_prepare(data)
    caps = tuple(
        replace(
            _capability(workflow),
            operation=node.operation,
            implementation=ImplementationRef(name=f"r4_{index}", revision=1),
        )
        for index, node in enumerate(nodes)
    )
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=4 * len(nodes), max_entries=len(nodes), max_parent_depth=1),
        bound_inputs=tuple(
            BoundInput(
                target=target, source=target if port == "subject" else sources[port], port=port, artifact_type=artifact
            )
            for target in targets.values()
            for port in ("subject", *keys)
        ),
        configuration=PreparationConfiguration(
            purpose="protection", required_protection_outcomes=frozenset(targets), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset({StateRevision(effect=read, revision=read_revision)})),
        selections=tuple(
            ImplementationSelection(node=node.id, implementation=cap.implementation, configuration=cap.configuration)
            for node, cap in zip(nodes, caps, strict=True)
        ),
        capabilities=caps,
        limits=replace(
            _limits(capabilities=len(caps), slots=len(nodes) * len(targets)),
            max_bound_inputs=len(targets) * (1 + len(keys)),
        ),
    )
    execution = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=caps,
        policies=tuple(
            OperationExecutionPolicy(
                node=node.id,
                kind="local" if node.id == assessor else "decision",
                request=None,
                safe_detachment="forbidden",
                implementations=(
                    ExecutionImplementation(
                        implementation=cap.implementation, configuration=cap.configuration, capability=cap, request=None
                    ),
                ),
                result_outcomes=frozenset(targets) if node.id == assessor else frozenset(),
                runtime_outcomes=_valid_runtime_rows("local", frozenset(targets))
                if node.id == assessor
                else _valid_runtime_rows("decision", frozenset()),
            )
            for node, cap in zip(nodes, caps, strict=True)
        ),
        decisions=tuple(
            DecisionDeclaration(
                node=node,
                artifact_port="input",
                outcomes=(DecisionOutcome(decision="approve", outcome="ok"),),
                max_lifetime_ns=10,
            )
            for node in decision_nodes.values()
        ),
        assessment_productions=tuple(
            EvidenceProductionDecl(
                node=assessor,
                outcome=name,
                promise=f"support_{name}",
                evidence_port=f"evidence_{name}",
                absence_queries=frozenset(),
                supported_findings=frozenset({AssessmentFinding(status="satisfied", code="checked")}),
            )
            for name in targets
        ),
        assessment_limits=AssessmentLimits(
            max_productions=len(targets),
            max_findings_per_production=1,
            max_finding_code_bytes=16,
            max_absence_queries=0,
            max_assessment_facts=len(targets),
            max_port_facts=128,
            max_provenance_edges=128,
        ),
    )
    callback = _Verifier()
    running = await start_execution(
        admitted=execution,
        capabilities=caps,
        services=ExecutionServices(
            handles=tuple(
                ImplementationHandle(
                    implementation=cap.implementation,
                    operation=node.operation,
                    configuration=cap.configuration,
                    local=callback if node.id == assessor else _Callback(decision=True),
                    transport=None,
                    resource=None,
                )
                for node, cap in zip(nodes, caps, strict=True)
            ),
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=0,
                max_runtime_artifacts=2 * len(targets) + len(keys),
                max_runtime_artifact_bytes=512,
                max_collection_items=0,
            ),
            decision_limits=DecisionLimits(max_pending=len(targets) * len(keys), max_lifetime_ns=10),
            clock=_ZeroClock(),
        ),
    )
    submitted = set()
    async with asyncio.timeout(5):
        while len(submitted) < len(targets) * len(keys):
            for wait in running.pending_decisions():
                if wait.wait not in submitted:
                    running.submit_decision(
                        DecisionResponse(
                            wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve"
                        )
                    )
                    submitted.add(wait.wait)
            await asyncio.sleep(0)
    result = await running.wait()
    assert callback.calls == len(targets)
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=replace(_qualification_limits(max_port_facts=128, max_provenance_edges=128), max_revision_entries=128),
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        absences=(),
        configurations=tuple({(fact.node, fact.environment.configuration) for fact in result.assessments}),
        state=prepared.state,
    )
    submissions = tuple(AssessmentSubmission(fact=fact) for fact in result.assessments)
    qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(qualified.qualified) == len(targets)
    return GraphWitness(
        case,
        tuple(targets.items()),
        tuple(decision_nodes.items()),
        assessor,
        coverage,
        execution,
        result,
        qualified,
        current,
    )
