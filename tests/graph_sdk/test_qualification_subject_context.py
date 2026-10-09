# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real qualification with independent subject and consumed-context ports."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import AtomicGroup, DataGraph, DataLimits, DatumDependency
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission, admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentFinding,
    AssessmentLimits,
    DecisionLimits,
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
)
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.graph.workflow import (
    CoverageAtom,
    DynamicScope,
    EvidencePromise,
    InputBinding,
    InputPort,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ProtectionRequirement,
    StateEffect,
    WorkflowInputRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_effects_production_conformance import _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.test_evidence import _qualification_limits
from tests.graph_sdk.test_preparation import _capability, _prepare, _workflow


@dataclass
class _SeparateAssessment:
    finding: AssessmentFinding | None
    target_findings: tuple[tuple[str, AssessmentFinding], ...] = ()
    returned_evidence_port: str = "evidence"
    additional_evidence: bool = False
    calls: int = 0

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
                    consumed_context_ports=frozenset(),
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


async def _execute_separate_subject_context(
    *,
    extra_evidence_dependency: bool = False,
    coverage: frozenset[CoverageAtom] | None = None,
    requirement_coverage: frozenset[CoverageAtom] | None = None,
    environment: bool = False,
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
) -> tuple[AdmittedExecutionPlan, ExecutionResult]:
    base, node, artifact = _workflow(with_input=True)
    raw = base.workflow
    coverage = coverage if coverage is not None else frozenset({CoverageAtom(kind="field", name="text")})
    read = StateEffect(kind="read", name="read")
    operation = replace(
        raw.interface,
        inputs=tuple(InputPort(name=name, artifact_type=artifact) for name in ("subject", "context")),
        outputs=tuple(OutputPort(name=name, artifact_type=artifact) for name in ("result", "evidence")),
        output_dependencies=(
            OutputDependency(output="result", inputs=frozenset({"subject", "context"}), identity_input="subject"),
            OutputDependency(
                output="evidence",
                inputs=frozenset({"subject", "context"}) if extra_evidence_dependency else frozenset({"context"}),
                identity_input=None,
            ),
        ),
        outcomes=(
            replace(
                raw.interface.outcomes[0],
                produced_ports=frozenset({"result", "evidence"}),
                state_effects=frozenset({read}) if environment else frozenset(),
                evidence=frozenset(
                    {
                        EvidencePromise(
                            name="checked",
                            meaning="test assessment",
                            subject_port="subject",
                            consumed_ports=frozenset({"context"}),
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
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=operation
        if expose_evidence
        else replace(
            operation,
            outputs=tuple(port for port in operation.outputs if port.name == "result"),
            output_dependencies=tuple(dep for dep in operation.output_dependencies if dep.output == "result"),
            outcomes=tuple(replace(outcome, produced_ports=frozenset({"result"})) for outcome in operation.outcomes),
        ),
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=tuple(
            InputBinding(source=WorkflowInputRef(port=name), destination=NodeInputRef(node=node, port=name))
            for name in ("subject", "context")
        ),
        output_bindings=tuple(
            OutputBinding(source=NodeOutputRef(node=node, port=name), destination=WorkflowOutputRef(port=name))
            for name in (("result", "evidence") if expose_evidence else ("result",))
        ),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(),
        choices=(),
        protection=()
        if execution_only
        else (
            ProtectionRequirement(
                outcome="ok",
                meaning="test assessment",
                subject_port="subject",
                consumed_ports=frozenset({"context"}),
                coverage=coverage if requirement_coverage is None else requirement_coverage,
            ),
        ),
        limits=replace(raw.limits, max_bindings=5),
    )
    workflow = admit_activation_workflow(
        workflow=static, scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),), limits=base.limits
    )
    capability = _capability(workflow)
    graph = DataGraph.new()
    targets = []
    for label in target_labels:
        graph, target = graph.add_text(label)
        targets.append(target)
    graph, context_source = graph.add_text("X")
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
            max_datums=target_count + 1,
            max_targets=target_count,
            max_text_bytes=sum(len(label.encode()) for label in target_labels) + 1,
            max_declarations=len(dependencies) + len(atomic_groups),
            max_group_members=target_count,
        ),
    )
    prepared = _prepare(
        workflow=workflow,
        capability=capability,
        data=data,
        configuration=PreparationConfiguration(
            purpose="execution_only" if execution_only else "protection",
            required_protection_outcomes=frozenset() if execution_only else frozenset({"ok"}),
            hard_request_limit=None,
        ),
        bound_inputs=tuple(
            BoundInput(target=target, source=source, port=name, artifact_type=artifact)
            for target in targets
            for name, source in (("subject", target), ("context", context_source))
        ),
        state=StateRevisionView(
            revisions=frozenset({StateRevision(effect=read, revision=1)}) if environment else frozenset()
        ),
    )
    finding = finding if finding is not None else AssessmentFinding(status="satisfied", code="observed")
    assert not target_findings or len(target_findings) == target_count
    supported_findings = frozenset(target_findings) if target_findings else frozenset({finding})
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=node,
                kind="local",
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
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_valid_runtime_rows("local", frozenset({"ok"})),
            ),
        ),
        decisions=(),
        assessment_productions=()
        if execution_only
        else tuple(
            EvidenceProductionDecl(
                node=node,
                outcome="ok",
                promise=promise,
                evidence_port=port,
                absence_queries=frozenset({0}) if environment else frozenset(),
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
            max_absence_queries=1 if environment else 0,
            max_assessment_facts=(1 + (additional_coverage is not None)) * target_count,
            max_port_facts=(4 + (additional_coverage is not None)) * target_count,
            max_provenance_edges=(3 + (additional_coverage is not None)) * target_count
            if provenance_edge_limit is None
            else provenance_edge_limit,
        ),
    )
    callback = _SeparateAssessment(
        None if execution_only else finding,
        tuple(zip(target_labels, target_findings, strict=True)) if target_findings else (),
        returned_evidence_port,
        additional_coverage is not None,
    )
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
                max_runtime_artifacts=(3 + (additional_coverage is not None)) * target_count,
                max_runtime_artifact_bytes=32 * target_count,
                max_collection_items=0,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_ZeroClock(),
            absence_revisions=((0, 1),) if environment and not execution_only else (),
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
        absences=(),
        configurations=((result.assessments[0].node, result.assessments[0].environment.configuration),),
        state=execution.context.prepared.state,
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
        absences=(),
        configurations=(),
        state=StateRevisionView(revisions=frozenset()),
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
