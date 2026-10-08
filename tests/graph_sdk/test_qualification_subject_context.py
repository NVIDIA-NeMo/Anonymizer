# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real qualification with independent subject and consumed-context ports."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission, admit_qualification, evidence_revision_view
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
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration
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
    finding: AssessmentFinding
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        (item,) = request
        assert isinstance(item.association, SemanticAssociation)
        subject = next(port for port in item.inputs if port.port == "subject")
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port="result", artifact_type=subject.artifact_type, artifact=None, value=subject.value
                        ),
                        PortArtifact(
                            port="evidence",
                            artifact_type=subject.artifact_type,
                            artifact=None,
                            value=TextArtifactValue(text="E"),
                        ),
                    ),
                    consumed_context_ports=frozenset(),
                ),
            ),
            assessments=(
                LocalAssessmentResult(
                    association=item.association, promise="checked", evidence_port="evidence", finding=self.finding
                ),
            ),
        )


async def _execute_separate_subject_context(*, extra_evidence_dependency: bool = False):
    base, node, artifact = _workflow(with_input=True)
    raw = base.workflow
    coverage = frozenset({CoverageAtom(kind="field", name="text")})
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
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=tuple(
            InputBinding(source=WorkflowInputRef(port=name), destination=NodeInputRef(node=node, port=name))
            for name in ("subject", "context")
        ),
        output_bindings=tuple(
            OutputBinding(source=NodeOutputRef(node=node, port=name), destination=WorkflowOutputRef(port=name))
            for name in ("result", "evidence")
        ),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(),
        choices=(),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning="test assessment",
                subject_port="subject",
                consumed_ports=frozenset({"context"}),
                coverage=coverage,
            ),
        ),
        limits=replace(raw.limits, max_bindings=5),
    )
    workflow = admit_activation_workflow(
        workflow=static, scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),), limits=base.limits
    )
    capability = _capability(workflow)
    graph = DataGraph.new()
    graph, target = graph.add_text("A")
    graph, context_source = graph.add_text("X")
    data = graph.validate(
        targets=(target,),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(max_datums=2, max_targets=1, max_text_bytes=2, max_declarations=0, max_group_members=0),
    )
    prepared = _prepare(
        workflow=workflow,
        capability=capability,
        data=data,
        configuration=PreparationConfiguration(
            purpose="protection", required_protection_outcomes=frozenset({"ok"}), hard_request_limit=None
        ),
        bound_inputs=tuple(
            BoundInput(target=target, source=source, port=name, artifact_type=artifact)
            for name, source in (("subject", target), ("context", context_source))
        ),
    )
    finding = AssessmentFinding(status="satisfied", code="observed")
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
        assessment_productions=(
            EvidenceProductionDecl(
                node=node,
                outcome="ok",
                promise="checked",
                evidence_port="evidence",
                absence_queries=frozenset(),
                supported_findings=frozenset({finding}),
            ),
        ),
        assessment_limits=AssessmentLimits(
            max_productions=1,
            max_findings_per_production=1,
            max_finding_code_bytes=16,
            max_absence_queries=0,
            max_assessment_facts=1,
            max_port_facts=4,
            max_provenance_edges=3,
        ),
    )
    callback = _SeparateAssessment(finding)
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
                max_runtime_artifacts=3,
                max_runtime_artifact_bytes=32,
                max_collection_items=0,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_ZeroClock(),
        ),
    )
    result = await running.wait()
    assert callback.calls == 1
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
