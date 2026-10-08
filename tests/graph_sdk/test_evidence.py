# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence authentication through real local execution and retained facts."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from typing import cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    QualificationLimits,
    VerifiedEvidence,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentFinding,
    AssessmentLimits,
    DecisionLimits,
    EvidenceProductionDecl,
    ExecutionAssessmentFact,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView
from anonymizer.engine.graph_sdk.records import CandidateRef
from anonymizer.graph.workflow import (
    DynamicScope,
    ProtectionRequirement,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_adaptive_executor import _adaptive_workflow, _AssessingLocal
from tests.graph_sdk.test_local_executor import _Clock, _runtime_rows
from tests.graph_sdk.test_preparation import _capability, _data, _prepare


def _qualification_limits(**changes: int) -> QualificationLimits:
    return QualificationLimits(**{item.name: changes.get(item.name, 16) for item in fields(QualificationLimits)})


async def _execute_assessment() -> tuple[AdmittedExecutionPlan, ExecutionResult]:
    base, node, artifact = _adaptive_workflow(assessment=True, requests=0)
    raw = base.workflow
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=raw.interface,
        nodes=tuple(raw.nodes),
        input_bindings=tuple(raw.input_bindings),
        output_bindings=tuple(raw.output_bindings),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=tuple(raw.sequence),
        choices=tuple(raw.choices),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning="test assessment",
                subject_port="context",
                consumed_ports=frozenset({"input"}),
                coverage=frozenset(),
            ),
        ),
        limits=raw.limits,
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),),
        limits=base.limits,
    )
    data = _data(1)
    target = next(iter(data.targets))
    capability = _capability(workflow)
    prepared = _prepare(
        data=data,
        workflow=workflow,
        capability=capability,
        configuration=PreparationConfiguration(
            purpose="protection", required_protection_outcomes=frozenset({"ok"}), hard_request_limit=None
        ),
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact),),
    )
    finding = AssessmentFinding(status="satisfied", code="observed")
    declaration = EvidenceProductionDecl(
        node=node,
        outcome="ok",
        promise="checked",
        evidence_port="context",
        absence_queries=frozenset(),
        supported_findings=frozenset({finding}),
    )
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
                runtime_outcomes=_runtime_rows(),
            ),
        ),
        decisions=(),
        assessment_productions=(declaration,),
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
    services = ExecutionServices(
        handles=(
            ImplementationHandle(
                implementation=capability.implementation,
                operation=capability.operation,
                configuration=capability.configuration,
                local=_AssessingLocal(finding=finding, evidence_port="context", alias_evidence=False),
                transport=None,
                resource=None,
            ),
        ),
        context_resources=(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=0,
            max_runtime_artifacts=2,
            max_runtime_artifact_bytes=100,
            max_collection_items=0,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_Clock(),
    )
    running = await start_execution(admitted=admitted, capabilities=(capability,), services=services)
    return admitted, await running.wait()


@pytest.fixture
def assessment_execution() -> tuple[AdmittedExecutionPlan, ExecutionResult]:
    return asyncio.run(_execute_assessment())


def test_verify_derives_exact_subject_dependencies_and_retained_environment(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    assert verified.subject == result.final_outputs[0].candidate
    assert verified.reference.artifact == fact.evidence_artifact
    input_fact = next(item for item in result.ports if item.port == "input")
    assert verified.consumed_by_port == (("input", input_fact.artifact),)
    assert verified.reference.consumed == frozenset({input_fact.artifact})
    assert verified.environment is fact.environment
    assert verified.finding is fact.finding
    assert verified.coverage == verified.promise.coverage
    assert verified.subject == CandidateRef(artifact=fact.evidence_artifact, target=input_fact.target)
    assert verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),)) == (
        verified,
    )


def test_copied_fields_do_not_authenticate_an_assessment(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    original = result.assessments[0]
    copied = object.__new__(ExecutionAssessmentFact)
    for item in fields(original):
        object.__setattr__(copied, item.name, getattr(original, item.name))
    assert copied == original and copied is not original
    with pytest.raises(EffectRejected) as error:
        verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=copied),))
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_submission_duplicates_and_cross_invocation_facts_reject(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    submission = AssessmentSubmission(fact=result.assessments[0])
    with pytest.raises(EffectRejected) as error:
        verify_evidence(admitted=admitted, result=result, submissions=(submission, submission))
    assert error.value.code is EffectCode.DUPLICATE
    _, foreign = asyncio.run(_execute_assessment())
    with pytest.raises(EffectRejected) as error:
        verify_evidence(
            admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=foreign.assessments[0]),)
        )
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_submission_limit_precedes_member_validation(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_submissions=0),
    )
    with pytest.raises(EffectRejected) as error:
        verify_evidence(
            admitted=admitted, result=result, submissions=cast(tuple[AssessmentSubmission, ...], (object(),))
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_factories_reject_direct_construction_and_boolean_limits() -> None:
    with pytest.raises(TypeError):
        VerifiedEvidence(_key=object())
    with pytest.raises(EffectRejected) as error:
        replace(_qualification_limits(), max_submissions=True)
    assert error.value.code is EffectCode.INVALID_TYPE


def test_explicit_current_view_and_stale_precedes_missing(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    artifacts = tuple(reference for reference, _ in result.artifacts)
    configurations = ((fact.node, fact.environment.configuration),)
    state = StateRevisionView(revisions=frozenset())
    current = evidence_revision_view(
        admitted=admitted, result=result, artifacts=artifacts, absences=(), configurations=configurations, state=state
    )
    assert evidence_validity(evidence=verified, current=current) == "current"
    missing = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(item for item in artifacts if item != fact.evidence_artifact),
        absences=(),
        configurations=configurations,
        state=state,
    )
    assert evidence_validity(evidence=verified, current=missing) == "unknown"
    changed = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=(),
        absences=(),
        configurations=((fact.node, FrozenConfig(fields=())),),
        state=state,
    )
    assert evidence_validity(evidence=verified, current=changed) == "stale"
    assert verified.environment is fact.environment
    assert evidence_validity(evidence=verified, current=current) == "current"


def test_revision_duplicates_and_foreign_result_binding_reject(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    reference = fact.evidence_artifact
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=(reference, reference),
            absences=(),
            configurations=(),
            state=StateRevisionView(revisions=frozenset()),
        )
    assert error.value.code is EffectCode.DUPLICATE
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    other_execution, other_result = asyncio.run(_execute_assessment())
    other_admitted = admit_qualification(
        execution=other_execution, productions=other_execution.assessment_productions, limits=_qualification_limits()
    )
    other_view = evidence_revision_view(
        admitted=other_admitted,
        result=other_result,
        artifacts=(),
        absences=(),
        configurations=(),
        state=StateRevisionView(revisions=frozenset()),
    )
    with pytest.raises(EffectRejected) as error:
        evidence_validity(evidence=verified, current=other_view)
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_revision_aggregate_limit_is_checked_before_members(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_revision_entries=0),
    )
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=cast(tuple, (object(),)),
            absences=(),
            configurations=(),
            state=StateRevisionView(revisions=frozenset()),
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED
