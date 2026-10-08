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
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    VerifiedEvidence,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    ExecutionAssessmentFact,
    ExecutionResult,
    admit_execution_plan,
)
from anonymizer.engine.graph_sdk.preparation import (
    StateRevisionView,
)
from anonymizer.engine.graph_sdk.records import AbsenceRef, CandidateRef, DecisionRef
from anonymizer.graph._values import ArtifactRef, InvocationId
from anonymizer.graph.workflow import (
    CoverageAtom,
)
from tests.graph_sdk.evidence_fixtures import _execute_assessment as _execute_assessment
from tests.graph_sdk.evidence_fixtures import _qualification_limits as _qualification_limits


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


@pytest.mark.parametrize("factory", ["verify", "view"])
def test_same_prepared_plan_does_not_authenticate_a_different_execution(
    factory: str, assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult]
) -> None:
    execution, result = assessment_execution
    other = admit_execution_plan(
        context=execution.context,
        capabilities=execution.capabilities,
        policies=tuple(execution.policies),
        decisions=tuple(execution.decisions),
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
        map_expansions=execution.map_expansions,
    )
    assert other is not execution and other.context.prepared is execution.context.prepared
    admitted = admit_qualification(
        execution=other, productions=other.assessment_productions, limits=_qualification_limits()
    )
    with pytest.raises(EffectRejected) as error:
        if factory == "verify":
            verify_evidence(
                admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=result.assessments[0]),)
            )
        else:
            evidence_revision_view(
                admitted=admitted,
                result=result,
                artifacts=(),
                absences=(),
                configurations=(),
                state=StateRevisionView(revisions=frozenset()),
            )
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_absence_capture_does_not_consume_a_port_dependency_slot() -> None:
    execution, result = asyncio.run(_execute_assessment(environment=True))
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_consumed_per_assessment=1),
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    assert len(verified.consumed_by_port) == 1
    assert {(item.query, item.scope_revision) for item in verified.environment.absences} == {(7, 4)}
    assert {(item.effect.name, item.revision) for item in verified.environment.state.revisions} == {
        ("assessment-policy", 3)
    }
    with pytest.raises(EffectRejected) as error:
        admit_qualification(
            execution=execution,
            productions=execution.assessment_productions,
            limits=_qualification_limits(max_consumed_per_assessment=0),
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


@pytest.mark.parametrize(
    ("dependency", "expected"),
    [
        ("none", "current"),
        ("absence-missing", "unknown"),
        ("absence-changed", "stale"),
        ("state-missing", "unknown"),
        ("state-changed", "stale"),
        ("input-missing", "unknown"),
        ("candidate-missing", "unknown"),
        ("changed-and-missing", "stale"),
    ],
)
def test_validity_selectively_tracks_real_environment(dependency: str, expected: str) -> None:
    execution, result = asyncio.run(_execute_assessment(environment=True))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    artifacts = tuple(reference for reference, _ in result.artifacts)
    absences = tuple(fact.environment.absences)
    state = fact.environment.state
    if dependency == "absence-missing":
        absences = ()
    if dependency in {"absence-changed", "changed-and-missing"}:
        absences = tuple(replace(item, scope_revision=item.scope_revision + 1) for item in absences)
    if dependency in {"state-missing", "changed-and-missing"}:
        state = StateRevisionView(revisions=frozenset())
    if dependency == "state-changed":
        state = StateRevisionView(
            revisions=frozenset(replace(item, revision=item.revision + 1) for item in state.revisions)
        )
    if dependency == "input-missing":
        consumed = next(item.artifact for item in result.ports if item.port == "input")
        artifacts = tuple(item for item in artifacts if item != consumed)
    if dependency == "candidate-missing":
        artifacts = tuple(item for item in artifacts if item != verified.subject.artifact)
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=artifacts,
        absences=absences,
        configurations=((fact.node, fact.environment.configuration),),
        state=state,
    )
    assert evidence_validity(evidence=verified, current=current) == expected


def test_missing_current_artifact_precedes_unsupported_absence(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    missing = ArtifactRef(invocation=result.record.invocation, key=999, version=1)
    unsupported = AbsenceRef(invocation=result.record.invocation, query=999, scope_revision=1)
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=(missing,),
            absences=(unsupported,),
            configurations=(),
            state=StateRevisionView(revisions=frozenset()),
        )
    assert error.value.code is EffectCode.MISSING


@pytest.mark.parametrize(
    ("defect", "code"),
    [("foreign", EffectCode.FOREIGN_OWNER), ("duplicate", EffectCode.DUPLICATE), ("missing", EffectCode.MISSING)],
)
def test_environment_identity_errors_precede_configuration_contradiction(defect: str, code: EffectCode) -> None:
    execution, result = asyncio.run(_execute_assessment(environment=True))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    original = fact.environment
    absence = next(iter(original.absences))
    absences = frozenset()
    if defect == "foreign":
        absences = frozenset({replace(absence, invocation=InvocationId.new(plan=result.record.plan))})
    if defect == "duplicate":
        absences = frozenset({absence, replace(absence, scope_revision=absence.scope_revision + 1)})
    # Corrupt the retained fact solely to exercise validation of simultaneous defects.
    object.__setattr__(fact, "environment", replace(original, configuration=FrozenConfig(fields=()), absences=absences))
    try:
        with pytest.raises(EffectRejected) as error:
            verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
        assert error.value.code is code
    finally:
        object.__setattr__(fact, "environment", original)


@pytest.mark.parametrize("bound", ["max_port_facts", "max_provenance_edges"])
def test_result_bounds_are_exact_and_count_actual_edges(
    bound: str, assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult]
) -> None:
    execution, result = assessment_execution
    count = len(result.ports) if bound == "max_port_facts" else sum(len(item.parents) for item in result.provenance)
    assert count > 0
    for limit in (count, count - 1):
        admitted = admit_qualification(
            execution=execution,
            productions=execution.assessment_productions,
            limits=_qualification_limits(**{bound: limit}),
        )
        if limit == count:
            assert verify_evidence(
                admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=result.assessments[0]),)
            )
        else:
            with pytest.raises(EffectRejected) as error:
                verify_evidence(
                    admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=result.assessments[0]),)
                )
            assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_consumed_candidate_retains_target_and_alias_identity() -> None:
    execution, result = asyncio.run(_execute_assessment(candidate_input=True))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    assert verified.consumed_by_port == (("input", verified.subject),)
    assert verified.subject == result.final_outputs[0].candidate
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        absences=(),
        configurations=((fact.node, fact.environment.configuration),),
        state=fact.environment.state,
    )
    assert evidence_validity(evidence=verified, current=current) == "current"


def test_coverage_capacity_accepts_exact_and_rejects_one_over() -> None:
    coverage = frozenset({CoverageAtom(kind="field", name="person")})
    execution, result = asyncio.run(_execute_assessment(coverage=coverage))
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_coverage_atoms=1),
    )
    (verified,) = verify_evidence(
        admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=result.assessments[0]),)
    )
    assert verified.coverage == coverage
    with pytest.raises(EffectRejected) as error:
        admit_qualification(
            execution=execution,
            productions=execution.assessment_productions,
            limits=_qualification_limits(max_coverage_atoms=0),
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_fixed_point_outer_bound_precedes_missing_production() -> None:
    execution, _ = asyncio.run(_execute_assessment(target_count=2))
    with pytest.raises(EffectRejected) as error:
        admit_qualification(execution=execution, productions=(), limits=_qualification_limits(max_fixed_point_steps=1))
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_consumed_decision_is_derived_from_an_actual_resumed_producer() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    ((port, decision),) = verified.consumed_by_port
    assert port == "input" and isinstance(decision, DecisionRef)
    producers = [item for item in result.provenance if item.decision]
    assert len(producers) == 1 and producers[0].artifact == decision.artifact
    assert len(result.record.terminals) == 2
    assert all(item.category == "success" for item in result.record.terminals)
    artifacts = tuple(reference for reference, _ in result.artifacts)
    for missing in (False, True):
        selected = tuple(item for item in artifacts if not missing or item != decision.artifact)
        current = evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=selected,
            absences=(),
            configurations=((fact.node, fact.environment.configuration),),
            state=fact.environment.state,
        )
        assert evidence_validity(evidence=verified, current=current) == ("unknown" if missing else "current")


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("role", ["artifact", "decision"])
def test_evidence_occurrence_authenticates_executor_role_precedence(nested: bool, role: str) -> None:
    execution, result = asyncio.run(_execute_assessment(candidate_input=True, nested=nested))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    evidence = next(item for item in result.ports if item.activation == fact.activation and item.port == "assessment")
    assert evidence.role == ("evidence" if nested else "candidate")
    submissions = (AssessmentSubmission(fact=fact),)
    assert len(verify_evidence(admitted=admitted, result=result, submissions=submissions)) == 1
    original = evidence.role
    object.__setattr__(evidence, "role", role)
    try:
        with pytest.raises(EffectRejected) as error:
            verify_evidence(admitted=admitted, result=result, submissions=submissions)
        assert error.value.code is EffectCode.CONTRADICTORY
    finally:
        object.__setattr__(evidence, "role", original)


@pytest.mark.parametrize(
    ("bound", "boundary"),
    [
        ("max_productions", "admit"),
        ("max_submissions", "verify"),
        ("max_verified_evidence", "verify"),
        ("max_revision_entries", "view"),
        ("max_absence_revisions", "view"),
    ],
)
def test_actual_evidence_collections_accept_exact_limits_and_reject_one_over(bound: str, boundary: str) -> None:
    execution, result = asyncio.run(_execute_assessment(environment=True))
    fact = result.assessments[0]
    artifacts = tuple(reference for reference, _ in result.artifacts)
    configurations = ((fact.node, fact.environment.configuration),)
    state = fact.environment.state
    count = len(artifacts) + len(configurations) + len(state.revisions) if bound == "max_revision_entries" else 1

    def invoke(limit: int) -> object:
        admitted = admit_qualification(
            execution=execution,
            productions=execution.assessment_productions,
            limits=_qualification_limits(**{bound: limit}),
        )
        if boundary == "admit":
            return admitted
        if boundary == "verify":
            return verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
        return evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=artifacts,
            absences=tuple(fact.environment.absences),
            configurations=configurations,
            state=state,
        )

    assert invoke(count) is not None
    with pytest.raises(EffectRejected) as error:
        invoke(count - 1)
    assert error.value.code is EffectCode.LIMIT_EXCEEDED
