# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification of actual provider-free execution results."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.data import AtomicGroup, CoherenceScope, DataGraph, DataLimits, DatumDependency
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission, admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, AssessmentFinding, ExecutionResult
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.test_evidence import _execute_assessment, _qualification_limits


def _inputs(execution: AdmittedExecutionPlan, result: ExecutionResult, *, decisions: int = 16):
    admitted = admit_qualification(
        execution=execution,
        productions=()
        if execution.context.prepared.configuration.purpose == "execution_only"
        else execution.assessment_productions,
        limits=_qualification_limits(max_required_decisions=decisions),
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        absences=tuple({absence for fact in result.assessments for absence in fact.environment.absences}),
        configurations=tuple({(fact.node, fact.environment.configuration) for fact in result.assessments}),
        state=execution.context.prepared.state,
    )
    submissions = tuple(AssessmentSubmission(fact=fact) for fact in result.assessments)
    return admitted, current, submissions


@pytest.mark.parametrize("environment", [False, True])
def test_successful_qualification_preserves_execution_record(environment: bool) -> None:
    execution, result = asyncio.run(_execute_assessment(environment=environment))
    admitted, current, submissions = _inputs(execution, result)
    qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(qualified.qualified) == 1
    assert qualified.qualified[0].candidate == result.final_outputs[0].candidate
    assert qualified.qualified[0].evidence == frozenset({qualified.verified[0].reference})
    assert qualified.record.memberships is result.record.memberships
    assert qualified.record.terminals is result.record.terminals
    assert qualified.record.artifacts is result.record.artifacts
    assert qualified.record.invocation is result.record.invocation
    assert qualified.record.statuses[0].qualification == "met"
    assert qualified.record.statuses[0].protection_available
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == qualified


@pytest.mark.parametrize(
    ("status", "withholding", "qualification"),
    [("unsatisfied", "assessment_unsatisfied", "unmet"), ("unknown", "assessment_unknown", "unknown")],
)
def test_actual_assessment_findings_withhold(status, withholding, qualification) -> None:
    execution, result = asyncio.run(_execute_assessment(finding=AssessmentFinding(status=status, code="observed")))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert output.targets[0].withholding == frozenset({withholding})
    assert output.record.statuses[0].qualification == qualification


def test_missing_assessment_is_withholding_not_invented_evidence() -> None:
    execution, result = asyncio.run(_execute_assessment())
    admitted, current, _ = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=())
    assert not output.qualified and not output.verified and not output.record.evidence
    assert output.targets[0].withholding == frozenset({"missing_assessment"})


def test_execution_only_never_consumes_submitted_assessments() -> None:
    execution, result = asyncio.run(_execute_assessment(execution_only=True))
    admitted, current, _ = _inputs(execution, result)
    output = qualify(
        admitted=admitted,
        result=result,
        current=current,
        submissions=cast(tuple[AssessmentSubmission, ...], (object(),)),
    )
    assert not output.qualified and not output.verified
    assert output.record.statuses[0].qualification == "not_assessed"
    assert not output.record.statuses[0].protection_available
    assert output.record.memberships is result.record.memberships


def test_required_decision_is_the_executed_transitive_producer_and_bound_is_exact() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True))
    admitted, current, submissions = _inputs(execution, result, decisions=1)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.required_decisions) == 1
    assert output.qualified[0].required_decisions == output.required_decisions
    assert next(iter(output.required_decisions)).artifact == next(
        item.artifact for item in result.provenance if item.decision
    )
    admitted, current, submissions = _inputs(execution, result, decisions=0)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


@pytest.mark.parametrize("defect", ["root", "terminal"])
def test_missing_canonical_accounting_withholds_actual_success(defect: str) -> None:
    execution, result = asyncio.run(_execute_assessment())
    original = result.record
    corrupted = replace(
        original,
        memberships=() if defect == "root" else original.memberships,
        terminals=(),
    )
    object.__setattr__(result, "record", corrupted)
    try:
        admitted, current, _ = _inputs(execution, result)
        output = qualify(admitted=admitted, result=result, current=current, submissions=())
        assert not output.qualified
        assert "incomplete_membership" in output.targets[0].withholding
        assert output.record.statuses[0].completion == "pending"
    finally:
        object.__setattr__(result, "record", original)


@pytest.mark.parametrize("relation", ["none", "dependency", "atomic", "coherence"])
def test_withholding_propagates_only_along_dependencies_and_atomic_groups(relation: str) -> None:
    graph = DataGraph.new()
    graph, a = graph.add_text("A")
    graph, b = graph.add_text("B")
    graph, c = graph.add_text("C")
    data = graph.validate(
        targets=(a, b, c),
        source_relations=(),
        contexts=(),
        dependencies=(DatumDependency(prerequisite=a, dependent=b),) if relation == "dependency" else (),
        atomic=(AtomicGroup(members=(a, b)),) if relation == "atomic" else (),
        coherence=(CoherenceScope(members=(a, b)),) if relation == "coherence" else (),
        output_regions=(),
        limits=DataLimits(max_datums=3, max_targets=3, max_text_bytes=10, max_declarations=1, max_group_members=2),
    )
    execution, result = asyncio.run(_execute_assessment(data=data))
    admitted, current, submissions = _inputs(execution, result)
    omitted = next(item.activation for item in result.ports if item.target == a)
    submissions = tuple(item for item in submissions if item.fact.activation != omitted)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    released = {item.target for item in output.qualified}
    assert released == ({c} if relation in {"dependency", "atomic"} else {b, c})
    if relation in {"dependency", "atomic"}:
        withheld_b = next(item for item in output.targets if item.target == b)
        assert ("dependency" if relation == "dependency" else "atomic_group") in withheld_b.withholding


@pytest.mark.parametrize(
    ("nested", "rename_ports", "decision"),
    [(False, True, False), (True, False, False), (True, True, False), (True, True, True)],
)
def test_requirement_projection_and_provenance_through_real_nested_execution(
    nested: bool, rename_ports: bool, decision: bool
) -> None:
    execution, result = asyncio.run(
        _execute_assessment(nested=nested, rename_ports=rename_ports, decision_input=decision)
    )
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.record.statuses[0].qualification == "met"
    assert any(item.structural for item in output.record.terminals) == nested
    assert len(output.required_decisions) == int(decision)
    assert output.verified[0].promise.subject_port == "context"
    assert result.final_outputs[0].port == ("protected" if rename_ports else "context")


def test_distinct_candidate_and_decision_occurrences_preserve_identity_alias() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True, alias_output=True))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    candidate = output.qualified[0].candidate
    assert next(iter(output.required_decisions)).artifact == candidate.artifact
    assert {item.role for item in result.ports if item.artifact == candidate.artifact} >= {"decision", "candidate"}
    assert output.verified[0].subject == candidate


def test_caller_owned_resource_stays_open_without_withholding_release() -> None:
    from anonymizer.engine.graph_sdk.resources import ResourceLease

    class FailedClose:
        calls = 0

        async def close(self) -> None:
            self.calls += 1
            raise RuntimeError("local close failure")

    owner = "caller"
    handle = FailedClose()
    lease = ResourceLease.create(owner=owner, safe_detachment="forbidden", handle=handle)
    execution, result = asyncio.run(_execute_assessment(resource=lease))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(result.cleanup) == 1
    assert bool(output.qualified) == (owner == "caller")
    assert handle.calls == int(owner == "sdk")
    assert result.cleanup[0].disposition == ("left_open" if owner == "caller" else "close_failed")


def test_zero_capacity_member_configuration_remains_a_valid_current_key() -> None:
    from anonymizer.engine.graph_sdk.preparation import StateRevisionView
    from tests.graph_sdk.test_effects_map_production_conformance import _execute_membership

    _, result, _ = asyncio.run(_execute_membership(0, max_children=0, outward_scalar="join"))
    execution = result._execution
    prepared = execution.context.prepared
    admitted = admit_qualification(execution=execution, productions=(), limits=_qualification_limits())
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=(),
        absences=(),
        configurations=tuple((item.node, item.capability.configuration) for item in prepared.implementations),
        state=StateRevisionView(revisions=frozenset()),
    )
    assert len(current.configurations) == len(prepared.implementations)
    output = qualify(admitted=admitted, result=result, current=current, submissions=())
    assert not output.qualified
    assert all(item.qualification == "not_assessed" for item in output.record.statuses)
    assert any(item.parent is not None and not item.members and item.closed for item in output.record.memberships)


def test_protection_submission_member_type_precedes_foreign_current_view() -> None:
    execution, result = asyncio.run(_execute_assessment())
    admitted, _, _ = _inputs(execution, result)
    other_execution, other_result = asyncio.run(_execute_assessment())
    _, foreign, _ = _inputs(other_execution, other_result)
    with pytest.raises(EffectRejected) as error:
        qualify(
            admitted=admitted,
            result=result,
            current=foreign,
            submissions=cast(tuple[AssessmentSubmission, ...], (object(),)),
        )
    assert error.value.code is EffectCode.INVALID_TYPE


def test_provenance_cannot_add_a_causal_cycle_to_a_real_output() -> None:
    execution, result = asyncio.run(_execute_assessment())
    admitted, current, submissions = _inputs(execution, result)
    producer = next(item for item in result.provenance if item.key == result.final_outputs[0].producer)
    original = producer.parents
    object.__setattr__(producer, "parents", frozenset({producer.key}))
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.CONTRADICTORY
    finally:
        object.__setattr__(producer, "parents", original)


def test_required_decision_bound_applies_to_union_across_released_targets() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True, target_count=2))
    admitted, current, submissions = _inputs(execution, result, decisions=2)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 2
    assert all(len(item.required_decisions) == 1 for item in output.qualified)
    assert len(output.required_decisions) == 2
    admitted, current, submissions = _inputs(execution, result, decisions=1)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.LIMIT_EXCEEDED
