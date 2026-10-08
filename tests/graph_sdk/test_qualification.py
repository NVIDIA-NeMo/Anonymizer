# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification of actual provider-free execution results."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.data import AtomicGroup, CoherenceScope, DataGraph, DataLimits, DatumDependency
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission, admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, AssessmentFinding, ExecutionResult
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    FailureClass,
    PortArtifact,
    StopConfirmed,
    TransportFailure,
    TransportResult,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
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


@dataclass
class _RecoveringTransport:
    failure: FailureClass | None = None
    unknown_usage: bool = False
    fail_close: bool = False
    calls: int = 0
    closes: int = 0

    async def dispatch(self, request: DispatchEnvelope) -> TransportResult:
        self.calls += 1
        settlement = ExternalSettlement(
            request=request.request,
            disposition="completed",
            remote_stopped=True,
            usage=UnknownUsage()
            if self.unknown_usage and self.calls == 1
            else ExactUsage(input_units=1, output_units=1),
        )
        if self.calls == 1 and self.failure is not None:
            return TransportFailure(failure=self.failure, settlement=settlement)
        item = request.associations[0]
        return TransportSuccess(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        PortArtifact(
                            port="context",
                            artifact_type=item.inputs[0].artifact_type,
                            artifact=None,
                            value=item.inputs[0].value,
                        ),
                    ),
                ),
            ),
            settlement=settlement,
        )

    async def cancel(self, request: object) -> StopConfirmed:
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))

    async def close(self) -> None:
        self.closes += 1
        if self.fail_close:
            raise RuntimeError("test close failure")


@pytest.mark.parametrize(
    ("failure", "purpose"), [(None, None), ("retryable", "retry"), ("malformed_response", "correction")]
)
def test_settled_external_recovery_can_qualify_after_local_verification(failure, purpose) -> None:
    transport = _RecoveringTransport(failure=failure)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.record.statuses[0].qualification == "met"
    assert [item.purpose for item in result.requests.dispatches] == (
        ["initial"] if purpose is None else ["initial", purpose]
    )
    assert transport.calls == (1 if purpose is None else 2)
    assert transport.closes == 1
    assert result.cleanup[0].disposition == "closed"
    assert not output.required_decisions


@pytest.mark.parametrize("defect", ["unknown-usage", "cleanup"])
def test_recovery_does_not_erase_accounting_or_cleanup_defects(defect: str) -> None:
    transport = _RecoveringTransport(
        failure="retryable", unknown_usage=defect == "unknown-usage", fail_close=defect == "cleanup"
    )
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert transport.calls == 2 and transport.closes == 1
    assert result.assessments[0].finding.status == "satisfied"
    assert not output.qualified
    assert output.record.statuses[0].qualification == ("unknown" if defect == "unknown-usage" else "unmet")
    assert output.targets[0].withholding == frozenset(
        {"request_accounting" if defect == "unknown-usage" else "cleanup_accounting"}
    )


@pytest.mark.parametrize("input_subject", [False, True])
def test_auxiliary_final_output_does_not_replace_the_assessed_candidate(input_subject: bool) -> None:
    execution, result = asyncio.run(_execute_assessment(auxiliary_output=True, candidate_input=input_subject))
    assert len(result.final_outputs) == 2
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    subject = next(item.candidate for item in result.final_outputs if item.port != "metadata")
    assert output.qualified[0].candidate == subject
    assert current.candidates == frozenset({subject})
    assert output.record.statuses[0].qualification == "met"


def test_transport_only_cleanup_can_have_no_affected_targets() -> None:
    transport = _RecoveringTransport(fail_close=True)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    association = result.cleanup_associations[0]
    original = association.targets, association.purpose
    # Exercise the D08 transport-only extension point; current P5 emits accounting here.
    object.__setattr__(association, "targets", frozenset())
    object.__setattr__(association, "purpose", "transport_only")
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert len(output.qualified) == 1
        assert output.record.statuses[0].qualification == "met"
    finally:
        object.__setattr__(association, "targets", original[0])
        object.__setattr__(association, "purpose", original[1])


@pytest.mark.parametrize("mutation", ["missing", "extra", "aliased-occurrence"])
def test_output_provenance_requires_exact_executed_input_occurrences(mutation: str) -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True))
    admitted, current, submissions = _inputs(execution, result)
    output = next(item for item in result.provenance if item.key == result.final_outputs[0].producer)
    decision = next(item for item in result.provenance if item.decision)
    root = next(item for item in result.provenance if not item.parents)
    assert root.artifact == decision.artifact and root.key != decision.key
    original = output.parents
    changed = (
        frozenset()
        if mutation == "missing"
        else original | {root.key}
        if mutation == "extra"
        else frozenset({root.key})
    )
    object.__setattr__(output, "parents", changed)
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.CONTRADICTORY
    finally:
        object.__setattr__(output, "parents", original)


def test_fully_settled_exhausted_failure_is_unmet_not_unknown() -> None:
    transport = _RecoveringTransport(failure="permanent")
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert transport.calls == 1 and transport.closes == 1
    assert not output.qualified
    assert output.record.statuses[0].qualification == "unmet"
    assert "request_accounting" in output.targets[0].withholding


@pytest.mark.parametrize(("count", "mode", "status"), [(2, "valid", "overflow"), (1, "missing", "failed")])
def test_failed_expansion_reconciles_actual_members(count: int, mode: str, status: str) -> None:
    from anonymizer.graph._values import ActivationKey
    from tests.graph_sdk.test_effects_map_production_conformance import _execute_membership

    _, result, _ = asyncio.run(_execute_membership(count, response_mode=mode, max_children=1, outward_scalar="join"))
    execution = result._execution
    admitted = admit_qualification(execution=execution, productions=(), limits=_qualification_limits())
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=(),
        absences=(),
        configurations=(),
        state=execution.context.prepared.state,
    )
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == status and not expansion.members
    output = qualify(admitted=admitted, result=result, current=current, submissions=())
    assert output.record.statuses[0].completion == "closed"
    original = expansion.members
    missing = ActivationKey(
        invocation=result.record.invocation, occurrence=100, parent=expansion.parent, iteration=None
    )
    object.__setattr__(expansion, "members", frozenset({missing}))
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=())
        assert output.record.statuses[0].completion == "pending"
        assert "incomplete_membership" in output.targets[0].withholding
    finally:
        object.__setattr__(expansion, "members", original)


@pytest.mark.parametrize("nested", [False, True])
def test_direct_root_input_output_retains_its_actual_producer_and_qualifies(nested: bool) -> None:
    from anonymizer.engine.graph_sdk.executor import OperationOutputKey, RootInputKey

    execution, result = asyncio.run(_execute_assessment(candidate_input=True, root_passthrough=True, nested=nested))
    assert len(result.final_outputs) == 1
    assert isinstance(result.final_outputs[0].producer, OperationOutputKey if nested else RootInputKey)
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.qualified[0].candidate == result.final_outputs[0].candidate
    assert not output.required_decisions
    assert output.record.statuses[0].qualification == "met"


@pytest.mark.parametrize("mutation", ["missing", "aliased-occurrence"])
def test_nested_passthrough_provenance_preserves_the_resolved_input_source(mutation: str) -> None:
    from anonymizer.engine.graph_sdk.executor import OperationOutputKey

    execution, result = asyncio.run(_execute_assessment(candidate_input=True, root_passthrough=True, nested=True))
    admitted, current, submissions = _inputs(execution, result)
    output = next(item for item in result.provenance if item.key == result.final_outputs[0].producer)
    assessment = result.assessments[0]
    alias = next(
        item
        for item in result.provenance
        if isinstance(item.key, OperationOutputKey) and item.key.activation == assessment.activation
    )
    assert alias.artifact == output.artifact
    assert alias.key not in output.parents
    original = output.parents
    object.__setattr__(output, "parents", frozenset() if mutation == "missing" else frozenset({alias.key}))
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.CONTRADICTORY
    finally:
        object.__setattr__(output, "parents", original)


@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("passthrough", [False, True])
def test_deep_nested_outputs_close_before_their_parent_outputs(depth: int, passthrough: bool) -> None:
    execution, result = asyncio.run(
        _execute_assessment(nested_depth=depth, candidate_input=passthrough, root_passthrough=passthrough)
    )
    assert len(result.final_outputs) == 1
    assert sum(item.structural for item in result.record.terminals) == depth
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.record.statuses[0].qualification == "met"
