# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remaining request and decision discriminators from the frozen P7 corpus."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.context import (
    ContextSelector,
    ContextSourceRef,
    RetrievalBounds,
    SourceItem,
    SourceResponse,
)
from anonymizer.engine.graph_sdk.evidence import evidence_revision_view, evidence_validity
from anonymizer.engine.graph_sdk.executor import AssessmentFinding, BoundInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.records import DecisionRef
from anonymizer.engine.graph_sdk.requests import ExactUsage, ExternalSettlement, PhysicalRequestId, RequestAssociation
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource, _inputs, _RecoveringTransport
from tests.graph_sdk.test_adaptive_executor import _UnusedTransport
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_retry_reconciles_associations_independent_of_result_order(reverse: bool) -> None:
    transport = _RecoveringTransport(failure="retryable", reverse_results=reverse)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(
        _execute_assessment(external=(transport, lease), external_shared=True, target_count=2)
    )
    first, second = transport.envelopes
    first_order = tuple(item.association for item in first.associations)
    second_order = tuple(item.association for item in second.associations)
    assert len(set(first_order)) == 2 and first_order == second_order
    initial, retry = result.requests.dispatches
    assert initial.associations == retry.associations == frozenset(first_order)
    assert retry.associations == frozenset(reversed(first_order))  # Typed receipts erase list order, not membership.
    assert (initial.purpose, retry.purpose) == ("initial", "retry")
    terminal_order = tuple(item.association for item in result.requests.terminals[1].results)
    assert terminal_order == (tuple(reversed(second_order)) if reverse else second_order)
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == len(submissions) == 2
    assert all(item.qualification == "met" for item in output.record.statuses)
    assert all(not item.withholding for item in output.targets)
    assert len(result.requests.settlements) == 2
    assert transport.calls == 2 and transport.closes == 1
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    assert transport.calls == 2


def test_receipt_with_an_association_from_another_real_invocation_loses_attribution() -> None:
    results = []
    transports = []
    for _ in range(2):
        transport = _RecoveringTransport()
        lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
        results.append(asyncio.run(_execute_assessment(external=(transport, lease), target_count=2)))
        transports.append(transport)
    execution, result = results[0]
    _, foreign_result = results[1]
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    dispatch = result.requests.dispatches[0]
    foreign = foreign_result.requests.dispatches[0].associations
    assert not foreign & frozenset(
        association for item in result.requests.dispatches for association in item.associations
    )
    original = dispatch.associations
    object.__setattr__(dispatch, "associations", foreign)
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        assert all("inconsistent_attribution" in item.withholding for item in output.targets)
        assert all(item.qualification == "unknown" for item in output.record.statuses)
    finally:
        object.__setattr__(dispatch, "associations", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
    assert all(transport.calls == 2 and transport.closes == 1 for transport in transports)


@pytest.mark.parametrize("selection", ["decision", "b_only", "a_b", "a_only"])
@pytest.mark.parametrize("revision", [2, 1, None], ids=["current", "stale", "unknown"])
def test_actual_consumed_decision_revision_is_selective(selection: str, revision: int | None) -> None:
    consumed = (
        frozenset({"a", "b"}) if selection == "a_b" else frozenset({"a"}) if selection == "a_only" else frozenset({"b"})
    )
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=consumed,
            decision_context=True,
            expose_evidence=False,
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    versions = {
        fact.artifact.version: fact.artifact
        for fact in result.provenance
        if isinstance(fact.key, BoundInputKey) and fact.key.port == "b"
    }
    assert set(versions) == {1, 2} and versions[1].key == versions[2].key
    changed = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref in current.artifacts if ref.key != versions[2].key)
        + (() if revision is None else (versions[revision],)),
        configurations=current.configurations,
    )
    output = qualify(admitted=admitted, result=result, current=changed, submissions=submissions)
    (verified,) = output.verified
    consumed_by_port = dict(verified.consumed_by_port)
    assert set(consumed_by_port) == consumed
    expected = "current" if selection == "a_only" or revision == 2 else "unknown" if revision is None else "stale"
    assert evidence_validity(evidence=verified, current=changed) == expected
    if "b" in consumed:
        assert consumed_by_port["b"] == DecisionRef(artifact=versions[2])
    if "a" in consumed:
        assert not isinstance(consumed_by_port["a"], DecisionRef)
    assert bool(output.qualified) == (expected == "current")
    assert (
        output.record.statuses[0].qualification == {"current": "met", "stale": "unmet", "unknown": "unknown"}[expected]
    )
    assert output.required_decisions == (
        frozenset({DecisionRef(artifact=versions[2])}) if expected == "current" and "b" in consumed else frozenset()
    )
    restored = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(restored.qualified) == 1
    assert qualify(admitted=admitted, result=result, current=changed, submissions=submissions) == output
    assert len(result.record.terminals) == 2
    assert execution.context.bound_context is not None
    assert execution.context.bound_context.receipt.requests.dispatched_count == 2
    assert not result.requests.dispatches


@pytest.mark.parametrize("released_index", [0, 1])
def test_withheld_target_and_unrelated_executed_decisions_contribute_nothing(released_index: int) -> None:
    findings = tuple(
        AssessmentFinding(status="satisfied" if index == released_index else "unsatisfied", code="observed")
        for index in range(2)
    )
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a"}),
            decision_context=True,
            expose_evidence=False,
            target_labels=("A", "B"),
            target_findings=findings,
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True, port_facts=32, provenance_edges=32)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1 and not output.required_decisions
    assert len([fact for fact in result.provenance if fact.decision]) == 2
    labels = {datum.id: datum.text for datum in execution.context.prepared.data.datums}
    (released,) = output.qualified
    assert labels[released.target] == ("A", "B")[released_index]
    other = next(item for item in output.targets if item.target != released.target)
    assert other.withholding == frozenset({"assessment_unsatisfied"})
    assert len(submissions) == 2 and len(result.record.terminals) == 4
    assert all(item.category == "success" for item in result.record.terminals)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output


class _IdentityRetrievalSource(_BindingProvider):
    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse:
        self.calls += 1
        assert bounds.max_items == 1
        # A real source may return the same bytes as its selector. The operation
        # explicitly declares identity; publication must enforce that declaration.
        return SourceResponse(
            source=self.source,
            items=(
                SourceItem(
                    association=association,
                    key=0,
                    version=1,
                    text=selector.fields[0].value,
                ),
            ),
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                usage=ExactUsage(input_units=1, output_units=1),
                remote_stopped=True,
            ),
        )


def test_physical_adaptive_retrieval_is_verified_and_qualified() -> None:
    provider = _IdentityRetrievalSource(source=ContextSourceRef(name="adaptive-qualified", revision=1))
    resource = _initial_resource(provider)
    resource = replace(
        resource,
        capability=replace(
            resource.capability,
            uses=frozenset({"adaptive_retrieval"}),
            request=replace(resource.capability.request, max_attempts=2),
        ),
    )
    transport = _UnusedTransport()
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease), adaptive_resource=resource))
    (dispatch,) = result.requests.dispatches
    assert dispatch.purpose == "adaptive_retrieval"
    assert result.requests.terminals[0].category == "success"
    assert len(result.requests.settlements) == 1
    assert len(result.assessments) == len(result.final_outputs) == 1
    assert len(result.record.terminals) == 2
    assert all(item.category == "success" for item in result.record.terminals)
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert not output.targets[0].withholding and output.record.statuses[0].qualification == "met"
    assert provider.calls == provider.closes == 1
    assert not result.requests.local_in_flight and not result.requests.remote_outstanding
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    assert provider.calls == provider.closes == 1


def test_exact_decision_version_still_requires_current_role_membership() -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"b"}), decision_context=True, expose_evidence=False
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == len(current.decisions) == 1
    original = current.decisions
    object.__setattr__(current, "decisions", frozenset())
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert evidence_validity(evidence=output.verified[0], current=current) == "unknown"
        assert not output.qualified and not output.required_decisions
        assert output.record.statuses[0].qualification == "unknown"
    finally:
        object.__setattr__(current, "decisions", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline


def test_known_decision_change_outranks_missing_ordinary_context() -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a", "b"}), decision_context=True, expose_evidence=False
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    roots = {
        (fact.key.port, fact.artifact.version): fact.artifact
        for fact in result.provenance
        if isinstance(fact.key, BoundInputKey)
    }
    removed_keys = {roots["a", 2].key, roots["b", 2].key}
    changed = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref in current.artifacts if ref.key not in removed_keys) + (roots["b", 1],),
        configurations=current.configurations,
    )
    assert not changed.decisions  # The older retained version was never a decision occurrence.
    output = qualify(admitted=admitted, result=result, current=changed, submissions=submissions)
    assert evidence_validity(evidence=output.verified[0], current=changed) == "stale"
    assert not output.qualified and output.record.statuses[0].qualification == "unmet"
    assert not output.required_decisions


def test_another_targets_decision_lineage_cannot_supply_missing_current_key() -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"b"}),
            decision_context=True,
            expose_evidence=False,
            target_labels=("A", "B"),
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True, port_facts=32, provenance_edges=32)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    first, second = baseline.qualified
    (removed,) = first.required_decisions
    (retained,) = second.required_decisions
    assert removed.artifact.key != retained.artifact.key
    changed = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref in current.artifacts if ref != removed.artifact),
        configurations=current.configurations,
    )
    assert changed.decisions == frozenset({retained})
    output = qualify(admitted=admitted, result=result, current=changed, submissions=submissions)
    assert output.qualified == (second,) and output.required_decisions == frozenset({retained})
    assert next(item for item in output.record.statuses if item.target == first.target).qualification == "unknown"
