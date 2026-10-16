# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assessment-bearing publication rollback with an already accepted peer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts
from anonymizer.engine.graph_sdk.executor import (
    ExecutionAssessmentFact,
    ExecutionResult,
    MapItemKey,
    OperationOutputKey,
)
from anonymizer.engine.graph_sdk.requests import SemanticAssociation, TextCollectionValue
from anonymizer.graph.workflow import NodeId
from tests.graph_sdk.effects_map_execution import _execute_membership
from tests.graph_sdk.effects_map_fixtures import _MapCallback, _MapFixture


@dataclass(frozen=True)
class _Discard:
    prior: tuple[ExecutionAssessmentFact, ...]
    rejected: ExecutionAssessmentFact


@pytest.mark.compat
def test_rejected_membership_leaves_exact_capacity_for_a_later_peer() -> None:
    fixture, result, callbacks = asyncio.run(
        _execute_membership(
            2,
            item_counts=(1, 2, 1),
            target_count=3,
            artifact_headroom=4,
            artifact_byte_headroom=128,
            max_local_in_flight=1,
        )
    )
    first, rejected, last = (call[0].association for call in callbacks[fixture.expander].calls)
    assert len(result.artifacts) == 7
    assert len(result.assessments) == len(result.final_outputs) == 2
    assert {fact.activation for fact in result.assessments} == {first.task.activation, last.task.activation}
    (failure,) = result.publication_failures
    assert failure.association == rejected and failure.condition == "artifact_limit_exhausted"
    assert all(fact.artifact in dict(result.artifacts) for fact in result.provenance)
    assert len(callbacks[fixture.member].calls) == len(callbacks[fixture.join].calls) == 2
    assert not result.requests.dispatches and not result.requests.remote_outstanding


def _execute(
    monkeypatch: pytest.MonkeyPatch, *, retain_rejected: bool = False
) -> tuple[_MapFixture, ExecutionResult, dict[NodeId, _MapCallback], _Discard]:
    deltas: list[_ExecutionFacts] = []
    create_delta = _ExecutionFacts.delta

    def observe(facts: _ExecutionFacts) -> _ExecutionFacts:
        delta = create_delta(facts)
        deltas.append(delta)
        return delta

    monkeypatch.setattr(_ExecutionFacts, "delta", observe)
    fixture, result, callbacks = asyncio.run(
        _execute_membership(
            2,
            item_counts=(1, 2),
            target_count=2,
            artifact_headroom=4,
            artifact_byte_headroom=128,
            max_local_in_flight=1,
        )
    )
    rejected_deltas = [delta for delta in deltas if delta._parent is not None and delta.assessments.maps[0]]
    (delta,) = rejected_deltas
    (rejected,) = delta.assessments.maps[0].values()
    assert delta._parent is not None
    prior = tuple(delta._parent.assessments.values())
    assert len(prior) == 1
    assert rejected.evidence_artifact in delta.values
    staged = delta.values[rejected.evidence_artifact]
    assert isinstance(staged, TextCollectionValue) and len(staged.items) == 2
    assert any(port.artifact == rejected.evidence_artifact for port in delta.ports.values())
    assert any(producer.artifact == rejected.evidence_artifact for producer in delta.provenance.values())
    if retain_rejected:
        # Corrupt the valid fixture by retaining the real rejected delta's fact.
        object.__setattr__(result, "assessments", (*result.assessments, rejected))
    return fixture, result, callbacks, _Discard(prior, rejected)


def _assert_rollback(
    fixture: _MapFixture,
    result: ExecutionResult,
    callbacks: dict[NodeId, _MapCallback],
    discard: _Discard,
) -> None:
    (survivor,) = discard.prior
    rejected = discard.rejected
    assert result.assessments == (survivor,), "rejected staged assessment survived rollback"
    assert result.assessments[0] is survivor
    calls = callbacks[fixture.expander].calls
    assert len(calls) == 2
    accepted_association, rejected_association = (call[0].association for call in calls)
    assert isinstance(accepted_association, SemanticAssociation)
    assert isinstance(rejected_association, SemanticAssociation)
    assert survivor.activation == accepted_association.task.activation
    assert rejected.activation == rejected_association.task.activation
    assert survivor.node == rejected.node == fixture.expander
    assert survivor.promise == rejected.promise == "assessment0"
    assert survivor.finding.status == rejected.finding.status == "satisfied"

    artifacts = dict(result.artifacts)
    assert rejected.evidence_artifact not in artifacts
    assert all(port.artifact != rejected.evidence_artifact for port in result.ports)
    assert all(producer.artifact != rejected.evidence_artifact for producer in result.provenance)
    assert all(port.activation != rejected.activation or port.port == "default" for port in result.ports)
    producer = next(fact for fact in result.provenance if fact.artifact == survivor.evidence_artifact)
    assert isinstance(producer.key, OperationOutputKey)
    assert producer.key.activation == survivor.activation and producer.key.port == "members"
    port = next(port for port in result.ports if port.artifact == survivor.evidence_artifact)
    assert port.activation == survivor.activation and port.target == producer.key.target and port.port == "members"
    accepted_value = artifacts[survivor.evidence_artifact]
    assert isinstance(accepted_value, TextCollectionValue) and len(accepted_value.items) == 1

    memberships = {entry.parent: entry for state in result.states for entry in state.expansions}
    assert memberships[survivor.activation].status == "closed"
    assert len(memberships[survivor.activation].members) == 1
    assert memberships[rejected.activation].status == "failed"
    assert not memberships[rejected.activation].members
    items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    assert len(items) == 1
    item = items[0]
    assert isinstance(item.key, MapItemKey)
    assert item.key.expander == survivor.activation and item.key.target == port.target
    assert item.key.member in memberships[survivor.activation].members
    assert item.parents == frozenset({producer.key})
    assert item.key.item_key == 0 and item.key.item_version == 1
    assert {ref for ref, _ in result.artifacts} == {fact.artifact for fact in result.provenance}
    assert len(result.artifacts) == 4  # Two initial inputs, accepted collection and accepted item.

    (final,) = result.final_outputs
    assert final.target == final.candidate.target == port.target
    assert final.candidate.artifact == survivor.evidence_artifact and final.producer == producer.key
    (failure,) = result.publication_failures
    assert failure.association == rejected_association and failure.target != final.target
    assert failure.condition == "artifact_limit_exhausted" and failure.failure is None
    terminals = {terminal.activation: terminal for terminal in result.record.terminals}
    assert terminals[survivor.activation].category == "success"
    assert terminals[survivor.activation].attempt == accepted_association.task
    assert terminals[rejected.activation].category == "blocked"
    assert terminals[rejected.activation].attempt == rejected_association.task
    assert terminals[rejected.activation].reasons == frozenset({"artifact_limit_exhausted"})
    assert sorted(terminal.category for terminal in result.record.terminals) == [
        "blocked",
        "blocked",
        "success",
        "success",
        "success",
    ]
    assert len(callbacks[fixture.member].calls) == len(callbacks[fixture.join].calls) == 1
    assert result.requests.dispatched_count == 0  # These are local callbacks, not physical requests.
    assert not result.requests.dispatches and not result.requests.terminals and not result.requests.settlements
    assert not result.requests.local_in_flight and not result.requests.remote_outstanding
    assert not result.requests.defects and not result.cleanup


def test_rejected_assessment_delta_preserves_accepted_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_rollback(*_execute(monkeypatch))


def test_witness_detects_suppressed_assessment_discard(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _execute(monkeypatch, retain_rejected=True)
    with pytest.raises(AssertionError, match="rejected staged assessment survived rollback"):
        _assert_rollback(*observed)
