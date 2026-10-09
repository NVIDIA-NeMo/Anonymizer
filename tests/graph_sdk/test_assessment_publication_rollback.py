# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Assessment-bearing publication rollback with an already accepted peer."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts, _FactCheckpoint
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


def _execute(
    monkeypatch: pytest.MonkeyPatch, *, retain_rejected: bool = False
) -> tuple[_MapFixture, ExecutionResult, dict[NodeId, _MapCallback], _Discard]:
    discarded: list[_Discard] = []
    restore = _ExecutionFacts.restore

    def observe(facts: _ExecutionFacts, checkpoint: _FactCheckpoint) -> None:
        # Successful commits already append to checkpoint.assessments. Only a
        # rejected, uncommitted assessment is absent from that authoritative base.
        pending = [fact for fact in facts.assessments if all(fact is not prior for prior in checkpoint.assessments)]
        if pending:
            (rejected,) = pending
            assert len(checkpoint.assessments) == 1
            assert rejected.evidence_artifact in facts.values
            staged = facts.values[rejected.evidence_artifact]
            assert isinstance(staged, TextCollectionValue) and len(staged.items) == 2
            assert any(port.artifact == rejected.evidence_artifact for port in facts.ports)
            assert any(producer.artifact == rejected.evidence_artifact for producer in facts.provenance)
            discarded.append(_Discard(tuple(checkpoint.assessments), rejected))
        restore(facts, checkpoint)
        if pending and retain_rejected:
            # Fault injection retains the real staged fact. No assessment is
            # fabricated, and every other owner follows its ordinary rollback.
            facts.assessments.extend(pending)

    monkeypatch.setattr(_ExecutionFacts, "restore", observe)
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
    (discard,) = discarded
    return fixture, result, callbacks, discard


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
    assert not result.requests.defects and not result.cleanup and not result.pending_decisions


def test_rejected_assessment_delta_preserves_accepted_peer(monkeypatch: pytest.MonkeyPatch) -> None:
    _assert_rollback(*_execute(monkeypatch))


def test_witness_detects_suppressed_assessment_discard(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = _execute(monkeypatch, retain_rejected=True)
    with pytest.raises(AssertionError, match="rejected staged assessment survived rollback"):
        _assert_rollback(*observed)
