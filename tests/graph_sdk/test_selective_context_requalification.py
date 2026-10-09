# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Requalify real retained evidence against independent context revisions."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.evidence import evidence_revision_view, evidence_validity
from anonymizer.engine.graph_sdk.executor import BoundInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


@pytest.mark.parametrize("consumed", [frozenset(), frozenset({"a"}), frozenset({"b"}), frozenset({"a", "b"})])
@pytest.mark.parametrize("revision", [None, 1, 2])
@pytest.mark.parametrize("changed_port", ["a", "b"])
def test_context_revision_only_invalidates_evidence_that_consumed_it(
    consumed: frozenset[str], revision: int | None, changed_port: str
) -> None:
    execution, result = asyncio.run(_execute_separate_subject_context(versioned_context_dependencies=consumed))
    admitted, current, submissions = _inputs(execution, result, latest=True)
    bound = execution.context.bound_context
    assert bound is not None
    assert bound.receipt.requests.dispatched_count == 2
    assert len(bound.receipt.requests.settlements) == len(bound.receipt.cleanup) == 2
    versions = {
        fact.artifact.version: fact.artifact
        for fact in result.provenance
        if isinstance(fact.key, BoundInputKey) and fact.key.port == changed_port
    }
    assert set(versions) == {1, 2}
    assert versions[1].key == versions[2].key
    selected = tuple(ref for ref in current.artifacts if ref.key != versions[2].key)
    changed = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=selected + (() if revision is None else (versions[revision],)),
        absences=tuple(current.absences),
        configurations=current.configurations,
        state=current.state,
    )
    output = qualify(admitted=admitted, result=result, current=changed, submissions=submissions)
    (verified,) = output.verified
    assert {port for port, _ in verified.consumed_by_port} == consumed
    expected = (
        "current" if changed_port not in consumed or revision == 2 else "unknown" if revision is None else "stale"
    )
    assert evidence_validity(evidence=verified, current=changed) == expected
    assert bool(output.qualified) == (expected == "current")
    assert (
        output.record.statuses[0].qualification == {"current": "met", "stale": "unmet", "unknown": "unknown"}[expected]
    )
    assert qualify(admitted=admitted, result=result, current=changed, submissions=submissions) == output
    restored = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(restored.qualified) == 1
    assert restored.qualified[0].candidate == output.targets[0].candidate
    assert bound.receipt.requests.dispatched_count == 2


@pytest.mark.parametrize("relation", ["independent", "dependency", "atomic", "mixed"])
def test_context_currentness_propagates_only_through_declared_relations(relation: str) -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a", "b"}),
            target_labels=("A", "B", "C"),
            dependencies=((0, 1),) if relation == "dependency" else ((1, 2),) if relation == "mixed" else (),
            atomic_groups=((0, 1),) if relation in {"atomic", "mixed"} else (),
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True, port_facts=32, provenance_edges=32)
    labels = {datum.id: datum.text for datum in execution.context.prepared.data.datums}
    target_a = next(target for target in execution.context.prepared.data.targets if labels[target] == "A")
    b_port = next(port for port in result.ports if port.target == target_a and port.port == "b")
    changed = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref in current.artifacts if ref != b_port.artifact),
        absences=tuple(current.absences),
        configurations=current.configurations,
        state=current.state,
    )
    output = qualify(admitted=admitted, result=result, current=changed, submissions=submissions)
    expected = {"B", "C"} if relation == "independent" else set() if relation == "mixed" else {"C"}
    assert {labels[item.target] for item in output.qualified} == expected
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 3
