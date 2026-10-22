# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded snapshot projections from actual executor evidence to frozen cases."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.evidence import evidence_revision_view, evidence_validity
from anonymizer.engine.graph_sdk.executor import BoundInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.records import CandidateRef
from anonymizer.graph import CoverageAtom
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context

TOPOLOGIES = {
    "independent": ((), ()),
    "chain": (((0, 1), (1, 2)), ()),
    "mixed": (((0, 1),), ((1, 2),)),
    "reverse": (((1, 0),), ()),
}
REFERENCE = {
    row["id"]: row
    for line in (Path(__file__).parent / "reference/evidence_v1/cases.jsonl").read_text().splitlines()
    if (row := json.loads(line))
}
# Frozen ordering: selection, topology, revision axis, revision state, submission.
SNAPSHOTS = [
    (topology, axis, state, submitted, REFERENCE[f"P7E-{((index * 7 + axis) * 3 + state) * 4 + submitted:05d}"])
    for index, topology in enumerate(TOPOLOGIES)
    for axis, states in ((0, (0, 2)), (1, (0, 1, 2)))
    for state in states
    for submitted in range(4)
]


@pytest.fixture(scope="module")
def executions() -> dict[str, Any]:
    runs = {}
    for topology, (dependencies, atomic) in TOPOLOGIES.items():
        execution, result = asyncio.run(
            _execute_separate_subject_context(
                versioned_context_dependencies=frozenset({"a"}),
                coverage=frozenset(CoverageAtom(kind="field", name=name) for name in ("K0", "K1")),
                requirement_coverage=frozenset({CoverageAtom(kind="field", name="K0")}),
                expose_evidence=False,
                target_labels=("A", "B", "C"),
                dependencies=dependencies,
                atomic_groups=atomic,
            )
        )
        admitted, current, submissions = _inputs(execution, result, latest=True, port_facts=32, provenance_edges=32)
        labels = {datum.id: datum.text for datum in execution.context.prepared.data.datums}
        assert len(result.assessments) == len(result.final_outputs) == 3
        bound = execution.context.bound_context
        assert bound is not None and bound.receipt.requests.dispatched_count == 6
        assert len(bound.receipt.requests.settlements) == 6 and len(bound.receipt.cleanup) == 2
        assert not result.requests.dispatches and not result.cleanup
        runs[topology] = admitted, current, submissions, result, labels
    return runs


@pytest.mark.parametrize("snapshot", SNAPSHOTS, ids=lambda item: item[-1]["id"])
def test_compact_evidence_snapshot(snapshot: tuple[Any, ...], executions: dict[str, Any]) -> None:
    topology, axis, state, submitted, reference = snapshot
    admitted, current, submissions, result, labels = executions[topology]
    targets = {p.activation: p.target for p in result.ports}
    target_label = "A" if axis == 0 else "B"
    port_name = "subject" if axis == 0 else "a"
    (port,) = [p for p in result.ports if labels[p.target] == target_label and p.port == port_name]
    selected = port.artifact
    replacement = ()
    if state == 0:
        replacement = (selected,)
    elif state == 1:
        assert axis == 1
        (older,) = [
            fact.artifact
            for fact in result.provenance
            if isinstance(fact.key, BoundInputKey) and fact.artifact.key == selected.key and fact.artifact.version == 1
        ]
        assert selected.version == 2 and older.invocation == selected.invocation
        replacement = (older,)
    changed = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref in current.artifacts if ref.key != selected.key) + replacement,
        configurations=current.configurations,
    )
    submitted_label = ("all", "A", "B", "none")[submitted]
    chosen = tuple(
        s for s in submissions if submitted_label == "all" or labels[targets[s.fact.activation]] == submitted_label
    )
    output = qualify(admitted=admitted, result=result, current=changed, submissions=chosen)
    expected_verified, _, expected_released, expected_decisions = reference["expected"]
    assert {labels[item.target] for item in output.qualified} == set(expected_released)
    assert expected_decisions == []  # The reference models required decisions; the SDK has none.
    actual = {}
    for verified in output.verified:
        assert isinstance(verified.subject, CandidateRef)
        target = labels[verified.subject.target]
        (submission,) = [s for s in chosen if labels[targets[s.fact.activation]] == target]
        fact = submission.fact
        (final,) = [f for f in result.final_outputs if f.target == targets[fact.activation]]
        assert verified.subject == final.candidate
        assert verified.reference.artifact == fact.evidence_artifact
        assert verified.reference.artifact.invocation == result.record.invocation
        assert verified.promise.coverage == frozenset(CoverageAtom(kind="field", name=n) for n in ("K0", "K1"))
        ((consumed_port, consumed_ref),) = verified.consumed_by_port
        assert consumed_port == "a"
        (input_port,) = [p for p in result.ports if p.target == targets[fact.activation] and p.port == "a"]
        assert consumed_ref == input_port.artifact and input_port.activation == fact.activation
        assert input_port.node == fact.node
        actual[target] = evidence_validity(evidence=verified, current=changed)
    assert actual == {row[0]: row[7] for row in expected_verified}
    assert qualify(admitted=admitted, result=result, current=changed, submissions=chosen) == output
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 3
