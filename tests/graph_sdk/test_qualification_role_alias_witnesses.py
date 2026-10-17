# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification validates roles per occurrence while preserving lawful aliases."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.executor import AssessmentFinding
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_candidate_and_evidence_can_identify_the_same_artifact() -> None:
    execution, result = asyncio.run(_execute_assessment(alias_output=True))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == len(output.verified) == 1
    candidate = output.qualified[0].candidate
    assert output.verified[0].reference.artifact == candidate.artifact
    assert output.verified[0].subject == candidate


def test_distinct_predecessor_and_candidate_occurrences_preserve_identity_alias() -> None:
    execution, result = asyncio.run(_execute_assessment(predecessor_input=True, alias_output=True))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    candidate = output.qualified[0].candidate
    (root,) = (fact for fact in result.provenance if not fact.parents)
    assert root.artifact == candidate.artifact
    occurrences = [item for item in result.ports if item.artifact == candidate.artifact]
    assert {item.role for item in occurrences} == {"artifact", "candidate"}
    assert len({(item.activation, item.port) for item in occurrences}) == len(occurrences)
    assert output.verified[0].subject == candidate


@pytest.mark.parametrize("released_index", [0, 1])
def test_withheld_target_and_unrelated_executed_hops_contribute_nothing(released_index: int) -> None:
    findings = tuple(
        AssessmentFinding(status="satisfied" if index == released_index else "unsatisfied", code="observed")
        for index in range(2)
    )
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a"}),
            ordinary_context_hop=True,
            expose_evidence=False,
            target_labels=("A", "B"),
            target_findings=findings,
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True, port_facts=32, provenance_edges=32)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    labels = {datum.id: datum.text for datum in execution.context.prepared.data.datums}
    (released,) = output.qualified
    assert labels[released.target] == ("A", "B")[released_index]
    assert not next(item for item in output.targets if item.target == released.target).withholding
    other = next(item for item in output.targets if item.target != released.target)
    assert other.withholding == frozenset({"assessment_unsatisfied"})
    assert len(submissions) == 2 and len(result.record.terminals) == 4
    assert all(item.category == "success" for item in result.record.terminals)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
