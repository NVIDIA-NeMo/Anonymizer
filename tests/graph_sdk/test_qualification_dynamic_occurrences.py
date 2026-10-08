# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence production follows actual successful dynamic occurrences."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission, admit_qualification, verify_evidence
from anonymizer.engine.graph_sdk.executor import MapItemKey, OperationOutputKey, RootInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.test_effects_map_production_conformance import _execute_membership
from tests.graph_sdk.test_evidence import _qualification_limits
from tests.graph_sdk.test_qualification import _inputs


@pytest.mark.parametrize("member_count", [0, 1, 2])
def test_each_successful_map_member_retains_its_own_assessment(member_count: int) -> None:
    fixture, result, callbacks = asyncio.run(
        _execute_membership(member_count, member_assessment=True, artifact_byte_headroom=64)
    )
    execution = result._execution
    entries = [entry for state in result.states for entry in state.entries]
    members = [entry for entry in entries if entry.template == fixture.member_implementation]
    assert len(members) == member_count
    assert all(entry.status == "success" for entry in entries)
    facts = [fact for fact in result.assessments if fact.promise == "member_checked"]
    assert len(facts) == member_count
    assert {fact.activation for fact in facts} == {entry.activation for entry in members}
    assert len(result.assessments) == member_count + 1
    assert len(callbacks[fixture.member_implementation].calls) == member_count
    assert len({fact.evidence_artifact for fact in facts}) == member_count

    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    verified = verify_evidence(
        admitted=admitted,
        result=result,
        submissions=tuple(AssessmentSubmission(fact=fact) for fact in result.assessments),
    )
    assert len(verified) == member_count + 1
    for fact in facts:
        ports = {port.port: port for port in result.ports if port.activation == fact.activation}
        assert set(ports) == {"item", "subject", "evidence"}
        assert ports["subject"].role == "candidate"
        assert ports["item"].role == "artifact"
        assert fact.evidence_artifact == ports["evidence"].artifact
        evidence = next(item for item in verified if item.activation == fact.activation)
        assert evidence.subject.artifact == ports["subject"].artifact
        assert evidence.consumed_by_port == (("subject", evidence.subject),)
        output = next(item for item in result.provenance if item.artifact == fact.evidence_artifact)
        assert isinstance(output.key, OperationOutputKey)
        assert len(output.parents) == 1
        assert isinstance(next(iter(output.parents)), RootInputKey)
        dynamic_item = next(item for item in result.provenance if item.artifact == ports["item"].artifact)
        assert isinstance(dynamic_item.key, MapItemKey)
        assert dynamic_item.key.member == fact.activation
        assert len(dynamic_item.parents) == 1
        assert isinstance(next(iter(dynamic_item.parents)), OperationOutputKey)


def test_failed_expansion_does_not_fabricate_member_assessments() -> None:
    fixture, result, callbacks = asyncio.run(
        _execute_membership(2, member_assessment=True, response_mode="missing", artifact_byte_headroom=64)
    )
    assert not callbacks[fixture.member_implementation].calls
    assert not result.assessments
    assert all(
        entry.template != fixture.member_implementation or entry.status != "success"
        for state in result.states
        for entry in state.entries
    )
    admitted, current, submissions = _inputs(result._execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.verified
    assert not output.qualified
    assert output.targets[0].withholding


@pytest.mark.parametrize("mutation,code", [("missing", EffectCode.MISSING), ("duplicate", EffectCode.DUPLICATE)])
def test_retained_member_inventory_is_independent_of_submission_selection(mutation: str, code: EffectCode) -> None:
    _, result, _ = asyncio.run(_execute_membership(2, member_assessment=True, artifact_byte_headroom=64))
    admitted, current, _ = _inputs(result._execution, result)
    fact = next(item for item in result.assessments if item.promise == "member_checked")
    if mutation == "missing":
        object.__setattr__(result, "assessments", tuple(item for item in result.assessments if item is not fact))
    else:
        object.__setattr__(result, "assessments", (*result.assessments, fact))
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=())
    assert error.value.code is code


@pytest.mark.parametrize("submitted", [False, True])
def test_missing_member_terminal_keeps_unsubmitted_fact_unauthenticated(submitted: bool) -> None:
    _, result, _ = asyncio.run(_execute_membership(2, member_assessment=True, artifact_byte_headroom=64))
    fact = next(item for item in result.assessments if item.promise == "member_checked")
    object.__setattr__(
        result,
        "record",
        replace(
            result.record,
            terminals=tuple(item for item in result.record.terminals if item.activation != fact.activation),
        ),
    )
    admitted, current, _ = _inputs(result._execution, result)
    if submitted:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=(AssessmentSubmission(fact=fact),))
        assert error.value.code is EffectCode.MISSING
    else:
        output = qualify(admitted=admitted, result=result, current=current, submissions=())
        assert not output.qualified
        assert not output.verified
        assert "incomplete_membership" in output.targets[0].withholding
        assert output.record.statuses[0].completion == "pending"
