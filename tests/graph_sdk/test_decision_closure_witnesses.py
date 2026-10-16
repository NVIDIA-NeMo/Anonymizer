# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Required decisions follow candidate ancestry and actual consumed decisions."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.evidence import verify_evidence
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.records import DecisionRef
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


def test_decision_closure_traverses_an_actual_extra_nondecision_hop() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True, decision_hop=True))
    admitted, current, submissions = _inputs(execution, result, decisions=1)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    (final,) = result.final_outputs
    producers = {item.key: item for item in result.provenance}
    final_fact = producers[final.producer]
    (hop,) = (producers[key] for key in final_fact.parents)
    (decision,) = (producers[key] for key in hop.parents)
    assert not final_fact.decision and not hop.decision and decision.decision
    assert final_fact.key != hop.key != decision.key
    assert hop.artifact == decision.artifact  # Identity does not collapse producer occurrences.
    assert output.required_decisions == frozenset({DecisionRef(artifact=decision.artifact)})
    assert len(result.record.terminals) == 3
    assert all(item.category == "success" for item in result.record.terminals)
    assert not result.requests.dispatches and not result.cleanup


def test_evidence_consumed_decision_is_required_without_candidate_decision_ancestry() -> None:
    execution, result = asyncio.run(
        _execute_assessment(
            decision_input=True,
            candidate_input=True,
            root_passthrough=True,
            separate_decision_subject=True,
        )
    )
    admitted, current, submissions = _inputs(execution, result, decisions=1)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    (final,) = result.final_outputs
    producer = next(item for item in result.provenance if item.key == final.producer)
    assert not producer.decision and not producer.parents
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    ((port, decision),) = verified.consumed_by_port
    assert port == "input" and isinstance(decision, DecisionRef)
    assert verified.subject.artifact == decision.artifact
    subject = next(item for item in result.ports if item.port == "subject")
    consumed = next(item for item in result.ports if item.activation == subject.activation and item.port == "input")
    assert subject.role == "candidate" and consumed.role == "decision"
    assert subject.artifact == consumed.artifact == decision.artifact
    assert output.required_decisions == frozenset({decision})
    assert len(result.record.terminals) == 2


def test_executed_decision_outside_candidate_and_direct_consumed_decisions_is_excluded() -> None:
    execution, result = asyncio.run(
        _execute_assessment(
            decision_input=True,
            decision_hop=True,
            candidate_input=True,
            root_passthrough=True,
        )
    )
    admitted, current, submissions = _inputs(execution, result, decisions=0)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert len([item for item in result.provenance if item.decision]) == 1
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    assert verified.consumed_by_port
    assert not any(isinstance(item, DecisionRef) for _, item in verified.consumed_by_port)
    (final,) = result.final_outputs
    producer = next(item for item in result.provenance if item.key == final.producer)
    assert not producer.decision and not producer.parents
    assert not output.required_decisions
    assert len(result.record.terminals) == 3


def test_candidate_ancestry_and_evidence_share_one_decision_budget_entry() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True))
    admitted, current, submissions = _inputs(execution, result, decisions=1)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    ((_, decision),) = verified.consumed_by_port
    assert isinstance(decision, DecisionRef)
    final = next(item for item in result.provenance if item.key == result.final_outputs[0].producer)
    assert any(
        item.key in final.parents and item.decision and item.artifact == decision.artifact for item in result.provenance
    )
    assert output.required_decisions == output.qualified[0].required_decisions == frozenset({decision})
    admitted, current, submissions = _inputs(execution, result, decisions=0)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_retained_decision_producer_rejects_another_executed_target_owner() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True, target_count=2))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    first, second = (item for item in result.provenance if item.decision)
    assert first.key.target != second.key.target
    original_key = first.key
    changed_key = replace(original_key, target=second.key.target)
    original_inputs = result._input_parents
    changed_parents = [(item, item.parents) for item in result.provenance if original_key in item.parents]
    object.__setattr__(first, "key", changed_key)
    # Preserve reference resolution: owner corruption must not become a dangling-key test.
    object.__setattr__(
        result,
        "_input_parents",
        tuple(
            (target, activation, port, changed_key if parent == original_key else parent)
            for target, activation, port, parent in original_inputs
        ),
    )
    for item, parents in changed_parents:
        object.__setattr__(item, "parents", (parents - {original_key}) | {changed_key})
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.FOREIGN_OWNER
    finally:
        object.__setattr__(first, "key", original_key)
        object.__setattr__(result, "_input_parents", original_inputs)
        for item, parents in changed_parents:
            object.__setattr__(item, "parents", parents)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
