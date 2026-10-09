# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only released targets contribute their actual decision ancestry."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


@pytest.mark.parametrize("released_index", [0, 1])
def test_withheld_target_does_not_contribute_unrelated_decision(released_index: int) -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True, target_count=2))
    admitted, current, submissions = _inputs(execution, result, decisions=2)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == len(baseline.required_decisions) == 2
    released = baseline.qualified[released_index]
    selected = tuple(
        item
        for item in submissions
        if any(port.activation == item.fact.activation and port.target == released.target for port in result.ports)
    )
    assert len(selected) == 1
    output = qualify(admitted=admitted, result=result, current=current, submissions=selected)
    assert output.qualified == (released,)
    assert output.required_decisions == released.required_decisions
    assert len(output.required_decisions) == 1
    other = baseline.qualified[1 - released_index]
    assert not output.required_decisions & other.required_decisions
    assert next(t for t in output.targets if t.target == other.target).withholding == frozenset({"missing_assessment"})
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline


def test_decision_producer_requires_its_retained_artifact() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.required_decisions) == 1
    artifact = next(p.artifact for p in result.provenance if p.decision)
    original = result.record.artifacts
    stored = result.artifacts
    object.__setattr__(result, "artifacts", tuple(pair for pair in stored if pair[0] != artifact))
    object.__setattr__(result.record, "artifacts", original - {artifact})
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.MISSING
    finally:
        object.__setattr__(result, "artifacts", stored)
        object.__setattr__(result.record, "artifacts", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
