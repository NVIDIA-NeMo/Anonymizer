# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact retained root, final output, and membership discriminators."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.executor import RootInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from anonymizer.graph._values import ContractViolation
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import (
    _execute_separate_subject_context,
)


def test_final_root_projection_cannot_select_another_input() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    final = result.final_outputs[0]
    other = next(
        fact
        for fact in result.provenance
        if isinstance(fact.key, RootInputKey) and fact.artifact != final.candidate.artifact
    )
    original = final.producer
    object.__setattr__(final, "producer", other.key)
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(
                admitted=admitted,
                result=result,
                current=current,
                submissions=submissions,
            )
        assert error.value.code is EffectCode.CONTRADICTORY
    finally:
        object.__setattr__(final, "producer", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline


def test_execution_only_open_root_membership_is_pending_not_assessed() -> None:
    execution, result = asyncio.run(_execute_assessment(execution_only=True))
    admitted, current, submissions = _inputs(execution, result)
    (root,) = result.record.memberships
    assert root.parent is None and root.closed
    original = root.closed
    object.__setattr__(root, "closed", False)
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        assert output.record.statuses[0].completion == "pending"
        assert output.record.statuses[0].qualification == "not_assessed"
        assert output.targets[0].withholding == frozenset({"execution_only", "incomplete_membership"})
    finally:
        object.__setattr__(root, "closed", original)


@pytest.mark.parametrize("inventory", ["membership"])
def test_owned_entry_missing_retained_inventory_is_missing(inventory: str) -> None:
    execution, result = asyncio.run(_execute_assessment())
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    entry = next(iter(result.states[0].entries))
    assert any(seed.activation == entry.activation for seed in result.states[0].reservations)
    assert any(entry.activation in membership.members for membership in result.record.memberships)
    if inventory == "terminal":
        object.__setattr__(
            result.record,
            "terminals",
            tuple(row for row in result.record.terminals if row.activation != entry.activation),
        )
    else:
        object.__setattr__(result.record, "memberships", ())
    with pytest.raises((EffectRejected, ContractViolation)) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == "missing"
