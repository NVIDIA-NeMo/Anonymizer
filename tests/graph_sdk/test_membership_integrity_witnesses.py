# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check retained membership and terminal integrity after real nested execution."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import ActivationKey, ContractViolation, TaskAttemptId
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


@pytest.mark.parametrize(
    "corruption,expected",
    [
        ("duplicate-membership", "duplicate"),
        ("duplicate-terminal", "duplicate"),
        ("stray-terminal", "missing"),
        ("stray-member", "missing"),
        ("container-as-operation", "contradictory"),
        ("operation-as-container", "contradictory"),
        ("container-attempt", "contradictory"),
        ("category-mismatch", "contradictory"),
        ("failure-without-reason", "contradictory"),
    ],
)
def test_nested_retained_integrity_rejects_exact_corruption(
    corruption: str,
    expected: str,
) -> None:
    execution, result = asyncio.run(_execute_assessment(nested=True))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    container = next(t for t in result.record.terminals if t.structural)
    operation = next(t for t in result.record.terminals if not t.structural)
    unknown = ActivationKey(invocation=result.record.invocation, occurrence=100, parent=None, iteration=None)
    if corruption == "duplicate-membership":
        object.__setattr__(result.record, "memberships", (*result.record.memberships, result.record.memberships[0]))
    elif corruption == "duplicate-terminal":
        object.__setattr__(result.record, "terminals", (*result.record.terminals, operation))
    elif corruption == "stray-terminal":
        object.__setattr__(
            result.record, "terminals", (*result.record.terminals, replace(container, activation=unknown))
        )
    elif corruption == "stray-member":
        root = next(m for m in result.record.memberships if m.parent is None)
        object.__setattr__(root, "members", root.members | {unknown})
    elif corruption == "container-as-operation":
        object.__setattr__(container, "structural", False)
    elif corruption == "operation-as-container":
        object.__setattr__(operation, "structural", True)
    elif corruption == "container-attempt":
        object.__setattr__(container, "attempt", TaskAttemptId.new(activation=container.activation))
    else:
        object.__setattr__(operation, "category", "failure")
        if corruption == "category-mismatch":
            object.__setattr__(operation, "reasons", frozenset({"execution_failed"}))
    with pytest.raises((EffectRejected, ContractViolation)) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == expected


@pytest.mark.parametrize("corruption", ["missing-container-terminal", "nested-open"])
def test_incomplete_nested_membership_withholds_retained_success(corruption: str) -> None:
    execution, result = asyncio.run(_execute_assessment(nested=True))
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    if corruption == "missing-container-terminal":
        object.__setattr__(result.record, "terminals", tuple(t for t in result.record.terminals if not t.structural))
    else:
        nested = next(m for m in result.record.memberships if m.parent is not None)
        object.__setattr__(nested, "closed", False)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert "incomplete_membership" in output.targets[0].withholding
    assert output.record.statuses[0].completion == "pending"
