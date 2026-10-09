# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise structural ownership through actual retained execution fields."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph._values import ContractViolation, InvocationId, TaskAttemptId, ValidationCode
from tests.graph_sdk.test_evidence import _execute_assessment
from tests.graph_sdk.test_qualification import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_target_ownership_cannot_be_changed_by_reordering_retained_states() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context(target_labels=("A", "B")))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    assert len(result.states) == 2
    object.__setattr__(result, "states", tuple(reversed(result.states)))
    with pytest.raises(EffectRejected) as rejected:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert rejected.value.code is EffectCode.FOREIGN_OWNER


def test_terminal_cannot_rebind_its_activation_to_another_invocation() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    (terminal,) = result.record.terminals
    foreign = replace(terminal.activation, invocation=InvocationId.new(plan=result.record.plan))
    object.__setattr__(terminal, "activation", foreign)
    # Foreign invocation ownership precedes terminal-to-entry presence.
    with pytest.raises(EffectRejected) as rejected:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert rejected.value.code is EffectCode.FOREIGN_OWNER


def test_duplicate_retained_membership_rows_preserve_observable_multiplicity() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    (root,) = result.record.memberships
    assert root.parent is None and len(root.members) == 1
    object.__setattr__(result.record, "memberships", (root, root))
    with pytest.raises(EffectRejected) as rejected:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert rejected.value.code is EffectCode.DUPLICATE


def test_admitted_container_requires_a_structural_terminal() -> None:
    execution, result = asyncio.run(_execute_assessment(nested=True))
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    (container,) = tuple(terminal for terminal in result.record.terminals if terminal.structural)
    assert container.attempt is None
    object.__setattr__(container, "structural", False)
    with pytest.raises(ContractViolation) as rejected:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert rejected.value.code is ValidationCode.CONTRADICTORY


def test_child_membership_cannot_cross_prepared_target_owners() -> None:
    execution, result = asyncio.run(_execute_assessment(nested=True, target_count=2))
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 2
    first, second = result.states
    (child,) = tuple(entry for entry in first.entries if entry.activation.parent is not None)
    (other_root,) = tuple(entry.activation for entry in second.entries if entry.activation.parent is None)
    corrupted = replace(child, activation=replace(child.activation, parent=other_root))
    object.__setattr__(first, "entries", frozenset(corrupted if entry is child else entry for entry in first.entries))
    object.__setattr__(
        first,
        "reservations",
        frozenset(
            replace(seed, activation=corrupted.activation) if seed.activation == child.activation else seed
            for seed in first.reservations
        ),
    )
    with pytest.raises(EffectRejected) as rejected:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert rejected.value.code is EffectCode.FOREIGN_OWNER


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "foreign-duplicate", "invalid-foreign", "scalar-foreign", "oversized-invalid"]
)
def test_terminal_inventory_precedence(fault: str) -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    original = result.record.terminals
    (terminal,) = original
    changed = replace(terminal)
    expected: EffectCode | ValidationCode
    error_type: type[EffectRejected] | type[ContractViolation] = EffectRejected
    if fault == "missing":
        absent = replace(terminal.activation, occurrence=999)
        changed = replace(terminal, activation=absent, attempt=TaskAttemptId.new(activation=absent))
        rows = (changed,)
        expected = EffectCode.MISSING
    elif fault == "duplicate":
        rows = (terminal, terminal)
        expected = EffectCode.DUPLICATE
    elif fault == "oversized-invalid":
        rows = (None,) * (execution.context.prepared.limits.max_total_activation_slots + 1)
        expected = EffectCode.LIMIT_EXCEEDED
    else:
        foreign = replace(terminal.activation, invocation=InvocationId.new(plan=result.record.plan))
        object.__setattr__(changed, "activation", foreign)
        rows = (changed, changed)
        expected = EffectCode.FOREIGN_OWNER
        if fault in ("invalid-foreign", "scalar-foreign"):
            object.__setattr__(changed, "category", 1 if fault == "invalid-foreign" else "invalid")
            error_type = ContractViolation
            expected = ValidationCode.INVALID_TYPE if fault == "invalid-foreign" else ValidationCode.INVALID_VALUE
    try:
        object.__setattr__(result.record, "terminals", rows)
        with pytest.raises(error_type) as rejected:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert rejected.value.code is expected
    finally:
        object.__setattr__(result.record, "terminals", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
