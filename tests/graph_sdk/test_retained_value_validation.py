# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained values obey their constructor shape and scalar contracts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.executor import MapItemKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph._values import ContractViolation, InvocationId
from tests.graph_sdk import qualification_map_records as maps
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


@pytest.mark.parametrize(
    "field,value,expected,other",
    [
        (field, value, expected, other)
        for field, value, expected in [
            ("item_key", -1, "invalid_value"),
            ("item_version", 0, "invalid_value"),
            ("item_key", True, "invalid_type"),
            ("item_version", False, "invalid_type"),
            ("item_key", 999, "missing"),
            ("item_version", 999, "missing"),
        ]
        for other in [None, "owner", "contradiction"]
        if expected != "missing" or other != "contradiction"
    ],
)
def test_retained_map_scalars_precede_lookup_and_relations(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: int,
    expected: str,
    other: str | None,
) -> None:
    baseline = next(case for case in maps.CORPUS if case["case_id"] == "map_item_evidence/direct_1")
    original = maps.qualify

    def check(**kwargs: Any) -> Any:
        result = kwargs["result"]
        assert original(**kwargs).qualified
        fact = next(row for row in result.provenance if isinstance(row.key, MapItemKey))
        old = fact.key
        invocation = result.record.invocation
        if expected != "missing":
            with pytest.raises(EffectRejected) as error:
                replace(old, **{field: value})
            assert error.value.code.value == expected
        new = replace(old)
        object.__setattr__(new, field, value)
        # Replace every hash-bearing reference rather than mutating a key in a set.
        for row in result.provenance:
            if row.key == old:
                object.__setattr__(row, "key", new)
            object.__setattr__(row, "parents", frozenset(new if p == old else p for p in row.parents))
        object.__setattr__(
            result,
            "_input_parents",
            tuple(
                (target, activation, port, new if parent == old else parent)
                for target, activation, port, parent in result._input_parents
            ),
        )
        if other == "owner":
            object.__setattr__(result.record, "invocation", InvocationId.new(plan=result.record.plan))
        elif other == "contradiction":
            object.__setattr__(fact, "decision", True)
        with pytest.raises((EffectRejected, ContractViolation)) as error:
            original(**kwargs)
        assert error.value.code.value == ("foreign_owner" if expected == "missing" and other == "owner" else expected)
        object.__setattr__(result.record, "invocation", invocation)
        return original_result

    # The adapter only needs a successful original result for its post-qualification projection.
    original_result: Any = None

    def capture(**kwargs: Any) -> Any:
        nonlocal original_result
        original_result = original(**kwargs)
        check(**kwargs)
        return original_result

    monkeypatch.setattr(maps, "qualify", capture)
    asyncio.run(maps._run_reference_map_case(baseline))


@pytest.mark.parametrize("shape", ["tuple", "list", "set", "member", "closed", "parent", "invocation"])
@pytest.mark.parametrize("other", [None, "owner", "duplicate"])
def test_retained_membership_shape_rejects_before_relations(shape: str, other: str | None) -> None:
    execution, result = asyncio.run(_execute_assessment(nested=True))
    admitted, current, submissions = _inputs(execution, result)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified
    root = result.record.memberships[0]
    member = next(iter(root.members))
    field, value = {
        "tuple": ("members", (member, member)),
        "list": ("members", [member, member]),
        "set": ("members", {member}),
        "member": ("members", frozenset({"invalid"})),
        "closed": ("closed", 1),
        "parent": ("parent", "invalid"),
        "invocation": ("invocation", "invalid"),
    }[shape]
    with pytest.raises(ContractViolation) as error:
        replace(root, **{field: value})
    assert error.value.code.value == "invalid_type"
    object.__setattr__(root, field, value)
    if other == "owner":
        object.__setattr__(result.record, "invocation", InvocationId.new(plan=result.record.plan))
    elif other == "duplicate":
        object.__setattr__(result.record, "memberships", (*result.record.memberships, root))
    with pytest.raises((EffectRejected, ContractViolation)) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == "invalid_type"


@pytest.mark.parametrize("collection", ["provenance", "memberships", "members", "ports"])
def test_retained_limits_precede_malformed_membership_fields(collection: str) -> None:
    execution, result = asyncio.run(_execute_assessment(nested=True))
    admitted, current, submissions = _inputs(execution, result)
    root = result.record.memberships[0]
    limit = execution.context.prepared.limits.max_total_activation_slots
    if collection == "provenance":
        object.__setattr__(result, "provenance", (result.provenance[0],) * (admitted.limits.max_port_facts + 1))
        object.__setattr__(root, "closed", 1)
    elif collection == "memberships":
        object.__setattr__(result.record, "memberships", (root,) * (limit + 2))
        object.__setattr__(root, "closed", 1)
    elif collection == "members":
        object.__setattr__(root, "members", (next(iter(root.members)),) * (limit + 1))
    else:
        object.__setattr__(result, "ports", (result.ports[0],) * (admitted.limits.max_port_facts + 1))
        object.__setattr__(root, "closed", 1)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == "limit_exceeded"
