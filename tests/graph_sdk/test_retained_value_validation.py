# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained values obey their constructor shape and scalar contracts."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.executor import MapItemKey
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import ActivationKey, ContractViolation, DatumId, GraphId, InvocationId, PlanId
from tests.graph_sdk import qualification_map_records as maps


def _map_item_key(**changes: Any) -> MapItemKey:
    invocation = InvocationId.new(plan=PlanId.new())
    expander = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    member = ActivationKey(invocation=invocation, occurrence=1, parent=expander, iteration=None)
    key = MapItemKey(
        expander=expander,
        member=member,
        target=DatumId.new(graph=GraphId.new()),
        port="item",
        item_key=0,
        item_version=1,
    )
    return replace(key, **changes)


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("item_key", -1, "invalid_value"),
        ("item_version", 0, "invalid_value"),
        ("item_key", True, "invalid_type"),
        ("item_version", False, "invalid_type"),
    ],
)
def test_map_item_key_scalars_reject_at_construction(field: str, value: object, expected: str) -> None:
    _map_item_key()
    with pytest.raises(EffectRejected) as error:
        _map_item_key(**{field: value})
    assert error.value.code.value == expected


@pytest.mark.parametrize(
    "field,value,expected,other",
    [
        (field, value, expected, other)
        for field, value, expected in [("item_key", 999, "missing"), ("item_version", 999, "missing")]
        for other in [None]
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
            object.__setattr__(fact, "parents", frozenset())
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
