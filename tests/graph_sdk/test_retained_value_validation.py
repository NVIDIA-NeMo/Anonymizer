# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained values obey their constructor shape and scalar contracts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._graph_values import GraphId, PlanId
from anonymizer.engine.graph_sdk.executor import MapItemKey
from anonymizer.graph import ActivationKey, DatumId, EffectRejected, InvocationId


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
