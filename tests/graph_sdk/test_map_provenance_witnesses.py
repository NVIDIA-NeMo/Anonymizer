# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained map coordinates and collection parents after actual execution."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.executor import MapItemKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectRejected
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.qualification_map_execution import _execute_reference_map


@pytest.mark.parametrize("two_maps", [False, True])
def test_map_items_retain_map_item_producers(two_maps: bool) -> None:
    execution, result, _ = asyncio.run(_execute_reference_map(1 if two_maps else 2, two_maps=two_maps))
    admitted, current, submissions = _inputs(execution, result, port_facts=64, provenance_edges=64)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    first, second = items[:2]
    assert isinstance(first.key, MapItemKey) and isinstance(second.key, MapItemKey)


@pytest.mark.parametrize(
    "corruption",
    ["membership-port"],
)
def test_map_collection_output_identity_and_declaration(corruption: str) -> None:
    execution, result, nodes = asyncio.run(_execute_reference_map(1))
    admitted, current, submissions = _inputs(execution, result)
    assert (
        len(
            qualify(
                admitted=admitted,
                result=result,
                current=current,
                submissions=submissions,
            ).qualified
        )
        == 1
    )
    (publication,) = execution.map_expansions
    object.__setattr__(publication, "membership_port", "not_members")
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == "contradictory"
