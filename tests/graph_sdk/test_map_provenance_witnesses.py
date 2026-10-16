# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained map coordinates and collection parents after actual execution."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.executor import (
    ExecutionResult,
    MapItemKey,
    OperationOutputKey,
    RootInputKey,
)
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import TextCollectionValue
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import ActivationKey, DatumId
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.qualification_map_execution import _execute_reference_map


def _replace_map_key(result: ExecutionResult, old: MapItemKey, new: MapItemKey) -> None:
    """Keep every retained reference to the neutral producer identity coherent."""
    for fact in result.provenance:
        if fact.key == old:
            object.__setattr__(fact, "key", new)
        object.__setattr__(
            fact,
            "parents",
            frozenset(new if key == old else key for key in fact.parents),
        )
    object.__setattr__(
        result,
        "_input_parents",
        tuple(
            (target, activation, port, new if key == old else key)
            for target, activation, port, key in result._input_parents
        ),
    )


@pytest.mark.parametrize(
    "corruption,expected",
    [
        ("wrong-port", "contradictory"),
        ("missing-parent", "contradictory"),
        ("wrong-parent", "contradictory"),
        ("multiple-parents", "contradictory"),
        ("unrelated-collection", "contradictory"),
        ("swapped-items", "contradictory"),
        ("foreign-target", "foreign_owner"),
        ("missing-expander", "missing"),
    ],
)
def test_actual_map_provenance_rejects_exact_coordinate_corruption(corruption: str, expected: str) -> None:
    execution, result, _ = asyncio.run(
        _execute_reference_map(
            1 if corruption == "unrelated-collection" else 2,
            two_maps=corruption == "unrelated-collection",
        )
    )
    admitted, current, submissions = _inputs(execution, result, port_facts=64, provenance_edges=64)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    first, second = items[:2]
    old = first.key
    assert isinstance(old, MapItemKey) and isinstance(second.key, MapItemKey)
    if corruption == "wrong-port":
        _replace_map_key(result, old, replace(old, port="not_item_input"))
    elif corruption == "foreign-target":
        _replace_map_key(result, old, replace(old, target=DatumId.new(graph=old.target.graph)))
    elif corruption == "missing-expander":
        other = ActivationKey(
            invocation=old.expander.invocation,
            occurrence=999,
            parent=None,
            iteration=None,
        )
        changed = replace(old)
        object.__setattr__(changed, "expander", other)
        _replace_map_key(result, old, changed)
    elif corruption == "swapped-items":
        assert old.item_key != second.key.item_key
        _replace_map_key(
            result,
            old,
            replace(old, item_key=second.key.item_key, item_version=second.key.item_version),
        )
    else:
        root = next(fact.key for fact in result.provenance if isinstance(fact.key, RootInputKey))
        if corruption == "missing-parent":
            parents = frozenset()
        elif corruption == "wrong-parent":
            parents = frozenset({root})
        elif corruption == "multiple-parents":
            parents = first.parents | {root}
        else:
            other_item = next(
                fact for fact in items if isinstance(fact.key, MapItemKey) and fact.key.expander != old.expander
            )
            parents = other_item.parents
            assert parents != first.parents and all(isinstance(key, OperationOutputKey) for key in parents)
        object.__setattr__(first, "parents", parents)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == expected


@pytest.mark.parametrize(
    "corruption",
    ["collection-artifact", "collection-items", "producer-activation", "membership-port"],
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
    item = next(fact for fact in result.provenance if isinstance(fact.key, MapItemKey))
    (parent_key,) = item.parents
    assert isinstance(parent_key, OperationOutputKey)
    parent = next(fact for fact in result.provenance if fact.key == parent_key)
    if corruption == "collection-artifact":
        port = next(
            port for port in result.ports if port.activation == parent_key.activation and port.port == parent_key.port
        )
        root = next(fact for fact in result.provenance if isinstance(fact.key, RootInputKey))
        assert root.artifact != parent.artifact
        object.__setattr__(port, "artifact", root.artifact)
    elif corruption == "collection-items":
        collection = dict(result.artifacts)[parent.artifact]
        assert isinstance(collection, TextCollectionValue) and len(collection.items) == 1
        object.__setattr__(collection, "items", (replace(collection.items[0], key=1),))
    elif corruption == "expansion-outcome":
        mapping = next(mapping for scope in execution.context.prepared.workflow.bodies() for mapping in scope.maps)
        assert mapping.expansion_outcomes == frozenset({"ok"})
        state = result.states[0]
        expander = next(entry for entry in state.entries if entry.template == mapping.expander)
        object.__setattr__(
            state,
            "entries",
            frozenset(replace(entry, outcome="other") if entry is expander else entry for entry in state.entries),
        )
        assert mapping.expansion_outcomes == frozenset({"ok"})
    elif corruption == "producer-activation":
        other = next(entry.activation for entry in result.states[0].entries if entry.template == nodes["N"])
        changed = replace(parent_key, activation=other)
        object.__setattr__(parent, "key", changed)
        for fact in result.provenance:
            object.__setattr__(
                fact,
                "parents",
                frozenset(changed if key == parent_key else key for key in fact.parents),
            )
        object.__setattr__(
            result,
            "_input_parents",
            tuple((t, a, p, changed if key == parent_key else key) for t, a, p, key in result._input_parents),
        )
    else:
        (publication,) = execution.map_expansions
        object.__setattr__(publication, "membership_port", "not_members")
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == ("missing" if corruption == "collection-items" else "contradictory")
