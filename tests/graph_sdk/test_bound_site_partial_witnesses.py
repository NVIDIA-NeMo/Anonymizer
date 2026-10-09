# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound provenance preserves actual declaration nodes and source versions."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.executor import BoundInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


@pytest.mark.parametrize("decision_context", [False, True])
def test_bound_declarations_join_their_exact_node_port_and_versions(decision_context: bool) -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a", "b"}), decision_context=decision_context
        )
    )
    bound = execution.context.bound_context
    assert bound is not None
    facts = tuple(item for item in result.provenance if isinstance(item.key, BoundInputKey))
    assert len(facts) == 4
    nodes = {cast(BoundInputKey, item.key).node for item in facts}
    assert len(nodes) == (2 if decision_context else 1)
    assert {item.key.port for item in facts} == {"a", "b"}
    for port in ("a", "b"):
        group = [item for item in facts if item.key.port == port]
        assert len(group) == 2
        assert {item.artifact.version for item in group} == {1, 2}
        assert len({item.artifact.key for item in group}) == 1
        for fact in group:
            assert isinstance(fact.key, BoundInputKey)
            (retained,) = tuple(item for item in bound.artifacts if item.reference == fact.key.binding_artifact)
            assert retained.node == fact.key.node
            assert retained.port == fact.key.port
            assert retained.target == fact.key.target
            assert retained.reference.version == fact.artifact.version
            assert not fact.parents and not fact.decision
        selected = next(
            item for item in result.ports if item.node == cast(BoundInputKey, group[0].key).node and item.port == port
        )
        assert selected.artifact == next(item.artifact for item in group if item.artifact.version == 2)
    assert len({item.artifact.key for item in facts}) == 2
    admitted, current, submissions = _inputs(execution, result, latest=True)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.record.statuses[0].qualification == "met"
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output


def test_retained_binding_reference_zero_version_rejects_before_lookup() -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a"}), versioned_context_ports=("a",)
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    fact = next(item for item in result.provenance if isinstance(item.key, BoundInputKey))
    original = fact.key
    assert isinstance(original, BoundInputKey)
    changed = replace(original, binding_artifact=replace(original.binding_artifact))
    object.__setattr__(changed.binding_artifact, "version", 0)
    parents = tuple((row, row.parents) for row in result.provenance)
    captures = result._input_parents
    object.__setattr__(fact, "key", changed)
    for row, old in parents:
        object.__setattr__(row, "parents", frozenset(changed if key == original else key for key in old))
    object.__setattr__(
        result, "_input_parents", tuple((t, a, p, changed if key == original else key) for t, a, p, key in captures)
    )
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.INVALID_VALUE
    finally:
        object.__setattr__(fact, "key", original)
        for row, old in parents:
            object.__setattr__(row, "parents", old)
        object.__setattr__(result, "_input_parents", captures)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
