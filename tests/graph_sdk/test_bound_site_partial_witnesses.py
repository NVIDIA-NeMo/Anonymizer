# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bound provenance preserves actual declaration nodes and source versions."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from anonymizer.engine.graph_sdk.executor import BoundInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


@pytest.mark.parametrize("ordinary_context_hop", [False, True])
def test_bound_declarations_join_their_exact_node_port_and_versions(ordinary_context_hop: bool) -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a", "b"}), ordinary_context_hop=ordinary_context_hop
        )
    )
    bound = execution.context.bound_context
    assert bound is not None
    facts = tuple(item for item in result.provenance if isinstance(item.key, BoundInputKey))
    assert len(facts) == 4
    nodes = {cast(BoundInputKey, item.key).node for item in facts}
    assert len(nodes) == (2 if ordinary_context_hop else 1)
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
            assert not fact.parents
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
