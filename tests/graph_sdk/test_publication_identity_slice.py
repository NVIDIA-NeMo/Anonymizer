# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Join all publication fact kinds in a real equal-text map evidence pipeline."""

from __future__ import annotations

import asyncio
from copy import copy
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.executor import LocalCompleted, LocalFailure, MapItemKey, OperationOutputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import AssociationInput, TextArtifactValue, TextCollectionValue
from anonymizer.graph import EffectRejected
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.qualification_map_callbacks import _ReferenceMapCallback
from tests.graph_sdk.qualification_map_execution import _execute_reference_map


@pytest.mark.parametrize("versioned", [False, True])
def test_equal_text_map_publication_keeps_all_identity_coordinates(
    monkeypatch: pytest.MonkeyPatch,
    versioned: bool,
) -> None:
    original = _ReferenceMapCallback.run

    async def run(self: _ReferenceMapCallback, request: tuple[AssociationInput, ...]) -> LocalCompleted | LocalFailure:
        returned = await original(self, request)
        if self.role == "EXP":
            assert isinstance(returned, LocalCompleted)
            row = returned.results[0]
            output = row.outputs[0]
            assert isinstance(output.value, TextCollectionValue)
            value = TextCollectionValue(
                items=tuple(replace(item, value=TextArtifactValue(text="equal")) for item in output.value.items)
            )
            return replace(returned, results=(replace(row, outputs=(replace(output, value=value),)),))
        return returned

    monkeypatch.setattr(_ReferenceMapCallback, "run", run)
    execution, result, nodes = asyncio.run(_execute_reference_map(2, versioned_items=versioned))
    target = next(iter(execution.context.prepared.data.targets))
    state = result.states[0]
    expander = next(entry.activation for entry in state.entries if entry.template == nodes["EXP"])
    members = {entry.activation for entry in state.entries if entry.template == nodes["MN"]}
    membership = next(item for item in state.expansions if item.parent == expander)
    assert membership.status == "closed" and membership.members == members and len(members) == 2
    items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    keys = [fact.key for fact in items if isinstance(fact.key, MapItemKey)]
    assert len(items) == 2 and {key.member for key in keys} == members
    assert {(key.item_key, key.item_version) for key in keys} == ({(0, 1), (0, 2)} if versioned else {(0, 1), (1, 1)})
    values = dict(result.artifacts)
    for fact in items:
        assert isinstance(fact.key, MapItemKey)
        key = fact.key
        assert (
            key.expander == expander and key.member.parent == expander and key.target == target and key.port == "item"
        )
        assert fact.artifact.invocation == expander.invocation and fact.artifact.version == key.item_version
        assert values[fact.artifact] == TextArtifactValue(text="equal")
        assert fact.parents == frozenset({OperationOutputKey(activation=expander, target=target, port="members")})
        port = next(port for port in result.ports if port.activation == key.member and port.port == "item")
        assert port.target == target and port.artifact == fact.artifact
        assessment = next(assessment for assessment in result.assessments if assessment.activation == key.member)
        evidence = next(port for port in result.ports if port.activation == key.member and port.port == "evidence")
        assert (
            assessment.node == nodes["MN"]
            and assessment.promise == "P_ITEM"
            and assessment.evidence_artifact == evidence.artifact
        )
        evidence_producer = next(producer for producer in result.provenance if producer.artifact == evidence.artifact)
        assert evidence_producer.parents == frozenset({key})
    assert len(result.assessments) == 3 and len(result.final_outputs) == 1
    final = result.final_outputs[0]
    root = next(entry.activation for entry in state.entries if entry.template == nodes["N"])
    assert final.target == final.candidate.target == target
    assert final.producer == OperationOutputKey(activation=root, target=target, port="result")
    assert (
        next(port.artifact for port in result.ports if port.activation == root and port.port == "result")
        == final.candidate.artifact
    )
    admitted, current, submissions = _inputs(execution, result, latest=versioned)
    qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(qualified.qualified) == (0 if versioned else 1)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == qualified
    # Another valid, equal-valued member is not interchangeable with this producer.
    corrupted = copy(result)
    substituted = copy(items[0])
    object.__setattr__(substituted, "artifact", items[1].artifact)
    object.__setattr__(
        corrupted, "provenance", tuple(substituted if fact is items[0] else fact for fact in result.provenance)
    )
    with pytest.raises(EffectRejected):
        qualify(admitted=admitted, result=corrupted, current=current, submissions=submissions)
