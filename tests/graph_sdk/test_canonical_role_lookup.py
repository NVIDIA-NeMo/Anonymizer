# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical lookup preserves role and source identity without global scans."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from copy import copy
from typing import TypeVar

import pytest

from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts
from anonymizer.engine.graph_sdk._execution_topology import _inherited_artifact_role
from anonymizer.engine.graph_sdk.executor import ExecutionResult, MapItemKey, OperationOutputKey
from tests.graph_sdk.qualification_map_execution import _execute_reference_map

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class _KeyedFacts(Mapping[_Key, _Value]):
    def __init__(self, facts: Mapping[_Key, _Value]) -> None:
        self.facts = facts

    def __getitem__(self, key: _Key) -> _Value:
        return self.facts[key]

    def __iter__(self) -> Iterator[_Key]:
        raise AssertionError("inherited roles must not enumerate unrelated facts")

    def __len__(self) -> int:
        return len(self.facts)


def _result() -> ExecutionResult:
    _, result, _ = asyncio.run(_execute_reference_map(2))
    return result


def test_real_map_parent_roles_use_canonical_identity_without_global_scans() -> None:
    result = _result()
    parent = result.final_outputs[0].producer
    assert isinstance(parent, OperationOutputKey)
    provenance = {fact.key: fact for fact in result.provenance}
    ports = {(fact.activation, fact.port): fact for fact in result.ports}
    assert ports[parent.activation, parent.port].role == "candidate"
    assert _inherited_artifact_role(parent, _KeyedFacts(provenance), _KeyedFacts(ports)) == "candidate"
    member = next(fact.key for fact in result.provenance if isinstance(fact.key, MapItemKey))
    assert _inherited_artifact_role(member, _KeyedFacts(provenance), _KeyedFacts(ports)) == "artifact"
    assert _inherited_artifact_role(None, _KeyedFacts(provenance), _KeyedFacts(ports)) == "artifact"


@pytest.mark.parametrize("coordinate", ["target", "activation", "port", "producer"])
def test_keyed_role_lookup_rejects_a_foreign_retained_coordinate(coordinate: str) -> None:
    result = _result()
    parent = result.final_outputs[0].producer
    assert isinstance(parent, OperationOutputKey)
    provenance = {fact.key: fact for fact in result.provenance}
    ports = {(fact.activation, fact.port): fact for fact in result.ports}
    port = ports[parent.activation, parent.port]
    other = next(fact for fact in result.ports if fact.activation != parent.activation)
    foreign = copy(port)
    if coordinate == "target":
        # Another valid datum identity cannot claim this canonical producer.
        object.__setattr__(foreign, "target", _result().final_outputs[0].target)
    elif coordinate == "activation":
        object.__setattr__(foreign, "activation", other.activation)
    elif coordinate == "port":
        object.__setattr__(foreign, "port", "foreign")
    else:
        producer = copy(provenance[parent])
        object.__setattr__(producer, "key", next(key for key in provenance if key != parent))
        provenance[parent] = producer
        foreign = port
    ports[parent.activation, parent.port] = foreign
    assert _inherited_artifact_role(parent, provenance, ports) == "artifact"


def test_role_lookup_uses_staged_facts_without_retaining_a_discarded_delta() -> None:
    result = _result()
    parent = result.final_outputs[0].producer
    assert isinstance(parent, OperationOutputKey)
    facts = _ExecutionFacts()
    staged = facts.delta()
    producer = next(fact for fact in result.provenance if fact.key == parent)
    port = next(fact for fact in result.ports if fact.activation == parent.activation and fact.port == parent.port)
    staged.add_provenance(producer)
    staged.ports[parent.activation, parent.port] = port
    assert _inherited_artifact_role(parent, staged.provenance, staged.ports) == "candidate"
    assert _inherited_artifact_role(parent, facts.provenance, facts.ports) == "artifact"
    accepted = facts.delta()
    accepted.add_provenance(producer)
    accepted.ports[parent.activation, parent.port] = port
    facts.publish(accepted)
    assert _inherited_artifact_role(parent, facts.provenance, facts.ports) == "candidate"
