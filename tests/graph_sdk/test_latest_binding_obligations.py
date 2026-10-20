# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Maintained latest-binding obligations, compared without allocator ordinals."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.context import BindingResult
from anonymizer.engine.graph_sdk.executor import (
    BoundInputKey,
    ExecutionResult,
    OperationOutputKey,
    ProvenanceKey,
    RootInputKey,
)
from anonymizer.engine.graph_sdk.requests import BindingDeclarationId, TextArtifactValue
from anonymizer.graph import ArtifactRef, NodeId
from tests.graph_sdk.effects_provider_fixtures import _CaseProvider
from tests.graph_sdk.effects_request_fixtures import _normalize, _policy
from tests.graph_sdk.latest_binding_execution import _execute_case
from tests.graph_sdk.reference.corpora import load_cases

CASES = [
    case
    for case in load_cases("effects")
    if case["case_id"]
    in {
        "materialization/latest_bound_with_optional_omission",
        "materialization/latest_publication_rollback_then_reuse",
    }
]


def _binding(case: dict[str, Any], bound: BindingResult, provider: _CaseProvider) -> dict[BindingDeclarationId, str]:
    expected = case["expected"]["state"]
    receipt = bound.receipt
    assert bound.context is not None and bound.context.receipt is receipt
    names = {
        fact.identity: spec["declaration"]
        for fact, spec in zip(receipt.sources, case["declaration"]["materializations"], strict=True)
    }
    assert receipt.terminal == expected["binding_terminal"]
    assert {names[f.identity]: f.terminal for f in receipt.sources} == expected["binding_sources"]
    assert [
        dict(
            association=names[item.reference.declaration],
            source=item.source.name,
            key=item.reference.key,
            version=item.reference.version,
            text=item.text,
        )
        for item in receipt.artifacts
    ] == expected["binding_artifacts"]
    policies = {
        "P0": next(iter(next(iter(receipt.requests.bindings)).policies)),
        "P1": _policy(case["declaration"]["policies"]["P1"]),
    }
    projection = _normalize(
        receipt.requests,
        {name: association for association, name in provider.association_names.items()},
        {name: request for request, name in provider.request_names.items()},
        policies,
    )
    for terminal in receipt.requests.terminals:
        if terminal.category == "success":
            for returned in terminal.results:
                cast(dict[str, object], projection["association_terminals"])[
                    provider.association_names[returned.association]
                ] = {"request": provider.request_names[terminal.request], "outcome": returned.outcome, "policy": "P0"}
    for key, value in projection.items():
        assert value == expected[key], key
    assert provider.closes == 1 and not provider.events
    (cleanup,), (association,) = receipt.cleanup, receipt.cleanup_associations
    assert cleanup.resource == association.resource
    assert cleanup.owner == "sdk" and cleanup.disposition == "closed"
    assert association.purpose == "accounting"
    assert association.targets == frozenset({receipt.sources[0].declaration.target})
    assert set(expected["binding_cleanup"].values()) == {cleanup.disposition}
    return names


def _publication(
    case: dict[str, Any],
    result: ExecutionResult,
    bound: BindingResult,
    nodes: dict[str, NodeId],
    names: dict[BindingDeclarationId, str],
) -> None:
    expected = case["expected"]["state"]["materialization"]
    entries = {entry.template: entry for entry in result.states[0].entries}
    target = bound.receipt.sources[0].declaration.target
    node_names = {node: name for name, node in nodes.items()}
    activations = {
        entries[nodes[spec["node"]]].activation: spec["activation"]
        for spec in case["declaration"]["operation_occurrences"]
    }

    def producer(key: ProvenanceKey) -> str:
        assert key.target == target
        if isinstance(key, RootInputKey):
            return f"RootInputKey:T0:{key.port}"
        if isinstance(key, BoundInputKey):
            binding = key.binding_artifact
            return f"BoundInputKey:T0:{node_names[key.node]}:{key.port}:{names[binding.declaration]}:{binding.key}:{binding.version}"
        assert isinstance(key, OperationOutputKey)
        node = next(entry.template for entry in entries.values() if entry.activation == key.activation)
        return (
            f"OperationOutputKey:{activations[key.activation]}:T0:{node_names[node]}:{entries[node].outcome}:{key.port}"
        )

    values: dict[ArtifactRef, TextArtifactValue] = {}
    for reference, value in result.artifacts:
        assert isinstance(value, TextArtifactValue)
        values[reference] = value
    assert len(values) == len(result.artifacts) == len(result.provenance) == expected["artifact_count"]
    assert result.record.artifacts == frozenset(values) == frozenset(f.artifact for f in result.provenance)
    actual = {producer(f.key): f for f in result.provenance}
    assert set(actual) == set(expected["provenance"])
    # Match by producer identity, then compare versions and joins. Physical key
    # ordinals and reuse after rollback are intentionally not contractual.
    aliases = {fact.artifact: expected["provenance"][name]["artifact"] for name, fact in actual.items()}
    assert len(set(aliases.values())) == len(aliases)
    for left, left_name in aliases.items():
        for right, right_name in aliases.items():
            assert (left.key == right.key) == (left_name.rsplit(":", 1)[0] == right_name.rsplit(":", 1)[0])
    expected_values = {item["identity"]: item for item in case["expected"]["state"]["artifacts"]}
    lineages = {}
    for name, fact in actual.items():
        assert fact.artifact.invocation == result.record.invocation
        assert fact.artifact.version == int(aliases[fact.artifact].rsplit(":", 1)[1])
        assert sorted(producer(parent) for parent in fact.parents) == expected["provenance"][name]["parents"]
        value = values[fact.artifact]
        assert isinstance(value, TextArtifactValue)
        assert value.text == expected_values[aliases[fact.artifact]]["text"]
        if isinstance(fact.key, BoundInputKey):
            key = fact.key.binding_artifact
            assert fact.key.node == nodes["N0"] and fact.key.port == "context"
            assert fact.artifact.version == key.version
            lineages.setdefault((key.declaration, key.key), set()).add(fact.artifact.key)
            (source,) = [item for item in bound.receipt.artifacts if item.reference == key]
            assert source.text == value.text
    assert lineages and all(len(keys) == 1 for keys in lineages.values())
    assert sum(len(value.text.encode()) for value in values.values()) == expected["artifact_bytes"]
    assert sum(len(f.parents) for f in result.provenance) == expected["provenance_edges"]
    ports, parents = {}, set()
    for port in result.ports:
        assert port.target == target and port.activation == entries[port.node].activation
        assert port.artifact_type == bound.receipt.sources[0].declaration.artifact_type
        if port.port == "result":
            assert port.role == "candidate"
            continue
        assert port.role == "artifact"
        (fact,) = [fact for fact in result.provenance if fact.artifact == port.artifact]
        parents.add((target, port.activation, port.port, fact.key))
        name = f"root:T0:{node_names[port.node]}:{port.port}"
        if isinstance(fact.key, BoundInputKey):
            name = f"initial:T0:{node_names[port.node]}:{port.port}:{names[fact.key.binding_artifact.declaration]}"
            assert port.artifact.version == max(item.reference.version for item in bound.receipt.artifacts)
        ports[name] = dict(
            artifact_type=port.artifact_type.name, key=aliases[port.artifact], value=values[port.artifact].text
        )
    assert ports == expected["ports"]
    assert len(result._input_parents) == len(parents) and set(result._input_parents) == parents
    assert not result._passthrough_parents and len(result.ports) == len(ports) + 1
    (final,) = result.final_outputs
    final_node = nodes["N1" if case["declaration"].get("root_publications") else "N0"]
    assert final.target == target and final.port == "result" and final.outcome == "ok"
    assert isinstance(final.producer, OperationOutputKey)
    assert final.producer.activation == entries[final_node].activation
    assert final.producer.port == "result" and final.producer.target == target
    assert actual[producer(final.producer)].artifact == final.candidate.artifact
    if "final_outputs" in case["expected"]["state"]:
        assert [aliases[final.candidate.artifact]] == case["expected"]["state"]["final_outputs"]
    (output,) = [port for port in result.ports if port.port == "result"]
    assert output.node == final_node and output.artifact == final.candidate.artifact


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["case_id"])
def test_latest_binding_obligation(case: dict[str, Any]) -> None:
    result, bound, nodes, callbacks, provider = asyncio.run(_execute_case(case))
    rollback = bool(case["declaration"].get("root_publications"))
    assert [callback.calls for callback in callbacks] == [1, int(rollback)]
    names = _binding(case, bound, provider)
    _publication(case, result, bound, nodes, names)
    entries = {entry.template: entry for entry in result.states[0].entries}
    assert result.states[0].complete and len(result.record.terminals) == 2
    assert result.record.targets == frozenset({bound.receipt.sources[0].declaration.target})
    for spec in case["declaration"]["operation_occurrences"]:
        entry = entries[nodes[spec["node"]]]
        (terminal,) = [fact for fact in result.record.terminals if fact.activation == entry.activation]
        expected = case["expected"]["state"]["operation_occurrences"][f"{spec['activation']}:{spec['attempt']}"]
        assert terminal.category == entry.status == expected["terminal"]["category"]
        assert (
            terminal.reasons == frozenset({expected["terminal"]["reason"]})
            if "reason" in expected["terminal"]
            else not terminal.reasons
        )
        assert not terminal.structural
        assert (terminal.attempt is not None) == (spec["node"] == "N0" or rollback)
        if terminal.attempt is not None:
            assert terminal.attempt.activation == entry.activation
        assert (
            sorted(
                port.port for port in result.ports if port.activation == entry.activation and port.role == "candidate"
            )
            == expected["published_ports"]
        )
    assert len(result.publication_failures) == int(rollback)
    if rollback:
        (failure,) = result.publication_failures
        assert failure.association.task.activation == entries[nodes["N0"]].activation
        assert failure.condition == "artifact_limit_exhausted" and failure.failure is None
        assert failure.association == callbacks[0].associations[0].association
        assert failure.target == result.final_outputs[0].target
        assert failure.association != callbacks[1].associations[0].association
        assert failure.association.task == next(
            t.attempt for t in result.record.terminals if t.activation == failure.association.task.activation
        )
    assert not result.cleanup and not result.cleanup_associations
    assert not result.assessments
    assert result.requests.dispatched_count == 0
    assert not result.requests.dispatches and not result.requests.terminals and not result.requests.bindings
    assert (
        not result.requests.settlements
        and not result.requests.local_in_flight
        and not result.requests.remote_outstanding
    )
    assert not result.requests.defects
