# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed candidate spans retain source ownership across graph processing."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE, EntitySpansValue
from anonymizer.engine.detection.postprocess import EntitySpan
from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape
from anonymizer.engine.graph_sdk._ingress import inspect_artifact
from anonymizer.engine.graph_sdk.executor import ExecutionResult, IngressLimits, start_execution
from anonymizer.engine.graph_sdk.requests import TextArtifactValue, TextCollectionItem, TextCollectionValue
from anonymizer.graph import EffectCode, EffectRejected
from anonymizer.graph._values import ArtifactRef, InvocationId, PlanId
from anonymizer.graph.workflow import ArtifactType, NodeId
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.typed_artifact_fixtures import ENTITIES, SOURCE, execute_spans, span_graph


def test_text_spans_filter_and_candidate_text_use_real_graph() -> None:
    execution, result, nodes = asyncio.run(execute_spans())
    assert [terminal.category for terminal in result.record.terminals] == ["success"] * 3
    values = dict(result.artifacts)
    spans = [value for value in values.values() if isinstance(value, EntitySpansValue)]
    expected = (
        EntitySpan("a", "Alice", "first_name", 0, 5, 0.9, "detector"),
        EntitySpan("b", "Bob", "last_name", 10, 13, 0.8, "detector"),
    )
    assert [value.entities for value in spans] == [expected, expected[:1]]
    assert all(values[value.source] == TextArtifactValue(text=SOURCE) for value in spans)
    final = values[result.final_outputs[0].candidate.artifact]
    assert final == TextArtifactValue(text="[name] met Bob.")
    assert {p.node for p in result.ports} >= set(nodes)
    assert ENTITIES[0].value == "Alice" and ENTITIES[1].value == "Bob"
    assert all(value.entities[0] is not ENTITIES[0] for value in spans)
    # Saved-result re-entry validates payloads as well as occurrence ownership.
    _inputs(execution, result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_position", True),
        ("start_position", -1),
        ("end_position", 100),
        ("end_position", 3),
        ("value", "wrong"),
        ("score", float("nan")),
        ("score", float("inf")),
        ("score", 1),
        ("label", []),
        ("value", "\ud800"),
    ],
)
def test_malformed_candidate_never_publishes(field: str, value: object) -> None:
    def corrupt(payload: EntitySpansValue) -> EntitySpansValue:
        object.__setattr__(payload.entities[0], field, value)
        return payload

    _, result, _ = asyncio.run(execute_spans(mutate=corrupt))
    assert not result.final_outputs
    assert len(result.artifacts) == 1
    assert not any(isinstance(v, EntitySpansValue) for _, v in result.artifacts)
    assert ENTITIES[0] == EntitySpan("a", "Alice", "first_name", 0, 5, 0.9, "detector")


@pytest.mark.parametrize("foreign", [False, True])
def test_source_reference_must_identify_the_consumed_text(foreign: bool) -> None:
    def corrupt(value: EntitySpansValue) -> EntitySpansValue:
        source = value.source
        return replace(
            value,
            source=replace(
                source,
                invocation=InvocationId.new(plan=PlanId.new()) if foreign else source.invocation,
                key=source.key if foreign else source.key + 100,
            ),
        )

    _, result, _ = asyncio.run(execute_spans(mutate=corrupt))
    assert not result.final_outputs and len(result.artifacts) == 1


@pytest.mark.parametrize("field,exact", [("max_text_bytes", 45), ("max_collection_items", 2)])
def test_domain_ingress_exact_and_one_under(field: str, exact: int) -> None:
    for maximum, success in ((exact, True), (exact - 1, False)):
        _, result, _ = asyncio.run(execute_spans(ingress=replace(IngressLimits(), **{field: maximum})))
        assert bool(result.final_outputs) is success
        if not success:
            assert len(result.artifacts) == 1


def test_forged_size_is_rejected_and_size_remains_content_only() -> None:
    def corrupt(value: EntitySpansValue) -> EntitySpansValue:
        object.__setattr__(value, "shape", ArtifactShape(0, 0))
        return value

    _, result, _ = asyncio.run(execute_spans(mutate=corrupt))
    assert not result.final_outputs
    source = ArtifactRef(invocation=InvocationId.new(plan=PlanId.new()), key=0, version=1)
    value = EntitySpansValue(source=source, entities=ENTITIES)
    assert value.shape == ArtifactShape(content_bytes=45, items=2)
    assert inspect_artifact(value, IngressLimits())
    object.__setattr__(value.entities[0], "score", float("nan"))
    assert not inspect_artifact(value, IngressLimits())


def test_runtime_capacity_failure_keeps_prior_publication_only() -> None:
    async def run() -> tuple[ExecutionResult, tuple[NodeId, ...]]:
        admitted, capabilities, services, nodes = span_graph()
        services = replace(services, limits=replace(services.limits, max_runtime_artifacts=2))
        result = await (await start_execution(admitted=admitted, capabilities=capabilities, services=services)).wait()
        return result, nodes

    result, nodes = asyncio.run(run())
    assert not result.final_outputs
    assert len(result.artifacts) == 2
    assert {p.node for p in result.ports if p.port == "value"} == {nodes[0]}
    assert len(result.publication_failures) == 1
    assert all(fact.artifact in dict(result.artifacts) for fact in result.provenance)


def test_constructor_rejects_mutable_or_open_span_shapes() -> None:
    source = ArtifactRef(invocation=InvocationId.new(plan=PlanId.new()), key=0, version=1)
    with pytest.raises(EffectRejected):
        EntitySpansValue(source=source, entities=cast(tuple[EntitySpan, ...], list(ENTITIES)))
    assert ENTITY_SPANS_TYPE.name == "anonymizer.candidate-spans" and ENTITY_SPANS_TYPE.revision == 1


def test_identity_alias_preserves_artifact_and_producing_occurrences() -> None:
    execution, result, nodes = asyncio.run(execute_spans(alias_filter=True))
    assert result.final_outputs
    ports = [p for p in result.ports if p.port == "value" and p.node in nodes[:2]]
    assert len(ports) == 2
    assert ports[0].artifact == ports[1].artifact
    assert ports[0].node != ports[1].node
    assert dict(result.artifacts)[result.final_outputs[0].candidate.artifact] == TextArtifactValue(
        text="[name] met Bob."
    )
    _inputs(execution, result)


def test_nested_collection_cannot_hide_forged_child_metrics() -> None:
    child = TextArtifactValue(text="Alice")
    collection = TextCollectionValue(items=(TextCollectionItem(key=0, version=1, value=child),))
    object.__setattr__(child, "shape", ArtifactShape(0))
    assert not inspect_artifact(collection, IngressLimits())


def test_unqualified_candidate_revision_is_rejected_before_execution() -> None:
    with pytest.raises(EffectRejected) as error:
        span_graph(span_type=ArtifactType(name=ENTITY_SPANS_TYPE.name, revision=2))
    assert error.value.code is EffectCode.UNSUPPORTED


def test_candidate_spans_cannot_be_bound_as_root_text() -> None:
    with pytest.raises(EffectRejected) as error:
        span_graph(source_type=ENTITY_SPANS_TYPE)
    assert error.value.code is EffectCode.UNSUPPORTED


def test_candidate_spans_are_content_private_and_not_text_collection_members() -> None:
    source = ArtifactRef(invocation=InvocationId.new(plan=PlanId.new()), key=0, version=1)
    value = EntitySpansValue(source=source, entities=ENTITIES)
    assert "Alice" not in repr(value) and "first_name" not in repr(value)
    with pytest.raises(EffectRejected):
        TextCollectionItem(key=0, version=1, value=cast(TextArtifactValue, value))


@pytest.mark.parametrize("maximum,success", [(98, True), (97, False)])
def test_exact_runtime_bytes_include_span_content(maximum: int, success: bool) -> None:
    async def run() -> ExecutionResult:
        admitted, capabilities, services, _ = span_graph()
        services = replace(services, limits=replace(services.limits, max_runtime_artifact_bytes=maximum))
        return await (await start_execution(admitted=admitted, capabilities=capabilities, services=services)).wait()

    result = asyncio.run(run())
    # Source 14 + detected 45 + filtered 24 + rendered 15 = 98 content bytes.
    assert bool(result.final_outputs) is success
    if not success:
        assert len(result.artifacts) == 3
        assert len(result.publication_failures) == 1
