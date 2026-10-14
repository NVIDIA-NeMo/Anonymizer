# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed callback boundaries, aggregate budgets and retained payload ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from dataclasses import fields, replace
from typing import cast, get_args

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk._ingress import inspect_associations, inspect_source
from anonymizer.engine.graph_sdk.context import ContextSourceRef, SourceItem, SourceResponse
from anonymizer.engine.graph_sdk.executor import AssessmentFinding, IngressLimits, LocalAssessmentResult
from anonymizer.engine.graph_sdk.records import ReasonCode, TerminalFact
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    InvocationRequestScope,
    PhysicalRequestId,
    PortArtifact,
    RequestAssociationSummary,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
    TransportResult,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph._values import InvocationId, PlanId
from anonymizer.graph.workflow import ArtifactType
from tests.graph_sdk import evidence_fixtures
from tests.graph_sdk.qualification_fixtures import _RecoveringTransport
from tests.graph_sdk.test_requests import _association


@pytest.mark.parametrize("field", [item.name for item in fields(IngressLimits)])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_ingress_limits_require_positive_exact_integers(field: str, value: object) -> None:
    with pytest.raises(EffectRejected):
        IngressLimits(**{field: cast(int, value)})


def _rows(*, collection: bool = False) -> tuple[AssociationResult, ...]:
    invocation = InvocationId.new(plan=PlanId.new())
    value = TextArtifactValue(text="é")
    return tuple(
        AssociationResult(
            association=_association(invocation, index),
            outcome="ok",
            consumed_context_ports=frozenset(),
            outputs=(
                PortArtifact(
                    port="p",
                    artifact_type=ArtifactType(name="text", revision=1),
                    artifact=None,
                    value=TextCollectionValue(items=(TextCollectionItem(key=0, version=1, value=value),))
                    if collection
                    else value,
                ),
            ),
        )
        for index in range(2)
    )


@pytest.mark.parametrize(
    "field,exact", [("max_text_bytes", 4), ("max_metadata_bytes", 14), ("max_metadata_items", 8), ("max_depth", 7)]
)
def test_scalar_response_exact_and_one_over_aggregate(field: str, exact: int) -> None:
    rows = _rows()
    expected = frozenset(row.association for row in rows)
    assert inspect_associations(rows, expected, IngressLimits(**{field: exact})) == frozenset()
    assert inspect_associations(rows, expected, IngressLimits(**{field: exact - 1})) == expected


def test_collection_count_is_aggregate_and_counts_repeated_values() -> None:
    rows = _rows(collection=True)
    expected = frozenset(row.association for row in rows)
    assert inspect_associations(rows, expected, IngressLimits(max_collection_items=2, max_depth=10)) == frozenset()
    assert inspect_associations(rows, expected, IngressLimits(max_collection_items=1)) == expected
    assert inspect_associations(rows, expected, IngressLimits(max_depth=9)) == expected


def test_invalid_correlated_value_and_per_member_output_bound_are_local() -> None:
    rows = _rows()
    expected = frozenset(row.association for row in rows)
    first = rows[0]
    malformed = replace(first, outputs=(replace(first.outputs[0], value=TextArtifactValue(text=chr(0xD800))),))
    assert inspect_associations((malformed, rows[1]), expected, IngressLimits()) == frozenset({first.association})
    repeated = replace(first, outputs=(first.outputs[0], first.outputs[0]))
    assert inspect_associations(
        (repeated, rows[1]), expected, IngressLimits(max_outputs_per_association=1)
    ) == frozenset({first.association})


@pytest.mark.parametrize("mode", ["rows", "member", "outputs", "surrogate", "nested", "settlement", "usage"])
def test_external_foreign_values_preserve_accounting_and_safe_cleanup(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    original = _RecoveringTransport.dispatch

    async def dispatch(self: _RecoveringTransport, request: DispatchEnvelope) -> TransportResult:
        result = await original(self, request)
        assert isinstance(result, TransportSuccess)
        if mode == "rows":
            object.__setattr__(result, "results", None)
        elif mode == "member":
            object.__setattr__(result, "results", (object(),))
        elif mode == "outputs":
            object.__setattr__(result.results[0], "outputs", None)
        elif mode == "surrogate":
            row = result.results[0]
            result = replace(
                result,
                results=(replace(row, outputs=(replace(row.outputs[0], value=TextArtifactValue(text=chr(0xD800))),)),),
            )
        elif mode == "nested":
            item = TextCollectionItem(key=0, version=1, value=TextArtifactValue(text="x"))
            object.__setattr__(item, "value", TextCollectionValue(items=()))
            row = result.results[0]
            result = replace(
                result,
                results=(replace(row, outputs=(replace(row.outputs[0], value=TextCollectionValue(items=(item,))),)),),
            )
        elif mode == "settlement":
            object.__setattr__(result, "settlement", object())
        else:
            assert result.settlement is not None
            object.__setattr__(result.settlement, "usage", object())
        return result

    monkeypatch.setattr(_RecoveringTransport, "dispatch", dispatch)
    transport = _RecoveringTransport()
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    _, result = asyncio.run(evidence_fixtures._execute_assessment(external=(transport, lease)))
    assert result.requests.dispatched_count == transport.calls > 0
    assert len(result.requests.terminals) == transport.calls
    assert len(result.requests.settlements) == transport.calls
    assert not result.requests.local_in_flight
    if mode in {"settlement", "usage"}:
        assert result.requests.remote_outstanding
        assert all(isinstance(item.usage, UnknownUsage) for item in result.requests.settlements)
        assert transport.closes == 0
        assert any(item.code == "invalid_settlement" for item in result.requests.defects)
    else:
        assert not result.requests.remote_outstanding
        assert transport.closes == 1
    for terminal in result.requests.terminals:
        assert terminal.schema_version == 2
        for summary in terminal.results:
            assert type(summary) is RequestAssociationSummary
            assert summary.category == "failure" and not summary.artifacts
            assert not hasattr(summary, "outputs")


@pytest.mark.parametrize("capacity", [1, 2, 3, 4, 5, 6])
def test_malformed_utf8_aggregate_exhaustion_is_independent_of_row_order(capacity: int) -> None:
    rows = _rows()
    malformed = replace(rows[0], outputs=(replace(rows[0].outputs[0], value=TextArtifactValue(text=chr(0xD800))),))
    expected = frozenset(row.association for row in rows)
    limits = IngressLimits(max_text_bytes=capacity)
    forward = inspect_associations((malformed, rows[1]), expected, limits)
    reverse = inspect_associations((rows[1], malformed), expected, limits)
    assert forward == reverse == (expected if capacity < 5 else frozenset({rows[0].association}))


def test_association_and_assessment_bounds_precede_payload_traversal() -> None:
    rows = _rows()
    expected = frozenset(row.association for row in rows)
    assert inspect_associations(rows, expected, IngressLimits(max_associations=2)) == frozenset()
    assert inspect_associations(rows, expected, IngressLimits(max_associations=1)) is None
    assessments = tuple(
        LocalAssessmentResult(
            association=cast(SemanticAssociation, row.association),
            promise="p",
            evidence_port="e",
            finding=AssessmentFinding(status="satisfied", code="ok"),
        )
        for row in rows
    )
    assert (
        inspect_associations(rows, expected, IngressLimits(max_assessments=2), assessments=assessments) == frozenset()
    )
    assert inspect_associations(rows, expected, IngressLimits(max_assessments=1), assessments=assessments) == expected


class _ForeignTuple(tuple[object, ...]):
    def __iter__(self) -> Iterator[object]:
        raise AssertionError("foreign iteration")

    def __len__(self) -> int:
        raise AssertionError("foreign length")


@pytest.mark.parametrize("field", ["envelope", "outputs", "assessments"])
def test_ingress_rejects_custom_container_without_hooks(field: str) -> None:
    rows = _rows()
    expected = frozenset(row.association for row in rows)
    foreign = _ForeignTuple(())
    if field == "envelope":
        assert inspect_associations(foreign, expected, IngressLimits()) is None
    elif field == "outputs":
        object.__setattr__(rows[0], "outputs", foreign)
        assert inspect_associations(rows, expected, IngressLimits()) == frozenset({rows[0].association})
    else:
        assert inspect_associations(rows, expected, IngressLimits(), assessments=foreign) == expected


@pytest.mark.parametrize("kind", ["outputs", "collection", "metadata"])
@pytest.mark.parametrize("reverse", [False, True])
def test_malformed_sibling_cannot_hide_aggregate_payload(kind: str, reverse: bool) -> None:
    rows = _rows()
    bad = replace(rows[0].outputs[0], value=TextArtifactValue(text=chr(0xD800)))
    large = replace(rows[0].outputs[0], value=TextArtifactValue(text="x" * 100))
    if kind == "outputs":
        outputs = (bad, large)
    elif kind == "collection":
        outputs = (
            replace(
                large,
                value=TextCollectionValue(
                    items=(
                        TextCollectionItem(key=0, version=1, value=TextArtifactValue(text=chr(0xD800))),
                        TextCollectionItem(key=1, version=1, value=TextArtifactValue(text="x" * 100)),
                    )
                ),
            ),
        )
    else:
        outputs = (replace(large, port=chr(0xD800)),)
    changed = (replace(rows[0], outputs=outputs), rows[1])
    expected = frozenset(row.association for row in rows)
    assert (
        inspect_associations(
            tuple(reversed(changed)) if reverse else changed, expected, IngressLimits(max_text_bytes=5)
        )
        == expected
    )


def test_source_and_assessment_depth_include_scalar_leaves() -> None:
    row = _rows()[0]
    association = cast(SemanticAssociation, row.association)
    request = PhysicalRequestId.new(scope=InvocationRequestScope(invocation=association.task.activation.invocation))
    source = SourceResponse(
        source=ContextSourceRef(name="test", revision=1),
        items=(SourceItem(association=association, key=0, version=1, text="x"),),
        settlement=ExternalSettlement(
            request=request,
            disposition="completed",
            usage=ExactUsage(input_units=1, output_units=0),
            remote_stopped=True,
        ),
    )
    assert inspect_source(source, association, IngressLimits(max_depth=4))
    assert not inspect_source(source, association, IngressLimits(max_depth=3))
    bare = replace(row, outputs=())
    assessments = (
        LocalAssessmentResult(
            association=association,
            promise="p",
            evidence_port="e",
            finding=AssessmentFinding(status="satisfied", code="ok"),
        ),
    )
    expected = frozenset({association})
    assert inspect_associations((bare,), expected, IngressLimits(max_depth=5), assessments=assessments) == frozenset()
    assert inspect_associations((bare,), expected, IngressLimits(max_depth=4), assessments=assessments) == expected


@pytest.mark.parametrize("reason", ["omitted_optional", "artifact_limit_exhausted"])
def test_accepted_terminal_reason_type_matches_constructor(reason: str) -> None:
    assert reason in get_args(ReasonCode)
    association = cast(SemanticAssociation, _rows()[0].association)
    fact = TerminalFact(
        activation=association.task.activation,
        attempt=None if reason == "omitted_optional" else association.task,
        category="blocked",
        reasons=frozenset({cast(ReasonCode, reason)}),
    )
    assert reason in fact.reasons


def test_large_finite_depth_never_uses_python_recursion_for_foreign_values() -> None:
    rows = _rows()
    value: TextArtifactValue | TextCollectionValue = TextArtifactValue(text="x")
    for index in range(1100):
        item = TextCollectionItem(key=index, version=1, value=TextArtifactValue(text="x"))
        object.__setattr__(item, "value", value)
        value = TextCollectionValue(items=(item,))
    changed = (replace(rows[0], outputs=(replace(rows[0].outputs[0], value=value),)), rows[1])
    expected = frozenset(row.association for row in rows)
    assert inspect_associations(changed, expected, IngressLimits(max_depth=10_000)) == frozenset({rows[0].association})
    assert inspect_associations(changed, expected, IngressLimits()) == expected


def test_unconfigured_nested_items_still_recount_and_preserve_explicit_caps():
    from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape
    from anonymizer.engine.graph_sdk._ingress import inspect_artifact
    from anonymizer.engine.rewrite.artifacts import QualityAnswer, QualityAnswersValue

    value = QualityAnswersValue(answers=tuple(QualityAnswer(index, "answer") for index in range(12)))
    unlimited = IngressLimits(max_collection_items=None, max_metadata_items=None)
    assert inspect_artifact(value, unlimited)
    assert not inspect_artifact(value, replace(unlimited, max_collection_items=11))
    assert inspect_artifact(value, replace(unlimited, max_collection_items=12))
    assert not inspect_artifact(value, replace(unlimited, max_metadata_items=11))
    object.__setattr__(value, "shape", ArtifactShape(content_bytes=value.shape.content_bytes, items=11))
    assert not inspect_artifact(value, unlimited)
