# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained metrics and unsupported binding schemas fail at their owning crossings."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from unittest.mock import Mock

import pytest

from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE, EntitySpansValue
from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape
from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk._ingress import inspect_artifact
from anonymizer.engine.graph_sdk.executor import ExecutionResult, IngressLimits
from anonymizer.engine.graph_sdk.inspection import InspectionSource, bind_inspection_source, inspect
from anonymizer.engine.graph_sdk.requests import BindingId, TextArtifactValue, TextCollectionValue
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_effects_materialization_schema import (
    _bind_schema_fixture,
    _build_schema_fixture,
    _execute_distinct_sites,
    _prepare_schema_fixture,
)
from tests.graph_sdk.test_observations import inspection_limits
from tests.graph_sdk.typed_artifact_fixtures import SOURCE, execute_spans


def _inspection_source(result: ExecutionResult) -> InspectionSource:
    execution = result._execution
    return bind_inspection_source(
        prepared=execution.context.prepared,
        context=execution.context,
        execution_plan=execution,
        execution=result,
        limits=inspection_limits(max_count=10000),
    )


def _assert_rejected_at_retained_crossings(result: ExecutionResult, source: InspectionSource) -> None:
    with pytest.raises(EffectRejected):
        _inputs(result._execution, result)
    with pytest.raises(EffectRejected):
        _inspection_source(result)
    # Recheck at consumption too: the source may have been bound before mutation.
    with pytest.raises(EffectRejected):
        inspect(source=source)


@pytest.mark.parametrize("kind", [TextArtifactValue, EntitySpansValue])
@pytest.mark.parametrize("dimension", ["content_bytes", "items"])
@pytest.mark.parametrize("delta", [-1, 1])
def test_forged_cached_dimensions_rejected_on_reentry_and_inspection(kind: type, dimension: str, delta: int) -> None:
    _, result, _ = asyncio.run(execute_spans())
    source = _inspection_source(result)
    resources = inspect(source=source).resources
    assert resources is not None and resources.artifact_bytes == 98
    value = next(value for _, value in result.artifacts if type(value) is kind)
    object.__setattr__(value, "shape", replace(value.shape, **{dimension: getattr(value.shape, dimension) + delta}))
    _assert_rejected_at_retained_crossings(result, source)


@pytest.mark.parametrize("child", [False, True])
def test_collection_and_nested_text_metrics_revalidated(child: bool) -> None:
    async def run() -> ExecutionResult:
        case = deepcopy(
            next(c for c in load_cases("effects") if c["case_id"] == "materialization/same_port_distinct_nodes")
        )
        fixture = _build_schema_fixture(case)
        bound = await _bind_schema_fixture(fixture)
        return await _execute_distinct_sites(fixture, bound, _prepare_schema_fixture(fixture))

    result = asyncio.run(run())
    source = _inspection_source(result)
    _inputs(result._execution, result)
    value = next(value for _, value in result.artifacts if isinstance(value, TextCollectionValue))
    target = value.items[0].value if child else value
    object.__setattr__(target, "shape", ArtifactShape(0, 0))
    _assert_rejected_at_retained_crossings(result, source)


def test_valid_root_larger_than_callback_budget_remains_inspectable() -> None:
    # Root data uses its own 100-byte budget. Rendering is correctly denied by the
    # smaller callback budget, leaving a valid result with the larger root retained.
    execution, result, _ = asyncio.run(
        execute_spans(source_text=SOURCE + " x" * 40, ingress=replace(IngressLimits(), max_text_bytes=45))
    )
    assert not result.final_outputs and len(result.artifacts) == 3
    _inputs(execution, result)
    resources = inspect(source=_inspection_source(result)).resources
    assert resources is not None and resources.artifact_bytes == 163


def test_bound_context_larger_than_callback_budget_remains_inspectable() -> None:
    async def run() -> ExecutionResult:
        case = deepcopy(
            next(c for c in load_cases("effects") if c["case_id"] == "materialization/same_port_distinct_nodes")
        )
        fixture = _build_schema_fixture(case)
        for provider in fixture.providers.values():
            provider.values = ("aaa", "bbb")
        bound = await _bind_schema_fixture(fixture)
        return await _execute_distinct_sites(
            fixture,
            bound,
            _prepare_schema_fixture(fixture),
            ingress_limits=replace(IngressLimits(), max_text_bytes=1),
            expected_texts=("aaa", "bbb"),
        )

    result = asyncio.run(run())
    _inputs(result._execution, result)
    resources = inspect(source=_inspection_source(result)).resources
    assert resources is not None and resources.artifact_bytes == 12


@pytest.mark.parametrize(
    "dimension,value", [("content_bytes", True), ("items", True), ("content_bytes", -1), ("items", -1)]
)
def test_shape_consistency_requires_exact_nonnegative_counts(dimension: str, value: int) -> None:
    text = TextArtifactValue(text="a")
    object.__setattr__(text, "shape", replace(text.shape, **{dimension: value}))
    assert not inspect_artifact(text)


@pytest.mark.parametrize("field", ["item_type", "output_type", "both"])
@pytest.mark.parametrize("revision", [1, 2])
def test_reserved_span_binding_rejected_before_identity_resources_and_provider(
    field: str, revision: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = deepcopy(
        next(c for c in load_cases("effects") if c["case_id"] == "materialization/same_port_distinct_nodes")
    )
    for row in case["declaration"]["materializations"]:
        if field == "both":
            row.update(kind="single", max_items=1, item_type=ENTITY_SPANS_TYPE.name, output_type=ENTITY_SPANS_TYPE.name)
        else:
            row[field] = ENTITY_SPANS_TYPE.name
    fixture = _build_schema_fixture(case, revision=revision)
    identity = Mock(side_effect=AssertionError("binding identity must not be allocated"))
    acquire = Mock(side_effect=AssertionError("resources must not be acquired"))
    monkeypatch.setattr(BindingId, "new", identity)
    monkeypatch.setattr("anonymizer.engine.graph_sdk.binding._acquire_resources", acquire)
    with pytest.raises(EffectRejected) as error:
        asyncio.run(_bind_schema_fixture(fixture))
    assert error.value.code is EffectCode.UNSUPPORTED
    identity.assert_not_called()
    acquire.assert_not_called()
    assert all(provider.calls == 0 for provider in fixture.providers.values())


def test_arbitrary_text_type_names_remain_supported_for_initial_binding() -> None:
    case = deepcopy(
        next(c for c in load_cases("effects") if c["case_id"] == "materialization/same_port_distinct_nodes")
    )
    for row in case["declaration"]["materializations"]:
        row.update(item_type="custom.text", output_type="custom.text-collection")
    fixture = _build_schema_fixture(case, revision=3)
    bound = asyncio.run(_bind_schema_fixture(fixture))
    assert bound.context is not None and len(bound.context.artifacts) == 2
    assert [provider.calls for provider in fixture.providers.values()] == [2]
