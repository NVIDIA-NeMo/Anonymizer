# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public equivalents of closed materialization boundary cases."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
    ContextMaterialization,
    ContextSourceRef,
    RetrievalBounds,
)
from anonymizer.engine.graph_sdk.requests import BindingDeclarationId, BindingId
from anonymizer.graph.workflow import ArtifactType, NodeId, WorkflowId
from tests.graph_sdk.test_context_source_execution import (
    test_initial_collection_schema_rejects_same_typed_caller_root_before_provider_effects as _assert_collection_root_rejection,
)
from tests.graph_sdk.test_effects_production_conformance import CORPUS, _assert_adaptive_materialization_case


def test_adaptive_binding_identity_is_not_a_public_declaration_field() -> None:
    corpus = json.loads(CORPUS.read_bytes())
    case = next(case for case in corpus if case["case_id"] == "materialization/adaptive_binding_identity")
    positive = next(case for case in corpus if case["case_id"] == "materialization/adaptive_collection_1")
    raw = case["declaration"]["materializations"][0]
    assert raw["declaration"] == "D0"
    assert {**raw, "declaration": None} == positive["declaration"]["materializations"][0]
    assert case["expected"] == {"status": "rejected", "code": "contradictory"}
    # There is no binding-identity slot in the production adaptive declaration.
    # The closed constructor rejects that reducer-only field before admission.
    with pytest.raises(TypeError, match="declaration"):
        cast(Any, AdaptiveRetrievalDecl)(
            node=NodeId.new(workflow=WorkflowId.new()),
            source=ContextSourceRef(name="source", revision=1),
            selector_ports=("input",),
            output_port=raw["port"],
            bounds=RetrievalBounds(max_items=raw["max_items"], max_bytes=raw["max_bytes"], max_requests=2),
            materialization=ContextMaterialization(
                kind=raw["kind"], item_type=ArtifactType(name=raw["item_type"], revision=1)
            ),
            declaration=BindingDeclarationId.new(binding=BindingId.new(), ordinal=0),
        )
    asyncio.run(_assert_adaptive_materialization_case(positive))


def test_known_collection_cannot_be_supplied_as_a_caller_root(monkeypatch: pytest.MonkeyPatch) -> None:
    case = next(
        case for case in json.loads(CORPUS.read_bytes()) if case["case_id"] == "materialization/collection_root_input"
    )
    raw = case["declaration"]["materializations"][0]
    assert raw["kind"] == "collection"
    assert raw["output_type"] in case["declaration"]["root_input_types"]
    assert raw["item_type"] != raw["output_type"]
    assert case["expected"] == {"status": "rejected", "code": "contradictory"}
    # The shared production fixture alpha-renames text/text_collection to
    # item/collection and proves zero BindingIds and provider factory calls.
    _assert_collection_root_rejection(monkeypatch)
