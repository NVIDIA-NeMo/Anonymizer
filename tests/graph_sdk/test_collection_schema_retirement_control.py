# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
    ContextMaterialization,
    RetrievalBounds,
    admit_context_plan,
)
from tests.graph_sdk.test_effects_materialization_schema import (
    _bind_schema_fixture,
    _build_schema_fixture,
    _prepare_schema_fixture,
)


def test_initial_and_adaptive_text_collection_schema_is_admitted() -> None:
    raw = json.loads(
        (Path(__file__).parent / "reference" / "retirement_witnesses" / "schema-positive-59.json").read_text()
    )
    assert (
        hashlib.sha256(
            json.dumps(
                {k: v for k, v in raw.items() if k != "hash"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        == raw["hash"]
    )
    assert raw["return"] == {"status": "accepted"}
    # This identifier selects the existing two-site initial/adaptive fixture topology;
    # its declaration and accepted historical answer remain exactly the captured bytes.
    case = {
        "case_id": "materialization/nested_collection_schema",
        "declaration": raw["before"]["declaration"],
        "expected": raw["return"],
    }
    fixture = _build_schema_fixture(case)
    bound = asyncio.run(_bind_schema_fixture(fixture))
    assert bound.context is not None
    prepared = _prepare_schema_fixture(fixture)
    initial, adaptive = case["declaration"]["materializations"]
    assert initial["item_type"] == adaptive["item_type"] == "text"
    assert initial["output_type"] == adaptive["output_type"] == "text_collection"
    assert case["declaration"]["root_input_types"] == ["text"]
    declaration = AdaptiveRetrievalDecl(
        node=fixture.nodes[1],
        source=fixture.source_caps[1].source,
        selector_ports=("input",),
        output_port="context",
        bounds=RetrievalBounds(
            max_items=adaptive["max_items"],
            max_bytes=adaptive["max_bytes"],
            max_requests=2,
        ),
        materialization=ContextMaterialization(kind=adaptive["kind"], item_type=fixture.types[adaptive["item_type"]]),
    )
    receipt = bound.receipt
    admitted = admit_context_plan(
        prepared=prepared,
        bound_context=bound.context,
        adaptive_retrievals=(declaration,),
        context_capabilities=(fixture.source_caps[1],),
    )
    assert admitted is not None and bound.context.receipt is receipt
    assert sum(provider.calls for provider in fixture.providers.values()) == 1
    assert fixture.source_caps[1].source not in fixture.providers
