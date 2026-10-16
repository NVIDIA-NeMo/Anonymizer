# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A raised bound admits execution of the same successful binding receipt."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from tests.graph_sdk import effects_binding_assertions as fixture
from tests.graph_sdk.reference.corpora import corpus_bytes

CASES = [
    case
    for case in json.loads(corpus_bytes("effects"))
    if case["case_id"]
    in {
        "materialization/collection_ceiling",
        "materialization/initial_artifact_count_one_over",
        "materialization/initial_logical_bytes_one_over",
        "materialization/initial_provenance_one_over",
    }
]


class _CompletedControl(Exception):
    pass


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_same_successful_receipt_executes_when_preflight_capacity_is_raised(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    original_projection = fixture._assert_initial_execution_projection
    original_start = fixture.start_execution
    results = []

    async def completed_start(**kwargs: Any) -> Any:
        running = await original_start(**kwargs)
        result = await running.wait()
        assert result.artifacts
        assert all(entry.status == "success" for state in result.states for entry in state.entries)
        results.append(result)
        raise _CompletedControl

    async def projected(**kwargs: Any) -> None:
        await original_projection(**kwargs)
        receipt = kwargs["bound_context"].receipt
        raised = dict(kwargs)
        raised["limits"] = {
            **kwargs["limits"],
            "max_artifacts": 100,
            "max_artifact_bytes": 1000,
            "max_provenance_edges": 100,
            "max_collection_items": 100,
        }
        raised["expected_rejection"] = None
        with monkeypatch.context() as patch:
            patch.setattr(fixture, "start_execution", completed_start)
            with pytest.raises(_CompletedControl):
                await original_projection(**raised)
        assert kwargs["bound_context"].receipt is receipt

    monkeypatch.setattr(fixture, "_assert_initial_execution_projection", projected)
    asyncio.run(fixture._assert_binding_corpus_case(case))
    assert len(results) == 1
