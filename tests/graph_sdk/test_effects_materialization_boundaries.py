# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public equivalents of closed materialization boundary cases."""

from __future__ import annotations

import pytest

from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_context_source_execution import (
    _assert_collection_root_preflight,
)


def test_known_collection_cannot_be_supplied_as_a_caller_root(monkeypatch: pytest.MonkeyPatch) -> None:
    case = next(case for case in load_cases("effects") if case["case_id"] == "materialization/collection_root_input")
    raw = case["declaration"]["materializations"][0]
    assert raw["kind"] == "collection"
    assert raw["output_type"] in case["declaration"]["root_input_types"]
    assert raw["item_type"] != raw["output_type"]
    assert case["expected"] == {"status": "rejected", "code": "contradictory"}
    # The shared production fixture alpha-renames text/text_collection to
    # item/collection and proves zero BindingIds and provider factory calls.
    _assert_collection_root_preflight(monkeypatch)
