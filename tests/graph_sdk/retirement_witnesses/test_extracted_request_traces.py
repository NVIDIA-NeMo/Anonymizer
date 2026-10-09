# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute extracted legacy selftest histories through the real request reducer."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from tests.graph_sdk import test_effects_production_conformance as request_fixture
from tests.graph_sdk.retirement_witnesses.test_owner_boundaries import (
    test_binding_finish_and_wrong_source as check_binding,
)
from tests.graph_sdk.test_effects_production_conformance import (
    test_request_corpus_case_through_production as compare_request,
)

_ALLOWED = frozenset(
    {
        "bind_policy",
        "reserve",
        "dispatch",
        "result",
        "failure",
        "cancel",
        "scope_cancel",
        "stop",
        "lost",
        "settlement",
        "source_failure",
        "source_result",
    }
)
_CASES = [
    row
    for row in json.loads(Path(__file__).with_name("extra-effects-traces.json").read_text())
    if {event["kind"] for event in row["events"]} <= _ALLOWED
]


@pytest.mark.parametrize("case", _CASES, ids=lambda row: row["case_id"])
def test_extracted_history(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    # Reviewed439/447: semantic binding completion is outside the physical ledger.
    if case["case_id"] == "retirement-33":
        assert case["expected"]["state"]["binding_sources"] == {"D0": "bound"}
        check_binding("binding/exact_item_byte_bounds")
        return
    if case["case_id"] == "retirement-23":
        assert case["expected"] == {"status": "rejected", "code": "missing_reservation"}
        mapped = deepcopy(case)
        mapped["expected"]["code"] = "missing"
        apply = request_fixture._apply
        checked = False

        def check_dispatch(state: Any, event: dict[str, Any], *args: Any) -> Any:
            nonlocal checked
            if event["kind"] == "dispatch":
                checked = True
                assert not state.dispatched and not state.dispatches
                try:
                    return apply(state, event, *args)
                finally:
                    assert not state.dispatched and not state.dispatches
            return apply(state, event, *args)

        monkeypatch.setattr(request_fixture, "_apply", check_dispatch)
        compare_request(mapped)
        assert checked
        assert case["expected"]["code"] == "missing_reservation"
        return
    compare_request(case)
