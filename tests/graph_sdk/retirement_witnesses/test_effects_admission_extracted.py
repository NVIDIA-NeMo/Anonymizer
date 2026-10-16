# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.graph_sdk.test_effects_production_conformance import (
    test_admission_corpus_case_through_production as compare,
)

CASES = json.loads(Path(__file__).with_name("effects-admission-extracted.json").read_text())


def _decision_policy(case: dict[str, Any]) -> bool:
    """Decision waits are outside the SDK, so their admission rows are not replayed."""
    policies = case["declaration"].get("execution_policies")
    return isinstance(policies, list) and any(
        isinstance(item, dict) and item.get("kind") == "decision" for item in policies
    )


@pytest.mark.parametrize(
    "case",
    [
        c
        for c in CASES
        if c["case_id"].split("/")[-1] not in {"d2ab4f9d57ae", "1cd372f706bd", "56b2234c20a7", "59a40bd72ae0"}
        and not _decision_policy(c)
    ],
    ids=lambda c: c["case_id"],
)
def test_extracted_effects_admission(case: dict[str, Any]) -> None:
    compare(case)
