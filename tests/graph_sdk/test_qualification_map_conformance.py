# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execute the independent map-evidence topology through public SDK owners."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from tests.graph_sdk.qualification_map_execution import CORPUS, FLAT_CASE_IDS
from tests.graph_sdk.qualification_map_execution import _execute_reference_map as _execute_reference_map
from tests.graph_sdk.qualification_map_records import _run_reference_map_case


@pytest.mark.parametrize(
    "case", [case for case in CORPUS if case["case_id"] in FLAT_CASE_IDS], ids=lambda case: case["case_id"]
)
def test_reference_map_case_through_real_execution(case: dict[str, Any]) -> None:
    try:
        actual = asyncio.run(_run_reference_map_case(case))
    except EffectRejected as exc:
        actual = {"status": "rejected", "code": exc.code.value}
    expected = json.loads(json.dumps(case["expected"]))
    if "verified" in expected:
        expected["verified"].sort(key=lambda row: (row["target"], row["activation"], row["evidence_artifact"]))
    if "record" in expected:
        for membership in expected["record"]["memberships"].values():
            membership["members"].sort()
    assert actual == expected
