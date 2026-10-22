# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from tests.graph_sdk.test_data_reference_conformance import _adapt
from tests.graph_sdk.test_workflow_static import _assert_case

ROOT = Path(__file__).parent


@pytest.mark.parametrize(
    "case",
    json.loads((ROOT / "data-extracted-cases.json").read_text()),
    ids=lambda c: c["case_id"],
)
def test_data_precedence_extraction(case: dict[str, Any]) -> None:
    assert _adapt(cast(Any, case)) == case["expected"]


@pytest.mark.parametrize(
    "case",
    json.loads((ROOT / "workflow_static-extracted-cases.json").read_text()),
    ids=lambda c: c["case_id"],
)
def test_static_substitution_extraction(case: dict[str, Any]) -> None:
    _assert_case(case)
