# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tests.graph_sdk.retirement_witnesses.witness_sources import activation_differences
from tests.graph_sdk.test_activation_reference_conformance import (
    test_frozen_activation_case_matches_product as compare,
)

DIFFERENCES = {c["case_id"] for c in activation_differences()}
CASES = json.loads(Path(__file__).with_name("activation-extracted-cases.json").read_text())


@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c["case_id"] not in DIFFERENCES],
    ids=lambda c: c["case_id"],
)
def test_extracted_activation(case: dict[str, Any]) -> None:
    compare("I0", case)
