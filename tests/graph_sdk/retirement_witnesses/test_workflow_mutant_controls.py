# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual compiler controls for retained independent workflow mutant witnesses."""

from __future__ import annotations

from typing import Any

import pytest

from tests.graph_sdk.retirement_witnesses.witness_sources import workflow_mutant_cases
from tests.graph_sdk.test_workflow_static import _assert_case

CASES = workflow_mutant_cases()


@pytest.mark.parametrize("case", CASES, ids=lambda row: row["case_id"])
def test_named_workflow_witness_on_real_compiler(case: dict[str, Any]) -> None:
    _assert_case(case)
