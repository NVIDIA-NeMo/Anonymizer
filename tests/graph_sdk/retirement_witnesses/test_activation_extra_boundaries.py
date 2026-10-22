# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve initialization exhaustion separately from static admission."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer.graph import ContractViolation, ValidationCode
from tests.graph_sdk.test_activation_reference_conformance import _Adapter

ROWS = json.loads(Path(__file__).with_name("extra-activation-traces.json").read_text())


def test_zero_entries_rejects_at_initializer_after_valid_static_admission() -> None:
    case = deepcopy(next(row["case"] for row in ROWS if row["case"]["declaration"]["limits"]["max_entries"] == 0))
    case["declaration"]["limits"]["max_entries"] = 1
    adapter = _Adapter(case)
    adapter.build_dynamic()
    control = adapter.initialize()
    assert not control.entries
    cast(dict[str, Any], adapter.declaration["limits"])["max_entries"] = 0
    with pytest.raises(ContractViolation) as rejected:
        adapter.initialize()
    assert rejected.value.code is ValidationCode.LIMIT_EXCEEDED
    assert not control.entries


def test_carry_only_input_graph_rejects_before_activation() -> None:
    case = next(row["case"] for row in ROWS if row["case"]["case_id"] == "retirement/carry-only")
    adapter = _Adapter(case)
    with pytest.raises(ContractViolation) as rejected:
        adapter.build_dynamic()
    assert rejected.value.code is ValidationCode.MISSING
    assert adapter.workflow is None
