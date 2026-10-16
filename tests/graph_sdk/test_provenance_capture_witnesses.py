# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Captured producer absence differs from a contradictory output-parent set."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


@pytest.mark.parametrize("capture", ["input", "passthrough"])
def test_missing_captured_producer_rejects_as_missing(capture: str) -> None:
    execution, result = asyncio.run(
        _execute_assessment(
            nested=capture == "passthrough", root_passthrough=capture == "passthrough", candidate_input=True
        )
    )
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    field = "_input_parents" if capture == "input" else "_passthrough_parents"
    parents = getattr(result, field)
    assert parents
    object.__setattr__(result, field, parents[1:])
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.MISSING
    finally:
        object.__setattr__(result, field, parents)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
