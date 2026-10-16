# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Derived subject and target fields retain actual occurrence ownership."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


@pytest.mark.parametrize("field", ["artifact"])
def test_actual_subject_occurrence_cannot_borrow_another_targets_fields(field: str) -> None:
    execution, result = asyncio.run(_execute_separate_subject_context(target_labels=("A", "B")))
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    first, second = tuple(item for item in result.ports if item.port == "subject")
    assert first.target != second.target and first.artifact != second.artifact
    old = getattr(first, field)
    object.__setattr__(first, field, getattr(second, field))
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is (EffectCode.CONTRADICTORY if field == "artifact" else EffectCode.FOREIGN_OWNER)
    finally:
        object.__setattr__(first, field, old)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
