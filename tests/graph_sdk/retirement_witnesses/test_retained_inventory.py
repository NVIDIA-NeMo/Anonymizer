# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Intermediate retained facts survive ordered qualification submission choices."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


def test_retained_inventory_is_independent_of_submission_order_and_selection() -> None:
    execution, result = asyncio.run(_execute_assessment(target_count=3))
    admitted, current, submissions = _inputs(execution, result, port_facts=32, provenance_edges=32)
    inventory = result.assessments
    assert len(inventory) == 3
    all_output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    reversed_output = qualify(
        admitted=admitted,
        result=result,
        current=current,
        submissions=tuple(reversed(submissions)),
    )
    assert all_output == reversed_output
    assert tuple(item.reference for item in all_output.verified) == tuple(
        item.reference for item in reversed_output.verified
    )
    assert len(all_output.qualified) == 3
    for selected in ((), submissions[:1], submissions[1:], submissions):
        output = qualify(admitted=admitted, result=result, current=current, submissions=selected)
        assert len(output.verified) == len(output.qualified) == len(selected)
        assert result.assessments is inventory
        assert all(left is right for left, right in zip(result.assessments, inventory, strict=True))
    with pytest.raises(EffectRejected) as rejected:
        qualify(
            admitted=admitted,
            result=result,
            current=current,
            submissions=(submissions[0], submissions[0]),
        )
    assert rejected.value.code is EffectCode.DUPLICATE
    assert result.assessments is inventory
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == all_output
