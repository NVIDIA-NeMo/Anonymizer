# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Empty-map retained outcome checks preserve the admitted declaration."""

from __future__ import annotations

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.test_versioned_map_and_membership_witnesses import _execute_root_protected_map, _root_inputs


def test_empty_map_retained_outcome_cannot_disagree_with_unchanged_declaration(monkeypatch: pytest.MonkeyPatch) -> None:
    execution, result, nodes = _execute_root_protected_map(monkeypatch, count=0)
    admitted, current, submissions = _root_inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    (expansion,) = result.states[0].expansions
    assert expansion.status == "closed" and not expansion.members
    entry = next(item for item in result.states[0].entries if item.template == nodes["EXP"])
    mapping = next(mapping for scope in execution.context.prepared.workflow.bodies() for mapping in scope.maps)
    original_outcomes = mapping.expansion_outcomes
    original = entry.outcome
    assert original in original_outcomes and "other" not in original_outcomes
    object.__setattr__(entry, "outcome", "other")
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is EffectCode.CONTRADICTORY
        assert mapping.expansion_outcomes is original_outcomes
        assert expansion.status == "closed" and not expansion.members
    finally:
        object.__setattr__(entry, "outcome", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
