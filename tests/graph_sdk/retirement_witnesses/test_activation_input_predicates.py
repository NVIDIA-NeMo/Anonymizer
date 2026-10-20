# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Known answers retained from legacy parent-history and duplicate witnesses."""

from __future__ import annotations

import pytest

from tests.graph_sdk.retirement_witnesses.activation_input_predicates import forest_code, unique_names


@pytest.mark.parametrize("depth", [0, 1, 2, 12])
def test_parent_cycle_does_not_charge_fictitious_root_depth(depth: int) -> None:
    assert forest_code({"A0": "A1", "A1": "A0"}, max_depth=depth) == "cycle"


def test_real_long_path_precedes_unrelated_parent_cycle() -> None:
    parents = {"A0": "A1", "A1": "A0", "A2": None, "A3": "A2"}
    assert forest_code(parents, max_depth=1) == "limit_exceeded"
    assert forest_code(parents, max_depth=2) == "cycle"


def test_finite_parent_chains_and_bijective_renaming() -> None:
    for size in range(1, 5):
        parents = {str(index): str(index - 1) if index else None for index in range(size)}
        renamed = {"renamed" + key: None if value is None else "renamed" + value for key, value in parents.items()}
        for depth in range(5):
            expected = None if depth >= size else "limit_exceeded"
            assert forest_code(parents, max_depth=depth) == expected
            assert forest_code(renamed, max_depth=depth) == expected


def test_duplicate_outputs_and_members_must_precede_set_conversion() -> None:
    assert unique_names(["A1"])
    assert not unique_names(["A1", "A1"])
    assert unique_names(["result", "carry"])
    assert not unique_names(["result", "result"])


def test_precedence_and_duplicate_mutants_are_killed() -> None:
    with pytest.raises(AssertionError):
        assert forest_code({"A0": "A1", "A1": "A0"}, max_depth=0) == "limit_exceeded"
    with pytest.raises(AssertionError):
        assert unique_names(["A1", "A1"])
    with pytest.raises(AssertionError):
        assert forest_code({"A0": "A1", "A1": "A0", "A2": None, "A3": "A2"}, max_depth=1) == "cycle"
