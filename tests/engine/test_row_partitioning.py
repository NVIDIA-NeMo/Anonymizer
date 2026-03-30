# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from anonymizer.engine.row_partitioning import merge_and_reorder, split_rows

_MIXED_VALUES = [1, 0, 3, 0, 5]

# ---------------------------------------------------------------------------
# split_rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "values, expected_match, expected_non_match",
    [
        (_MIXED_VALUES, [1, 3, 5], [0, 0]),
        ([1, 2, 3], [1, 2, 3], []),
        ([0, 0, 0], [], [0, 0, 0]),
    ],
    ids=["mixed", "all-matching", "none-matching"],
)
def test_split_rows_partitions(
    values: list[int], expected_match: list[int], expected_non_match: list[int]
) -> None:
    df = pd.DataFrame({"val": values})
    matching, non_matching = split_rows(df, column="val", predicate=bool)

    assert list(matching["val"]) == expected_match
    assert list(non_matching["val"]) == expected_non_match
    assert "_anonymizer_row_order" in matching.columns
    assert "_anonymizer_row_order" in non_matching.columns


def test_split_rows_empty_df() -> None:
    df = pd.DataFrame({"val": pd.Series([], dtype=int)})
    matching, non_matching = split_rows(df, column="val", predicate=bool)
    assert len(matching) == 0
    assert len(non_matching) == 0


def test_split_rows_does_not_mutate_input() -> None:
    df = pd.DataFrame({"val": _MIXED_VALUES})
    original_cols = list(df.columns)
    split_rows(df, column="val", predicate=bool)
    assert list(df.columns) == original_cols


def test_split_rows_custom_predicate() -> None:
    df = pd.DataFrame({"name": ["alice", "bob", "charlie"]})
    matching, _ = split_rows(df, column="name", predicate=lambda x: len(x) > 3)
    assert list(matching["name"]) == ["alice", "charlie"]


# ---------------------------------------------------------------------------
# merge_and_reorder
# ---------------------------------------------------------------------------


def test_merge_and_reorder_restores_order() -> None:
    df = pd.DataFrame({"val": _MIXED_VALUES})
    matching, non_matching = split_rows(df, column="val", predicate=bool)
    combined = merge_and_reorder(matching, non_matching, attrs={})

    assert list(combined["val"]) == _MIXED_VALUES
    assert "_anonymizer_row_order" not in combined.columns
    assert list(combined.index) == list(range(len(combined)))


def test_merge_and_reorder_propagates_attrs() -> None:
    df = pd.DataFrame({"val": _MIXED_VALUES})
    matching, non_matching = split_rows(df, column="val", predicate=bool)
    attrs = {"key": "value", "nested": {"a": 1}}
    combined = merge_and_reorder(matching, non_matching, attrs=attrs)
    assert combined.attrs == attrs


def test_merge_and_reorder_single_partition() -> None:
    df = pd.DataFrame({"val": [1, 2, 3]})
    matching, _ = split_rows(df, column="val", predicate=bool)
    combined = merge_and_reorder(matching, attrs={})
    assert list(combined["val"]) == [1, 2, 3]


def test_roundtrip_preserves_columns_added_after_split() -> None:
    df = pd.DataFrame({"val": [1, 0, 3]})
    matching, non_matching = split_rows(df, column="val", predicate=bool)

    matching["extra"] = "processed"
    non_matching["extra"] = "skipped"

    combined = merge_and_reorder(matching, non_matching, attrs={})
    assert list(combined["extra"]) == ["processed", "skipped", "processed"]
