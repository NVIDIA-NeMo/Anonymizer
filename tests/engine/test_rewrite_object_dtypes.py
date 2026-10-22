# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Object columns of the lowered Rewrite frame keep their dtype when the published frame is rebuilt row by row."""

from __future__ import annotations

import pandas as pd
import pytest

from anonymizer.engine.rewrite._semantic_evaluation import _restore_object_dtypes


def _restore(cells: list[str | None], *, held_missing: frozenset[str] = frozenset()) -> pd.DataFrame:
    frame = pd.DataFrame({"text": pd.Series(["x", "y"], dtype=object), "other": [1, 2]})
    rows = [{"text": cell, "other": index} for index, cell in enumerate(cells)]
    projected = pd.DataFrame({"text": pd.Series(["a", "b"], dtype="string"), "other": [0, 1]})
    _restore_object_dtypes(projected, frame, rows, held_missing)
    return projected


@pytest.mark.parametrize("cells", [[None, "b"], [None, None]], ids=["partial-none", "all-none"])
def test_object_column_holding_none_stays_object(cells: list[str | None]) -> None:
    projected = _restore(cells)

    assert projected["text"].dtype == object
    assert projected["text"].tolist() == cells
    assert projected["other"].dtype == "int64"


def test_object_column_that_held_missing_cells_stays_object_once_repair_fills_them() -> None:
    projected = _restore(["a", "b"], held_missing=frozenset({"text"}))

    assert projected["text"].dtype == object
    assert projected["text"].tolist() == ["a", "b"]


def test_object_column_that_never_held_missing_cells_is_rebuilt() -> None:
    assert _restore(["a", "b"]).equals(pd.DataFrame({"text": pd.Series(["a", "b"], dtype="string"), "other": [0, 1]}))
