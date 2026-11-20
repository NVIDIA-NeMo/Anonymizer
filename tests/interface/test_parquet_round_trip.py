# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pytest

from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_JUDGE_EVALUATION
from anonymizer.interface.anonymizer import _build_user_dataframe

_ENTITIES = {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
_JUDGE = {"privacy": {"score": 9, "reasoning": "ok"}, "quality": {"score": 8, "reasoning": "ok"}}


def _arrow_column(cell: dict, rows: int) -> pd.Series:
    """A column as run() and evaluate() can load it: an Arrow struct, one cell per row."""
    array = pa.array([cell] * rows)
    return pd.Series(array, dtype=pd.ArrowDtype(array.type))


@pytest.mark.parametrize(
    ("output_column", "nested_column", "cell"),
    [("text_replaced", COL_FINAL_ENTITIES, _ENTITIES), ("text_rewritten", COL_JUDGE_EVALUATION, _JUDGE)],
    ids=["substitute-run", "rewrite-evaluate"],
)
def test_nested_result_columns_round_trip_through_parquet(
    tmp_path: Path, output_column: str, nested_column: str, cell: dict
) -> None:
    trace = pd.DataFrame({"text": ["Alice works at Acme"] * 2, output_column: ["X works at Acme"] * 2})
    trace[nested_column] = _arrow_column(cell, 2)

    user_df = _build_user_dataframe(trace, resolved_text_column="text")
    user_df.to_parquet(tmp_path / "output.parquet")

    saved = pd.read_parquet(tmp_path / "output.parquet")
    assert user_df[nested_column].tolist() == [cell, cell]
    assert len(saved) == 2
