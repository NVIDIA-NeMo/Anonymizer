# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nullable and non-string inputs complete on each route, keep every row accounted for and keep passthrough dtypes.

The library documents no output for a null or non-string text, so these tests assert the invariants only.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from anonymizer import Anonymizer
from anonymizer.config.anonymizer_config import AnonymizerInput
from tests.parity.harness import CASES, RECORDS, Case, scripted


def _base() -> pd.DataFrame:
    return pd.DataFrame({"id": [i for i, _ in RECORDS], "text": [t for _, t in RECORDS]})


def _with_passthrough(frame: pd.DataFrame) -> pd.DataFrame:
    n = len(frame)
    return frame.assign(
        kind=pd.Categorical(["x", "y"] * (n // 2) + ["x"] * (n % 2)),
        count=pd.array([1, None] + [3] * (n - 2), dtype="Int64"),
        seen=pd.to_datetime(["2026-01-01"] * n, utc=True),
        flag=pd.array([True, None] + [False] * (n - 2), dtype="boolean"),
        score=[0.5, float("nan")] + [1.0] * (n - 2),
    )


VARIANTS: dict[str, tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = {
    "none": ("parquet", lambda f: f.assign(text=f.text.astype(object).where(f.index != 2, None))),
    "csv_empty": ("csv", lambda f: f.assign(text=f.text.where(f.index != 2, ""))),
    "whitespace": ("parquet", lambda f: f.assign(text=f.text.where(f.index != 2, "   "))),
    "string_pyarrow_na": (
        "parquet",
        lambda f: f.assign(text=pd.array([*f.text[:2], pd.NA, *f.text[3:]], dtype="string[pyarrow]")),
    ),
    "category": ("parquet", lambda f: f.assign(text=pd.Categorical(f.text))),
    "int": ("csv", lambda f: f.assign(text=range(1000, 1000 + len(f)))),
    "all_null": ("parquet", lambda f: f.assign(text=None)),
    "zero_rows": ("parquet", lambda f: f.iloc[:0]),
    "odd_passthrough": ("parquet", _with_passthrough),
}


def _write(variant: str, directory: Path) -> tuple[Path, pd.DataFrame]:
    kind, build = VARIANTS[variant]
    frame = build(_base())
    path = directory / f"{variant}.{kind}"
    if kind == "csv":
        frame.to_csv(path, index=False)
    else:
        frame.to_parquet(path, index=False)
    return path, frame


def _passthrough(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("id", "text")]


ROUTES = [case for case in CASES if case.name in {"redact", "substitute", "rewrite_staged"}]


@pytest.mark.parametrize("case", ROUTES, ids=lambda case: case.name)
@pytest.mark.parametrize("variant", VARIANTS, ids=str)
def test_input_variant_completes_and_keeps_rows_and_dtypes(case: Case, variant: str, tmp_path: Path) -> None:
    source, frame = _write(variant, tmp_path)
    with scripted(case):
        result = Anonymizer(artifact_path=tmp_path / "artifacts").run(
            config=case.config(), data=AnonymizerInput(source=str(source), text_column="text")
        )
    # The scripted adapter picks the rows it drops by text, so a variant may change which rows fail: count, not pin.
    assert len(result.dataframe) + len(result.failed_records) == len(frame)
    for column in _passthrough(frame):
        assert result.trace_dataframe[column].dtype == frame[column].dtype, column
    buffer = io.BytesIO()
    result.dataframe.to_parquet(buffer)
    assert list(pd.read_parquet(io.BytesIO(buffer.getvalue())).columns) == list(result.dataframe.columns)
