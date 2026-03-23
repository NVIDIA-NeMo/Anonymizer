# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.cli._output import format_summary, write_result
from anonymizer.interface.results import AnonymizerResult


def _make_result(num_rows: int = 3, num_failures: int = 1) -> AnonymizerResult:
    df = pd.DataFrame({"text_replaced": [f"row{i}" for i in range(num_rows)]})
    failures = [FailedRecord(record_id=str(i), step="detect", reason="test") for i in range(num_failures)]
    return AnonymizerResult(dataframe=df, trace_dataframe=df.copy(), failed_records=failures)


def test_format_summary_contains_counts() -> None:
    """format_summary returns a string mentioning record count and failure count."""
    result = _make_result(num_rows=3, num_failures=1)
    summary = format_summary(result)
    assert isinstance(summary, str)
    assert "3" in summary
    assert "1" in summary


def test_write_result_csv(tmp_path: Path) -> None:
    """write_result writes a CSV file that can be read back."""
    result = _make_result(num_rows=2, num_failures=0)
    out_path = tmp_path / "out.csv"
    returned = write_result(result, out_path)
    assert out_path.exists()
    assert returned == out_path
    loaded = pd.read_csv(out_path)
    assert list(loaded.columns) == list(result.dataframe.columns)
    assert len(loaded) == 2


def test_write_result_parquet(tmp_path: Path) -> None:
    """write_result writes a Parquet file that can be read back."""
    result = _make_result(num_rows=2, num_failures=0)
    out_path = tmp_path / "out.parquet"
    returned = write_result(result, out_path)
    assert out_path.exists()
    assert returned == out_path
    loaded = pd.read_parquet(out_path)
    assert list(loaded.columns) == list(result.dataframe.columns)
    assert len(loaded) == 2
