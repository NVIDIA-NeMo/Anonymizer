# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.interface.cli._output import write_result
from anonymizer.interface.cli.main import app
from anonymizer.interface.results import AnonymizerResult, PreviewResult


def _make_result(num_rows: int = 2, num_failures: int = 0) -> AnonymizerResult:
    df = pd.DataFrame(
        {"bio": [f"original {i}" for i in range(num_rows)], "bio_replaced": [f"REDACTED_{i}" for i in range(num_rows)]}
    )
    failures = [FailedRecord(record_id=str(i), step="detect", reason="test") for i in range(num_failures)]
    return AnonymizerResult(
        dataframe=df,
        trace_dataframe=df.copy(),
        original_text_column="bio",
        failed_records=failures,
    )


@pytest.fixture
def csv_source(tmp_path: Path) -> Path:
    f = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello", "world"]}).to_csv(f, index=False)
    return f


# ---------------------------------------------------------------------------
# write_result unit tests
# ---------------------------------------------------------------------------


def test_write_result_csv(tmp_path: Path) -> None:
    """write_result writes a CSV file that can be read back."""
    result = _make_result(num_rows=2)
    out_path = tmp_path / "out.csv"
    returned = write_result(result, out_path)
    assert out_path.exists()
    assert returned == out_path
    loaded = pd.read_csv(out_path)
    assert list(loaded.columns) == list(result.dataframe.columns)
    assert len(loaded) == 2


def test_write_result_parquet(tmp_path: Path) -> None:
    """write_result writes a Parquet file that can be read back."""
    result = _make_result(num_rows=2)
    out_path = tmp_path / "out.parquet"
    returned = write_result(result, out_path)
    assert out_path.exists()
    assert returned == out_path
    loaded = pd.read_parquet(out_path)
    assert list(loaded.columns) == list(result.dataframe.columns)
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# run subcommand output tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ext", [".csv", ".parquet"])
def test_run_default_output_path(tmp_path: Path, capsys: pytest.CaptureFixture, ext: str) -> None:
    """run with no --output writes to {stem}_anonymized{ext} next to the source."""
    source = tmp_path / f"data{ext}"
    df = pd.DataFrame({"text": ["hello"]})
    if ext == ".csv":
        df.to_csv(source, index=False)
    else:
        df.to_parquet(source, index=False)

    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result()

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(source), "--replace", "redact"])
    assert exc_info.value.code == 0

    expected = tmp_path / f"data_anonymized{ext}"
    assert expected.exists()
    assert f"data_anonymized{ext}" in capsys.readouterr().out


def test_run_rewrite_default_output_path(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    """run --rewrite with no --output writes to {stem}_rewritten.csv."""
    source = tmp_path / "data.csv"
    pd.DataFrame({"text": ["hello"]}).to_csv(source, index=False)

    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result()

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(source), "--rewrite"])
    assert exc_info.value.code == 0

    expected = tmp_path / "data_rewritten.csv"
    assert expected.exists()
    assert "data_rewritten.csv" in capsys.readouterr().out


def test_run_explicit_output(tmp_path: Path, capsys: pytest.CaptureFixture, csv_source: Path) -> None:
    """run with --output writes to the specified path and prints it."""
    out_file = tmp_path / "custom_out.csv"

    mock_anonymizer = MagicMock()
    mock_anonymizer.run.return_value = _make_result()

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["run", "--source", str(csv_source), "--replace", "redact", "--output", str(out_file)])
    assert exc_info.value.code == 0

    assert out_file.exists()
    assert str(out_file) in capsys.readouterr().out


# ---------------------------------------------------------------------------
# preview subcommand output tests
# ---------------------------------------------------------------------------


def test_preview_prints_dataframe(tmp_path: Path, capsys: pytest.CaptureFixture, csv_source: Path) -> None:
    """preview prints the result dataframe to stdout."""
    result = _make_result(num_rows=2)
    preview_result = PreviewResult(
        dataframe=result.dataframe,
        trace_dataframe=result.trace_dataframe,
        original_text_column="bio",
        failed_records=[],
        preview_num_records=2,
    )

    mock_anonymizer = MagicMock()
    mock_anonymizer.preview.return_value = preview_result

    with patch("anonymizer.interface.cli.main.Anonymizer", return_value=mock_anonymizer):
        with pytest.raises(SystemExit) as exc_info:
            app(["preview", "--source", str(csv_source), "--replace", "redact"])
    assert exc_info.value.code == 0

    out = capsys.readouterr().out
    assert "bio_replaced" in out
    assert "REDACTED_0" in out
