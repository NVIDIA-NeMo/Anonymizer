# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerInput
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.io.writer import write_output
from anonymizer.interface.errors import InvalidInputError


def test_write_output_csv_roundtrips(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "out.csv"
    write_output(stub_dataframe, out_path)
    loaded = pd.read_csv(out_path)
    assert loaded[COL_TEXT].tolist() == stub_dataframe[COL_TEXT].tolist()


def test_write_output_parquet_roundtrips(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "out.parquet"
    write_output(stub_dataframe, out_path)
    loaded = pd.read_parquet(out_path)
    assert loaded[COL_TEXT].tolist() == stub_dataframe[COL_TEXT].tolist()


def test_write_output_unsupported_format_raises(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="Unsupported output format"):
        write_output(stub_dataframe, tmp_path / "out.xlsx")


@pytest.mark.parametrize(
    "suffix,writer",
    [
        (".csv", lambda df, p: df.to_csv(p, index=False)),
        (".parquet", lambda df, p: df.to_parquet(p, index=False)),
    ],
)
def test_read_input_from_file(suffix: str, writer: object, tmp_path: Path) -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    file_path = tmp_path / f"data{suffix}"
    writer(input_df, file_path)
    inp = AnonymizerInput(source=str(file_path))
    result = read_input(inp)
    assert COL_TEXT in result.columns


def test_read_input_renames_text_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"content": ["hello world"]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    inp = AnonymizerInput(source=str(file_path), text_column="content")
    result = read_input(inp)
    assert COL_TEXT in result.columns
    assert result.attrs["original_text_column"] == "content"


def test_read_input_missing_text_column_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"other": ["hello"]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    inp = AnonymizerInput(source=str(file_path), text_column="missing_col")
    with pytest.raises(InvalidInputError, match="Input text column 'missing_col' not found"):
        read_input(inp)


def test_read_input_internal_text_collision_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({COL_TEXT: ["hello"], "bio": ["other"]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    inp = AnonymizerInput(source=str(file_path), text_column="bio")
    with pytest.raises(InvalidInputError, match="reserved internal column"):
        read_input(inp)


def test_read_input_unsupported_format_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "data.json"
    file_path.write_text('{"a":[1]}')
    inp = AnonymizerInput(source=str(file_path))
    with pytest.raises(InvalidInputError, match="Unsupported input format"):
        read_input(inp)


def test_read_input_preserves_text_attr_when_column_exists(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": ["hello"]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    inp = AnonymizerInput(source=str(file_path))
    result = read_input(inp)
    assert result.attrs["original_text_column"] == "text"


def test_read_input_missing_path_raises(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing" / "data.csv"
    inp = AnonymizerInput(source=str(missing_file))
    with pytest.raises(InvalidInputError, match="does not exist"):
        read_input(inp)


def test_write_output_creates_parent_directories(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "nested" / "deep" / "out.csv"
    write_output(stub_dataframe, out_path)
    assert out_path.exists()


def test_read_input_directory_path_raises(tmp_path: Path) -> None:
    directory_path = tmp_path / "data.csv"
    directory_path.mkdir()
    inp = AnonymizerInput(source=str(directory_path))
    with pytest.raises(InvalidInputError, match="is not a file"):
        read_input(inp)


def test_read_input_pandas_failure_raises_invalid_input_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "data.csv"
    file_path.write_text("text\nhello\n")

    def _raise_read_error(*args: object, **kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(pd, "read_csv", _raise_read_error)
    inp = AnonymizerInput(source=str(file_path))
    with pytest.raises(InvalidInputError, match="Failed to read input data"):
        read_input(inp)


def test_write_output_unwritable_path_raises_invalid_input_error(
    stub_dataframe: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "nested" / "out.csv"

    def _raise_write_error(*args: object, **kwargs: object) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _raise_write_error)
    with pytest.raises(InvalidInputError, match="Failed to write output data"):
        write_output(stub_dataframe, out_path)
