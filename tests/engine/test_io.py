# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerInput, InputSourceType
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.io.writer import write_output
from anonymizer.interface.errors import InvalidInputError

# ---------------------------------------------------------------------------
# writer: write_output
# ---------------------------------------------------------------------------


def test_write_output_csv_roundtrips(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "out.csv"
    write_output(stub_dataframe, out_path)
    loaded = pd.read_csv(out_path)
    assert loaded["text"].tolist() == stub_dataframe["text"].tolist()


def test_write_output_parquet_roundtrips(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "out.parquet"
    write_output(stub_dataframe, out_path)
    loaded = pd.read_parquet(out_path)
    assert loaded["text"].tolist() == stub_dataframe["text"].tolist()


def test_write_output_unsupported_format_raises(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="Unsupported output format"):
        write_output(stub_dataframe, tmp_path / "out.xlsx")


# ---------------------------------------------------------------------------
# reader: read_input — DataFrame path
# ---------------------------------------------------------------------------


def test_read_input_from_dataframe(stub_dataframe: pd.DataFrame) -> None:
    result = read_input(stub_dataframe)
    assert "text" in result.columns
    assert result["text"].tolist() == stub_dataframe["text"].tolist()


def test_read_input_from_dataframe_does_not_mutate_original(stub_dataframe: pd.DataFrame) -> None:
    result = read_input(stub_dataframe)
    result["text"] = "modified"
    assert stub_dataframe["text"].iloc[0] != "modified"


# ---------------------------------------------------------------------------
# reader: read_input — AnonymizerInput with file sources
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source_type,suffix,writer",
    [
        (InputSourceType.csv, ".csv", lambda df, p: df.to_csv(p, index=False)),
        (InputSourceType.parquet, ".parquet", lambda df, p: df.to_parquet(p, index=False)),
        (InputSourceType.json, ".json", lambda df, p: df.to_json(p)),
    ],
)
def test_read_input_from_file(
    stub_dataframe: pd.DataFrame,
    source_type: InputSourceType,
    suffix: str,
    writer: object,
    tmp_path: Path,
) -> None:
    file_path = tmp_path / f"data{suffix}"
    writer(stub_dataframe, file_path)
    inp = AnonymizerInput(source=str(file_path), source_type=source_type)
    result = read_input(inp)
    assert "text" in result.columns


def test_read_input_renames_text_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"content": ["hello world"]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    inp = AnonymizerInput(source=str(file_path), source_type=InputSourceType.csv, text_column="content")
    result = read_input(inp)
    assert "text" in result.columns
    assert result.attrs["original_text_column"] == "content"


def test_read_input_missing_text_column_raises(tmp_path: Path) -> None:
    df = pd.DataFrame({"other": ["hello"]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    inp = AnonymizerInput(source=str(file_path), source_type=InputSourceType.csv, text_column="missing_col")
    with pytest.raises(InvalidInputError, match="Input text column 'missing_col' not found"):
        read_input(inp)


def test_read_input_preserves_text_attr_when_column_exists(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": ["hello"]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)
    inp = AnonymizerInput(source=str(file_path), source_type=InputSourceType.csv)
    result = read_input(inp)
    assert result.attrs["original_text_column"] == "text"
