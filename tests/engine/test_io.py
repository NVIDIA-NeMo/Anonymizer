# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerInput
from anonymizer.engine.detection.constants import COL_TEXT
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
    assert loaded[COL_TEXT].tolist() == stub_dataframe[COL_TEXT].tolist()


def test_write_output_parquet_roundtrips(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "out.parquet"
    write_output(stub_dataframe, out_path)
    loaded = pd.read_parquet(out_path)
    assert loaded[COL_TEXT].tolist() == stub_dataframe[COL_TEXT].tolist()


def test_write_output_unsupported_format_raises(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError, match="Unsupported output format"):
        write_output(stub_dataframe, tmp_path / "out.xlsx")


# ---------------------------------------------------------------------------
# reader: read_input — DataFrame path
# ---------------------------------------------------------------------------


def test_read_input_from_dataframe() -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    result = read_input(input_df)
    assert COL_TEXT in result.columns
    assert result[COL_TEXT].tolist() == input_df["text"].tolist()


def test_read_input_from_dataframe_does_not_mutate_original() -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    result = read_input(input_df)
    result[COL_TEXT] = "modified"
    assert input_df["text"].iloc[0] != "modified"


# ---------------------------------------------------------------------------
# reader: read_input — AnonymizerInput with file sources
# ---------------------------------------------------------------------------


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
