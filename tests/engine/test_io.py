# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import AnonymizerInput
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.io.reader import read_input
from anonymizer.engine.io.writer import write_output
from anonymizer.interface.errors import AnonymizerIOError, InvalidInputError


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


def test_read_input_from_remote_csv_url(monkeypatch: pytest.MonkeyPatch) -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    source = "https://example.com/data.csv"

    def _read_csv(url: str, *args: object, **kwargs: object) -> pd.DataFrame:
        assert url == source
        return input_df

    monkeypatch.setattr(pd, "read_csv", _read_csv)
    result = read_input(AnonymizerInput(source=source))
    assert result[COL_TEXT].tolist() == ["Alice works at Acme"]
    assert result.attrs["original_text_column"] == "text"


def test_read_input_from_remote_parquet_url_with_query_params(monkeypatch: pytest.MonkeyPatch) -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    source = "https://example.com/data.parquet?download=1"

    def _read_parquet(url: str, *args: object, **kwargs: object) -> pd.DataFrame:
        assert url == source
        return input_df

    monkeypatch.setattr(pd, "read_parquet", _read_parquet)
    result = read_input(AnonymizerInput(source=source))
    assert result[COL_TEXT].tolist() == ["Alice works at Acme"]
    assert result.attrs["original_text_column"] == "text"


def test_read_input_from_remote_csv_url_with_fragment(monkeypatch: pytest.MonkeyPatch) -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    source = "https://example.com/data.csv#section-1"

    def _read_csv(url: str, *args: object, **kwargs: object) -> pd.DataFrame:
        assert url == source
        return input_df

    monkeypatch.setattr(pd, "read_csv", _read_csv)
    result = read_input(AnonymizerInput(source=source))
    assert result[COL_TEXT].tolist() == ["Alice works at Acme"]
    assert result.attrs["original_text_column"] == "text"


def test_read_input_remote_url_with_unsupported_format_raises() -> None:
    inp = AnonymizerInput(source="https://example.com/data.json")
    with pytest.raises(InvalidInputError, match="Unsupported input format"):
        read_input(inp)


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


def test_anonymizer_input_missing_path_raises_validation_error(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing" / "data.csv"
    with pytest.raises(ValidationError, match="Input path does not exist"):
        AnonymizerInput(source=str(missing_file))


def test_write_output_creates_parent_directories(stub_dataframe: pd.DataFrame, tmp_path: Path) -> None:
    out_path = tmp_path / "nested" / "deep" / "out.csv"
    write_output(stub_dataframe, out_path)
    assert out_path.exists()


def test_anonymizer_input_directory_path_raises_validation_error(tmp_path: Path) -> None:
    directory_path = tmp_path / "data.csv"
    directory_path.mkdir()
    with pytest.raises(ValidationError, match="Input path is not a file"):
        AnonymizerInput(source=str(directory_path))


def test_read_input_pandas_failure_raises_anonymizer_io_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "data.csv"
    file_path.write_text("text\nhello\n")

    def _raise_read_error(*args: object, **kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(pd, "read_csv", _raise_read_error)
    inp = AnonymizerInput(source=str(file_path))
    with pytest.raises(AnonymizerIOError, match="Failed to read input data"):
        read_input(inp)


def test_read_input_remote_csv_failure_raises_anonymizer_io_error(monkeypatch: pytest.MonkeyPatch) -> None:
    source = "https://example.com/data.csv"

    def _raise_read_error(*args: object, **kwargs: object) -> None:
        raise OSError("network failure")

    monkeypatch.setattr(pd, "read_csv", _raise_read_error)
    inp = AnonymizerInput(source=source)
    with pytest.raises(AnonymizerIOError, match="Failed to read input data"):
        read_input(inp)


def test_write_output_unwritable_path_raises_anonymizer_io_error(
    stub_dataframe: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out_path = tmp_path / "nested" / "out.csv"

    def _raise_write_error(*args: object, **kwargs: object) -> None:
        raise PermissionError("permission denied")

    monkeypatch.setattr(pd.DataFrame, "to_csv", _raise_write_error)
    with pytest.raises(AnonymizerIOError, match="Failed to write output data"):
        write_output(stub_dataframe, out_path)


# ---------------------------------------------------------------------------
# nrows (preview row limit) tests
# ---------------------------------------------------------------------------


def test_read_input_nrows_truncates_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": [f"row {i}" for i in range(20)]})
    file_path = tmp_path / "big.csv"
    df.to_csv(file_path, index=False)

    result = read_input(AnonymizerInput(source=str(file_path)), nrows=5)
    assert len(result) == 5
    assert result[COL_TEXT].tolist() == [f"row {i}" for i in range(5)]


def test_read_input_nrows_truncates_parquet(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": [f"row {i}" for i in range(20)]})
    file_path = tmp_path / "big.parquet"
    df.to_parquet(file_path, index=False)

    result = read_input(AnonymizerInput(source=str(file_path)), nrows=5)
    assert len(result) == 5
    assert result[COL_TEXT].tolist() == [f"row {i}" for i in range(5)]


def test_read_input_nrows_larger_than_file_returns_all(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": ["a", "b", "c"]})
    file_path = tmp_path / "small.csv"
    df.to_csv(file_path, index=False)

    result = read_input(AnonymizerInput(source=str(file_path)), nrows=100)
    assert len(result) == 3


def test_read_input_nrows_none_returns_all(tmp_path: Path) -> None:
    df = pd.DataFrame({"text": [f"row {i}" for i in range(20)]})
    file_path = tmp_path / "full.csv"
    df.to_csv(file_path, index=False)

    result = read_input(AnonymizerInput(source=str(file_path)), nrows=None)
    assert len(result) == 20


def test_read_input_nrows_preserves_attrs(tmp_path: Path) -> None:
    df = pd.DataFrame({"bio": [f"row {i}" for i in range(10)]})
    file_path = tmp_path / "data.csv"
    df.to_csv(file_path, index=False)

    result = read_input(AnonymizerInput(source=str(file_path), text_column="bio"), nrows=3)
    assert len(result) == 3
    assert result.attrs["original_text_column"] == "bio"
    assert COL_TEXT in result.columns


def test_read_input_nrows_remote_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    full_df = pd.DataFrame({"text": [f"row {i}" for i in range(50)]})
    source = "https://example.com/data.csv"

    def _read_csv(url: str, *args: object, **kwargs: object) -> pd.DataFrame:
        nrows = kwargs.get("nrows")
        if nrows is not None:
            return full_df.head(nrows)
        return full_df

    monkeypatch.setattr(pd, "read_csv", _read_csv)
    result = read_input(AnonymizerInput(source=source), nrows=5)
    assert len(result) == 5
