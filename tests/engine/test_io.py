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

_WRITERS = {
    ".csv": lambda df, p: df.to_csv(p, index=False),
    ".parquet": lambda df, p: df.to_parquet(p, index=False),
}


def _write_input(
    df: pd.DataFrame,
    tmp_path: Path,
    suffix: str = ".csv",
    **input_kwargs: object,
) -> AnonymizerInput:
    """Write *df* to a temp file and return a ready-to-use ``AnonymizerInput``."""
    file_path = tmp_path / f"data{suffix}"
    _WRITERS[suffix](df, file_path)
    return AnonymizerInput(source=str(file_path), **input_kwargs)


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
    assert COL_TEXT in result.dataframe.columns


def test_read_input_from_remote_csv_url(monkeypatch: pytest.MonkeyPatch) -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    source = "https://example.com/data.csv"

    def _read_csv(url: str, *args: object, **kwargs: object) -> pd.DataFrame:
        assert url == source
        return input_df

    monkeypatch.setattr(pd, "read_csv", _read_csv)
    result = read_input(AnonymizerInput(source=source))
    assert result.dataframe[COL_TEXT].tolist() == ["Alice works at Acme"]
    assert result.original_text_column == "text"


def test_read_input_from_remote_parquet_url_with_query_params(monkeypatch: pytest.MonkeyPatch) -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    source = "https://example.com/data.parquet?download=1"

    def _read_parquet(url: str, *args: object, **kwargs: object) -> pd.DataFrame:
        assert url == source
        return input_df

    monkeypatch.setattr(pd, "read_parquet", _read_parquet)
    result = read_input(AnonymizerInput(source=source))
    assert result.dataframe[COL_TEXT].tolist() == ["Alice works at Acme"]
    assert result.original_text_column == "text"


def test_read_input_from_remote_csv_url_with_fragment(monkeypatch: pytest.MonkeyPatch) -> None:
    input_df = pd.DataFrame({"text": ["Alice works at Acme"]})
    source = "https://example.com/data.csv#section-1"

    def _read_csv(url: str, *args: object, **kwargs: object) -> pd.DataFrame:
        assert url == source
        return input_df

    monkeypatch.setattr(pd, "read_csv", _read_csv)
    result = read_input(AnonymizerInput(source=source))
    assert result.dataframe[COL_TEXT].tolist() == ["Alice works at Acme"]
    assert result.original_text_column == "text"


def test_read_input_remote_url_with_unsupported_format_raises() -> None:
    inp = AnonymizerInput(source="https://example.com/data.json")
    with pytest.raises(InvalidInputError, match="Unsupported input format"):
        read_input(inp)


def test_read_input_renames_text_column(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"content": ["hello world"]}), tmp_path, text_column="content")
    result = read_input(inp)
    assert COL_TEXT in result.dataframe.columns
    assert result.original_text_column == "content"


def test_read_input_missing_text_column_raises(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"other": ["hello"]}), tmp_path, text_column="missing_col")
    with pytest.raises(InvalidInputError, match="Input text column 'missing_col' not found"):
        read_input(inp)


def test_read_input_internal_text_collision_raises(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({COL_TEXT: ["hello"], "bio": ["other"]}), tmp_path, text_column="bio")
    with pytest.raises(InvalidInputError, match="reserved internal column"):
        read_input(inp)


@pytest.mark.parametrize("suffix", ["_replaced", "_with_spans", "_rewritten"])
def test_read_input_output_column_collision_renames_with_warning(
    tmp_path: Path, suffix: str, caplog: pytest.LogCaptureFixture
) -> None:
    colliding_col = f"bio{suffix}"
    inp = _write_input(
        pd.DataFrame({"bio": ["hello"], colliding_col: ["existing"]}),
        tmp_path,
        text_column="bio",
    )
    with caplog.at_level("WARNING", logger="anonymizer"):
        result = read_input(inp)
    assert COL_TEXT in result.dataframe.columns
    assert colliding_col not in result.dataframe.columns
    renamed = f"{colliding_col}__input"
    assert renamed in result.dataframe.columns
    assert result.dataframe[renamed].tolist() == ["existing"]
    assert any("collide with Anonymizer output column names" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("static_column", ["final_entities", "utility_score", "needs_human_review"])
def test_read_input_static_output_column_collision_renames_with_warning(
    tmp_path: Path, static_column: str, caplog: pytest.LogCaptureFixture
) -> None:
    inp = _write_input(
        pd.DataFrame({"bio": ["hello"], static_column: ["existing"]}),
        tmp_path,
        text_column="bio",
    )
    with caplog.at_level("WARNING", logger="anonymizer"):
        result = read_input(inp)
    assert static_column not in result.dataframe.columns
    assert f"{static_column}__input" in result.dataframe.columns
    assert any("collide with Anonymizer output column names" in rec.message for rec in caplog.records)


@pytest.mark.parametrize(
    "static_column",
    [
        "final_entities",
        "utility_score",
        "leakage_mass",
        "weighted_leakage_rate",
        "any_high_leaked",
        "needs_human_review",
    ],
)
def test_read_input_text_column_equal_to_static_output_renames_with_warning(
    tmp_path: Path, static_column: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Selecting a text column whose name matches a fixed output column is treated
    like any other collision: the input column is renamed with the ``__input``
    suffix and ``original_text_column`` reflects the renamed identifier so the
    end-of-pipeline rename does not clash with the pipeline's own output column.
    """
    inp = _write_input(
        pd.DataFrame({static_column: ["hello"]}),
        tmp_path,
        text_column=static_column,
    )
    with caplog.at_level("WARNING", logger="anonymizer"):
        result = read_input(inp)
    assert COL_TEXT in result.dataframe.columns
    assert static_column not in result.dataframe.columns
    renamed = f"{static_column}__input"
    assert result.original_text_column == renamed
    assert list(result.dataframe.columns).count(COL_TEXT) == 1
    assert any("collide with Anonymizer output column names" in rec.message for rec in caplog.records)


def test_read_input_text_column_is_static_plus_other_static_collision(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A static-name text column coexisting with other static collisions: each
    input column gets its own ``__input`` rename, independent of whether it is
    the selected text column.
    """
    inp = _write_input(
        pd.DataFrame(
            {
                "final_entities": ["hello"],
                "utility_score": [0.9],
                "needs_human_review": [False],
            }
        ),
        tmp_path,
        text_column="final_entities",
    )
    with caplog.at_level("WARNING", logger="anonymizer"):
        result = read_input(inp)
    assert result.original_text_column == "final_entities__input"
    assert {"utility_score__input", "needs_human_review__input"}.issubset(result.dataframe.columns)
    assert "final_entities" not in result.dataframe.columns
    assert "utility_score" not in result.dataframe.columns
    assert "needs_human_review" not in result.dataframe.columns
    assert result.dataframe["utility_score__input"].iloc[0] == 0.9
    assert bool(result.dataframe["needs_human_review__input"].iloc[0]) is False


def test_read_input_output_column_collision_renames_all(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    inp = _write_input(
        pd.DataFrame(
            {
                "bio": ["hello"],
                "bio_replaced": ["x"],
                "bio_rewritten": ["y"],
                "final_entities": ["z"],
            }
        ),
        tmp_path,
        text_column="bio",
    )
    with caplog.at_level("WARNING", logger="anonymizer"):
        result = read_input(inp)
    assert {"bio_replaced__input", "bio_rewritten__input", "final_entities__input"}.issubset(result.dataframe.columns)
    warning = next(
        rec.message for rec in caplog.records if "collide with Anonymizer output column names" in rec.message
    )
    for original in ("bio_replaced", "bio_rewritten", "final_entities"):
        assert f"'{original}'" in warning


def test_read_input_output_column_collision_disambiguates_when_input_suffix_taken(
    tmp_path: Path,
) -> None:
    inp = _write_input(
        pd.DataFrame(
            {
                "bio": ["hello"],
                "bio_replaced": ["new"],
                "bio_replaced__input": ["already_renamed"],
            }
        ),
        tmp_path,
        text_column="bio",
    )
    result = read_input(inp)
    assert "bio_replaced" not in result.dataframe.columns
    assert result.dataframe["bio_replaced__input"].tolist() == ["already_renamed"]
    assert result.dataframe["bio_replaced__input_2"].tolist() == ["new"]


def test_read_input_non_colliding_columns_pass(tmp_path: Path) -> None:
    inp = _write_input(
        pd.DataFrame({"bio": ["hello"], "other_replaced": ["x"], "score": [0.5]}),
        tmp_path,
        text_column="bio",
    )
    result = read_input(inp)
    assert COL_TEXT in result.dataframe.columns
    assert "other_replaced" in result.dataframe.columns
    assert "score" in result.dataframe.columns


def test_read_input_unsupported_format_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "data.json"
    file_path.write_text('{"a":[1]}')
    inp = AnonymizerInput(source=str(file_path))
    with pytest.raises(InvalidInputError, match="Unsupported input format"):
        read_input(inp)


def test_read_input_preserves_text_attr_when_column_exists(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"text": ["hello"]}), tmp_path)
    result = read_input(inp)
    assert result.original_text_column == "text"


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
    inp = _write_input(pd.DataFrame({"text": ["hello"]}), tmp_path)

    def _raise_read_error(*args: object, **kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(pd, "read_csv", _raise_read_error)
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
    inp = _write_input(pd.DataFrame({"text": [f"row {i}" for i in range(20)]}), tmp_path)
    result = read_input(inp, nrows=5)
    assert len(result.dataframe) == 5
    assert result.dataframe[COL_TEXT].tolist() == [f"row {i}" for i in range(5)]


def test_read_input_nrows_truncates_parquet(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"text": [f"row {i}" for i in range(20)]}), tmp_path, suffix=".parquet")
    result = read_input(inp, nrows=5)
    assert len(result.dataframe) == 5
    assert result.dataframe[COL_TEXT].tolist() == [f"row {i}" for i in range(5)]


def test_read_input_nrows_larger_than_file_returns_all(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"text": ["a", "b", "c"]}), tmp_path)
    result = read_input(inp, nrows=100)
    assert len(result.dataframe) == 3


def test_read_input_nrows_none_returns_all(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"text": [f"row {i}" for i in range(20)]}), tmp_path)
    result = read_input(inp, nrows=None)
    assert len(result.dataframe) == 20


def test_read_input_nrows_preserves_context(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"bio": [f"row {i}" for i in range(10)]}), tmp_path, text_column="bio")
    result = read_input(inp, nrows=3)
    assert len(result.dataframe) == 3
    assert result.original_text_column == "bio"
    assert COL_TEXT in result.dataframe.columns


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
    assert len(result.dataframe) == 5


# ---------------------------------------------------------------------------
# nrows edge-cases: zero, negative, and empty input
# ---------------------------------------------------------------------------


def test_read_input_nrows_zero_returns_empty_parquet(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"text": ["a", "b", "c"]}), tmp_path, suffix=".parquet")
    result = read_input(inp, nrows=0)
    assert len(result.dataframe) == 0
    assert COL_TEXT in result.dataframe.columns


def test_read_input_nrows_zero_returns_empty_csv(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"text": ["a", "b", "c"]}), tmp_path)
    result = read_input(inp, nrows=0)
    assert len(result.dataframe) == 0
    assert COL_TEXT in result.dataframe.columns


def test_read_input_nrows_negative_returns_empty_parquet(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"text": ["a", "b", "c"]}), tmp_path, suffix=".parquet")
    result = read_input(inp, nrows=-5)
    assert len(result.dataframe) == 0
    assert COL_TEXT in result.dataframe.columns


def test_read_input_empty_parquet_returns_empty(tmp_path: Path) -> None:
    inp = _write_input(pd.DataFrame({"text": pd.Series([], dtype="object")}), tmp_path, suffix=".parquet")
    result = read_input(inp, nrows=5)
    assert len(result.dataframe) == 0
    assert COL_TEXT in result.dataframe.columns
