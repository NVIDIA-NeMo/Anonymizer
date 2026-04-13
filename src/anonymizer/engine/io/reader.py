# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerInput, infer_input_source_suffix
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.io.constants import SUPPORTED_IO_FORMATS
from anonymizer.interface.errors import AnonymizerIOError, InvalidInputError

logger = logging.getLogger("anonymizer")


def read_input(input_data: AnonymizerInput, *, nrows: int | None = None) -> pd.DataFrame:
    """Load input into a normalized dataframe with canonical internal text column.

    Args:
        input_data: Input source definition.
        nrows: Maximum rows to read.  ``None`` reads the entire file.
    """
    dataframe = _load_dataframe(input_data, nrows=nrows)
    selected_text_column = input_data.text_column
    if selected_text_column not in dataframe.columns:
        raise InvalidInputError(f"Input text column '{selected_text_column}' not found.")
    _validate_internal_column_collision(dataframe, selected_text_column=selected_text_column)
    dataframe = dataframe.rename(columns={selected_text_column: COL_TEXT})
    dataframe.attrs["original_text_column"] = selected_text_column
    return dataframe


def _validate_internal_column_collision(dataframe: pd.DataFrame, *, selected_text_column: str) -> None:
    if COL_TEXT not in dataframe.columns or selected_text_column == COL_TEXT:
        return
    raise InvalidInputError(
        f"Input contains reserved internal column {COL_TEXT!r} while text_column={selected_text_column!r}. "
        f"Either set text_column={COL_TEXT!r} or remove/rename {COL_TEXT!r} from input."
    )


def _load_dataframe(input_data: AnonymizerInput, *, nrows: int | None = None) -> pd.DataFrame:
    source_str = str(input_data.source)
    suffix = infer_input_source_suffix(source_str)
    if suffix not in SUPPORTED_IO_FORMATS:
        supported_formats = " or ".join(SUPPORTED_IO_FORMATS)
        raise InvalidInputError(f"Unsupported input format: {suffix}. Use {supported_formats}.")
    try:
        if suffix == ".csv":
            df = pd.read_csv(source_str, nrows=nrows)
        else:
            # TODO: pd.read_parquet loads the entire file; for a true partial
            # read use pyarrow.parquet.ParquetFile(...).iter_batches(...).
            df = pd.read_parquet(source_str)
            if nrows is not None:
                df = df.head(nrows)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise AnonymizerIOError(f"Failed to read input data from: {source_str}") from error
    if nrows is not None:
        logger.info(
            "👀 Preview mode: 📂 Loaded %d records from %s (column: '%s')",
            len(df),
            source_str,
            input_data.text_column,
        )
    else:
        logger.info("📂 Loaded %d records from %s (column: '%s')", len(df), source_str, input_data.text_column)
    return df
