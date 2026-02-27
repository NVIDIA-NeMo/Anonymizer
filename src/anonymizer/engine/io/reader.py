# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerInput
from anonymizer.engine.constants import COL_TEXT
from anonymizer.interface.errors import InvalidInputError


def read_input(input_data: AnonymizerInput) -> pd.DataFrame:
    """Load input into a normalized dataframe with canonical internal text column."""
    dataframe = _load_dataframe(input_data)
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


def _load_dataframe(input_data: AnonymizerInput) -> pd.DataFrame:
    source = Path(str(input_data.source))
    if not source.exists():
        raise InvalidInputError(f"Input path does not exist: {source}")
    if not source.is_file():
        raise InvalidInputError(f"Input path is not a file: {source}")
    suffix = source.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(source)
        if suffix == ".parquet":
            return pd.read_parquet(source)
    except (OSError, pd.errors.ParserError, ValueError) as error:
        raise InvalidInputError(f"Failed to read input data from path: {source}") from error
    raise InvalidInputError(f"Unsupported input format: {suffix}. Use .csv or .parquet.")
