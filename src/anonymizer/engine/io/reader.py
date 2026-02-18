# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anonymizer.config.anonymizer_config import AnonymizerInput, InputSourceType
from anonymizer.interface.errors import InvalidInputError


def read_input(input_data: pd.DataFrame | AnonymizerInput) -> pd.DataFrame:
    """Load input into a normalized dataframe with canonical `text` column."""
    dataframe = _load_dataframe(input_data=input_data)
    if "text" not in dataframe.columns and isinstance(input_data, AnonymizerInput):
        if input_data.text_column not in dataframe.columns:
            raise InvalidInputError(f"Input text column '{input_data.text_column}' not found.")
        dataframe = dataframe.rename(columns={input_data.text_column: "text"})
        dataframe.attrs["original_text_column"] = input_data.text_column
    elif "text" in dataframe.columns:
        dataframe.attrs["original_text_column"] = "text"
    return dataframe


def _load_dataframe(input_data: pd.DataFrame | AnonymizerInput) -> pd.DataFrame:
    if isinstance(input_data, pd.DataFrame):
        return input_data.copy()
    source = Path(str(input_data.source))
    if input_data.source_type == InputSourceType.parquet:
        return pd.read_parquet(source)
    if input_data.source_type == InputSourceType.csv:
        return pd.read_csv(source)
    if input_data.source_type == InputSourceType.json:
        return pd.read_json(source)
    raise InvalidInputError(f"Unsupported input source_type: {input_data.source_type}")
