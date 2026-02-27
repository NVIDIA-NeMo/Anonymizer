# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pandas as pd

from anonymizer.interface.errors import InvalidInputError


def write_output(dataframe: pd.DataFrame, output_path: str | Path) -> Path:
    """Write dataframe to csv/parquet based on output suffix."""
    path = Path(output_path)
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".parquet"}:
        raise InvalidInputError(f"Unsupported output format for path: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if suffix == ".csv":
            dataframe.to_csv(path, index=False)
            return path
        dataframe.to_parquet(path, index=False)
        return path
    except (OSError, ValueError) as error:
        raise InvalidInputError(f"Failed to write output data to path: {path}") from error
