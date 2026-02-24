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
    if suffix == ".csv":
        dataframe.to_csv(path, index=False)
        return path
    if suffix == ".parquet":
        dataframe.to_parquet(path, index=False)
        return path
    raise InvalidInputError(f"Unsupported output format for path: {path}")
