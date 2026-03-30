# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pandas as pd
from data_designer.config.column_types import ColumnConfigT

from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN


def derive_seed_columns(
    columns: list[ColumnConfigT],
    df: pd.DataFrame,
) -> list[str]:
    """Compute the minimal seed column set from column config dependencies.

    Columns produced by a config in *columns* (via ``name`` or
    ``side_effect_columns``) are internal to the workflow and excluded.
    Only columns that must come from the input dataframe are returned,
    preserving their original dataframe ordering.
    """
    produced: set[str] = set()
    for col in columns:
        produced.add(col.name)
        produced.update(col.side_effect_columns)

    required: set[str] = set()
    for col in columns:
        required.update(col.required_columns)

    external = (required - produced) | {RECORD_ID_COLUMN}
    return [c for c in df.columns if c in external]


def select_seed_cols(df: pd.DataFrame, seed_cols: list[str]) -> pd.DataFrame:
    """Select *seed_cols* from *df* for an adapter call.

    Non-string object columns are JSON-stringified so they survive the
    parquet round-trip through the NDD adapter.
    """
    present = [c for c in seed_cols if c in df.columns]
    seed = df[present].copy()
    for col in seed.columns:
        if seed[col].dtype == "object" and seed[col].notna().any():
            sample = seed[col].dropna().iloc[0]
            if not isinstance(sample, str):
                seed[col] = seed[col].apply(
                    lambda v: json.dumps(v, default=str, ensure_ascii=False) if v is not None else v
                )
    return seed
