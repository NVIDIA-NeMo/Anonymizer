# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Sequence

import pandas as pd
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig

from anonymizer.engine.constants import RECORD_ID_COLUMN
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.backend import BackendRun, DataDesignerBackend
from anonymizer.engine.schemas.shared import to_python


def derive_seed_columns(
    columns: Sequence[ColumnConfigT],
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

    Non-string object cells are converted to plain Python, then JSON-encoded: numpy
    arrays (as in a frame reloaded from parquet) become lists and models their fields.
    """
    present = [c for c in seed_cols if c in df.columns]
    seed = df[present].copy()
    for col in seed.columns:
        if seed[col].dtype == "object" and seed[col].notna().any():
            sample = seed[col].dropna().iloc[0]
            if not isinstance(sample, str):
                seed[col] = seed[col].apply(
                    lambda v: json.dumps(to_python(v), default=str, ensure_ascii=False) if v is not None else v
                )
    return seed


def run_columns(
    adapter: NddAdapter,
    frame: pd.DataFrame,
    columns: list[ColumnConfigT],
    *,
    model_configs: list[ModelConfig],
    workflow_name: str,
    preview_num_records: int | None = None,
) -> tuple[BackendRun, list[str]]:
    """Seed *frame* for *columns* and run them; returns the run and the seed column names."""
    seed_cols = derive_seed_columns(columns, frame)
    run = DataDesignerBackend(adapter).run(
        select_seed_cols(frame, seed_cols),
        columns=columns,
        workflow_name=workflow_name,
        model_configs=model_configs,
        preview_num_records=preview_num_records,
    )
    return run, seed_cols
