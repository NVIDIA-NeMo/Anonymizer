# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for partitioning DataFrames by a row predicate and recombining them."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

ROW_ORDER_COL = "_anonymizer_row_order"


def split_rows(
    df: pd.DataFrame,
    *,
    column: str,
    predicate: Callable[[Any], bool],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition *df* into (matching, non-matching) rows with row-order tracking.

    A ``_anonymizer_row_order`` column is added so that :func:`merge_and_reorder`
    can restore the original ordering after the two partitions are processed
    independently.
    """
    working = df.copy()
    working[ROW_ORDER_COL] = range(len(working))
    mask = working[column].apply(predicate)
    return working[mask].copy(), working[~mask].copy()


def merge_and_reorder(
    *parts: pd.DataFrame,
    attrs: dict,
) -> pd.DataFrame:
    """Concat partitions, restore original row order, and propagate *attrs*."""
    if not parts:
        raise ValueError("merge_and_reorder requires at least one partition")
    combined = (
        pd.concat(list(parts), ignore_index=True)
        .sort_values(ROW_ORDER_COL)
        .drop(columns=[ROW_ORDER_COL])
        .reset_index(drop=True)
    )
    combined.attrs = {**attrs}
    return combined
