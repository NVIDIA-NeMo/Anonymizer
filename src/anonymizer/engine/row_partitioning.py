# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for partitioning DataFrames by a row predicate and recombining them."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pandas as pd

from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN
from anonymizer.interface.errors import AnonymizerWorkflowError

warnings.filterwarnings(
    "ignore",
    message=".*DataFrame concatenation with empty or all-NA entries.*",
    category=FutureWarning,
)

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


def merge_and_reorder(*parts: pd.DataFrame) -> pd.DataFrame:
    """Concat partitions and restore the original row order."""
    if not parts:
        raise ValueError("merge_and_reorder requires at least one partition")
    return (
        pd.concat(list(parts), ignore_index=True)
        .sort_values(ROW_ORDER_COL)
        .drop(columns=[ROW_ORDER_COL])
        .reset_index(drop=True)
    )


def resolve_source_positions(
    source: pd.DataFrame, returned: pd.DataFrame, positions: tuple[int, ...] | None
) -> tuple[int, ...]:
    """Validate adapter positions; None denotes an unassociated legacy result.

    Native adapter results always supply a tuple, including empty output. Legacy
    injected/fake adapters retain positional alignment at equal length or unique
    record-ID alignment after drops. Neither form establishes graph identity.
    """
    if positions is None:
        if len(source) == len(returned):
            positions = tuple(range(len(source)))
        elif RECORD_ID_COLUMN in source and RECORD_ID_COLUMN in returned:
            original = source[RECORD_ID_COLUMN].astype(str).tolist()
            surviving = returned[RECORD_ID_COLUMN].astype(str).tolist()
            if len(set(original)) != len(original) or any(value not in original for value in surviving):
                raise AnonymizerWorkflowError("Unassociated workflow result has ambiguous source correspondence")
            lookup = {value: index for index, value in enumerate(original)}
            positions = tuple(lookup[value] for value in surviving)
        else:
            raise AnonymizerWorkflowError("Workflow result is missing source correspondence")
    if (
        type(positions) is not tuple
        or len(positions) != len(returned)
        or any(type(position) is not int or not 0 <= position < len(source) for position in positions)
        or len(set(positions)) != len(positions)
    ):
        raise AnonymizerWorkflowError("Workflow result has invalid source correspondence")
    return positions
