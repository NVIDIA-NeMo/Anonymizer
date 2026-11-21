# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for partitioning DataFrames by a row predicate and recombining them, and for placing returned rows
against their source rows by record id."""

from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Callable
from typing import Any

import pandas as pd

from anonymizer.engine.errors import AnonymizerWorkflowError
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, NddAdapter
from anonymizer.engine.ndd.failed_record import FailedRecord
from anonymizer.engine.protection_policy import FailureReason

warnings.filterwarnings(
    "ignore",
    message=".*DataFrame concatenation with empty or all-NA entries.*",
    category=FutureWarning,
)

ROW_ORDER_COL = "_anonymizer_row_order"


def cell_text(cell: object) -> str:
    """A source cell as the Replace check reads it: ``str(cell)``, as 0.4.0 (a number, a boolean or a blank cell
    becomes its string form)."""
    return str(cell)


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
    # An empty frame yields an object-typed mask, which pandas would read as a column selector.
    mask = working[column].apply(predicate).astype(bool)
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
    """Validate adapter positions; None denotes a result that reports no positions.

    Native adapter results always supply a tuple, including empty output. A result
    without positions corresponds only through unique record IDs; equal row counts
    or equal text do not establish correspondence (``uncorrelated_output``).
    """
    if positions is None:
        if returned.empty:
            positions = ()
        elif RECORD_ID_COLUMN in source and RECORD_ID_COLUMN in returned:
            original = source[RECORD_ID_COLUMN].astype(str).tolist()
            surviving = returned[RECORD_ID_COLUMN].astype(str).tolist()
            lookup = {value: index for index, value in enumerate(original)}
            if len(lookup) != len(original) or any(value not in lookup for value in surviving):
                raise AnonymizerWorkflowError("Unassociated workflow result has ambiguous source correspondence")
            positions = tuple(lookup[value] for value in surviving)
        else:
            raise AnonymizerWorkflowError("Workflow result has no source correspondence (uncorrelated_output)")
    if (
        type(positions) is not tuple
        or len(positions) != len(returned)
        or any(type(position) is not int or not 0 <= position < len(source) for position in positions)
        or len(set(positions)) != len(positions)
    ):
        raise AnonymizerWorkflowError("Workflow result has invalid source correspondence")
    return positions


def try_resolve_source_positions(
    source: pd.DataFrame, returned: pd.DataFrame, positions: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    """``resolve_source_positions``, or ``None`` when the result cannot be aligned to its source."""
    try:
        return resolve_source_positions(source, returned, positions)
    except AnonymizerWorkflowError:
        return None


def restore_diagnostic_columns(
    target: pd.DataFrame,
    returned: pd.DataFrame,
    defaults: dict[str, object],
    *,
    source_positions: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Restore diagnostic outputs to their sources, defaulting dropped rows.

    Keep input columns, order and index. Complete outputs retain their dtypes;
    partial outputs use the caller's defaults, including literal None values.
    """
    positions = resolve_source_positions(target, returned, source_positions)
    result = target.copy()
    for column, default in defaults.items():
        if len(positions) == len(target) and column in returned:
            order = sorted(range(len(positions)), key=positions.__getitem__)
            result[column] = returned[column].iloc[order].array
        else:
            result[column] = default
            if column in returned:
                column_index = result.columns.get_loc(column)
                for row_index, source_index in enumerate(positions):
                    result.iat[source_index, column_index] = returned[column].iloc[row_index]
    return result


def with_record_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """``frame`` with an ``_anonymizer_record_id`` per row: its own column when present (``TextRecordsInput`` ids),
    else the adapter's uuid5 of the row, so detection, every later stage and every failure name the same record."""
    if RECORD_ID_COLUMN in frame.columns:
        return frame
    frame = frame.copy()
    ids = [NddAdapter.compute_record_id(row) for _, row in frame.iterrows()]
    frame[RECORD_ID_COLUMN] = pd.Series(ids, index=frame.index, dtype=object)
    return frame


def record_ids(frame: pd.DataFrame) -> tuple[str, ...] | None:
    """The record id of each row, ``None`` when the frame carries none."""
    if RECORD_ID_COLUMN not in frame.columns:
        return None
    return tuple(str(i) for i in frame[RECORD_ID_COLUMN])


def source_positions(source: pd.DataFrame, returned: pd.DataFrame) -> tuple[int | None, ...]:
    """The ``source`` position of each ``returned`` row, ``None`` for a row that cannot be placed.

    A row is placed only by its record id, and only when that id occurs exactly once in ``source`` and once in
    ``returned``. Row order and text are never read: records may share their text, and a judge may reorder rows.
    A row without an id (missing column or a null), with an unknown id or with a repeated id cannot be placed. Ids
    compare as text, so a judge that retypes an id onto another record's text (``"1"`` and ``1.0`` as ``"1.0"``)
    relabels the row, which the id contract trusts."""
    if RECORD_ID_COLUMN not in source.columns or RECORD_ID_COLUMN not in returned.columns:
        return (None,) * len(returned)
    source_ids, ids = (tuple(None if pd.isna(i) else str(i) for i in f[RECORD_ID_COLUMN]) for f in (source, returned))
    in_source, in_returned = Counter(source_ids), Counter(ids)
    where = {i: p for p, i in enumerate(source_ids) if i is not None and in_source[i] == 1}
    return tuple(where.get(i) if i is not None and in_returned[i] == 1 else None for i in ids)


def placed_output(
    source: pd.DataFrame,
    output: pd.DataFrame,
    detection_failures: list[FailedRecord],
    runner_failures: list[FailedRecord],
    step: str,
) -> tuple[pd.DataFrame, list[FailedRecord]]:
    """An override runner's ``output``, checked at the seam: only rows placed by a unique input record id are
    released, and each ``source`` record that no placed row answers and no failure names fails at ``step`` with
    ``FailureReason.UNPLACED_OUTPUT``. Returns the released rows and ``runner_failures`` plus those failures."""
    positions = source_positions(source, output)
    placed = [i for i, p in enumerate(positions) if p is not None]
    ids = record_ids(source) or ()
    answered = {ids[p] for p in positions if p is not None}
    named = {f.record_id for f in (*detection_failures, *runner_failures)}
    unplaced = [
        FailedRecord(i, step, FailureReason.UNPLACED_OUTPUT.value) for i in ids if i not in answered and i not in named
    ]
    released = output if len(placed) == len(output) else output.iloc[placed].reset_index(drop=True)
    return released, [*runner_failures, *unplaced]
