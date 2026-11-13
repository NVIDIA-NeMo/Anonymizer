# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logical JSON storage contracts for workflow seed handoffs.

Only engine-owned structured columns are registered here. User text that happens
to contain JSON remains text; existing DD field-level contracts remain intact.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.utils.storage_encoding import read_dataframe_storage_plan
from data_designer.config.utils.storage_schema import JSON_STORAGE_RUN_KEY, JsonStoragePlan, compile_json_storage_schema

from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTATION_ADMISSION_RESULTS,
    COL_AUGMENTATION_ATTEMPTS,
    COL_AUGMENTATION_DETECTION_RESULT,
    COL_AUGMENTED_CANDIDATES,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITIES_FOR_REPLACE,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_MERGED_ENTITIES,
    COL_RECONCILED_CANDIDATES,
    COL_REPLACEMENT_MAP,
    COL_SEED_DETECTION_RESULT,
    COL_SEED_ENTITIES,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    COL_VALIDATION_SKELETON,
)
from anonymizer.engine.schemas.shared import to_python

if TYPE_CHECKING:
    from data_designer.config.column_types import ColumnConfigT


JSON_STORAGE_COLUMNS = frozenset(
    {
        COL_ADMISSION_RESULTS,
        COL_ADMITTED_CANDIDATES,
        COL_AUGMENTATION_ADMISSION_RESULTS,
        COL_AUGMENTATION_ATTEMPTS,
        COL_AUGMENTATION_DETECTION_RESULT,
        COL_AUGMENTED_CANDIDATES,
        COL_AUGMENTED_ENTITIES,
        COL_DETECTED_ENTITIES,
        COL_ENTITIES_BY_VALUE,
        COL_ENTITIES_FOR_REPLACE,
        COL_FINAL_ENTITIES,
        COL_LATENT_ENTITIES,
        COL_MERGED_ENTITIES,
        COL_RECONCILED_CANDIDATES,
        COL_REPLACEMENT_MAP,
        COL_SEED_DETECTION_RESULT,
        COL_SEED_ENTITIES,
        COL_SEED_VALIDATION_CANDIDATES,
        COL_VALIDATION_CANDIDATES,
        COL_VALIDATION_DECISIONS,
        COL_VALIDATION_INVENTORIES,
        COL_VALIDATION_SKELETON,
    }
)


def prepare_json_storage_seed(
    dataframe: pd.DataFrame, *, json_columns: Iterable[str] = JSON_STORAGE_COLUMNS
) -> pd.DataFrame:
    """Carry explicit DD storage metadata without mutating input logical values.

    Each planned column is rebuilt as an object column of plain Python values.
    pandas would otherwise reuse an Arrow-backed column's old nested dtype on
    write, and DD's JSON encoder rejects the numpy arrays, numpy scalars and NaN
    that a batch read can return. No guessed parsing or string conversion is
    used, and unrelated columns are left untouched.
    """
    frame = dataframe.copy()
    plan = read_dataframe_storage_plan(frame)
    columns = dict(plan.columns)
    for name in json_columns:
        if name in frame.columns:
            columns.setdefault(name, [[]])
    for name in columns:
        values = [None if _is_missing(value) else to_python(value) for value in frame[name].tolist()]
        frame[name] = pd.Series(values, index=frame.index, dtype=object)
    if columns:
        frame.attrs[JSON_STORAGE_RUN_KEY] = JsonStoragePlan(columns=columns).model_dump()
    return frame


def prepare_json_storage_column(column: ColumnConfigT) -> ColumnConfigT:
    """Declare storage for known named outputs, never implicitly for side effects."""
    if isinstance(column, LLMStructuredColumnConfig):
        schema = column.output_format
        if not isinstance(schema, dict):
            schema = schema.model_json_schema()
        _, paths = compile_json_storage_schema(schema)
        if paths:
            return column
    if column.name in JSON_STORAGE_COLUMNS and column.storage_encoding is None:
        return column.model_copy(update={"storage_encoding": "json"})
    return column


def _is_missing(value: object) -> bool:
    return value is None or value is pd.NA or (isinstance(value, float) and value != value)
