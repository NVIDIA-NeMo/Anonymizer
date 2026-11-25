# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shape of nested values in the frames a result returns, and its reverse for frames handed back to ``evaluate``.

At the output boundary every nested cell the library produces follows one rule: plain JSON types (dict, list, str,
int, float, bool, None); keys that are names, never data; None for an empty object and for NaN. Two values were keyed
by data and become lists of records: the entity classification (``{label: classification}``) and the skipped-span
label counts inside the replacement application (``{label: count}``). Inside the engine, and in prompts, the values
keep their working form; :func:`from_output_shape` restores it when a returned frame comes back.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from anonymizer.engine import constants
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    COL_DETECTION_INVALID_ENTITIES,
    COL_ENTITY_CLASSIFICATION,
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_MISSED_ENTITIES,
    COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
    COL_REPLACEMENT_APPLICATION,
    COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
)

# The user-facing nested columns. The internal columns are named by constant below, so a caller's own input column is
# never reshaped, whatever its name.
USER_NESTED_COLUMNS = frozenset(
    {
        COL_FINAL_ENTITIES,
        COL_JUDGE_EVALUATION,
        COL_MISSED_ENTITIES,
        COL_DETECTION_INVALID_ENTITIES,
        COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
        COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
        COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    }
)
_LABEL_COUNTS = "skipped_span_label_counts"


def _missing(value: object) -> bool:
    return value is None or value is pd.NA or value is pd.NaT or (isinstance(value, float) and value != value)


def plain(value: Any) -> Any:
    """``value`` as plain JSON types; NaN and an empty object become None, at any depth."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()} or None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [plain(v) for v in (value.tolist() if isinstance(value, np.ndarray) else value)]
    if isinstance(value, np.generic):
        value = value.item()
    if _missing(value):
        return None
    return value


def _records(mapping: object, key: str, value: str) -> list[dict[str, Any]] | None:
    if isinstance(mapping, str):
        mapping = json.loads(mapping)
    if not isinstance(mapping, dict):
        return None
    return [{key: str(k), value: plain(v)} for k, v in mapping.items()]


def _shape(column: str, cell: Any) -> Any:
    if column == COL_ENTITY_CLASSIFICATION and isinstance(cell, dict):
        return _records(cell, "label", "classification") or None
    if column == COL_REPLACEMENT_APPLICATION and isinstance(cell, dict) and _LABEL_COUNTS in cell:
        return plain({**cell, _LABEL_COUNTS: None}) | {
            _LABEL_COUNTS: _records(cell[_LABEL_COUNTS], "label", "count") or None
        }
    return plain(cell)


_INTERNAL_COLUMNS = frozenset(
    value
    for name, value in vars(constants).items()
    if name.startswith("COL_") and isinstance(value, str) and value.startswith("_")
)
_ROUND = "__iteration_"  # combined Rewrite suffixes a repair round's columns, as in ``_privacy_qa__iteration_0``


def _shaped(column: object) -> bool:
    if not isinstance(column, str):
        return False
    return column in USER_NESTED_COLUMNS or column.split(_ROUND, 1)[0] in _INTERNAL_COLUMNS


def to_output_shape(frame: pd.DataFrame) -> pd.DataFrame:
    """A copy of ``frame`` whose produced object columns follow the nested-value rule."""
    out = frame.copy()
    for column in out.columns:
        if _shaped(column) and out[column].dtype == object:
            out[column] = pd.Series([_shape(column, v) for v in out[column]], index=out.index, dtype=object)
    return out


def from_output_shape(frame: pd.DataFrame) -> pd.DataFrame:
    """The working form of the values :func:`to_output_shape` changed by key; other cells are already readable."""
    out = frame.copy()
    if COL_ENTITY_CLASSIFICATION in out.columns:
        out[COL_ENTITY_CLASSIFICATION] = pd.Series(
            [_mapping(v, "label", "classification") for v in out[COL_ENTITY_CLASSIFICATION]],
            index=out.index,
            dtype=object,
        )
    if COL_REPLACEMENT_APPLICATION in out.columns:
        out[COL_REPLACEMENT_APPLICATION] = pd.Series(
            [_application(v) for v in out[COL_REPLACEMENT_APPLICATION]], index=out.index, dtype=object
        )
    return out


def _mapping(cell: Any, key: str, value: str) -> Any:
    if isinstance(cell, (list, np.ndarray)):
        return {item[key]: item[value] for item in cell}
    return cell


def _application(cell: Any) -> Any:
    if isinstance(cell, dict) and isinstance(cell.get(_LABEL_COUNTS), (list, np.ndarray, type(None))):
        counts = _mapping(cell[_LABEL_COUNTS], "label", "count") or {}
        return {**cell, _LABEL_COUNTS: json.dumps(counts, sort_keys=True)}
    return cell
