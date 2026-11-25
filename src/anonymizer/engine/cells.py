# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reading and writing one cell of a frame: the one place that knows the forms a nested cell arrives in.

A cell may hold a pydantic model, plain Python, numpy arrays and scalars (after a Parquet or DataFrame round trip),
JSON text (after JSON storage), or a missing marker (None, NaN, ``pd.NA``, ``pd.NaT``).
"""

from __future__ import annotations

import json
from typing import Any, Protocol, TypeGuard

import pandas as pd
from pydantic import BaseModel


class ListConvertible(Protocol):
    def tolist(self) -> object: ...


def is_list_convertible(value: object) -> TypeGuard[ListConvertible]:
    """A numpy array, pandas array or Series: anything with ``tolist``."""
    return callable(getattr(value, "tolist", None))


def is_missing(value: object) -> bool:
    """None, NaN or ``pd.NA``; never an empty container."""
    return value is None or value is pd.NA or (isinstance(value, float) and value != value)


def to_python(raw: object) -> object:
    """Recursively normalize nested model/array/scalar payloads to plain Python."""
    if isinstance(raw, BaseModel):
        return to_python(raw.model_dump(mode="python"))
    if isinstance(raw, dict):
        return {key: to_python(value) for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        return [to_python(value) for value in raw]
    if is_list_convertible(raw):
        return to_python(raw.tolist())
    item = getattr(raw, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return raw


def normalize_payload(raw: Any) -> Any:
    """Normalize values that may have been JSON-stringified by parquet round-trip."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return to_python(raw)


def plain(value: Any) -> Any:
    """``value`` as plain JSON types for a returned frame; NaN and an empty object become None, at any depth."""
    value = to_python(value)
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()} or None
    if isinstance(value, list):
        return [plain(v) for v in value]
    return None if is_missing(value) or value is pd.NaT else value
