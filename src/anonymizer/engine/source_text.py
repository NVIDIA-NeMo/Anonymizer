# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The text a source cell contributes to the graph, as the released routes read it."""

from __future__ import annotations

import pandas as pd


def source_text(value: object) -> str:
    """Return a text cell's root text: the cell itself, or its `str()` rendering.

    The released routes never rejected a null or non-string text cell: they
    rendered it into prompts and wrote `str(text)` (`None` -> "None", NaN ->
    "nan", `pd.NA` -> "<NA>", 31337 -> "31337"). The source frame keeps the
    original cell, so the legacy lowerings see and publish exactly what they
    did; only the graph's root datum, which admits exact `str` only, receives
    this rendering.
    """
    return value if type(value) is str else str(value)


def same_source_text(returned: object, source: object) -> bool:
    """Whether a returned text cell still carries its source cell's root text.

    A DataDesigner round trip may change a null cell's missing marker (`None`,
    NaN, `pd.NA`) or a non-string cell's scalar type, as it did in the release;
    the root text of a string cell must be returned unchanged.
    """
    if source_text(returned) == source_text(source):
        return True
    return _missing(returned) and _missing(source)


def _missing(value: object) -> bool:
    return pd.api.types.is_scalar(value) and bool(pd.isna(value))
