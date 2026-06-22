#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small aggregation helpers shared by measurement analysis tools."""

from __future__ import annotations

from typing import cast

import pandas as pd


def none_if_nan(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def median_or_none(dataframe: pd.DataFrame, column: str) -> float | None:
    if column not in dataframe.columns:
        return None
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.median())


def sum_int_or_zero(dataframe: pd.DataFrame, column: str) -> int:
    return int(sum_or_zero(dataframe, column))


def sum_or_zero(dataframe: pd.DataFrame, column: str) -> float:
    value = sum_or_none(dataframe, column)
    return 0.0 if value is None else value


def sum_or_none(dataframe: pd.DataFrame, column: str) -> float | None:
    if column not in dataframe.columns:
        return None
    values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
    if values.empty:
        return None
    return float(values.sum())


def optional_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(cast(float, value))
