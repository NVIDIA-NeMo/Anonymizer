# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owns the mapping between internal `COL_*` column names and the user-facing
output dataframe (rename, un-rename, and the public column allowlist per mode).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_VALID,
    COL_FINAL_ENTITIES,
    COL_JUDGE_EVALUATION,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
    COL_TYPE_FIDELITY_VALID,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["build_user_dataframe", "rename_output_columns", "unrename_output_columns"]


def rename_output_columns(df: pd.DataFrame, *, resolved_text_column: str) -> pd.DataFrame:
    """Rename internal column names to user-facing names."""
    rename_map: dict[str, str] = {}
    if COL_TEXT in df.columns:
        rename_map[COL_TEXT] = resolved_text_column
    if COL_REPLACED_TEXT in df.columns:
        rename_map[COL_REPLACED_TEXT] = f"{resolved_text_column}_replaced"
    if COL_TAGGED_TEXT in df.columns:
        rename_map[COL_TAGGED_TEXT] = f"{resolved_text_column}_with_spans"
    if COL_REWRITTEN_TEXT in df.columns:
        rename_map[COL_REWRITTEN_TEXT] = f"{resolved_text_column}_rewritten"
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def unrename_output_columns(df: pd.DataFrame, *, resolved_text_column: str) -> pd.DataFrame:
    """Reverse of :func:`rename_output_columns`.

    Converts user-facing column names (``biography``, ``biography_replaced``, …)
    back to the internal names (``__nemo_anonymizer_text_input__``, …) that the
    judges' prompt templates reference. No-op if the dataframe is already in
    internal form (``COL_TEXT`` already present).
    """
    if COL_TEXT in df.columns:
        return df
    rename_map: dict[str, str] = {}
    if resolved_text_column in df.columns:
        rename_map[resolved_text_column] = COL_TEXT
    if f"{resolved_text_column}_replaced" in df.columns:
        rename_map[f"{resolved_text_column}_replaced"] = COL_REPLACED_TEXT
    if f"{resolved_text_column}_with_spans" in df.columns:
        rename_map[f"{resolved_text_column}_with_spans"] = COL_TAGGED_TEXT
    if f"{resolved_text_column}_rewritten" in df.columns:
        rename_map[f"{resolved_text_column}_rewritten"] = COL_REWRITTEN_TEXT
    if not rename_map:
        return df
    return df.rename(columns=rename_map)


def build_user_dataframe(trace_dataframe: pd.DataFrame, *, resolved_text_column: str) -> pd.DataFrame:
    """Filter trace dataframe to the public column set for the active mode.

    Replace:     {text_col}, {text_col}_replaced, {text_col}_with_spans, final_entities,
                 optional judge verdict columns when available
    Rewrite:     {text_col}, {text_col}_rewritten, utility_score, leakage_mass, weighted_leakage_rate,
                 any_high_leaked, needs_human_review
    Detect-only: {text_col}, {text_col}_with_spans, final_entities
    """
    t = trace_dataframe
    text_col = resolved_text_column

    if f"{text_col}_rewritten" in t.columns:
        allowed = {
            text_col,
            f"{text_col}_rewritten",
            COL_UTILITY_SCORE,
            COL_LEAKAGE_MASS,
            COL_WEIGHTED_LEAKAGE_RATE,
            COL_ANY_HIGH_LEAKED,
            COL_NEEDS_HUMAN_REVIEW,
            COL_DETECTION_VALID,  # only present after evaluate()
            COL_DETECTION_INVALID_ENTITIES,  # only present after evaluate()
            COL_JUDGE_EVALUATION,  # only present after evaluate()
        }
    elif f"{text_col}_replaced" in t.columns:
        allowed = {
            text_col,
            f"{text_col}_replaced",
            f"{text_col}_with_spans",
            COL_FINAL_ENTITIES,
            COL_DETECTION_VALID,
            COL_DETECTION_INVALID_ENTITIES,
            COL_TYPE_FIDELITY_VALID,
            COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
            COL_RELATIONAL_CONSISTENCY_VALID,
            COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
            COL_ATTRIBUTE_FIDELITY_VALID,
            COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
        }
    else:
        allowed = {
            text_col,
            f"{text_col}_with_spans",
            COL_FINAL_ENTITIES,
        }

    return t[[col for col in t.columns if col in allowed]].copy()
