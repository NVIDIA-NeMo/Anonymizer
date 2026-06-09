# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import Counter
from typing import Any

from anonymizer.measurement._coerce import _coerce_bool, _coerce_float


def _rewrite_record_fields(row: Any, *, columns: set[str]) -> dict[str, Any]:
    from anonymizer.engine.constants import (
        COL_ANY_HIGH_LEAKED,
        COL_LEAKAGE_MASS,
        COL_NEEDS_HUMAN_REVIEW,
        COL_NEEDS_REPAIR,
        COL_UTILITY_SCORE,
        COL_WEIGHTED_LEAKAGE_RATE,
    )

    return {
        "utility_score": _coerce_float(row.get(COL_UTILITY_SCORE)) if COL_UTILITY_SCORE in columns else None,
        "leakage_mass": _coerce_float(row.get(COL_LEAKAGE_MASS)) if COL_LEAKAGE_MASS in columns else None,
        "weighted_leakage_rate": (
            _coerce_float(row.get(COL_WEIGHTED_LEAKAGE_RATE)) if COL_WEIGHTED_LEAKAGE_RATE in columns else None
        ),
        "any_high_leaked": _coerce_bool(row.get(COL_ANY_HIGH_LEAKED)) if COL_ANY_HIGH_LEAKED in columns else None,
        "needs_human_review": (
            _coerce_bool(row.get(COL_NEEDS_HUMAN_REVIEW)) if COL_NEEDS_HUMAN_REVIEW in columns else None
        ),
        "needs_repair": _coerce_bool(row.get(COL_NEEDS_REPAIR)) if COL_NEEDS_REPAIR in columns else None,
    }


def _original_value_leak_record_fields(
    row: Any,
    *,
    columns: set[str],
    final_entities: list[dict[str, Any]],
) -> dict[str, Any]:
    output_column = _output_text_column(columns)
    if output_column is None:
        return {"original_value_leak_count": None, "original_value_leak_label_counts": {}}
    output_text = str(row.get(output_column, ""))
    leaked = [
        entity
        for entity in final_entities
        if entity.get("value") and _output_contains_original_value(output_text, str(entity.get("value")))
    ]
    return {
        "original_value_leak_count": len(leaked),
        "original_value_leak_label_counts": dict(
            sorted(Counter(str(entity.get("label") or "") for entity in leaked if entity.get("label")).items())
        ),
    }


def _output_contains_original_value(output_text: str, value: str) -> bool:
    if _needs_boundary_sensitive_leak_match(value):
        return _contains_with_alnum_boundaries(output_text, value)
    return value in output_text


def _needs_boundary_sensitive_leak_match(value: str) -> bool:
    return len(value) <= 4 or value.isdigit()


def _contains_with_alnum_boundaries(output_text: str, value: str) -> bool:
    start = 0
    while True:
        match_start = output_text.find(value, start)
        if match_start < 0:
            return False
        match_end = match_start + len(value)
        if _has_alnum_boundaries(output_text, match_start, match_end):
            return True
        start = match_start + 1


def _has_alnum_boundaries(text: str, start: int, end: int) -> bool:
    before_is_alnum = start > 0 and text[start - 1].isalnum()
    after_is_alnum = end < len(text) and text[end].isalnum()
    return not before_is_alnum and not after_is_alnum


def _output_text_column(columns: set[str]) -> str | None:
    from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_REWRITTEN_TEXT

    if COL_REPLACED_TEXT in columns:
        return COL_REPLACED_TEXT
    if COL_REWRITTEN_TEXT in columns:
        return COL_REWRITTEN_TEXT
    return None
