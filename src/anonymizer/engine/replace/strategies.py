# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd

from anonymizer.config.replace_strategies import LocalReplaceStrategy
from anonymizer.engine.detection.constants import COL_REPLACED_TEXT, COL_TEXT


@dataclass(frozen=True)
class ReplacementEntry:
    original: str
    label: str
    synthetic: str


def apply_local_replace_strategy(
    dataframe: pd.DataFrame,
    *,
    strategy: LocalReplaceStrategy,
    text_column: str = COL_TEXT,
    entities_column: str = "_detected_entities",
) -> pd.DataFrame:
    """Apply deterministic local replace strategy on detected entities."""
    output_df = dataframe.copy()
    output_df["_replacement_map"] = output_df.apply(
        lambda row: _build_local_replacement_map(
            row=row,
            strategy=strategy,
            entities_column=entities_column,
        ),
        axis=1,
    )
    output_df[COL_REPLACED_TEXT] = output_df.apply(
        lambda row: _apply_replacement_map_to_text(
            text=str(row.get(text_column, "")),
            entities=row.get(entities_column, []),
            replacement_map=row["_replacement_map"],
        ),
        axis=1,
    )
    return output_df


def apply_replacement_map(
    dataframe: pd.DataFrame,
    *,
    text_column: str = COL_TEXT,
    entities_column: str = "_detected_entities",
    replacement_map_column: str = "_replacement_map",
) -> pd.DataFrame:
    """Apply pre-generated replacement map to text."""
    output_df = dataframe.copy()
    output_df[COL_REPLACED_TEXT] = output_df.apply(
        lambda row: _apply_replacement_map_to_text(
            text=str(row.get(text_column, "")),
            entities=row.get(entities_column, []),
            replacement_map=row.get(replacement_map_column, {"replacements": []}),
        ),
        axis=1,
    )
    return output_df


def _build_local_replacement_map(
    row: pd.Series, strategy: LocalReplaceStrategy, entities_column: str
) -> dict[str, Any]:
    entities = [e for e in row.get(entities_column, []) if isinstance(e, dict)]
    if not entities:
        return {"replacements": []}
    replacements: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities:
        value = str(entity.get("value", ""))
        label = str(entity.get("label", ""))
        if not value or not label:
            continue
        key = (value, label)
        if key in seen:
            continue
        seen.add(key)
        synthetic = strategy.replace(text=value, label=label)
        replacements.append({"original": value, "label": label, "synthetic": synthetic})
    return {"replacements": replacements}


def _apply_replacement_map_to_text(text: str, entities: Any, replacement_map: Any) -> str:
    normalized_entities = [e for e in entities if isinstance(e, dict)]
    replacements = _normalize_replacements(replacement_map)
    if not normalized_entities or not replacements:
        return text

    by_value_label: dict[tuple[str, str], str] = {}
    by_value: dict[str, str] = {}
    for replacement in replacements:
        by_value_label[(replacement.original, replacement.label)] = replacement.synthetic
        by_value[replacement.original] = replacement.synthetic

    spans = sorted(
        (
            (
                int(entity.get("start_position", 0)),
                int(entity.get("end_position", 0)),
                str(entity.get("value", "")),
                str(entity.get("label", "")),
            )
            for entity in normalized_entities
            if entity.get("start_position") is not None and entity.get("end_position") is not None
        ),
        key=lambda item: item[0],
    )

    parts: list[str] = []
    cursor = 0
    for start, end, value, label in spans:
        if start < cursor or end <= start or end > len(text):
            continue
        parts.append(text[cursor:start])
        synthetic = by_value_label.get((value, label), by_value.get(value, text[start:end]))
        parts.append(synthetic)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


def _normalize_replacements(raw: Any) -> list[ReplacementEntry]:
    parsed = raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(parsed, dict):
        return []
    replacements = parsed.get("replacements", [])
    if not isinstance(replacements, list):
        return []
    normalized: list[ReplacementEntry] = []
    for replacement in replacements:
        if not isinstance(replacement, dict):
            continue
        original = str(replacement.get("original", ""))
        label = str(replacement.get("label", ""))
        synthetic = str(replacement.get("synthetic", ""))
        if not original or not label or not synthetic:
            continue
        normalized.append(ReplacementEntry(original=original, label=label, synthetic=synthetic))
    return normalized
