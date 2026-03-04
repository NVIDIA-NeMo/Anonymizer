# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from anonymizer.config.replace_strategies import LocalReplaceMethod
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.schemas import EntitiesSchema


@dataclass(frozen=True)
class ReplacementEntry:
    original: str
    label: str
    synthetic: str


def apply_local_replace_strategy(
    dataframe: pd.DataFrame,
    *,
    strategy: LocalReplaceMethod,
    text_column: str = COL_TEXT,
    entities_column: str = COL_FINAL_ENTITIES,
) -> pd.DataFrame:
    """Apply deterministic local replace strategy on detected entities."""
    output_df = dataframe.copy()
    output_df[COL_REPLACEMENT_MAP] = output_df.apply(
        lambda row: _build_local_replacement_map(
            entities=EntitiesSchema.from_raw(row.get(entities_column, [])),
            strategy=strategy,
        ),
        axis=1,
    )
    output_df[COL_REPLACED_TEXT] = output_df.apply(
        lambda row: _apply_replacement_map_to_text(
            text=str(row.get(text_column, "")),
            entities=EntitiesSchema.from_raw(row.get(entities_column, [])),
            replacements=_parse_replacements(row[COL_REPLACEMENT_MAP]),
        ),
        axis=1,
    )
    return output_df


def apply_replacement_map(
    dataframe: pd.DataFrame,
    *,
    text_column: str = COL_TEXT,
    entities_column: str = COL_FINAL_ENTITIES,
    replacement_map_column: str = COL_REPLACEMENT_MAP,
) -> pd.DataFrame:
    """Apply pre-generated replacement map to text."""
    output_df = dataframe.copy()
    output_df[COL_REPLACED_TEXT] = output_df.apply(
        lambda row: _apply_replacement_map_to_text(
            text=str(row.get(text_column, "")),
            entities=EntitiesSchema.from_raw(row.get(entities_column, [])),
            replacements=_parse_replacements(row.get(replacement_map_column, {"replacements": []})),
        ),
        axis=1,
    )
    return output_df


def _build_local_replacement_map(
    entities: EntitiesSchema, strategy: LocalReplaceMethod
) -> dict[str, list[dict[str, str]]]:
    if not entities.entities:
        return {"replacements": []}
    replacements: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entity in entities.entities:
        if not entity.value or not entity.label:
            continue
        key = (entity.value, entity.label)
        if key in seen:
            continue
        seen.add(key)
        synthetic = strategy.replace(text=entity.value, label=entity.label)
        replacements.append({"original": entity.value, "label": entity.label, "synthetic": synthetic})
    return {"replacements": replacements}


def _apply_replacement_map_to_text(text: str, entities: EntitiesSchema, replacements: list[ReplacementEntry]) -> str:
    if not entities.entities or not replacements:
        return text

    by_value_label: dict[tuple[str, str], str] = {}
    by_value: dict[str, str] = {}
    for replacement in replacements:
        by_value_label[(replacement.original, replacement.label)] = replacement.synthetic
        by_value[replacement.original] = replacement.synthetic

    spans = sorted(
        ((entity.start_position, entity.end_position, entity.value, entity.label) for entity in entities.entities),
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


def _parse_replacements(raw: str | dict | object) -> list[ReplacementEntry]:
    """Parse raw replacement map (JSON string or dict) into typed entries."""
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
