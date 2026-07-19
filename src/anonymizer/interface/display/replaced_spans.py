# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Owns reconstruction of entity span positions in original and replaced text when detection output and the replacement map disagree or are missing."""

from __future__ import annotations

import re
from dataclasses import dataclass

from anonymizer.engine.schemas import (
    EntitySchema,
)

__all__ = [
    "build_original_entities_from_map",
    "build_replaced_entities",
    "build_replaced_entities_from_map",
]


@dataclass(frozen=True)
class _SyntheticLookupMaps:
    by_value_label: dict[tuple[str, str], str]
    by_value: dict[str, str]
    by_value_label_ci: dict[tuple[str, str], str]
    by_value_ci: dict[str, str]


def build_replaced_entities(
    original_entities: list[EntitySchema],
    replacement_map: list[dict[str, str]],
    original_text: str,
    replaced_text: str,
) -> list[EntitySchema]:
    """Compute entity positions in the replaced text by locating synthetic values directly.

    Instead of replaying cursor arithmetic (which drifts when entity values
    differ in case from the replacement map keys), we resolve each entity's
    synthetic value and find it in the replaced text by scanning forward.
    """
    by_value_label: dict[tuple[str, str], str] = {}
    by_value: dict[str, str] = {}
    by_value_label_ci: dict[tuple[str, str], str] = {}
    by_value_ci: dict[str, str] = {}
    for entry in replacement_map:
        orig = entry.get("original", "")
        label = entry.get("label", "")
        synth = entry.get("synthetic", "")
        by_value_label[(orig, label)] = synth
        by_value[orig] = synth
        by_value_label_ci[(orig.lower(), label)] = synth
        by_value_ci[orig.lower()] = synth
    lookups = _SyntheticLookupMaps(
        by_value_label=by_value_label,
        by_value=by_value,
        by_value_label_ci=by_value_label_ci,
        by_value_ci=by_value_ci,
    )

    sorted_entities = sorted(original_entities, key=lambda e: (e.start_position, e.end_position))
    replaced_entities: list[EntitySchema] = []
    original_cursor = 0
    search_from = 0

    for entity in sorted_entities:
        start = entity.start_position
        end = entity.end_position
        value = entity.value
        label = entity.label
        if start < original_cursor or end <= start or end > len(original_text):
            continue

        synthetic = _resolve_synthetic(value, label, lookups)
        original_span = original_text[start:end]

        pos = replaced_text.find(synthetic, search_from) if synthetic else -1
        if pos < 0 and synthetic != original_span:
            pos = replaced_text.find(original_span, search_from)
            if pos >= 0:
                synthetic = original_span

        if pos < 0:
            original_cursor = end
            continue

        replaced_entities.append(
            EntitySchema(
                value=replaced_text[pos : pos + len(synthetic)],
                label=label,
                start_position=pos,
                end_position=pos + len(synthetic),
            )
        )
        search_from = pos + len(synthetic)
        original_cursor = end

    return replaced_entities


def _resolve_synthetic(
    value: str,
    label: str,
    lookups: _SyntheticLookupMaps,
) -> str:
    """Look up synthetic value with exact-match first, then case-insensitive fallback."""
    result = lookups.by_value_label.get((value, label))
    if result is not None:
        return result
    result = lookups.by_value.get(value)
    if result is not None:
        return result
    result = lookups.by_value_label_ci.get((value.lower(), label))
    if result is not None:
        return result
    result = lookups.by_value_ci.get(value.lower())
    if result is not None:
        return result
    return value


def build_original_entities_from_map(
    replacement_map: list[dict[str, str]],
    original_text: str,
) -> list[EntitySchema]:
    """Build entity positions by finding original values in original text.

    Fallback when _detected_entities is empty but replacement_map exists.
    Uses case-insensitive matching to align with how the detection engine
    finds entities (e.g. "The Lantern" matching "the Lantern" in text).
    """
    result: list[EntitySchema] = []
    for entry in replacement_map:
        original = str(entry.get("original", ""))
        label = str(entry.get("label", ""))
        if not original or not label:
            continue
        for match in re.finditer(re.escape(original), original_text, flags=re.IGNORECASE):
            result.append(
                EntitySchema(
                    value=original_text[match.start() : match.end()],
                    label=label,
                    start_position=match.start(),
                    end_position=match.end(),
                )
            )
    return sorted(result, key=lambda e: (e.start_position, e.end_position))


def build_replaced_entities_from_map(
    replacement_map: list[dict[str, str]],
    replaced_text: str,
) -> list[EntitySchema]:
    """Build entity positions by finding synthetic values in replaced text.

    Fallback when _detected_entities is empty but replacement_map exists (e.g. LLM
    replace path where entity format differs).
    """
    result: list[EntitySchema] = []
    for entry in replacement_map:
        synthetic = str(entry.get("synthetic", ""))
        label = str(entry.get("label", ""))
        if not synthetic or not label:
            continue
        start = 0
        while True:
            pos = replaced_text.find(synthetic, start)
            if pos < 0:
                break
            result.append(
                EntitySchema(
                    value=synthetic,
                    label=label,
                    start_position=pos,
                    end_position=pos + len(synthetic),
                )
            )
            start = pos + len(synthetic)
    return sorted(result, key=lambda e: (e.start_position, e.end_position))
