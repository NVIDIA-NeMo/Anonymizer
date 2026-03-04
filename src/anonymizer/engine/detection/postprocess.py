# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from hashlib import sha1

logger = logging.getLogger(__name__)

VALIDATION_CONTEXT_WINDOW = 32


@dataclass(frozen=True)
class EntitySpan:
    """Canonical standoff entity representation."""

    entity_id: str
    value: str
    label: str
    start_position: int
    end_position: int
    score: float
    source: str

    def as_dict(self) -> dict[str, str | int | float]:
        return {
            "id": self.entity_id,
            "value": self.value,
            "label": self.label,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "score": self.score,
            "source": self.source,
        }


class TagNotation(str, Enum):
    xml = "xml"
    bracket = "bracket"
    paren = "paren"
    sentinel = "sentinel"


def parse_raw_entities(raw_response: str, text: str) -> list[EntitySpan]:
    """Parse hosted GLiNER JSON response into canonical standoff entities.

    Threshold filtering is handled server-side by the GLiNER API; this
    function only validates structural integrity of the returned spans.
    """
    payload = _safe_json_loads(raw_response)
    raw_entities = payload.get("entities", [])
    if not isinstance(raw_entities, list):
        return []

    parsed: list[EntitySpan] = []
    for idx, raw_entity in enumerate(raw_entities):
        if not isinstance(raw_entity, dict):
            continue
        value = str(raw_entity.get("text", "")).strip()
        label = str(raw_entity.get("label", "")).strip()
        start = _coerce_int(raw_entity.get("start"))
        end = _coerce_int(raw_entity.get("end"))
        score = _coerce_float(raw_entity.get("score"), default=0.0)
        if not value or not label:
            continue
        if start is None or end is None or start < 0 or end <= start or end > len(text):
            continue
        entity_id = _build_entity_id(index=idx, value=value, label=label, start=start, end=end)
        parsed.append(
            EntitySpan(
                entity_id=entity_id,
                value=value,
                label=label,
                start_position=start,
                end_position=end,
                score=score,
                source="detector",
            )
        )
    return resolve_overlaps(parsed)


def build_validation_candidates(text: str, entities: list[EntitySpan]) -> list[dict[str, str]]:
    """Build per-entity validation payload with context windows."""
    # TODO(lramaswamy): make validation context window configurable from AnonymizerConfig.
    candidates: list[dict[str, str]] = []
    for entity in entities:
        before_start = max(0, entity.start_position - VALIDATION_CONTEXT_WINDOW)
        after_end = min(len(text), entity.end_position + VALIDATION_CONTEXT_WINDOW)
        candidates.append(
            {
                "id": entity.entity_id,
                "value": entity.value,
                "label": entity.label,
                "context_before": text[before_start : entity.start_position],
                "context_after": text[entity.end_position : after_end],
            }
        )
    return candidates


def apply_validation_decisions(entities: list[EntitySpan], validation_output: dict | str) -> list[EntitySpan]:
    """Apply keep/reclass/drop validation decisions to canonical entities.

    - keep: retain entity with its original label
    - reclass: retain entity but change its label to ``proposed_label``
    - drop: remove entity entirely

    Entities without a matching decision are kept unchanged.
    """
    payload = _safe_json_loads(validation_output) if isinstance(validation_output, str) else validation_output
    decisions = payload.get("decisions", []) if isinstance(payload, dict) else []
    if not isinstance(decisions, list):
        return entities

    decision_map: dict[str, dict[str, str]] = {}
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        entity_id = str(decision.get("id", "")).strip()
        result = str(decision.get("decision", "")).strip().lower()
        if not entity_id or result not in {"keep", "reclass", "drop"}:
            continue
        decision_map[entity_id] = {
            "decision": result,
            "proposed_label": str(decision.get("proposed_label", "")).strip(),
        }

    validated: list[EntitySpan] = []
    for entity in entities:
        entry = decision_map.get(entity.entity_id)
        if entry is None:
            validated.append(entity)
            continue
        if entry["decision"] == "drop":
            continue
        if entry["decision"] == "reclass" and entry["proposed_label"]:
            validated.append(
                EntitySpan(
                    entity_id=entity.entity_id,
                    value=entity.value,
                    label=entry["proposed_label"],
                    start_position=entity.start_position,
                    end_position=entity.end_position,
                    score=entity.score,
                    source=entity.source,
                )
            )
        else:
            validated.append(entity)
    return validated


def apply_augmented_entities(
    text: str,
    entities: list[EntitySpan],
    augmented_output: dict | str,
) -> list[EntitySpan]:
    """Add augmented entities, split full names, and resolve overlaps on merged set."""
    payload = _safe_json_loads(augmented_output) if isinstance(augmented_output, str) else augmented_output
    augmented = payload.get("entities", []) if isinstance(payload, dict) else []
    if not isinstance(augmented, list):
        augmented = []

    merged = list(entities)
    for idx, suggestion in enumerate(augmented):
        if not isinstance(suggestion, dict):
            continue
        value = str(suggestion.get("value", "")).strip()
        label = str(suggestion.get("label", "")).strip()
        if not value or not label:
            continue
        for start, end in _find_all_occurrences(text=text, needle=value):
            entity_id = _build_entity_id(
                index=idx,
                value=value,
                label=label,
                start=start,
                end=end,
                source="augmenter",
            )
            merged.append(
                EntitySpan(
                    entity_id=entity_id,
                    value=value,
                    label=label,
                    start_position=start,
                    end_position=end,
                    score=1.0,
                    source="augmenter",
                )
            )

    merged = _split_full_names(text=text, entities=merged)
    return resolve_overlaps(merged)


def _split_full_names(text: str, entities: list[EntitySpan]) -> list[EntitySpan]:
    """Split ``full_name`` entities into first/middle/last name parts.

    When a ``full_name`` span like "John Smith" is detected, this adds
    separate ``first_name``/``last_name``/``middle_name`` entities for
    each part so that standalone occurrences elsewhere in the text are
    also caught.
    """
    existing_values: set[str] = {entity.value.lower() for entity in entities}
    extra: list[EntitySpan] = []

    for entity in entities:
        if entity.label != "full_name":
            continue
        parts = entity.value.split()
        if len(parts) < 2:
            continue
        for idx, part in enumerate(parts):
            if len(part) <= 1 or part.lower() in existing_values:
                continue
            if idx == 0:
                part_label = "first_name"
            elif idx == len(parts) - 1:
                part_label = "last_name"
            else:
                part_label = "middle_name"
            for start, end in _find_all_occurrences(text=text, needle=part):
                entity_id = _build_entity_id(
                    index=idx,
                    value=part,
                    label=part_label,
                    start=start,
                    end=end,
                    source="name_split",
                )
                extra.append(
                    EntitySpan(
                        entity_id=entity_id,
                        value=part,
                        label=part_label,
                        start_position=start,
                        end_position=end,
                        score=entity.score,
                        source="name_split",
                    )
                )
            existing_values.add(part.lower())

    return [*entities, *extra]


def resolve_overlaps(entities: list[EntitySpan]) -> list[EntitySpan]:
    """Resolve span conflicts by preferring longer spans, then earlier starts."""
    sorted_entities = sorted(
        entities,
        key=lambda item: (
            -(item.end_position - item.start_position),
            item.start_position,
            item.end_position,
            item.label,
        ),
    )
    accepted: list[EntitySpan] = []
    for candidate in sorted_entities:
        if any(_spans_overlap(candidate, existing) for existing in accepted):
            continue
        accepted.append(candidate)
    return sorted(accepted, key=lambda item: (item.start_position, item.end_position, item.label))


def build_tagged_text(text: str, entities: list[EntitySpan]) -> str:
    """Render human-readable tagged text for downstream LLM prompts."""
    if not entities:
        return text
    notation = _choose_tag_notation(text)
    cursor = 0
    parts: list[str] = []
    for entity in sorted(entities, key=lambda item: (item.start_position, item.end_position)):
        if entity.start_position < cursor:
            continue
        parts.append(text[cursor : entity.start_position])
        parts.append(
            _format_entity_tag(
                value=text[entity.start_position : entity.end_position],
                label=entity.label,
                notation=notation,
            )
        )
        cursor = entity.end_position
    parts.append(text[cursor:])
    return "".join(parts)


def get_tag_notation(text: str) -> str:
    """Return the tag notation name chosen for *text* (xml, bracket, paren, sentinel)."""
    return _choose_tag_notation(text).value


def expand_entity_occurrences(text: str, entities: list[EntitySpan]) -> list[EntitySpan]:
    """Expand each validated entity to ALL its occurrences in the text.

    After validation, entities only have the positions where the detector
    originally found them. This function finds every word-boundary-matched
    occurrence of each unique entity value in the text, creating new spans
    for positions not already covered. Overlaps are resolved by preferring
    longer spans.
    """
    entity_map: dict[str, str] = {}
    for entity in entities:
        key = entity.value.lower()
        if key not in entity_map:
            entity_map[key] = entity.label

    expanded: list[EntitySpan] = []
    for idx, (key, label) in enumerate(entity_map.items()):
        original_value = next(e.value for e in entities if e.value.lower() == key)
        for start, end in _find_all_occurrences(text=text, needle=original_value):
            entity_id = _build_entity_id(
                index=idx,
                value=text[start:end],
                label=label,
                start=start,
                end=end,
                source="propagation",
            )
            expanded.append(
                EntitySpan(
                    entity_id=entity_id,
                    value=text[start:end],
                    label=label,
                    start_position=start,
                    end_position=end,
                    score=1.0,
                    source="propagation",
                )
            )

    all_entities = [*entities, *expanded]
    return resolve_overlaps(all_entities)


def group_entities_by_value(entities: list[EntitySpan]) -> list[dict[str, str | list[str]]]:
    """Group entities by normalized value for consistent replacement mapping."""
    grouped: dict[str, set[str]] = {}
    for entity in entities:
        key = entity.value
        grouped.setdefault(key, set()).add(entity.label)
    return [
        {"value": value, "labels": sorted(labels)}
        for value, labels in sorted(grouped.items(), key=lambda item: item[0])
    ]


def _safe_json_loads(value: dict | str) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str):
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError as exc:
        logger.warning(
            "Failed to parse JSON in postprocessing pipeline (error=%s, length=%d)",
            exc.msg,
            len(value),
        )
        return {}


def _coerce_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object, default: float) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _spans_overlap(left: EntitySpan, right: EntitySpan) -> bool:
    return left.start_position < right.end_position and right.start_position < left.end_position


def _find_all_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    escaped = re.escape(needle)
    if needle[0].isalnum() or needle[0] == "_":
        escaped = rf"(?<![A-Za-z0-9_]){escaped}"
    if needle[-1].isalnum() or needle[-1] == "_":
        escaped = rf"{escaped}(?![A-Za-z0-9_])"

    positions: list[tuple[int, int]] = []
    for match in re.finditer(escaped, text, flags=re.IGNORECASE):
        positions.append((match.start(), match.end()))
    return positions


def _build_entity_id(
    *,
    index: int,
    value: str,
    label: str,
    start: int,
    end: int,
    source: str = "detector",
) -> str:
    digest = sha1(f"{source}:{index}:{value}:{label}:{start}:{end}".encode("utf-8")).hexdigest()
    return digest[:16]


def _choose_tag_notation(text: str) -> TagNotation:
    candidates: tuple[tuple[TagNotation, tuple[str, ...]], ...] = (
        (TagNotation.xml, ("<", "</")),
        (TagNotation.bracket, ("[[", "]]")),
        (TagNotation.paren, ("((PII:", "))")),
        (TagNotation.sentinel, ("<<PII:", "<</PII:")),
    )
    scored = sorted(
        ((sum(text.count(marker) for marker in markers), notation) for notation, markers in candidates),
        key=lambda item: item[0],
    )
    return scored[0][1]


def _format_entity_tag(*, value: str, label: str, notation: TagNotation) -> str:
    if notation == TagNotation.xml:
        return f"<{label}>{value}</{label}>"
    if notation == TagNotation.bracket:
        return f"[[{value}|{label}]]"
    if notation == TagNotation.paren:
        return f"((PII:{label}|{value}))"
    return f"<<PII:{label}>>{value}<</PII:{label}>>"
