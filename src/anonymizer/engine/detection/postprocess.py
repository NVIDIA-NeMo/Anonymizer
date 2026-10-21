# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from anonymizer.engine.constants import SOURCE_KIND_AUGMENTER
from anonymizer.engine.schemas import DetectionResult

logger = logging.getLogger(__name__)

VALIDATION_CONTEXT_WINDOW = 32
_SpanT = TypeVar("_SpanT")


def group_overlapping_spans(
    ordered_spans: Iterable[_SpanT],
    *,
    start_position: Callable[[_SpanT], int],
    end_position: Callable[[_SpanT], int],
) -> list[tuple[_SpanT, ...]]:
    """Group start-ordered half-open spans into connected overlap components.

    Callers supply spans sorted by start position and choose their own tie
    ordering. Touching spans remain separate; nested and transitively
    overlapping spans stay together.
    """
    groups: list[tuple[_SpanT, ...]] = []
    current: list[_SpanT] = []
    current_end = -1
    for span in ordered_spans:
        if current and start_position(span) >= current_end:
            groups.append(tuple(current))
            current = []
            current_end = -1
        current.append(span)
        current_end = max(current_end, end_position(span))
    if current:
        groups.append(tuple(current))
    return groups


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


def normalize_label(label: str) -> str:
    """Canonical normalization for entity label comparisons: strip + casefold."""
    return label.strip().casefold()


def normalize_labels(labels: Iterable[str] | None) -> set[str]:
    """Normalize a collection of labels, dropping empty/whitespace-only entries."""
    return {normalized for label in labels or [] if (normalized := normalize_label(label))}


def is_reserved_entity_label(label: str) -> bool:
    """Return whether a label is reserved for internal prompt markup."""
    return re.fullmatch(r"nagroup_g[1-9][0-9]*", normalize_label(label)) is not None


def filter_excluded_entity_spans(
    entities: list[EntitySpan],
    excluded_entity_labels: Iterable[str] | None,
) -> list[EntitySpan]:
    """Remove entity spans whose normalized labels are explicitly excluded."""
    excluded = normalize_labels(excluded_entity_labels)
    if not excluded:
        return list(entities)
    return [entity for entity in entities if normalize_label(entity.label) not in excluded]


class TagNotation(str, Enum):
    xml = "xml"
    bracket = "bracket"
    paren = "paren"
    sentinel = "sentinel"


def project_detection_result_to_entity_spans(result: DetectionResult) -> list[EntitySpan]:
    """Project lifecycle observations into the legacy seed-entity payload."""
    return [
        EntitySpan(
            entity_id=_build_entity_id(
                label=observation.label,
                start=observation.start_position,
                end=observation.end_position,
            ),
            value=observation.value,
            label=observation.label,
            start_position=observation.start_position,
            end_position=observation.end_position,
            score=observation.native_score or 0.0,
            source="detector",
        )
        for observation in result.observations
    ]


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


def apply_augmented_entities(
    text: str,
    entities: list[EntitySpan],
    augmented_output: dict | str,
    excluded_entity_labels: set[str] | None = None,
) -> list[EntitySpan]:
    """Add allowed augmented entities, split full names, and resolve overlaps."""
    payload = _safe_json_loads(augmented_output) if isinstance(augmented_output, str) else augmented_output
    augmented = payload.get("entities", []) if isinstance(payload, dict) else []
    if not isinstance(augmented, list):
        augmented = []
    excluded = normalize_labels(excluded_entity_labels)

    merged = filter_excluded_entity_spans(entities, excluded)
    for idx, suggestion in enumerate(augmented):
        if not isinstance(suggestion, dict):
            continue
        value = str(suggestion.get("value", "")).strip()
        label = str(suggestion.get("label", "")).strip()
        if not value or not label or normalize_label(label) in excluded:
            continue
        for start, end in _find_all_occurrences(text=text, needle=value):
            entity_id = _build_entity_id(label=label, start=start, end=end)
            merged.append(
                EntitySpan(
                    entity_id=entity_id,
                    value=value,
                    label=label,
                    start_position=start,
                    end_position=end,
                    score=1.0,
                    source=SOURCE_KIND_AUGMENTER,
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
                entity_id = _build_entity_id(label=part_label, start=start, end=end)
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
    """Resolve span conflicts by preferring longer spans, then earlier starts and labels."""
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


def build_tagged_text(
    text: str,
    entities: list[EntitySpan],
    *,
    notation: TagNotation | str | None = None,
) -> str:
    """Render human-readable tagged text for downstream LLM prompts.

    Args:
        text: Source text to annotate.
        entities: Entities to tag within ``text``; positions are relative to ``text``.
        notation: Optional override of the tag notation. When ``None`` the
            notation is chosen heuristically from ``text`` (default behaviour).
            Callers that tag a substring of a larger document should pass the
            parent document's notation so tags remain stable across excerpts.
    """
    if not entities:
        return text
    if notation is None:
        resolved_notation = _choose_tag_notation(text)
    elif isinstance(notation, TagNotation):
        resolved_notation = notation
    else:
        resolved_notation = TagNotation(notation)
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
                notation=resolved_notation,
            )
        )
        cursor = entity.end_position
    parts.append(text[cursor:])
    return "".join(parts)


def build_conflict_aware_tagged_text(text: str, entities: list[EntitySpan]) -> str:
    """Render one neutral marker for each connected group of overlapping spans."""
    ordered = sorted(
        entities,
        key=lambda entity: (
            entity.start_position,
            entity.end_position,
            normalize_label(entity.label),
            entity.entity_id,
        ),
    )
    groups = group_overlapping_spans(
        ordered,
        start_position=lambda entity: entity.start_position,
        end_position=lambda entity: entity.end_position,
    )

    markers: list[EntitySpan] = []
    next_group_id = 1
    for group in groups:
        if len(group) == 1:
            markers.append(group[0])
            continue
        start = min(entity.start_position for entity in group)
        end = max(entity.end_position for entity in group)
        group_id = f"g{next_group_id}"
        next_group_id += 1
        markers.append(
            EntitySpan(
                entity_id=group_id,
                value=text[start:end],
                label=f"NAGROUP_{group_id}",
                start_position=start,
                end_position=end,
                score=0.0,
                source="prompt_marker",
            )
        )
    return build_tagged_text(text=text, entities=markers)


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

    original_positions: set[tuple[int, int]] = {(e.start_position, e.end_position) for e in entities}
    expanded: list[EntitySpan] = []
    for idx, (key, label) in enumerate(entity_map.items()):
        original_value = next(e.value for e in entities if e.value.lower() == key)
        for start, end in _find_all_occurrences(text=text, needle=original_value):
            if (start, end) in original_positions:
                continue  # already covered by a detector span; skip to preserve its provenance
            entity_id = _build_entity_id(label=label, start=start, end=end)
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


def _build_entity_id(*, label: str, start: int, end: int) -> str:
    return f"{label}_{start}_{end}"


def _choose_tag_notation(text: str) -> TagNotation:
    candidates: tuple[tuple[TagNotation, tuple[str, ...]], ...] = (
        (TagNotation.xml, ("<", "</")),
        (TagNotation.bracket, ("[[", "]]")),
        (TagNotation.paren, ("((SENSITIVE:", "))")),
        (TagNotation.sentinel, ("<<SENSITIVE:", "<</SENSITIVE:")),
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
        return f"((SENSITIVE:{label}|{value}))"
    return f"<<SENSITIVE:{label}>>{value}<</SENSITIVE:{label}>>"
