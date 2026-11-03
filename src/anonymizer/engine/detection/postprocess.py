# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar

from anonymizer.engine.schemas import DetectionResult

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
