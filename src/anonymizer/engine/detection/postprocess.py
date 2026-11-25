# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from itertools import count

from anonymizer.config.entity_labels import label_keys
from anonymizer.config.entity_labels import normalize_entity_label as normalize_label
from anonymizer.engine.edits import Edit, Edits, Prefer
from anonymizer.engine.schemas import DetectionResult
from anonymizer.engine.spans import BY_START_END, Positioned, Span, group_overlapping_spans, within

VALIDATION_CONTEXT_WINDOW = 32


@dataclass(frozen=True)
class EntitySpan(Positioned):
    """Canonical standoff entity representation."""

    _noun = "entity span"

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


normalize_labels = label_keys
"""The keys of a collection of labels, dropping empty and whitespace-only ones (``label_keys``)."""


def is_reserved_entity_label(label: str) -> bool:
    """Return whether a label is reserved for internal prompt markup."""
    return re.fullmatch(r"nagroup_g[1-9][0-9]*", normalize_label(label)) is not None


class TagNotation(str, Enum):
    """How prompt text marks an entity: the ``markers`` whose count in a text decides the notation, and the tag
    ``template``. A text gets the first notation, in member order, whose markers occur least often in it."""

    markers: tuple[str, ...]
    template: str

    def __new__(cls, value: str, markers: tuple[str, ...], template: str) -> TagNotation:
        member = str.__new__(cls, value)
        member._value_ = value
        member.markers = markers
        member.template = template
        return member

    xml = "xml", ("<", "</"), "<{label}>{value}</{label}>"
    bracket = "bracket", ("[[", "]]"), "[[{value}|{label}]]"
    paren = "paren", ("((SENSITIVE:", "))"), "((SENSITIVE:{label}|{value}))"
    sentinel = "sentinel", ("<<SENSITIVE:", "<</SENSITIVE:"), "<<SENSITIVE:{label}>>{value}<</SENSITIVE:{label}>>"

    def marker_count(self, text: str) -> int:
        return sum(text.count(marker) for marker in self.markers)

    def tag(self, value: str, label: str) -> str:
        return self.template.format(value=value, label=label)


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
            source=(
                "detector"
                if observation.source_kind == "initial_detector"
                else f"{observation.source_kind}:{observation.source_instance}"
            ),
        )
        for observation in result.observations
    ]


def build_validation_candidates(text: str, entities: list[EntitySpan]) -> list[dict[str, str]]:
    """Build per-entity validation payload with context windows."""
    # TODO(lramaswamy): make validation context window configurable from AnonymizerConfig.
    return [
        {
            "id": entity.entity_id,
            "value": entity.value,
            "label": entity.label,
            "context_before": text[max(0, entity.start_position - VALIDATION_CONTEXT_WINDOW) : entity.start_position],
            "context_after": text[
                entity.end_position : min(len(text), entity.end_position + VALIDATION_CONTEXT_WINDOW)
            ],
        }
        for entity in entities
    ]


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
    # A tag whose range is not in the text is dropped.
    tags = [Edit(span, resolved_notation.tag(text[span.slice], e.label)) for e, span in within(len(text), entities)]
    # Among overlapping tags the earlier, then the shorter, stands; identical spans keep input order.
    edits, _ = Edits.admit(text, tags, Prefer(BY_START_END))
    return edits.apply()


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
    groups = group_overlapping_spans(ordered, span=lambda entity: entity.span)

    group_ids = (f"g{n}" for n in count(1))  # numbered over the groups of two or more, in order

    def marker(group: tuple[EntitySpan, ...]) -> EntitySpan:
        if len(group) == 1:
            return group[0]
        bounds = Span.cover(entity.span for entity in group)
        group_id = next(group_ids)
        return EntitySpan(
            entity_id=group_id,
            value=text[bounds.slice],
            label=f"NAGROUP_{group_id}",
            start_position=bounds.start,
            end_position=bounds.end,
            score=0.0,
            source="prompt_marker",
        )

    return build_tagged_text(text=text, entities=[marker(group) for group in groups])


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
    return min(TagNotation, key=lambda notation: notation.marker_count(text))
