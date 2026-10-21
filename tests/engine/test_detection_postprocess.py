# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    build_conflict_aware_tagged_text,
    build_tagged_text,
    build_validation_candidates,
    get_tag_notation,
    group_entities_by_value,
    group_overlapping_spans,
    normalize_label,
    normalize_labels,
)


@pytest.mark.parametrize(
    ("spans", "expected"),
    [
        ([], []),
        ([(0, 5)], [((0, 5),)]),
        ([(0, 5), (5, 8)], [((0, 5),), ((5, 8),)]),
        ([(0, 5), (7, 8)], [((0, 5),), ((7, 8),)]),
        ([(0, 10), (2, 3), (9, 11)], [((0, 10), (2, 3), (9, 11))]),
        ([(0, 5), (3, 10), (9, 11)], [((0, 5), (3, 10), (9, 11))]),
        ([(0, 5), (0, 5)], [((0, 5), (0, 5))]),
    ],
)
def test_group_overlapping_spans_uses_only_source_bounds(
    spans: list[tuple[int, int]], expected: list[tuple[tuple[int, int], ...]]
) -> None:
    assert (
        group_overlapping_spans(spans, start_position=lambda span: span[0], end_position=lambda span: span[1])
        == expected
    )


def test_normalize_label_strips_and_casefolds() -> None:
    assert normalize_label(" Health_Condition ") == "health_condition"


def test_normalize_labels_dedupes_and_drops_empty_entries() -> None:
    assert normalize_labels([" Email ", "email", "  ", "City"]) == {"email", "city"}


def test_normalize_labels_none_returns_empty_set() -> None:
    assert normalize_labels(None) == set()


def test_build_tagged_text_renders_xml_style_tags() -> None:
    text = "Alice Smith"
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert tagged == "<first_name>Alice</first_name> Smith"


def test_conflict_aware_tagged_text_marks_overlapping_region_once() -> None:
    text = "John Smith arrived"
    entities = [
        EntitySpan("o1", "John Smith", "full_name", 0, 10, 0.8, "detector"),
        EntitySpan("o2", "John", "first_name", 0, 4, 0.9, "detector"),
        EntitySpan("o3", "Smith", "last_name", 5, 10, 0.9, "detector"),
    ]

    tagged = build_conflict_aware_tagged_text(text, entities)

    assert tagged == "<NAGROUP_g1>John Smith</NAGROUP_g1> arrived"
    assert tagged.count("John Smith") == 1


def test_build_tagged_text_avoids_xml_when_input_has_xml() -> None:
    text = "<p>Alice Smith</p>"
    entities = [EntitySpan("id1", "Alice", "first_name", 3, 8, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert "<first_name>Alice</first_name>" not in tagged
    assert "[[Alice|first_name]]" in tagged


def test_build_tagged_text_empty_entities_returns_text() -> None:
    assert build_tagged_text(text="hello world", entities=[]) == "hello world"


def test_build_tagged_text_entity_at_start() -> None:
    text = "Alice works here"
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert tagged.startswith("<first_name>Alice</first_name>")


def test_build_tagged_text_entity_at_end() -> None:
    text = "works at Acme"
    entities = [EntitySpan("id1", "Acme", "organization", 9, 13, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert tagged.endswith("<organization>Acme</organization>")


def test_build_tagged_text_adjacent_entities() -> None:
    """Two entities with no gap between them."""
    text = "AliceBob"
    entities = [
        EntitySpan("a", "Alice", "first_name", 0, 5, 1.0, "detector"),
        EntitySpan("b", "Bob", "first_name", 5, 8, 1.0, "detector"),
    ]
    tagged = build_tagged_text(text=text, entities=entities)
    assert "<first_name>Alice</first_name><first_name>Bob</first_name>" == tagged


def test_build_tagged_text_skips_overlapping_entity() -> None:
    """If entities overlap in input, the later start is skipped."""
    text = "John Doe went"
    entities = [
        EntitySpan("a", "John Doe", "full_name", 0, 8, 1.0, "detector"),
        EntitySpan("b", "Doe", "last_name", 5, 8, 0.5, "detector"),
    ]
    tagged = build_tagged_text(text=text, entities=entities)
    assert "full_name" in tagged
    assert "last_name" not in tagged


def test_build_tagged_text_uses_paren_notation_when_xml_and_bracket_conflict() -> None:
    """Text with both < and [[ should fall back to paren or sentinel."""
    text = "<div>[[Alice]] is here</div>"
    entities = [EntitySpan("id1", "Alice", "first_name", 10, 15, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert "<first_name>" not in tagged
    assert "[[Alice|first_name]]" not in tagged


def test_build_tagged_text_uses_sentinel_notation_when_others_conflict() -> None:
    text = "<div>[[Alice]] ((SENSITIVE:first_name|Alice)) is here</div>"
    entities = [EntitySpan("id1", "Alice", "first_name", 7, 12, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert "<<SENSITIVE:first_name>>Alice<</SENSITIVE:first_name>>" in tagged
    assert tagged.startswith("<div>[[<<SENSITIVE:first_name>>Alice<</SENSITIVE:first_name>>]]")
    assert "<first_name>Alice</first_name>" not in tagged
    assert "[[Alice|first_name]]" not in tagged


def test_validation_candidates_include_context_window() -> None:
    text = "Dr. Alice Smith is a cardiologist at Regional Medical Center."
    entities = [EntitySpan("e1", "Alice Smith", "full_name", 4, 15, 1.0, "detector")]
    candidates = build_validation_candidates(text=text, entities=entities)
    assert len(candidates) == 1
    assert candidates[0]["context_before"] == "Dr. "
    assert candidates[0]["context_after"].startswith(" is a ")


def test_validation_candidates_clip_at_text_start() -> None:
    text = "Alice works here"
    entities = [EntitySpan("e1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    candidates = build_validation_candidates(text=text, entities=entities)
    assert candidates[0]["context_before"] == ""


def test_validation_candidates_clip_at_text_end() -> None:
    text = "works at Acme"
    entities = [EntitySpan("e1", "Acme", "organization", 9, 13, 1.0, "detector")]
    candidates = build_validation_candidates(text=text, entities=entities)
    assert candidates[0]["context_after"] == ""


def test_get_tag_notation_returns_xml_for_plain_text() -> None:
    assert get_tag_notation("Hello world") == "xml"


def test_get_tag_notation_avoids_xml_for_html_text() -> None:
    assert get_tag_notation("<p>Hello <b>world</b></p>") != "xml"


def test_group_entities_by_value_groups_labels() -> None:
    entities = [
        EntitySpan("a", "Alice", "first_name", 0, 5, 1.0, "detector"),
        EntitySpan("b", "Alice", "user_name", 20, 25, 1.0, "detector"),
    ]
    grouped = group_entities_by_value(entities=entities)
    assert len(grouped) == 1
    assert grouped[0]["value"] == "Alice"
    assert set(grouped[0]["labels"]) == {"first_name", "user_name"}


def test_group_entities_by_value_sorts_by_value() -> None:
    entities = [
        EntitySpan("b", "Zara", "first_name", 0, 4, 1.0, "detector"),
        EntitySpan("a", "Alice", "first_name", 10, 15, 1.0, "detector"),
    ]
    grouped = group_entities_by_value(entities=entities)
    assert grouped[0]["value"] == "Alice"
    assert grouped[1]["value"] == "Zara"


def test_group_entities_by_value_empty() -> None:
    assert group_entities_by_value(entities=[]) == []
