# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

import pytest

from anonymizer.engine.detection.postprocess import (
    EntitySpan,
    apply_augmented_entities,
    apply_validation_decisions,
    build_tagged_text,
    build_validation_candidates,
    expand_entity_occurrences,
    get_tag_notation,
    group_entities_by_value,
    parse_raw_entities,
    resolve_overlaps,
)


def test_parse_raw_entities_parses_valid_spans() -> None:
    text = "Call me at (555) 123-4567"
    raw = '{"entities":[{"text":"(555) 123-4567","label":"phone_number","start":11,"end":25,"score":0.9}]}'
    entities = parse_raw_entities(raw_response=raw, text=text)
    assert len(entities) == 1
    assert entities[0].label == "phone_number"


def test_overlap_resolution_prefers_longer_span() -> None:
    short = EntitySpan("a", "John", "first_name", 0, 4, 1.0, "detector")
    long = EntitySpan("b", "John Doe", "full_name", 0, 8, 1.0, "detector")
    resolved = resolve_overlaps([short, long])
    assert len(resolved) == 1
    assert resolved[0].value == "John Doe"


def test_apply_validation_decisions_drops_entities() -> None:
    entities = [
        EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector"),
        EntitySpan("id2", "Seattle", "city", 10, 17, 1.0, "detector"),
    ]
    validated = apply_validation_decisions(
        entities=entities,
        validation_output={"decisions": [{"id": "id1", "decision": "keep"}, {"id": "id2", "decision": "drop"}]},
    )
    assert [item.entity_id for item in validated] == ["id1"]


def test_apply_validation_decisions_reclassifies_label() -> None:
    entities = [
        EntitySpan("id1", "San Diego", "country", 5, 14, 1.0, "detector"),
    ]
    validated = apply_validation_decisions(
        entities=entities,
        validation_output={
            "decisions": [{"id": "id1", "decision": "reclass", "proposed_label": "city"}],
        },
    )
    assert len(validated) == 1
    assert validated[0].label == "city"
    assert validated[0].value == "San Diego"
    assert validated[0].start_position == 5


def test_apply_validation_decisions_reclass_without_label_keeps_original() -> None:
    """If reclass has empty proposed_label, keep original label."""
    entities = [
        EntitySpan("id1", "Portland", "country", 0, 8, 1.0, "detector"),
    ]
    validated = apply_validation_decisions(
        entities=entities,
        validation_output={
            "decisions": [{"id": "id1", "decision": "reclass", "proposed_label": ""}],
        },
    )
    assert len(validated) == 1
    assert validated[0].label == "country"


def test_apply_validation_decisions_unknown_id_defaults_to_keep() -> None:
    entities = [
        EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector"),
    ]
    validated = apply_validation_decisions(
        entities=entities,
        validation_output={"decisions": [{"id": "id_unknown", "decision": "drop"}]},
    )
    assert len(validated) == 1


def test_apply_validation_decisions_last_decision_wins_for_duplicate_ids() -> None:
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    validated = apply_validation_decisions(
        entities=entities,
        validation_output={
            "decisions": [
                {"id": "id1", "decision": "drop"},
                {"id": "id1", "decision": "keep"},
            ]
        },
    )
    assert len(validated) == 1


def test_apply_augmented_entities_adds_occurrences() -> None:
    text = "Alice met Bob. Bob called Alice."
    merged = apply_augmented_entities(
        text=text,
        entities=[],
        augmented_output={"entities": [{"value": "Bob", "label": "first_name"}]},
    )
    assert len(merged) == 2
    assert all(entity.source == "augmenter" for entity in merged)


def test_apply_augmented_entities_avoids_substring_matches() -> None:
    text = "Annex contains Ann."
    merged = apply_augmented_entities(
        text=text,
        entities=[],
        augmented_output={"entities": [{"value": "Ann", "label": "first_name"}]},
    )
    assert len(merged) == 1
    assert merged[0].value == "Ann"
    assert merged[0].start_position == 15


def test_augmented_splits_full_name_into_parts() -> None:
    text = "John Smith called. Later, John met Smith."
    merged = apply_augmented_entities(
        text=text,
        entities=[EntitySpan("fn", "John Smith", "full_name", 0, 10, 1.0, "detector")],
        augmented_output={"entities": []},
    )
    labels = {entity.label for entity in merged}
    assert "full_name" in labels
    assert "first_name" in labels
    assert "last_name" in labels
    # "John" at 0-4 overlaps with "John Smith" at 0-10, so only standalone at 26 survives
    johns = [e for e in merged if e.value == "John"]
    assert len(johns) == 1
    assert johns[0].start_position == 26
    smiths = [e for e in merged if e.value == "Smith"]
    assert len(smiths) == 1
    assert smiths[0].start_position == 35


def test_apply_augmented_entities_does_not_split_single_token_full_name() -> None:
    text = "Madonna performed tonight."
    merged = apply_augmented_entities(
        text=text,
        entities=[EntitySpan("fn", "Madonna", "full_name", 0, 7, 1.0, "detector")],
        augmented_output={"entities": []},
    )
    labels = {entity.label for entity in merged}
    assert "full_name" in labels
    assert "first_name" not in labels
    assert "last_name" not in labels


def test_name_split_skips_single_letter_parts() -> None:
    text = "John A. Smith went home."
    merged = apply_augmented_entities(
        text=text,
        entities=[EntitySpan("fn", "John A. Smith", "full_name", 0, 13, 1.0, "detector")],
        augmented_output={"entities": []},
    )
    values = [e.value.lower() for e in merged]
    assert "a." not in values


def test_name_split_does_not_duplicate_existing_entities() -> None:
    text = "John Smith lives here. John agrees and Smith too."
    existing = [
        EntitySpan("fn", "John Smith", "full_name", 0, 10, 1.0, "detector"),
        EntitySpan("jn", "John", "first_name", 23, 27, 1.0, "detector"),
    ]
    merged = apply_augmented_entities(
        text=text,
        entities=existing,
        augmented_output={"entities": []},
    )
    # "John" already exists, so name split should NOT add duplicates for it
    johns = [e for e in merged if e.value == "John" and e.source == "name_split"]
    assert len(johns) == 0
    # "Smith" should be added at the standalone occurrence (position 39)
    smiths = [e for e in merged if e.value == "Smith"]
    assert len(smiths) == 1
    assert smiths[0].label == "last_name"


def test_build_tagged_text_renders_xml_style_tags() -> None:
    text = "Alice Smith"
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert tagged == "<first_name>Alice</first_name> Smith"


def test_build_tagged_text_avoids_xml_when_input_has_xml() -> None:
    text = "<p>Alice Smith</p>"
    entities = [EntitySpan("id1", "Alice", "first_name", 3, 8, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert "<first_name>Alice</first_name>" not in tagged
    assert "[[Alice|first_name]]" in tagged


def test_parse_raw_entities_returns_empty_on_malformed_json() -> None:
    assert parse_raw_entities(raw_response="not json {{{", text="hello") == []


def test_parse_raw_entities_returns_empty_when_entities_not_a_list() -> None:
    assert parse_raw_entities(raw_response='{"entities": "not a list"}', text="hello") == []


def test_parse_raw_entities_drops_non_dict_items() -> None:
    raw = json.dumps({"entities": ["string_item", 42, None, {"text": "x", "label": "email"}]})
    result = parse_raw_entities(raw_response=raw, text="has x in it")
    assert len(result) == 0  # "x" has no valid position


def test_parse_raw_entities_drops_entities_with_empty_value() -> None:
    raw = json.dumps(
        {
            "entities": [
                {"text": "", "label": "email", "start": 0, "end": 5, "score": 0.9},
            ]
        }
    )
    assert parse_raw_entities(raw_response=raw, text="hello") == []


def test_parse_raw_entities_drops_entities_with_empty_label() -> None:
    raw = json.dumps(
        {
            "entities": [
                {"text": "hello", "label": "", "start": 0, "end": 5, "score": 0.9},
            ]
        }
    )
    assert parse_raw_entities(raw_response=raw, text="hello") == []


def test_parse_raw_entities_drops_negative_start() -> None:
    raw = json.dumps(
        {
            "entities": [
                {"text": "hello", "label": "email", "start": -1, "end": 5, "score": 0.9},
            ]
        }
    )
    assert parse_raw_entities(raw_response=raw, text="hello") == []


def test_parse_raw_entities_drops_end_equal_to_start() -> None:
    raw = json.dumps(
        {
            "entities": [
                {"text": "hello", "label": "email", "start": 3, "end": 3, "score": 0.9},
            ]
        }
    )
    assert parse_raw_entities(raw_response=raw, text="hello") == []


def test_parse_raw_entities_drops_end_beyond_text() -> None:
    raw = json.dumps(
        {
            "entities": [
                {"text": "hello", "label": "email", "start": 0, "end": 999, "score": 0.9},
            ]
        }
    )
    assert parse_raw_entities(raw_response=raw, text="hello") == []


def test_parse_raw_entities_handles_non_numeric_positions() -> None:
    raw = json.dumps(
        {
            "entities": [
                {
                    "text": "hello",
                    "label": "email",
                    "start": "abc",
                    "end": 5,
                    "score": 0.9,
                },
            ]
        }
    )
    assert parse_raw_entities(raw_response=raw, text="hello world") == []


def test_parse_raw_entities_handles_non_numeric_score() -> None:
    """Non-numeric score should default to 0.0 instead of crashing."""
    raw = json.dumps(
        {
            "entities": [
                {
                    "text": "Alice",
                    "label": "first_name",
                    "start": 0,
                    "end": 5,
                    "score": "bad",
                },
            ]
        }
    )
    result = parse_raw_entities(raw_response=raw, text="Alice")
    assert len(result) == 1
    assert result[0].score == 0.0


def test_parse_raw_entities_resolves_overlapping_spans() -> None:
    raw = json.dumps(
        {
            "entities": [
                {
                    "text": "John",
                    "label": "first_name",
                    "start": 0,
                    "end": 4,
                    "score": 0.8,
                },
                {
                    "text": "John Doe",
                    "label": "full_name",
                    "start": 0,
                    "end": 8,
                    "score": 0.9,
                },
            ]
        }
    )
    result = parse_raw_entities(raw_response=raw, text="John Doe went home")
    assert len(result) == 1
    assert result[0].value == "John Doe"


def test_resolve_overlaps_keeps_non_overlapping_spans() -> None:
    a = EntitySpan("a", "Alice", "first_name", 0, 5, 1.0, "detector")
    b = EntitySpan("b", "Acme", "organization", 15, 19, 1.0, "detector")
    resolved = resolve_overlaps([a, b])
    assert len(resolved) == 2


def test_resolve_overlaps_returns_sorted_by_position() -> None:
    b = EntitySpan("b", "Acme", "organization", 15, 19, 1.0, "detector")
    a = EntitySpan("a", "Alice", "first_name", 0, 5, 1.0, "detector")
    resolved = resolve_overlaps([b, a])  # input order reversed
    assert resolved[0].value == "Alice"
    assert resolved[1].value == "Acme"


def test_resolve_overlaps_nested_span_keeps_outer() -> None:
    """Inner 'York' inside 'New York' should be dropped."""
    outer = EntitySpan("a", "New York", "city", 0, 8, 1.0, "detector")
    inner = EntitySpan("b", "York", "city", 4, 8, 0.8, "detector")
    resolved = resolve_overlaps([inner, outer])
    assert len(resolved) == 1
    assert resolved[0].value == "New York"


def test_resolve_overlaps_empty_input() -> None:
    assert resolve_overlaps([]) == []


def test_validation_decisions_from_json_string() -> None:
    """Validation output arrives as JSON string after parquet round-trip."""
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    json_str = json.dumps({"decisions": [{"id": "id1", "decision": "drop"}]})
    result = apply_validation_decisions(entities=entities, validation_output=json_str)
    assert len(result) == 0


def test_validation_decisions_with_invalid_json_keeps_all() -> None:
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    result = apply_validation_decisions(entities=entities, validation_output="not valid json {{")
    assert len(result) == 1


def test_validation_decisions_with_non_list_decisions_keeps_all() -> None:
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    result = apply_validation_decisions(entities=entities, validation_output={"decisions": "not a list"})
    assert len(result) == 1


def test_validation_decisions_skips_invalid_decision_values() -> None:
    """Unknown decision like 'maybe' should be ignored — entity kept."""
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    result = apply_validation_decisions(
        entities=entities,
        validation_output={"decisions": [{"id": "id1", "decision": "maybe"}]},
    )
    assert len(result) == 1


def test_validation_decisions_skips_non_dict_decision_items() -> None:
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    result = apply_validation_decisions(
        entities=entities,
        validation_output={"decisions": ["not a dict", {"id": "id1", "decision": "keep"}]},
    )
    assert len(result) == 1


def test_augmented_entities_from_json_string() -> None:
    text = "Alice works here"
    result = apply_augmented_entities(
        text=text,
        entities=[],
        augmented_output=json.dumps({"entities": [{"value": "Alice", "label": "first_name"}]}),
    )
    assert len(result) == 1


def test_augmented_entities_with_invalid_json_returns_originals() -> None:
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    result = apply_augmented_entities(text="Alice", entities=entities, augmented_output="bad json {{{")
    assert len(result) == 1
    assert result[0].entity_id == "id1"


def test_augmented_entities_with_non_list_entities_returns_originals() -> None:
    entities = [EntitySpan("id1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    result = apply_augmented_entities(text="Alice", entities=entities, augmented_output={"entities": "not a list"})
    assert len(result) == 1


def test_augmented_entities_skips_non_dict_suggestions() -> None:
    text = "Alice and Bob"
    result = apply_augmented_entities(
        text=text,
        entities=[],
        augmented_output={"entities": ["not_a_dict", {"value": "Bob", "label": "first_name"}]},
    )
    bobs = [e for e in result if e.value == "Bob"]
    assert len(bobs) == 1


def test_augmented_entities_skips_empty_value_or_label() -> None:
    text = "Alice works here"
    result = apply_augmented_entities(
        text=text,
        entities=[],
        augmented_output={"entities": [{"value": "", "label": "first_name"}, {"value": "Alice", "label": ""}]},
    )
    assert len(result) == 0


def test_augmented_entities_case_insensitive_occurrence_finding() -> None:
    """'alice' in augmented output should match 'Alice' in text."""
    text = "Alice works here"
    result = apply_augmented_entities(
        text=text,
        entities=[],
        augmented_output={"entities": [{"value": "alice", "label": "first_name"}]},
    )
    assert len(result) == 1


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
    text = "<div>[[Alice]] ((PII:first_name|Alice)) is here</div>"
    entities = [EntitySpan("id1", "Alice", "first_name", 7, 12, 1.0, "detector")]
    tagged = build_tagged_text(text=text, entities=entities)
    assert "<<PII:first_name>>Alice<</PII:first_name>>" in tagged
    assert tagged.startswith("<div>[[<<PII:first_name>>Alice<</PII:first_name>>]]")
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


def test_expand_finds_all_occurrences_of_detected_entity() -> None:
    """'Mara' detected at position 0 should expand to all 4 occurrences."""
    text = "Mara is a director. Mara learned storytelling. Mara makes films."
    entities = [EntitySpan("e1", "Mara", "first_name", 0, 4, 1.0, "detector")]
    expanded = expand_entity_occurrences(text=text, entities=entities)
    mara_spans = [e for e in expanded if e.value == "Mara"]
    assert len(mara_spans) == 3
    assert {e.start_position for e in mara_spans} == {0, 20, 47}


def test_expand_preserves_original_entities() -> None:
    text = "Alice works at Acme"
    entities = [
        EntitySpan("e1", "Alice", "first_name", 0, 5, 0.95, "detector"),
        EntitySpan("e2", "Acme", "organization", 15, 19, 0.9, "detector"),
    ]
    expanded = expand_entity_occurrences(text=text, entities=entities)
    assert len(expanded) == 2
    assert expanded[0].value == "Alice"
    assert expanded[1].value == "Acme"


def test_expand_avoids_substring_matches() -> None:
    """'Ann' should not match inside 'Annex'."""
    text = "Annex contains Ann. Ann is here."
    entities = [EntitySpan("e1", "Ann", "first_name", 15, 18, 1.0, "detector")]
    expanded = expand_entity_occurrences(text=text, entities=entities)
    ann_spans = [e for e in expanded if e.value == "Ann"]
    assert len(ann_spans) == 2
    positions = {e.start_position for e in ann_spans}
    assert 0 not in positions  # "Annex" should not match


def test_expand_resolves_overlaps_with_longer_span() -> None:
    text = "John Doe met John later"
    entities = [
        EntitySpan("e1", "John Doe", "full_name", 0, 8, 1.0, "detector"),
        EntitySpan("e2", "John", "first_name", 13, 17, 1.0, "detector"),
    ]
    expanded = expand_entity_occurrences(text=text, entities=entities)
    assert any(e.value == "John Doe" for e in expanded)
    johns = [e for e in expanded if e.value == "John"]
    assert len(johns) == 1
    assert johns[0].start_position == 13


def test_expand_handles_empty_entities() -> None:
    assert expand_entity_occurrences(text="hello world", entities=[]) == []


def test_expand_case_insensitive_matching() -> None:
    """'alice' in text should match 'Alice' entity."""
    text = "Alice met alice later"
    entities = [EntitySpan("e1", "Alice", "first_name", 0, 5, 1.0, "detector")]
    expanded = expand_entity_occurrences(text=text, entities=entities)
    assert len(expanded) == 2


def test_safe_json_loads_logs_warning_on_invalid_json(caplog: pytest.LogCaptureFixture) -> None:
    from anonymizer.engine.detection.postprocess import _safe_json_loads

    with caplog.at_level(logging.WARNING, logger="anonymizer.engine.detection.postprocess"):
        result = _safe_json_loads("{invalid json")
    assert result == {}
    assert any("Failed to parse JSON" in m for m in caplog.messages)
