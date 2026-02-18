# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
import pytest

from anonymizer.interface.display import (
    _build_replaced_entities,
    _normalize_replacement_map,
    _render_highlighted_text,
    render_record_html,
)
from anonymizer.interface.results import PreviewResult

# ---------------------------------------------------------------------------
# _render_highlighted_text
# ---------------------------------------------------------------------------


def test_highlighted_text_wraps_entity_in_styled_span() -> None:
    entities = [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]
    result = _render_highlighted_text("Alice works here", entities)
    assert "Alice" in result
    assert "first_name" in result
    assert "border:" in result


def test_highlighted_text_escapes_html_in_plain_text() -> None:
    entities = [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]
    result = _render_highlighted_text("Alice <b>works</b> here", entities)
    assert "&lt;b&gt;" in result


def test_highlighted_text_preserves_text_between_entities() -> None:
    entities = [
        {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
        {"value": "Acme", "label": "organization", "start_position": 15, "end_position": 19},
    ]
    result = _render_highlighted_text("Alice works at Acme", entities)
    assert " works at " in result


def test_highlighted_text_no_entities_returns_escaped_text() -> None:
    result = _render_highlighted_text("Hello <world>", [])
    assert "&lt;world&gt;" in result


def test_highlighted_text_skips_overlapping_entity() -> None:
    entities = [
        {"value": "John Doe", "label": "full_name", "start_position": 0, "end_position": 8},
        {"value": "Doe", "label": "last_name", "start_position": 5, "end_position": 8},
    ]
    result = _render_highlighted_text("John Doe went home", entities)
    assert "full_name" in result
    assert "last_name" not in result


def test_highlighted_text_consistent_color_per_label() -> None:
    entities = [
        {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
        {"value": "Bob", "label": "first_name", "start_position": 10, "end_position": 13},
    ]
    result = _render_highlighted_text("Alice met Bob here", entities)
    border_colors = [
        s.split("border:1.5px solid ")[1].split(";")[0] for s in result.split("<span") if "border:1.5px solid" in s
    ]
    assert len(border_colors) == 2
    assert border_colors[0] == border_colors[1]


# ---------------------------------------------------------------------------
# _build_replaced_entities — position tracking in replaced text
# ---------------------------------------------------------------------------


def test_replaced_entities_tracks_shifted_positions() -> None:
    original_entities = [
        {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
        {"value": "Acme", "label": "organization", "start_position": 15, "end_position": 19},
    ]
    replacement_map = [
        {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
        {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
    ]
    replaced_text = "Maya works at NovaCorp"

    result = _build_replaced_entities(original_entities, replacement_map, "Alice works at Acme", replaced_text)
    assert len(result) == 2
    assert result[0]["value"] == "Maya"
    assert result[0]["start_position"] == 0
    assert result[0]["end_position"] == 4
    assert result[1]["value"] == "NovaCorp"
    assert result[1]["start_position"] == 14


def test_replaced_entities_empty_map_uses_original_values() -> None:
    original_entities = [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]
    result = _build_replaced_entities(original_entities, [], "Alice works", "Alice works")
    assert len(result) == 1
    assert result[0]["value"] == "Alice"


# ---------------------------------------------------------------------------
# _normalize_replacement_map
# ---------------------------------------------------------------------------


def test_normalize_replacement_map_from_dict() -> None:
    raw = {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]}
    result = _normalize_replacement_map(raw)
    assert len(result) == 1
    assert result[0]["synthetic"] == "Maya"


def test_normalize_replacement_map_from_json_string() -> None:
    raw = '{"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]}'
    result = _normalize_replacement_map(raw)
    assert len(result) == 1


def test_normalize_replacement_map_invalid_json_returns_empty() -> None:
    assert _normalize_replacement_map("bad json {{{") == []


def test_normalize_replacement_map_non_dict_returns_empty() -> None:
    assert _normalize_replacement_map([1, 2, 3]) == []


# ---------------------------------------------------------------------------
# render_record_html — full integration
# ---------------------------------------------------------------------------


def test_render_record_html_contains_all_sections() -> None:
    row = pd.Series(
        {
            "text": "Alice works at Acme",
            "replaced_text": "[REDACTED_FIRST_NAME] works at [REDACTED_ORGANIZATION]",
            "_detected_entities": [
                {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
                {"value": "Acme", "label": "organization", "start_position": 15, "end_position": 19},
            ],
            "_replacement_map": {
                "replacements": [
                    {"original": "Alice", "label": "first_name", "synthetic": "[REDACTED_FIRST_NAME]"},
                    {"original": "Acme", "label": "organization", "synthetic": "[REDACTED_ORGANIZATION]"},
                ]
            },
        }
    )
    result = render_record_html(row, record_index=0)
    assert "Original" in result
    assert "Replaced" in result
    assert "Replacement Map" in result
    assert "Alice" in result
    assert "REDACTED_FIRST_NAME" in result
    assert "record 0" in result


def test_render_record_html_without_replacement_map() -> None:
    row = pd.Series(
        {
            "text": "Alice works here",
            "replaced_text": "Alice works here",
            "_detected_entities": [],
            "_replacement_map": {},
        }
    )
    result = render_record_html(row)
    assert "No replacement map available" in result


# ---------------------------------------------------------------------------
# PreviewResult.display_record
# ---------------------------------------------------------------------------


def test_display_record_cycles_on_repeated_calls() -> None:
    df = pd.DataFrame(
        {
            "text": ["Alice", "Bob"],
            "replaced_text": ["[R]", "[R]"],
            "_detected_entities": [[], []],
            "_replacement_map": [{}, {}],
        }
    )
    preview = PreviewResult(dataframe=df, failed_records=[], preview_num_records=2)
    assert preview._display_cycle_index == 0
    preview.display_record()
    assert preview._display_cycle_index == 1
    preview.display_record()
    assert preview._display_cycle_index == 0


def test_display_record_explicit_index_does_not_advance_cycle() -> None:
    df = pd.DataFrame(
        {
            "text": ["Alice", "Bob"],
            "replaced_text": ["[R]", "[R]"],
            "_detected_entities": [[], []],
            "_replacement_map": [{}, {}],
        }
    )
    preview = PreviewResult(dataframe=df, failed_records=[], preview_num_records=2)
    preview.display_record(index=1)
    assert preview._display_cycle_index == 0


def test_display_record_out_of_bounds_raises() -> None:
    df = pd.DataFrame(
        {
            "text": ["Alice"],
            "replaced_text": ["[R]"],
            "_detected_entities": [[]],
            "_replacement_map": [{}],
        }
    )
    preview = PreviewResult(dataframe=df, failed_records=[], preview_num_records=1)
    with pytest.raises(IndexError, match="out of bounds"):
        preview.display_record(index=5)
