# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_FINAL_ENTITIES, COL_REPLACEMENT_MAP
from anonymizer.engine.schemas import EntitiesSchema, EntitySchema
from anonymizer.interface.display import (
    _build_replaced_entities,
    _normalize_replacement_map,
    _render_highlighted_text,
    render_record_html,
)
from anonymizer.interface.results import PreviewResult


def _entity(value: str, label: str, start: int, end: int) -> EntitySchema:
    return EntitySchema(value=value, label=label, start_position=start, end_position=end)


def _build_detected_entities_payload(payload_kind: str) -> object:
    entities_list = [
        {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
        {"value": "Acme", "label": "organization", "start_position": 15, "end_position": 19},
    ]
    if payload_kind == "dict_wrapper":
        return {"entities": entities_list}
    if payload_kind == "numpy_wrapped_dict_wrapper":
        return {"entities": np.array(entities_list, dtype=object)}
    if payload_kind == "entities_schema":
        return EntitiesSchema(entities=entities_list)
    raise ValueError(f"Unsupported payload kind: {payload_kind}")


def test_highlighted_text_wraps_entity_in_styled_span() -> None:
    entities = [_entity("Alice", "first_name", 0, 5)]
    result = _render_highlighted_text("Alice works here", entities)
    assert "Alice" in result
    assert "first_name" in result
    assert "border:" in result


def test_highlighted_text_escapes_html_in_plain_text() -> None:
    entities = [_entity("Alice", "first_name", 0, 5)]
    result = _render_highlighted_text("Alice <b>works</b> here", entities)
    assert "&lt;b&gt;" in result


def test_highlighted_text_preserves_text_between_entities() -> None:
    entities = [
        _entity("Alice", "first_name", 0, 5),
        _entity("Acme", "organization", 15, 19),
    ]
    result = _render_highlighted_text("Alice works at Acme", entities)
    assert " works at " in result


def test_highlighted_text_no_entities_returns_escaped_text() -> None:
    result = _render_highlighted_text("Hello <world>", [])
    assert "&lt;world&gt;" in result


def test_highlighted_text_skips_overlapping_entity() -> None:
    entities = [
        _entity("John Doe", "full_name", 0, 8),
        _entity("Doe", "last_name", 5, 8),
    ]
    result = _render_highlighted_text("John Doe went home", entities)
    assert "full_name" in result
    assert "last_name" not in result


def test_highlighted_text_consistent_color_per_label() -> None:
    entities = [
        _entity("Alice", "first_name", 0, 5),
        _entity("Bob", "first_name", 10, 13),
    ]
    result = _render_highlighted_text("Alice met Bob here", entities)
    border_colors = [
        s.split("border:1.5px solid ")[1].split(";")[0] for s in result.split("<span") if "border:1.5px solid" in s
    ]
    assert len(border_colors) == 2
    assert border_colors[0] == border_colors[1]


def test_replaced_entities_tracks_shifted_positions() -> None:
    original_entities = [
        _entity("Alice", "first_name", 0, 5),
        _entity("Acme", "organization", 15, 19),
    ]
    replacement_map = [
        {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
        {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
    ]
    result = _build_replaced_entities(
        original_entities,
        replacement_map,
        "Alice works at Acme",
        "Maya works at NovaCorp",
    )
    assert len(result) == 2
    assert result[0].value == "Maya"
    assert result[0].start_position == 0
    assert result[0].end_position == 4
    assert result[1].value == "NovaCorp"
    assert result[1].start_position == 14


def test_replaced_entities_empty_map_uses_original_values() -> None:
    original_entities = [_entity("Alice", "first_name", 0, 5)]
    result = _build_replaced_entities(original_entities, [], "Alice works", "Alice works")
    assert len(result) == 1
    assert result[0].value == "Alice"


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


@pytest.mark.parametrize(
    "payload_kind",
    ["dict_wrapper", "numpy_wrapped_dict_wrapper", "entities_schema"],
)
def test_render_record_html_contains_all_sections(payload_kind: str) -> None:
    row = pd.Series(
        {
            "text": "Alice works at Acme",
            "text_replaced": "[REDACTED_FIRST_NAME] works at [REDACTED_ORGANIZATION]",
            COL_DETECTED_ENTITIES: _build_detected_entities_payload(payload_kind),
            COL_REPLACEMENT_MAP: {
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


def test_render_record_html_original_highlights_from_map_when_entities_empty() -> None:
    """When _detected_entities is empty but replacement_map exists, Original gets highlights."""
    row = {
        "text": "Alice works at Acme",
        "text_replaced": "Maya works at NovaCorp",
        COL_DETECTED_ENTITIES: {"entities": []},
        COL_REPLACEMENT_MAP: {
            "replacements": [
                {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
            ]
        },
    }
    html_str = render_record_html(pd.Series(row))
    assert "Original" in html_str
    assert "Alice" in html_str
    assert "Acme" in html_str
    assert "first_name" in html_str


def test_render_record_html_replaced_highlights_from_map_when_entities_empty() -> None:
    """When _detected_entities is empty but replacement_map exists, highlights come from map."""
    row = {
        "text": "Alice works at Acme",
        "text_replaced": "Maya works at NovaCorp",
        COL_DETECTED_ENTITIES: {"entities": []},
        COL_REPLACEMENT_MAP: {
            "replacements": [
                {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
            ]
        },
    }
    html_str = render_record_html(pd.Series(row))
    assert "Maya" in html_str
    assert "NovaCorp" in html_str
    assert "border:1.5px solid" in html_str


def test_render_record_html_without_replacement_map() -> None:
    row = pd.Series(
        {
            "text": "Alice works here",
            "text_replaced": "Alice works here",
            COL_DETECTED_ENTITIES: {"entities": []},
            COL_REPLACEMENT_MAP: {},
        }
    )
    result = render_record_html(row)
    assert "No replacement map available" in result


def _make_preview(rows: int = 2) -> PreviewResult:
    df = pd.DataFrame(
        {
            "text": ["Alice", "Bob"][:rows],
            "text_replaced": ["[R]", "[R]"][:rows],
            COL_DETECTED_ENTITIES: [{"entities": []} for _ in range(rows)],
            COL_REPLACEMENT_MAP: [{} for _ in range(rows)],
        }
    )
    df.attrs["original_text_column"] = "text"
    return PreviewResult(dataframe=df, trace_dataframe=df, failed_records=[], preview_num_records=rows)


def test_render_record_html_uses_detected_entities_over_map_scan() -> None:
    """When _detected_entities is populated, use them directly instead of map-based scanning."""
    row = pd.Series(
        {
            "text": "She works at the Lantern in Austin",
            "text_replaced": "She works at [REDACTED_COMPANY] in [REDACTED_CITY]",
            COL_DETECTED_ENTITIES: {
                "entities": [
                    {"value": "The Lantern", "label": "company_name", "start_position": 13, "end_position": 24},
                    {"value": "Austin", "label": "city", "start_position": 28, "end_position": 34},
                ]
            },
            COL_REPLACEMENT_MAP: {
                "replacements": [
                    {"original": "The Lantern", "label": "company_name", "synthetic": "[REDACTED_COMPANY]"},
                    {"original": "Austin", "label": "city", "synthetic": "[REDACTED_CITY]"},
                ]
            },
        }
    )
    result = render_record_html(row)
    assert "company_name" in result
    assert "city" in result
    assert "[REDACTED_COMPANY]" in result
    assert "[REDACTED_CITY]" in result


def test_render_record_html_prefers_final_entities_for_filtered_replace_preview() -> None:
    row = pd.Series(
        {
            "text": "Mara Delgado works in Austin",
            "text_replaced": "[REDACTED_FIRST_NAME] Delgado works in Austin",
            COL_DETECTED_ENTITIES: {
                "entities": [
                    {"value": "Mara", "label": "first_name", "start_position": 0, "end_position": 4},
                    {"value": "Delgado", "label": "last_name", "start_position": 5, "end_position": 12},
                    {"value": "Austin", "label": "city", "start_position": 22, "end_position": 28},
                ]
            },
            COL_FINAL_ENTITIES: {
                "entities": [
                    {"value": "Mara", "label": "first_name", "start_position": 0, "end_position": 4},
                ]
            },
            COL_REPLACEMENT_MAP: {
                "replacements": [
                    {"original": "Mara", "label": "first_name", "synthetic": "[REDACTED_FIRST_NAME]"},
                ]
            },
        }
    )

    result = render_record_html(row)

    assert "first_name" in result
    assert "| last_name" not in result
    assert "| city" not in result


def test_render_record_html_does_not_fallback_to_detected_when_final_entities_empty() -> None:
    row = pd.Series(
        {
            "text": "BackupAgent config_path=/opt/app/config.yaml",
            "text_replaced": "BackupAgent config_path=/opt/app/config.yaml",
            COL_DETECTED_ENTITIES: {
                "entities": [
                    {"value": "BackupAgent", "label": "process_name", "start_position": 0, "end_position": 11},
                    {"value": "/opt/app/config.yaml", "label": "file_path", "start_position": 24, "end_position": 44},
                ]
            },
            COL_FINAL_ENTITIES: {"entities": []},
            COL_REPLACEMENT_MAP: {"replacements": []},
        }
    )

    result = render_record_html(row)

    assert "| process_name" not in result
    assert "| file_path" not in result


def test_render_record_html_replaced_tags_positioned_correctly_with_case_mismatch() -> None:
    """Tags in replaced text must not drift when entity value case differs from actual text."""
    row = pd.Series(
        {
            "text": "Hi Mara at the Lantern in Austin TX",
            "text_replaced": "Hi [REDACTED_NAME] at [REDACTED_COMPANY] in [REDACTED_CITY] TX",
            COL_DETECTED_ENTITIES: {
                "entities": [
                    {"value": "Mara", "label": "first_name", "start_position": 3, "end_position": 7},
                    {"value": "The Lantern", "label": "company_name", "start_position": 11, "end_position": 22},
                    {"value": "Austin", "label": "city", "start_position": 26, "end_position": 32},
                ]
            },
            COL_REPLACEMENT_MAP: {
                "replacements": [
                    {"original": "Mara", "label": "first_name", "synthetic": "[REDACTED_NAME]"},
                    {"original": "The Lantern", "label": "company_name", "synthetic": "[REDACTED_COMPANY]"},
                    {"original": "Austin", "label": "city", "synthetic": "[REDACTED_CITY]"},
                ]
            },
        }
    )
    result = render_record_html(row)
    assert "[REDACTED_CITY]" in result
    assert "city" in result
    assert "[REDACTED| city" not in result


def test_render_record_html_mask_strategy_labels_are_distinct() -> None:
    """When all replacements are identical (e.g. '*****'), each entity keeps its own label."""
    row = pd.Series(
        {
            "text": "Mara in Austin",
            "text_replaced": "***** in *****",
            COL_DETECTED_ENTITIES: {
                "entities": [
                    {"value": "Mara", "label": "first_name", "start_position": 0, "end_position": 4},
                    {"value": "Austin", "label": "city", "start_position": 8, "end_position": 14},
                ]
            },
            COL_REPLACEMENT_MAP: {
                "replacements": [
                    {"original": "Mara", "label": "first_name", "synthetic": "*****"},
                    {"original": "Austin", "label": "city", "synthetic": "*****"},
                ]
            },
        }
    )
    result = render_record_html(row)
    assert "first_name" in result
    assert "city" in result


def test_build_original_entities_from_map_case_insensitive() -> None:
    """Fallback map scanning finds entities regardless of case."""
    from anonymizer.interface.display import _build_original_entities_from_map

    replacement_map = [{"original": "The Lantern", "label": "company_name"}]
    text = "She works at the Lantern daily"
    result = _build_original_entities_from_map(replacement_map, text)
    assert len(result) == 1
    assert result[0].value == "the Lantern"
    assert result[0].start_position == 13


def test_display_record_cycles_on_repeated_calls() -> None:
    preview = _make_preview(rows=2)
    assert preview._display_cycle_index == 0
    preview.display_record()
    assert preview._display_cycle_index == 1
    preview.display_record()
    assert preview._display_cycle_index == 0


def test_display_record_explicit_index_does_not_advance_cycle() -> None:
    preview = _make_preview(rows=2)
    preview.display_record(index=1)
    assert preview._display_cycle_index == 0


def test_build_replaced_entities_no_drift_with_case_mismatch() -> None:
    """Regression for #15: case-insensitive expanded entities must resolve correctly.

    expand_entity_occurrences stores value=text[start:end] which may differ in case
    from the replacement map key (e.g. "the Lantern" vs "The Lantern"). The old
    cursor-replay approach missed the map entry and used a wrong-length fallback,
    causing cumulative drift on all subsequent entities.
    """
    original_text = "Mara works at The Lantern in Austin. the Lantern is her home. Luis visits."
    original_entities = [
        _entity("Mara", "first_name", 0, 4),
        _entity("The Lantern", "company_name", 14, 25),
        _entity("Austin", "city", 29, 35),
        _entity("the Lantern", "company_name", 37, 48),
        _entity("Luis", "first_name", 65, 69),
    ]
    replacement_map = [
        {"original": "Mara", "label": "first_name", "synthetic": "Leila"},
        {"original": "The Lantern", "label": "company_name", "synthetic": "The Ember"},
        {"original": "Austin", "label": "city", "synthetic": "Boulder"},
        {"original": "Luis", "label": "first_name", "synthetic": "Diego"},
    ]
    replaced_text = "Leila works at The Ember in Boulder. The Ember is her home. Diego visits."

    result = _build_replaced_entities(original_entities, replacement_map, original_text, replaced_text)
    assert len(result) == 5
    for entity in result:
        actual = replaced_text[entity.start_position : entity.end_position]
        assert actual == entity.value, (
            f"expected {entity.value!r} at [{entity.start_position}:{entity.end_position}], got {actual!r}"
        )


def test_build_replaced_entities_no_drift_when_entity_absent_from_map() -> None:
    """When an entity is not in the map, its original text span stays; no drift on later entities."""
    original_text = "Contact Sofia and Diego at Acme"
    original_entities = [
        _entity("Sofia", "first_name", 8, 13),
        _entity("Diego", "first_name", 18, 23),
        _entity("Acme", "organization", 27, 31),
    ]
    replacement_map = [
        {"original": "Diego", "label": "first_name", "synthetic": "Carlos"},
        {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
    ]
    replaced_text = "Contact Sofia and Carlos at NovaCorp"

    result = _build_replaced_entities(original_entities, replacement_map, original_text, replaced_text)
    assert len(result) == 3
    for entity in result:
        actual = replaced_text[entity.start_position : entity.end_position]
        assert actual == entity.value


def test_display_record_out_of_bounds_raises() -> None:
    preview = _make_preview(rows=1)
    with pytest.raises(IndexError, match="out of bounds"):
        preview.display_record(index=5)


# ---------------------------------------------------------------------------
# Rewrite-mode display tests
# ---------------------------------------------------------------------------


def test_render_record_html_rewrite_mode_shows_rewrite_layout() -> None:
    row = pd.Series(
        {
            "text": "Alice works at Acme",
            "text_rewritten": "Beth works at Globex",
            COL_DETECTED_ENTITIES: {
                "entities": [
                    {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
                    {"value": "Acme", "label": "organization", "start_position": 15, "end_position": 19},
                ]
            },
            "utility_score": 0.85,
            "leakage_mass": 0.3,
            "needs_human_review": False,
        }
    )
    result = render_record_html(row, record_index=0)
    assert "Rewrite Preview" in result
    assert "Original" in result
    assert "Rewritten" in result
    assert "Scores" in result
    assert "0.85" in result
    assert "0.30" in result
    assert "Replaced" not in result
    assert "Replacement Map" not in result


def test_render_record_html_rewrite_mode_with_judge_scores() -> None:
    row = pd.Series(
        {
            "text": "Alice works at Acme",
            "text_rewritten": "Beth works at Globex",
            COL_DETECTED_ENTITIES: {"entities": []},
            "utility_score": 0.9,
            "leakage_mass": 0.1,
            "needs_human_review": False,
            "_judge_evaluation": {
                "privacy": {"score": 8, "reasoning": "good privacy"},
                "quality": {"score": 9, "reasoning": "high quality"},
                "naturalness": {"score": 7, "reasoning": "mostly natural"},
            },
        }
    )
    result = render_record_html(row, record_index=0)
    assert "privacy: 8/10" in result
    assert "quality: 9/10" in result
    assert "naturalness: 7/10" in result


def test_render_record_html_rewrite_mode_with_disposition() -> None:
    row = pd.Series(
        {
            "text": "Alice works at Acme",
            "text_rewritten": "Beth works at Globex",
            COL_DETECTED_ENTITIES: {"entities": []},
            "utility_score": 0.85,
            "leakage_mass": 0.3,
            "needs_human_review": False,
            "_sensitivity_disposition": {
                "sensitivity_disposition": [
                    {
                        "entity_value": "Alice",
                        "entity_label": "first_name",
                        "sensitivity": "high",
                        "protection_method_suggestion": "replace",
                    },
                ]
            },
        }
    )
    result = render_record_html(row, record_index=0)
    assert "Entity Disposition" in result
    assert "Alice" in result
    assert "first_name" in result
    assert "high" in result
    assert "replace" in result


def test_render_record_html_replace_mode_unchanged_when_no_rewritten_column() -> None:
    """Existing replace-mode rows still render the Original/Replaced/Replacement Map layout."""
    row = pd.Series(
        {
            "text": "Alice works at Acme",
            "text_replaced": "[REDACTED] works at [REDACTED]",
            COL_DETECTED_ENTITIES: {"entities": []},
            COL_REPLACEMENT_MAP: {"replacements": []},
        }
    )
    result = render_record_html(row)
    assert "Replaced" in result
    assert "Replacement Map" in result
    assert "Rewritten" not in result
    assert "Scores" not in result
