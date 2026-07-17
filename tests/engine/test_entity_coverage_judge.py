# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.engine.evaluation.entity_coverage_judge import (
    _coverage_prompt,
    _filter_covered_leaked_entities,
    _is_leaked_value_covered,
    _parse_leaked_entities,
)


def test_coverage_prompt_omits_data_summary_context_when_summary_absent() -> None:
    without_summary = _coverage_prompt(entity_labels=None, strict_entity_protection=False)
    with_blank_summary = _coverage_prompt(
        entity_labels=None,
        strict_entity_protection=False,
        data_summary="   ",
    )

    assert without_summary == with_blank_summary
    assert "<data_summary_context>" not in without_summary


def test_coverage_prompt_includes_data_summary_as_interpretive_context() -> None:
    prompt = _coverage_prompt(
        entity_labels=["first_name"],
        strict_entity_protection=False,
        data_summary="Customer support transcripts.",
    )

    assert "<data_summary_context>\nCustomer support transcripts.\n" in prompt
    assert "Use this context only to interpret literal values and their semantic types." in prompt
    assert "Do not infer or invent entities that are absent from the original text." in prompt


def test_filter_covered_leaked_entities_removes_subspans_and_composites() -> None:
    detected = [
        {"value": "Mstr Marzella", "label": "givenname"},
        {"value": "Nawabganj", "label": "city"},
        {"value": "382210", "label": "zipcode"},
        {"value": "44 Dunsfold Drive", "label": "street"},
        {"value": "Chihuahuan Desert", "label": "location"},
        {"value": "Annex Building", "label": "place_name"},
    ]
    leaked = [
        {"value": "Mstr", "label": "title"},  # subspan of a single final
        {"value": "Nawabganj - 382210", "label": "city"},  # composite of two whole finals
        {"value": "44", "label": "buildingnum"},  # short subspan
        # "Chihuahuan Desert Festival" adds the content token "festival" on top of the
        # detected "Chihuahuan Desert" — a named event, so it is a real leak (NOT covered).
        {"value": "Chihuahuan Desert Festival", "label": "event"},
        {"value": "m", "label": "sex"},  # short token, not covered
        {"value": "Ann", "label": "first_name"},  # partial token of "Annex", not covered
        {"value": "uncovered value", "label": "unique_id"},
    ]

    assert _filter_covered_leaked_entities(leaked, detected) == [
        {"value": "Chihuahuan Desert Festival", "label": "event"},
        {"value": "m", "label": "sex"},
        {"value": "Ann", "label": "first_name"},
        {"value": "uncovered value", "label": "unique_id"},
    ]


@pytest.mark.parametrize(
    ("leaked_value", "final_values"),
    [
        ("Mstr", ["Mstr Marzella"]),  # subspan of a single final entity
        ("the Nawabganj", ["Nawabganj"]),  # grammatical stopword ignored
        ("44", ["44 Dunsfold Drive"]),  # short numeric subspan
        ("Nawabganj - 382210", ["Nawabganj", "382210"]),  # composite of whole finals
        ("Nawabganj", ["Nawabganj", "382210"]),  # exact match against one final
        ("José", ["José García"]),  # accented subspan (Unicode tokenizer)
        ("Zürich", ["Zürich"]),  # accented exact match
    ],
)
def test_is_leaked_value_covered_true(leaked_value: str, final_values: list[str]) -> None:
    assert _is_leaked_value_covered(leaked_value, final_values) is True


@pytest.mark.parametrize(
    ("leaked_value", "final_values"),
    [
        # Cross-entity: pieces come from unrelated final entities -> a real, distinct leak.
        ("John Smith", ["John Doe", "Jane Smith"]),
        # Content descriptor is NOT ignored: a named event is a distinct leak.
        ("Davos Summit", ["Davos"]),
        ("Chihuahuan Desert Festival", ["Chihuahuan Desert"]),
        # Partial-token substrings must NOT count as covered (no raw substring matching).
        ("Ann", ["Annex Building"]),
        ("Sara", ["Sarah Connor"]),
        ("ana", ["Banana Republic"]),
        # Short-token safeguard: a single letter is not covered by a longer token it prefixes.
        ("m", ["Mstr Marzella"]),
        # Nothing in common.
        ("uncovered value", ["Mstr Marzella", "Nawabganj"]),
        # No final entities -> nothing can be covered.
        ("Alice", []),
    ],
)
def test_is_leaked_value_covered_false(leaked_value: str, final_values: list[str]) -> None:
    assert _is_leaked_value_covered(leaked_value, final_values) is False


def test_filter_covered_leaked_entities_keeps_cross_entity_reconstruction() -> None:
    """A leak whose tokens are spread across unrelated final entities is a real leak."""
    detected = [
        {"value": "John Doe", "label": "first_name"},
        {"value": "Jane Smith", "label": "first_name"},
    ]
    leaked = [{"value": "John Smith", "label": "first_name"}]

    assert _filter_covered_leaked_entities(leaked, detected) == leaked


def test_filter_covered_leaked_entities_passthrough_on_no_final_entities() -> None:
    leaked = [{"value": "Alice", "label": "first_name"}]

    assert _filter_covered_leaked_entities(leaked, []) == leaked
    assert _filter_covered_leaked_entities(leaked, None) == leaked


def test_parse_leaked_entities_accepts_gemma_thought_prefixed_json_fence() -> None:
    raw = """thought
```json
{
  "leaked_entities": [
    {
      "value": "Alice",
      "label": "givenname",
      "reasoning": "The given name was not detected."
    }
  ]
}
```"""

    assert _parse_leaked_entities(raw) == [
        {
            "value": "Alice",
            "label": "givenname",
            "reasoning": "The given name was not detected.",
        }
    ]


def test_coverage_prompt_extracts_independently_before_deterministic_filtering() -> None:
    prompt = _coverage_prompt(entity_labels=["sex", "title"], strict_entity_protection=False)

    assert "Work independently from the anonymizer" in prompt
    assert "deterministic postprocessing step" in prompt
    assert "<anonymizer_final_entities>" not in prompt
    assert "_final_entities_for_coverage_judge" not in prompt


def test_coverage_prompt_requires_systematic_structured_text_scan() -> None:
    prompt = _coverage_prompt(entity_labels=["sex", "title"], strict_entity_protection=False)

    assert "salutations, signatures" in prompt
    assert "Tables, bullets, forms" in prompt
    assert "Short or single-token values" in prompt
    assert "Honorifics attached to person names" in prompt
