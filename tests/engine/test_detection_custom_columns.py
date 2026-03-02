# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the custom column generator pipeline steps.

These test the *composed* behavior: raw inputs flowing through
parse → merge → validate → finalize, with the same kinds of
tricky strings and edge cases that real detector output produces.
"""

from __future__ import annotations

import json
from typing import Any

from anonymizer.engine.constants import (
    COL_INITIAL_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
)
from anonymizer.engine.detection.custom_columns import (
    _from_dicts,
    enrich_validation_decisions,
    parse_detected_entities,
)


def test_from_dicts_handles_empty_list() -> None:
    assert _from_dicts([]) == []


def test_from_dicts_defaults_missing_keys() -> None:
    """After a parquet round-trip some keys may be absent."""
    spans = _from_dicts([{"value": "Bob", "label": "first_name"}])
    assert spans[0].entity_id == ""
    assert spans[0].start_position == 0
    assert spans[0].score == 0.0
    assert spans[0].source == "detector"


def _raw(entities: list[dict[str, Any]]) -> str:
    return json.dumps({"entities": entities})


def test_parse_produces_valid_tagged_text_and_notation() -> None:
    text = "Call (555) 123-4567 today"
    raw = _raw(
        [
            {
                "text": "(555) 123-4567",
                "label": "phone_number",
                "start": 5,
                "end": 19,
                "score": 0.95,
            },
        ]
    )
    row: dict[str, Any] = {COL_TEXT: text, COL_RAW_DETECTED: raw}
    result = parse_detected_entities(row)
    assert "(555) 123-4567" in result[COL_INITIAL_TAGGED_TEXT]
    assert "phone_number" in result[COL_INITIAL_TAGGED_TEXT]
    assert result[COL_TAG_NOTATION] in {"xml", "bracket", "paren", "sentinel"}
    assert json.loads(result[COL_SEED_ENTITIES_JSON]) == result[COL_SEED_ENTITIES]


def test_enrich_validation_decisions_adds_value_from_candidates() -> None:
    row = {
        COL_VALIDATION_DECISIONS: {
            "decisions": [
                {"id": "id1", "decision": "keep", "proposed_label": "", "reason": "direct identifier"},
                {"id": "id2", "decision": "drop", "proposed_label": "", "reason": "placeholder"},
            ]
        },
        COL_VALIDATION_CANDIDATES: [
            {"id": "id1", "value": "Alice", "label": "first_name", "context_before": "", "context_after": ""},
            {"id": "id2", "value": "name", "label": "first_name", "context_before": "", "context_after": ""},
        ],
    }
    result = enrich_validation_decisions(row)
    decisions = result[COL_VALIDATED_ENTITIES]["decisions"]
    assert decisions[0]["value"] == "Alice"
    assert decisions[1]["value"] == "name"


def test_enrich_validation_decisions_filters_unknown_ids() -> None:
    row = {
        COL_VALIDATION_DECISIONS: {"decisions": [{"id": "unknown_id", "decision": "keep", "proposed_label": ""}]},
        COL_VALIDATION_CANDIDATES: [],
    }
    result = enrich_validation_decisions(row)
    assert result[COL_VALIDATED_ENTITIES]["decisions"] == []


def test_enrich_validation_decisions_ignores_non_dict_validation_payload() -> None:
    row = {
        COL_VALIDATION_DECISIONS: "unexpected-string-payload",
        COL_VALIDATION_CANDIDATES: [
            {"id": "id1", "value": "Alice", "label": "first_name", "context_before": "", "context_after": ""}
        ],
    }
    result = enrich_validation_decisions(row)
    assert result == row
