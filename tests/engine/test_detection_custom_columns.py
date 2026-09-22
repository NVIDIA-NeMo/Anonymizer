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

import pytest

from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_REGEX_ACCEPTED_ENTITIES,
    COL_REGEX_ENTITIES,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATED_SEED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
)
from anonymizer.engine.detection.custom_columns import (
    _parse_entity_spans,
    apply_validation_and_finalize,
    apply_validation_to_seed_entities,
    enrich_validation_decisions,
    merge_and_build_candidates,
    parse_detected_entities,
    prepare_validation_inputs,
)


def test_parse_entity_spans_handles_malformed_payload() -> None:
    assert _parse_entity_spans([]) == []


def test_parse_entity_spans_defaults_missing_keys() -> None:
    """After a parquet round-trip some keys may be absent."""
    spans = _parse_entity_spans({"entities": [{"value": "Bob", "label": "first_name"}]})
    assert spans[0].entity_id == ""
    assert spans[0].start_position == 0
    assert spans[0].score == 0.0
    assert spans[0].source == "detector"


def _raw(entities: list[dict[str, Any]]) -> str:
    return json.dumps({"entities": entities})


def test_parse_produces_seed_entities_and_notation() -> None:
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
    row: dict[str, Any] = {
        COL_TEXT: text,
        COL_RAW_DETECTED: raw,
        COL_REGEX_ENTITIES: {"entities": []},
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": []},
    }
    result = parse_detected_entities(row)
    assert len(result[COL_SEED_ENTITIES]["entities"]) == 1
    assert result[COL_SEED_ENTITIES]["entities"][0]["value"] == "(555) 123-4567"
    assert result[COL_TAG_NOTATION] in {"xml", "bracket", "paren", "sentinel"}


def test_parse_excludes_detector_candidates_for_regex_only_label_but_keeps_regex_candidates() -> None:
    text = "allow:TKT-1 deny:TKT-2"
    row: dict[str, Any] = {
        COL_TEXT: text,
        COL_RAW_DETECTED: _raw(
            [
                {
                    "text": "TKT-2",
                    "label": "ticket",
                    "start": 17,
                    "end": 22,
                    "score": 0.9,
                }
            ]
        ),
        COL_REGEX_ENTITIES: {
            "entities": [
                {
                    "id": "ticket_6_11",
                    "value": "TKT-1",
                    "label": "ticket",
                    "start_position": 6,
                    "end_position": 11,
                    "score": 1.0,
                    "source": "regex_user:user:ticket:v1",
                }
            ]
        },
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": []},
    }

    result = parse_detected_entities(row, excluded_entity_labels=["ticket"])

    assert [(entity["value"], entity["start_position"]) for entity in result[COL_SEED_ENTITIES]["entities"]] == [
        ("TKT-1", 6)
    ]


def test_overlapping_fallback_survives_when_longer_candidate_is_dropped() -> None:
    text = "ABC-123"
    row: dict[str, Any] = {
        COL_TEXT: text,
        COL_RAW_DETECTED: _raw(
            [
                {
                    "text": "123",
                    "label": "account_number",
                    "start": 4,
                    "end": 7,
                    "score": 0.8,
                }
            ]
        ),
        COL_REGEX_ENTITIES: {
            "entities": [
                {
                    "id": "secret_0_7",
                    "value": text,
                    "label": "secret",
                    "start_position": 0,
                    "end_position": 7,
                    "score": 1.0,
                    "source": "regex_user:user:secret:v1",
                }
            ]
        },
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": []},
    }

    parse_detected_entities(row)
    prepare_validation_inputs(row)

    assert [(entity["value"], entity["label"]) for entity in row[COL_SEED_ENTITIES]["entities"]] == [
        (text, "secret"),
        ("123", "account_number"),
    ]
    assert {candidate["id"] for candidate in row[COL_SEED_VALIDATION_CANDIDATES]["candidates"]} == {
        "secret_0_7",
        "account_number_4_7",
    }

    row[COL_VALIDATED_ENTITIES] = {"decisions": []}
    kept_result = apply_validation_to_seed_entities(row)
    assert [(entity["value"], entity["label"]) for entity in kept_result[COL_VALIDATED_SEED_ENTITIES]["entities"]] == [
        (text, "secret")
    ]

    row[COL_VALIDATED_ENTITIES] = {
        "decisions": [
            {
                "id": "secret_0_7",
                "value": text,
                "label": "secret",
                "decision": "drop",
                "proposed_label": "",
                "reason": "not sensitive in context",
            }
        ]
    }
    result = apply_validation_to_seed_entities(row)

    assert [(entity["value"], entity["label"]) for entity in result[COL_VALIDATED_SEED_ENTITIES]["entities"]] == [
        ("123", "account_number")
    ]


def test_exact_accepted_duplicate_skips_llm_validation_and_retains_origins() -> None:
    text = "alice@example.com"
    accepted = {
        "id": "email_0_17",
        "value": text,
        "label": "email",
        "start_position": 0,
        "end_position": 17,
        "score": 1.0,
        "source": "regex_builtin:nemo-anonymizer.email.v1",
    }
    row: dict[str, Any] = {
        COL_TEXT: text,
        COL_RAW_DETECTED: _raw(
            [
                {
                    "text": text,
                    "label": "email",
                    "start": 0,
                    "end": 17,
                    "score": 0.9,
                }
            ]
        ),
        COL_REGEX_ENTITIES: {"entities": []},
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": [accepted]},
    }

    parse_detected_entities(row)
    prepare_validation_inputs(row)

    assert row[COL_SEED_VALIDATION_CANDIDATES] == {"candidates": []}
    row[COL_VALIDATED_ENTITIES] = {"decisions": []}
    result = apply_validation_to_seed_entities(row)

    assert result[COL_VALIDATED_SEED_ENTITIES]["entities"] == [
        {
            **accepted,
            "source": "regex_builtin:nemo-anonymizer.email.v1|detector",
        }
    ]


def test_regex_candidate_bypassing_llm_survives_a_drop_decision() -> None:
    entity = {
        "id": "email_6_23",
        "value": "alice@example.com",
        "label": "email",
        "start_position": 6,
        "end_position": 23,
        "score": 1.0,
        "source": "regex_builtin:nemo-anonymizer.email.v1",
    }
    row: dict[str, Any] = {
        COL_TEXT: "Email alice@example.com",
        COL_SEED_ENTITIES: {"entities": [entity]},
        COL_VALIDATED_ENTITIES: {
            "decisions": [
                {
                    "id": "email_6_23",
                    "value": "alice@example.com",
                    "label": "email",
                    "decision": "drop",
                    "proposed_label": "",
                    "reason": "test",
                }
            ]
        },
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": [entity]},
    }

    result = apply_validation_to_seed_entities(row)

    assert result[COL_VALIDATED_SEED_ENTITIES]["entities"] == [entity]


@pytest.mark.parametrize(
    ("accepted_source", "validated_source"),
    [
        ("regex_builtin:nemo-anonymizer.email.v1", "regex_user:user:contact:v1"),
        ("regex_user:user:contact:v1", "regex_builtin:nemo-anonymizer.email.v1"),
    ],
)
def test_user_regex_wins_same_span_across_validation_routes(
    accepted_source: str,
    validated_source: str,
) -> None:
    def entity(label: str, source: str) -> dict[str, Any]:
        return {
            "id": f"{label}_0_17",
            "value": "alice@example.com",
            "label": label,
            "start_position": 0,
            "end_position": 17,
            "score": 1.0,
            "source": source,
        }

    accepted_label = "user_contact" if accepted_source.startswith("regex_user:") else "email"
    validated_label = "user_contact" if validated_source.startswith("regex_user:") else "email"
    accepted = entity(accepted_label, accepted_source)
    validated = entity(validated_label, validated_source)
    row: dict[str, Any] = {
        COL_TEXT: "alice@example.com",
        COL_SEED_ENTITIES: {"entities": [validated]},
        COL_VALIDATED_ENTITIES: {"decisions": []},
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": [accepted]},
    }

    result = apply_validation_to_seed_entities(row)

    assert result[COL_VALIDATED_SEED_ENTITIES]["entities"] == [entity("user_contact", "regex_user:user:contact:v1")]


def test_user_regex_wins_same_span_during_finalization() -> None:
    user_entity = {
        "id": "user_contact_0_17",
        "value": "alice@example.com",
        "label": "user_contact",
        "start_position": 0,
        "end_position": 17,
        "score": 1.0,
        "source": "regex_user:user:contact:v1",
    }
    builtin_entity = {
        **user_entity,
        "id": "email_0_17",
        "label": "email",
        "source": "regex_builtin:nemo-anonymizer.email.v1",
    }
    row: dict[str, Any] = {
        COL_TEXT: "alice@example.com",
        COL_MERGED_ENTITIES: {"entities": [user_entity]},
        COL_VALIDATED_ENTITIES: {"decisions": []},
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": [builtin_entity]},
    }

    result = apply_validation_and_finalize(row)

    assert result[COL_DETECTED_ENTITIES]["entities"] == [user_entity]


def test_merge_and_build_candidates_writes_schema_shaped_payloads() -> None:
    row: dict[str, Any] = {
        COL_TEXT: "Alice works at Acme in Seattle.",
        COL_VALIDATED_SEED_ENTITIES: {
            "entities": [
                {
                    "id": "first_name_0_5",
                    "value": "Alice",
                    "label": "first_name",
                    "start_position": 0,
                    "end_position": 5,
                    "score": 0.95,
                    "source": "detector",
                }
            ]
        },
        COL_AUGMENTED_ENTITIES: {"entities": []},
    }

    result = merge_and_build_candidates(row)
    assert "entities" in result[COL_MERGED_ENTITIES]
    assert isinstance(result[COL_MERGED_ENTITIES]["entities"], list)
    assert "candidates" in result[COL_VALIDATION_CANDIDATES]
    assert isinstance(result[COL_VALIDATION_CANDIDATES]["candidates"], list)


def test_merge_filters_denied_augmentation_before_overlap_resolution() -> None:
    row: dict[str, Any] = {
        COL_TEXT: "Alice Johnson",
        COL_VALIDATED_SEED_ENTITIES: {
            "entities": [
                {
                    "id": "first_name_0_5",
                    "value": "Alice",
                    "label": "first_name",
                    "start_position": 0,
                    "end_position": 5,
                    "score": 0.95,
                    "source": "detector",
                }
            ]
        },
        COL_AUGMENTED_ENTITIES: {
            "entities": [
                {
                    "value": "Alice Johnson",
                    "label": " Full_Name ",
                    "reason": "longer overlapping span",
                }
            ]
        },
    }

    result = merge_and_build_candidates(row, excluded_entity_labels=["full_name"])

    merged = result[COL_MERGED_ENTITIES]["entities"]
    assert [(entity["value"], entity["label"]) for entity in merged] == [("Alice", "first_name")]


def test_validation_reclassification_to_excluded_label_is_filtered_before_augmentation() -> None:
    row: dict[str, Any] = {
        COL_TEXT: "San Diego",
        COL_SEED_ENTITIES: {
            "entities": [
                {
                    "id": "country_0_9",
                    "value": "San Diego",
                    "label": "country",
                    "start_position": 0,
                    "end_position": 9,
                    "score": 0.95,
                    "source": "detector",
                }
            ]
        },
        COL_VALIDATED_ENTITIES: {
            "decisions": [
                {
                    "id": "country_0_9",
                    "value": "San Diego",
                    "label": "country",
                    "decision": "reclass",
                    "proposed_label": "city",
                    "reason": "San Diego is a city",
                }
            ]
        },
    }

    result = apply_validation_to_seed_entities(row, excluded_entity_labels=[" CITY "])

    assert result[COL_VALIDATED_SEED_ENTITIES]["entities"] == []
    assert json.loads(result[COL_SEED_ENTITIES_JSON]) == []
    assert result[COL_INITIAL_TAGGED_TEXT] == "San Diego"


def test_excluded_label_filters_locally_accepted_regex_before_augmentation() -> None:
    accepted = {
        "id": "email_0_17",
        "value": "alice@example.com",
        "label": "email",
        "start_position": 0,
        "end_position": 17,
        "score": 1.0,
        "source": "regex_builtin:nemo-anonymizer.email.v1",
    }
    row: dict[str, Any] = {
        COL_TEXT: "alice@example.com",
        COL_SEED_ENTITIES: {"entities": [accepted]},
        COL_VALIDATED_ENTITIES: {"decisions": []},
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": [accepted]},
    }

    result = apply_validation_to_seed_entities(row, excluded_entity_labels=["email"])

    assert result[COL_VALIDATED_SEED_ENTITIES]["entities"] == []
    assert result[COL_INITIAL_TAGGED_TEXT] == "alice@example.com"


def test_merge_filters_excluded_validated_seed_entities() -> None:
    row: dict[str, Any] = {
        COL_TEXT: "San Diego",
        COL_VALIDATED_SEED_ENTITIES: {
            "entities": [
                {
                    "id": "country_0_9",
                    "value": "San Diego",
                    "label": " City ",
                    "start_position": 0,
                    "end_position": 9,
                    "score": 0.95,
                    "source": "detector",
                }
            ]
        },
        COL_AUGMENTED_ENTITIES: {"entities": []},
    }

    result = merge_and_build_candidates(row, excluded_entity_labels=["city"])

    assert result[COL_MERGED_ENTITIES]["entities"] == []
    assert result[COL_VALIDATION_CANDIDATES]["candidates"] == []
    assert result[COL_MERGED_TAGGED_TEXT] == "San Diego"


def test_merge_filters_regex_only_label_from_augmentation_but_preserves_regex_seed() -> None:
    regex_entity = {
        "id": "ticket_6_11",
        "value": "TKT-1",
        "label": "ticket",
        "start_position": 6,
        "end_position": 11,
        "score": 1.0,
        "source": "regex_user:user:ticket:v1",
    }
    row: dict[str, Any] = {
        COL_TEXT: "allow:TKT-1 deny:TKT-2",
        COL_VALIDATED_SEED_ENTITIES: {"entities": [regex_entity]},
        COL_AUGMENTED_ENTITIES: {"entities": [{"value": "TKT-2", "label": "ticket"}]},
    }

    result = merge_and_build_candidates(row, excluded_augmented_entity_labels=["ticket"])

    assert result[COL_MERGED_ENTITIES]["entities"] == [regex_entity]


def test_merge_does_not_split_regex_only_full_name() -> None:
    regex_entity = {
        "id": "full_name_0_10",
        "value": "John Smith",
        "label": "full_name",
        "start_position": 0,
        "end_position": 10,
        "score": 1.0,
        "source": "regex_user:user:full_name:v1",
    }
    row: dict[str, Any] = {
        COL_TEXT: "John Smith met John",
        COL_VALIDATED_SEED_ENTITIES: {"entities": [regex_entity]},
        COL_AUGMENTED_ENTITIES: {"entities": []},
    }

    result = merge_and_build_candidates(row, excluded_augmented_entity_labels=["full_name"])

    assert result[COL_MERGED_ENTITIES]["entities"] == [regex_entity]


def test_finalize_filters_reclassification_to_excluded_label() -> None:
    row: dict[str, Any] = {
        COL_TEXT: "San Diego",
        COL_MERGED_ENTITIES: {
            "entities": [
                {
                    "id": "country_0_9",
                    "value": "San Diego",
                    "label": "country",
                    "start_position": 0,
                    "end_position": 9,
                    "score": 0.95,
                    "source": "augmenter",
                }
            ]
        },
        COL_VALIDATED_ENTITIES: {
            "decisions": [
                {
                    "id": "country_0_9",
                    "value": "San Diego",
                    "label": "country",
                    "decision": "reclass",
                    "proposed_label": "city",
                    "reason": "San Diego is a city",
                }
            ]
        },
    }

    result = apply_validation_and_finalize(row, excluded_entity_labels=["city"])

    assert result[COL_DETECTED_ENTITIES]["entities"] == []
    assert result[COL_TAGGED_TEXT] == "San Diego"


def test_excluded_label_filters_locally_accepted_regex_during_finalization() -> None:
    accepted = {
        "id": "email_0_17",
        "value": "alice@example.com",
        "label": "email",
        "start_position": 0,
        "end_position": 17,
        "score": 1.0,
        "source": "regex_builtin:nemo-anonymizer.email.v1",
    }
    row: dict[str, Any] = {
        COL_TEXT: "alice@example.com",
        COL_MERGED_ENTITIES: {"entities": []},
        COL_VALIDATED_ENTITIES: {"decisions": []},
        COL_REGEX_ACCEPTED_ENTITIES: {"entities": [accepted]},
    }

    result = apply_validation_and_finalize(row, excluded_entity_labels=["email"])

    assert result[COL_DETECTED_ENTITIES]["entities"] == []
    assert result[COL_TAGGED_TEXT] == "alice@example.com"


def test_enrich_validation_decisions_adds_value_from_candidates() -> None:
    row = {
        COL_VALIDATION_DECISIONS: {
            "decisions": [
                {"id": "id1", "decision": "keep", "proposed_label": "", "reason": "direct identifier"},
                {"id": "id2", "decision": "drop", "proposed_label": "", "reason": "placeholder"},
            ]
        },
        COL_SEED_VALIDATION_CANDIDATES: {
            "candidates": [
                {"id": "id1", "value": "Alice", "label": "first_name", "context_before": "", "context_after": ""},
                {"id": "id2", "value": "name", "label": "first_name", "context_before": "", "context_after": ""},
            ]
        },
    }
    result = enrich_validation_decisions(row)
    decisions = result[COL_VALIDATED_ENTITIES]["decisions"]
    assert decisions[0]["value"] == "Alice"
    assert decisions[1]["value"] == "name"


def test_enrich_validation_decisions_ignores_numeric_value_echo() -> None:
    row = {
        COL_VALIDATION_DECISIONS: {
            "decisions": [
                {
                    "id": "id1",
                    "value": 42,
                    "decision": "keep",
                    "proposed_label": "",
                    "reason": "numeric identifier",
                }
            ]
        },
        COL_SEED_VALIDATION_CANDIDATES: {
            "candidates": [
                {
                    "id": "id1",
                    "value": "42",
                    "label": "account_number",
                    "context_before": "",
                    "context_after": "",
                }
            ]
        },
    }

    result = enrich_validation_decisions(row)

    decisions = result[COL_VALIDATED_ENTITIES]["decisions"]
    assert len(decisions) == 1
    assert decisions[0]["value"] == "42"


def test_enrich_validation_decisions_filters_unknown_ids() -> None:
    row = {
        COL_VALIDATION_DECISIONS: {"decisions": [{"id": "unknown_id", "decision": "keep", "proposed_label": ""}]},
        COL_SEED_VALIDATION_CANDIDATES: {"candidates": []},
    }
    result = enrich_validation_decisions(row)
    assert result[COL_VALIDATED_ENTITIES]["decisions"] == []


def test_enrich_validation_decisions_ignores_non_dict_validation_payload() -> None:
    row = {
        COL_VALIDATION_DECISIONS: "unexpected-string-payload",
        COL_SEED_VALIDATION_CANDIDATES: {
            "candidates": [
                {"id": "id1", "value": "Alice", "label": "first_name", "context_before": "", "context_after": ""}
            ]
        },
    }
    result = enrich_validation_decisions(row)
    assert result[COL_VALIDATED_ENTITIES] == {"decisions": []}


def test_apply_validation_and_finalize_handles_malformed_merged_entities() -> None:
    row: dict[str, Any] = {
        COL_TEXT: "Alice works at Acme.",
        COL_MERGED_ENTITIES: ["bad-shape"],
        COL_VALIDATED_ENTITIES: {"decisions": []},
    }

    result = apply_validation_and_finalize(row)
    assert result[COL_DETECTED_ENTITIES] == {"entities": []}
