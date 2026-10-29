# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging

import pandas as pd
import pytest

from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_REPLACEMENT_APPLICATION, COL_TEXT
from anonymizer.engine.detection.backend import _GlinerDetectionBackend
from anonymizer.engine.detection.postprocess import project_detection_result_to_entity_spans
from anonymizer.engine.replace.strategies import apply_local_replace_strategy
from anonymizer.engine.schemas import DetectionResult, SourceTextRef, SourceView


def _build_result(
    *,
    canonical_text: str,
    raw_entities: list[dict[str, object]] | None = None,
    view_start: int = 0,
    view_end: int | None = None,
    raw_output: str | None = None,
) -> DetectionResult:
    source_text_ref = SourceTextRef(record_id="record-1", field_id="text")
    return _GlinerDetectionBackend(source_instance="gliner-pii-detector").to_detection_result(
        detection_result_id="r1",
        source_view=SourceView(
            source_text_ref=source_text_ref,
            start_position=view_start,
            end_position=len(canonical_text) if view_end is None else view_end,
        ),
        canonical_text=canonical_text,
        raw_output=raw_output if raw_output is not None else json.dumps({"entities": raw_entities or []}),
    )


def test_gliner_backend_assigns_ids_after_deterministic_sort() -> None:
    result = _build_result(
        canonical_text="Alice Johnson",
        raw_entities=[
            {
                "text": "Alice Johnson",
                "label": "full_name",
                "start": 0,
                "end": 13,
                "score": 0.9,
            },
            {
                "text": "Alice",
                "label": "first_name",
                "start": 0,
                "end": 5,
                "score": 0.8,
            },
        ],
    )

    assert [(item.observation_id, item.value) for item in result.observations] == [
        ("o1", "Alice"),
        ("o2", "Alice Johnson"),
    ]


def test_gliner_backend_retains_exact_duplicates_and_overlaps() -> None:
    result = _build_result(
        canonical_text="Alice Johnson",
        raw_entities=[
            {
                "text": "Alice",
                "label": "first_name",
                "start": 0,
                "end": 5,
                "score": 0.9,
            },
            {
                "text": "Alice",
                "label": "first_name",
                "start": 0,
                "end": 5,
                "score": 0.8,
            },
            {
                "text": "Alice Johnson",
                "label": "full_name",
                "start": 0,
                "end": 13,
                "score": 0.7,
            },
        ],
    )

    assert [item.observation_id for item in result.observations] == ["o1", "o2", "o3"]
    assert [item.native_score for item in result.observations] == [0.9, 0.8, 0.7]


def test_gliner_backend_retains_same_span_labels_in_deterministic_order() -> None:
    result = _build_result(
        canonical_text="She called Mum every day.",
        raw_entities=[
            {"text": "Mum", "label": "relationship", "start": 11, "end": 14, "score": 0.941},
            {"text": "Mum", "label": "last_name", "start": 11, "end": 14, "score": 0.719},
        ],
    )

    assert [(item.label, item.native_score) for item in result.observations] == [
        ("last_name", 0.719),
        ("relationship", 0.941),
    ]


@pytest.mark.parametrize(
    "raw_output",
    ["not json {{{", '{"entities": "not a list"}', "[]", "null"],
)
def test_gliner_backend_ignores_invalid_payloads(raw_output: str) -> None:
    assert _build_result(canonical_text="hello", raw_output=raw_output).observations == ()


def test_gliner_backend_ignores_non_object_entities() -> None:
    result = _build_result(
        canonical_text="hello",
        raw_output=json.dumps(
            {"entities": ["string_item", 42, None, {"text": "hello", "label": "email", "start": 0, "end": 5}]}
        ),
    )

    assert [item.value for item in result.observations] == ["hello"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"text": ""},
        {"label": ""},
        {"start": -1},
        {"start": 3, "end": 3},
        {"end": 999},
        {"start": "abc"},
        {"end": "abc"},
    ],
)
def test_gliner_backend_ignores_invalid_entities(overrides: dict[str, object]) -> None:
    entity: dict[str, object] = {"text": "hello", "label": "email", "start": 0, "end": 5, "score": 0.9}
    entity.update(overrides)

    assert _build_result(canonical_text="hello", raw_entities=[entity]).observations == ()


@pytest.mark.parametrize("field", ["text", "label"])
@pytest.mark.parametrize("invalid_value", [None, True, 42, [], {}])
def test_gliner_backend_skips_non_string_fields_without_losing_valid_observations(
    field: str, invalid_value: object
) -> None:
    invalid_entity: dict[str, object] = {"text": "Alice", "label": "first_name", "start": 0, "end": 5}
    invalid_entity[field] = invalid_value
    result = _build_result(
        canonical_text="Alice",
        raw_entities=[invalid_entity, {"text": "Alice", "label": "first_name", "start": 0, "end": 5}],
    )

    assert [(item.value, item.label) for item in result.observations] == [("Alice", "first_name")]


@pytest.mark.parametrize("field", ["start", "end"])
@pytest.mark.parametrize("invalid_offset", [True, False, 1.7, float("nan"), float("inf"), float("-inf")])
def test_gliner_backend_rejects_boolean_and_non_integral_offsets(field: str, invalid_offset: object) -> None:
    entity: dict[str, object] = {"text": "Alice", "label": "first_name", "start": 1, "end": 6}
    entity[field] = invalid_offset

    assert _build_result(canonical_text=" Alice ", raw_entities=[entity]).observations == ()


def test_gliner_backend_accepts_whole_number_float_offsets() -> None:
    result = _build_result(
        canonical_text=" Alice ",
        raw_entities=[{"text": "Alice", "label": "first_name", "start": 1.0, "end": 6.0}],
    )

    assert [(item.value, item.start_position, item.end_position) for item in result.observations] == [("Alice", 1, 6)]


@pytest.mark.parametrize("value", [" Alice", "Alice ", "\tAlice\n", "\u2003Alice\u00a0"])
def test_gliner_backend_trims_source_whitespace_and_preserves_it_during_replacement(value: str) -> None:
    text = f"prefix {value}suffix"
    result = _build_result(
        canonical_text=text,
        view_start=7,
        view_end=7 + len(value),
        raw_entities=[{"text": value, "label": "first_name", "start": 0, "end": len(value)}],
    )
    observation = result.observations[0]
    expected_start = text.index("Alice")
    assert (observation.value, observation.start_position, observation.end_position) == (
        "Alice",
        expected_start,
        expected_start + 5,
    )

    entities = project_detection_result_to_entity_spans(result)
    replaced = apply_local_replace_strategy(
        pd.DataFrame({COL_TEXT: [text], COL_FINAL_ENTITIES: [{"entities": [entities[0].as_dict()]}]}),
        strategy=Redact(),
    )
    assert replaced[COL_REPLACED_TEXT].iloc[0] == text.replace("Alice", "[REDACTED_FIRST_NAME]")


def test_gliner_backend_logs_malformed_json_without_source_payload(caplog: pytest.LogCaptureFixture) -> None:
    payload = '{"name":"Alice","ssn":"123-45-6789",invalid}'
    with caplog.at_level(logging.WARNING, logger="anonymizer.engine.detection.backend"):
        result = _build_result(canonical_text="hello", raw_output=payload)

    assert result.observations == ()
    assert any("Failed to parse GLiNER JSON output" in message for message in caplog.messages)
    assert any("length=" in message for message in caplog.messages)
    assert payload not in "\n".join(caplog.messages)


def test_gliner_source_value_projection_is_redacted_after_detector_normalization() -> None:
    text = "Alice joined Acme"
    result = _build_result(
        canonical_text=text,
        raw_entities=[{"text": "alice", "label": "first_name", "start": 0, "end": 5, "score": 0.9}],
    )
    entities = project_detection_result_to_entity_spans(result)
    assert entities[0].value == "Alice"

    replaced = apply_local_replace_strategy(
        pd.DataFrame({COL_TEXT: [text], COL_FINAL_ENTITIES: [{"entities": [entities[0].as_dict()]}]}),
        strategy=Redact(),
    )
    assert replaced[COL_REPLACED_TEXT].iloc[0] == "[REDACTED_FIRST_NAME] joined Acme"
    assert replaced[COL_REPLACEMENT_APPLICATION].iloc[0] == {
        "targeted_span_count": 1,
        "applied_span_count": 1,
        "skipped_span_count": 0,
        "skipped_span_label_counts": {},
    }


@pytest.mark.parametrize("raw_score", [0, 0.0, "0"])
def test_gliner_backend_preserves_zero_scores(raw_score: object) -> None:
    result = _build_result(
        canonical_text="Alice",
        raw_entities=[
            {
                "text": "Alice",
                "label": "first_name",
                "start": 0,
                "end": 5,
                "score": raw_score,
            }
        ],
    )

    assert result.observations[0].native_score == 0.0


@pytest.mark.parametrize(
    "score_payload",
    [
        {},
        {"score": "nan"},
        {"score": float("inf")},
        {"score": "abc"},
        {"score": True},
    ],
)
def test_gliner_backend_maps_unusable_scores_to_none(score_payload: dict[str, object]) -> None:
    result = _build_result(
        canonical_text="Alice",
        raw_entities=[
            {
                "text": "Alice",
                "label": "first_name",
                "start": 0,
                "end": 5,
                **score_payload,
            }
        ],
    )

    assert result.observations[0].native_score is None


def test_gliner_backend_orders_scored_duplicate_before_unscored_duplicate() -> None:
    result = _build_result(
        canonical_text="Alice",
        raw_entities=[
            {
                "text": "Alice",
                "label": "first_name",
                "start": 0,
                "end": 5,
            },
            {
                "text": "Alice",
                "label": "first_name",
                "start": 0,
                "end": 5,
                "score": 0.7,
            },
        ],
    )

    assert [observation.native_score for observation in result.observations] == [0.7, None]


def test_gliner_backend_maps_view_relative_offsets_to_canonical_source() -> None:
    result = _build_result(
        canonical_text="prefix Alice suffix",
        view_start=7,
        view_end=12,
        raw_entities=[
            {
                "text": "alice",
                "label": "first_name",
                "start": 0,
                "end": 5,
                "score": 0.9,
            }
        ],
    )

    observation = result.observations[0]
    assert observation.value == "Alice"
    assert (observation.start_position, observation.end_position) == (7, 12)


def test_gliner_backend_skips_source_mismatch_without_losing_valid_observations(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer.engine.detection.backend"):
        result = _build_result(
            canonical_text="Alice met Bob",
            raw_entities=[
                {"text": "Carol", "label": "first_name", "start": 0, "end": 5, "score": 0.9},
                {"text": "Bob", "label": "first_name", "start": 10, "end": 13, "score": 0.8},
            ],
        )

    assert [(item.value, item.start_position, item.end_position) for item in result.observations] == [("Bob", 10, 13)]
    assert "source offsets [0, 5)" in caplog.text
    assert all(value not in caplog.text for value in ("Alice", "Carol", "Bob", "first_name"))


def test_gliner_backend_publishes_empty_result_for_no_valid_observations() -> None:
    result = _build_result(
        canonical_text="Alice",
        raw_entities=[
            {
                "text": "Alice",
                "label": "first_name",
                "start": -1,
                "end": 5,
                "score": 0.9,
            }
        ],
    )

    assert result.observations == ()
