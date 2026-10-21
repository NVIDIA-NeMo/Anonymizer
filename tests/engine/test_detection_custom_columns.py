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
from unittest.mock import Mock

import pytest

from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_SEED_DETECTION_RESULT,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_SEED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.detection.custom_columns import (
    _parse_entity_spans,
    admit_validation_decisions,
    apply_validation_to_seed_entities,
    finalize_entities,
    merge_and_build_candidates,
    parse_detected_entities,
)
from anonymizer.engine.schemas import (
    AdmissionResults,
    AdmissionRoute,
    AdmissionStatus,
    DetectionResult,
    EffectiveLabelPolicy,
    Observation,
    RawValidationDecisionSchema,
    SourceTextRef,
    SourceView,
    ValidationChoice,
    ValidationInventoryEntry,
    ValidationRequestDecisionSet,
    ValidationRequestDecisionSets,
    ValidationRequestInventory,
)
from anonymizer.engine.workflow_columns.detection.config import DetectionTransformConfig, DetectionTransformOperation
from anonymizer.engine.workflow_columns.detection.impl import DetectionTransformGenerator


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


def _validation_row(
    *,
    text: str,
    label: str = "first_name",
    decision: ValidationChoice = ValidationChoice.keep,
    proposed_label: str = "",
    label_policy: EffectiveLabelPolicy | None = None,
) -> dict[str, Any]:
    source_ref = SourceTextRef(record_id="record-1", field_id=COL_TEXT)
    observation = Observation(
        detection_result_id="r1",
        observation_id="o1",
        source_text_ref=source_ref,
        value=text,
        label=label,
        start_position=0,
        end_position=len(text),
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
        admission_route=AdmissionRoute.llm_validation,
        native_score=0.9,
    )
    detection_result = DetectionResult(
        detection_result_id="r1",
        source_view=SourceView(source_text_ref=source_ref, start_position=0, end_position=len(text)),
        source_kind=observation.source_kind,
        source_instance=observation.source_instance,
        observations=(observation,),
    )
    inventory = ValidationRequestInventory(
        request_id="validation-request-1",
        source_text_ref=source_ref,
        context_view=detection_result.source_view,
        source_context=text,
        effective_label_policy=label_policy or EffectiveLabelPolicy(),
        entries=(
            ValidationInventoryEntry(
                alias="c1",
                observation_ref=observation.ref,
                value=text,
                label=label,
                start_position=0,
                end_position=len(text),
            ),
        ),
    )
    return {
        COL_TEXT: text,
        COL_SEED_DETECTION_RESULT: detection_result.model_dump(mode="json"),
        COL_VALIDATION_INVENTORIES: [inventory.model_dump(mode="json")],
        COL_VALIDATION_DECISIONS: ValidationRequestDecisionSets(
            responses=(
                ValidationRequestDecisionSet(
                    request_id=inventory.request_id,
                    decisions=(RawValidationDecisionSchema(id="c1", decision=decision, proposed_label=proposed_label),),
                ),
            )
        ).model_dump(mode="json"),
    }


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
        RECORD_ID_COLUMN: "record-1",
        COL_TEXT: text,
        COL_RAW_DETECTED: raw,
    }
    result = parse_detected_entities(row, detection_result_id="r1", source_instance="gliner")
    assert len(result[COL_SEED_ENTITIES]["entities"]) == 1
    assert result[COL_SEED_ENTITIES]["entities"][0]["value"] == "(555) 123-4567"
    detection_result = DetectionResult.model_validate(result[COL_SEED_DETECTION_RESULT])
    assert detection_result.detection_result_id == "r1"
    assert detection_result.observations[0].observation_id == "o1"
    assert detection_result.source_view.source_text_ref.record_id == "record-1"
    assert result[COL_TAG_NOTATION] in {"xml", "bracket", "paren", "sentinel"}


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
    row = _validation_row(
        text="San Diego",
        label="country",
        decision=ValidationChoice.reclass,
        proposed_label="city",
        label_policy=EffectiveLabelPolicy(excluded_labels=(" CITY ",)),
    )
    admit_validation_decisions(row, excluded_entity_labels=[" CITY "])

    result = apply_validation_to_seed_entities(row)

    assert result[COL_VALIDATED_SEED_ENTITIES]["entities"] == []
    assert json.loads(result[COL_SEED_ENTITIES_JSON]) == []
    assert result[COL_INITIAL_TAGGED_TEXT] == "San Diego"


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


def test_finalize_filters_reclassification_to_excluded_label() -> None:
    row = _validation_row(text="San Diego", label="country", decision=ValidationChoice.reclass, proposed_label="city")
    admit_validation_decisions(row)
    apply_validation_to_seed_entities(row)
    row[COL_AUGMENTED_ENTITIES] = {"entities": []}
    merge_and_build_candidates(row)

    result = finalize_entities(row, excluded_entity_labels=["city"])

    assert result[COL_DETECTED_ENTITIES]["entities"] == []
    assert result[COL_TAGGED_TEXT] == "San Diego"


def test_finalize_filters_reclassification_outside_explicit_label_set() -> None:
    row = _validation_row(text="San Diego", label="country", decision=ValidationChoice.reclass, proposed_label="city")
    admit_validation_decisions(row)
    apply_validation_to_seed_entities(row)
    row[COL_AUGMENTED_ENTITIES] = {"entities": []}
    merge_and_build_candidates(row)

    result = finalize_entities(row, allowed_entity_labels=["country"])

    assert result[COL_DETECTED_ENTITIES]["entities"] == []
    assert result[COL_TAGGED_TEXT] == "San Diego"


def test_validation_rejection_is_not_replayed_against_augmenter_observation() -> None:
    row = _validation_row(text="Alice", decision=ValidationChoice.drop)

    admit_validation_decisions(row)
    apply_validation_to_seed_entities(row)
    assert row[COL_VALIDATED_SEED_ENTITIES] == {"entities": []}
    assert row[COL_ADMITTED_CANDIDATES] == {"candidates": []}

    row[COL_AUGMENTED_ENTITIES] = {"entities": [{"value": "Alice", "label": "first_name"}]}
    merge_and_build_candidates(row)
    finalize_entities(row)

    assert [(item["value"], item["source"]) for item in row[COL_DETECTED_ENTITIES]["entities"]] == [
        ("Alice", "augmenter")
    ]
    assert COL_ADMISSION_RESULTS in row


def test_admit_validation_decisions_only_publishes_admission_results() -> None:
    row = _validation_row(text="Alice")
    input_columns = set(row)

    result = admit_validation_decisions(row)

    assert set(result) - input_columns == {COL_ADMISSION_RESULTS}
    admissions = AdmissionResults.model_validate(result[COL_ADMISSION_RESULTS])
    assert len(admissions.results) == 1
    assert (
        admissions.results[0].observation_ref
        == DetectionResult.model_validate(row[COL_SEED_DETECTION_RESULT]).observations[0].ref
    )
    assert admissions.results[0].effective_label == "first_name"


def test_admission_transform_forwards_configured_label_policy() -> None:
    policy = EffectiveLabelPolicy.from_labels(allowed_labels=["country", "city"], excluded_labels=["city"])
    row = _validation_row(
        text="San Diego",
        label="country",
        decision=ValidationChoice.reclass,
        proposed_label="city",
        label_policy=policy,
    )
    config = DetectionTransformConfig(
        name=COL_ADMISSION_RESULTS,
        operation=DetectionTransformOperation.ADMIT_VALIDATION_DECISIONS,
        allowed_entity_labels=["country", "city"],
        excluded_entity_labels=["city"],
    )
    generator = DetectionTransformGenerator(config, resource_provider=Mock())

    result = generator.generate(row)

    admissions = AdmissionResults.model_validate(result[COL_ADMISSION_RESULTS])
    assert admissions.results[0].status is AdmissionStatus.rejected


@pytest.mark.parametrize("column", [COL_SEED_DETECTION_RESULT, COL_VALIDATION_INVENTORIES, COL_VALIDATION_DECISIONS])
def test_admit_validation_decisions_requires_lifecycle_columns(column: str) -> None:
    row = _validation_row(text="Alice")
    del row[column]

    with pytest.raises(KeyError, match=column):
        admit_validation_decisions(row)


def test_admission_projection_uses_source_value_not_model_echo() -> None:
    row = _validation_row(text="Alice")
    row[COL_VALIDATION_DECISIONS]["responses"][0]["decisions"][0]["value"] = 42

    admit_validation_decisions(row)
    apply_validation_to_seed_entities(row)

    entity = row[COL_VALIDATED_SEED_ENTITIES]["entities"][0]
    assert (entity["value"], entity["label"], entity["start_position"], entity["end_position"]) == (
        "Alice",
        "first_name",
        0,
        5,
    )


def test_finalize_entities_handles_malformed_merged_entities() -> None:
    row = _validation_row(text="Alice works at Acme.")
    admit_validation_decisions(row)
    row[COL_MERGED_ENTITIES] = ["bad-shape"]

    result = finalize_entities(row)
    assert result[COL_DETECTED_ENTITIES] == {"entities": []}
