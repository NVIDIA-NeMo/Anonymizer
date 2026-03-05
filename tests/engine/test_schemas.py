# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from anonymizer.engine.schemas import (
    EntitiesByValueSchema,
    EntitiesSchema,
    RawValidationDecisionsSchema,
    ValidatedDecisionsSchema,
    ValidationCandidatesSchema,
    ValidationSkeletonSchema,
)


def test_entities_payload_from_raw_list() -> None:
    raw = {
        "entities": [
            {
                "id": "first_name_0_5",
                "value": "Alice",
                "label": "first_name",
                "start_position": 0,
                "end_position": 5,
                "score": 0.9,
                "source": "detector",
            }
        ]
    }
    payload = EntitiesSchema.from_raw(raw)
    assert len(payload.entities) == 1
    assert payload.entities[0].value == "Alice"


def test_entities_payload_from_raw_dict() -> None:
    raw = {
        "entities": [
            {
                "id": "org_15_19",
                "value": "Acme",
                "label": "organization",
                "start_position": 15,
                "end_position": 19,
                "score": 0.8,
                "source": "augmenter",
            }
        ]
    }
    payload = EntitiesSchema.from_raw(raw)
    assert len(payload.entities) == 1
    assert payload.entities[0].label == "organization"


def test_entities_payload_from_raw_model_instance() -> None:
    raw = EntitiesSchema(
        entities=[
            {
                "id": "org_15_19",
                "value": "Acme",
                "label": "organization",
                "start_position": 15,
                "end_position": 19,
                "score": 0.8,
                "source": "augmenter",
            }
        ]
    )
    payload = EntitiesSchema.from_raw(raw)
    assert len(payload.entities) == 1
    assert payload.entities[0].label == "organization"


def test_entities_payload_from_raw_numpy_array() -> None:
    raw = {
        "entities": np.array(
            [{"id": "city_23_30", "value": "Seattle", "label": "city", "start_position": 23, "end_position": 30}],
            dtype=object,
        )
    }
    payload = EntitiesSchema.from_raw(raw)
    assert len(payload.entities) == 1
    assert payload.entities[0].value == "Seattle"


def test_entities_payload_from_raw_dict_wrapping_numpy_array() -> None:
    """Regression: parquet round-trips produce {"entities": numpy_array}."""
    raw = {
        "entities": np.array(
            [{"id": "city_23_30", "value": "Seattle", "label": "city", "start_position": 23, "end_position": 30}],
            dtype=object,
        )
    }
    payload = EntitiesSchema.from_raw(raw)
    assert len(payload.entities) == 1
    assert payload.entities[0].value == "Seattle"


def test_entities_payload_from_raw_invalid_returns_empty() -> None:
    payload = EntitiesSchema.from_raw("not-a-payload")
    assert payload.entities == []


def test_entities_payload_from_malformed_list_returns_empty() -> None:
    payload = EntitiesSchema.from_raw(["not-an-entity"])
    assert payload.entities == []


def test_entities_payload_from_bare_list_returns_empty() -> None:
    payload = EntitiesSchema.from_raw(
        [{"id": "first_name_0_5", "value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]
    )
    assert payload.entities == []


def test_validation_candidates_payload_from_raw_list() -> None:
    payload = ValidationCandidatesSchema.from_raw(
        {
            "candidates": [
                {
                    "id": "city_3_10",
                    "value": "Seattle",
                    "label": "city",
                    "context_before": "in ",
                    "context_after": ", WA",
                }
            ]
        }
    )
    assert len(payload.candidates) == 1
    assert payload.candidates[0].label == "city"


def test_raw_validation_decisions_payload_from_raw_list() -> None:
    payload = RawValidationDecisionsSchema.from_raw(
        {"decisions": [{"id": "city_3_10", "decision": "keep", "proposed_label": "", "reason": "quasi-identifier"}]}
    )
    assert len(payload.decisions) == 1
    assert payload.decisions[0].decision is not None
    assert payload.decisions[0].decision.value == "keep"


def test_raw_validation_decisions_payload_from_malformed_list_returns_empty() -> None:
    payload = RawValidationDecisionsSchema.from_raw({"decisions": ["bad-item"]})
    assert payload.decisions == []


def test_validated_decisions_payload_from_raw_list() -> None:
    payload = ValidatedDecisionsSchema.from_raw(
        {
            "decisions": [
                {
                    "id": "city_3_10",
                    "decision": "keep",
                    "proposed_label": "",
                    "reason": "quasi-identifier",
                    "value": "Seattle",
                    "label": "city",
                }
            ]
        }
    )
    assert len(payload.decisions) == 1
    assert payload.decisions[0].value == "Seattle"


def test_validation_skeleton_payload_from_raw_dict() -> None:
    payload = ValidationSkeletonSchema.from_raw(
        {
            "decisions": [
                {
                    "id": "city_3_10",
                    "value": "Seattle",
                    "label": "city",
                    "decision": None,
                    "proposed_label": None,
                    "reason": None,
                }
            ]
        }
    )
    assert len(payload.decisions) == 1
    assert payload.decisions[0].decision is None


def test_entities_by_value_payload_from_raw_list() -> None:
    payload = EntitiesByValueSchema.from_raw({"entities_by_value": [{"value": "Seattle", "labels": ["city"]}]})
    assert len(payload.entities_by_value) == 1
    assert payload.entities_by_value[0].labels == ["city"]
