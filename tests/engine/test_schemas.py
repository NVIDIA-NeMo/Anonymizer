# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from anonymizer.engine.schemas import (
    EntitiesByValueSchema,
    EntitiesSchema,
    JudgeEvaluationSchema,
    JudgeScoreSchema,
    RawValidationDecisionsSchema,
    SensitivityDispositionSchema,
    ValidatedDecisionsSchema,
    ValidationCandidatesSchema,
    ValidationSkeletonSchema,
    generate_privacy_qa_from_disposition,
)
from anonymizer.engine.schemas.rewrite import EntityDispositionSchema


def test_entities_payload_from_raw_dict() -> None:
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
    assert payload.entities[0].label == "first_name"


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


# ---------------------------------------------------------------------------
# Rewrite schemas
# ---------------------------------------------------------------------------


def _make_entity(**kwargs) -> dict:
    """Factory for a valid EntityDispositionSchema dict; override any field via kwargs."""
    defaults = {
        "id": 1,
        "source": "tagged",
        "category": "direct_identifier",
        "sensitivity": "high",
        "entity_label": "first_name",
        "entity_value": "Alice",
        "does_need_protection": True,
        "protection_reason": "Direct identifier that uniquely identifies the individual.",
        "protection_method_suggestion": "replace",
        "combined_risk_level": "high",
    }
    return {**defaults, **kwargs}


@pytest.fixture()
def mixed_disposition() -> SensitivityDispositionSchema:
    """Disposition with one protected entity (Alice/first_name) and one unprotected (Portland/city)."""
    return SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(id=1),
                _make_entity(
                    id=2,
                    entity_label="city",
                    entity_value="Portland",
                    does_need_protection=False,
                    protection_method_suggestion="left_as_is",
                ),
            ]
        }
    )


# EntityDispositionSchema — protection consistency


def test_entity_disposition_invalid_no_protection_but_method_set() -> None:
    with pytest.raises(ValidationError, match="does_need_protection=False"):
        EntityDispositionSchema.model_validate(
            _make_entity(does_need_protection=False, protection_method_suggestion="replace")
        )


def test_entity_disposition_invalid_needs_protection_but_left_as_is() -> None:
    with pytest.raises(ValidationError, match="does_need_protection=True"):
        EntityDispositionSchema.model_validate(
            _make_entity(does_need_protection=True, protection_method_suggestion="left_as_is")
        )


# SensitivityDispositionSchema — sequential ID validator


def test_sensitivity_disposition_invalid_non_sequential_ids() -> None:
    with pytest.raises(ValidationError, match="sequential"):
        SensitivityDispositionSchema.model_validate(
            {
                "sensitivity_disposition": [
                    _make_entity(id=1),
                    _make_entity(id=3, entity_label="last_name", entity_value="Smith"),
                ],
            }
        )


def test_sensitivity_disposition_invalid_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="sequential"):
        SensitivityDispositionSchema.model_validate(
            {
                "sensitivity_disposition": [
                    _make_entity(id=1),
                    _make_entity(id=1, entity_label="last_name", entity_value="Smith"),
                ],
            }
        )


def test_sensitivity_disposition_entities_needing_protection(mixed_disposition: SensitivityDispositionSchema) -> None:
    protected = mixed_disposition.entities_needing_protection()
    assert len(protected) == 1
    assert protected[0].entity_label == "first_name"


def test_sensitivity_disposition_entities_by_method(mixed_disposition: SensitivityDispositionSchema) -> None:
    replaceable = mixed_disposition.entities_by_method("replace")
    assert len(replaceable) == 1
    assert replaceable[0].entity_label == "first_name"
    left = mixed_disposition.entities_by_method("left_as_is")
    assert len(left) == 1
    assert left[0].entity_label == "city"


def test_sensitivity_disposition_medium_and_high_sensitivity(mixed_disposition: SensitivityDispositionSchema) -> None:
    # Both entities in mixed_disposition have sensitivity=high
    result = mixed_disposition.medium_and_high_sensitivity()
    assert len(result) == 2


def test_sensitivity_disposition_to_rewrite_context(mixed_disposition: SensitivityDispositionSchema) -> None:
    context = mixed_disposition.to_rewrite_context()
    assert "[HIGH]" in context
    assert "first_name" in context
    assert "Alice" in context
    assert "→ replace" in context


def test_sensitivity_disposition_to_rewrite_context_empty_when_all_low() -> None:
    schema = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(
                    id=1, sensitivity="low", does_need_protection=False, protection_method_suggestion="left_as_is"
                ),
            ]
        }
    )
    assert schema.to_rewrite_context() == "No medium or high sensitivity entities identified."


# generate_privacy_qa_from_disposition


def test_generate_privacy_qa_from_schema_only_protected_entities(
    mixed_disposition: SensitivityDispositionSchema,
) -> None:
    qa = generate_privacy_qa_from_disposition(mixed_disposition)
    assert len(qa.items) == 1
    assert "Alice" in qa.items[0].question


def test_generate_privacy_qa_from_dict_input(mixed_disposition: SensitivityDispositionSchema) -> None:
    qa = generate_privacy_qa_from_disposition(mixed_disposition.model_dump())
    assert len(qa.items) == 1
    assert "Alice" in qa.items[0].question


def test_generate_privacy_qa_empty_when_nothing_to_protect() -> None:
    schema = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(id=1, does_need_protection=False, protection_method_suggestion="left_as_is")
            ]
        }
    )
    assert generate_privacy_qa_from_disposition(schema).items == []


def test_generate_privacy_qa_ids_are_sequential() -> None:
    schema = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(id=1),
                _make_entity(id=2, entity_label="last_name", entity_value="Smith"),
            ],
        }
    )
    qa = generate_privacy_qa_from_disposition(schema)
    assert [item.id for item in qa.items] == [1, 2]


# JudgeEvaluationSchema


def test_judge_evaluation_parses_all_rubrics() -> None:
    judge = JudgeEvaluationSchema(
        privacy=JudgeScoreSchema(score=8, reason="good"),
        quality=JudgeScoreSchema(score=7, reason="acceptable"),
        naturalness=JudgeScoreSchema(score=9, reason="fluent"),
    )
    assert judge.privacy.score == 8
    assert judge.quality.score == 7
    assert judge.naturalness.score == 9


def test_judge_evaluation_requires_all_rubrics() -> None:
    with pytest.raises(ValidationError):
        JudgeEvaluationSchema(privacy=JudgeScoreSchema(score=8, reason="good"))
