# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from anonymizer.engine.schemas import (
    EntitiesByValueSchema,
    EntitiesSchema,
    PrivacyAnswersSchema,
    QACompareResultsSchema,
    QualityAnswersSchema,
    RawValidationDecisionsSchema,
    SensitivityDispositionSchema,
    ValidatedDecisionsSchema,
    ValidationCandidatesSchema,
    ValidationSkeletonSchema,
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


def test_entities_by_value_from_json_string() -> None:
    """Regression: _select_seed_cols JSON-serializes dicts before parquet round-trip."""
    import json

    raw = json.dumps({"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]})
    payload = EntitiesByValueSchema.from_raw(raw)
    assert len(payload.entities_by_value) == 1
    assert payload.entities_by_value[0].value == "Alice"


def test_entities_from_json_string() -> None:
    import json

    raw = json.dumps(
        {
            "entities": [
                {"id": "fn_0_5", "value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}
            ]
        }
    )
    payload = EntitiesSchema.from_raw(raw)
    assert len(payload.entities) == 1
    assert payload.entities[0].value == "Alice"


def test_from_raw_invalid_json_string_returns_empty() -> None:
    payload = EntitiesByValueSchema.from_raw("{bad json")
    assert payload.entities_by_value == []


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
        "needs_protection": True,
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
                    needs_protection=False,
                    protection_method_suggestion="leave_as_is",
                ),
            ]
        }
    )


# EntityDispositionSchema — protection consistency


def test_entity_disposition_invalid_no_protection_but_method_set() -> None:
    with pytest.raises(ValidationError, match="needs_protection=False"):
        EntityDispositionSchema.model_validate(
            _make_entity(needs_protection=False, protection_method_suggestion="replace")
        )


def test_entity_disposition_invalid_needs_protection_but_leave_as_is() -> None:
    with pytest.raises(ValidationError, match="needs_protection=True"):
        EntityDispositionSchema.model_validate(
            _make_entity(needs_protection=True, protection_method_suggestion="leave_as_is")
        )


# SensitivityDispositionSchema — ID normalization


def test_sensitivity_disposition_renumbers_non_sequential_ids() -> None:
    schema = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(id=1),
                _make_entity(id=3, entity_label="last_name", entity_value="Smith"),
            ],
        }
    )
    assert [e.id for e in schema.sensitivity_disposition] == [1, 2]


def test_sensitivity_disposition_renumbers_duplicate_ids() -> None:
    schema = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(id=1),
                _make_entity(id=1, entity_label="last_name", entity_value="Smith"),
            ],
        }
    )
    assert [e.id for e in schema.sensitivity_disposition] == [1, 2]
    assert schema.sensitivity_disposition[0].entity_value == "Alice"
    assert schema.sensitivity_disposition[1].entity_value == "Smith"


def test_sensitivity_disposition_protected_entities(mixed_disposition: SensitivityDispositionSchema) -> None:
    protected = mixed_disposition.protected_entities
    assert len(protected) == 1
    assert protected[0].entity_label == "first_name"


def test_sensitivity_disposition_get_entities_by_method(mixed_disposition: SensitivityDispositionSchema) -> None:
    replaceable = mixed_disposition.get_entities_by_method("replace")
    assert len(replaceable) == 1
    assert replaceable[0].entity_label == "first_name"
    left = mixed_disposition.get_entities_by_method("leave_as_is")
    assert len(left) == 1
    assert left[0].entity_label == "city"


def test_sensitivity_disposition_medium_and_high_sensitivity_entities(
    mixed_disposition: SensitivityDispositionSchema,
) -> None:
    # Both entities in mixed_disposition have sensitivity=high
    result = mixed_disposition.medium_and_high_sensitivity_entities
    assert len(result) == 2


def test_sensitivity_disposition_format_for_rewrite_context(mixed_disposition: SensitivityDispositionSchema) -> None:
    context = mixed_disposition.format_for_rewrite_context()
    assert "[HIGH]" in context
    assert "first_name" in context
    assert "Alice" in context
    assert "→ replace" in context


def test_sensitivity_disposition_format_for_rewrite_context_empty_when_no_protection() -> None:
    schema = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(
                    id=1, sensitivity="low", needs_protection=False, protection_method_suggestion="leave_as_is"
                ),
            ]
        }
    )
    assert schema.format_for_rewrite_context() == "No entities needing protection."


def test_sensitivity_disposition_format_for_rewrite_context_includes_low_when_protected() -> None:
    schema = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(
                    id=1,
                    sensitivity="low",
                    entity_label="city",
                    entity_value="Portland",
                    needs_protection=True,
                    protection_method_suggestion="generalize",
                    combined_risk_level="medium",
                    protection_reason="City combined with other quasi-identifiers enables re-identification",
                ),
            ]
        }
    )
    context = schema.format_for_rewrite_context()
    assert "[LOW]" in context
    assert "Portland" in context
    assert "→ generalize" in context


def test_quality_answers_use_integer_ids() -> None:
    answers = QualityAnswersSchema.model_validate({"answers": [{"id": 1, "answer": "A concise answer"}]})
    assert answers.answers[0].id == 1


def test_privacy_answers_reject_unknown_and_use_integer_ids() -> None:
    with pytest.raises(ValidationError):
        PrivacyAnswersSchema.model_validate(
            {"answers": [{"id": 1, "answer": "unknown", "confidence": 0.9, "reason": "unsupported enum value"}]}
        )


def test_privacy_answers_require_confidence_and_reason() -> None:
    with pytest.raises(ValidationError):
        PrivacyAnswersSchema.model_validate({"answers": [{"id": 1, "answer": "yes"}]})


def test_qa_compare_results_use_integer_ids() -> None:
    results = QACompareResultsSchema.model_validate({"per_item": [{"id": 1, "score": 0.8, "reason": "close match"}]})
    assert results.per_item[0].id == 1


# Context-validated answer coverage


def test_quality_answers_backfill_missing_ids_with_context() -> None:
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}]},
        context={"expected_ids": [1, 2]},
    )
    assert len(result.answers) == 2
    backfilled = next(a for a in result.answers if a.id == 2)
    assert backfilled.answer == ""


def test_quality_answers_accept_complete_with_context() -> None:
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}, {"id": 2, "answer": "no"}]},
        context={"expected_ids": [1, 2]},
    )
    assert len(result.answers) == 2


def test_quality_answers_no_enforcement_without_context() -> None:
    result = QualityAnswersSchema.model_validate({"answers": [{"id": 1, "answer": "yes"}]})
    assert len(result.answers) == 1


def test_privacy_answers_backfill_missing_ids_with_context() -> None:
    result = PrivacyAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "no", "confidence": 0.0, "reason": "not inferable"}]},
        context={"expected_ids": [1, 2]},
    )
    assert len(result.answers) == 2
    backfilled = next(a for a in result.answers if a.id == 2)
    assert backfilled.answer.value == "yes"
    assert backfilled.confidence == 1.0


def test_qa_compare_backfill_missing_ids_with_context() -> None:
    result = QACompareResultsSchema.model_validate(
        {"per_item": [{"id": 1, "score": 0.9}]},
        context={"expected_ids": [1, 2]},
    )
    assert len(result.per_item) == 2
    backfilled = next(a for a in result.per_item if a.id == 2)
    assert backfilled.score == 0.0


def test_quality_answers_tolerate_duplicate_ids() -> None:
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}, {"id": 1, "answer": "no"}, {"id": 2, "answer": "yes"}]},
        context={"expected_ids": [1, 2]},
    )
    assert len(result.answers) == 3


def test_quality_answers_tolerate_extra_ids() -> None:
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}, {"id": 2, "answer": "no"}, {"id": 99, "answer": "yes"}]},
        context={"expected_ids": [1, 2]},
    )
    assert len(result.answers) == 3
