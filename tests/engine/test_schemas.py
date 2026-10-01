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
from anonymizer.engine.schemas.rewrite import (
    EntityDispositionSchema,
    StrictEntityDispositionSchema,
    StrictSensitivityDispositionSchema,
)


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


def test_raw_validation_decision_normalizes_explicit_null_proposed_label() -> None:
    payload = RawValidationDecisionsSchema.model_validate(
        {"decisions": [{"id": "city_3_10", "decision": "keep", "proposed_label": None}]}
    )
    assert payload.decisions[0].proposed_label == ""


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
        "protection_reason": "Direct identifier that uniquely identifies the individual.",
        "protection_method_suggestion": "replace",
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
                    sensitivity="low",
                    protection_method_suggestion="leave_as_is",
                ),
            ],
        }
    )


# EntityDispositionSchema — protection consistency


@pytest.mark.parametrize("method", ["replace", "generalize", "remove", "suppress_inference"])
def test_entity_disposition_low_sensitivity_with_protection_promotes_to_medium(method: str) -> None:
    entity = EntityDispositionSchema.model_validate(
        _make_entity(sensitivity="low", protection_method_suggestion=method)
    )
    assert entity.sensitivity == "medium"
    assert entity.protection_method_suggestion == method
    # Coerced and non-coerced paths must produce the same serialization type (plain string, not enum).
    dumped = entity.model_dump()
    assert dumped["sensitivity"] == "medium"
    assert type(dumped["sensitivity"]) is str
    assert "combined_risk_level" not in dumped


@pytest.mark.parametrize("sensitivity", ["medium", "high"])
def test_entity_disposition_rejects_protected_level_with_leave_as_is(sensitivity: str) -> None:
    with pytest.raises(ValidationError, match="cannot have protection_method_suggestion='leave_as_is'"):
        EntityDispositionSchema.model_validate(
            _make_entity(sensitivity=sensitivity, protection_method_suggestion="leave_as_is")
        )


def test_entity_disposition_accepts_low_leave_as_is() -> None:
    entity = EntityDispositionSchema.model_validate(
        _make_entity(sensitivity="low", protection_method_suggestion="leave_as_is")
    )
    assert entity.needs_protection is False


# SensitivityDispositionSchema — ID identity and plan validation


def _plan_names_entity_1() -> dict:
    return {"mandatory_protections": [{"entity_id": 1, "requirement": "Always replace person names."}]}


def test_sensitivity_disposition_rejects_non_sequential_ids() -> None:
    with pytest.raises(ValidationError, match="sequential 1..2"):
        SensitivityDispositionSchema.model_validate(
            {
                "sensitivity_disposition": [
                    _make_entity(id=1),
                    _make_entity(id=3, entity_label="last_name", entity_value="Smith"),
                ],
            }
        )


def test_sensitivity_disposition_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="sequential 1..2"):
        SensitivityDispositionSchema.model_validate(
            {
                "sensitivity_disposition": [
                    _make_entity(id=1),
                    _make_entity(id=1, entity_label="last_name", entity_value="Smith"),
                ],
            }
        )


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
    # Alice is high; Portland is low (leave_as_is), so only Alice qualifies.
    result = mixed_disposition.medium_and_high_sensitivity_entities
    assert [e.entity_label for e in result] == ["first_name"]


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
                    id=1,
                    sensitivity="low",
                    protection_method_suggestion="leave_as_is",
                ),
            ],
        }
    )
    assert schema.format_for_rewrite_context() == "No entities needing protection."


def test_sensitivity_disposition_format_for_rewrite_context_promotes_low_when_protected() -> None:
    schema = SensitivityDispositionSchema.model_validate(
        {
            "sensitivity_disposition": [
                _make_entity(
                    id=1,
                    sensitivity="low",
                    entity_label="city",
                    entity_value="Portland",
                    protection_method_suggestion="generalize",
                    protection_reason="City combined with other quasi-identifiers enables re-identification",
                ),
            ],
        }
    )
    context = schema.format_for_rewrite_context()
    assert "[MEDIUM]" in context  # low + protecting method is promoted to medium
    assert "Portland" in context
    assert "→ generalize" in context


def test_quality_answers_use_integer_ids() -> None:
    answers = QualityAnswersSchema.model_validate({"answers": [{"id": 1, "answer": "A concise answer"}]})
    assert answers.answers[0].id == 1


def test_quality_answers_normalize_explicit_null_conservatively() -> None:
    answers = QualityAnswersSchema.model_validate({"answers": [{"id": 1, "answer": None}]})
    assert answers.answers[0].answer == "unknown"


def test_privacy_answers_reject_unknown_and_use_integer_ids() -> None:
    with pytest.raises(ValidationError):
        PrivacyAnswersSchema.model_validate(
            {"answers": [{"id": 1, "answer": "unknown", "confidence": 0.9, "reason": "unsupported enum value"}]}
        )


def test_privacy_answers_require_confidence_and_reason() -> None:
    with pytest.raises(ValidationError):
        PrivacyAnswersSchema.model_validate({"answers": [{"id": 1, "answer": "yes"}]})


def test_privacy_answers_normalize_explicit_nulls_conservatively() -> None:
    answers = PrivacyAnswersSchema.model_validate(
        {
            "answers": [
                {
                    "id": 1,
                    "answer": None,
                    "confidence": None,
                    "reason": None,
                    "evidence": None,
                }
            ]
        }
    )
    answer = answers.answers[0]
    assert answer.answer == "yes"
    assert answer.confidence == 1.0
    assert answer.reason == "Model returned null answer; defaulted to highest-confidence leak."
    assert answer.evidence == []


def test_privacy_answers_normalize_null_answer_atomically() -> None:
    answers = PrivacyAnswersSchema.model_validate(
        {
            "answers": [
                {
                    "id": 1,
                    "answer": None,
                    "confidence": 0.0,
                    "reason": "No leak detected.",
                    "evidence": ["unrelated evidence"],
                }
            ]
        }
    )
    answer = answers.answers[0]
    assert answer.answer == "yes"
    assert answer.confidence == 1.0
    assert answer.reason == "Model returned null answer; defaulted to highest-confidence leak."
    assert answer.evidence == ["unrelated evidence"]


def test_privacy_answers_normalize_null_reason_without_changing_verdict() -> None:
    answers = PrivacyAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "no", "confidence": 0.4, "reason": None}]}
    )
    answer = answers.answers[0]
    assert answer.answer == "no"
    assert answer.confidence == 0.4
    assert answer.reason == "Model returned no reason."


def test_privacy_answers_normalize_null_confidence_without_changing_verdict() -> None:
    answers = PrivacyAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "no", "confidence": None, "reason": "No supporting evidence."}]}
    )
    answer = answers.answers[0]
    assert answer.answer == "no"
    assert answer.confidence == 1.0
    assert answer.reason == "No supporting evidence."


def test_privacy_answers_normalize_null_evidence_without_changing_verdict() -> None:
    answers = PrivacyAnswersSchema.model_validate(
        {
            "answers": [
                {
                    "id": 1,
                    "answer": "no",
                    "confidence": 0.4,
                    "reason": "No supporting evidence.",
                    "evidence": None,
                }
            ]
        }
    )
    answer = answers.answers[0]
    assert answer.answer == "no"
    assert answer.confidence == 0.4
    assert answer.reason == "No supporting evidence."
    assert answer.evidence == []


def test_qa_compare_results_use_integer_ids() -> None:
    results = QACompareResultsSchema.model_validate({"per_item": [{"id": 1, "score": 0.8, "reason": "close match"}]})
    assert results.per_item[0].id == 1


def test_qa_compare_results_normalize_explicit_null_score_conservatively() -> None:
    results = QACompareResultsSchema.model_validate({"per_item": [{"id": 1, "score": None, "reason": None}]})
    assert results.per_item[0].score == 0.0
    assert results.per_item[0].reason is None


# Context-validated answer coverage


def test_quality_answers_reject_missing_ids_with_context() -> None:
    with pytest.raises(ValidationError, match="Missing answer IDs"):
        QualityAnswersSchema.model_validate(
            {"answers": [{"id": 1, "answer": "yes"}]},
            context={"expected_ids": [1, 2]},
        )


def test_quality_answers_accept_complete_with_context() -> None:
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}, {"id": 2, "answer": "no"}]},
        context={"expected_ids": [1, 2]},
    )
    assert len(result.answers) == 2


def test_quality_answers_no_enforcement_without_context() -> None:
    result = QualityAnswersSchema.model_validate({"answers": [{"id": 1, "answer": "yes"}]})
    assert len(result.answers) == 1


def test_privacy_answers_reject_missing_ids_with_context() -> None:
    with pytest.raises(ValidationError, match="Missing answer IDs"):
        PrivacyAnswersSchema.model_validate(
            {"answers": [{"id": 1, "answer": "no", "confidence": 0.0, "reason": "not inferable"}]},
            context={"expected_ids": [1, 2]},
        )


def test_qa_compare_reject_missing_ids_with_context() -> None:
    with pytest.raises(ValidationError, match="Missing compare IDs"):
        QACompareResultsSchema.model_validate(
            {"per_item": [{"id": 1, "score": 0.9}]},
            context={"expected_ids": [1, 2]},
        )


def test_quality_answers_reject_duplicate_ids() -> None:
    with pytest.raises(ValidationError, match="Duplicate answer IDs"):
        QualityAnswersSchema.model_validate(
            {"answers": [{"id": 1, "answer": "yes"}, {"id": 1, "answer": "no"}, {"id": 2, "answer": "yes"}]},
            context={"expected_ids": [1, 2]},
        )


def test_quality_answers_reject_extra_ids() -> None:
    with pytest.raises(ValidationError, match="Extra answer IDs"):
        QualityAnswersSchema.model_validate(
            {"answers": [{"id": 1, "answer": "yes"}, {"id": 2, "answer": "no"}, {"id": 99, "answer": "yes"}]},
            context={"expected_ids": [1, 2]},
        )


# ---------------------------------------------------------------------------
# StrictEntityDispositionSchema / StrictSensitivityDispositionSchema
# ---------------------------------------------------------------------------


def _make_strict_entity(**kwargs) -> dict:
    """Factory for a valid StrictEntityDispositionSchema dict."""
    defaults = {
        "id": 1,
        "source": "tagged",
        "category": "direct_identifier",
        "sensitivity": "high",
        "entity_label": "first_name",
        "entity_value": "Alice",
        "protection_reason": "Direct identifier that uniquely identifies the individual.",
        "protection_method_suggestion": "replace",
    }
    return {**defaults, **kwargs}


def test_strict_entity_rejects_leave_as_is() -> None:
    with pytest.raises(ValidationError):
        StrictEntityDispositionSchema.model_validate(
            _make_strict_entity(protection_method_suggestion="leave_as_is", sensitivity="medium")
        )


def test_strict_entity_rejects_low_sensitivity() -> None:
    with pytest.raises(ValidationError):
        StrictEntityDispositionSchema.model_validate(
            _make_strict_entity(sensitivity="low", protection_method_suggestion="replace")
        )


def test_strict_entity_accepts_valid_protected_entity() -> None:
    entity = StrictEntityDispositionSchema.model_validate(_make_strict_entity())
    assert entity.needs_protection is True
    assert entity.sensitivity == "high"


def test_strict_sensitivity_disposition_inherits_id_validation() -> None:
    payload = {
        "sensitivity_disposition": [
            _make_strict_entity(id=1),
            _make_strict_entity(id=5, entity_label="last_name", entity_value="Smith"),
        ],
    }
    with pytest.raises(ValidationError, match="sequential 1..2"):
        StrictSensitivityDispositionSchema.model_validate(payload)
    payload["sensitivity_disposition"][1]["id"] = 2
    schema = StrictSensitivityDispositionSchema.model_validate(payload)
    assert isinstance(schema, SensitivityDispositionSchema)


def test_sensitivity_disposition_output_contains_only_entities() -> None:
    payload = {"sensitivity_disposition": [_make_entity(id=1)]}
    disposition = SensitivityDispositionSchema.model_validate(payload)
    assert set(disposition.model_dump()) == {"sensitivity_disposition"}
    assert set(SensitivityDispositionSchema.model_json_schema()["properties"]) == {"sensitivity_disposition"}
