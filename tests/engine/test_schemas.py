# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS
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
    _ENTITY_LABEL_TO_CATEGORY,
    DomainClassificationSchema,
    EntityDispositionSchema,
    MeaningUnitsSchema,
    PrivacyAnswerItemSchema,
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


def test_entity_disposition_rejects_no_protection_but_method_set() -> None:
    """EntityDispositionSchema is strict — inconsistent (needs_protection,
    method) pairs are rejected. Class K reconciliation is the
    reconstructor's job (derive_needs_protection makes the pair
    tautologically consistent), so the schema must reject anything that
    bypasses the reconstructor."""
    with pytest.raises(ValidationError):
        EntityDispositionSchema.model_validate(
            _make_entity(needs_protection=False, protection_method_suggestion="replace")
        )


def test_entity_disposition_rejects_needs_protection_but_leave_as_is() -> None:
    """Inverse class K: also rejected by the strict schema. Class K
    reconciliation lives in reconstruct_full_disposition, not here."""
    with pytest.raises(ValidationError):
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


def test_quality_answers_pad_missing_ids_with_context() -> None:
    """Missing ids are padded with a placeholder so the record survives."""
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}]},
        context={"expected_ids": [1, 2]},
    )
    ids = [a.id for a in result.answers]
    assert ids == [1, 2]
    # Padded entry has the placeholder answer
    assert any(a.id == 2 and a.answer == "missing" for a in result.answers)


def test_quality_answers_accept_complete_with_context() -> None:
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}, {"id": 2, "answer": "no"}]},
        context={"expected_ids": [1, 2]},
    )
    assert len(result.answers) == 2


def test_quality_answers_no_enforcement_without_context() -> None:
    result = QualityAnswersSchema.model_validate({"answers": [{"id": 1, "answer": "yes"}]})
    assert len(result.answers) == 1


def test_privacy_answers_pad_missing_ids_with_context() -> None:
    """Missing ids are padded with a pessimistic 'yes' so the record survives
    and downstream triggers human review rather than silently passing."""
    result = PrivacyAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "no", "confidence": 0.0, "reason": "not inferable"}]},
        context={"expected_ids": [1, 2]},
    )
    ids = [a.id for a in result.answers]
    assert ids == [1, 2]
    padded = next(a for a in result.answers if a.id == 2)
    assert padded.answer.value == "yes"  # pessimistic default
    assert padded.confidence == 0.5


def test_qa_compare_pad_missing_ids_with_context() -> None:
    """Missing ids are padded with a neutral 0.5 score so the record survives."""
    result = QACompareResultsSchema.model_validate(
        {"per_item": [{"id": 1, "score": 0.9}]},
        context={"expected_ids": [1, 2]},
    )
    ids = [a.id for a in result.per_item]
    assert ids == [1, 2]
    padded = next(a for a in result.per_item if a.id == 2)
    assert padded.score == 0.5


def test_quality_answers_dedup_duplicate_ids() -> None:
    """Duplicate ids are deduplicated (first occurrence wins) instead of rejected."""
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}, {"id": 1, "answer": "no"}, {"id": 2, "answer": "yes"}]},
        context={"expected_ids": [1, 2]},
    )
    ids = [a.id for a in result.answers]
    assert ids == [1, 2]
    # First occurrence wins
    assert next(a for a in result.answers if a.id == 1).answer == "yes"


def test_quality_answers_drop_extra_ids() -> None:
    """Extra ids outside the expected set are dropped instead of rejecting the record."""
    result = QualityAnswersSchema.model_validate(
        {"answers": [{"id": 1, "answer": "yes"}, {"id": 2, "answer": "no"}, {"id": 99, "answer": "yes"}]},
        context={"expected_ids": [1, 2]},
    )
    ids = [a.id for a in result.answers]
    assert ids == [1, 2]  # 99 dropped


def test_meaning_units_renumber_when_ids_omitted() -> None:
    """MeaningUnitSchema.id defaults to 1 to keep the wire schema permissive
    for small-model output. When the LLM emits multiple units and omits ids
    on all of them, _ensure_list reassigns sequential 1..N so downstream
    prompts that reference units by id stay unambiguous."""
    raw = {"units": [
        {"aspect": "role", "unit": "individual is a software engineer"},
        {"aspect": "environment", "unit": "they work remotely"},
        {"aspect": "temporal_sequence", "unit": "for the past two years"},
    ]}
    result = MeaningUnitsSchema.model_validate(raw)
    assert [u.id for u in result.units] == [1, 2, 3]


def test_meaning_units_renumber_on_id_collision() -> None:
    """If the LLM emits duplicate ids, renumber to avoid downstream
    ambiguity. Otherwise downstream prompts referencing 'unit 1' would be
    ambiguous."""
    raw = {"units": [
        {"id": 1, "aspect": "role", "unit": "engineer"},
        {"id": 1, "aspect": "environment", "unit": "remote"},
    ]}
    result = MeaningUnitsSchema.model_validate(raw)
    assert [u.id for u in result.units] == [1, 2]


def test_meaning_units_preserve_explicit_unique_ids() -> None:
    """When the LLM emits valid, unique ids, keep them verbatim — even if
    not strictly sequential."""
    raw = {"units": [
        {"id": 5, "aspect": "role", "unit": "engineer"},
        {"id": 7, "aspect": "environment", "unit": "remote"},
    ]}
    result = MeaningUnitsSchema.model_validate(raw)
    assert [u.id for u in result.units] == [5, 7]


def test_entity_label_to_category_covers_default_labels() -> None:
    """Every label in DEFAULT_ENTITY_LABELS must have a category assignment.
    Without this guard, adding a new label to engine/constants.py without
    updating the category sets in schemas/rewrite.py would silently lose
    the entity-label-stuffed-in-category fallback for that label."""
    missing = set(DEFAULT_ENTITY_LABELS) - set(_ENTITY_LABEL_TO_CATEGORY)
    assert not missing, (
        f"DEFAULT_ENTITY_LABELS contains labels with no category assignment "
        f"in _ENTITY_LABEL_TO_CATEGORY: {sorted(missing)}. Add each to one "
        f"of _DIRECT_ID_LABELS / _QUASI_ID_LABELS / _SENSITIVE_ATTR_LABELS "
        f"in src/anonymizer/engine/schemas/rewrite.py."
    )


def test_privacy_answer_truncates_overlong_reason() -> None:
    """nemotron-3-nano on vLLM observed emitting 250+ char reasons; the
    schema must truncate rather than reject the record (max_length=200)."""
    long_reason = "x" * 250
    obj = PrivacyAnswerItemSchema.model_validate(
        {"id": 1, "answer": "yes", "confidence": 0.9, "reason": long_reason}
    )
    assert len(obj.reason) <= 200
    assert obj.reason.endswith("...")


def test_privacy_answer_reason_at_boundary_unchanged() -> None:
    """A reason of exactly 200 chars passes through untouched."""
    boundary = "y" * 200
    obj = PrivacyAnswerItemSchema.model_validate(
        {"id": 1, "answer": "yes", "confidence": 0.9, "reason": boundary}
    )
    assert obj.reason == boundary


def test_privacy_answer_none_reason_defaults_to_placeholder() -> None:
    """None reason coerces to a placeholder string rather than raising."""
    obj = PrivacyAnswerItemSchema.model_validate(
        {"id": 1, "answer": "yes", "confidence": 0.9, "reason": None}
    )
    assert obj.reason == "no reason provided"


def test_domain_confidence_coerces_float_string() -> None:
    """Small models occasionally return numeric confidences as strings —
    accept ``"0.95"`` and similar."""
    obj = DomainClassificationSchema.model_validate(
        {"domain": "MEDICAL", "domain_confidence": "0.95"}
    )
    assert obj.domain_confidence == 0.95


def test_domain_confidence_coerces_percent_string() -> None:
    """Accept percentage-style strings (``"85%"`` -> 0.85)."""
    obj = DomainClassificationSchema.model_validate(
        {"domain": "MEDICAL", "domain_confidence": "85%"}
    )
    assert obj.domain_confidence == pytest.approx(0.85)


def test_domain_confidence_coerces_unparseable_to_default() -> None:
    """Non-numeric strings (``"high"``) fall back to the 0.5 default."""
    obj = DomainClassificationSchema.model_validate(
        {"domain": "MEDICAL", "domain_confidence": "high"}
    )
    assert obj.domain_confidence == 0.5


def test_domain_confidence_clamps_out_of_range() -> None:
    """String confidence > 1.0 clamps to 1.0; negatives clamp to 0.0."""
    obj_hi = DomainClassificationSchema.model_validate(
        {"domain": "MEDICAL", "domain_confidence": "1.5"}
    )
    assert obj_hi.domain_confidence == 1.0
    obj_lo = DomainClassificationSchema.model_validate(
        {"domain": "MEDICAL", "domain_confidence": "-0.2"}
    )
    assert obj_lo.domain_confidence == 0.0
