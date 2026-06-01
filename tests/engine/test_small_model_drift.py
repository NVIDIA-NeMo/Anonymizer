# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests covering small-model output drift on rewrite + detection schemas.

These regressions cover the drift modes observed during PR #130 small-model
benchmarks (gemma4-e2b, gemma4-e4b, nemotron-3-nano:4b, qwen3.5:4b on legal
court / medical visit / employee notes datasets). Each test pins one drift
class so a future schema change that re-tightens the wire contract surfaces
here rather than silently dropping records on small-model runs.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anonymizer.engine.schemas.detection import (
    LatentEntitySchema,
    RawValidationDecisionSchema,
    ValidationDecisionSchema,
)
from anonymizer.engine.schemas.rewrite import (
    Domain,
    DomainClassificationSchema,
    MeaningUnitImportance,
    MeaningUnitsSchema,
    PrivacyAnswerItemSchema,
    SimpleDispositionResult,
)

# ---------------------------------------------------------------------------
# DomainClassificationSchema — wire-loose domain + confidence coercion
# ---------------------------------------------------------------------------


class TestDomainClassificationDrift:
    def test_lowercase_domain_normalizes_to_canonical(self) -> None:
        result = DomainClassificationSchema.model_validate({"domain": "biography_profile", "domain_confidence": 0.9})
        assert result.domain == Domain.BIOGRAPHY_PROFILE.value

    def test_substring_match_to_canonical_value(self) -> None:
        result = DomainClassificationSchema.model_validate(
            {"domain": "this is medical clinical content", "domain_confidence": 0.7}
        )
        assert result.domain == Domain.MEDICAL_CLINICAL.value

    def test_unknown_domain_falls_back_to_other(self) -> None:
        result = DomainClassificationSchema.model_validate(
            {"domain": "completely-novel-bucket", "domain_confidence": 0.5}
        )
        assert result.domain == Domain.OTHER.value

    def test_missing_or_blank_domain_falls_back_to_other(self) -> None:
        for raw in [None, "", "   ", 123]:
            result = DomainClassificationSchema.model_validate({"domain": raw, "domain_confidence": 0.5})
            assert result.domain == Domain.OTHER.value

    def test_string_confidence_coerces_to_float(self) -> None:
        result = DomainClassificationSchema.model_validate({"domain": "LEGAL", "domain_confidence": "0.85"})
        assert result.domain_confidence == pytest.approx(0.85)

    def test_percent_confidence_coerces_to_fractional_float(self) -> None:
        result = DomainClassificationSchema.model_validate({"domain": "LEGAL", "domain_confidence": "85%"})
        assert result.domain_confidence == pytest.approx(0.85)

    def test_unparseable_confidence_falls_back_to_default(self) -> None:
        result = DomainClassificationSchema.model_validate({"domain": "LEGAL", "domain_confidence": "very high"})
        assert result.domain_confidence == 0.5

    def test_out_of_range_confidence_clamps(self) -> None:
        high = DomainClassificationSchema.model_validate({"domain": "LEGAL", "domain_confidence": 1.5})
        assert high.domain_confidence == 1.0
        low = DomainClassificationSchema.model_validate({"domain": "LEGAL", "domain_confidence": -0.2})
        assert low.domain_confidence == 0.0


# ---------------------------------------------------------------------------
# SimpleDispositionResult — bare-list tolerance + scalar coercion
# ---------------------------------------------------------------------------


class TestSimpleDispositionDrift:
    def test_canonical_wrapper_validates(self) -> None:
        result = SimpleDispositionResult.model_validate(
            {"sensitivity_disposition": [{"id": 1, "category": "direct_identifier"}]}
        )
        assert len(result.sensitivity_disposition) == 1

    def test_bare_list_at_top_level_wraps_into_canonical(self) -> None:
        """nemotron-3-nano:4b consistently emits the bare list shape on
        rewrite-mode disposition; without the ``oneOf`` widening +
        ``accept_bare_list`` validator this would fail DD's pre-validate."""
        result = SimpleDispositionResult.model_validate([{"id": 1, "category": "direct_identifier"}])
        assert len(result.sensitivity_disposition) == 1
        assert result.sensitivity_disposition[0].id == 1

    def test_json_schema_widens_to_oneof_for_dd_pre_validate(self) -> None:
        """DataDesigner runs ``jsonschema.validate`` BEFORE pydantic's
        before-validators, so the emitted schema must accept both shapes."""
        schema = SimpleDispositionResult.model_json_schema()
        assert "oneOf" in schema, schema
        oneof_types = {variant.get("type") for variant in schema["oneOf"]}
        assert oneof_types == {"object", "array"}

    def test_int_values_in_str_fields_coerce(self) -> None:
        """gemma4-e4b observed echoing ints in entity_value when the value
        is numeric (age, ssn). Loose wire coerces to str so the row
        survives; reconstructor uses the trusted context anyway."""
        result = SimpleDispositionResult.model_validate(
            [{"id": 1, "entity_value": 42, "entity_label": "age", "category": "quasi_identifier"}]
        )
        assert result.sensitivity_disposition[0].entity_value == "42"

    def test_none_in_str_fields_coerces_to_empty(self) -> None:
        result = SimpleDispositionResult.model_validate(
            [{"id": 1, "entity_value": None, "entity_label": None, "category": None}]
        )
        item = result.sensitivity_disposition[0]
        assert item.entity_value == ""
        assert item.entity_label == ""
        assert item.category == ""


# ---------------------------------------------------------------------------
# MeaningUnits — bare list, aspect normalize, importance default, id renumber
# ---------------------------------------------------------------------------


class TestMeaningUnitsDrift:
    def test_bare_list_top_level_wraps(self) -> None:
        """qwen3.5:4b on legal-court emits the bare-list shape."""
        result = MeaningUnitsSchema.model_validate([{"id": 1, "aspect": "role", "unit": "judge"}])
        assert len(result.units) == 1

    def test_aspect_case_normalizes(self) -> None:
        result = MeaningUnitsSchema.model_validate({"units": [{"id": 1, "aspect": "ROLE", "unit": "lawyer"}]})
        assert result.units[0].aspect == "role"

    def test_aspect_substring_match(self) -> None:
        result = MeaningUnitsSchema.model_validate(
            {"units": [{"id": 1, "aspect": "the procedural status of the case", "unit": "pending"}]}
        )
        assert result.units[0].aspect == "procedural_status"

    def test_unknown_aspect_falls_through_to_empty(self) -> None:
        result = MeaningUnitsSchema.model_validate({"units": [{"id": 1, "aspect": "xyz", "unit": "u"}]})
        assert result.units[0].aspect == ""

    def test_missing_importance_defaults_to_important(self) -> None:
        """Pre-#130 ``MeaningUnitSchema.importance`` was a required enum;
        small models that omit it would drop the row. Default to
        ``important`` (the safer of the two values for downstream QA)."""
        result = MeaningUnitsSchema.model_validate({"units": [{"id": 1, "aspect": "role", "unit": "lawyer"}]})
        assert result.units[0].importance == MeaningUnitImportance.important.value

    def test_drift_on_importance_normalizes(self) -> None:
        result = MeaningUnitsSchema.model_validate(
            {"units": [{"id": 1, "aspect": "role", "unit": "lawyer", "importance": "Critical"}]}
        )
        assert result.units[0].importance == MeaningUnitImportance.critical.value

    def test_duplicate_ids_renumber_sequentially(self) -> None:
        """Every-unit-has-id=1 collapse mode (model omitted ids and the
        ``id=1`` default kicked in for all of them)."""
        result = MeaningUnitsSchema.model_validate(
            {
                "units": [
                    {"aspect": "role", "unit": "a"},
                    {"aspect": "role", "unit": "b"},
                    {"aspect": "role", "unit": "c"},
                ]
            }
        )
        assert [u.id for u in result.units] == [1, 2, 3]

    def test_explicit_unique_ids_preserved(self) -> None:
        result = MeaningUnitsSchema.model_validate(
            {"units": [{"id": 5, "aspect": "role", "unit": "a"}, {"id": 7, "aspect": "role", "unit": "b"}]}
        )
        assert [u.id for u in result.units] == [5, 7]

    def test_empty_list_does_not_drop_record(self) -> None:
        """Outer min_length=1 was relaxed to default_factory=list — empty
        units should validate so downstream can decide what to do."""
        result = MeaningUnitsSchema.model_validate({"units": []})
        assert result.units == []


# ---------------------------------------------------------------------------
# RawValidationDecisionSchema — chunked-validation drift
# ---------------------------------------------------------------------------


class TestRawValidationDecisionDrift:
    def test_freeform_prose_decision_coerces_to_keep(self) -> None:
        """gemma4-e4b on legal_court emits prose like "No specific entity
        type for the date placeholder." in the decision slot. Conservative
        coercion to ``keep`` so the detection survives downstream."""
        result = RawValidationDecisionSchema.model_validate(
            {"id": "x", "decision": "No specific entity type for the date placeholder."}
        )
        assert result.decision is not None
        assert result.decision.value == "keep"

    def test_explicit_drop_substring_wins(self) -> None:
        result = RawValidationDecisionSchema.model_validate({"id": "x", "decision": "DROP."})
        assert result.decision.value == "drop"

    def test_reclass_substring_wins_over_keep(self) -> None:
        """``"reclass entity (was previously kept)"`` — the more-specific
        choice should win even when both substrings are present."""
        result = RawValidationDecisionSchema.model_validate(
            {"id": "x", "decision": "reclass entity (was previously kept)"}
        )
        assert result.decision.value == "reclass"

    def test_none_decision_preserved(self) -> None:
        """``decision=None`` is "no answer" in the chunked merger; must NOT
        be coerced into a verdict or downstream merge logic breaks."""
        result = RawValidationDecisionSchema.model_validate({"id": "x", "decision": None})
        assert result.decision is None

    def test_blank_string_decision_treated_as_none(self) -> None:
        result = RawValidationDecisionSchema.model_validate({"id": "x", "decision": "   "})
        assert result.decision is None

    def test_proposed_label_none_coerces_to_empty(self) -> None:
        result = RawValidationDecisionSchema.model_validate({"id": "x", "decision": "keep", "proposed_label": None})
        assert result.proposed_label == ""

    def test_int_proposed_label_coerces_to_str(self) -> None:
        result = RawValidationDecisionSchema.model_validate({"id": "x", "decision": "keep", "proposed_label": 42})
        assert result.proposed_label == "42"


# ---------------------------------------------------------------------------
# ValidationDecisionSchema — wire schema drops value/label
# ---------------------------------------------------------------------------


class TestValidationDecisionWireShape:
    def test_value_label_stripped_from_wire(self) -> None:
        """The wire-loose schema dropped value and label as drift surface;
        downstream ``enrich_validation_decisions`` re-fills them from the
        trusted ``candidate_lookup``."""
        result = ValidationDecisionSchema.model_validate(
            {"id": "x", "decision": "keep", "value": "Alice", "label": "first_name"}
        )
        dumped = result.model_dump()
        assert "value" not in dumped
        assert "label" not in dumped

    def test_decision_freeform_coerces_to_keep(self) -> None:
        # ValidationDecisionSchema.decision is ``str`` (not the ValidationChoice
        # enum) — the wire-loose contract — so the result is a plain string.
        result = ValidationDecisionSchema.model_validate({"id": "x", "decision": "free-form prose"})
        assert result.decision == "keep"

    def test_proposed_label_none_coerces_to_empty(self) -> None:
        result = ValidationDecisionSchema.model_validate({"id": "x", "decision": "keep", "proposed_label": None})
        assert result.proposed_label == ""


# ---------------------------------------------------------------------------
# LatentEntitySchema — defaults + rationale clamp
# ---------------------------------------------------------------------------


class TestLatentEntityDrift:
    def test_overlong_rationale_truncates(self) -> None:
        """Some models emit 200+ char rationales; clamp to 147 + ``"..."``
        to fit the 150-char schema cap rather than dropping the row."""
        long = "A" * 250
        result = LatentEntitySchema.model_validate({"label": "occupation", "value": "doctor", "rationale": long})
        assert len(result.rationale) <= 150
        assert result.rationale.endswith("...")

    def test_empty_required_fields_default_to_empty_string(self) -> None:
        """Pre-#130 these were required ``min_length=1`` and would drop
        the row; loose wire allows empty so the parquet-pad sentinel
        path can build a placeholder row when needed."""
        result = LatentEntitySchema()
        assert result.label == ""
        assert result.value == ""

    def test_invalid_confidence_coerces_to_medium(self) -> None:
        result = LatentEntitySchema.model_validate(
            {"label": "x", "value": "y", "confidence": "very-high", "rationale": "ok"}
        )
        assert result.confidence == "medium"

    def test_sensitive_category_drift_normalizes_to_latent_identifier(self) -> None:
        """A category string containing "sensitive" must normalize to the lone
        LatentCategory member rather than raising AttributeError (sensitive
        attributes were folded into quasi_identifier on the rewrite side)."""
        result = LatentEntitySchema.model_validate(
            {"label": "x", "value": "y", "category": "latent_sensitive_attribute"}
        )
        assert result.category == "latent_identifier"

    def test_unknown_category_drift_normalizes_to_latent_identifier(self) -> None:
        result = LatentEntitySchema.model_validate({"label": "x", "value": "y", "category": "some-novel-bucket"})
        assert result.category == "latent_identifier"


# ---------------------------------------------------------------------------
# PrivacyAnswerItemSchema — reason field coercion
# ---------------------------------------------------------------------------


class TestPrivacyAnswerReasonCoercion:
    def _base(self, **overrides: object) -> dict:
        return {
            "id": 1,
            "answer": "no",
            "confidence": 0.9,
            "reason": "no leak observed in rewrite",
            **overrides,
        }

    def test_overlong_reason_truncates(self) -> None:
        long = "x" * 400
        result = PrivacyAnswerItemSchema.model_validate(self._base(reason=long))
        assert len(result.reason) <= 200
        assert result.reason.endswith("...")

    def test_none_reason_replaced_with_placeholder(self) -> None:
        result = PrivacyAnswerItemSchema.model_validate(self._base(reason=None))
        assert "no reason provided" in result.reason

    def test_blank_reason_replaced_with_placeholder(self) -> None:
        result = PrivacyAnswerItemSchema.model_validate(self._base(reason="   "))
        assert "no reason provided" in result.reason

    def test_short_valid_reason_passthrough(self) -> None:
        result = PrivacyAnswerItemSchema.model_validate(self._base(reason="ok"))
        assert result.reason == "ok"


# ---------------------------------------------------------------------------
# Catch the regression: the strict wire contract used to drop these rows
# ---------------------------------------------------------------------------


def test_simple_disposition_does_not_validate_under_strict_disposition() -> None:
    """Sanity check: wire-loose ``SimpleDispositionResult`` accepts inputs
    that the strict ``SensitivityDispositionSchema`` would reject. This
    is the whole point of PR #130's two-step pipeline; if a future schema
    change makes them equivalent we want this test to surface that."""
    from anonymizer.engine.schemas.rewrite import SensitivityDispositionSchema

    drifted = [
        {
            "id": 1,
            "category": "DIRECT IDENTIFIER",  # display-variant drift
            "sensitivity": "HIGH",
            "protection_method_suggestion": "Replace_With_Surrogate",
            # missing entity_label, entity_value, protection_reason, combined_risk_level
        }
    ]
    SimpleDispositionResult.model_validate(drifted)  # OK
    with pytest.raises(ValidationError):
        SensitivityDispositionSchema.model_validate({"sensitivity_disposition": drifted})
