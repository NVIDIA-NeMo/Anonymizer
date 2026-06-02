# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests covering small-model output drift on detection schemas.

These regressions cover the drift modes observed during small-model
benchmarks (gemma4-e2b, gemma4-e4b, nemotron-3-nano:4b, qwen3.5:4b on legal
court / medical visit / employee notes datasets). Each test pins one drift
class so a future schema change that re-tightens the wire contract surfaces
here rather than silently dropping records on small-model runs.
"""

from __future__ import annotations

from anonymizer.engine.schemas.detection import (
    LatentEntitySchema,
    RawValidationDecisionSchema,
    ValidationDecisionSchema,
)

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
        """Pre-loosening these were required ``min_length=1`` and would drop
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

    def test_bare_string_evidence_wrapped_not_dropped(self) -> None:
        """Small models sometimes emit a single evidence quote as a bare string
        instead of a one-element list; it should be kept, not silently dropped."""
        result = LatentEntitySchema.model_validate({"label": "x", "value": "y", "evidence": "lives near the clinic"})
        assert result.evidence == ["lives near the clinic"]
