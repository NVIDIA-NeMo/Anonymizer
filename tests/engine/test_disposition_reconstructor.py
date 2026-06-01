# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the server-side disposition reconstructor.

Covers:
  * ``_normalize_category`` — display variants, merged enums, entity-label
    confusion, fallback to quasi_identifier.
  * ``_normalize_method`` — case drift, substring match priority, omission.
  * ``derive_combined_risk_level`` — invariant-passing pick from method.
  * ``reconstruct_full_disposition`` — pairing, orphan skip, dedupe, empty
    fallback contract.
  * ``_pessimistic_fallback_disposition`` — direct=replace, others=generalize.
  * ``_reconstruct_full_disposition_column`` — workflow-column glue + empty
    + ValidationError fallbacks.

These tests pin the contracts the rewrite + QA + repair pipelines depend on
when small-model drift collapses the LLM disposition into a sparse
``SimpleDispositionResult``.
"""

from __future__ import annotations

import pandas as pd

from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_SENSITIVITY_DISPOSITION,
    COL_SIMPLE_DISPOSITION,
)
from anonymizer.engine.rewrite.disposition_derivation import (
    _normalize_category,
    _normalize_method,
    derive_combined_risk_level,
    reconstruct_full_disposition,
    template_protection_reason,
)
from anonymizer.engine.rewrite.sensitivity_disposition import (
    _pessimistic_fallback_disposition,
    _reconstruct_full_disposition_column,
)
from anonymizer.engine.schemas import SimpleDispositionResult

# ---------------------------------------------------------------------------
# _normalize_category
# ---------------------------------------------------------------------------


class TestNormalizeCategory:
    def test_canonical_passthrough(self) -> None:
        assert _normalize_category("direct_identifier") == "direct_identifier"
        assert _normalize_category("quasi_identifier") == "quasi_identifier"
        assert _normalize_category("latent_identifier") == "latent_identifier"

    def test_display_variants(self) -> None:
        assert _normalize_category("Direct-Identifier") == "direct_identifier"
        assert _normalize_category("DIRECT IDENTIFIER") == "direct_identifier"
        assert _normalize_category("DIRECT IDENTIFIERS") == "direct_identifier"

    def test_merged_enum_strongest_protection_wins(self) -> None:
        """Nemotron emits ``"latent_direct_identifier"`` — direct should win
        because re-id risk > inference risk."""
        assert _normalize_category("latent_direct_identifier") == "direct_identifier"

    def test_sensitive_substring_folds_into_quasi(self) -> None:
        """``sensitive_attribute`` was removed from ``EntityCategory`` (#150);
        the substring branch folds it to quasi_identifier as the conservative
        protect-cautiously bucket."""
        assert _normalize_category("latent_sensitive_attribute") == "quasi_identifier"
        assert _normalize_category("sensitive_attribute") == "quasi_identifier"

    def test_entity_label_in_category_slot_resolves(self) -> None:
        assert _normalize_category("first_name") == "direct_identifier"
        assert _normalize_category("date_of_birth") == "quasi_identifier"
        assert _normalize_category("gender") == "quasi_identifier"  # post-#150 fold

    def test_entity_label_echo_falls_back_to_quasi(self) -> None:
        """When the model echoes the same entity_label into both fields,
        quasi_identifier is the conservative fallback."""
        assert _normalize_category("zzz_unknown", entity_label="zzz_unknown") == "quasi_identifier"

    def test_blank_or_none_falls_back(self) -> None:
        assert _normalize_category(None) == "quasi_identifier"
        assert _normalize_category("") == "quasi_identifier"
        assert _normalize_category("   ") == "quasi_identifier"
        assert _normalize_category(123) == "quasi_identifier"

    def test_truly_unknown_falls_back_to_quasi(self) -> None:
        assert _normalize_category("xyz_zzz_random_999") == "quasi_identifier"


# ---------------------------------------------------------------------------
# _normalize_method
# ---------------------------------------------------------------------------


class TestNormalizeMethod:
    def test_canonical_passthrough(self) -> None:
        for choice in ("replace", "generalize", "remove", "suppress_inference", "leave_as_is"):
            assert _normalize_method(choice) == choice

    def test_uppercase_drift(self) -> None:
        assert _normalize_method("REPLACE") == "replace"
        assert _normalize_method("Generalize") == "generalize"

    def test_substring_match_priority(self) -> None:
        """``"replace_with_surrogate"`` -> replace; ``"leave_as_is_for_now"``
        -> leave_as_is. Substring match is on the canonical underscored form
        so it sees the wire-typical ``"suppress_inference_of_the_value"``;
        space-form variants fall through to the empty-string return (which
        the reconstructor handles via its pessimistic-default path)."""
        assert _normalize_method("replace_with_surrogate") == "replace"
        assert _normalize_method("leave_as_is_for_now") == "leave_as_is"
        assert _normalize_method("suppress_inference_of_the_value") == "suppress_inference"

    def test_unknown_returns_empty_string(self) -> None:
        """Empty signals to the caller to apply a pessimistic default."""
        assert _normalize_method("totally novel method") == ""
        assert _normalize_method("") == ""
        assert _normalize_method(None) == ""  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# derive_combined_risk_level
# ---------------------------------------------------------------------------


class TestDeriveCombinedRiskLevel:
    def test_leave_as_is_yields_low(self) -> None:
        """``low + leave_as_is`` is the only combination the
        ``_validate_protection_consistency`` invariant accepts for low; the
        reconstructor mirrors that."""
        for cat in ("direct_identifier", "quasi_identifier", "latent_identifier"):
            for sens in ("low", "medium", "high"):
                assert derive_combined_risk_level(cat, "leave_as_is", sens) == "low"

    def test_direct_identifier_with_protection_yields_high(self) -> None:
        assert derive_combined_risk_level("direct_identifier", "replace", "low") == "high"
        assert derive_combined_risk_level("direct_identifier", "remove", "medium") == "high"

    def test_high_sensitivity_yields_high(self) -> None:
        assert derive_combined_risk_level("quasi_identifier", "replace", "high") == "high"
        assert derive_combined_risk_level("latent_identifier", "remove", "high") == "high"

    def test_otherwise_yields_medium(self) -> None:
        assert derive_combined_risk_level("quasi_identifier", "generalize", "medium") == "medium"
        assert derive_combined_risk_level("latent_identifier", "suppress_inference", "low") == "medium"

    def test_picks_pass_validate_protection_consistency(self) -> None:
        """Spot-check: reconstructor's pick is always accepted by the strict
        schema's ``_validate_protection_consistency``. Any combination of
        (category, method, sensitivity) the reconstructor sees should
        produce a valid (combined_risk_level, method) pair."""
        from anonymizer.engine.schemas.rewrite import EntityDispositionSchema

        for cat in ("direct_identifier", "quasi_identifier", "latent_identifier"):
            for method in ("replace", "generalize", "remove", "suppress_inference", "leave_as_is"):
                for sens in ("low", "medium", "high"):
                    risk = derive_combined_risk_level(cat, method, sens)
                    EntityDispositionSchema(
                        id=1,
                        source="tagged",
                        category=cat,
                        sensitivity=sens,
                        entity_label="x",
                        entity_value="y",
                        protection_reason=template_protection_reason(cat, method, sens),
                        protection_method_suggestion=method,
                        combined_risk_level=risk,
                    )


# ---------------------------------------------------------------------------
# reconstruct_full_disposition
# ---------------------------------------------------------------------------


class TestReconstructFullDisposition:
    def test_id_indexed_pairing_uses_context_over_echo(self) -> None:
        """Belt-and-braces: when context has the entity at id-1, trust it
        over the echoed labels (gemma4-e2b emits garbage in the echoes)."""
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        simple = SimpleDispositionResult.model_validate(
            [
                {
                    "id": 1,
                    "source": "garbage",
                    "entity_label": "wrong_label",
                    "entity_value": "wrong_value",
                    "category": "direct_identifier",
                    "sensitivity": "high",
                    "protection_method_suggestion": "replace",
                }
            ]
        )
        result = reconstruct_full_disposition(simple, ebv, [])
        item = result.sensitivity_disposition[0]
        assert item.entity_label == "first_name"
        assert item.entity_value == "Alice"
        assert item.source == "tagged"

    def test_orphan_skipped_with_warning(self) -> None:
        """Items with id outside the context range AND missing/garbage
        echoes are skipped — better to return a smaller valid schema than
        to drop the row."""
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        simple = SimpleDispositionResult.model_validate(
            [
                {"id": 1, "category": "direct_identifier", "sensitivity": "high"},
                {"id": 99, "category": "direct_identifier"},  # orphan
            ]
        )
        result = reconstruct_full_disposition(simple, ebv, [])
        assert [item.id for item in result.sensitivity_disposition] == [1]

    def test_duplicate_ids_first_wins(self) -> None:
        ebv = [
            {"value": "Alice", "labels": ["first_name"]},
            {"value": "Bob", "labels": ["first_name"]},
        ]
        simple = SimpleDispositionResult.model_validate(
            [
                {"id": 1, "category": "direct_identifier", "sensitivity": "high"},
                {"id": 1, "category": "quasi_identifier", "sensitivity": "low"},  # duplicate
                {"id": 2, "category": "direct_identifier", "sensitivity": "high"},
            ]
        )
        result = reconstruct_full_disposition(simple, ebv, [])
        assert [item.id for item in result.sensitivity_disposition] == [1, 2]
        assert result.sensitivity_disposition[0].category == "direct_identifier"

    def test_short_reason_replaced_by_template(self) -> None:
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        simple = SimpleDispositionResult.model_validate(
            [
                {
                    "id": 1,
                    "category": "direct_identifier",
                    "sensitivity": "high",
                    "protection_method_suggestion": "replace",
                    "protection_reason": "ok",
                }
            ]
        )
        result = reconstruct_full_disposition(simple, ebv, [])
        # Templated reason is >=10 chars and reflects (high, replace, direct).
        assert len(result.sensitivity_disposition[0].protection_reason) >= 10
        assert "direct identifier" in result.sensitivity_disposition[0].protection_reason.lower()

    def test_long_llm_reason_kept_verbatim(self) -> None:
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        long_reason = "A document-specific judgement that the model should preserve"
        simple = SimpleDispositionResult.model_validate(
            [
                {
                    "id": 1,
                    "category": "direct_identifier",
                    "sensitivity": "high",
                    "protection_method_suggestion": "replace",
                    "protection_reason": long_reason,
                }
            ]
        )
        result = reconstruct_full_disposition(simple, ebv, [])
        assert result.sensitivity_disposition[0].protection_reason == long_reason

    def test_rambling_reason_is_capped(self) -> None:
        """The schema no longer enforces max_length on protection_reason, so
        the reconstructor must cap a runaway small-model reason itself to keep
        rewrite prompts and parquet bounded (silent truncate, never a drop)."""
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        rambling = "This entity is sensitive because " + ("blah " * 300)
        simple = SimpleDispositionResult.model_validate(
            [
                {
                    "id": 1,
                    "category": "direct_identifier",
                    "sensitivity": "high",
                    "protection_method_suggestion": "replace",
                    "protection_reason": rambling,
                }
            ]
        )
        result = reconstruct_full_disposition(simple, ebv, [])
        capped = result.sensitivity_disposition[0].protection_reason
        assert len(capped) <= 500
        assert capped.endswith("...")

    def test_omitted_method_pessimistic_default(self) -> None:
        """When the LLM omits ``protection_method_suggestion``, default to
        ``replace`` for direct/medium-or-high sensitivity, else
        ``leave_as_is``."""
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        simple = SimpleDispositionResult.model_validate(
            [{"id": 1, "category": "direct_identifier", "sensitivity": "high"}]
        )
        result = reconstruct_full_disposition(simple, ebv, [])
        assert result.sensitivity_disposition[0].protection_method_suggestion == "replace"

    def test_combined_risk_level_derived_from_method(self) -> None:
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        simple = SimpleDispositionResult.model_validate(
            [
                {
                    "id": 1,
                    "category": "direct_identifier",
                    "sensitivity": "high",
                    "protection_method_suggestion": "replace",
                }
            ]
        )
        result = reconstruct_full_disposition(simple, ebv, [])
        assert result.sensitivity_disposition[0].combined_risk_level == "high"


# ---------------------------------------------------------------------------
# _pessimistic_fallback_disposition
# ---------------------------------------------------------------------------


class TestPessimisticFallback:
    def test_direct_identifiers_get_replace(self) -> None:
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        result = _pessimistic_fallback_disposition(ebv, [])
        item = result.sensitivity_disposition[0]
        assert item.protection_method_suggestion == "replace"
        assert item.combined_risk_level == "high"

    def test_quasi_identifiers_get_generalize(self) -> None:
        ebv = [{"value": "40", "labels": ["age"]}]
        result = _pessimistic_fallback_disposition(ebv, [])
        item = result.sensitivity_disposition[0]
        assert item.protection_method_suggestion == "generalize"

    def test_latent_entities_marked_latent_identifier(self) -> None:
        result = _pessimistic_fallback_disposition([], [{"label": "occupation", "value": "doctor"}])
        item = result.sensitivity_disposition[0]
        assert item.source == "latent"
        assert item.category == "latent_identifier"
        assert item.protection_method_suggestion == "generalize"

    def test_unmapped_label_falls_back_to_quasi(self) -> None:
        ebv = [{"value": "x", "labels": ["wholly_novel_label"]}]
        result = _pessimistic_fallback_disposition(ebv, [])
        item = result.sensitivity_disposition[0]
        assert item.category == "quasi_identifier"

    def test_empty_context_returns_valid_noop_instead_of_raising(self) -> None:
        """Empty context (a pipeline-invariant violation) must not raise on the
        SensitivityDispositionSchema min_length=1 tripwire and drop the row;
        the fallback emits a single no-op (leave_as_is/low) disposition."""
        result = _pessimistic_fallback_disposition([], [])
        assert len(result.sensitivity_disposition) == 1
        item = result.sensitivity_disposition[0]
        assert item.protection_method_suggestion == "leave_as_is"
        assert item.combined_risk_level == "low"
        assert result.protected_entities == []  # no-op never reaches the rewrite

    def test_all_blank_slots_returns_valid_noop(self) -> None:
        """Slots whose label/value strip to empty are skipped; if that empties
        the disposition, the no-op guarantee still holds (no raise/no drop)."""
        result = _pessimistic_fallback_disposition([{"value": "", "labels": [""]}], [])
        assert len(result.sensitivity_disposition) == 1
        assert result.sensitivity_disposition[0].protection_method_suggestion == "leave_as_is"


# ---------------------------------------------------------------------------
# _reconstruct_full_disposition_column (workflow glue)
# ---------------------------------------------------------------------------


class TestReconstructionColumn:
    def _row(self, simple_payload, ebv=None, latent=None) -> dict:
        return {
            COL_SIMPLE_DISPOSITION: simple_payload,
            COL_ENTITIES_BY_VALUE: ebv or [],
            COL_LATENT_ENTITIES: latent or [],
        }

    def test_dict_payload(self) -> None:
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        row = self._row(
            {"sensitivity_disposition": [{"id": 1, "category": "direct_identifier", "sensitivity": "high"}]},
            ebv=ebv,
        )
        out = _reconstruct_full_disposition_column(row)
        assert out[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"][0]["entity_label"] == "first_name"

    def test_json_string_payload(self) -> None:
        import json

        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        row = self._row(
            json.dumps(
                {"sensitivity_disposition": [{"id": 1, "category": "direct_identifier", "sensitivity": "high"}]}
            ),
            ebv=ebv,
        )
        out = _reconstruct_full_disposition_column(row)
        assert out[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"][0]["entity_label"] == "first_name"

    def test_empty_simple_falls_back_to_pessimistic(self) -> None:
        """Lipika/Andre's review concern: empty reconstruction must NOT
        emit ``{"sensitivity_disposition": []}`` (would fail downstream
        ``parse_sensitivity_disposition`` on min_length=1). Instead build
        a pessimistic disposition from the entity context."""
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        row = self._row({"sensitivity_disposition": []}, ebv=ebv)
        out = _reconstruct_full_disposition_column(row)
        items = out[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"]
        assert len(items) == 1
        assert items[0]["entity_label"] == "first_name"
        assert items[0]["protection_method_suggestion"] == "replace"

    def test_empty_simple_and_empty_context_does_not_drop_row(self) -> None:
        """Both unguarded fallback call-sites: empty simple output AND empty
        entity context must still yield a valid row (the column generator must
        never raise out and drop the record)."""
        row = self._row({"sensitivity_disposition": []}, ebv=[], latent=[])
        out = _reconstruct_full_disposition_column(row)
        items = out[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"]
        assert len(items) == 1
        assert items[0]["protection_method_suggestion"] == "leave_as_is"

    def test_invalid_simple_payload_falls_back(self) -> None:
        """If ``SimpleDispositionResult.model_validate`` raises, fall back
        to pessimistic disposition rather than dropping the row."""
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        # A list with a non-dict item is not a valid SimpleDispositionItem;
        # validation would raise — fallback should kick in.
        row = self._row(["not a dict at all"], ebv=ebv)
        out = _reconstruct_full_disposition_column(row)
        assert len(out[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"]) == 1

    def test_garbage_string_payload_falls_back(self) -> None:
        ebv = [{"value": "Alice", "labels": ["first_name"]}]
        row = self._row("not json {}{ at all", ebv=ebv)
        out = _reconstruct_full_disposition_column(row)
        assert len(out[COL_SENSITIVITY_DISPOSITION]["sensitivity_disposition"]) == 1


# ---------------------------------------------------------------------------
# Sanity: the reconstructor closes the loop via parse_sensitivity_disposition
# ---------------------------------------------------------------------------


def test_round_trip_reconstructed_disposition_parses() -> None:
    """End-to-end: reconstructor output -> parse_sensitivity_disposition
    must succeed. This is the contract that broke in PR #130's original
    empty-list fallback (Lipika+Andre review)."""
    from anonymizer.engine.rewrite.parsers import parse_sensitivity_disposition

    ebv = [{"value": "Alice", "labels": ["first_name"]}]
    simple = SimpleDispositionResult.model_validate([{"id": 1, "category": "direct_identifier", "sensitivity": "high"}])
    result = reconstruct_full_disposition(simple, ebv, [])
    parsed = parse_sensitivity_disposition(result.model_dump())
    assert len(parsed.sensitivity_disposition) == 1


def test_round_trip_pessimistic_fallback_parses() -> None:
    from anonymizer.engine.rewrite.parsers import parse_sensitivity_disposition

    ebv = [{"value": "Alice", "labels": ["first_name"]}]
    result = _pessimistic_fallback_disposition(ebv, [])
    parsed = parse_sensitivity_disposition(result.model_dump())
    assert len(parsed.sensitivity_disposition) == 1


# Suppress unused-import warning for pandas in pyright/strict modes; pandas
# may be needed by future fixtures and pre-importing it gives a more
# reliable failure surface than a lazy import inside a test that fires only
# on schema regression.
_ = pd
