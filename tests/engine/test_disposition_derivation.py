# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anonymizer.engine.rewrite.disposition_derivation import (
    _flatten_context,
    _normalize_category,
    derive_needs_protection,
    reconstruct_full_disposition,
    template_protection_reason,
)
from anonymizer.engine.schemas.rewrite import (
    EntityCategory,
    ProtectionMethod,
    SensitivityLevel,
    SimpleDispositionItem,
    SimpleDispositionResult,
)


# ---------------------------------------------------------------------------
# derive_needs_protection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method,expected",
    [
        ("leave_as_is", False),
        ("replace", True),
        ("generalize", True),
        ("remove", True),
        ("suppress_inference", True),
        ("", True),   # empty method -> treat as non-trivial (conservative)
    ],
)
def test_derive_needs_protection(method: str, expected: bool) -> None:
    assert derive_needs_protection(method) is expected


# ---------------------------------------------------------------------------
# template_protection_reason — all 4 categories x 5 methods x 3 sensitivities
# ---------------------------------------------------------------------------


def test_template_covers_all_combinations_with_valid_length() -> None:
    """Every (category, method, sensitivity) triple produces a string that
    satisfies EntityDispositionSchema.protection_reason min_length=10."""
    categories = [c.value for c in EntityCategory]
    methods = [m.value for m in ProtectionMethod]
    sensitivities = [s.value for s in SensitivityLevel]
    for cat in categories:
        for m in methods:
            for s in sensitivities:
                reason = template_protection_reason(cat, m, s)
                assert isinstance(reason, str)
                assert len(reason) >= 10, f"({cat}, {m}, {s}) -> {reason!r}"


def test_template_unknown_category_method_pair_fallback() -> None:
    reason = template_protection_reason("weird_category", "replace", "medium")
    assert len(reason) >= 10


def test_template_leave_as_is_produces_non_empty_reason() -> None:
    reason = template_protection_reason("quasi_identifier", "leave_as_is", "low")
    assert len(reason) >= 10
    assert "retained" in reason.lower() or "low-risk" in reason.lower()


# ---------------------------------------------------------------------------
# reconstruct_full_disposition — happy paths
# ---------------------------------------------------------------------------


def _make_simple(*items: dict) -> SimpleDispositionResult:
    return SimpleDispositionResult(sensitivity_disposition=[SimpleDispositionItem(**i) for i in items])


def test_reconstruct_tagged_only_happy_path() -> None:
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace",
         "protection_reason": "Name identifies the individual directly and clearly."},
        {"id": 2, "category": "quasi_identifier", "sensitivity": "low",
         "protection_method_suggestion": "leave_as_is",
         "protection_reason": "City alone does not meaningfully narrow the population."},
    )
    context_tagged = [
        {"value": "Alice", "labels": ["first_name"]},
        {"value": "Portland", "labels": ["city"]},
    ]
    full = reconstruct_full_disposition(simple, context_tagged, [])

    assert len(full.sensitivity_disposition) == 2
    e1 = full.sensitivity_disposition[0]
    assert e1.id == 1
    assert e1.source == "tagged"
    assert e1.entity_label == "first_name"
    assert e1.entity_value == "Alice"
    assert e1.category == "direct_identifier"
    assert e1.sensitivity == "high"
    assert e1.needs_protection is True
    assert e1.protection_method_suggestion == "replace"
    assert "Name identifies" in e1.protection_reason  # LLM reason kept
    e2 = full.sensitivity_disposition[1]
    assert e2.needs_protection is False
    assert e2.protection_method_suggestion == "leave_as_is"


def test_reconstruct_latent_only_happy_path() -> None:
    simple = _make_simple(
        {"id": 1, "category": "latent_identifier", "sensitivity": "medium",
         "protection_method_suggestion": "suppress_inference"},
    )
    latent = [{"label": "employer", "value": "UW"}]
    full = reconstruct_full_disposition(simple, [], latent)

    assert len(full.sensitivity_disposition) == 1
    e = full.sensitivity_disposition[0]
    assert e.source == "latent"
    assert e.entity_label == "employer"
    assert e.entity_value == "UW"
    assert e.needs_protection is True
    # protection_reason was omitted -> templated
    assert len(e.protection_reason) >= 10


def test_reconstruct_mixed_tagged_and_latent() -> None:
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace", "protection_reason": "x"},  # too short -> templated
        {"id": 2, "category": "quasi_identifier", "sensitivity": "medium",
         "protection_method_suggestion": "generalize"},
        {"id": 3, "category": "latent_identifier", "sensitivity": "medium",
         "protection_method_suggestion": "suppress_inference"},
    )
    context_tagged = [
        {"value": "Alice", "labels": ["first_name"]},
        {"value": "47", "labels": ["age"]},
    ]
    latent = [{"label": "employer", "value": "UW"}]
    full = reconstruct_full_disposition(simple, context_tagged, latent)

    assert len(full.sensitivity_disposition) == 3
    assert full.sensitivity_disposition[0].entity_label == "first_name"
    assert full.sensitivity_disposition[1].entity_label == "age"
    assert full.sensitivity_disposition[2].source == "latent"
    # Item 1 had a too-short LLM reason -> templated
    assert full.sensitivity_disposition[0].protection_reason != "x"
    assert len(full.sensitivity_disposition[0].protection_reason) >= 10


# ---------------------------------------------------------------------------
# reconstruct_full_disposition — edge cases
# ---------------------------------------------------------------------------


def test_leave_as_is_implies_needs_protection_false_but_valid_reason() -> None:
    simple = _make_simple(
        {"id": 1, "category": "quasi_identifier", "sensitivity": "low",
         "protection_method_suggestion": "leave_as_is"},
    )
    full = reconstruct_full_disposition(simple, [{"value": "x", "labels": ["age"]}], [])
    e = full.sensitivity_disposition[0]
    assert e.needs_protection is False
    assert e.protection_method_suggestion == "leave_as_is"
    assert len(e.protection_reason) >= 10


def test_llm_reason_below_minlength_is_templated() -> None:
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace", "protection_reason": "short"},
    )
    full = reconstruct_full_disposition(simple, [{"value": "Alice", "labels": ["first_name"]}], [])
    assert full.sensitivity_disposition[0].protection_reason != "short"


def test_llm_reason_at_or_above_minlength_is_kept_verbatim() -> None:
    long_reason = "Very specific to this document and 10+ chars."
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace", "protection_reason": long_reason},
    )
    full = reconstruct_full_disposition(simple, [{"value": "Alice", "labels": ["first_name"]}], [])
    assert full.sensitivity_disposition[0].protection_reason == long_reason


def test_llm_reason_none_or_int_coerced_to_empty_then_templated() -> None:
    # The @field_validator on SimpleDispositionItem coerces None/int to "".
    item = SimpleDispositionItem(
        id=1, category="direct_identifier", sensitivity="high",
        protection_method_suggestion="replace", protection_reason=None,  # type: ignore[arg-type]
    )
    assert item.protection_reason == ""
    simple = SimpleDispositionResult(sensitivity_disposition=[item])
    full = reconstruct_full_disposition(simple, [{"value": "A", "labels": ["first_name"]}], [])
    assert len(full.sensitivity_disposition[0].protection_reason) >= 10


def test_orphan_item_skipped_with_warning(caplog: pytest.LogCaptureFixture) -> None:
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace",
         "protection_reason": "Ten+ chars here."},
        {"id": 99, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace",
         "protection_reason": "Ten+ chars also."},  # id out of context range, no echo
    )
    context_tagged = [{"value": "Alice", "labels": ["first_name"]}]
    full = reconstruct_full_disposition(simple, context_tagged, [])

    # Orphan skipped, valid item retained.
    assert len(full.sensitivity_disposition) == 1
    assert full.sensitivity_disposition[0].id == 1


def test_orphan_item_with_echoes_still_reconstructable() -> None:
    # Model echoed enough to rebuild even if id is out of context range.
    simple = _make_simple(
        {"id": 99, "source": "tagged", "entity_label": "last_name", "entity_value": "Smith",
         "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace", "protection_reason": "Reason here."},
    )
    full = reconstruct_full_disposition(simple, [], [])
    assert len(full.sensitivity_disposition) == 1
    assert full.sensitivity_disposition[0].entity_label == "last_name"
    # After SensitivityDispositionSchema._normalize_ids, id re-sequenced to 1.
    assert full.sensitivity_disposition[0].id == 1


def test_duplicate_ids_deduplicated_keeping_first() -> None:
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace", "protection_reason": "First copy ok."},
        {"id": 1, "category": "quasi_identifier", "sensitivity": "low",
         "protection_method_suggestion": "leave_as_is", "protection_reason": "Second dup."},
    )
    full = reconstruct_full_disposition(simple, [{"value": "x", "labels": ["first_name"]}], [])
    assert len(full.sensitivity_disposition) == 1
    # First wins
    assert "First copy" in full.sensitivity_disposition[0].protection_reason


def test_empty_simple_result_produces_empty_schema() -> None:
    # SensitivityDispositionSchema requires min_length=1, so zero-item
    # reconstruction should raise at the strict-schema layer — caller is
    # expected to short-circuit before disposition when no entities exist.
    simple = SimpleDispositionResult(sensitivity_disposition=[])
    with pytest.raises(ValidationError):
        reconstruct_full_disposition(simple, [], [])


def test_empty_method_with_high_risk_defaults_to_replace() -> None:
    """Class K-adjacent: small models routinely omit
    protection_method_suggestion entirely. For high-sensitivity direct
    identifiers and sensitive attributes, the reconstructor must default
    to ``replace`` so the entity does not silently slip through as
    leave_as_is. Mirrors the pessimistic-default pattern used in
    PrivacyAnswersSchema (pad missing with ``yes``)."""
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "", "protection_reason": ""},
        {"id": 2, "category": "sensitive_attribute", "sensitivity": "low",
         "protection_method_suggestion": "", "protection_reason": ""},
        {"id": 3, "category": "quasi_identifier", "sensitivity": "low",
         "protection_method_suggestion": "", "protection_reason": ""},
    )
    context_tagged = [
        {"value": "Alice", "labels": ["first_name"]},
        {"value": "diabetic", "labels": ["medical_condition"]},
        {"value": "blue", "labels": ["favorite_color"]},
    ]
    full = reconstruct_full_disposition(simple, context_tagged, [])

    # direct_identifier + high -> replace, needs_protection True
    assert full.sensitivity_disposition[0].protection_method_suggestion == "replace"
    assert full.sensitivity_disposition[0].needs_protection is True
    # sensitive_attribute + low -> still replace by category rule
    assert full.sensitivity_disposition[1].protection_method_suggestion == "replace"
    assert full.sensitivity_disposition[1].needs_protection is True
    # quasi_identifier + low -> safe to leave_as_is
    assert full.sensitivity_disposition[2].protection_method_suggestion == "leave_as_is"
    assert full.sensitivity_disposition[2].needs_protection is False


def test_empty_method_with_medium_quasi_id_defaults_to_replace() -> None:
    """Sensitivity-driven branch of the pessimistic default — even a
    quasi_identifier becomes ``replace`` when sensitivity is medium/high."""
    simple = _make_simple(
        {"id": 1, "category": "quasi_identifier", "sensitivity": "medium",
         "protection_method_suggestion": "", "protection_reason": ""},
    )
    context_tagged = [{"value": "37", "labels": ["age"]}]
    full = reconstruct_full_disposition(simple, context_tagged, [])
    assert full.sensitivity_disposition[0].protection_method_suggestion == "replace"
    assert full.sensitivity_disposition[0].needs_protection is True


def test_multi_label_entity_flattens_to_multiple_slots() -> None:
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace", "protection_reason": "Flatten me OK."},
        {"id": 2, "category": "quasi_identifier", "sensitivity": "low",
         "protection_method_suggestion": "leave_as_is", "protection_reason": "Other label."},
    )
    # One value, two labels -> reconstructor allocates two id slots.
    context_tagged = [{"value": "John", "labels": ["first_name", "organization_name"]}]
    full = reconstruct_full_disposition(simple, context_tagged, [])
    assert full.sensitivity_disposition[0].entity_label == "first_name"
    assert full.sensitivity_disposition[1].entity_label == "organization_name"
    assert full.sensitivity_disposition[0].entity_value == "John"
    assert full.sensitivity_disposition[1].entity_value == "John"


# ---------------------------------------------------------------------------
# _normalize_category — entity-label-stuffed-in-category drift
# ---------------------------------------------------------------------------


def test_normalize_category_entity_label_to_direct_identifier() -> None:
    """Class L-style drift: small models (gemma4-e2b) write entity labels
    like ``last_name``, ``ssn`` into the category slot. The reconstructor
    must map them back to the most-likely category — direct_identifier
    here — using _ENTITY_LABEL_TO_CATEGORY."""
    assert _normalize_category("last_name") == "direct_identifier"
    assert _normalize_category("first_name") == "direct_identifier"
    assert _normalize_category("ssn") == "direct_identifier"


def test_normalize_category_entity_label_to_quasi_identifier() -> None:
    assert _normalize_category("city") == "quasi_identifier"
    assert _normalize_category("date_of_birth") == "quasi_identifier"
    assert _normalize_category("occupation") == "quasi_identifier"


def test_normalize_category_entity_label_to_sensitive_attribute() -> None:
    assert _normalize_category("religious_belief") == "sensitive_attribute"
    assert _normalize_category("blood_type") == "sensitive_attribute"


def test_normalize_category_display_variants() -> None:
    """Display-label drift: case, spacing, hyphenation, plurals."""
    assert _normalize_category("Direct-Identifier") == "direct_identifier"
    assert _normalize_category("DIRECT IDENTIFIER") == "direct_identifier"
    assert _normalize_category("direct_identifiers") == "direct_identifier"
    assert _normalize_category("quasi_identifiers") == "quasi_identifier"


def test_normalize_category_merged_enum_hallucination() -> None:
    """Nemotron-observed drift: model splices two enum values. Strongest
    protection wins so harm dimension ranks above inference dimension."""
    assert _normalize_category("latent_sensitive_attribute") == "sensitive_attribute"
    assert _normalize_category("direct_quasi_identifier") == "direct_identifier"


def test_normalize_category_unknown_falls_back_to_quasi() -> None:
    """Unknown / empty drops back to quasi_identifier (conservative default
    over rejecting the row)."""
    assert _normalize_category("") == "quasi_identifier"
    assert _normalize_category(None) == "quasi_identifier"
    assert _normalize_category("totally_made_up_thing") == "quasi_identifier"


def test_normalize_category_via_reconstructor_with_entity_label_drift() -> None:
    """End-to-end: SimpleDispositionItem with category=``last_name`` (the
    canonical gemma failure mode) flows through reconstruct_full_disposition
    and lands as direct_identifier on EntityDispositionSchema."""
    simple = _make_simple(
        {"id": 1, "category": "last_name", "sensitivity": "high",
         "protection_method_suggestion": "replace",
         "protection_reason": "Identifies the individual directly."},
    )
    context_tagged = [{"value": "Smith", "labels": ["last_name"]}]
    full = reconstruct_full_disposition(simple, context_tagged, [])
    assert full.sensitivity_disposition[0].category == "direct_identifier"
    assert full.sensitivity_disposition[0].entity_label == "last_name"


# ---------------------------------------------------------------------------
# Class-K-style reconciliation through the reconstructor
# (Schema-level reconciliation was deleted — derive_needs_protection makes
# the (needs_protection, method) pair tautologically consistent.)
# ---------------------------------------------------------------------------


def test_class_k_method_replace_implies_needs_protection_true() -> None:
    """SimpleDispositionItem has no needs_protection field — the
    reconstructor derives it from method via derive_needs_protection.
    Any non-trivial method => needs_protection=True."""
    simple = _make_simple(
        {"id": 1, "category": "direct_identifier", "sensitivity": "high",
         "protection_method_suggestion": "replace",
         "protection_reason": "Direct id; replaced."},
    )
    full = reconstruct_full_disposition(
        simple, [{"value": "Alice", "labels": ["first_name"]}], []
    )
    assert full.sensitivity_disposition[0].needs_protection is True


def test_class_k_method_generalize_implies_needs_protection_true() -> None:
    """Coverage for the generalize method — also implies needs_protection."""
    simple = _make_simple(
        {"id": 1, "category": "quasi_identifier", "sensitivity": "medium",
         "protection_method_suggestion": "generalize",
         "protection_reason": "Generalize to age band."},
    )
    full = reconstruct_full_disposition(
        simple, [{"value": "37", "labels": ["age"]}], []
    )
    assert full.sensitivity_disposition[0].needs_protection is True
    assert full.sensitivity_disposition[0].protection_method_suggestion == "generalize"


def test_class_k_method_leave_as_is_implies_needs_protection_false() -> None:
    """Symmetric: leave_as_is is the only method that derives False."""
    simple = _make_simple(
        {"id": 1, "category": "quasi_identifier", "sensitivity": "low",
         "protection_method_suggestion": "leave_as_is",
         "protection_reason": "Low risk in context."},
    )
    full = reconstruct_full_disposition(
        simple, [{"value": "blue", "labels": ["favorite_color"]}], []
    )
    assert full.sensitivity_disposition[0].needs_protection is False


# ---------------------------------------------------------------------------
# Prompt enumeration alignment with _flatten_context
# ---------------------------------------------------------------------------


def test_flatten_context_enumeration_matches_reconstruction_for_multi_label() -> None:
    """Pin the (value, label) pair contract: _flatten_context produces one
    slot per pair, and reconstruct_full_disposition expects the LLM to
    emit one SimpleDispositionItem per slot. With ids 1..N and zero
    orphans, every multi-label entity round-trips cleanly."""
    entities_by_value = [
        {"value": "John", "labels": ["first_name", "organization_name"]},
        {"value": "Portland", "labels": ["city"]},
    ]
    flat = _flatten_context(entities_by_value, [])
    # 3 slots: (John, first_name), (John, organization_name), (Portland, city)
    assert len(flat) == 3
    assert [s["entity_label"] for s in flat] == [
        "first_name", "organization_name", "city",
    ]
    # Construct one SimpleDispositionItem per slot, ids 1..N.
    simple = _make_simple(
        *[
            {"id": i + 1, "category": "direct_identifier", "sensitivity": "high",
             "protection_method_suggestion": "replace",
             "protection_reason": f"Aligned reason {i + 1}."}
            for i in range(len(flat))
        ]
    )
    full = reconstruct_full_disposition(simple, entities_by_value, [])
    # All three reconstructed; zero orphans.
    assert len(full.sensitivity_disposition) == 3
    assert [e.entity_label for e in full.sensitivity_disposition] == [
        "first_name", "organization_name", "city",
    ]
    assert [e.entity_value for e in full.sensitivity_disposition] == [
        "John", "John", "Portland",
    ]


def test_flatten_context_includes_latent_after_tagged() -> None:
    """Latent entries follow tagged entries; ids continue sequentially.
    Pins the prompt-rendering contract: the LLM sees tagged before latent
    in the prompt and must emit ids 1..N in that order."""
    entities_by_value = [{"value": "Alice", "labels": ["first_name"]}]
    latent = [{"label": "employer", "value": "UW"}]
    flat = _flatten_context(entities_by_value, latent)
    assert [s["source"] for s in flat] == ["tagged", "latent"]
    assert [s["entity_label"] for s in flat] == ["first_name", "employer"]
