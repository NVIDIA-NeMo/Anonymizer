# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import hashlib
import importlib
import json
from dataclasses import fields, replace
from pathlib import Path

import pytest

from tests.engine.execution.phase7_reference_model import (
    MAX_EXOGENOUS_OBSERVATIONS,
    OWNER_CASE_IDS,
    ReferenceCase,
    ReferenceEvent,
    ReferenceEventKind,
    canonical_corpus_bytes,
    canonical_events,
    case_by_name,
    events_commute,
    finite_reference_cases,
    owner_case_outcome,
    reduce_reference,
    reference_manifest,
)


def test_phase7_reference_model_defines_timestamp_free_declarations_and_events() -> None:
    model = importlib.import_module("tests.engine.execution.phase7_reference_model")

    required = {
        "ReferenceCase",
        "ReferenceDatum",
        "ReferenceDeclaration",
        "ReferenceEvent",
        "ReferenceEventKind",
        "ReferenceMention",
        "ReferenceRelation",
        "ReferenceScope",
        "canonical_events",
        "events_commute",
    }

    assert required.issubset(vars(model)), "Phase 7 reference grammar is incomplete"
    assert tuple(kind.value for kind in model.ReferenceEventKind) == model.EVENT_ALPHABET


def test_phase7_reference_model_exposes_an_independent_finite_reducer() -> None:
    model = importlib.import_module("tests.engine.execution.phase7_reference_model")

    required = {
        "canonical_corpus_bytes",
        "corpus_document",
        "finite_reference_cases",
        "reduce_reference",
        "reference_manifest",
    }

    assert required.issubset(vars(model)), "Phase 7 reference reducer or corpus generator is missing"


def test_phase7_reference_person_name_format_matches_the_p0_zs_only_contract() -> None:
    model = importlib.import_module("tests.engine.execution.phase7_reference_model")

    assert model._format_valid("unicode_person_name/v1", "A\u00a0B")
    assert not model._format_valid("unicode_person_name/v1", "A\u2028B")
    assert not model._format_valid("unicode_person_name/v1", "A\u2029B")


def test_phase7_reference_model_preserves_all_owner_contract_cases() -> None:
    cases = {case.owner_case: case for case in finite_reference_cases() if case.owner_case is not None}
    expected = {
        "valid_empty_scope_zero_dispatch": "planned_empty",
        "valid_single_given_name": "planned",
        "valid_given_family_email_relation": "planned",
        "valid_phone_source_mask": "planned",
        "unknown_contract_version": "contract_invalid",
        "contract_digest_mismatch": "digest_mismatch",
        "missing_detector_disposition": "detector_universe_incomplete",
        "unknown_role": "unsupported_role",
        "unknown_relation": "unsupported_constraint",
        "unknown_mask": "unsupported_mask",
        "unsupported_detector_label": "unsupported_label",
        "selector_resolves_zero_slots": "selector_missing",
        "selector_resolves_multiple_slots": "selector_ambiguous",
        "relation_crosses_scopes": "cross_scope_relation",
        "email_relation_wrong_roles": "relation_role_mismatch",
        "distinct_slots_same_canonical_value": "canonical_collision",
        "candidate_matches_own_original": "candidate_matches_original",
        "candidate_matches_other_slot_original": "candidate_matches_original",
        "email_local_part_omits_name": "relation_failed",
        "count_limits_exact": "planned",
        "count_limits_one_over": "limit_exceeded",
        "byte_limits_exact": "planned",
        "byte_limits_one_over": "limit_exceeded",
        "runtime_capability_missing": "missing_capability",
        "trusted_task_failure": "failed",
        "unattributable_failure": "inconsistent_global_embargo",
        "cleanup_attestation_verified": "release_eligible",
        "cleanup_attestation_missing": "inconsistent_global_embargo",
        "cleanup_attestation_contradictory": "inconsistent_global_embargo",
        "redact_policy_role_bearing_scope": "blocked_zero_effects",
    }

    assert tuple(cases) == OWNER_CASE_IDS
    assert {case_id: owner_case_outcome(case) for case_id, case in cases.items()} == expected


@pytest.mark.parametrize("slot_count", range(5))
def test_future_policy_enumerates_zero_through_four_slots_and_all_required_pairs(slot_count: int) -> None:
    result = reduce_reference(case_by_name(f"future-slots-{slot_count}"))

    assert result.admission == "admitted"
    assert result.scope_outcomes == ("planned",)
    assert len(result.manifests[0].slots) == slot_count
    assert len(result.manifests[0].required_pairs) == slot_count * (slot_count - 1) // 2
    assert result.dispatch_count == int(slot_count > 0)
    assert result.attempt_count == int(slot_count > 0)
    assert result.released_groups == (0,)


@pytest.mark.parametrize("slot_count", range(5))
def test_current_empty_policy_blocks_every_role_bearing_scope_without_dispatch(slot_count: int) -> None:
    result = reduce_reference(case_by_name(f"current-empty-policy-slots-{slot_count}"))

    assert result.dispatch_count == 0
    assert result.attempt_count == 0
    if slot_count == 0:
        assert result.scope_outcomes == ("planned",)
        assert result.released_groups == (0,)
    else:
        assert result.scope_outcomes == ("blocked",)
        assert result.released_groups == ()


def test_independent_scopes_have_separate_slots_pairs_tasks_and_release() -> None:
    result = reduce_reference(case_by_name("independent-scopes-2-2"))

    assert result.scope_outcomes == ("planned", "planned")
    assert tuple(len(manifest.slots) for manifest in result.manifests) == (2, 2)
    assert tuple(len(manifest.required_pairs) for manifest in result.manifests) == (1, 1)
    assert result.dispatch_count == 2
    assert result.released_groups == (0, 1)
    assert tuple(task[:2] for task in result.task_outcomes if task[0] == "scope") == (
        ("scope", "0"),
        ("scope", "1"),
    )


def test_phase7_reference_lifecycle_is_absorbing_and_release_is_fail_closed() -> None:
    expected = {
        "dispatch-rejected": ("failed", "completed", (), 0),
        "backend-exception": ("failed", "completed", (), 1),
        "contradictory-candidate-evidence": ("inconsistent", "inconsistent", (), 1),
        "cancel-before-dispatch": ("cancelled", "cancelled", (), 0),
        "dispatch-cancel-without-stop": ("lost", "lost", (), 1),
        "dispatch-cancel-trusted-stop": ("cancelled", "cancelled", (), 1),
        "late-candidate-after-stop": ("cancelled", "cancelled", (), 1),
        "late-candidate-after-loss": ("lost", "lost", (), 1),
        "foreign-candidate-before-acceptance": ("inconsistent", "inconsistent", (), 1),
        "partial-candidate": ("inconsistent", "inconsistent", (), 1),
        "planned-then-foreign-is-absorbing": ("planned", "completed", (0,), 1),
        "finalization-failure": ("planned", "inconsistent", (), 1),
        "cleanup-failure": ("planned", "inconsistent", (), 1),
        "teardown-failure-after-acceptance": ("planned", "completed", (0,), 1),
        "release-then-cancel-is-absorbing": ("planned", "completed", (0,), 1),
    }

    actual = {
        name: (
            reduce_reference(case_by_name(name)).scope_outcomes[0],
            reduce_reference(case_by_name(name)).invocation,
            reduce_reference(case_by_name(name)).released_groups,
            reduce_reference(case_by_name(name)).dispatch_count,
        )
        for name in expected
    }

    assert actual == expected


def test_phase4_group_and_dependency_outcomes_define_the_only_legal_release_set() -> None:
    atomic = reduce_reference(case_by_name("atomic-group-member-failure"))
    dependent = reduce_reference(case_by_name("dependent-datum-withheld"))
    independent = reduce_reference(case_by_name("independent-scope-local-failure"))

    assert atomic.released_groups == ()
    assert dependent.released_groups == ()
    assert independent.released_groups == (1,)
    assert independent.released_datums == ("d1",)


def test_anchored_application_is_non_cascading() -> None:
    result = reduce_reference(case_by_name("anchored-non-cascading-application"))

    assert result.outputs == (("d0", "Nova Blake met Vale"),)
    assert result.released_groups == (0,)


def test_canonicalization_collapses_only_commuting_independent_observations() -> None:
    case = case_by_name("independent-scopes-1-1")
    first_dispatch, first_candidate, second_dispatch, second_candidate = case.events[:4]

    assert events_commute(first_dispatch, second_dispatch, case.declaration)
    assert canonical_events((first_dispatch, second_dispatch), case.declaration) == canonical_events(
        (second_dispatch, first_dispatch), case.declaration
    )
    assert not events_commute(first_dispatch, first_candidate, case.declaration)
    assert canonical_events((first_dispatch, first_candidate), case.declaration) != canonical_events(
        (first_candidate, first_dispatch), case.declaration
    )
    assert reduce_reference(case) == reduce_reference(
        replace(case, events=(second_dispatch, second_candidate, first_dispatch, first_candidate, *case.events[4:]))
    )


def test_reference_model_rejects_a_trace_over_the_frozen_bound() -> None:
    case = case_by_name("future-slots-1")
    excessive = replace(
        case,
        events=tuple(
            ReferenceEvent(ReferenceEventKind.CANCELLATION) for _index in range(MAX_EXOGENOUS_OBSERVATIONS + 1)
        ),
    )

    with pytest.raises(AssertionError, match="16-observation"):
        reduce_reference(excessive)


def test_reference_model_forbidden_import_and_input_boundary_is_structural() -> None:
    source_path = Path(__file__).with_name("phase7_reference_model.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module is not None
    } | {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}

    assert not any(
        module == forbidden or module.startswith(f"{forbidden}.")
        for module in imported_modules
        for forbidden in ("anonymizer", "pandas", "data_designer", "datadesigner")
    )
    assert tuple(field.name for field in fields(ReferenceCase)) == (
        "name",
        "declaration",
        "events",
        "owner_case",
    )


def test_phase7_reference_manifest_freezes_exact_counts_serialization_and_digest() -> None:
    frozen = json.loads(Path(__file__).with_name("phase7_reference_manifest.json").read_text(encoding="utf-8"))
    generated = reference_manifest()

    assert generated == frozen
    assert generated["reference_model_version"] == "phase7-reference-model/v1"
    assert generated["generator_version"] == "phase7-finite-envelope/v1"
    assert generated["graph_count"] == 54
    assert generated["case_count"] == 83
    assert generated["canonical_trace_count"] == 78
    assert generated["actual_event_count"] == 549
    assert generated["owner_case_count"] == 30
    assert generated["max_exogenous_observations"] == 16
    assert generated["digest"] == hashlib.sha256(canonical_corpus_bytes()).hexdigest()
    assert not canonical_corpus_bytes().endswith(b"\n")


def test_corpus_serialization_is_invariant_to_case_and_commuting_schedule_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = importlib.import_module("tests.engine.execution.phase7_reference_model")
    cases = finite_reference_cases()
    baseline = canonical_corpus_bytes()

    monkeypatch.setattr(model, "finite_reference_cases", lambda: tuple(reversed(cases)))
    assert canonical_corpus_bytes() == baseline

    target = next(case for case in cases if case.name == "independent-scopes-1-1")
    first_dispatch, first_candidate, second_dispatch, second_candidate = target.events[:4]
    reordered = replace(
        target,
        events=(second_dispatch, second_candidate, first_dispatch, first_candidate, *target.events[4:]),
    )
    monkeypatch.setattr(
        model,
        "finite_reference_cases",
        lambda: tuple(reordered if case.name == target.name else case for case in cases),
    )
    assert canonical_corpus_bytes() == baseline
