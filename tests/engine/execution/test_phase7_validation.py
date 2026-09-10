# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import itertools
import pickle
from dataclasses import FrozenInstanceError, replace
from types import ModuleType
from typing import cast

import pytest

from anonymizer.engine.execution.graph import _CoherenceScope
from anonymizer.engine.execution.mention_resolution import _ClusterId
from anonymizer.engine.execution.phase6_runtime import _Phase6SubstituteHandoff
from anonymizer.engine.execution.phase7_admission import _Phase7Plan, _ScopeManifest
from anonymizer.engine.execution.phase7_contract import _load_phase7_contract
from anonymizer.engine.execution.phase7_validation import (
    _BundleRejected,
    _CandidateAssignment,
    _ValidatedBundle,
)
from tests.engine.execution.test_phase7_admission import (
    _compile_phase7,
    _ids,
    _person_relation_fixture,
    _Proposal,
    _qualified_phase6,
)


def _validation_module() -> ModuleType:
    module_name = "anonymizer.engine.execution.phase7_validation"
    assert importlib.util.find_spec(module_name) is not None, "the private Phase 7 validation module is missing"
    return importlib.import_module(module_name)


def _compiled_scope(
    texts: tuple[str, ...],
    scopes: tuple[tuple[str, ...], ...],
    proposals: dict[str, tuple[_Proposal, ...]],
    *,
    combined_scope: bool = False,
) -> tuple[_ScopeManifest, tuple[_Phase6SubstituteHandoff, ...]]:
    plan, _backend, execution = _qualified_phase6(texts, scopes, proposals)
    declared = (_CoherenceScope(_ids(plan)),) if combined_scope else plan.coherence_scopes
    compiled = _compile_phase7(plan, execution, declared)
    assert isinstance(compiled, _Phase7Plan)
    assert len(compiled.manifests) == 1
    return compiled.manifests[0], execution.handoffs


def _assignments_for_roles(manifest: _ScopeManifest, values: dict[str, str]) -> tuple[_CandidateAssignment, ...]:
    module = _validation_module()
    assignment_type = getattr(module, "_CandidateAssignment")
    slots = getattr(manifest, "slots")
    return tuple(assignment_type(slot.id, values[slot.role]) for slot in slots)


def _validate(
    manifest: _ScopeManifest,
    handoffs: object,
    assignments: object,
) -> _ValidatedBundle | _BundleRejected:
    module = _validation_module()
    validator = getattr(module, "_validate_scope_bundle", None)
    assert callable(validator), "the private Phase 7 complete-bundle validator is missing"
    return validator(manifest, handoffs, assignments, _load_phase7_contract())


def _code(result: object) -> str:
    code = getattr(result, "code", None)
    assert code is not None, "malformed Phase 7 candidate did not return a typed rejection"
    value = getattr(code, "value", None)
    assert isinstance(value, str)
    return value


def _validated_values(result: object) -> tuple[str, ...]:
    assignments = getattr(result, "assignments", None)
    assert isinstance(assignments, tuple)
    return tuple(getattr(assignment, "value") for assignment in assignments)


def _all_permutations(values: tuple[_CandidateAssignment, ...]) -> tuple[tuple[_CandidateAssignment, ...], ...]:
    return tuple(itertools.permutations(values))


def test_phase7_canonicalizer_exactly_matches_the_p0_unicode_algorithm() -> None:
    module = _validation_module()
    canonicalize = getattr(module, "_canonicalize_value", None)
    assert callable(canonicalize), "the single private Phase 7 canonicalizer is missing"

    assert canonicalize("\u2003 Ａ-lí_ce \u2003") == "alíce"
    assert canonicalize("Straße") == "strasse"
    assert canonicalize("É９") == "é9"
    assert canonicalize("e\u0301") == "é"
    assert canonicalize("-_ .") is None
    assert canonicalize(None) is None


@pytest.mark.parametrize(
    ("format_name", "value", "expected"),
    [
        pytest.param("unicode_person_name/v1", "Élodie Marie-José.", True, id="person-unicode"),
        pytest.param("unicode_person_name/v1", "A" * 128, True, id="person-128-bytes"),
        pytest.param("unicode_person_name/v1", "A" * 129, False, id="person-129-bytes"),
        pytest.param("unicode_person_name/v1", "é" * 64, True, id="person-128-multibyte-bytes"),
        pytest.param("unicode_person_name/v1", "é" * 65, False, id="person-130-multibyte-bytes"),
        pytest.param("unicode_person_name/v1", "A\u00a0B", True, id="person-zs"),
        pytest.param("unicode_person_name/v1", "A\u2028B", False, id="person-zl"),
        pytest.param("unicode_person_name/v1", "A\u2029B", False, id="person-zp"),
        pytest.param("unicode_person_name/v1", "A_1", False, id="person-disallowed"),
        pytest.param("username_ascii/v1", "a", True, id="username-min"),
        pytest.param("username_ascii/v1", "a" * 64, True, id="username-max"),
        pytest.param("username_ascii/v1", "a" * 65, False, id="username-over"),
        pytest.param("username_ascii/v1", "_alice", False, id="username-left-boundary"),
        pytest.param("username_ascii/v1", "alice-", False, id="username-right-boundary"),
        pytest.param("username_ascii/v1", "álîce", False, id="username-nonascii"),
        pytest.param("telephone_ascii/v1", "+1 (555) 010-0200", True, id="phone-valid"),
        pytest.param("telephone_ascii/v1", "123456", False, id="phone-six-digits"),
        pytest.param("telephone_ascii/v1", "1234567890123456", False, id="phone-sixteen-digits"),
        pytest.param("telephone_ascii/v1", "1+234567", False, id="phone-nonleading-plus"),
        pytest.param("telephone_ascii/v1", "+123+4567", False, id="phone-two-plus"),
        pytest.param("telephone_ascii/v1", "123/4567", False, id="phone-invalid-character"),
        pytest.param("email_addr_spec_ascii/v1", "a.b+tag@example-domain.com", True, id="email-valid"),
        pytest.param("email_addr_spec_ascii/v1", ".alice@example.com", False, id="email-leading-dot"),
        pytest.param("email_addr_spec_ascii/v1", "alice..x@example.com", False, id="email-double-dot"),
        pytest.param("email_addr_spec_ascii/v1", "alice@-example.com", False, id="email-label-hyphen"),
        pytest.param("email_addr_spec_ascii/v1", "alice@example.c", False, id="email-short-tld"),
        pytest.param("email_addr_spec_ascii/v1", "álîce@example.com", False, id="email-nonascii"),
        pytest.param("unknown/v1", "Alice", False, id="unknown-format"),
    ],
)
def test_phase7_format_predicates_are_total_and_closed(format_name: str, value: object, expected: bool) -> None:
    validator = getattr(_validation_module(), "_matches_format", None)
    assert callable(validator), "the private Phase 7 format validator is missing"

    assert validator(format_name, value) is expected


def test_phase7_email_format_enforces_every_exact_byte_and_label_boundary() -> None:
    validator = getattr(_validation_module(), "_matches_format", None)
    assert callable(validator)
    exact = f"{'a' * 64}@{'b' * 63}.{'c' * 63}.{'d' * 61}"

    assert len(exact.encode("utf-8")) == 254
    assert validator("email_addr_spec_ascii/v1", exact)
    assert not validator("email_addr_spec_ascii/v1", f"{exact}e")
    assert not validator("email_addr_spec_ascii/v1", f"{'a' * 65}@example.com")
    assert not validator("email_addr_spec_ascii/v1", f"alice@{'b' * 64}.com")
    assert validator("email_addr_spec_ascii/v1", f"alice@example.{'z' * 63}")
    assert not validator("email_addr_spec_ascii/v1", f"alice@example.{'z' * 64}")


@pytest.mark.parametrize(
    ("mask_name", "source", "candidate", "expected"),
    [
        pytest.param("none/v1", "anything", "different", True, id="none"),
        pytest.param("digit_literal/v1", "+１ (555)-0100", "+2 (212)-9999", True, id="nfkc-source"),
        pytest.param("digit_literal/v1", "555-0100", "212-9999", True, id="changed-digits"),
        pytest.param("digit_literal/v1", "555-0100", "212 9999", False, id="changed-literal"),
        pytest.param("digit_literal/v1", "555-0100", "212-999", False, id="length"),
        pytest.param("digit_literal/v1", "555-0100", "２12-9999", True, id="nfkc-candidate"),
        pytest.param("unknown/v1", "555-0100", "212-9999", False, id="unknown-mask"),
    ],
)
def test_phase7_source_masks_are_total_and_exact(
    mask_name: str,
    source: object,
    candidate: object,
    expected: bool,
) -> None:
    validator = getattr(_validation_module(), "_matches_mask", None)
    assert callable(validator), "the private Phase 7 source-mask validator is missing"

    assert validator(mask_name, source, candidate) is expected


def test_phase7_bundle_requires_every_exact_opaque_slot_token_once() -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice Adams",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"), _Proposal("Adams", "last_name", "person"))},
    )
    module = _validation_module()
    assignment_type = getattr(module, "_CandidateAssignment")
    assignments = _assignments_for_roles(
        manifest,
        {"person_given_name": "Nova", "person_family_name": "Vale"},
    )

    accepted = _validate(manifest, handoffs, assignments)
    assert isinstance(accepted, getattr(module, "_ValidatedBundle"))
    assert _validated_values(accepted) == ("Nova", "Vale")
    assert _validated_values(_validate(manifest, handoffs, tuple(reversed(assignments)))) == ("Nova", "Vale")
    assert _code(_validate(manifest, handoffs, assignments[:-1])) == "partial_bundle"
    assert _code(_validate(manifest, handoffs, (*assignments, assignments[0]))) == "duplicate_slot"
    duplicate_divergent = (*assignments, assignment_type(assignments[0].token, "Mira"))
    assert _code(_validate(manifest, handoffs, duplicate_divergent)) == "duplicate_slot"

    foreign_manifest, _foreign_handoffs = _compiled_scope(
        ("Mira",),
        (("target-0",),),
        {"target-0": (_Proposal("Mira", "first_name", "other"),)},
    )
    foreign = assignment_type(foreign_manifest.slots[0].id, "Tess")
    assert _code(_validate(manifest, handoffs, (foreign, *assignments[1:]))) == "foreign_slot"
    assert _code(_validate(manifest, handoffs, (*assignments, foreign))) == "foreign_slot"


@pytest.mark.parametrize(
    "malformed",
    [None, [], (), (object(),), (("not", "an assignment"),)],
    ids=["none", "list", "empty", "object", "pair"],
)
def test_phase7_bundle_validation_is_total_for_malformed_untrusted_input(malformed: object) -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
    )

    result = _validate(manifest, handoffs, malformed)

    assert _code(result) in {"invalid_input", "partial_bundle"}


def test_phase7_bundle_validation_rejects_nontext_empty_and_unencodable_values_without_repair() -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
    )
    assignment_type = getattr(_validation_module(), "_CandidateAssignment")
    token = manifest.slots[0].id

    assert _code(_validate(manifest, handoffs, (assignment_type(token, 7),))) == "invalid_input"
    assert _code(_validate(manifest, handoffs, (assignment_type(token, ""),))) == "candidate_matches_original"
    assert _code(_validate(manifest, handoffs, (assignment_type(token, " -_. "),))) == "candidate_matches_original"
    assert _code(_validate(manifest, handoffs, (assignment_type(token, "\ud800"),))) == "candidate_matches_original"


def test_phase7_candidate_must_differ_from_every_original_across_the_complete_scope() -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice", "Bob"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (_Proposal("Alice", "first_name", "alice"),),
            "target-1": (_Proposal("Bob", "first_name", "bob"),),
        },
        combined_scope=True,
    )
    module = _validation_module()
    assignment_type = getattr(module, "_CandidateAssignment")
    assignments = tuple(
        assignment_type(slot.id, "Nova" if index == 0 else "A-l.i.c.e") for index, slot in enumerate(manifest.slots)
    )

    assert _code(_validate(manifest, handoffs, assignments)) == "candidate_matches_original"


def test_phase7_candidate_collision_checks_do_not_cross_scope_boundaries() -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice", "Bob"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (_Proposal("Alice", "first_name", "alice"),),
            "target-1": (_Proposal("Bob", "first_name", "bob"),),
        },
    )
    compiled = _compile_phase7(plan, execution, plan.coherence_scopes)
    assert isinstance(compiled, _Phase7Plan)
    manifest = compiled.manifests[0]
    assignment_type = getattr(_validation_module(), "_CandidateAssignment")

    accepted = _validate(manifest, execution.handoffs, (assignment_type(manifest.slots[0].id, "Bob"),))

    assert isinstance(accepted, getattr(_validation_module(), "_ValidatedBundle"))


def test_phase7_every_compiled_required_pair_is_enforced() -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice Adams Brenda Chloe",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "a"),
                _Proposal("Adams", "last_name", "a"),
                _Proposal("Brenda", "first_name", "b"),
                _Proposal("Chloe", "first_name", "c"),
            ),
        },
    )
    module = _validation_module()
    assignment_type = getattr(module, "_CandidateAssignment")
    baseline = ("Nova", "Vale", "Mira", "Tess")
    positions = {slot.id: index for index, slot in enumerate(manifest.slots)}

    assert len(manifest.required_pairs) == 6
    for pair in manifest.required_pairs:
        values = list(baseline)
        values[positions[pair.left]] = "S-am"
        values[positions[pair.right]] = "S.am"
        assignments = tuple(assignment_type(slot.id, values[index]) for index, slot in enumerate(manifest.slots))
        assert _code(_validate(manifest, handoffs, assignments)) == "canonical_collision"


def test_phase7_shared_slot_accepts_one_value_and_rejects_divergent_duplicate_assignments() -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice Alicia",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "person"),
                _Proposal("Alicia", "first_name", "person"),
            )
        },
    )
    module = _validation_module()
    assignment_type = getattr(module, "_CandidateAssignment")
    token = manifest.slots[0].id
    one = (assignment_type(token, "Nova"),)

    assert len(manifest.slots) == 1
    assert len(manifest.slots[0].mention_ids) == 2
    assert isinstance(_validate(manifest, handoffs, one), getattr(module, "_ValidatedBundle"))
    assert _code(_validate(manifest, handoffs, (*one, assignment_type(token, "Nova")))) == "duplicate_slot"
    assert _code(_validate(manifest, handoffs, (*one, assignment_type(token, "Vale")))) == "duplicate_slot"


def test_phase7_reused_slot_checks_the_source_mask_for_every_bound_mention() -> None:
    manifest, handoffs = _compiled_scope(
        ("555-0100 and (555)0100",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("555-0100", "phone_number", "phone"),
                _Proposal("(555)0100", "phone_number", "phone"),
            )
        },
    )
    assignment = _assignments_for_roles(manifest, {"voice_phone_number": "212-9999"})

    assert _code(_validate(manifest, handoffs, assignment)) == "relation_failed"


def test_phase7_email_relation_is_a_canonical_local_part_bundle_predicate() -> None:
    plan, _backend, execution, cluster_value = _person_relation_fixture()
    cluster = cast(_ClusterId, cluster_value)
    admission = importlib.import_module("anonymizer.engine.execution.phase7_admission")
    relation = admission._RelationDeclaration(
        "email_from_name/v1",
        (
            admission._ClusterRoleSelector("cluster_role/v1", cluster, "person_given_name"),
            admission._ClusterRoleSelector("cluster_role/v1", cluster, "person_family_name"),
        ),
        admission._ClusterRoleSelector("cluster_role/v1", cluster, "email_address"),
    )
    compiled = _compile_phase7(plan, execution, plan.coherence_scopes, (relation,))
    assert isinstance(compiled, _Phase7Plan)
    manifest = compiled.manifests[0]

    accepted = _assignments_for_roles(
        manifest,
        {
            "person_given_name": "Nova",
            "person_family_name": "Vale",
            "email_address": "v-ale@example.com",
        },
    )
    domain_only = _assignments_for_roles(
        manifest,
        {
            "person_given_name": "Nova",
            "person_family_name": "Vale",
            "email_address": "opaque@nova.example.com",
        },
    )
    omitted = _assignments_for_roles(
        manifest,
        {
            "person_given_name": "Nova",
            "person_family_name": "Vale",
            "email_address": "opaque@example.com",
        },
    )

    assert isinstance(
        _validate(manifest, execution.handoffs, accepted), getattr(_validation_module(), "_ValidatedBundle")
    )
    assert _code(_validate(manifest, execution.handoffs, domain_only)) == "relation_failed"
    assert _code(_validate(manifest, execution.handoffs, omitted)) == "relation_failed"


def test_phase7_reachable_candidate_byte_limit_accepts_exact_and_rejects_one_over() -> None:
    source = "1" * 15 + " " * 241
    exact = "2" * 15 + " " * 241
    manifest, handoffs = _compiled_scope(
        (source,),
        (("target-0",),),
        {"target-0": (_Proposal(source, "phone_number", "phone"),)},
    )
    assignment_type = getattr(_validation_module(), "_CandidateAssignment")
    token = manifest.slots[0].id

    assert len(exact.encode("utf-8")) == 256
    assert isinstance(
        _validate(manifest, handoffs, (assignment_type(token, exact),)),
        getattr(_validation_module(), "_ValidatedBundle"),
    )
    assert _code(_validate(manifest, handoffs, (assignment_type(token, f"{exact} "),))) == "limit_exceeded"


def test_phase7_empty_manifest_accepts_only_the_complete_empty_bundle() -> None:
    manifest, handoffs = _compiled_scope(("plain text",), (("target-0",),), {})
    module = _validation_module()

    accepted = _validate(manifest, handoffs, ())

    assert isinstance(accepted, getattr(module, "_ValidatedBundle"))
    assert accepted.assignments == ()
    assert _code(_validate(manifest, handoffs, None)) == "invalid_input"


def test_phase7_validation_rejects_stale_manifest_and_handoff_inputs() -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
    )
    assignments = _assignments_for_roles(manifest, {"person_given_name": "Nova"})
    stale = replace(manifest, members=())

    assert _code(_validate(stale, handoffs, assignments)) == "invalid_input"
    assert _code(_validate(manifest, (), assignments)) == "invalid_input"
    assert _code(_validate(manifest, (object(),), assignments)) == "invalid_input"


def test_phase7_validated_bundle_is_private_immutable_and_value_order_is_permutation_invariant() -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice Adams alice@example.com",),
        (("target-0",),),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "person"),
                _Proposal("Adams", "last_name", "person"),
                _Proposal("alice@example.com", "email", "person"),
            )
        },
    )
    assignments = _assignments_for_roles(
        manifest,
        {
            "person_given_name": "Nova",
            "person_family_name": "Vale",
            "email_address": "nova@example.com",
        },
    )
    baseline = _validate(manifest, handoffs, assignments)

    for permutation in _all_permutations(assignments):
        assert _validated_values(_validate(manifest, handoffs, permutation)) == _validated_values(baseline)
    with pytest.raises(FrozenInstanceError):
        setattr(baseline, "assignments", ())
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(baseline)
    assert "Nova" not in repr(baseline)


def test_phase7_validated_bundle_recursively_rejects_nested_manifest_and_source_mutations() -> None:
    manifest, handoffs = _compiled_scope(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
    )
    assignments = _assignments_for_roles(manifest, {"person_given_name": "Nova"})
    validated = _validate(manifest, handoffs, assignments)
    module = _validation_module()
    admitted = getattr(module, "_is_validated_bundle", None)
    assert callable(admitted)
    assert isinstance(validated, _ValidatedBundle)
    assert admitted(validated)

    stale_manifest = replace(manifest, members=())
    assert not admitted(replace(validated, manifest=stale_manifest))

    handoff = handoffs[0]
    resolved = handoff.resolved
    resolved_mention = resolved.mentions[0]
    stale_mention = replace(resolved_mention.mention, source_slice="Mallory")
    stale_resolved_mention = replace(resolved_mention, mention=stale_mention)
    stale_resolved = replace(resolved, mentions=(stale_resolved_mention, *resolved.mentions[1:]))
    stale_handoff = replace(handoff, resolved=stale_resolved)
    stale_bundle = replace(validated, handoffs=(stale_handoff, *handoffs[1:]))

    assert not admitted(stale_bundle)
    assert _code(_validate(manifest, (stale_handoff, *handoffs[1:]), assignments)) == "invalid_input"


def test_phase7_validation_has_no_planner_or_runtime_effects() -> None:
    plan, backend, execution = _qualified_phase6(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
    )
    compiled = _compile_phase7(plan, execution, plan.coherence_scopes)
    assert isinstance(compiled, _Phase7Plan)
    manifest = compiled.manifests[0]
    assignments = _assignments_for_roles(manifest, {"person_given_name": "Nova"})
    calls_before = tuple(backend.calls)

    result = _validate(manifest, execution.handoffs, assignments)

    assert isinstance(result, getattr(_validation_module(), "_ValidatedBundle"))
    assert tuple(backend.calls) == calls_before
    assert backend.planner_effect_count == 0
    assert not {
        "_AccountingLedger",
        "_observe_context_boundary",
        "_build_context_workframes",
        "NddAdapter",
        "DataDesigner",
    } & set(vars(_validation_module()))
