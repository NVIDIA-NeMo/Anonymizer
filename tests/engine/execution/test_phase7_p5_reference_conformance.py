# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import cast

import pytest

from anonymizer.engine.execution.mention_resolution import _ClusterId
from anonymizer.engine.execution.phase7_admission import _Phase7Plan
from anonymizer.engine.execution.phase7_validation import (
    _BundleRejected,
    _CandidateAssignment,
    _ValidatedBundle,
)
from tests.engine.execution.phase7_reference_model import (
    _canonical_value as _reference_canonical_value,
)
from tests.engine.execution.phase7_reference_model import (
    _digit_mask_valid as _reference_digit_mask_valid,
)
from tests.engine.execution.phase7_reference_model import (
    _format_valid as _reference_format_valid,
)
from tests.engine.execution.phase7_reference_model import (
    case_by_name,
    reduce_reference,
)
from tests.engine.execution.test_phase7_admission import (
    _compile_phase7,
    _person_relation_fixture,
    _Proposal,
)
from tests.engine.execution.test_phase7_application import _apply, _outputs, _patch_tuple, _validated
from tests.engine.execution.test_phase7_validation import (
    _assignments_for_roles,
    _code,
    _compiled_scope,
    _validate,
)


@pytest.mark.parametrize(
    "value",
    ["\u2003 Ａ-lí_ce \u2003", "Straße", "É９", "e\u0301", "-_ .", "𐐀lice"],
)
def test_p5_canonicalization_matches_the_independent_p4_oracle(value: str) -> None:
    from anonymizer.engine.execution.phase7_validation import _canonicalize_value

    assert _canonicalize_value(value) == (_reference_canonical_value(value) or None)


@pytest.mark.parametrize(
    ("format_name", "value"),
    [
        ("unicode_person_name/v1", "Élodie Marie-José."),
        ("unicode_person_name/v1", "A\u2028B"),
        ("username_ascii/v1", "alice_01"),
        ("username_ascii/v1", "álîce"),
        ("telephone_ascii/v1", "+1 (555) 010-0200"),
        ("telephone_ascii/v1", "1+234567"),
        ("email_addr_spec_ascii/v1", "nova.vale@example.test"),
        ("email_addr_spec_ascii/v1", "nova@-example.test"),
    ],
)
def test_p5_format_predicates_match_the_independent_p4_oracle(format_name: str, value: str) -> None:
    from anonymizer.engine.execution.phase7_validation import _matches_format

    assert _matches_format(format_name, value) is _reference_format_valid(format_name, value)


@pytest.mark.parametrize(
    ("source", "candidate"),
    [("555-0100", "555-0199"), ("555-0100", "555 0199"), ("+１ (555)-0100", "+2 (212)-9999")],
)
def test_p5_digit_mask_matches_the_independent_p4_oracle(source: str, candidate: str) -> None:
    from anonymizer.engine.execution.phase7_validation import _matches_mask

    assert _matches_mask("digit_literal/v1", source, candidate) is _reference_digit_mask_valid(source, candidate)


@pytest.mark.parametrize(
    ("case_name", "text", "proposals", "values"),
    [
        (
            "owner-distinct_slots_same_canonical_value",
            "Alice Adams",
            (_Proposal("Alice", "first_name", "c0"), _Proposal("Adams", "last_name", "c0")),
            ("Nova", "Ｎｏｖａ"),
        ),
        (
            "owner-candidate_matches_own_original",
            "Alice",
            (_Proposal("Alice", "first_name", "c0"),),
            (" Alice ",),
        ),
        (
            "owner-candidate_matches_other_slot_original",
            "Alice Adams",
            (_Proposal("Alice", "first_name", "c0"), _Proposal("Adams", "last_name", "c0")),
            ("Adams", "Vale"),
        ),
    ],
)
def test_p5_scope_validation_reasons_match_the_independent_p4_oracle(
    case_name: str,
    text: str,
    proposals: tuple[_Proposal, ...],
    values: tuple[str, ...],
) -> None:
    manifest, handoffs = _compiled_scope((text,), (("target-0",),), {"target-0": proposals})
    assignments = tuple(
        _CandidateAssignment(slot.id, value) for slot, value in zip(manifest.slots, values, strict=True)
    )
    production = _validate(manifest, handoffs, assignments)
    reference = reduce_reference(case_by_name(case_name))

    assert isinstance(production, _BundleRejected)
    assert _code(production) == reference.reason_codes[0]


@pytest.mark.parametrize(
    ("case_name", "text", "proposals", "values"),
    [
        (
            "shared-slot-reuse",
            "Alice and Alicia",
            (_Proposal("Alice", "first_name", "c0"), _Proposal("Alicia", "first_name", "c0")),
            ("Nova",),
        ),
        (
            "owner-valid_phone_source_mask",
            "555-0100",
            (_Proposal("555-0100", "phone_number", "c0"),),
            ("555-0199",),
        ),
        (
            "anchored-non-cascading-application",
            "Alice met Nova",
            (_Proposal("Alice", "first_name", "c0"), _Proposal("Nova", "last_name", "c1")),
            ("Nova Blake", "Vale"),
        ),
    ],
)
def test_p5_reuse_mask_and_application_outputs_match_the_independent_p4_oracle(
    case_name: str,
    text: str,
    proposals: tuple[_Proposal, ...],
    values: tuple[str, ...],
) -> None:
    bundle, _backend = _validated((text,), (("target-0",),), {"target-0": proposals}, values)
    production = _apply(bundle, _patch_tuple(bundle))
    reference = reduce_reference(case_by_name(case_name))

    assert reference.reason_codes == (None,)
    assert _outputs(production)["target-0"][0] == reference.outputs[0][1]


def test_p5_email_relation_acceptance_and_rejection_match_the_independent_p4_oracle() -> None:
    plan, _backend, execution, cluster_value = _person_relation_fixture()
    cluster = cast(_ClusterId, cluster_value)
    from anonymizer.engine.execution import phase7_admission as admission

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

    accepted = _validate(
        manifest,
        execution.handoffs,
        _assignments_for_roles(
            manifest,
            {
                "person_given_name": "Nova",
                "person_family_name": "Vale",
                "email_address": "nova.vale@example.test",
            },
        ),
    )
    rejected = _validate(
        manifest,
        execution.handoffs,
        _assignments_for_roles(
            manifest,
            {
                "person_given_name": "Nova",
                "person_family_name": "Vale",
                "email_address": "other@example.test",
            },
        ),
    )
    accepted_reference = reduce_reference(case_by_name("owner-valid_given_family_email_relation"))
    rejected_reference = reduce_reference(case_by_name("owner-email_local_part_omits_name"))

    assert isinstance(accepted, _ValidatedBundle)
    assert _outputs(_apply(accepted, _patch_tuple(accepted)))["target-0"][0] == accepted_reference.outputs[0][1]
    assert _code(rejected) == rejected_reference.reason_codes[0]
