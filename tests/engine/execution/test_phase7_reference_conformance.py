# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable

import pytest

from anonymizer.engine.execution.graph import _CoherenceScope, _DatumId
from anonymizer.engine.execution.phase7_admission import _Phase7Plan, _ScopeManifest
from tests.engine.execution.phase7_reference_model import (
    ReferenceManifest,
    case_by_name,
    reduce_reference,
)
from tests.engine.execution.test_phase7_admission import (
    _compile_phase7,
    _ids,
    _phase7_module,
    _Proposal,
    _qualified_phase6,
    _rejection_code,
)

_LABELS = ("first_name", "last_name", "email", "phone_number")
_SOURCES = ("Alice", "Adams", "alice@example.com", "555-0100")


@pytest.mark.parametrize("slot_count", range(5))
def test_p3_sealed_manifest_matches_the_independent_zero_to_four_slot_oracle(slot_count: int) -> None:
    labels = _LABELS[:slot_count]
    sources = _SOURCES[:slot_count]
    text = " ".join(sources) if sources else "plain-0"
    plan, _backend, execution = _qualified_phase6(
        (text,),
        (("target-0",),),
        {"target-0": tuple(_Proposal(source, label, "c0") for label, source in zip(labels, sources, strict=True))},
    )
    production = _compile_phase7(plan, execution, plan.coherence_scopes)
    reference = reduce_reference(case_by_name(f"future-slots-{slot_count}"))
    module = _phase7_module()

    assert isinstance(production, module._Phase7Plan)
    assert module._is_admitted_phase7_plan(production)
    assert _production_manifest_shape(production.manifests) == _reference_manifest_shape(reference.manifests)


def test_p3_sealed_manifests_match_two_independent_reference_scopes() -> None:
    plan, _backend, execution = _qualified_phase6(
        ("Alice Adams", "Bob Stone"),
        (("target-0",), ("target-1",)),
        {
            "target-0": (
                _Proposal("Alice", "first_name", "c0"),
                _Proposal("Adams", "last_name", "c0"),
            ),
            "target-1": (
                _Proposal("Bob", "first_name", "c1"),
                _Proposal("Stone", "last_name", "c1"),
            ),
        },
    )

    production = _compile_phase7(plan, execution, plan.coherence_scopes)
    reference = reduce_reference(case_by_name("independent-scopes-2-2"))

    assert isinstance(production, _Phase7Plan)
    assert _production_manifest_shape(production.manifests) == _reference_manifest_shape(reference.manifests)


@pytest.mark.parametrize(
    ("case_name", "production_scopes", "expected"),
    [
        (
            "admission-empty-scope",
            lambda ids: (_CoherenceScope(()), _CoherenceScope(ids)),
            "empty_scope",
        ),
        (
            "admission-duplicate-scope",
            lambda ids: (_CoherenceScope(ids), _CoherenceScope(tuple(reversed(ids)))),
            "duplicate_scope",
        ),
        (
            "admission-duplicate-member",
            lambda ids: (
                _CoherenceScope((ids[0], ids[0], ids[1])),
                _CoherenceScope((ids[2], ids[3])),
            ),
            "duplicate_scope_member",
        ),
        (
            "admission-unknown-datum",
            lambda ids: (
                _CoherenceScope((ids[0], ids[1], ids[2])),
                _CoherenceScope((_DatumId("foreign"),)),
            ),
            "unknown_scope_datum",
        ),
        (
            "admission-coverage-gap",
            lambda ids: (_CoherenceScope((ids[0], ids[1], ids[2])),),
            "scope_coverage_gap",
        ),
        (
            "admission-overlap",
            lambda ids: (
                _CoherenceScope((ids[0], ids[1])),
                _CoherenceScope((ids[1], ids[2], ids[3])),
            ),
            "scope_overlap",
        ),
        (
            "admission-nesting",
            lambda ids: (_CoherenceScope(ids), _CoherenceScope((ids[0], ids[1]))),
            "unsupported_scope_nesting",
        ),
    ],
)
def test_p3_admission_rejection_matches_the_independent_oracle(
    case_name: str,
    production_scopes: Callable[[tuple[_DatumId, ...]], tuple[_CoherenceScope, ...]],
    expected: str,
) -> None:
    plan, _backend, execution = _qualified_phase6(
        ("one", "two", "three", "four"),
        (("target-0",), ("target-1",), ("target-2",), ("target-3",)),
        {},
    )
    scopes = production_scopes(_ids(plan))

    production = _compile_phase7(plan, execution, scopes)
    reference = reduce_reference(case_by_name(case_name))

    assert reference.admission == expected
    assert _rejection_code(production) == expected


def _production_manifest_shape(manifests: tuple[_ScopeManifest, ...]) -> tuple[object, ...]:
    normalized = []
    for manifest in manifests:
        slots = manifest.slots
        positions = {slot.id: index for index, slot in enumerate(slots)}
        normalized.append(
            (
                len(manifest.members),
                tuple((slot.role, len(slot.mention_ids)) for slot in slots),
                tuple((positions[pair.left], positions[pair.right]) for pair in manifest.required_pairs),
                tuple(
                    (
                        relation.version,
                        tuple(positions[slot_id] for slot_id in relation.upstream),
                        positions[relation.downstream],
                    )
                    for relation in manifest.relations
                ),
            )
        )
    return tuple(normalized)


def _reference_manifest_shape(manifests: tuple[ReferenceManifest, ...]) -> tuple[object, ...]:
    normalized = []
    for manifest in manifests:
        positions = {slot.key: index for index, slot in enumerate(manifest.slots)}
        normalized.append(
            (
                len(manifest.members),
                tuple((slot.role, len(slot.mention_indexes)) for slot in manifest.slots),
                tuple((positions[left], positions[right]) for left, right in manifest.required_pairs),
                tuple(
                    (
                        version,
                        tuple(positions[slot_key] for slot_key in upstream),
                        positions[downstream],
                    )
                    for version, upstream, downstream in manifest.relations
                ),
            )
        )
    return tuple(normalized)
