# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib
import importlib.util
import itertools
import pickle
from dataclasses import FrozenInstanceError, dataclass, replace
from types import ModuleType
from typing import Any

import pytest

from anonymizer.engine.execution.graph import _CoherenceScope
from anonymizer.engine.execution.mention_admission import _MentionId, _ValidationDecision, _ValidationDecisionKind
from anonymizer.engine.execution.mention_resolution import _EvidenceVersion, _SameSubjectEvidence
from anonymizer.engine.execution.phase6_plan import _compile_phase6_plan, _Phase6Plan, _Phase6ProfileVersion
from anonymizer.engine.execution.phase6_runtime import (
    _CandidateProposal,
    _Phase6CandidateWork,
    _Phase6Execution,
    _Phase6ResolverWork,
    _Phase6Runtime,
)
from anonymizer.engine.execution.phase7_admission import _Phase7Plan
from anonymizer.engine.execution.phase7_contract import _load_phase7_contract
from anonymizer.engine.execution.phase7_validation import (
    _CandidateAssignment,
    _validate_scope_bundle,
    _ValidatedBundle,
)
from tests.engine.execution.test_phase7_admission import (
    _ACCOUNTING_LIMITS,
    _MENTION_LIMITS,
    _compile_phase7,
    _contract_and_capability,
    _graph,
    _Proposal,
    _qualified_phase6,
)


@dataclass(frozen=True)
class _SpanProposal:
    start: int
    end: int
    source: str
    label: str
    cluster: str


class _SpanBackend:
    def __init__(self, proposals: dict[str, tuple[_SpanProposal, ...]]) -> None:
        self._proposals = proposals
        self.calls: list[str] = []

    def context_capability(self) -> object:
        return _contract_and_capability()[1]

    def detect(self, work: _Phase6CandidateWork) -> tuple[_CandidateProposal, ...]:
        self.calls.append("detect")
        return tuple(
            _CandidateProposal(item.start, item.end, item.source, item.label)
            for item in self._proposals.get(work.target.datum_id.value, ())
        )

    def augment(self, work: object) -> tuple[()]:
        del work
        self.calls.append("augment")
        return ()

    def validate(self, work: object) -> tuple[_ValidationDecision, ...]:
        self.calls.append("validate")
        candidates = getattr(work, "candidates")
        return tuple(_ValidationDecision(item.token, _ValidationDecisionKind.KEEP) for item in candidates)

    def resolve(self, work: _Phase6ResolverWork) -> tuple[_SameSubjectEvidence, ...]:
        self.calls.append("resolve")
        by_anchor = {
            (datum_id, item.start, item.end, item.source, item.label): item
            for datum_id, proposals in self._proposals.items()
            for item in proposals
        }
        grouped: dict[str, list[_MentionId]] = {}
        for mention in work.eligible_mentions:
            proposal = by_anchor[
                (
                    mention.target_datum_id.value,
                    mention.start,
                    mention.end,
                    mention.source_slice,
                    mention.detector_label,
                )
            ]
            grouped.setdefault(proposal.cluster, []).append(mention.id)
        return tuple(
            _SameSubjectEvidence(work.owner.token, left, right, _EvidenceVersion.V1)
            for mention_ids in grouped.values()
            for left, right in itertools.pairwise(mention_ids)
        )

    def close_phase6(self) -> bool:
        return True


def _application_module() -> ModuleType:
    module_name = "anonymizer.engine.execution.phase7_application"
    assert importlib.util.find_spec(module_name) is not None, "the private Phase 7 application module is missing"
    return importlib.import_module(module_name)


def _qualified_phase6_spans(
    texts: tuple[str, ...],
    scopes: tuple[tuple[str, ...], ...],
    proposals: dict[str, tuple[_SpanProposal, ...]],
) -> tuple[_Phase6Plan, _SpanBackend, _Phase6Execution]:
    contract, capability = _contract_and_capability()
    plan = _compile_phase6_plan(
        _graph(texts, scopes),
        accounting_limits=_ACCOUNTING_LIMITS,
        context_contract=contract,
        capability=capability,
        mention_limits=_MENTION_LIMITS,
        profile_version=_Phase6ProfileVersion.SUBSTITUTE_V1,
    )
    assert isinstance(plan, _Phase6Plan)
    backend = _SpanBackend(proposals)
    execution = _Phase6Runtime(backend).run(plan)
    assert len(execution.handoffs) == len(plan.components)
    return plan, backend, execution


def _validated(
    texts: tuple[str, ...],
    scopes: tuple[tuple[str, ...], ...],
    proposals: dict[str, tuple[_Proposal, ...]],
    values: tuple[str, ...],
    *,
    combined_scope: bool = False,
) -> tuple[_ValidatedBundle, object]:
    plan, backend, execution = _qualified_phase6(texts, scopes, proposals)
    return _validated_from_execution(plan, execution, values, backend, combined_scope=combined_scope)


def _validated_spans(
    texts: tuple[str, ...],
    scopes: tuple[tuple[str, ...], ...],
    proposals: dict[str, tuple[_SpanProposal, ...]],
    values: tuple[str, ...],
    *,
    combined_scope: bool = False,
) -> tuple[_ValidatedBundle, object]:
    plan, backend, execution = _qualified_phase6_spans(texts, scopes, proposals)
    return _validated_from_execution(plan, execution, values, backend, combined_scope=combined_scope)


def _validated_from_execution(
    plan: _Phase6Plan,
    execution: _Phase6Execution,
    values: tuple[str, ...],
    backend: object,
    *,
    combined_scope: bool,
) -> tuple[_ValidatedBundle, object]:
    declared = (
        (_CoherenceScope(tuple(datum.id for datum in plan.accounting.datums)),)
        if combined_scope
        else plan.coherence_scopes
    )
    compiled = _compile_phase7(plan, execution, declared)
    assert isinstance(compiled, _Phase7Plan)
    assert len(compiled.manifests) == 1
    manifest = compiled.manifests[0]
    assert len(manifest.slots) == len(values)
    assignments = tuple(
        _CandidateAssignment(slot.id, value) for slot, value in zip(manifest.slots, values, strict=True)
    )
    result = _validate_scope_bundle(manifest, execution.handoffs, assignments, _load_phase7_contract())
    assert isinstance(result, _ValidatedBundle)
    return result, backend


def _materialize(bundle: object) -> object:
    materialize = getattr(_application_module(), "_materialize_substitute_patches", None)
    assert callable(materialize), "the private anchored-patch materializer is missing"
    return materialize(bundle)


def _apply(bundle: object, patches: object) -> object:
    apply_patches = getattr(_application_module(), "_apply_substitute_patches", None)
    assert callable(apply_patches), "the private anchored reconstruction is missing"
    return apply_patches(bundle, patches)


def _code(result: object) -> str:
    code = getattr(result, "code", None)
    assert code is not None, "invalid application did not return a typed rejection"
    value = getattr(code, "value", None)
    assert isinstance(value, str)
    return value


def _outputs(result: object) -> dict[str, tuple[str, bool]]:
    datums = getattr(result, "datums", None)
    assert isinstance(datums, tuple)
    return {item.datum_id.value: (item.output, item.applied) for item in datums}


def _patch_tuple(bundle: _ValidatedBundle) -> tuple[Any, ...]:
    patches = _materialize(bundle)
    assert isinstance(patches, tuple)
    return patches


def test_phase7_application_requires_every_exact_opaque_mention_token_once() -> None:
    bundle, _backend = _validated(
        ("Alice Adams",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"), _Proposal("Adams", "last_name", "person"))},
        ("Nova", "Vale"),
    )
    patches = _patch_tuple(bundle)
    patch_type = getattr(_application_module(), "_SubstitutePatch")

    accepted = _apply(bundle, patches)

    assert _outputs(accepted) == {"target-0": ("Nova Vale", True)}
    assert _code(_apply(bundle, patches[:-1])) == "invalid_application"
    assert _code(_apply(bundle, (*patches, patches[0]))) == "invalid_application"
    assert _code(_apply(bundle, (*patches, replace(patches[0], replacement="Mira")))) == "invalid_application"

    foreign_bundle, _foreign_backend = _validated(
        ("Mallory",),
        (("target-0",),),
        {"target-0": (_Proposal("Mallory", "first_name", "other"),)},
        ("Tess",),
    )
    foreign = _patch_tuple(foreign_bundle)[0]
    assert isinstance(foreign, patch_type)
    assert _code(_apply(bundle, (foreign, *patches[1:]))) == "invalid_application"


@pytest.mark.parametrize("malformed", [None, [], (), (object(),)], ids=["none", "list", "empty", "object"])
def test_phase7_application_is_total_for_malformed_patch_bundles(malformed: object) -> None:
    bundle, _backend = _validated(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
        ("Nova",),
    )

    assert _code(_apply(bundle, malformed)) == "invalid_application"


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        pytest.param("start", 1, id="start"),
        pytest.param("end", 4, id="end"),
        pytest.param("source_slice", "Alic", id="source-slice"),
        pytest.param("replacement", "Vale", id="replacement"),
    ],
)
def test_phase7_application_rejects_every_non_authoritative_patch_field(
    field_name: str,
    replacement: object,
) -> None:
    bundle, _backend = _validated(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
        ("Nova",),
    )
    patch = _patch_tuple(bundle)[0]

    assert _code(_apply(bundle, (replace(patch, **{field_name: replacement}),))) == "invalid_application"


def test_phase7_application_rejects_foreign_target_and_mention_tokens() -> None:
    bundle, _backend = _validated(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
        ("Nova",),
    )
    foreign_bundle, _foreign_backend = _validated(
        ("Mallory",),
        (("target-0",),),
        {"target-0": (_Proposal("Mallory", "first_name", "other"),)},
        ("Tess",),
    )
    patch = _patch_tuple(bundle)[0]
    foreign = _patch_tuple(foreign_bundle)[0]

    assert _code(_apply(bundle, (replace(patch, mention_id=foreign.mention_id),))) == "invalid_application"
    assert _code(_apply(bundle, (replace(patch, target=foreign.target),))) == "invalid_application"


def test_phase7_repeated_text_uses_only_each_authoritative_span_without_raw_fallback() -> None:
    text = "Alice Alice Alice"
    bundle, _backend = _validated_spans(
        (text,),
        (("target-0",),),
        {
            "target-0": (
                _SpanProposal(6, 11, "Alice", "first_name", "second"),
                _SpanProposal(12, 17, "Alice", "first_name", "third"),
            )
        },
        ("Nova", "Vale"),
    )

    result = _apply(bundle, _patch_tuple(bundle))

    assert _outputs(result) == {"target-0": ("Alice Nova Vale", True)}


def test_phase7_shared_slot_consumes_each_repeated_mention_once() -> None:
    text = "Alice/Alice"
    bundle, _backend = _validated_spans(
        (text,),
        (("target-0",),),
        {
            "target-0": (
                _SpanProposal(0, 5, "Alice", "first_name", "person"),
                _SpanProposal(6, 11, "Alice", "first_name", "person"),
            )
        },
        ("Nova",),
    )
    patches = _patch_tuple(bundle)

    assert len(patches) == 2
    assert len({patch.mention_id for patch in patches}) == 2
    assert _outputs(_apply(bundle, patches)) == {"target-0": ("Nova/Nova", True)}


def test_phase7_adjacent_mentions_are_reconstructed_in_source_order() -> None:
    bundle, _backend = _validated_spans(
        ("AliceAdams!",),
        (("target-0",),),
        {
            "target-0": (
                _SpanProposal(0, 5, "Alice", "first_name", "person"),
                _SpanProposal(5, 10, "Adams", "last_name", "person"),
            )
        },
        ("Nova", "Vale"),
    )

    assert _outputs(_apply(bundle, _patch_tuple(bundle))) == {"target-0": ("NovaVale!", True)}


def test_phase7_combining_and_astral_unicode_offsets_are_python_source_intervals() -> None:
    text = "😀Jose\u0301 met 𐐀lice"
    first = "Jose\u0301"
    second = "𐐀lice"
    first_start = text.index(first)
    second_start = text.index(second)
    bundle, _backend = _validated_spans(
        (text,),
        (("target-0",),),
        {
            "target-0": (
                _SpanProposal(first_start, first_start + len(first), first, "first_name", "first"),
                _SpanProposal(second_start, second_start + len(second), second, "first_name", "second"),
            )
        },
        ("René", "Nova"),
    )

    assert _outputs(_apply(bundle, _patch_tuple(bundle))) == {"target-0": ("😀René met Nova", True)}


def test_phase7_reconstruction_preserves_every_unmentioned_source_interval() -> None:
    text = "AA Alice :: Adams ZZ"
    bundle, _backend = _validated(
        (text,),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"), _Proposal("Adams", "last_name", "person"))},
        ("Alexandria", "Li"),
    )

    assert _outputs(_apply(bundle, _patch_tuple(bundle))) == {"target-0": ("AA Alexandria :: Li ZZ", True)}


def test_phase7_patch_input_permutations_have_one_canonical_reconstruction() -> None:
    bundle, _backend = _validated(
        ("Alice Adams",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"), _Proposal("Adams", "last_name", "person"))},
        ("Nova", "Vale"),
    )
    patches = _patch_tuple(bundle)

    outputs = {_outputs(_apply(bundle, permutation))["target-0"] for permutation in itertools.permutations(patches)}

    assert outputs == {("Nova Vale", True)}


def test_phase7_application_never_searches_or_mutates_evolving_output() -> None:
    bundle, _backend = _validated(
        ("Alice met Nova",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "alice"), _Proposal("Nova", "first_name", "nova"))},
        ("Nova Blake", "Vale"),
    )

    result = _apply(bundle, _patch_tuple(bundle))

    assert _outputs(result) == {"target-0": ("Nova Blake met Vale", True)}


def test_phase7_missing_application_never_releases_an_admitted_original() -> None:
    bundle, _backend = _validated(
        ("Alice and Bob",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "alice"), _Proposal("Bob", "first_name", "bob"))},
        ("Nova", "Vale"),
    )
    patches = _patch_tuple(bundle)

    rejected = _apply(bundle, patches[:1])

    assert _code(rejected) == "invalid_application"
    assert not hasattr(rejected, "datums")


def test_phase7_empty_scope_and_unmentioned_members_pass_through_from_source() -> None:
    empty, _empty_backend = _validated(("plain 😀 text",), (("target-0",),), {}, ())
    mixed, _mixed_backend = _validated(
        ("Alice", "plain 😀 text"),
        (("target-0",), ("target-1",)),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
        ("Nova",),
        combined_scope=True,
    )

    assert _patch_tuple(empty) == ()
    assert _outputs(_apply(empty, ())) == {"target-0": ("plain 😀 text", False)}
    assert _outputs(_apply(mixed, _patch_tuple(mixed))) == {
        "target-0": ("Nova", True),
        "target-1": ("plain 😀 text", False),
    }


def test_phase7_application_rejects_a_stale_validated_bundle_before_materialization() -> None:
    bundle, _backend = _validated(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
        ("Nova",),
    )
    stale_manifest = replace(bundle.manifest, members=())
    stale_bundle = replace(bundle, manifest=stale_manifest)

    assert _code(_materialize(stale_bundle)) == "invalid_application"
    assert _code(_apply(stale_bundle, ())) == "invalid_application"


def test_phase7_applied_result_is_private_immutable_nonserializable_and_content_free_in_repr() -> None:
    bundle, _backend = _validated(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
        ("Nova",),
    )
    result = _apply(bundle, _patch_tuple(bundle))

    with pytest.raises(FrozenInstanceError):
        setattr(result, "datums", ())
    with pytest.raises(TypeError, match="not serializable"):
        pickle.dumps(result)
    assert "Alice" not in repr(result)
    assert "Nova" not in repr(result)


def test_phase7_application_has_no_backend_effects_or_public_substitute_dependency() -> None:
    bundle, backend = _validated(
        ("Alice",),
        (("target-0",),),
        {"target-0": (_Proposal("Alice", "first_name", "person"),)},
        ("Nova",),
    )
    calls_before = tuple(getattr(backend, "calls"))

    result = _apply(bundle, _patch_tuple(bundle))

    assert _outputs(result) == {"target-0": ("Nova", True)}
    assert tuple(getattr(backend, "calls")) == calls_before
    assert not {"Substitute", "ReplacementWorkflow", "apply_replacements_to_spans"} & set(vars(_application_module()))
