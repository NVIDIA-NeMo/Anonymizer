# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Anchored, non-cascading application for private Phase 7 Substitute scopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from anonymizer.engine.execution.graph import _DatumId
from anonymizer.engine.execution.mention_admission import _MentionId, _MentionTargetToken
from anonymizer.engine.execution.phase7_validation import (
    _index_scope_sources,
    _is_validated_bundle,
    _ValidatedBundle,
)


class _PrivatePhase7ApplicationValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 7 application values are not serializable")


class _ApplicationRejectionCode(str, Enum):
    INVALID_APPLICATION = "invalid_application"


@dataclass(frozen=True, slots=True, repr=False)
class _SubstitutePatch(_PrivatePhase7ApplicationValue):
    mention_id: _MentionId
    target: _MentionTargetToken
    start: int
    end: int
    source_slice: str
    replacement: str


@dataclass(frozen=True, slots=True, repr=False)
class _AppliedDatum(_PrivatePhase7ApplicationValue):
    datum_id: _DatumId
    output: str
    applied: bool


@dataclass(frozen=True, slots=True, repr=False)
class _AppliedScopeProof(_PrivatePhase7ApplicationValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _AppliedScope(_PrivatePhase7ApplicationValue):
    bundle: _ValidatedBundle
    datums: tuple[_AppliedDatum, ...]
    patches: tuple[_SubstitutePatch, ...]
    _proof: _AppliedScopeProof | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _ApplicationRejected(_PrivatePhase7ApplicationValue):
    code: _ApplicationRejectionCode


_APPLIED_SCOPE_SEAL = object()


def _materialize_substitute_patches(
    bundle: object,
) -> tuple[_SubstitutePatch, ...] | _ApplicationRejected:
    """Bind every admitted mention token to its authoritative source interval."""
    if not isinstance(bundle, _ValidatedBundle) or not _is_validated_bundle(bundle):
        return _rejected()
    source_index = _index_scope_sources(bundle.manifest, bundle.handoffs)
    if source_index is None:
        return _rejected()
    mention_by_id = dict(source_index.mentions)
    target_by_datum = {target.datum_id: target for target in source_index.targets}
    assignment_by_token = {assignment.token: assignment.value for assignment in bundle.assignments}
    patches: list[_SubstitutePatch] = []
    for slot in bundle.manifest.slots:
        replacement = assignment_by_token.get(slot.id)
        if replacement is None:
            return _rejected()
        for mention_id in slot.mention_ids:
            mention = mention_by_id.get(mention_id)
            if mention is None:
                return _rejected()
            target = target_by_datum.get(mention.target_datum_id)
            if target is None:
                return _rejected()
            patches.append(
                _SubstitutePatch(
                    mention.id,
                    target.token,
                    mention.start,
                    mention.end,
                    mention.source_slice,
                    replacement,
                )
            )
    return tuple(patches)


def _apply_substitute_patches(
    bundle: object,
    patches: object,
) -> _AppliedScope | _ApplicationRejected:
    """Validate and apply one complete patch set over immutable source text."""
    if not isinstance(bundle, _ValidatedBundle) or not _is_validated_bundle(bundle):
        return _rejected()
    expected = _materialize_substitute_patches(bundle)
    if isinstance(expected, _ApplicationRejected):
        return expected
    if not _patches_are_exact(expected, patches):
        return _rejected()
    datums = _reconstruct_scope(bundle, expected)
    if isinstance(datums, _ApplicationRejected):
        return datums

    values = (bundle, datums, expected)
    candidate = _AppliedScope(*values)
    snapshot = _applied_scope_snapshot(candidate)
    if snapshot is None:
        return _rejected()
    return _AppliedScope(*values, _AppliedScopeProof(_APPLIED_SCOPE_SEAL, snapshot))


def _patches_are_exact(expected: tuple[_SubstitutePatch, ...], patches: object) -> bool:
    if not isinstance(patches, tuple) or not all(isinstance(patch, _SubstitutePatch) for patch in patches):
        return False
    typed_patches = tuple(patch for patch in patches if isinstance(patch, _SubstitutePatch))
    expected_by_mention = {patch.mention_id: patch for patch in expected}
    observed_mentions = tuple(patch.mention_id for patch in typed_patches)
    if len(set(observed_mentions)) != len(observed_mentions) or set(observed_mentions) != set(expected_by_mention):
        return False
    observed_by_mention = {patch.mention_id: patch for patch in typed_patches}
    return all(
        _is_exact_patch(observed_by_mention[expected_patch.mention_id], expected_patch) for expected_patch in expected
    )


def _reconstruct_scope(
    bundle: _ValidatedBundle,
    patches: tuple[_SubstitutePatch, ...],
) -> tuple[_AppliedDatum, ...] | _ApplicationRejected:
    source_index = _index_scope_sources(bundle.manifest, bundle.handoffs)
    if source_index is None:
        return _rejected()
    patches_by_target: dict[_MentionTargetToken, list[_SubstitutePatch]] = {
        target.token: [] for target in source_index.targets
    }
    for patch in patches:
        target_patches = patches_by_target.get(patch.target)
        if target_patches is None:
            return _rejected()
        target_patches.append(patch)

    datums: list[_AppliedDatum] = []
    for target in source_index.targets:
        target_patches = tuple(patches_by_target[target.token])
        output = _reconstruct_source(target.text, target_patches)
        if output is None:
            return _rejected()
        datums.append(_AppliedDatum(target.datum_id, output, bool(target_patches)))
    return tuple(datums)


def _is_applied_scope(value: object) -> bool:
    if (
        not isinstance(value, _AppliedScope)
        or value._proof is None
        or value._proof.seal is not _APPLIED_SCOPE_SEAL
        or not _is_validated_bundle(value.bundle)
        or value._proof.snapshot != _applied_scope_snapshot(value)
        or tuple(datum.datum_id for datum in value.datums) != value.bundle.manifest.members
    ):
        return False
    expected = _materialize_substitute_patches(value.bundle)
    return isinstance(expected, tuple) and value.patches == expected


def _is_exact_patch(observed: _SubstitutePatch, expected: _SubstitutePatch) -> bool:
    return (
        observed.mention_id is expected.mention_id
        and observed.target is expected.target
        and type(observed.start) is int
        and observed.start == expected.start
        and type(observed.end) is int
        and observed.end == expected.end
        and type(observed.source_slice) is str
        and observed.source_slice == expected.source_slice
        and type(observed.replacement) is str
        and observed.replacement == expected.replacement
    )


def _reconstruct_source(text: object, patches: object) -> str | None:
    if (
        type(text) is not str
        or not isinstance(patches, tuple)
        or not all(isinstance(patch, _SubstitutePatch) for patch in patches)
    ):
        return None
    cursor = 0
    parts: list[str] = []
    for patch in sorted(patches, key=lambda item: (item.start, item.end)):
        if (
            type(patch.start) is not int
            or type(patch.end) is not int
            or patch.start < cursor
            or patch.start < 0
            or patch.end <= patch.start
            or patch.end > len(text)
            or type(patch.source_slice) is not str
            or text[patch.start : patch.end] != patch.source_slice
            or type(patch.replacement) is not str
        ):
            return None
        parts.extend((text[cursor : patch.start], patch.replacement))
        cursor = patch.end
    parts.append(text[cursor:])
    return "".join(parts)


def _applied_scope_snapshot(value: _AppliedScope) -> tuple[object, ...] | None:
    try:
        return (
            value.bundle,
            tuple((datum.datum_id, datum.output, datum.applied) for datum in value.datums),
            tuple(
                (
                    patch.mention_id,
                    patch.target,
                    patch.start,
                    patch.end,
                    patch.source_slice,
                    patch.replacement,
                )
                for patch in value.patches
            ),
        )
    except (AttributeError, TypeError):
        return None


def _rejected() -> _ApplicationRejected:
    return _ApplicationRejected(_ApplicationRejectionCode.INVALID_APPLICATION)
