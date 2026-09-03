# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mention-keyed local Redact patch construction and exact verification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from anonymizer.engine.execution.graph import _DatumId
from anonymizer.engine.execution.mention_admission import _MentionId, _MentionTargetToken
from anonymizer.engine.execution.role_policy import _ResolvedGraph


class _PrivatePatchValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Redact patch values are not serializable")


class _RedactProfileVersion(str, Enum):
    V1 = "phase6-redact/v1"


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _PatchToken(_PrivatePatchValue):
    """Invocation-private identity bound to one expected mention patch."""


@dataclass(frozen=True, slots=True, repr=False)
class _PatchManifestEntry(_PrivatePatchValue):
    mention_id: _MentionId


@dataclass(frozen=True, slots=True, repr=False)
class _PatchManifestProof(_PrivatePatchValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _PatchManifest(_PrivatePatchValue):
    resolved: _ResolvedGraph
    entries: tuple[_PatchManifestEntry, ...]
    profile_version: _RedactProfileVersion
    _proof: _PatchManifestProof | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _BoundPatchEntry(_PrivatePatchValue):
    token: _PatchToken
    mention_id: _MentionId


@dataclass(frozen=True, slots=True, repr=False)
class _BoundPatchProof(_PrivatePatchValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _BoundPatchManifest(_PrivatePatchValue):
    manifest: _PatchManifest
    entries: tuple[_BoundPatchEntry, ...]
    _proof: _BoundPatchProof | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _RedactPatch(_PrivatePatchValue):
    token: _PatchToken
    target: _MentionTargetToken
    start: int
    end: int
    replacement: str


@dataclass(frozen=True, slots=True, repr=False)
class _ReturnedRedact(_PrivatePatchValue):
    target: _MentionTargetToken
    output: str


@dataclass(frozen=True, slots=True, repr=False)
class _VerifiedDatum(_PrivatePatchValue):
    datum_id: _DatumId
    output: str
    applied: bool


@dataclass(frozen=True, slots=True, repr=False)
class _VerifiedGraph(_PrivatePatchValue):
    resolved: _ResolvedGraph
    datums: tuple[_VerifiedDatum, ...]
    patches: tuple[_RedactPatch, ...]
    profile_version: _RedactProfileVersion


class _PatchRejectionCode(str, Enum):
    FOREIGN_TOKEN = "foreign_token"
    STALE_TOKEN = "stale_token"
    INVALID_PATCH = "invalid_patch"
    RELEASE_PREDICATE_FAILED = "release_predicate_failed"


@dataclass(frozen=True, slots=True, repr=False)
class _PatchRejected(_PrivatePatchValue):
    code: _PatchRejectionCode
    owner: _MentionTargetToken | None = None


_PATCH_MANIFEST_SEAL = object()
_BOUND_PATCH_SEAL = object()
_REDACT_REPLACEMENT = "[REDACTED]"


def _build_patch_manifest(resolved: _ResolvedGraph) -> _PatchManifest | _PatchRejected:
    if not isinstance(resolved, _ResolvedGraph):
        return _PatchRejected(_PatchRejectionCode.STALE_TOKEN)
    detected_ids = tuple(mention.id for mention in resolved.clustered.detected.mentions)
    resolved_ids = tuple(item.mention.id for item in resolved.mentions)
    if len(set(detected_ids)) != len(detected_ids) or resolved_ids != detected_ids:
        return _PatchRejected(_PatchRejectionCode.INVALID_PATCH)
    entries = tuple(_PatchManifestEntry(mention_id) for mention_id in resolved_ids)
    snapshot = (resolved, tuple(entry.mention_id for entry in entries), _RedactProfileVersion.V1)
    return _PatchManifest(
        resolved,
        entries,
        _RedactProfileVersion.V1,
        _PatchManifestProof(_PATCH_MANIFEST_SEAL, snapshot),
    )


def _bind_patch_manifest(manifest: _PatchManifest) -> _BoundPatchManifest | _PatchRejected:
    if not _is_manifest(manifest):
        return _PatchRejected(_PatchRejectionCode.STALE_TOKEN)
    entries = tuple(_BoundPatchEntry(_PatchToken(), entry.mention_id) for entry in manifest.entries)
    snapshot = (manifest, tuple((entry.token, entry.mention_id) for entry in entries))
    return _BoundPatchManifest(manifest, entries, _BoundPatchProof(_BOUND_PATCH_SEAL, snapshot))


def _materialize_redact_patches(bound: _BoundPatchManifest) -> tuple[_RedactPatch, ...] | _PatchRejected:
    if not _is_bound(bound):
        return _PatchRejected(_PatchRejectionCode.STALE_TOKEN)
    resolved = bound.manifest.resolved
    mention_by_id = {item.mention.id: item.mention for item in resolved.mentions}
    target_by_datum = {target.datum_id: target.token for target in resolved.clustered.detected.targets}
    patches: list[_RedactPatch] = []
    for entry in bound.entries:
        mention = mention_by_id.get(entry.mention_id)
        if mention is None or mention.target_datum_id not in target_by_datum:
            return _PatchRejected(_PatchRejectionCode.STALE_TOKEN)
        if mention.source_slice in _REDACT_REPLACEMENT:
            return _PatchRejected(_PatchRejectionCode.INVALID_PATCH, target_by_datum[mention.target_datum_id])
        patches.append(
            _RedactPatch(
                entry.token,
                target_by_datum[mention.target_datum_id],
                mention.start,
                mention.end,
                _REDACT_REPLACEMENT,
            )
        )
    return tuple(patches)


def _apply_redact_patches(
    resolved: _ResolvedGraph,
    patches: tuple[_RedactPatch, ...],
) -> tuple[_ReturnedRedact, ...] | _PatchRejected:
    if not isinstance(resolved, _ResolvedGraph) or not isinstance(patches, tuple):
        return _PatchRejected(_PatchRejectionCode.INVALID_PATCH)
    by_target: dict[_MentionTargetToken, list[_RedactPatch]] = {
        target.token: [] for target in resolved.clustered.detected.targets
    }
    for patch in patches:
        if not isinstance(patch, _RedactPatch) or patch.target not in by_target:
            return _PatchRejected(_PatchRejectionCode.INVALID_PATCH)
        by_target[patch.target].append(patch)
    returned: list[_ReturnedRedact] = []
    for target in resolved.clustered.detected.targets:
        output = _apply_target_patches(target.text, by_target[target.token])
        if output is None:
            return _PatchRejected(_PatchRejectionCode.INVALID_PATCH, target.token)
        returned.append(_ReturnedRedact(target.token, output))
    return tuple(returned)


def _apply_target_patches(text: str, patches: list[_RedactPatch]) -> str | None:
    cursor = 0
    parts: list[str] = []
    for patch in sorted(patches, key=lambda item: (item.start, item.end)):
        if patch.start < cursor or patch.end > len(text):
            return None
        parts.extend((text[cursor : patch.start], patch.replacement))
        cursor = patch.end
    parts.append(text[cursor:])
    return "".join(parts)


def _verify_redact_patches(
    bound: _BoundPatchManifest,
    patches: tuple[_RedactPatch, ...],
    returned: tuple[_ReturnedRedact, ...],
) -> _VerifiedGraph | _PatchRejected:
    if not _is_bound(bound):
        return _PatchRejected(_PatchRejectionCode.STALE_TOKEN)
    validated = _validate_patches(bound, patches)
    if isinstance(validated, _PatchRejected):
        return validated
    expected = _reconstruct_outputs(bound, validated)
    if isinstance(expected, _PatchRejected):
        return expected
    returned_by_target = _validate_returned(bound, returned)
    if isinstance(returned_by_target, _PatchRejected):
        return returned_by_target
    targets = bound.manifest.resolved.clustered.detected.targets
    for target in targets:
        if returned_by_target[target.token] != expected[target.token]:
            return _PatchRejected(_PatchRejectionCode.RELEASE_PREDICATE_FAILED, target.token)
    mention_datums = {item.mention.target_datum_id for item in bound.manifest.resolved.mentions}
    datums = tuple(
        _VerifiedDatum(target.datum_id, expected[target.token], target.datum_id in mention_datums) for target in targets
    )
    return _VerifiedGraph(bound.manifest.resolved, datums, validated, bound.manifest.profile_version)


def _is_manifest(manifest: object) -> bool:
    if not isinstance(manifest, _PatchManifest) or manifest._proof is None:
        return False
    snapshot = (
        manifest.resolved,
        tuple(entry.mention_id for entry in manifest.entries),
        manifest.profile_version,
    )
    return manifest._proof.seal is _PATCH_MANIFEST_SEAL and manifest._proof.snapshot == snapshot


def _is_bound(bound: object) -> bool:
    if not isinstance(bound, _BoundPatchManifest) or bound._proof is None or not _is_manifest(bound.manifest):
        return False
    snapshot = (bound.manifest, tuple((entry.token, entry.mention_id) for entry in bound.entries))
    return (
        bound._proof.seal is _BOUND_PATCH_SEAL
        and bound._proof.snapshot == snapshot
        and len({entry.token for entry in bound.entries}) == len(bound.entries)
        and tuple(entry.mention_id for entry in bound.entries)
        == tuple(entry.mention_id for entry in bound.manifest.entries)
    )


def _validate_patches(
    bound: _BoundPatchManifest,
    patches: object,
) -> tuple[_RedactPatch, ...] | _PatchRejected:
    if not isinstance(patches, tuple) or not all(isinstance(patch, _RedactPatch) for patch in patches):
        return _PatchRejected(_PatchRejectionCode.INVALID_PATCH)
    expected_by_token = {entry.token: entry.mention_id for entry in bound.entries}
    observed_tokens = tuple(patch.token for patch in patches)
    if any(token not in expected_by_token for token in observed_tokens):
        return _PatchRejected(_PatchRejectionCode.FOREIGN_TOKEN)
    if len(set(observed_tokens)) != len(observed_tokens) or set(observed_tokens) != set(expected_by_token):
        return _PatchRejected(_PatchRejectionCode.INVALID_PATCH)
    mention_by_id = {item.mention.id: item.mention for item in bound.manifest.resolved.mentions}
    target_by_datum = {target.datum_id: target.token for target in bound.manifest.resolved.clustered.detected.targets}
    patch_by_token = {patch.token: patch for patch in patches}
    for entry in bound.entries:
        patch = patch_by_token[entry.token]
        mention = mention_by_id[entry.mention_id]
        if (
            patch.target is not target_by_datum[mention.target_datum_id]
            or type(patch.start) is not int
            or type(patch.end) is not int
            or patch.start != mention.start
            or patch.end != mention.end
            or patch.replacement != _REDACT_REPLACEMENT
            or mention.source_slice in patch.replacement
        ):
            return _PatchRejected(_PatchRejectionCode.INVALID_PATCH, target_by_datum[mention.target_datum_id])
    return tuple(patch_by_token[entry.token] for entry in bound.entries)


def _reconstruct_outputs(
    bound: _BoundPatchManifest,
    patches: tuple[_RedactPatch, ...],
) -> dict[_MentionTargetToken, str] | _PatchRejected:
    resolved = bound.manifest.resolved
    target_by_token = {target.token: target for target in resolved.clustered.detected.targets}
    by_target: dict[_MentionTargetToken, list[_RedactPatch]] = {token: [] for token in target_by_token}
    for patch in patches:
        if patch.target not in by_target:
            return _PatchRejected(_PatchRejectionCode.FOREIGN_TOKEN)
        by_target[patch.target].append(patch)
    outputs: dict[_MentionTargetToken, str] = {}
    for token, target in target_by_token.items():
        cursor = 0
        parts: list[str] = []
        for patch in sorted(by_target[token], key=lambda item: (item.start, item.end)):
            if patch.start < cursor or patch.end > len(target.text):
                return _PatchRejected(_PatchRejectionCode.INVALID_PATCH, token)
            parts.extend((target.text[cursor : patch.start], patch.replacement))
            cursor = patch.end
        parts.append(target.text[cursor:])
        outputs[token] = "".join(parts)
    return outputs


def _validate_returned(
    bound: _BoundPatchManifest,
    returned: object,
) -> dict[_MentionTargetToken, str] | _PatchRejected:
    if not isinstance(returned, tuple) or not all(isinstance(item, _ReturnedRedact) for item in returned):
        return _PatchRejected(_PatchRejectionCode.RELEASE_PREDICATE_FAILED)
    known = frozenset(target.token for target in bound.manifest.resolved.clustered.detected.targets)
    targets = tuple(item.target for item in returned)
    if any(target not in known for target in targets):
        return _PatchRejected(_PatchRejectionCode.FOREIGN_TOKEN)
    if len(set(targets)) != len(targets) or set(targets) != set(known):
        return _PatchRejected(_PatchRejectionCode.RELEASE_PREDICATE_FAILED)
    if not all(isinstance(item.output, str) for item in returned):
        return _PatchRejected(_PatchRejectionCode.RELEASE_PREDICATE_FAILED)
    return {item.target: item.output for item in returned}
