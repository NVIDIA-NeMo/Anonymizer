# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic candidate validation for private Phase 7 Substitute scopes."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum

from anonymizer.engine.execution.mention_admission import _AnchoredMention, _MentionId, _MentionTarget
from anonymizer.engine.execution.phase6_runtime import (
    _PHASE6_HANDOFF_SEAL,
    _handoff_snapshot,
    _Phase6SubstituteHandoff,
)
from anonymizer.engine.execution.phase7_admission import (
    _is_admitted_scope_manifest,
    _ReplacementSlot,
    _ReplacementSlotId,
    _scope_manifest_snapshot,
    _ScopeManifest,
)
from anonymizer.engine.execution.phase7_contract import (
    _is_admitted_phase7_contract,
    _Phase7Role,
    _Phase7StableSubstituteContract,
)
from anonymizer.engine.execution.role_policy import (
    _RESOLVED_GRAPH_SEAL,
    _resolved_graph_snapshot,
    _ResolvedGraph,
)


class _PrivatePhase7ValidationValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 7 validation values are not serializable")


class _BundleRejectionCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    DUPLICATE_SLOT = "duplicate_slot"
    FOREIGN_SLOT = "foreign_slot"
    PARTIAL_BUNDLE = "partial_bundle"
    CANDIDATE_MATCHES_ORIGINAL = "candidate_matches_original"
    LIMIT_EXCEEDED = "limit_exceeded"
    UNSUPPORTED_ROLE = "unsupported_role"
    CANONICAL_COLLISION = "canonical_collision"
    UNSUPPORTED_CONSTRAINT = "unsupported_constraint"
    RELATION_FAILED = "relation_failed"


@dataclass(frozen=True, slots=True, repr=False)
class _CandidateAssignment(_PrivatePhase7ValidationValue):
    token: _ReplacementSlotId
    value: str


@dataclass(frozen=True, slots=True, repr=False)
class _ValidatedAssignment(_PrivatePhase7ValidationValue):
    token: _ReplacementSlotId
    value: str


@dataclass(frozen=True, slots=True, repr=False)
class _ValidatedBundleProof(_PrivatePhase7ValidationValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _ValidatedBundle(_PrivatePhase7ValidationValue):
    manifest: _ScopeManifest
    handoffs: tuple[_Phase6SubstituteHandoff, ...]
    assignments: tuple[_ValidatedAssignment, ...]
    _proof: _ValidatedBundleProof | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _BundleRejected(_PrivatePhase7ValidationValue):
    code: _BundleRejectionCode


@dataclass(frozen=True, slots=True, repr=False)
class _ScopeSourceIndex(_PrivatePhase7ValidationValue):
    mentions: tuple[tuple[_MentionId, _AnchoredMention], ...]
    targets: tuple[_MentionTarget, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _PreparedBundle(_PrivatePhase7ValidationValue):
    manifest: _ScopeManifest
    handoffs: tuple[_Phase6SubstituteHandoff, ...]
    assignments: tuple[_CandidateAssignment, ...]
    sources: _ScopeSourceIndex


_VALIDATED_BUNDLE_SEAL = object()
_USERNAME_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,62}[A-Za-z0-9])?")
_TELEPHONE_PATTERN = re.compile(r"[0-9 ()+.-]+")
_EMAIL_LOCAL_PATTERN = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+")
_EMAIL_LABEL_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?")


def _validate_scope_bundle(
    manifest: object,
    handoffs: object,
    assignments: object,
    contract: object,
) -> _ValidatedBundle | _BundleRejected:
    """Validate one complete scope atomically against frozen source evidence."""
    prepared = _prepare_bundle(manifest, handoffs, assignments, contract)
    if isinstance(prepared, _BundleRejected):
        return prepared
    if not isinstance(contract, _Phase7StableSubstituteContract):
        return _BundleRejected(_BundleRejectionCode.INVALID_INPUT)
    rejected = _validate_bundle_values(prepared, contract)
    if rejected is not None:
        return rejected

    assignment_by_token = {item.token: item.value for item in prepared.assignments}
    expected_tokens = tuple(slot.id for slot in prepared.manifest.slots)
    validated_assignments = tuple(_ValidatedAssignment(token, assignment_by_token[token]) for token in expected_tokens)
    values = (prepared.manifest, prepared.handoffs, validated_assignments)
    candidate = _ValidatedBundle(*values)
    snapshot = _validated_bundle_snapshot(candidate)
    if snapshot is None:
        return _BundleRejected(_BundleRejectionCode.INVALID_INPUT)
    return _ValidatedBundle(*values, _ValidatedBundleProof(_VALIDATED_BUNDLE_SEAL, snapshot))


def _prepare_bundle(
    manifest: object,
    handoffs: object,
    assignments: object,
    contract: object,
) -> _PreparedBundle | _BundleRejected:
    if (
        not isinstance(manifest, _ScopeManifest)
        or not _is_admitted_scope_manifest(manifest)
        or not isinstance(handoffs, tuple)
        or not isinstance(assignments, tuple)
        or not isinstance(contract, _Phase7StableSubstituteContract)
        or not _is_admitted_phase7_contract(contract)
    ):
        return _BundleRejected(_BundleRejectionCode.INVALID_INPUT)
    source_index = _index_scope_sources(manifest, handoffs)
    if source_index is None:
        return _BundleRejected(_BundleRejectionCode.INVALID_INPUT)
    validated_assignments = _validate_assignment_set(manifest, assignments)
    if isinstance(validated_assignments, _BundleRejected):
        return validated_assignments
    typed_handoffs = tuple(item for item in handoffs if isinstance(item, _Phase6SubstituteHandoff))
    return _PreparedBundle(manifest, typed_handoffs, validated_assignments, source_index)


def _validate_assignment_set(
    manifest: _ScopeManifest,
    assignments: tuple[object, ...],
) -> tuple[_CandidateAssignment, ...] | _BundleRejected:
    if not all(isinstance(item, _CandidateAssignment) for item in assignments):
        return _BundleRejected(_BundleRejectionCode.INVALID_INPUT)
    typed_assignments = tuple(item for item in assignments if isinstance(item, _CandidateAssignment))
    if any(not isinstance(item.token, _ReplacementSlotId) for item in typed_assignments):
        return _BundleRejected(_BundleRejectionCode.INVALID_INPUT)
    observed_tokens = tuple(item.token for item in typed_assignments)
    if len(set(observed_tokens)) != len(observed_tokens):
        return _BundleRejected(_BundleRejectionCode.DUPLICATE_SLOT)
    expected_tokens = tuple(slot.id for slot in manifest.slots)
    expected_set = set(expected_tokens)
    observed_set = set(observed_tokens)
    if observed_set - expected_set:
        return _BundleRejected(_BundleRejectionCode.FOREIGN_SLOT)
    if observed_set != expected_set:
        return _BundleRejected(_BundleRejectionCode.PARTIAL_BUNDLE)
    if any(type(item.value) is not str for item in typed_assignments):
        return _BundleRejected(_BundleRejectionCode.INVALID_INPUT)
    return typed_assignments


def _validate_bundle_values(
    prepared: _PreparedBundle,
    contract: _Phase7StableSubstituteContract,
) -> _BundleRejected | None:
    assignment_by_token = {item.token: item.value for item in prepared.assignments}
    mention_by_id = dict(prepared.sources.mentions)
    originals = tuple(mention.source_slice for _mention_id, mention in prepared.sources.mentions)
    original_skeletons = {_canonicalize_value(original) for original in originals}
    canonical_by_token: dict[_ReplacementSlotId, str] = {}
    byte_limits = dict(contract.byte_limits)
    bundle_bytes = 0
    roles = {role.name: role for role in contract.roles}
    for slot in prepared.manifest.slots:
        value = assignment_by_token[slot.id]
        validated = _validate_slot_value(slot, value, mention_by_id, original_skeletons, roles, byte_limits)
        if isinstance(validated, _BundleRejected):
            return validated
        skeleton, value_bytes = validated
        bundle_bytes += value_bytes
        if bundle_bytes > byte_limits["max_candidate_bundle_bytes_per_scope"]:
            return _BundleRejected(_BundleRejectionCode.LIMIT_EXCEEDED)
        canonical_by_token[slot.id] = skeleton

    return _validate_bundle_constraints(prepared.manifest, assignment_by_token, canonical_by_token)


def _validate_bundle_constraints(
    manifest: _ScopeManifest,
    assignment_by_token: dict[_ReplacementSlotId, str],
    canonical_by_token: dict[_ReplacementSlotId, str],
) -> _BundleRejected | None:
    for pair in manifest.required_pairs:
        if canonical_by_token[pair.left] == canonical_by_token[pair.right]:
            return _BundleRejected(_BundleRejectionCode.CANONICAL_COLLISION)
    for relation in manifest.relations:
        if relation.version != "email_from_name/v1":
            return _BundleRejected(_BundleRejectionCode.UNSUPPORTED_CONSTRAINT)
        if not _matches_email_from_name_relation(
            relation.upstream, relation.downstream, assignment_by_token, canonical_by_token
        ):
            return _BundleRejected(_BundleRejectionCode.RELATION_FAILED)
    return None


def _validate_slot_value(
    slot: _ReplacementSlot,
    value: str,
    mention_by_id: dict[_MentionId, _AnchoredMention],
    original_skeletons: set[str | None],
    roles: dict[str, _Phase7Role],
    byte_limits: dict[str, int],
) -> tuple[str, int] | _BundleRejected:
    skeleton = _canonicalize_value(value)
    if skeleton is None or skeleton in original_skeletons:
        return _BundleRejected(_BundleRejectionCode.CANDIDATE_MATCHES_ORIGINAL)
    try:
        value_bytes = len(value.encode("utf-8"))
    except UnicodeEncodeError:
        return _BundleRejected(_BundleRejectionCode.UNSUPPORTED_ROLE)
    if value_bytes > byte_limits["max_candidate_value_bytes"]:
        return _BundleRejected(_BundleRejectionCode.LIMIT_EXCEEDED)
    role = roles.get(slot.role)
    if role is None or role.format != slot.format or role.mask != slot.mask or not _matches_format(slot.format, value):
        return _BundleRejected(_BundleRejectionCode.UNSUPPORTED_ROLE)
    if any(
        mention_id not in mention_by_id or not _matches_mask(slot.mask, mention_by_id[mention_id].source_slice, value)
        for mention_id in slot.mention_ids
    ):
        return _BundleRejected(_BundleRejectionCode.RELATION_FAILED)
    return skeleton, value_bytes


def _canonicalize_value(value: object) -> str | None:
    if type(value) is not str:
        return None
    try:
        normalized = unicodedata.normalize("NFKC", value).strip().casefold()
    except (TypeError, ValueError):
        return None
    canonical = "".join(character for character in normalized if unicodedata.category(character)[:1] in {"L", "N"})
    return canonical or None


def _matches_format(format_name: object, value: object) -> bool:
    if type(format_name) is not str or type(value) is not str:
        return False
    try:
        if format_name == "unicode_person_name/v1":
            normalized = unicodedata.normalize("NFKC", value)
            return (
                1 <= len(normalized.encode("utf-8")) <= 128
                and any(unicodedata.category(character).startswith("L") for character in normalized)
                and all(
                    unicodedata.category(character)[:1] in {"L", "M"}
                    or unicodedata.category(character) == "Zs"
                    or character in "'.-"
                    for character in normalized
                )
            )
        if format_name == "username_ascii/v1":
            return value.isascii() and _USERNAME_PATTERN.fullmatch(value) is not None
        if format_name == "telephone_ascii/v1":
            return (
                _TELEPHONE_PATTERN.fullmatch(value) is not None
                and 7 <= sum(character.isascii() and character.isdigit() for character in value) <= 15
                and value.count("+") <= 1
                and ("+" not in value or value.startswith("+"))
            )
        if format_name == "email_addr_spec_ascii/v1":
            return _matches_email_format(value)
    except (TypeError, UnicodeEncodeError, ValueError):
        return False
    return False


def _matches_mask(mask_name: object, source: object, candidate: object) -> bool:
    if type(mask_name) is not str or type(source) is not str or type(candidate) is not str:
        return False
    if mask_name == "none/v1":
        return True
    if mask_name != "digit_literal/v1":
        return False
    try:
        normalized_source = unicodedata.normalize("NFKC", source)
        normalized_candidate = unicodedata.normalize("NFKC", candidate)
    except (TypeError, ValueError):
        return False
    return len(normalized_source) == len(normalized_candidate) and all(
        (candidate_character.isascii() and candidate_character.isdigit())
        if source_character.isascii() and source_character.isdigit()
        else source_character == candidate_character
        for source_character, candidate_character in zip(normalized_source, normalized_candidate, strict=True)
    )


def _is_validated_bundle(value: object) -> bool:
    return (
        isinstance(value, _ValidatedBundle)
        and value._proof is not None
        and value._proof.seal is _VALIDATED_BUNDLE_SEAL
        and value._proof.snapshot == _validated_bundle_snapshot(value)
        and _is_admitted_scope_manifest(value.manifest)
        and all(_is_sealed_handoff(handoff) for handoff in value.handoffs)
        and _index_scope_sources(value.manifest, value.handoffs) is not None
        and tuple(assignment.token for assignment in value.assignments)
        == tuple(slot.id for slot in value.manifest.slots)
        and all(type(assignment.value) is str for assignment in value.assignments)
    )


def _index_scope_sources(
    manifest: _ScopeManifest,
    handoffs: tuple[object, ...],
) -> _ScopeSourceIndex | None:
    if not all(_is_sealed_handoff(handoff) for handoff in handoffs):
        return None
    members = set(manifest.members)
    targets: list[_MentionTarget] = []
    mentions: list[_AnchoredMention] = []
    for handoff in handoffs:
        if not isinstance(handoff, _Phase6SubstituteHandoff):
            return None
        targets.extend(target for target in handoff.resolved.clustered.detected.targets if target.datum_id in members)
        mentions.extend(item.mention for item in handoff.resolved.mentions if item.mention.target_datum_id in members)
    if (
        len({target.datum_id for target in targets}) != len(targets)
        or set(target.datum_id for target in targets) != members
        or len({mention.id for mention in mentions}) != len(mentions)
    ):
        return None
    expected_mentions = tuple(mention_id for slot in manifest.slots for mention_id in slot.mention_ids)
    if len(set(expected_mentions)) != len(expected_mentions) or set(expected_mentions) != {
        mention.id for mention in mentions
    }:
        return None
    position = {mention_id: index for index, mention_id in enumerate(expected_mentions)}
    mentions.sort(key=lambda mention: position[mention.id])
    target_position = {member: index for index, member in enumerate(manifest.members)}
    targets.sort(key=lambda target: target_position[target.datum_id])
    return _ScopeSourceIndex(tuple((mention.id, mention) for mention in mentions), tuple(targets))


def _is_sealed_handoff(value: object) -> bool:
    return (
        isinstance(value, _Phase6SubstituteHandoff)
        and value._proof is not None
        and value._proof.seal is _PHASE6_HANDOFF_SEAL
        and value._proof.snapshot == _handoff_snapshot(value)
        and _is_sealed_resolved_graph(value.resolved)
    )


def _is_sealed_resolved_graph(value: object) -> bool:
    return (
        isinstance(value, _ResolvedGraph)
        and value._proof is not None
        and value._proof.seal is _RESOLVED_GRAPH_SEAL
        and value._proof.snapshot
        == _resolved_graph_snapshot(
            value.clustered,
            value.mentions,
            value.policy_version,
            value.policy_digest,
            value.source_policy_version,
        )
    )


def _matches_email_format(value: str) -> bool:
    if not value.isascii() or not 3 <= len(value.encode("utf-8")) <= 254 or value.count("@") != 1:
        return False
    local, domain = value.split("@")
    labels = domain.split(".")
    return (
        1 <= len(local) <= 64
        and not local.startswith(".")
        and not local.endswith(".")
        and ".." not in local
        and _EMAIL_LOCAL_PATTERN.fullmatch(local) is not None
        and len(labels) >= 2
        and all(1 <= len(label) <= 63 and _EMAIL_LABEL_PATTERN.fullmatch(label) is not None for label in labels)
        and 2 <= len(labels[-1]) <= 63
        and labels[-1].isalpha()
    )


def _matches_email_from_name_relation(
    upstream: tuple[_ReplacementSlotId, ...],
    downstream: _ReplacementSlotId,
    assignment_by_token: dict[_ReplacementSlotId, str],
    canonical_by_token: dict[_ReplacementSlotId, str],
) -> bool:
    if downstream not in assignment_by_token or any(token not in canonical_by_token for token in upstream):
        return False
    local = assignment_by_token[downstream].split("@", maxsplit=1)[0]
    local_skeleton = _canonicalize_value(local)
    return local_skeleton is not None and any(canonical_by_token[token] in local_skeleton for token in upstream)


def _validated_bundle_snapshot(bundle: _ValidatedBundle) -> tuple[object, ...] | None:
    try:
        return (
            _scope_manifest_snapshot(bundle.manifest),
            tuple(
                (
                    _handoff_snapshot(handoff),
                    _resolved_graph_snapshot(
                        handoff.resolved.clustered,
                        handoff.resolved.mentions,
                        handoff.resolved.policy_version,
                        handoff.resolved.policy_digest,
                        handoff.resolved.source_policy_version,
                    ),
                )
                for handoff in bundle.handoffs
            ),
            tuple((assignment.token, assignment.value) for assignment in bundle.assignments),
        )
    except (AttributeError, TypeError):
        return None
