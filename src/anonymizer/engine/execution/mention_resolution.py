# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit-evidence clustering for private target-anchored mentions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from anonymizer.engine.execution.mention_admission import (
    _AnchoredMention,
    _DetectedGraph,
    _MentionId,
    _MentionTargetToken,
)


class _PrivateResolutionValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private resolution values are not serializable")


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _ClusterId(_PrivateResolutionValue):
    """Executor-issued graph-scoped cluster identity."""


class _EvidenceVersion(str, Enum):
    V1 = "same-subject-evidence/v1"


class _EvidenceProvenance(str, Enum):
    RESOLVER = "resolver"


@dataclass(frozen=True, slots=True, repr=False)
class _ResolverScope(_PrivateResolutionValue):
    owner: _MentionTargetToken
    eligible_targets: tuple[_MentionTargetToken, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _SameSubjectEvidence(_PrivateResolutionValue):
    owner: _MentionTargetToken
    left: _MentionId
    right: _MentionId
    version: _EvidenceVersion
    provenance: _EvidenceProvenance = _EvidenceProvenance.RESOLVER


@dataclass(frozen=True, slots=True, repr=False)
class _DistinctSubjectEvidence(_PrivateResolutionValue):
    owner: _MentionTargetToken
    left: _MentionId
    right: _MentionId
    version: _EvidenceVersion
    provenance: _EvidenceProvenance = _EvidenceProvenance.RESOLVER


_SubjectEvidence: TypeAlias = _SameSubjectEvidence | _DistinctSubjectEvidence


@dataclass(frozen=True, slots=True, repr=False)
class _EntityCluster(_PrivateResolutionValue):
    id: _ClusterId
    ordered_mention_ids: tuple[_MentionId, ...]
    accepted_evidence: tuple[_SameSubjectEvidence, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _ClusteredGraph(_PrivateResolutionValue):
    detected: _DetectedGraph
    clusters: tuple[_EntityCluster, ...]
    accepted_evidence: tuple[_SubjectEvidence, ...]


class _ResolutionRejectionCode(str, Enum):
    FOREIGN_TOKEN = "foreign_token"
    STALE_TOKEN = "stale_token"
    INVALID_EVIDENCE = "invalid_evidence"
    EVIDENCE_CONTRADICTION = "evidence_contradiction"


@dataclass(frozen=True, slots=True, repr=False)
class _ResolutionRejected(_PrivateResolutionValue):
    code: _ResolutionRejectionCode
    owner: _MentionTargetToken | None = None


def _resolve_mentions(
    detected: _DetectedGraph,
    scopes: tuple[_ResolverScope, ...],
    evidence: tuple[_SubjectEvidence, ...],
) -> _ClusteredGraph | _ResolutionRejected:
    indexed = _index_detected_graph(detected)
    if indexed is None:
        return _ResolutionRejected(_ResolutionRejectionCode.STALE_TOKEN)
    mention_by_id, mention_owner, mention_position, target_position = indexed
    scope_by_owner = _validate_scopes(detected, scopes)
    if scope_by_owner is None:
        return _ResolutionRejected(_ResolutionRejectionCode.INVALID_EVIDENCE)
    normalized = _normalize_evidence(
        evidence,
        scope_by_owner,
        mention_by_id,
        mention_owner,
        mention_position,
        target_position,
    )
    if isinstance(normalized, _ResolutionRejected):
        return normalized
    return _cluster(detected, normalized, mention_position)


def _index_detected_graph(
    detected: object,
) -> (
    tuple[
        dict[_MentionId, _AnchoredMention],
        dict[_MentionId, _MentionTargetToken],
        dict[_MentionId, int],
        dict[_MentionTargetToken, int],
    ]
    | None
):
    if not isinstance(detected, _DetectedGraph):
        return None
    target_position = {target.token: index for index, target in enumerate(detected.targets)}
    if len(target_position) != len(detected.targets):
        return None
    target_by_datum = {target.datum_id: target.token for target in detected.targets}
    if len(target_by_datum) != len(detected.targets):
        return None
    mention_by_id: dict[_MentionId, _AnchoredMention] = {}
    mention_owner: dict[_MentionId, _MentionTargetToken] = {}
    mention_position: dict[_MentionId, int] = {}
    for position, mention in enumerate(detected.mentions):
        if not isinstance(mention, _AnchoredMention):
            return None
        owner = target_by_datum.get(mention.target_datum_id)
        if owner is None or mention.id in mention_by_id:
            return None
        mention_by_id[mention.id] = mention
        mention_owner[mention.id] = owner
        mention_position[mention.id] = position
    return mention_by_id, mention_owner, mention_position, target_position


def _validate_scopes(
    detected: _DetectedGraph,
    scopes: object,
) -> dict[_MentionTargetToken, frozenset[_MentionTargetToken]] | None:
    if not isinstance(scopes, tuple) or not all(isinstance(scope, _ResolverScope) for scope in scopes):
        return None
    known = frozenset(target.token for target in detected.targets)
    by_owner: dict[_MentionTargetToken, frozenset[_MentionTargetToken]] = {}
    for scope in scopes:
        eligible = scope.eligible_targets
        if (
            scope.owner not in known
            or scope.owner in by_owner
            or not isinstance(eligible, tuple)
            or not eligible
            or len(set(eligible)) != len(eligible)
            or scope.owner not in eligible
            or not set(eligible).issubset(known)
        ):
            return None
        by_owner[scope.owner] = frozenset(eligible)
    return by_owner if set(by_owner) == set(known) else None


def _normalize_evidence(
    evidence: object,
    scopes: dict[_MentionTargetToken, frozenset[_MentionTargetToken]],
    mention_by_id: dict[_MentionId, _AnchoredMention],
    mention_owner: dict[_MentionId, _MentionTargetToken],
    mention_position: dict[_MentionId, int],
    target_position: dict[_MentionTargetToken, int],
) -> tuple[_SubjectEvidence, ...] | _ResolutionRejected:
    if not isinstance(evidence, tuple) or not all(
        isinstance(item, (_SameSubjectEvidence, _DistinctSubjectEvidence)) for item in evidence
    ):
        return _ResolutionRejected(_ResolutionRejectionCode.INVALID_EVIDENCE)
    selected: dict[tuple[type[object], _MentionId, _MentionId], _SubjectEvidence] = {}
    owner_keys: set[tuple[_MentionTargetToken, type[object], _MentionId, _MentionId]] = set()
    pair_kinds: dict[tuple[_MentionId, _MentionId], type[object]] = {}
    for item in evidence:
        validated = _validate_evidence_item(item, scopes, mention_by_id, mention_owner, mention_position)
        if isinstance(validated, _ResolutionRejected):
            return validated
        left, right = validated
        owner_key = (item.owner, type(item), left, right)
        if owner_key in owner_keys:
            return _ResolutionRejected(_ResolutionRejectionCode.INVALID_EVIDENCE, item.owner)
        owner_keys.add(owner_key)
        pair = (left, right)
        if pair in pair_kinds and pair_kinds[pair] is not type(item):
            return _ResolutionRejected(_ResolutionRejectionCode.EVIDENCE_CONTRADICTION)
        pair_kinds[pair] = type(item)
        key = (type(item), left, right)
        prior = selected.get(key)
        if prior is None or target_position[item.owner] < target_position[prior.owner]:
            selected[key] = type(item)(item.owner, left, right, item.version, item.provenance)
    return tuple(
        selected[key]
        for key in sorted(
            selected,
            key=lambda value: (
                0 if value[0] is _SameSubjectEvidence else 1,
                mention_position[value[1]],
                mention_position[value[2]],
            ),
        )
    )


def _validate_evidence_item(
    item: _SubjectEvidence,
    scopes: dict[_MentionTargetToken, frozenset[_MentionTargetToken]],
    mention_by_id: dict[_MentionId, _AnchoredMention],
    mention_owner: dict[_MentionId, _MentionTargetToken],
    mention_position: dict[_MentionId, int],
) -> tuple[_MentionId, _MentionId] | _ResolutionRejected:
    if (
        item.owner not in scopes
        or item.version is not _EvidenceVersion.V1
        or item.provenance is not _EvidenceProvenance.RESOLVER
    ):
        return _ResolutionRejected(_ResolutionRejectionCode.INVALID_EVIDENCE)
    if item.left not in mention_by_id or item.right not in mention_by_id:
        return _ResolutionRejected(_ResolutionRejectionCode.FOREIGN_TOKEN)
    if item.left is item.right:
        return _ResolutionRejected(_ResolutionRejectionCode.INVALID_EVIDENCE, item.owner)
    left_owner = mention_owner[item.left]
    right_owner = mention_owner[item.right]
    if (
        left_owner not in scopes[item.owner]
        or right_owner not in scopes[item.owner]
        or item.owner not in {left_owner, right_owner}
    ):
        return _ResolutionRejected(_ResolutionRejectionCode.INVALID_EVIDENCE, item.owner)
    if mention_position[item.left] < mention_position[item.right]:
        return item.left, item.right
    return item.right, item.left


def _cluster(
    detected: _DetectedGraph,
    evidence: tuple[_SubjectEvidence, ...],
    mention_position: dict[_MentionId, int],
) -> _ClusteredGraph | _ResolutionRejected:
    parents = {mention.id: mention.id for mention in detected.mentions}
    for item in evidence:
        if isinstance(item, _SameSubjectEvidence):
            _union(parents, item.left, item.right, mention_position)
    if any(
        _find(parents, item.left) is _find(parents, item.right)
        for item in evidence
        if isinstance(item, _DistinctSubjectEvidence)
    ):
        return _ResolutionRejected(_ResolutionRejectionCode.EVIDENCE_CONTRADICTION)
    components: dict[_MentionId, list[_MentionId]] = {}
    for mention in detected.mentions:
        components.setdefault(_find(parents, mention.id), []).append(mention.id)
    same = tuple(item for item in evidence if isinstance(item, _SameSubjectEvidence))
    clusters = tuple(
        _EntityCluster(
            _ClusterId(),
            tuple(members),
            tuple(item for item in same if item.left in members and item.right in members),
        )
        for _root, members in sorted(components.items(), key=lambda item: mention_position[item[1][0]])
    )
    return _ClusteredGraph(detected, clusters, evidence)


def _find(parents: dict[_MentionId, _MentionId], mention_id: _MentionId) -> _MentionId:
    root = mention_id
    while parents[root] is not root:
        root = parents[root]
    while parents[mention_id] is not mention_id:
        parent = parents[mention_id]
        parents[mention_id] = root
        mention_id = parent
    return root


def _union(
    parents: dict[_MentionId, _MentionId],
    left: _MentionId,
    right: _MentionId,
    positions: dict[_MentionId, int],
) -> None:
    left_root = _find(parents, left)
    right_root = _find(parents, right)
    if left_root is right_root:
        return
    if positions[left_root] < positions[right_root]:
        parents[right_root] = left_root
    else:
        parents[left_root] = right_root
