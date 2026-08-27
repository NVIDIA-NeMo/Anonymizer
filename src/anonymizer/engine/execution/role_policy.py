# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned structural replacement-role results for the private graph profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TypeAlias

from anonymizer.engine.execution.mention_admission import _AnchoredMention, _MentionId
from anonymizer.engine.execution.mention_resolution import _ClusteredGraph, _ClusterId


class _PrivateRoleValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private role values are not serializable")


class _RolePolicyVersion(str, Enum):
    V1 = "phase6-role-result/v1"


@dataclass(frozen=True, slots=True, repr=False)
class _ReplacementRole(_PrivateRoleValue):
    value: str


class _UnsupportedRoleReason(str, Enum):
    UNSUPPORTED_ROLE = "unsupported_role"


@dataclass(frozen=True, slots=True, repr=False)
class _ClassifiedRole(_PrivateRoleValue):
    role: _ReplacementRole
    policy_version: _RolePolicyVersion


@dataclass(frozen=True, slots=True, repr=False)
class _UnsupportedRole(_PrivateRoleValue):
    reason: _UnsupportedRoleReason
    policy_version: _RolePolicyVersion


_RoleResult: TypeAlias = _ClassifiedRole | _UnsupportedRole


@dataclass(frozen=True, slots=True, repr=False)
class _RolePolicyProof(_PrivateRoleValue):
    seal: object = field(compare=False)
    snapshot: tuple[object, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _RolePolicy(_PrivateRoleValue):
    version: _RolePolicyVersion
    mappings: tuple[tuple[str, _ReplacementRole], ...]
    digest: str
    _proof: _RolePolicyProof | None = field(default=None, compare=False)


@dataclass(frozen=True, slots=True, repr=False)
class _ResolvedMention(_PrivateRoleValue):
    mention: _AnchoredMention
    cluster_id: _ClusterId
    role_result: _RoleResult


@dataclass(frozen=True, slots=True, repr=False)
class _ResolvedGraph(_PrivateRoleValue):
    clustered: _ClusteredGraph
    mentions: tuple[_ResolvedMention, ...]
    policy_version: _RolePolicyVersion
    policy_digest: str


class _RolePolicyRejectionCode(str, Enum):
    UNSUPPORTED_ROLE = "unsupported_role"


@dataclass(frozen=True, slots=True, repr=False)
class _RolePolicyRejected(_PrivateRoleValue):
    code: _RolePolicyRejectionCode
    mention_id: _MentionId | None = None


_ROLE_POLICY_SEAL = object()


def _compile_role_policy(
    version: _RolePolicyVersion,
    mappings: tuple[tuple[str, str], ...],
) -> _RolePolicy | _RolePolicyRejected:
    if version is not _RolePolicyVersion.V1 or not isinstance(mappings, tuple):
        return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE)
    validated: list[tuple[str, _ReplacementRole]] = []
    labels: set[str] = set()
    for mapping in mappings:
        if not isinstance(mapping, tuple) or len(mapping) != 2:
            return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE)
        label, role = mapping
        if not _valid_text(label) or not _valid_text(role) or label in labels:
            return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE)
        labels.add(label)
        validated.append((label, _ReplacementRole(role)))
    ordered = tuple(sorted(validated, key=lambda item: item[0]))
    digest = _policy_digest(version, ordered)
    snapshot = (version.value, tuple((label, role.value) for label, role in ordered), digest)
    return _RolePolicy(version, ordered, digest, _RolePolicyProof(_ROLE_POLICY_SEAL, snapshot))


def _classify_roles(
    clustered: _ClusteredGraph,
    policy: _RolePolicy,
) -> _ResolvedGraph | _RolePolicyRejected:
    if not _is_admitted_policy(policy) or not isinstance(clustered, _ClusteredGraph):
        return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE)
    cluster_by_mention: dict[_MentionId, _ClusterId] = {}
    for cluster in clustered.clusters:
        for mention_id in cluster.ordered_mention_ids:
            if mention_id in cluster_by_mention:
                return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE, mention_id)
            cluster_by_mention[mention_id] = cluster.id
    expected = {mention.id for mention in clustered.detected.mentions}
    if set(cluster_by_mention) != expected:
        return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE)
    mapping = dict(policy.mappings)
    resolved = tuple(
        _ResolvedMention(
            mention,
            cluster_by_mention[mention.id],
            _ClassifiedRole(mapping[mention.detector_label], policy.version)
            if mention.detector_label in mapping
            else _UnsupportedRole(_UnsupportedRoleReason.UNSUPPORTED_ROLE, policy.version),
        )
        for mention in clustered.detected.mentions
    )
    return _ResolvedGraph(clustered, resolved, policy.version, policy.digest)


def _is_admitted_policy(policy: object) -> bool:
    if not isinstance(policy, _RolePolicy) or policy._proof is None:
        return False
    snapshot = (
        policy.version.value,
        tuple((label, role.value) for label, role in policy.mappings),
        policy.digest,
    )
    return (
        policy._proof.seal is _ROLE_POLICY_SEAL
        and policy._proof.snapshot == snapshot
        and policy.digest == _policy_digest(policy.version, policy.mappings)
    )


def _policy_digest(
    version: _RolePolicyVersion,
    mappings: tuple[tuple[str, _ReplacementRole], ...],
) -> str:
    payload = {
        "mappings": [[label, role.value] for label, role in mappings],
        "version": version.value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_text(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
