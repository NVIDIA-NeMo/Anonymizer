# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned structural replacement-role results for the private graph profile."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from importlib.resources import files
from typing import TypeAlias, cast

from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS
from anonymizer.engine.execution.mention_admission import _AnchoredMention, _MentionId
from anonymizer.engine.execution.mention_resolution import _ClusteredGraph, _ClusterId
from anonymizer.engine.execution.phase7_contract import (
    _canonical_digest,
    _load_phase7_contract,
    _Phase7ContractRejected,
    _Phase7StableSubstituteContract,
)


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
    policy_version: str = ""
    dispositions: tuple[tuple[str, str | None], ...] = ()
    _proof: _RolePolicyProof | None = field(default=None, compare=False)

    @property
    def result_version(self) -> _RolePolicyVersion:
        return self.version


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
    source_policy_version: str
    _proof: _RolePolicyProof | None = field(default=None, compare=False)


class _RolePolicyRejectionCode(str, Enum):
    UNSUPPORTED_ROLE = "unsupported_role"


@dataclass(frozen=True, slots=True, repr=False)
class _RolePolicyRejected(_PrivateRoleValue):
    code: _RolePolicyRejectionCode
    mention_id: _MentionId | None = None


_ROLE_POLICY_SEAL = object()
_RESOLVED_GRAPH_SEAL = object()
_REDACT_ROLE_POLICY_RESOURCE = "phase6_redact_role_policy.json"
_SUBSTITUTE_ROLE_POLICY_RESOURCE = "phase6_substitute_role_policy.json"


def _load_redact_role_policy() -> _RolePolicy | _RolePolicyRejected:
    try:
        payload = json.loads(
            files("anonymizer.engine.execution").joinpath(_REDACT_ROLE_POLICY_RESOURCE).read_text(encoding="utf-8")
        )
        if type(payload) is not dict or set(payload) != {"digest", "mappings", "version"}:
            raise TypeError
        digest = payload["digest"]
        mappings = payload["mappings"]
        version = payload["version"]
        if type(digest) is not str or type(mappings) is not list or type(version) is not str:
            raise TypeError
        if mappings:
            raise ValueError
        parsed_mappings: list[tuple[str, str]] = []
        for mapping in mappings:
            if type(mapping) is not list or len(mapping) != 2 or any(type(value) is not str for value in mapping):
                raise TypeError
            parsed_mappings.append((mapping[0], mapping[1]))
        policy = _compile_role_policy(_RolePolicyVersion(version), tuple(parsed_mappings))
        if isinstance(policy, _RolePolicyRejected) or digest != policy.digest:
            raise ValueError
        return policy
    except (OSError, TypeError, ValueError):
        return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE)


def _load_substitute_role_policy() -> _RolePolicy | _RolePolicyRejected:
    """Load only the owner-frozen P0 Substitute disposition resource."""
    try:
        contract = _load_phase7_contract()
        if isinstance(contract, _Phase7ContractRejected):
            raise ValueError
        text = (
            files("anonymizer.engine.execution").joinpath(_SUBSTITUTE_ROLE_POLICY_RESOURCE).read_text(encoding="utf-8")
        )
        payload = json.loads(text, object_pairs_hook=_object_without_duplicates)
        return _compile_substitute_policy(payload, contract)
    except (KeyError, OSError, TypeError, UnicodeEncodeError, ValueError):
        return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE)


def _compile_substitute_policy(
    payload: object,
    contract: _Phase7StableSubstituteContract,
) -> _RolePolicy:
    if type(payload) is not dict or set(payload) != {"dispositions", "result_version", "version"}:
        raise TypeError
    policy_payload = cast(dict[str, object], payload)
    result_version = policy_payload["result_version"]
    policy_version = policy_payload["version"]
    dispositions = _compile_substitute_dispositions(policy_payload["dispositions"], contract)
    if (
        type(result_version) is not str
        or type(policy_version) is not str
        or result_version != contract.phase6_result_version
        or policy_version != contract.phase6_policy_version
        or _canonical_digest(policy_payload) != contract.phase6_policy_digest
    ):
        raise ValueError
    mappings = tuple((label, role) for label, role in dispositions if role is not None)
    policy = _compile_role_policy(
        _RolePolicyVersion(result_version),
        mappings,
        policy_version=policy_version,
        dispositions=dispositions,
    )
    if isinstance(policy, _RolePolicyRejected) or policy.digest != contract.phase6_policy_digest:
        raise ValueError
    return policy


def _compile_substitute_dispositions(
    payload: object,
    contract: _Phase7StableSubstituteContract,
) -> tuple[tuple[str, str | None], ...]:
    if type(payload) is not dict:
        raise TypeError
    dispositions = cast(dict[str, object], payload)
    if set(dispositions) != set(DEFAULT_ENTITY_LABELS) or len(dispositions) != len(DEFAULT_ENTITY_LABELS):
        raise ValueError
    roles = {role.name for role in contract.roles}
    result: list[tuple[str, str | None]] = []
    for label in sorted(dispositions):
        role = dispositions[label]
        if role is not None and (type(role) is not str or role not in roles):
            raise ValueError
        result.append((label, role))
    return tuple(result)


def _compile_role_policy(
    version: _RolePolicyVersion,
    mappings: tuple[tuple[str, str], ...],
    *,
    policy_version: str | None = None,
    dispositions: tuple[tuple[str, str | None], ...] = (),
) -> _RolePolicy | _RolePolicyRejected:
    selected_version = (
        version.value if policy_version is None and isinstance(version, _RolePolicyVersion) else policy_version
    )
    if (
        version is not _RolePolicyVersion.V1
        or not isinstance(mappings, tuple)
        or not _valid_text(selected_version)
        or not isinstance(dispositions, tuple)
    ):
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
    if dispositions and not _valid_dispositions(dispositions, ordered):
        return _RolePolicyRejected(_RolePolicyRejectionCode.UNSUPPORTED_ROLE)
    canonical_dispositions = tuple(sorted(dispositions))
    digest = _policy_digest(version, ordered, selected_version, canonical_dispositions)
    values = (version, ordered, digest, selected_version, canonical_dispositions)
    snapshot = _role_policy_snapshot(*values)
    return _RolePolicy(*values, _RolePolicyProof(_ROLE_POLICY_SEAL, snapshot))


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
    values = (clustered, resolved, policy.version, policy.digest, policy.policy_version)
    snapshot = _resolved_graph_snapshot(*values)
    return _ResolvedGraph(*values, _RolePolicyProof(_RESOLVED_GRAPH_SEAL, snapshot))


def _is_admitted_policy(policy: object) -> bool:
    if not isinstance(policy, _RolePolicy) or policy._proof is None:
        return False
    snapshot = _role_policy_snapshot(
        policy.version,
        policy.mappings,
        policy.digest,
        policy.policy_version,
        policy.dispositions,
    )
    return (
        policy._proof.seal is _ROLE_POLICY_SEAL
        and policy._proof.snapshot == snapshot
        and policy.digest == _policy_digest(policy.version, policy.mappings, policy.policy_version, policy.dispositions)
    )


def _is_admitted_resolved_graph(value: object, policy: _RolePolicy) -> bool:
    if not isinstance(value, _ResolvedGraph) or value._proof is None or not _is_admitted_policy(policy):
        return False
    if (
        value.policy_version is not policy.version
        or value.source_policy_version != policy.policy_version
        or value.policy_digest != policy.digest
        or value._proof.seal is not _RESOLVED_GRAPH_SEAL
        or value._proof.snapshot
        != _resolved_graph_snapshot(
            value.clustered,
            value.mentions,
            value.policy_version,
            value.policy_digest,
            value.source_policy_version,
        )
    ):
        return False
    expected_mentions = value.clustered.detected.mentions
    if len(value.mentions) != len(expected_mentions) or any(
        result.mention is not mention for result, mention in zip(value.mentions, expected_mentions, strict=True)
    ):
        return False
    mapping = dict(policy.mappings)
    cluster_ids = {
        mention_id: cluster.id for cluster in value.clustered.clusters for mention_id in cluster.ordered_mention_ids
    }
    for result in value.mentions:
        if result.cluster_id is not cluster_ids.get(result.mention.id):
            return False
        expected_role = mapping.get(result.mention.detector_label)
        if expected_role is None:
            if not isinstance(result.role_result, _UnsupportedRole):
                return False
        elif not isinstance(result.role_result, _ClassifiedRole) or result.role_result.role != expected_role:
            return False
        if result.role_result.policy_version is not policy.version:
            return False
    return True


def _policy_digest(
    version: _RolePolicyVersion,
    mappings: tuple[tuple[str, _ReplacementRole], ...],
    policy_version: str,
    dispositions: tuple[tuple[str, str | None], ...],
) -> str:
    if dispositions:
        payload = {
            "dispositions": dict(dispositions),
            "result_version": version.value,
            "version": policy_version,
        }
    else:
        payload = {
            "mappings": [[label, role.value] for label, role in mappings],
            "version": version.value,
        }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _role_policy_snapshot(
    version: _RolePolicyVersion,
    mappings: tuple[tuple[str, _ReplacementRole], ...],
    digest: str,
    policy_version: str,
    dispositions: tuple[tuple[str, str | None], ...],
) -> tuple[object, ...]:
    return (
        version.value,
        tuple((label, role.value) for label, role in mappings),
        digest,
        policy_version,
        dispositions,
    )


def _resolved_graph_snapshot(
    clustered: _ClusteredGraph,
    mentions: tuple[_ResolvedMention, ...],
    result_version: _RolePolicyVersion,
    policy_digest: str,
    policy_version: str,
) -> tuple[object, ...]:
    return (
        clustered,
        tuple(
            (
                result.mention,
                result.cluster_id,
                type(result.role_result),
                result.role_result.role.value if isinstance(result.role_result, _ClassifiedRole) else None,
                result.role_result.policy_version.value,
            )
            for result in mentions
        ),
        result_version.value,
        policy_digest,
        policy_version,
    )


def _valid_dispositions(
    dispositions: tuple[tuple[str, str | None], ...],
    mappings: tuple[tuple[str, _ReplacementRole], ...],
) -> bool:
    if any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not _valid_text(item[0])
        or (item[1] is not None and not _valid_text(item[1]))
        for item in dispositions
    ):
        return False
    labels = tuple(label for label, _role in dispositions)
    return len(set(labels)) == len(labels) and {(label, role) for label, role in dispositions if role is not None} == {
        (label, role.value) for label, role in mappings
    }


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _valid_text(value: object) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True
