# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed private contract for bounded target and context execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class _PrivateContextContractValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private context execution values are not serializable")


class _ContextProfile(str, Enum):
    TARGET_CONTEXT_V1 = "target-context-v1"


class _ContextSchemaVersion(str, Enum):
    V1 = "context-workframe-v1"


class _ContextOrdering(str, Enum):
    DECLARED = "declared"


class _RetentionPosture(str, Enum):
    DISABLED = "retention_disabled"
    ENABLED = "retention_enabled"
    UNKNOWN = "unknown"


class _BackendArtifactClass(str, Enum):
    CONTEXT_REQUEST = "context_request"


@dataclass(frozen=True, slots=True, repr=False)
class _ContextLimits(_PrivateContextContractValue):
    max_context_members_per_target: int
    max_context_bytes_per_target: int
    max_total_context_references: int
    max_expanded_frame_bytes: int


@dataclass(frozen=True, slots=True, repr=False)
class _ContextExecutionContract(_PrivateContextContractValue):
    profile: _ContextProfile
    schema_version: _ContextSchemaVersion
    limits: _ContextLimits
    allow_target_as_context: bool
    ordering: _ContextOrdering
    required_artifacts: tuple[_BackendArtifactClass, ...]
    retention: _RetentionPosture = _RetentionPosture.DISABLED


@dataclass(frozen=True, slots=True, repr=False)
class _ContextBackendCapability(_PrivateContextContractValue):
    profile: _ContextProfile
    schema_version: _ContextSchemaVersion
    limits: _ContextLimits
    allow_target_as_context: bool
    ordering: _ContextOrdering
    artifact_classes: tuple[_BackendArtifactClass, ...]
    retention: _RetentionPosture


def _snapshot_context_capability(backend: object) -> _ContextBackendCapability | None:
    """Take one fail-closed typed capability snapshot from a private backend."""
    try:
        capability_getter = getattr(backend, "context_capability", None)
        capability = capability_getter() if callable(capability_getter) else None
    except Exception:
        return None
    return capability if isinstance(capability, _ContextBackendCapability) else None


def _capability_satisfies(
    capability: object,
    contract: object,
) -> bool:
    """Return whether one immutable backend snapshot satisfies the frozen contract."""
    if not isinstance(capability, _ContextBackendCapability) or not isinstance(contract, _ContextExecutionContract):
        return False
    try:
        actual = capability.limits
        required = contract.limits
        return (
            _valid_context_limits(actual)
            and _valid_context_limits(required)
            and capability.profile is contract.profile
            and capability.schema_version is contract.schema_version
            and capability.ordering is contract.ordering
            and capability.retention is contract.retention is _RetentionPosture.DISABLED
            and (capability.allow_target_as_context or not contract.allow_target_as_context)
            and set(contract.required_artifacts).issubset(capability.artifact_classes)
            and actual.max_context_members_per_target >= required.max_context_members_per_target
            and actual.max_context_bytes_per_target >= required.max_context_bytes_per_target
            and actual.max_total_context_references >= required.max_total_context_references
            and actual.max_expanded_frame_bytes >= required.max_expanded_frame_bytes
        )
    except (AttributeError, TypeError):
        return False


def _valid_context_limits(limits: object) -> bool:
    if not isinstance(limits, _ContextLimits):
        return False
    values = (
        limits.max_context_members_per_target,
        limits.max_context_bytes_per_target,
        limits.max_total_context_references,
        limits.max_expanded_frame_bytes,
    )
    return all(type(value) is int and value >= 0 for value in values)
