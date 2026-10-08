# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource leases and immutable cleanup facts for graph effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    OpaqueIdentity,
    PrivateValue,
    reject,
    require_instance,
    require_literal,
)

ResourceOwner: TypeAlias = Literal["caller", "sdk"]
SafeDetachment: TypeAlias = Literal["forbidden", "independent_after_dispatch"]
CleanupDisposition: TypeAlias = Literal["closed", "close_failed", "close_unknown", "left_open"]


class ResourceId(OpaqueIdentity):
    """Opaque identity for one process-local resource."""

    __slots__ = ()

    @classmethod
    def new(cls) -> ResourceId:
        return cls._new()  # type: ignore[return-value]


@runtime_checkable
class CloseableResource(Protocol):
    async def close(self) -> None: ...


_LEASE_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ResourceLease(PrivateValue):
    resource: ResourceId
    owner: ResourceOwner
    safe_detachment: SafeDetachment
    handle: object

    def __init__(
        self,
        *,
        _key: object,
        resource: ResourceId,
        owner: ResourceOwner,
        safe_detachment: SafeDetachment,
        handle: object,
    ) -> None:
        if _key is not _LEASE_KEY:
            raise TypeError("resource leases must be created by their owner boundary")
        require_instance(resource, ResourceId)
        require_literal(owner, frozenset({"caller", "sdk"}))
        require_literal(safe_detachment, frozenset({"forbidden", "independent_after_dispatch"}))
        if handle is None:
            reject(EffectCode.INVALID_TYPE)
        object.__setattr__(self, "resource", resource)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "safe_detachment", safe_detachment)
        object.__setattr__(self, "handle", handle)

    @classmethod
    def create(
        cls,
        *,
        owner: ResourceOwner,
        safe_detachment: SafeDetachment,
        handle: object,
    ) -> ResourceLease:
        """Create a lease after its domain owner validates the live handle."""
        return cls(
            _key=_LEASE_KEY,
            resource=ResourceId.new(),
            owner=owner,
            safe_detachment=safe_detachment,
            handle=handle,
        )

    def __hash__(self) -> int:
        return hash(self.resource)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ResourceLease) and self.resource == other.resource


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CleanupFact(PrivateValue):
    resource: ResourceId
    owner: ResourceOwner
    disposition: CleanupDisposition

    def __post_init__(self) -> None:
        require_instance(self.resource, ResourceId)
        require_literal(self.owner, frozenset({"caller", "sdk"}))
        require_literal(self.disposition, frozenset({"closed", "close_failed", "close_unknown", "left_open"}))


async def close_resource(lease: ResourceLease) -> CleanupFact:
    """Close an SDK-owned lease once its owner has proved cleanup eligibility."""
    require_instance(lease, ResourceLease)
    if lease.owner == "caller":
        return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="left_open")
    handle = lease.handle
    if not isinstance(handle, CloseableResource):
        return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="close_unknown")
    try:
        await handle.close()
    except Exception:
        return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="close_failed")
    return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="closed")
