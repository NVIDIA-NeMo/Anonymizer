# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource leases and immutable cleanup facts for graph effects."""

from __future__ import annotations

import asyncio
from dataclasses import InitVar, dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    OpaqueIdentity,
    PrivateValue,
    reject,
    require_instance,
    require_literal,
    require_owner,
)
from anonymizer.engine.graph_sdk._graph_values import DatumId
from anonymizer.engine.graph_sdk.observations import _Span

ResourceOwner: TypeAlias = Literal["caller", "sdk"]
SafeDetachment: TypeAlias = Literal["forbidden", "independent_after_dispatch"]
CleanupPurpose: TypeAlias = Literal["verification", "accounting", "transport_only"]
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


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ResourceLease(PrivateValue):
    _key: InitVar[object]
    resource: ResourceId
    owner: ResourceOwner
    safe_detachment: SafeDetachment
    handle: object

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _LEASE_KEY)
        # create() is the public verb: owner, detachment and handle are caller-supplied ingress.
        require_literal(self.owner, frozenset({"caller", "sdk"}))
        require_literal(self.safe_detachment, frozenset({"forbidden", "independent_after_dispatch"}))
        if self.handle is None:
            reject(EffectCode.INVALID_TYPE)

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


_ASSOCIATION_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CleanupAssociation(PrivateValue):
    _key: InitVar[object]
    resource: ResourceId
    targets: frozenset[DatumId]
    purpose: CleanupPurpose

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _ASSOCIATION_KEY)


def _cleanup_association(
    *, resource: ResourceId, targets: frozenset[DatumId], purpose: CleanupPurpose
) -> CleanupAssociation:
    """Capture admitted ownership before the owner attempts resource cleanup."""
    return CleanupAssociation(_key=_ASSOCIATION_KEY, resource=resource, targets=targets, purpose=purpose)


async def close_resource(lease: ResourceLease) -> CleanupFact:
    with _Span("cleanup") as span:
        result = await _close_resource(lease)
        _observe_cleanup(span, result)
        return result


def _observe_cleanup(span: _Span, result: CleanupFact) -> None:
    if result.disposition == "close_failed":
        span.outcome = "failure"
        span.reasons = frozenset({"cleanup_failed"})
    elif result.disposition == "close_unknown" or (result.owner == "sdk" and result.disposition == "left_open"):
        span.outcome = "unknown"
        span.reasons = frozenset({"cleanup_unknown"})


async def _close_resource(lease: ResourceLease) -> CleanupFact:
    """Close an SDK-owned lease once its owner has proved cleanup eligibility."""
    require_instance(lease, ResourceLease)
    if lease.owner == "caller":
        return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="left_open")
    handle = lease.handle
    # A Lost request is no longer locally in-flight in the request reducer,
    # but an ordinary worker may still be touching this shared handle.
    from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
    from anonymizer.engine.graph_sdk.blocking import _BlockingAdapter

    if isinstance(handle, (_BlockingAdapter, ForeignLoopTransport)) and handle.local_users:
        return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="left_open")
    if not isinstance(handle, CloseableResource):
        return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="close_unknown")
    try:
        await handle.close()
    except asyncio.CancelledError:
        if not isinstance(handle, ForeignLoopTransport):
            raise
        return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="close_unknown")
    except Exception:
        return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="close_failed")
    return CleanupFact(resource=lease.resource, owner=lease.owner, disposition="closed")


def _retain_resource(lease: ResourceLease) -> CleanupFact:
    """Record an entered cleanup boundary whose owner cannot yet close it."""
    with _Span("cleanup") as span:
        result = CleanupFact(resource=lease.resource, owner=lease.owner, disposition="left_open")
        _observe_cleanup(span, result)
        return result
