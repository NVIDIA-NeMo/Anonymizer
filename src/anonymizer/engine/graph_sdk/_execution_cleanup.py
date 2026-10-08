# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution resource acquisition and cleanup."""

from __future__ import annotations

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
)
from anonymizer.engine.graph_sdk._execution_values import AdmittedExecutionPlan, ImplementationHandle
from anonymizer.engine.graph_sdk.context import (
    ContextProvider,
    ContextResource,
)
from anonymizer.engine.graph_sdk.requests import (
    PhysicalRequestId,
    RequestState,
)
from anonymizer.engine.graph_sdk.resources import (
    CleanupAssociation,
    CleanupFact,
    ResourceId,
    ResourceLease,
    _cleanup_association,
    close_resource,
)


def _acquire_context_resources(
    resources: tuple[ContextResource, ...],
) -> tuple[dict[object, ResourceLease], frozenset[object]]:
    leases: dict[object, ResourceLease] = {}
    failed: set[object] = set()
    for resource in resources:
        if resource.lease is not None:
            leases[resource.source] = resource.lease
            continue
        if resource.factory is None:
            reject(EffectCode.MISSING)
        try:
            provider = resource.factory()
        except Exception:
            failed.add(resource.source)
            continue
        if not isinstance(provider, ContextProvider):
            failed.add(resource.source)
            continue
        leases[resource.source] = ResourceLease.create(
            owner="sdk",
            safe_detachment=resource.capability.safe_detachment,
            handle=provider,
        )
    return leases, frozenset(failed)


async def _cleanup_execution(
    admitted: AdmittedExecutionPlan,
    handles: tuple[ImplementationHandle, ...],
    context_leases: dict[object, ResourceLease],
    requests: RequestState,
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> tuple[tuple[CleanupFact, ...], tuple[CleanupAssociation, ...]]:
    leases = {item.resource.resource: item.resource for item in handles if item.resource is not None}
    leases.update({item.resource: item for item in context_leases.values()})
    local_resources = {
        request_resources[request] for request in requests.local_in_flight if request in request_resources
    }
    remote_resources = {
        request_resources[request] for request in requests.remote_outstanding if request in request_resources
    }
    all_targets = admitted.context.prepared.data.targets
    external_resources = {
        handle.resource.resource
        for policy in admitted.policies
        if policy.kind == "external"
        for implementation in policy.implementations
        for handle in handles
        if handle.resource is not None
        and handle.implementation == implementation.implementation
        and handle.operation == implementation.capability.operation
        and handle.configuration == implementation.configuration
    }
    context_resource_ids = {item.resource for item in context_leases.values()}
    context_targets = {
        lease.resource: frozenset(
            target.target
            for target in admitted.context.prepared.target_occurrences
            if any(item.source == source for item in admitted.context.adaptive_retrievals)
        )
        for source, lease in context_leases.items()
    }
    associations = tuple(
        _cleanup_association(
            resource=resource,
            targets=context_targets.get(resource, all_targets),
            purpose="accounting" if resource in external_resources | context_resource_ids else "verification",
        )
        for resource in leases
    )
    cleanup_values: list[CleanupFact] = []
    for lease in leases.values():
        if lease.owner == "sdk" and (
            lease.resource in local_resources
            or (lease.resource in remote_resources and lease.safe_detachment == "forbidden")
        ):
            cleanup_values.append(CleanupFact(resource=lease.resource, owner=lease.owner, disposition="left_open"))
        else:
            cleanup_values.append(await close_resource(lease))
    cleanup = tuple(cleanup_values)
    return cleanup, associations
