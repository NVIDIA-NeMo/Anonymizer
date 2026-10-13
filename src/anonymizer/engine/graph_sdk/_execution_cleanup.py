# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution resource acquisition and cleanup."""

from __future__ import annotations

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
)
from anonymizer.engine.graph_sdk._execution_values import (
    AdaptiveResourceAcquisition,
    AdmittedExecutionPlan,
    ImplementationHandle,
)
from anonymizer.engine.graph_sdk.blocking import _validate_provider_execution
from anonymizer.engine.graph_sdk.context import (
    ContextProvider,
    ContextResource,
    ContextSourceCapability,
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
    _retain_resource,
    close_resource,
)
from anonymizer.graph._values import InvocationId


def _acquire_context_resources(
    resources: tuple[ContextResource, ...],
    invocation: InvocationId,
) -> tuple[
    dict[ContextSourceCapability, ResourceLease],
    frozenset[ContextSourceCapability],
    tuple[AdaptiveResourceAcquisition, ...],
]:
    leases: dict[ContextSourceCapability, ResourceLease] = {}
    failed: set[ContextSourceCapability] = set()
    acquisitions = []
    for resource in resources:
        capability = resource.capability
        lease = resource.lease
        if lease is None:
            if resource.factory is None:
                reject(EffectCode.MISSING)
            try:
                provider = resource.factory()
                lease = ResourceLease.create(
                    owner="sdk",
                    safe_detachment=capability.safe_detachment,
                    handle=provider,
                )
                if not isinstance(provider, ContextProvider):
                    reject(EffectCode.INVALID_TYPE)
                _validate_provider_execution(provider, capability.execution)
            except Exception:
                failed.add(capability)
        if lease is not None:
            leases[capability] = lease
        acquisitions.append(
            AdaptiveResourceAcquisition(
                invocation=invocation,
                capability=capability,
                outcome="acquired" if lease is not None else "not_transferred",
                resource=lease.resource if lease is not None else None,
                owner=lease.owner if lease is not None else capability.resource_owner,
                safe_detachment=lease.safe_detachment if lease is not None else capability.safe_detachment,
            )
        )
    return leases, frozenset(failed), tuple(acquisitions)


async def _cleanup_execution(
    admitted: AdmittedExecutionPlan,
    handles: tuple[ImplementationHandle, ...],
    context_leases: dict[ContextSourceCapability, ResourceLease],
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
        if policy.kind in {"external", "delegated"}
        for implementation in policy.implementations
        for handle in handles
        if handle.resource is not None
        and handle.implementation == implementation.implementation
        and handle.operation == implementation.capability.operation
        and handle.configuration == implementation.configuration
    }
    context_resource_ids = {item.resource for item in context_leases.values()}
    associations = tuple(
        _cleanup_association(
            resource=resource,
            targets=all_targets,
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
            cleanup_values.append(_retain_resource(lease))
        else:
            cleanup_values.append(await close_resource(lease))
    cleanup = tuple(cleanup_values)
    return cleanup, associations
