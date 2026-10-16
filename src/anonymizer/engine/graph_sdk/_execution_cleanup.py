# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution resource cleanup."""

from __future__ import annotations

from anonymizer.engine.graph_sdk._execution_values import (
    AdmittedExecutionPlan,
    ImplementationHandle,
)
from anonymizer.engine.graph_sdk.requests import (
    PhysicalRequestId,
    RequestState,
)
from anonymizer.engine.graph_sdk.resources import (
    CleanupAssociation,
    CleanupFact,
    ResourceId,
    _cleanup_association,
    _retain_resource,
    close_resource,
)


async def _cleanup_execution(
    admitted: AdmittedExecutionPlan,
    handles: tuple[ImplementationHandle, ...],
    requests: RequestState,
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> tuple[tuple[CleanupFact, ...], tuple[CleanupAssociation, ...]]:
    leases = {item.resource.resource: item.resource for item in handles if item.resource is not None}
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
    associations = tuple(
        _cleanup_association(
            resource=resource,
            targets=all_targets,
            purpose="accounting" if resource in external_resources else "verification",
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
