# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind an existing capability without changing caller-owned execution details."""

from __future__ import annotations

from anonymizer.engine.graph_sdk.capabilities import ImplementationCapability, ImplementationSelection
from anonymizer.engine.graph_sdk.executor import (
    ExecutionImplementation,
    ImplementationHandle,
    LocalCallable,
    RequestTransport,
)
from anonymizer.engine.graph_sdk.requests import PhysicalRequestPolicy
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.workflow import NodeId


def _selection(node: NodeId, capability: ImplementationCapability) -> ImplementationSelection:
    return ImplementationSelection(
        node=node, implementation=capability.implementation, configuration=capability.configuration
    )


def _execution_implementation(
    capability: ImplementationCapability, *, request: PhysicalRequestPolicy | None
) -> ExecutionImplementation:
    return ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=request,
    )


def _implementation_handle(
    capability: ImplementationCapability,
    *,
    local: LocalCallable | None,
    transport: RequestTransport | None,
    resource: ResourceLease | None,
) -> ImplementationHandle:
    return ImplementationHandle(
        implementation=capability.implementation,
        operation=capability.operation,
        configuration=capability.configuration,
        local=local,
        transport=transport,
        resource=resource,
    )
