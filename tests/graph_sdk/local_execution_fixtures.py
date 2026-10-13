# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local implementation bindings shared by map and loop execution fixtures."""

from __future__ import annotations

from collections.abc import Mapping

from anonymizer.engine.graph_sdk.capabilities import ImplementationCapability, ImplementationSelection
from anonymizer.engine.graph_sdk.executor import (
    ExecutionImplementation,
    ImplementationHandle,
    LocalCallable,
    OperationExecutionPolicy,
)
from anonymizer.graph.workflow import NodeId, WorkflowLimits
from tests.graph_sdk.test_dynamic_executor import _rows


def _selections(by_node: Mapping[NodeId, ImplementationCapability]) -> tuple[ImplementationSelection, ...]:
    return tuple(
        ImplementationSelection(
            node=node, implementation=capability.implementation, configuration=capability.configuration
        )
        for node, capability in by_node.items()
    )


def _local_policies(by_node: Mapping[NodeId, ImplementationCapability]) -> tuple[OperationExecutionPolicy, ...]:
    return tuple(
        OperationExecutionPolicy(
            node=node,
            kind="local",
            request=None,
            safe_detachment="forbidden",
            implementations=(
                ExecutionImplementation(
                    implementation=capability.implementation,
                    configuration=capability.configuration,
                    capability=capability,
                    request=None,
                ),
            ),
            result_outcomes=frozenset(outcome.name for outcome in capability.operation.outcomes),
            runtime_outcomes=_rows(frozenset(outcome.name for outcome in capability.operation.outcomes)),
        )
        for node, capability in by_node.items()
    )


def _local_handle(capability: ImplementationCapability, callback: LocalCallable) -> ImplementationHandle:
    return ImplementationHandle(
        implementation=capability.implementation,
        operation=capability.operation,
        configuration=capability.configuration,
        local=callback,
        transport=None,
        resource=None,
    )


def _no_choice_limits(*, nodes: int, bindings: int, sequence_edges: int, subgraph_depth: int = 1) -> WorkflowLimits:
    return WorkflowLimits(
        max_nodes=nodes,
        max_bindings=bindings,
        max_sequence_edges=sequence_edges,
        max_choices=0,
        max_branch_members=0,
        max_subgraph_depth=subgraph_depth,
        max_choice_states=1,
    )
