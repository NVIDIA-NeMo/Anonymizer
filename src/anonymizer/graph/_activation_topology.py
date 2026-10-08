# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Activation scope and parent lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Never

from anonymizer.graph._values import ActivationKey, ContractViolation, ValidationCode
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    AdmittedWorkflow,
    DynamicScope,
    KeyedJoinDecl,
    LoopDecl,
    MapDecl,
    Node,
    NodeId,
)

if TYPE_CHECKING:
    from anonymizer.graph._activation_values import ActivationSeed


def _nodes(workflow: AdmittedWorkflow) -> dict[NodeId, Node]:
    return {node.id: node for node in workflow.nodes}


def _scope_for(workflow: AdmittedActivationWorkflow, template: NodeId) -> DynamicScope:
    matches = [scope for scope in workflow.scopes if template in _nodes(scope.workflow)]
    if not matches:
        _reject(ValidationCode.MISSING)
    return matches[0]


def _node(workflow: AdmittedActivationWorkflow, template: NodeId) -> Node:
    return _nodes(_scope_for(workflow, template).workflow)[template]


def _map_for(scope: DynamicScope, template: NodeId) -> MapDecl | None:
    return next((item for item in scope.maps if item.expander == template), None)


def _loop_for(scope: DynamicScope, template: NodeId) -> LoopDecl | None:
    return next((item for item in scope.loops if item.starter == template), None)


def _join_for(scope: DynamicScope, source: NodeId) -> KeyedJoinDecl | None:
    return next((item for item in scope.joins if item.source == source), None)


def _depth(key: ActivationKey) -> int:
    seen: set[int] = set()
    depth = 0
    current: ActivationKey | None = key
    while current is not None:
        if id(current) in seen:
            _reject(ValidationCode.CYCLE)
        seen.add(id(current))
        depth += 1
        current = current.parent
    return depth


def _context(seed: ActivationSeed, scope: DynamicScope) -> ActivationKey | None:
    dynamic_member = any(seed.template == item.member for item in (*scope.maps, *scope.loops))
    return (
        seed.activation.parent.parent
        if dynamic_member and seed.activation.parent is not None
        else seed.activation.parent
    )


def _reject(code: ValidationCode) -> Never:
    raise ContractViolation(code) from None
