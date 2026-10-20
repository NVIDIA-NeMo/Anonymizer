# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate historical staged fixture declarations to the unified graph.

The independent literal models remain unchanged. Static fixture cases exercise
body compilation; activation fixtures attach their declared controls and exercise
control admission. Production callers use compile_graph once with all nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from anonymizer import graph as _workflow
from anonymizer.engine.graph_sdk._workflow_admission import _compile_body, _substitute_body
from anonymizer.engine.graph_sdk._workflow_admission import substitute as _admitted_substitute
from anonymizer.engine.graph_sdk._workflow_analysis import _bounded_analysis
from anonymizer.engine.graph_sdk._workflow_values import _analysis_limit, _reject
from anonymizer.graph import (
    DynamicLimits,
    Graph,
    KeyedJoinDecl,
    LoopDecl,
    MapDecl,
    NodeId,
    SubgraphNode,
    ValidationCode,
)

AdmittedWorkflow = Graph
AdmittedActivationWorkflow = Graph


def __getattr__(name: str) -> Any:
    return getattr(_workflow, name)


@dataclass(frozen=True, kw_only=True)
class DynamicScope:
    workflow: Graph
    maps: tuple[MapDecl, ...]
    joins: tuple[KeyedJoinDecl, ...]
    loops: tuple[LoopDecl, ...]


@_bounded_analysis(_analysis_limit)
def admit_static_workflow(*, choices: tuple[Any, ...], **declaration: Any) -> Graph:
    nodes = declaration.pop("nodes")
    return _compile_body(nodes=(*nodes, *choices), **declaration)


@_bounded_analysis(_analysis_limit)
def admit_activation_workflow(
    *, workflow: Graph, scopes: tuple[DynamicScope | Graph, ...], limits: DynamicLimits
) -> Graph:
    if not isinstance(workflow, Graph) or not isinstance(limits, DynamicLimits):
        _reject(ValidationCode.INVALID_TYPE)
    if not isinstance(scopes, tuple) or any(not isinstance(scope, (DynamicScope, Graph)) for scope in scopes):
        _reject(ValidationCode.INVALID_TYPE)
    by_body = {id(scope if isinstance(scope, Graph) else scope.workflow): scope for scope in scopes}
    if len(by_body) != len(scopes):
        _reject(ValidationCode.DUPLICATE)
    if set(by_body) != {id(body) for body in workflow.bodies()}:
        _reject(ValidationCode.MISSING)
    rewritten: dict[int, Graph] = {}
    pending = [(workflow, False)]
    while pending:
        body, visited = pending.pop()
        if id(body) in rewritten:
            continue
        children = [node.body for node in body.operations if isinstance(node, SubgraphNode)]
        if not visited:
            pending.append((body, True))
            pending.extend((child, False) for child in children)
            continue
        scope = by_body[id(body)]
        copied = _workflow.compile_graph(
            workflow=body.workflow,
            interface=body.interface,
            nodes=tuple(
                (
                    *(
                        SubgraphNode(id=node.id, operation=node.operation, body=rewritten[id(node.body)])
                        if isinstance(node, SubgraphNode)
                        else node
                        for node in body.operations
                    ),
                    *body.choices,
                    *scope.maps,
                    *scope.joins,
                    *scope.loops,
                )
            ),
            input_bindings=tuple(body.input_bindings),
            output_bindings=tuple(body.output_bindings),
            outcome_bindings=tuple(body.outcome_bindings),
            sequence=tuple(body.sequence),
            protection=tuple(body.protection_requirements),
            limits=body.limits,
            dynamic_limits=limits,
        )
        rewritten[id(body)] = copied
    return rewritten[id(workflow)]


def substitute_activation_workflow(*, workflow: Graph, target: NodeId, replacement: Graph) -> Graph:
    return _admitted_substitute(workflow=workflow, target=target, replacement=replacement)


def substitute(*, workflow: Graph, target: NodeId, replacement: Graph) -> Graph:
    return _substitute_body(workflow=workflow, target=target, replacement=replacement)


def replace_scope(scope: Graph | DynamicScope, **changes: Any) -> DynamicScope:
    """Translate edits to the former scope declaration, preserving every field."""
    declaration = (
        scope
        if isinstance(scope, DynamicScope)
        else DynamicScope(
            workflow=scope,
            maps=scope.maps,
            joins=scope.joins,
            loops=scope.loops,
        )
    )
    return replace(declaration, **changes)
