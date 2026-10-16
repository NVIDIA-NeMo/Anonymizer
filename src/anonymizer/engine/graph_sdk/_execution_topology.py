# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow and activation lookup for execution."""

from __future__ import annotations

from collections.abc import Mapping

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
)
from anonymizer.engine.graph_sdk._execution_values import (
    AdmittedExecutionPlan,
    ArtifactProvenanceFact,
    ArtifactRole,
    ExecutionPortFact,
    OperationOutputKey,
    ProvenanceKey,
)
from anonymizer.graph._values import (
    ActivationKey,
    DatumId,
)
from anonymizer.graph.activation import (
    ActivationEntry,
    ActivationState,
)
from anonymizer.graph.workflow import (
    ContextInputRef,
    Graph,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    Operation,
    OperationSpec,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
)


def _lexical_path(
    activation: ActivationKey,
    entries: Mapping[ActivationKey, ActivationEntry],
    nodes: Mapping[NodeId, Operation],
) -> tuple[NodeId, ...] | None:
    """Resolve lexical containers without mistaking map/loop parents for them."""
    path = []
    child = entries.get(activation)
    if child is None:
        return None
    parent = activation.parent
    while parent is not None:
        entry = entries.get(parent)
        if entry is None:
            return None
        node = nodes.get(entry.template)
        if isinstance(node, SubgraphNode) and child.template.workflow == node.body.workflow:
            path.append(entry.template)
        child = entry
        parent = parent.parent
    return tuple(reversed(path))


def _loop_input_source(
    admitted: AdmittedExecutionPlan,
    state: ActivationState,
    node: NodeId,
    activation: ActivationKey,
    port: str,
    default: WorkflowInputRef | ContextInputRef | NodeOutputRef | None,
) -> WorkflowInputRef | ContextInputRef | NodeOutputRef | None:
    if activation.parent is None or activation.iteration is None:
        return default
    parent = next(
        (item.template for item in state.entries if item.activation == activation.parent),
        None,
    )
    declaration = next(
        (
            item
            for scope in admitted.context.prepared.workflow.bodies()
            for item in scope.loops
            if item.member == node and item.starter == parent
        ),
        None,
    )
    if declaration is None:
        return default
    bindings = declaration.initial if activation.iteration == 0 else declaration.carried
    selected = [item.source for item in bindings if item.destination == NodeInputRef(node=node, port=port)]
    if len(selected) > 1:
        reject(EffectCode.DUPLICATE)
    return selected[0] if selected else default


def _is_mapped_item_input(
    admitted: AdmittedExecutionPlan,
    state: ActivationState,
    node: NodeId,
    activation: ActivationKey,
    port: str,
) -> bool:
    if activation.parent is None:
        return False
    parent = next(
        (item.template for item in state.entries if item.activation == activation.parent),
        None,
    )
    if parent is None:
        return False
    return any(
        declaration.member == node and declaration.expander == parent and declaration.item_input == port
        for scope in admitted.context.prepared.workflow.bodies()
        for declaration in scope.maps
    )


def _operation_has_omitted_context(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    node: NodeId,
) -> bool:
    context = admitted.context.bound_context
    if context is None:
        return False
    return any(
        fact.declaration.target == target
        and fact.declaration.node == node
        and fact.declaration.requirement == "optional"
        and fact.terminal == "omitted_optional"
        for fact in context.receipt.sources
    )


def _inherited_artifact_role(
    parent: ProvenanceKey | None,
    provenance: Mapping[ProvenanceKey, ArtifactProvenanceFact],
    ports: Mapping[tuple[ActivationKey, str], ExecutionPortFact],
) -> ArtifactRole:
    if parent is None:
        return "artifact"
    source = provenance.get(parent)
    if source is None or source.key != parent:
        return "artifact"
    if isinstance(parent, OperationOutputKey):
        occurrence = ports.get((parent.activation, parent.port))
        if (
            occurrence is not None
            and occurrence.activation == parent.activation
            and occurrence.target == parent.target
            and occurrence.port == parent.port
            and occurrence.role == "candidate"
        ):
            return "candidate"
    return "artifact"


def _source_activation(
    state: ActivationState,
    destination: ActivationKey,
    source: NodeId,
) -> ActivationKey | None:
    candidates = [item.activation for item in state.entries if item.template == source]
    destination_entry = next((item for item in state.entries if item.activation == destination), None)
    context = destination.parent
    if (
        destination_entry is not None
        and context is not None
        and any(
            declaration.member == destination_entry.template
            for scope in state.workflow.bodies()
            for declaration in (*scope.maps, *scope.loops)
        )
    ):
        context = context.parent
    if destination.iteration is not None:
        previous = [
            item
            for item in candidates
            if item.parent == destination.parent
            and item.iteration is not None
            and item.iteration == destination.iteration - 1
        ]
        if len(previous) == 1:
            return previous[0]
    dynamic_map = next(
        (
            declaration
            for scope in state.workflow.bodies()
            for declaration in scope.maps
            if declaration.member == source
        ),
        None,
    )
    if dynamic_map is not None:
        expanders = [
            item.activation
            for item in state.entries
            if item.template == dynamic_map.expander and item.activation.parent == context
        ]
        if len(expanders) != 1:
            return None
        expansion = next((item for item in state.expansions if item.parent == expanders[0]), None)
        if expansion is None or expansion.status != "closed" or len(expansion.members) != 1:
            return None
        selected = next(iter(expansion.members))
        return selected if selected in candidates else None
    dynamic_loop = next(
        (
            declaration
            for scope in state.workflow.bodies()
            for declaration in scope.loops
            if declaration.member == source
        ),
        None,
    )
    if dynamic_loop is not None:
        starters = [
            item.activation
            for item in state.entries
            if item.template == dynamic_loop.starter and item.activation.parent == context
        ]
        if len(starters) != 1:
            return None
        exited = [
            item.activation
            for item in state.entries
            if item.template == source
            and item.activation.parent == starters[0]
            and item.outcome in dynamic_loop.exit_outcomes
        ]
        return exited[0] if len(exited) == 1 else None
    same_context = [item for item in candidates if item.parent == context]
    if len(same_context) == 1:
        return same_context[0]
    if destination.parent in candidates:
        return destination.parent
    parent_context = [item for item in candidates if item.parent == destination]
    return parent_context[0] if len(parent_context) == 1 else None


def _workflow_owners(root: Graph) -> frozenset[WorkflowId]:
    return frozenset(body.workflow for body in root.bodies())


def _node_operation(root: Graph, node: NodeId) -> OperationSpec:
    for body in root.bodies():
        for candidate in body.operations:
            if candidate.id == node:
                return candidate.operation
    reject(EffectCode.MISSING)
