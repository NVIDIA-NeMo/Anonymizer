# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow and activation lookup for execution."""

from __future__ import annotations

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
    ActivationState,
)
from anonymizer.graph.workflow import (
    AdmittedWorkflow,
    ContextInputRef,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
)


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
            for scope in admitted.context.prepared.workflow.scopes
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
        for scope in admitted.context.prepared.workflow.scopes
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
    provenance: list[ArtifactProvenanceFact],
    ports: list[ExecutionPortFact],
) -> ArtifactRole:
    if parent is None:
        return "artifact"
    source = next((item for item in provenance if item.key == parent), None)
    if source is None:
        return "artifact"
    if source.decision:
        return "decision"
    if isinstance(parent, OperationOutputKey):
        occurrence = next(
            (
                item
                for item in ports
                if item.activation == parent.activation and item.target == parent.target and item.port == parent.port
            ),
            None,
        )
        if occurrence is not None and occurrence.role == "candidate":
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
            for scope in state.workflow.scopes
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
        (declaration for scope in state.workflow.scopes for declaration in scope.maps if declaration.member == source),
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
        (declaration for scope in state.workflow.scopes for declaration in scope.loops if declaration.member == source),
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


def _operation_owner(root: AdmittedWorkflow, node: NodeId) -> tuple[AdmittedWorkflow, OperationNode]:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in current.nodes:
            if isinstance(candidate, OperationNode) and candidate.id == node:
                return current, candidate
            if isinstance(candidate, SubgraphNode):
                pending.append(candidate.body)
    reject(EffectCode.MISSING)


def _workflow_owners(root: AdmittedWorkflow) -> frozenset[WorkflowId]:
    owners: set[WorkflowId] = set()
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        owners.add(current.workflow)
        pending.extend(candidate.body for candidate in current.nodes if isinstance(candidate, SubgraphNode))
    return frozenset(owners)


def _node_operation(root: AdmittedWorkflow, node: NodeId) -> OperationSpec:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in current.nodes:
            if candidate.id == node:
                return candidate.operation
            if isinstance(candidate, SubgraphNode):
                pending.append(candidate.body)
    reject(EffectCode.MISSING)


def _is_subgraph_node(root: AdmittedWorkflow, node: NodeId) -> bool:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in current.nodes:
            if isinstance(candidate, SubgraphNode):
                if candidate.id == node:
                    return True
                pending.append(candidate.body)
    return False


def _subgraph_owner(root: AdmittedWorkflow, node: NodeId) -> tuple[AdmittedWorkflow, SubgraphNode]:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in current.nodes:
            if isinstance(candidate, SubgraphNode):
                if candidate.id == node:
                    return current, candidate
                pending.append(candidate.body)
    reject(EffectCode.MISSING)
