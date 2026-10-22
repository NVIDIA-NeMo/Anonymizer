# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Activation scope and parent lookup."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from threading import Lock
from typing import TYPE_CHECKING, Never

from anonymizer.engine.graph_sdk._graph_values import ActivationKey, ContractViolation, ValidationCode
from anonymizer.engine.graph_sdk._workflow_values import Graph, KeyedJoinDecl, LoopDecl, MapDecl, NodeId, Operation

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk._activation_values import ActivationEntry, ActivationSeed


_TOPOLOGY_CACHE_SIZE = 32
_topology_lock = Lock()
# Graphs are immutable and compare by identity; each cached entry keeps its graph alive, so ids are not reused.
_topology: OrderedDict[int, tuple[Graph, dict[NodeId, tuple[Graph, Operation]]]] = OrderedDict()


def _placements(workflow: Graph) -> dict[NodeId, tuple[Graph, Operation]]:
    """Each template's first scope in ``bodies()`` order, computed once per graph."""
    with _topology_lock:
        cached = _topology.get(id(workflow))
        if cached is not None and cached[0] is workflow:
            _topology.move_to_end(id(workflow))
            return cached[1]
    placements: dict[NodeId, tuple[Graph, Operation]] = {}
    for scope in workflow.bodies():
        for node in scope.operations:
            placements.setdefault(node.id, (scope, node))
    with _topology_lock:
        _topology[id(workflow)] = (workflow, placements)
        _topology.move_to_end(id(workflow))
        while len(_topology) > _TOPOLOGY_CACHE_SIZE:
            _topology.popitem(last=False)
    return placements


def _scope_for(workflow: Graph, template: NodeId) -> Graph:
    placement = _placements(workflow).get(template)
    if placement is None:
        _reject(ValidationCode.MISSING)
    return placement[0]


def _node(workflow: Graph, template: NodeId) -> Operation:
    placement = _placements(workflow).get(template)
    if placement is None:
        _reject(ValidationCode.MISSING)
    return placement[1]


def _map_for(scope: Graph, template: NodeId) -> MapDecl | None:
    return next((item for item in scope.maps if item.expander == template), None)


def _loop_for(scope: Graph, template: NodeId) -> LoopDecl | None:
    return next((item for item in scope.loops if item.starter == template), None)


def _join_for(scope: Graph, source: NodeId) -> KeyedJoinDecl | None:
    return next((item for item in scope.joins if item.source == source), None)


def _selected_path(scope: Graph, entries: Mapping[NodeId, ActivationEntry]) -> tuple[frozenset[NodeId], NodeId | None]:
    """Return one scope occurrence's selected nodes and their unique sink.

    This is the runtime reading of authoring admission's path rule
    (``_workflow_composition._validate_paths``): choice branches join the path
    only when their selector's observed outcome names them, and the sink is the
    one selected node without an outgoing selected sequence edge. ``entries``
    maps each directly scoped template to its entry in that occurrence.
    """
    branch_members = {member for choice in scope.choices for branch in choice.branches for member in branch.members}
    selected = {node.id for node in scope.operations} - branch_members
    while True:
        additions = {
            member
            for choice in scope.choices
            if choice.selector in selected
            and (selector := entries.get(choice.selector)) is not None
            and selector.outcome is not None
            for branch in choice.branches
            if selector.outcome in branch.outcomes
            for member in branch.members
        }
        if additions <= selected:
            break
        selected |= additions
    outgoing = {edge.before for edge in scope.sequence if edge.before in selected and edge.after in selected}
    sinks = selected - outgoing
    return frozenset(selected), next(iter(sinks)) if len(sinks) == 1 else None


def _mapped_outcome(scope: Graph, entry: ActivationEntry) -> str | None:
    bound = {
        binding.destination.outcome
        for binding in scope.outcome_bindings
        if binding.source.node == entry.template and binding.source.outcome == entry.outcome
    }
    return next(iter(bound)) if entry.outcome is not None and len(bound) == 1 else None


def _selected_terminal(
    scope: Graph, entries: Mapping[NodeId, ActivationEntry]
) -> tuple[ActivationEntry | None, str | None]:
    """Return the selected sink's entry and the scope outcome its outcome binds to.

    The scope's outcome is exactly the binding of its selected sink's outcome;
    any other node's bound outcome is not the scope's outcome. A missing sink,
    or one without a bound outcome, yields no scope outcome.
    """
    _, sink = _selected_path(scope, entries)
    entry = entries.get(sink) if sink is not None else None
    return entry, None if entry is None else _mapped_outcome(scope, entry)


def _publication_outcome(scope: Graph, entries: Mapping[NodeId, ActivationEntry]) -> str | None:
    """Return the outcome labelling a scope's published outputs, if any.

    When the selected sink succeeded, the label is its bound outcome. When the
    selected sink exists but did not succeed, the scope has no success outcome:
    its candidate stays inspectable only when the selected path's successful
    bound outcomes are unanimous, and the sink's non-success terminal withholds
    it from release (RFC "Failure, Cancellation, and Release": withholding
    propagates through affected dependencies).
    """
    selected, sink = _selected_path(scope, entries)
    entry = entries.get(sink) if sink is not None else None
    if entry is None:
        return None
    if entry.status == "success":
        return _mapped_outcome(scope, entry)
    labels = {
        label
        for node in selected
        if (item := entries.get(node)) is not None
        and item.status == "success"
        and (label := _mapped_outcome(scope, item)) is not None
    }
    return next(iter(labels)) if len(labels) == 1 else None


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


def _context(seed: ActivationSeed, scope: Graph) -> ActivationKey | None:
    dynamic_member = any(seed.template == item.member for item in (*scope.maps, *scope.loops))
    return (
        seed.activation.parent.parent
        if dynamic_member and seed.activation.parent is not None
        else seed.activation.parent
    )


def _reject(code: ValidationCode) -> Never:
    raise ContractViolation(code) from None
