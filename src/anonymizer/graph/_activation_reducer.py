# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure activation normalization and transitions."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal, TypeAlias

from anonymizer.graph._activation_topology import _context, _depth, _join_for, _loop_for, _map_for, _node, _scope_for
from anonymizer.graph._activation_values import (
    _TERMINAL,
    ActivationEntry,
    ActivationEvent,
    ActivationLimits,
    ActivationSeed,
    ActivationState,
    ActivationStatus,
    CloseUnstarted,
    ExpansionEntry,
    ExpansionStatus,
    ObserveMembership,
    ObserveOverflow,
    ObserveTerminal,
    Select,
    Start,
    _entry_map,
    _expansion_map,
    _reject,
    _seed_map,
)
from anonymizer.graph._values import ActivationKey, InvocationId, ValidationCode
from anonymizer.graph.workflow import (
    Graph,
    LoopDecl,
    NodeId,
    NodeOutputRef,
    SubgraphNode,
)


def initialize_activation(
    *,
    workflow: Graph,
    invocation: InvocationId,
    reservations: frozenset[ActivationSeed],
    limits: ActivationLimits,
) -> ActivationState:
    """Validate a complete finite occurrence reservation without selecting work."""
    if not isinstance(workflow, Graph) or not isinstance(invocation, InvocationId):
        _reject(ValidationCode.INVALID_TYPE)
    if not isinstance(reservations, frozenset) or any(not isinstance(seed, ActivationSeed) for seed in reservations):
        _reject(ValidationCode.INVALID_TYPE)
    if not isinstance(limits, ActivationLimits):
        _reject(ValidationCode.INVALID_TYPE)
    keys = [seed.activation for seed in reservations]
    if (
        len(reservations) > limits.max_entries
        or (reservations and limits.max_parent_depth == 0)
        or limits.max_events < 3 * len(reservations)
    ):
        _reject(ValidationCode.LIMIT_EXCEEDED)
    if any(key.invocation != invocation for key in keys):
        _reject(ValidationCode.FOREIGN_OWNER)
    if len(keys) != len(set(keys)):
        _reject(ValidationCode.DUPLICATE)
    for seed in reservations:
        _scope_for(workflow, seed.template)
    if len(reservations) < workflow.activation_upper_bound:
        _reject(ValidationCode.MISSING)
    if len(reservations) > workflow.activation_upper_bound:
        _reject(ValidationCode.LIMIT_EXCEEDED)
    expected_templates: dict[NodeId, int] = {}
    pending_workflows: list[tuple[Graph, int]] = [(workflow, 1)]
    while pending_workflows:
        current, multiplier = pending_workflows.pop()
        scope = current
        factors = {item.member: item.max_children for item in scope.maps}
        factors.update({item.member: item.max_iterations for item in scope.loops})
        for node in current.operations:
            count = multiplier * factors.get(node.id, 1)
            expected_templates[node.id] = expected_templates.get(node.id, 0) + count
            if isinstance(node, SubgraphNode):
                pending_workflows.append((node.body, count))
    observed_templates: dict[NodeId, int] = {}
    for seed in reservations:
        observed_templates[seed.template] = observed_templates.get(seed.template, 0) + 1
    if any(observed_templates.get(template, 0) < count for template, count in expected_templates.items()):
        _reject(ValidationCode.MISSING)
    reservation_map = _seed_map(reservations)
    body_parents: dict[int, set[NodeId]] = {}
    for scope in workflow.bodies():
        for node in scope.operations:
            if isinstance(node, SubgraphNode):
                body_parents.setdefault(id(node.body), set()).add(node.id)
    for seed in reservations:
        scope = _scope_for(workflow, seed.template)
        map_decl = next((item for item in scope.maps if item.member == seed.template), None)
        loop_decl = next((item for item in scope.loops if item.member == seed.template), None)
        parent_seed = reservation_map.get(seed.activation.parent) if seed.activation.parent is not None else None
        if map_decl is not None:
            if (
                parent_seed is None
                or parent_seed.template != map_decl.expander
                or seed.activation.iteration is not None
            ):
                _reject(ValidationCode.CONTRADICTORY)
        elif loop_decl is not None:
            if (
                parent_seed is None
                or parent_seed.template != loop_decl.starter
                or seed.activation.iteration is None
                or seed.activation.iteration >= loop_decl.max_iterations
            ):
                _reject(ValidationCode.CONTRADICTORY)
        elif scope is workflow:
            if seed.activation.parent is not None or seed.activation.iteration is not None:
                _reject(ValidationCode.CONTRADICTORY)
        else:
            expected_parents = body_parents[id(scope)]
            if (
                parent_seed is None
                or parent_seed.template not in expected_parents
                or seed.activation.iteration is not None
            ):
                _reject(ValidationCode.CONTRADICTORY)
    if any(_depth(key) > limits.max_parent_depth for key in keys):
        _reject(ValidationCode.LIMIT_EXCEEDED)
    if len(reservations) > limits.max_entries:
        _reject(ValidationCode.LIMIT_EXCEEDED)
    map_expanders = sum(
        _map_for(_scope_for(workflow, seed.template), seed.template) is not None for seed in reservations
    )
    if limits.max_events < 3 * len(reservations) + map_expanders:
        _reject(ValidationCode.LIMIT_EXCEEDED)
    return ActivationState(
        workflow=workflow,
        invocation=invocation,
        entries=frozenset(),
        expansions=frozenset(),
        reservations=reservations,
        events_applied=0,
        limits=limits,
    )


def _reservation_for(
    reservations: dict[ActivationKey, ActivationSeed],
    template: NodeId,
    context: ActivationKey | None,
    iteration: int | None = None,
) -> ActivationSeed | None:
    candidates = [
        seed
        for seed in reservations.values()
        if seed.template == template and seed.activation.parent == context and seed.activation.iteration == iteration
    ]
    return candidates[0] if len(candidates) == 1 else None


def _materialize(seed: ActivationSeed, entries: dict[ActivationKey, ActivationEntry]) -> None:
    if seed.activation not in entries:
        entries[seed.activation] = ActivationEntry(
            template=seed.template, activation=seed.activation, status="unstarted", outcome=None
        )


def _produces(workflow: Graph, entry: ActivationEntry, port: str) -> bool:
    if entry.outcome is None:
        return False
    node = _node(workflow, entry.template)
    return any(outcome.name == entry.outcome and port in outcome.produced_ports for outcome in node.operation.outcomes)


_Availability: TypeAlias = Literal["unstarted", "ready", "blocked"]


def _resolve_availability(
    workflow: Graph,
    template: NodeId,
    context: ActivationKey | None,
    scope: Graph,
    entries: dict[ActivationKey, ActivationEntry],
    expansions: dict[ActivationKey, ExpansionEntry],
    reservations: dict[ActivationKey, ActivationSeed],
    output: NodeOutputRef | None = None,
) -> _Availability:
    """Resolve an ordinary obligation through its exact dynamic occurrence."""
    seed = _reservation_for(reservations, template, context)
    entry = entries.get(seed.activation) if seed is not None else None
    impossible = False
    map_decl = next((item for item in scope.maps if item.member == template and item.max_children <= 1), None)
    if entry is None and map_decl is not None:
        parent = _reservation_for(reservations, map_decl.expander, context)
        expansion = expansions.get(parent.activation) if parent is not None else None
        if expansion is not None and expansion.status != "pending":
            if expansion.status == "closed" and len(expansion.members) == 1:
                entry = entries.get(next(iter(expansion.members)))
            else:
                impossible = True
    loop = next((item for item in scope.loops if item.member == template), None)
    if entry is None and loop is not None:
        parent = _reservation_for(reservations, loop.starter, context)
        starter = entries.get(parent.activation) if parent is not None else None
        iterations = [
            item
            for item in entries.values()
            if parent is not None and item.template == template and item.activation.parent == parent.activation
        ]
        exited = [item for item in iterations if item.status == "success" and item.outcome in loop.exit_outcomes]
        if len(exited) == 1:
            entry = exited[0]
        elif (
            starter is not None and starter.status in _TERMINAL and all(item.status in _TERMINAL for item in iterations)
        ):
            impossible = True
    if entry is None or entry.status not in _TERMINAL:
        return "blocked" if impossible else "unstarted"
    if output is not None and not _produces(workflow, entry, output.port):
        impossible = True
    return "blocked" if impossible else "ready"


def _initial_values_available(
    workflow: Graph,
    entry: ActivationEntry,
    loop: LoopDecl,
    entries: dict[ActivationKey, ActivationEntry],
    reservations: dict[ActivationKey, ActivationSeed],
) -> bool:
    scope = _scope_for(workflow, entry.template)
    context = _context(reservations[entry.activation], scope)
    for binding in loop.initial:
        if not isinstance(binding.source, NodeOutputRef):
            continue
        source_seed = _reservation_for(reservations, binding.source.node, context)
        source_entry = entries.get(source_seed.activation) if source_seed is not None else None
        if (
            source_entry is None
            or source_entry.status not in _TERMINAL
            or not _produces(workflow, source_entry, binding.source.port)
        ):
            return False
    return True


def _normalize(
    workflow: Graph,
    entries: dict[ActivationKey, ActivationEntry],
    expansions: dict[ActivationKey, ExpansionEntry],
    reservations: dict[ActivationKey, ActivationSeed],
) -> None:
    while True:
        previous_entries, previous_expansions = entries.copy(), expansions.copy()
        # Named choice outcomes materialize exactly one matching branch.
        for entry in list(entries.values()):
            if entry.status not in _TERMINAL or entry.outcome is None:
                continue
            scope = _scope_for(workflow, entry.template)
            context = _context(reservations[entry.activation], scope)
            for choice in scope.choices:
                if choice.selector != entry.template:
                    continue
                for branch in choice.branches:
                    if entry.outcome in branch.outcomes:
                        for template in branch.members:
                            seed = _reservation_for(reservations, template, context)
                            if seed is not None and seed.activation not in entries:
                                _materialize(seed, entries)

        # Aggregate state follows observed control outcomes.
        for entry in list(entries.values()):
            if entry.status not in _TERMINAL:
                continue
            scope = _scope_for(workflow, entry.template)
            map_decl = _map_for(scope, entry.template)
            loop = _loop_for(scope, entry.template)
            if map_decl is not None and (entry.outcome is None or entry.outcome not in map_decl.expansion_outcomes):
                prior = expansions.get(entry.activation)
                members = prior.members if prior is not None else frozenset()
                replacement = ExpansionEntry(parent=entry.activation, members=members, status="failed")
                if prior != replacement:
                    expansions[entry.activation] = replacement
            if loop is not None and entry.activation not in expansions:
                if entry.outcome is None:
                    status: ExpansionStatus = "failed"
                elif entry.outcome in loop.bypass_outcomes:
                    status = "closed"
                elif entry.outcome in loop.enter_outcomes and loop.max_iterations == 0:
                    status = "overflow"
                elif entry.outcome in loop.enter_outcomes:
                    seed = _reservation_for(reservations, loop.member, entry.activation, 0)
                    if seed is None or not _initial_values_available(workflow, entry, loop, entries, reservations):
                        status = "failed"
                    else:
                        _materialize(seed, entries)
                        status = "pending"
                else:
                    status = "failed"
                expansions[entry.activation] = ExpansionEntry(
                    parent=entry.activation,
                    members=frozenset(
                        key
                        for key, seed in reservations.items()
                        if seed.template == loop.member and key in entries and key.parent == entry.activation
                    ),
                    status=status,
                )

        for entry in list(entries.values()):
            if entry.status not in _TERMINAL or entry.activation.iteration is None:
                continue
            scope = _scope_for(workflow, entry.template)
            loop = next((item for item in scope.loops if item.member == entry.template), None)
            if loop is None or entry.activation.parent is None:
                continue
            parent = entry.activation.parent
            prior = expansions.get(parent)
            if prior is None or prior.status != "pending":
                continue
            members = prior.members | {entry.activation}
            if entry.outcome is None:
                status = "failed"
            elif entry.outcome in loop.exit_outcomes:
                status = "closed"
            elif entry.outcome in loop.continue_outcomes and entry.activation.iteration + 1 >= loop.max_iterations:
                status = "overflow"
            elif entry.outcome in loop.continue_outcomes:
                next_seed = _reservation_for(reservations, loop.member, parent, entry.activation.iteration + 1)
                if next_seed is None or not all(
                    _produces(workflow, entry, binding.source.port) for binding in loop.carried
                ):
                    status = "failed"
                else:
                    _materialize(next_seed, entries)
                    members |= {next_seed.activation}
                    status = "pending"
            else:
                status = "failed"
            replacement = ExpansionEntry(parent=parent, members=frozenset(members), status=status)
            if replacement != prior:
                expansions[parent] = replacement

        # Dynamic join closure is conjunction over exact membership.
        for source, expansion in list(expansions.items()):
            source_seed = reservations.get(source)
            if source_seed is None:
                continue
            scope = _scope_for(workflow, source_seed.template)
            join_decl = _join_for(scope, source_seed.template)
            if join_decl is None:
                continue
            context = _context(source_seed, scope)
            join_seed = _reservation_for(reservations, join_decl.join, context)
            if join_seed is None:
                continue
            if expansion.status == "pending":
                continue
            _materialize(join_seed, entries)
            join_entry = entries[join_seed.activation]
            if join_entry.status in {"running", *_TERMINAL}:
                continue
            terminal_members = [entries.get(member) for member in expansion.members]
            map_decl = _map_for(scope, source_seed.template)
            source_entry = entries.get(source)
            if expansion.status == "overflow":
                target: ActivationStatus = "inconsistent"
            elif map_decl is not None and (
                source_entry is None
                or source_entry.status not in _TERMINAL
                or source_entry.outcome not in map_decl.expansion_outcomes
            ):
                if source_entry is None or source_entry.status not in _TERMINAL:
                    continue
                target = "blocked"
            elif expansion.status == "failed":
                target = "blocked"
            elif any(item is None or item.status not in _TERMINAL for item in terminal_members):
                continue
            elif any(item.status not in join_decl.accepted_categories for item in terminal_members if item is not None):
                target = "blocked"
            else:
                target = "ready"
            replacement = replace(join_entry, status=target)
            if replacement != join_entry:
                entries[join_seed.activation] = replacement

        # Sequence and input obligations share occurrence resolution, but only
        # inputs require a produced port. An impossible obligation dominates a
        # pending one; a terminal predecessor's category does not block sequence.
        for key, entry in list(entries.items()):
            if entry.status != "unstarted":
                continue
            scope = _scope_for(workflow, entry.template)
            if any(item.join == entry.template for item in scope.joins):
                continue
            loop_member = next((item for item in scope.loops if item.member == entry.template), None)
            map_member = next((item for item in scope.maps if item.member == entry.template), None)
            requirements: list[_Availability] = []
            if loop_member is None:
                context = _context(reservations[key], scope)
                for edge in scope.sequence:
                    if edge.after == entry.template and (map_member is None or edge.before != map_member.expander):
                        requirements.append(
                            _resolve_availability(
                                workflow, edge.before, context, scope, entries, expansions, reservations
                            )
                        )
                mapped_item_port = (
                    map_member.item_input
                    if map_member is not None
                    and any(
                        key in expansion.members and expansion.parent == key.parent for expansion in expansions.values()
                    )
                    else None
                )
                for binding in scope.input_bindings:
                    if binding.destination.node != entry.template or (
                        mapped_item_port is not None and binding.destination.port == mapped_item_port
                    ):
                        continue
                    if isinstance(binding.source, NodeOutputRef):
                        requirements.append(
                            _resolve_availability(
                                workflow,
                                binding.source.node,
                                context,
                                scope,
                                entries,
                                expansions,
                                reservations,
                                binding.source,
                            )
                        )
            target = "blocked" if "blocked" in requirements else "unstarted" if "unstarted" in requirements else "ready"
            if target != entry.status:
                entries[key] = replace(entry, status=target)

        # Running subgraph containers derive their terminal from body sinks.
        for key, entry in list(entries.items()):
            if entry.status != "running":
                continue
            node = _node(workflow, entry.template)
            if not isinstance(node, SubgraphNode):
                continue
            body_entries = [item for item in entries.values() if item.activation.parent == key]
            if not body_entries or any(item.status not in _TERMINAL for item in body_entries):
                continue
            sink = next(
                (
                    item
                    for item in body_entries
                    if any(binding.source.node == item.template for binding in node.body.outcome_bindings)
                ),
                body_entries[-1],
            )
            outcome = sink.outcome
            if outcome is not None:
                binding = next(
                    (
                        binding
                        for binding in node.body.outcome_bindings
                        if binding.source.node == sink.template and binding.source.outcome == outcome
                    ),
                    None,
                )
                outcome = binding.destination.outcome if binding is not None else outcome
            entries[key] = replace(entry, status=sink.status, outcome=outcome)

        if entries == previous_entries and expansions == previous_expansions:
            return


def _completion_reserve(
    state: ActivationState,
    entries: dict[ActivationKey, ActivationEntry],
    expansions: dict[ActivationKey, ExpansionEntry],
) -> int:
    reservations = _seed_map(state.reservations)
    reserve = 3 * len(set(reservations) - set(entries))
    reserve += 2 * sum(item.status in {"unstarted", "ready"} for item in entries.values())
    reserve += sum(
        item.status == "running" and not isinstance(_node(state.workflow, item.template), SubgraphNode)
        for item in entries.values()
    )
    reserve += sum(
        _map_for(_scope_for(state.workflow, seed.template), seed.template) is not None
        and (key not in expansions or expansions[key].status == "pending")
        for key, seed in reservations.items()
    )
    return reserve


def advance_activation(*, state: ActivationState, event: ActivationEvent) -> ActivationState:
    """Apply one explicit event and publish one fully normalized immutable state."""
    if not isinstance(state, ActivationState) or not isinstance(
        event, (Select, Start, ObserveTerminal, CloseUnstarted, ObserveMembership, ObserveOverflow)
    ):
        _reject(ValidationCode.INVALID_TYPE)
    if state.events_applied + 1 > state.limits.max_events:
        _reject(ValidationCode.LIMIT_EXCEEDED)
    entries = _entry_map(state.entries)
    expansions = _expansion_map(state.expansions)
    reservations = _seed_map(state.reservations)

    event_keys: tuple[ActivationKey, ...]
    if isinstance(event, Select):
        event_keys = tuple(seed.activation for seed in event.seeds)
    elif isinstance(event, (ObserveMembership, ObserveOverflow)):
        event_keys = (event.parent,)
    else:
        event_keys = (event.activation,)
    if any(key.invocation != state.invocation for key in event_keys):
        _reject(ValidationCode.FOREIGN_OWNER)

    if isinstance(event, Select):
        keys = [seed.activation for seed in event.seeds]
        if len(keys) != len(set(keys)) or any(
            seed.activation in entries and entries[seed.activation].template != seed.template for seed in event.seeds
        ):
            _reject(ValidationCode.DUPLICATE)
        if any(seed.activation not in reservations for seed in event.seeds):
            _reject(ValidationCode.MISSING)
        if any(reservations[seed.activation].template != seed.template for seed in event.seeds):
            _reject(ValidationCode.CONTRADICTORY)
        for seed in event.seeds:
            scope = _scope_for(state.workflow, seed.template)
            choice_member = any(
                seed.template in branch.members for choice in scope.choices for branch in choice.branches
            )
            if choice_member:
                _reject(ValidationCode.OVERLAP)
            dynamic_member = any(seed.template == item.member for item in (*scope.maps, *scope.loops))
            if seed.activation.parent is not None or dynamic_member:
                _reject(ValidationCode.CONTRADICTORY)
        overlap = {seed.activation for seed in event.seeds} & set(entries)
        if overlap and overlap != {seed.activation for seed in event.seeds}:
            _reject(ValidationCode.DUPLICATE)
        for seed in event.seeds:
            _materialize(seed, entries)
    elif isinstance(event, Start):
        entry = entries.get(event.activation)
        if entry is None:
            _reject(ValidationCode.MISSING)
        if entry.status != "ready":
            _reject(ValidationCode.CONTRADICTORY)
        entries[event.activation] = replace(entry, status="running")
        node = _node(state.workflow, entry.template)
        if isinstance(node, SubgraphNode):
            choice_members = {
                member for choice in node.body.choices for branch in choice.branches for member in branch.members
            }
            roots = {body_node.id for body_node in node.body.operations} - choice_members
            for seed in reservations.values():
                if seed.activation.parent == event.activation and seed.template in roots:
                    _materialize(seed, entries)
    elif isinstance(event, ObserveTerminal):
        entry = entries.get(event.activation)
        if entry is None:
            _reject(ValidationCode.MISSING)
        if isinstance(_node(state.workflow, entry.template), SubgraphNode):
            _reject(ValidationCode.CONTRADICTORY)
        if entry.status in _TERMINAL:
            _reject(ValidationCode.DUPLICATE)
        if entry.status != "running":
            _reject(ValidationCode.CONTRADICTORY)
        if event.outcome is not None:
            operation = _node(state.workflow, entry.template).operation
            outcome = next((item for item in operation.outcomes if item.name == event.outcome), None)
            if outcome is None:
                _reject(ValidationCode.INVALID_VALUE)
            if outcome.category != event.category:
                _reject(ValidationCode.CONTRADICTORY)
        entries[event.activation] = replace(entry, status=event.category, outcome=event.outcome)
    elif isinstance(event, CloseUnstarted):
        entry = entries.get(event.activation)
        if entry is None:
            _reject(ValidationCode.MISSING)
        if entry.status in _TERMINAL:
            _reject(ValidationCode.DUPLICATE)
        if entry.status == "running":
            _reject(ValidationCode.CONTRADICTORY)
        entries[event.activation] = replace(entry, status=event.category)
    elif isinstance(event, ObserveMembership):
        parent_seed = reservations.get(event.parent)
        if parent_seed is None or event.parent not in entries:
            _reject(ValidationCode.MISSING)
        scope = _scope_for(state.workflow, parent_seed.template)
        declaration = _map_for(scope, parent_seed.template)
        if declaration is None:
            _reject(ValidationCode.MISSING)
        if len(event.members) > declaration.max_children:
            _reject(ValidationCode.LIMIT_EXCEEDED)
        expected = {
            key
            for key, seed in reservations.items()
            if seed.template == declaration.member and key.parent == event.parent and key.iteration is None
        }
        if not event.members <= expected:
            _reject(ValidationCode.CONTRADICTORY)
        prior = expansions.get(event.parent)
        if prior is not None and prior.status in {"failed", "overflow"}:
            _reject(ValidationCode.CONTRADICTORY)
        if prior is not None and prior.status == "closed" and (not event.closed or prior.members != event.members):
            _reject(ValidationCode.CONTRADICTORY)
        if prior is not None and not prior.members <= event.members:
            _reject(ValidationCode.CONTRADICTORY)
        for key in event.members:
            _materialize(reservations[key], entries)
        expansions[event.parent] = ExpansionEntry(
            parent=event.parent, members=event.members, status="closed" if event.closed else "pending"
        )
    else:
        assert isinstance(event, ObserveOverflow)
        parent_seed = reservations.get(event.parent)
        if parent_seed is None or event.parent not in entries:
            _reject(ValidationCode.MISSING)
        scope = _scope_for(state.workflow, parent_seed.template)
        map_decl = _map_for(scope, parent_seed.template)
        loop = _loop_for(scope, parent_seed.template)
        bound = map_decl.max_children if map_decl is not None else loop.max_iterations if loop is not None else None
        if bound is None:
            _reject(ValidationCode.MISSING)
        if event.observed_count <= bound:
            _reject(ValidationCode.INVALID_VALUE)
        prior = expansions.get(event.parent)
        if prior is not None and prior.status != "pending":
            _reject(ValidationCode.CONTRADICTORY)
        expansions[event.parent] = ExpansionEntry(
            parent=event.parent, members=prior.members if prior is not None else frozenset(), status="overflow"
        )

    _normalize(state.workflow, entries, expansions, reservations)
    if len(entries) > state.limits.max_entries or len(entries) > state.workflow.activation_upper_bound:
        _reject(ValidationCode.LIMIT_EXCEEDED)
    candidate = replace(
        state,
        entries=frozenset(entries.values()),
        expansions=frozenset(expansions.values()),
        events_applied=state.events_applied + 1,
    )
    if candidate.events_applied + _completion_reserve(candidate, entries, expansions) > candidate.limits.max_events:
        _reject(ValidationCode.LIMIT_EXCEEDED)
    return candidate
