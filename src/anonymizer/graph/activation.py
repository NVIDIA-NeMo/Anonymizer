# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure bounded activation transitions for admitted protection workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Never, SupportsIndex, TypeAlias

from anonymizer.graph._values import ActivationKey, ContractViolation, InvocationId, ValidationCode
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    AdmittedWorkflow,
    DynamicScope,
    KeyedJoinDecl,
    LoopDecl,
    MapDecl,
    Node,
    NodeId,
    NodeOutputRef,
    OutcomeClass,
    SubgraphNode,
)

ActivationStatus: TypeAlias = Literal[
    "unstarted", "ready", "running", "success", "failure", "cancelled", "lost", "blocked", "inconsistent"
]
ExpansionStatus: TypeAlias = Literal["pending", "closed", "failed", "overflow"]
_TERMINAL = frozenset(("success", "failure", "cancelled", "lost", "blocked", "inconsistent"))
_CATEGORIES = _TERMINAL
_PICKLE_ERROR = "activation value serialization is not supported"


def _reject(code: ValidationCode) -> Never:
    raise ContractViolation(code) from None


class _PrivateValue:
    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def __copy__(self) -> _PrivateValue:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _PrivateValue:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Never:
        del protocol
        raise TypeError(_PICKLE_ERROR)


def _integer(value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(ValidationCode.INVALID_TYPE)
    if value < (1 if positive else 0):
        _reject(ValidationCode.INVALID_VALUE)
    return value


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActivationSeed(_PrivateValue):
    template: NodeId
    activation: ActivationKey

    def __post_init__(self) -> None:
        if not isinstance(self.template, NodeId) or not isinstance(self.activation, ActivationKey):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActivationEntry(_PrivateValue):
    template: NodeId
    activation: ActivationKey
    status: ActivationStatus
    outcome: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.template, NodeId) or not isinstance(self.activation, ActivationKey):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.status, str) or (self.outcome is not None and not isinstance(self.outcome, str)):
            _reject(ValidationCode.INVALID_TYPE)
        if self.status not in {"unstarted", "ready", "running", *_TERMINAL} or self.outcome == "":
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExpansionEntry(_PrivateValue):
    parent: ActivationKey
    members: frozenset[ActivationKey]
    status: ExpansionStatus

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent, ActivationKey)
            or not isinstance(self.members, frozenset)
            or any(not isinstance(member, ActivationKey) for member in self.members)
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.status, str):
            _reject(ValidationCode.INVALID_TYPE)
        if self.status not in {"pending", "closed", "failed", "overflow"}:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActivationLimits(_PrivateValue):
    max_events: int
    max_entries: int
    max_parent_depth: int

    def __post_init__(self) -> None:
        _integer(self.max_events)
        _integer(self.max_entries)
        _integer(self.max_parent_depth)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActivationState(_PrivateValue):
    workflow: AdmittedActivationWorkflow
    invocation: InvocationId
    entries: frozenset[ActivationEntry]
    expansions: frozenset[ExpansionEntry]
    reservations: frozenset[ActivationSeed]
    events_applied: int
    limits: ActivationLimits

    def __post_init__(self) -> None:
        if not isinstance(self.workflow, AdmittedActivationWorkflow) or not isinstance(self.invocation, InvocationId):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.entries, frozenset) or any(
            not isinstance(item, ActivationEntry) for item in self.entries
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.expansions, frozenset) or any(
            not isinstance(item, ExpansionEntry) for item in self.expansions
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.reservations, frozenset) or any(
            not isinstance(item, ActivationSeed) for item in self.reservations
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.limits, ActivationLimits):
            _reject(ValidationCode.INVALID_TYPE)
        _integer(self.events_applied)
        entry_keys = [item.activation for item in self.entries]
        expansion_keys = [item.parent for item in self.expansions]
        reservation_keys = [item.activation for item in self.reservations]
        if (
            len(entry_keys) != len(set(entry_keys))
            or len(expansion_keys) != len(set(expansion_keys))
            or len(reservation_keys) != len(set(reservation_keys))
        ):
            _reject(ValidationCode.DUPLICATE)
        if any(key.invocation != self.invocation for key in (*entry_keys, *expansion_keys, *reservation_keys)):
            _reject(ValidationCode.FOREIGN_OWNER)
        reservations = _seed_map(self.reservations)
        if any(
            item.activation not in reservations or reservations[item.activation].template != item.template
            for item in self.entries
        ) or any(item.parent not in reservations for item in self.expansions):
            _reject(ValidationCode.MISSING)
        if (
            self.events_applied > self.limits.max_events
            or len(self.entries) > self.limits.max_entries
            or len(self.entries) > self.workflow.activation_upper_bound
        ):
            _reject(ValidationCode.LIMIT_EXCEEDED)

    @property
    def complete(self) -> bool:
        """Whether every selected obligation is terminal and every expansion is settled."""
        entries = _entry_map(self.entries)
        if not entries:
            return False
        reservations = _seed_map(self.reservations)
        required: set[ActivationKey] = set()
        for key, seed in reservations.items():
            scope = _scope_for(self.workflow, seed.template)
            choice_member = any(
                seed.template in branch.members for choice in scope.workflow.choices for branch in choice.branches
            )
            dynamic_member = any(seed.template == item.member for item in (*scope.maps, *scope.loops))
            if key.parent is None and not choice_member and not dynamic_member:
                required.add(key)
            if (
                key.parent is not None
                and key.parent in entries
                and isinstance(_node(self.workflow, entries[key.parent].template), SubgraphNode)
            ):
                if not choice_member and not dynamic_member:
                    required.add(key)
        for entry in entries.values():
            if entry.outcome is None:
                continue
            scope = _scope_for(self.workflow, entry.template)
            context = _context(reservations[entry.activation], scope)
            for choice in scope.workflow.choices:
                if choice.selector == entry.template:
                    required.update(
                        seed.activation
                        for seed in reservations.values()
                        for branch in choice.branches
                        if entry.outcome in branch.outcomes and seed.template in branch.members
                        if _context(seed, scope) == context
                    )
        for expansion in self.expansions:
            required.update(expansion.members)
        return (
            required <= entries.keys()
            and all(entries[key].status in _TERMINAL for key in required)
            and all(
                expansion.status != "pending"
                and all(member in entries and entries[member].status in _TERMINAL for member in expansion.members)
                for expansion in self.expansions
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Select(_PrivateValue):
    seeds: frozenset[ActivationSeed]

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, frozenset) or any(not isinstance(seed, ActivationSeed) for seed in self.seeds):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Start(_PrivateValue):
    activation: ActivationKey

    def __post_init__(self) -> None:
        if not isinstance(self.activation, ActivationKey):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObserveTerminal(_PrivateValue):
    activation: ActivationKey
    outcome: str | None
    category: OutcomeClass

    def __post_init__(self) -> None:
        if not isinstance(self.activation, ActivationKey) or (
            self.outcome is not None and not isinstance(self.outcome, str)
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.category, str):
            _reject(ValidationCode.INVALID_TYPE)
        if (
            self.category not in _CATEGORIES
            or self.outcome == ""
            or (self.outcome is None and self.category == "success")
        ):
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CloseUnstarted(_PrivateValue):
    activation: ActivationKey
    category: Literal["blocked", "inconsistent"]

    def __post_init__(self) -> None:
        if not isinstance(self.activation, ActivationKey) or not isinstance(self.category, str):
            _reject(ValidationCode.INVALID_TYPE)
        if self.category not in {"blocked", "inconsistent"}:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObserveMembership(_PrivateValue):
    parent: ActivationKey
    members: frozenset[ActivationKey]
    closed: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent, ActivationKey)
            or not isinstance(self.members, frozenset)
            or any(not isinstance(item, ActivationKey) for item in self.members)
            or not isinstance(self.closed, bool)
        ):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObserveOverflow(_PrivateValue):
    parent: ActivationKey
    observed_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.parent, ActivationKey):
            _reject(ValidationCode.INVALID_TYPE)
        _integer(self.observed_count)


ActivationEvent: TypeAlias = Select | Start | ObserveTerminal | CloseUnstarted | ObserveMembership | ObserveOverflow


def _entry_map(items: frozenset[ActivationEntry]) -> dict[ActivationKey, ActivationEntry]:
    return {item.activation: item for item in items}


def _expansion_map(items: frozenset[ExpansionEntry]) -> dict[ActivationKey, ExpansionEntry]:
    return {item.parent: item for item in items}


def _seed_map(items: frozenset[ActivationSeed]) -> dict[ActivationKey, ActivationSeed]:
    return {item.activation: item for item in items}


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


def initialize_activation(
    *,
    workflow: AdmittedActivationWorkflow,
    invocation: InvocationId,
    reservations: frozenset[ActivationSeed],
    limits: ActivationLimits,
) -> ActivationState:
    """Validate a complete finite occurrence reservation without selecting work."""
    if not isinstance(workflow, AdmittedActivationWorkflow) or not isinstance(invocation, InvocationId):
        _reject(ValidationCode.INVALID_TYPE)
    if not isinstance(reservations, frozenset) or any(not isinstance(seed, ActivationSeed) for seed in reservations):
        _reject(ValidationCode.INVALID_TYPE)
    if not isinstance(limits, ActivationLimits):
        _reject(ValidationCode.INVALID_TYPE)
    keys = [seed.activation for seed in reservations]
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
    pending_workflows: list[tuple[AdmittedWorkflow, int]] = [(workflow.workflow, 1)]
    while pending_workflows:
        current, multiplier = pending_workflows.pop()
        scope = next(item for item in workflow.scopes if item.workflow is current)
        factors = {item.member: item.max_children for item in scope.maps}
        factors.update({item.member: item.max_iterations for item in scope.loops})
        for node in current.nodes:
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
    body_parents = {
        id(node.body): node.id
        for scope in workflow.scopes
        for node in scope.workflow.nodes
        if isinstance(node, SubgraphNode)
    }
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
        elif scope.workflow is workflow.workflow:
            if seed.activation.parent is not None or seed.activation.iteration is not None:
                _reject(ValidationCode.CONTRADICTORY)
        else:
            expected_parent = body_parents[id(scope.workflow)]
            if parent_seed is None or parent_seed.template != expected_parent or seed.activation.iteration is not None:
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


def _context(seed: ActivationSeed, scope: DynamicScope) -> ActivationKey | None:
    dynamic_member = any(seed.template == item.member for item in (*scope.maps, *scope.loops))
    return (
        seed.activation.parent.parent
        if dynamic_member and seed.activation.parent is not None
        else seed.activation.parent
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


def _produces(workflow: AdmittedActivationWorkflow, entry: ActivationEntry, port: str) -> bool:
    if entry.outcome is None:
        return False
    node = _node(workflow, entry.template)
    return any(outcome.name == entry.outcome and port in outcome.produced_ports for outcome in node.operation.outcomes)


def _initial_values_available(workflow: AdmittedActivationWorkflow, entry: ActivationEntry, loop: LoopDecl) -> bool:
    return all(
        not isinstance(binding.source, NodeOutputRef)
        or binding.source.node != entry.template
        or _produces(workflow, entry, binding.source.port)
        for binding in loop.initial
    )


def _normalize(
    workflow: AdmittedActivationWorkflow,
    entries: dict[ActivationKey, ActivationEntry],
    expansions: dict[ActivationKey, ExpansionEntry],
    reservations: dict[ActivationKey, ActivationSeed],
) -> None:
    changed = True
    while changed:
        changed = False
        # Named choice outcomes materialize exactly one matching branch.
        for entry in list(entries.values()):
            if entry.status not in _TERMINAL or entry.outcome is None:
                continue
            scope = _scope_for(workflow, entry.template)
            context = _context(reservations[entry.activation], scope)
            for choice in scope.workflow.choices:
                if choice.selector != entry.template:
                    continue
                for branch in choice.branches:
                    if entry.outcome in branch.outcomes:
                        for template in branch.members:
                            seed = _reservation_for(reservations, template, context)
                            if seed is not None and seed.activation not in entries:
                                _materialize(seed, entries)
                                changed = True

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
                    changed = True
            if loop is not None and entry.activation not in expansions:
                if entry.outcome is None:
                    status: ExpansionStatus = "failed"
                elif entry.outcome in loop.bypass_outcomes:
                    status = "closed"
                elif entry.outcome in loop.enter_outcomes and loop.max_iterations == 0:
                    status = "overflow"
                elif entry.outcome in loop.enter_outcomes:
                    seed = _reservation_for(reservations, loop.member, entry.activation, 0)
                    if seed is None or not _initial_values_available(workflow, entry, loop):
                        status = "failed"
                    else:
                        _materialize(seed, entries)
                        status = "pending"
                    changed = True
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
                changed = True

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
                    was_absent = next_seed.activation not in entries
                    _materialize(next_seed, entries)
                    members |= {next_seed.activation}
                    status = "pending"
                    changed |= was_absent
            else:
                status = "failed"
            replacement = ExpansionEntry(parent=parent, members=frozenset(members), status=status)
            if replacement != prior:
                expansions[parent] = replacement
                changed = True

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
                changed = True

        # Ordinary readiness and impossible-input closure.
        for key, entry in list(entries.items()):
            if entry.status != "unstarted":
                continue
            scope = _scope_for(workflow, entry.template)
            if any(item.join == entry.template for item in scope.joins):
                continue
            if any(item.member == entry.template for item in (*scope.maps, *scope.loops)):
                ready = True
                blocked = False
            else:
                context = _context(reservations[key], scope)
                predecessors = [edge.before for edge in scope.workflow.sequence if edge.after == entry.template]
                prior_entries: list[ActivationEntry] = []
                missing_prior = False
                for template in predecessors:
                    seed = _reservation_for(reservations, template, context)
                    prior = entries.get(seed.activation) if seed is not None else None
                    if prior is None or prior.status not in _TERMINAL:
                        missing_prior = True
                    else:
                        prior_entries.append(prior)
                bindings = [
                    binding for binding in scope.workflow.input_bindings if binding.destination.node == entry.template
                ]
                impossible = False
                inputs_ready = True
                for binding in bindings:
                    source = binding.source
                    if not isinstance(source, NodeOutputRef):
                        continue
                    source_seed = _reservation_for(reservations, source.node, context)
                    source_entry = entries.get(source_seed.activation) if source_seed is not None else None
                    if source_entry is None or source_entry.status not in _TERMINAL:
                        inputs_ready = False
                    elif not _produces(workflow, source_entry, source.port):
                        impossible = True
                blocked = impossible
                ready = not missing_prior and inputs_ready and not blocked
            target = "blocked" if blocked else "ready" if ready else "unstarted"
            if target != entry.status:
                entries[key] = replace(entry, status=target)
                changed = True

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
            changed = True


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
                seed.template in branch.members for choice in scope.workflow.choices for branch in choice.branches
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
            roots = {node.id for node in node.body.nodes}
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
