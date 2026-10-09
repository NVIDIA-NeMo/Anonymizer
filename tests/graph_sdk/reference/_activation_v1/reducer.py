# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent activation_v1 reference: reducer."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from tests.graph_sdk.reference._activation_v1.model import (
    CATEGORIES,
    TERMINAL,
    Close,
    Code,
    Declaration,
    Entry,
    Event,
    Expansion,
    Initialize,
    LoopAggregate,
    MapAggregate,
    Membership,
    Overflow,
    Rejected,
    Role,
    Seed,
    Select,
    Start,
    Subgraph,
    TerminalEvent,
    completion_reserve,
)
from tests.graph_sdk.reference._activation_v1.parsing import (
    _raw_depth,
)


def _initialize(declaration: Declaration) -> None:
    seen: dict[str, Seed] = {}
    raw_parents: dict[str, list[str | None]] = {}
    duplicate = False
    for seed in declaration.seeds:
        duplicate |= seed.key in seen
        seen.setdefault(seed.key, seed)
        raw_parents.setdefault(seed.key, []).append(seed.parent)
    foreign = any(seed.invocation != declaration.invocation for seed in declaration.seeds)
    missing = not set(declaration.required) <= seen.keys()
    duplicate |= len(declaration.required) != len(set(declaration.required))
    duplicate |= len(declaration.edges) != len(set(declaration.edges))
    duplicate |= len(declaration.input_dependencies) != len(set(declaration.input_dependencies))
    duplicate |= len(declaration.outcomes) != len(
        {(outcome.scope, outcome.template, outcome.name) for outcome in declaration.outcomes}
    )
    duplicate |= any(
        len(outcome.produced_ports) != len(set(outcome.produced_ports)) for outcome in declaration.outcomes
    )
    map_count = sum(seed.role == "map_expander" for seed in declaration.seeds)
    limit = (
        declaration.limits.max_entries < len(declaration.seeds)
        or declaration.limits.max_events < 3 * len(declaration.seeds) + map_count
    )
    parent_facts = {key: tuple(dict.fromkeys(parents)) for key, parents in raw_parents.items()}
    depth_facts = tuple(_raw_depth(key, parent_facts) for key in parent_facts)
    maximum_depth = max((depth for depth, _, _ in depth_facts), default=0)
    limit |= declaration.limits.max_parent_depth < maximum_depth
    missing |= any(parent_missing for _, parent_missing, _ in depth_facts)
    cycle_from_parents = any(parent_cycle for _, _, parent_cycle in depth_facts)
    contradictory = False
    dynamic_overlap = False
    outcome_names = {(outcome.scope, outcome.template, outcome.name) for outcome in declaration.outcomes}
    missing |= any(before not in seen or after not in seen for before, after in declaration.edges)
    missing |= any(
        dependency.source not in seen or dependency.destination not in seen
        for dependency in declaration.input_dependencies
    )
    duplicate |= any(before == after for before, after in declaration.edges)
    contradictory |= any(
        (dependency.source, dependency.destination) not in declaration.edges
        for dependency in declaration.input_dependencies
    )
    for choice in declaration.choices:
        selector = seen.get(choice.selector)
        missing |= selector is None
        branch_members = [member for _, members in choice.branches for member in members]
        duplicate |= any(len(members) != len(set(members)) for _, members in choice.branches)
        dynamic_overlap |= any(
            set(left_members) & set(right_members)
            for index, (_, left_members) in enumerate(choice.branches)
            for _, right_members in choice.branches[index + 1 :]
        )
        missing |= any(member not in seen for member in branch_members)
        missing |= bool(
            selector
            and any((selector.scope, selector.template, outcome) not in outcome_names for outcome, _ in choice.branches)
        )
    for subgraph in declaration.subgraphs:
        parent_seed = seen.get(subgraph.parent)
        missing |= parent_seed is None or subgraph.sink not in seen or any(root not in seen for root in subgraph.roots)
        duplicate |= len(subgraph.roots) != len(set(subgraph.roots))
        contradictory |= bool(parent_seed and parent_seed.role not in ("subgraph", "map_member"))
        contradictory |= any(seen[root].parent != subgraph.parent for root in subgraph.roots if root in seen)
        contradictory |= bool(subgraph.sink in seen and seen[subgraph.sink].parent != subgraph.parent)
    for aggregate in declaration.aggregates:
        parent = aggregate.parent if isinstance(aggregate, MapAggregate) else aggregate.starter
        parent_role: Role = "map_expander" if isinstance(aggregate, MapAggregate) else "loop_starter"
        member_role: Role = "map_member" if isinstance(aggregate, MapAggregate) else "loop_member"
        parent_seed, join_seed = seen.get(parent), seen.get(aggregate.join)
        missing |= parent_seed is None or join_seed is None or any(member not in seen for member in aggregate.members)
        contradictory |= bool(parent_seed and parent_seed.role != parent_role)
        contradictory |= bool(join_seed and join_seed.role != "join")
        duplicate |= len(aggregate.members) != len(set(aggregate.members))
        limit |= len(aggregate.members) > aggregate.bound
        contradictory |= len(aggregate.members) != aggregate.bound
        for index, member in enumerate(aggregate.members):
            seed = seen.get(member)
            if seed is None:
                continue
            contradictory |= seed.role != member_role or seed.parent != parent
            expected_iteration = index if isinstance(aggregate, LoopAggregate) else None
            contradictory |= seed.iteration != expected_iteration
        if isinstance(aggregate, MapAggregate):
            contradictory |= not aggregate.expansion_outcomes or not aggregate.accepted_categories
            missing |= bool(
                parent_seed
                and any(
                    (parent_seed.scope, parent_seed.template, outcome) not in outcome_names
                    for outcome in aggregate.expansion_outcomes
                )
            )
            duplicate |= len(aggregate.expansion_outcomes) != len(set(aggregate.expansion_outcomes))
            duplicate |= len(aggregate.accepted_categories) != len(set(aggregate.accepted_categories))
        else:
            contradictory |= bool(set(aggregate.enter_outcomes) & set(aggregate.bypass_outcomes))
            contradictory |= bool(set(aggregate.continue_outcomes) & set(aggregate.exit_outcomes))
            duplicate |= any(
                len(values) != len(set(values))
                for values in (
                    aggregate.enter_outcomes,
                    aggregate.bypass_outcomes,
                    aggregate.continue_outcomes,
                    aggregate.exit_outcomes,
                )
            )
            contradictory |= not all(
                (
                    aggregate.enter_outcomes,
                    aggregate.bypass_outcomes,
                    aggregate.continue_outcomes,
                    aggregate.exit_outcomes,
                )
            )
            missing |= bool(
                parent_seed
                and any(
                    (parent_seed.scope, parent_seed.template, outcome) not in outcome_names
                    for outcome in (*aggregate.enter_outcomes, *aggregate.bypass_outcomes)
                )
            )
            for member in aggregate.members:
                member_seed = seen.get(member)
                missing |= bool(
                    member_seed
                    and any(
                        (member_seed.scope, member_seed.template, outcome) not in outcome_names
                        for outcome in (*aggregate.continue_outcomes, *aggregate.exit_outcomes)
                    )
                )
    codes: list[Code] = []
    if limit:
        codes.append("limit_exceeded")
    if foreign:
        codes.append("foreign_owner")
    if duplicate:
        codes.append("duplicate")
    if missing:
        codes.append("missing")
    overlap = dynamic_overlap or any(
        set(left) & set(right)
        for index, left in enumerate(declaration.static_groups)
        for right in declaration.static_groups[index + 1 :]
    )
    cycle = cycle_from_parents or any(
        (after, before) in declaration.static_edges for before, after in declaration.static_edges
    )
    static_codes: list[Code] = []
    if overlap:
        static_codes.append("overlap")
    if cycle:
        static_codes.append("cycle")
    if not declaration.compatible or contradictory:
        static_codes.append("contradictory")
    if codes or static_codes:
        raise Rejected(*(codes + static_codes))


def _normalize(
    entries: dict[str, Entry], expansions: dict[str, Expansion], outputs: set[str], declaration: Declaration
) -> None:
    changed = True
    while changed:
        changed = False
        for key, entry in tuple(entries.items()):
            if entry.status != "unstarted":
                continue
            seed = next(item for item in declaration.seeds if item.key == key)
            if seed.role == "join":
                continue
            predecessors = [before for before, after in declaration.edges if after == key]
            if all(before in entries and entries[before].status in TERMINAL for before in predecessors):
                required_outputs = [
                    dependency for dependency in declaration.input_dependencies if dependency.destination == key
                ]
                blocked = any(
                    dependency.source not in entries
                    or dependency.source_port not in entries[dependency.source].produced_ports
                    for dependency in required_outputs
                )
                entries[key] = replace(
                    entry, status="blocked" if blocked else "ready", category="blocked" if blocked else None
                )
                changed = True
        for subgraph in declaration.subgraphs:
            parent, sink = entries.get(subgraph.parent), entries.get(subgraph.sink)
            if (
                parent
                and sink
                and parent.status == "running"
                and sink.status in TERMINAL
                and _subgraph_body_closed(subgraph, entries, expansions, declaration)
            ):
                projected = next(
                    (
                        outcome
                        for outcome in declaration.outcomes
                        if outcome.scope
                        == next(seed.scope for seed in declaration.seeds if seed.key == subgraph.parent)
                        and outcome.template == parent.template
                        and outcome.name == sink.outcome
                    ),
                    None,
                )
                produced_ports = projected.produced_ports if projected else ()
                entries[subgraph.parent] = replace(
                    parent,
                    status=sink.status,
                    category=sink.category,
                    outcome=sink.outcome,
                    produced_ports=produced_ports,
                )
                if produced_ports:
                    outputs.add(subgraph.parent)
                changed = True
        for aggregate in declaration.aggregates:
            aggregate_parent = aggregate.parent if isinstance(aggregate, MapAggregate) else aggregate.starter
            expansion, join, parent = (
                expansions.get(aggregate_parent),
                entries.get(aggregate.join),
                entries.get(aggregate_parent),
            )
            if (
                isinstance(aggregate, MapAggregate)
                and parent
                and parent.status in TERMINAL
                and (parent.outcome is None or parent.outcome not in aggregate.expansion_outcomes)
            ):
                members = expansion.members if expansion else frozenset()
                expansions[aggregate_parent] = Expansion(aggregate_parent, "failed", members)
                if join and join.status in ("unstarted", "ready"):
                    entries[aggregate.join] = replace(join, status="blocked", category="blocked")
            elif expansion and join and expansion.status == "failed" and join.status in ("unstarted", "ready"):
                entries[aggregate.join] = replace(join, status="blocked", category="blocked")
            elif expansion and join and expansion.status == "overflow" and join.status in ("unstarted", "ready"):
                entries[aggregate.join] = replace(join, status="inconsistent", category="inconsistent")
            elif expansion and join and expansion.status == "closed":
                children = [entries.get(member) for member in expansion.members]
                parent_admitted = not isinstance(aggregate, MapAggregate) or bool(
                    parent and parent.status in TERMINAL and parent.outcome in aggregate.expansion_outcomes
                )
                if (
                    parent_admitted
                    and join.status in ("unstarted", "ready")
                    and all(child and child.status in TERMINAL for child in children)
                ):
                    accepted_categories = (
                        aggregate.accepted_categories if isinstance(aggregate, MapAggregate) else ("success",)
                    )
                    accepted = all(child and child.category in accepted_categories for child in children)
                    entries[aggregate.join] = replace(
                        join, status="ready" if accepted else "blocked", category=None if accepted else "blocked"
                    )


def _subgraph_body_closed(
    subgraph: Subgraph,
    entries: Mapping[str, Entry],
    expansions: Mapping[str, Expansion],
    declaration: Declaration,
) -> bool:
    seeds = {seed.key: seed for seed in declaration.seeds}

    def belongs_to_body(key: str) -> bool:
        current = seeds[key].parent
        while current is not None:
            if current == subgraph.parent:
                return True
            current = seeds[current].parent
        return False

    def selected_choice_member(key: str) -> bool:
        memberships = [
            (choice.selector, outcome)
            for choice in declaration.choices
            for outcome, members in choice.branches
            if key in members
        ]
        return not memberships or any(
            selector in entries and entries[selector].outcome == outcome for selector, outcome in memberships
        )

    required = {
        seed.key
        for seed in declaration.seeds
        if belongs_to_body(seed.key)
        and selected_choice_member(seed.key)
        and (seed.role not in ("map_member", "loop_member") or seed.key in entries)
    }
    required.update((subgraph.sink, *subgraph.roots))
    if not required <= entries.keys() or any(entries[key].status not in TERMINAL for key in required):
        return False
    for aggregate in declaration.aggregates:
        parent = aggregate.parent if isinstance(aggregate, MapAggregate) else aggregate.starter
        if parent not in entries or not belongs_to_body(parent):
            continue
        expansion = expansions.get(parent)
        if expansion is None or expansion.status == "pending":
            return False
        if any(member not in entries or entries[member].status not in TERMINAL for member in expansion.members):
            return False
        if aggregate.join not in entries or entries[aggregate.join].status not in TERMINAL:
            return False
    return True


def _materialize(key: str, entries: dict[str, Entry], declaration: Declaration) -> None:
    seed = next((item for item in declaration.seeds if item.key == key), None)
    if seed is None:
        raise Rejected("missing")
    if key in entries:
        raise Rejected("duplicate")
    entries[key] = Entry(key, seed.scope, seed.template, "unstarted", None, None, ())


def _apply(
    event: Event,
    entries: dict[str, Entry],
    expansions: dict[str, Expansion],
    outputs: set[str],
    declaration: Declaration,
) -> None:
    if isinstance(event, Initialize):
        raise Rejected("contradictory")
    if event.invocation != declaration.invocation:
        raise Rejected("foreign_owner")
    if isinstance(event, Select):
        repeated = len(event.keys) != len(set(event.keys))
        overlap = {key for key in event.keys if key in entries}
        duplicate = repeated or bool(overlap) and overlap != set(event.keys)
        selected_seeds = [next((seed for seed in declaration.seeds if seed.key == key), None) for key in event.keys]
        missing = any(seed is None for seed in selected_seeds)
        dynamic_member = any(seed and seed.role in ("map_member", "loop_member") for seed in selected_seeds)
        premature_choice = any(
            key in members and (choice.selector not in entries or entries[choice.selector].outcome != outcome)
            for key in event.keys
            for choice in declaration.choices
            for outcome, members in choice.branches
        )
        if duplicate or missing or premature_choice or dynamic_member:
            raise Rejected(
                *(
                    (["duplicate"] if duplicate else [])
                    + (["missing"] if missing else [])
                    + (["overlap"] if premature_choice else [])
                    + (["contradictory"] if dynamic_member else [])
                )
            )
        if not overlap:
            for key in event.keys:
                _materialize(key, entries, declaration)
    elif isinstance(event, Start):
        entry = entries.get(event.key)
        if entry is None:
            raise Rejected("missing")
        if entry.status != "ready":
            raise Rejected("contradictory")
        entries[event.key] = replace(entry, status="running")
        subgraph = next((item for item in declaration.subgraphs if item.parent == event.key), None)
        if subgraph:
            for seed in declaration.seeds:
                if seed.parent == subgraph.parent:
                    _materialize(seed.key, entries, declaration)
    elif isinstance(event, TerminalEvent):
        entry = entries.get(event.key)
        if entry is None:
            raise Rejected("missing")
        if entry.status in TERMINAL:
            raise Rejected("duplicate")
        seed = next(item for item in declaration.seeds if item.key == event.key)
        if entry.status != "running" or seed.role == "subgraph":
            raise Rejected("contradictory")
        outcome_spec = next(
            (
                outcome
                for outcome in declaration.outcomes
                if outcome.scope == seed.scope and outcome.template == seed.template and outcome.name == event.outcome
            ),
            None,
        )
        if event.outcome is not None and outcome_spec is None:
            raise Rejected("invalid_value")
        if outcome_spec is not None and outcome_spec.category != event.category:
            raise Rejected("contradictory")
        produced_ports = outcome_spec.produced_ports if outcome_spec else ()
        entries[event.key] = replace(
            entry,
            status=event.category,
            category=event.category,
            outcome=event.outcome,
            produced_ports=produced_ports,
        )
        if produced_ports:
            outputs.add(event.key)
        for choice in declaration.choices:
            if choice.selector == event.key and event.outcome is not None:
                for outcome, members in choice.branches:
                    if outcome == event.outcome:
                        for member in members:
                            _materialize(member, entries, declaration)
        for aggregate in declaration.aggregates:
            if not isinstance(aggregate, LoopAggregate):
                continue
            if event.key == aggregate.starter:
                if event.outcome is None:
                    expansions[event.key] = Expansion(event.key, "failed", frozenset())
                elif event.outcome in aggregate.bypass_outcomes:
                    expansions[event.key] = Expansion(event.key, "closed", frozenset())
                elif event.outcome in aggregate.enter_outcomes and aggregate.bound == 0:
                    expansions[event.key] = Expansion(event.key, "overflow", frozenset())
                elif event.outcome in aggregate.enter_outcomes:
                    first = aggregate.members[0]
                    expansions[event.key] = Expansion(event.key, "pending", frozenset({first}))
                    _materialize(first, entries, declaration)
            elif event.key in aggregate.members:
                index = aggregate.members.index(event.key)
                known = frozenset(aggregate.members[: index + 1])
                if event.outcome is None:
                    expansions[aggregate.starter] = Expansion(aggregate.starter, "failed", known)
                elif event.outcome in aggregate.exit_outcomes:
                    expansions[aggregate.starter] = Expansion(aggregate.starter, "closed", known)
                elif (
                    event.outcome in aggregate.continue_outcomes
                    and aggregate.carried_binding is not None
                    and aggregate.carried_binding[1] not in produced_ports
                ):
                    expansions[aggregate.starter] = Expansion(aggregate.starter, "failed", known)
                elif event.outcome in aggregate.continue_outcomes and index + 1 >= aggregate.bound:
                    expansions[aggregate.starter] = Expansion(aggregate.starter, "overflow", known)
                elif event.outcome in aggregate.continue_outcomes:
                    expansions[aggregate.starter] = Expansion(aggregate.starter, "pending", known)
                    _materialize(aggregate.members[index + 1], entries, declaration)
    elif isinstance(event, Close):
        entry = entries.get(event.key)
        if entry is None:
            raise Rejected("missing")
        if entry.status in TERMINAL:
            raise Rejected("duplicate")
        if entry.status == "running":
            raise Rejected("contradictory")
        entries[event.key] = replace(entry, status=event.category, category=event.category)
    elif isinstance(event, Membership):
        aggregate = next(
            (item for item in declaration.aggregates if isinstance(item, MapAggregate) and item.parent == event.parent),
            None,
        )
        if aggregate is None:
            raise Rejected("missing")
        parent = entries.get(event.parent)
        if parent is None:
            raise Rejected("missing")
        if len(event.members) != len(set(event.members)):
            raise Rejected("duplicate")
        if len(event.members) > aggregate.bound:
            raise Rejected("limit_exceeded")
        if any(member not in aggregate.members for member in event.members):
            raise Rejected("contradictory")
        prior = expansions.get(event.parent)
        observed = frozenset(event.members)
        if prior and prior.status in ("failed", "overflow"):
            raise Rejected("contradictory")
        if prior and prior.status == "closed" and (not event.closed or prior.members != observed):
            raise Rejected("contradictory")
        if prior and prior.status == "pending" and not prior.members <= observed:
            raise Rejected("contradictory")
        for member in event.members:
            if member not in entries:
                _materialize(member, entries, declaration)
        expansions[event.parent] = Expansion(event.parent, "closed" if event.closed else "pending", observed)
    elif isinstance(event, Overflow):
        aggregate = next(
            (
                item
                for item in declaration.aggregates
                if (item.parent if isinstance(item, MapAggregate) else item.starter) == event.parent
            ),
            None,
        )
        if aggregate is None:
            raise Rejected("missing")
        if event.parent not in entries:
            raise Rejected("missing")
        if event.count <= aggregate.bound:
            raise Rejected("invalid_value")
        prior = expansions.get(event.parent)
        if prior and prior.status != "pending":
            raise Rejected("contradictory")
        expansions[event.parent] = Expansion(event.parent, "overflow", prior.members if prior else frozenset())
    _normalize(entries, expansions, outputs, declaration)


def _reserve(entries: Mapping[str, Entry], expansions: Mapping[str, Expansion], declaration: Declaration) -> int:
    absent = len({seed.key for seed in declaration.seeds} - entries.keys())
    waiting = sum(entry.status in ("unstarted", "ready") for entry in entries.values())
    roles = {seed.key: seed.role for seed in declaration.seeds}
    running = sum(entry.status == "running" and roles[entry.activation] != "subgraph" for entry in entries.values())
    open_maps = sum(
        isinstance(item, MapAggregate)
        and (item.parent not in expansions or expansions[item.parent].status == "pending")
        for item in declaration.aggregates
    )
    return completion_reserve(
        absent=absent, unstarted_or_ready=waiting, running_ordinary=running, open_map_expanders=open_maps
    )


def _complete(entries: Mapping[str, Entry], expansions: Mapping[str, Expansion], declaration: Declaration) -> bool:
    required = set(declaration.required)
    for expansion in expansions.values():
        required.update(expansion.members)
    for choice in declaration.choices:
        selector = entries.get(choice.selector)
        if selector and selector.outcome:
            required.update(
                member for outcome, members in choice.branches if outcome == selector.outcome for member in members
            )
    for subgraph in declaration.subgraphs:
        if subgraph.parent in entries and entries[subgraph.parent].status in ("running", *CATEGORIES):
            required.update((*subgraph.roots, subgraph.sink))
    return (
        required <= entries.keys()
        and all(entries[key].status in TERMINAL for key in required)
        and all(
            expansion.status != "pending"
            and all(member in entries and entries[member].status in TERMINAL for member in expansion.members)
            for expansion in expansions.values()
        )
        and all(
            (item.parent if isinstance(item, MapAggregate) else item.starter) in expansions
            and expansions[item.parent if isinstance(item, MapAggregate) else item.starter].status != "pending"
            and item.join in entries
            and entries[item.join].status in TERMINAL
            for item in declaration.aggregates
            if (item.parent if isinstance(item, MapAggregate) else item.starter) in entries
        )
    )
