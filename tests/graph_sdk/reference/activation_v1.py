# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite, independent, typed event reducer for workflow activation v1."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]
Object: TypeAlias = dict[str, Json]
Category: TypeAlias = Literal["success", "failure", "cancelled", "lost", "blocked", "inconsistent"]
Status: TypeAlias = Literal[
    "unstarted", "ready", "running", "success", "failure", "cancelled", "lost", "blocked", "inconsistent"
]
Role: TypeAlias = Literal["ordinary", "subgraph", "map_expander", "map_member", "join", "loop_starter", "loop_member"]
Code: TypeAlias = Literal[
    "invalid_type",
    "invalid_value",
    "limit_exceeded",
    "foreign_owner",
    "duplicate",
    "missing",
    "overlap",
    "cycle",
    "contradictory",
]

CONTRACT_SHA256 = "9f58d60ad4ecc25065cc6c74d784cd05ad6fb2a72f183a615122eb981c7b265b"
CORPUS_PATH = "tests/graph_sdk/reference/activation_v1_cases.json"
GENERATOR_VERSION = "workflow-activation-v1-generator-3"
SELF_TEST_VERSION = "workflow-activation-v1-self-test-3"
TEMPLATES = ("N0", "N1", "N2")
INVOCATIONS = ("I0", "I1")
ACTIVATIONS = tuple(f"A{i}" for i in range(12))
CATEGORIES: tuple[Category, ...] = ("success", "failure", "cancelled", "lost", "blocked", "inconsistent")
TERMINAL = frozenset(CATEGORIES)
ALPHABET = (
    "initialize",
    "select",
    "start",
    "terminal_success",
    "terminal_failure",
    "terminal_cancelled",
    "terminal_lost",
    "terminal_blocked",
    "terminal_inconsistent",
    "close_blocked",
    "close_inconsistent",
    "membership_open",
    "membership_close",
    "membership_overflow",
)
FAMILY_IDS = (
    "sequence_single",
    "sequence_linked_pair",
    "sequence_independent_siblings",
    "sequence_mutations",
    "choice",
    "subgraph",
    "map",
    "join",
    "loop",
    "nested_map_loop",
    "precedence",
    "terminal_coverage",
)
RULE_IDS = (
    "different_parent_activations",
    "same_parent_unrelated_siblings",
    "no_sequence_choice_membership_loop_join_dependency",
)
ERROR_ORDER: tuple[Code, ...] = (
    "invalid_type",
    "invalid_value",
    "limit_exceeded",
    "foreign_owner",
    "duplicate",
    "missing",
    "overlap",
    "cycle",
    "contradictory",
)
RENAME_TEMPLATE = {"N0": "N2", "N1": "N0", "N2": "N1"}
INVERSE_TEMPLATE = {value: key for key, value in RENAME_TEMPLATE.items()}
RENAME_ACTIVATION = {f"A{i}": f"A{11 - i}" for i in range(12)}
INVERSE_ACTIVATION = {value: key for key, value in RENAME_ACTIVATION.items()}


@dataclass(frozen=True, slots=True)
class Limits:
    max_events: int
    max_entries: int
    max_parent_depth: int


@dataclass(frozen=True, slots=True)
class Seed:
    key: str
    template: str
    invocation: str
    parent: str | None
    iteration: int | None
    role: Role


@dataclass(frozen=True, slots=True)
class Outcome:
    template: str
    name: str
    category: Category
    produced_ports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Choice:
    selector: str
    branches: tuple[tuple[str, tuple[str, ...]], ...]


@dataclass(frozen=True, slots=True)
class Subgraph:
    parent: str
    roots: tuple[str, ...]
    sink: str


@dataclass(frozen=True, slots=True)
class MapAggregate:
    parent: str
    members: tuple[str, ...]
    join: str
    bound: int
    expansion_outcomes: tuple[str, ...]
    accepted_categories: tuple[Category, ...]


@dataclass(frozen=True, slots=True)
class LoopAggregate:
    starter: str
    members: tuple[str, ...]
    join: str
    bound: int
    enter_outcomes: tuple[str, ...]
    bypass_outcomes: tuple[str, ...]
    continue_outcomes: tuple[str, ...]
    exit_outcomes: tuple[str, ...]
    initial: bool
    carried: bool


Aggregate: TypeAlias = MapAggregate | LoopAggregate


@dataclass(frozen=True, slots=True)
class Declaration:
    invocation: str
    seeds: tuple[Seed, ...]
    required: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    outcomes: tuple[Outcome, ...]
    choices: tuple[Choice, ...]
    subgraphs: tuple[Subgraph, ...]
    aggregates: tuple[Aggregate, ...]
    limits: Limits
    static_groups: tuple[tuple[str, ...], ...]
    static_edges: tuple[tuple[str, str], ...]
    compatible: bool


@dataclass(frozen=True, slots=True)
class Initialize: ...


@dataclass(frozen=True, slots=True)
class Select:
    keys: tuple[str, ...]
    invocation: str


@dataclass(frozen=True, slots=True)
class Start:
    key: str
    invocation: str


@dataclass(frozen=True, slots=True)
class TerminalEvent:
    key: str
    invocation: str
    category: Category
    outcome: str | None


@dataclass(frozen=True, slots=True)
class Close:
    key: str
    invocation: str
    category: Literal["blocked", "inconsistent"]


@dataclass(frozen=True, slots=True)
class Membership:
    parent: str
    invocation: str
    members: tuple[str, ...]
    closed: bool


@dataclass(frozen=True, slots=True)
class Overflow:
    parent: str
    invocation: str
    count: int


Event: TypeAlias = Initialize | Select | Start | TerminalEvent | Close | Membership | Overflow


@dataclass(frozen=True, slots=True)
class Entry:
    activation: str
    template: str
    status: Status
    outcome: str | None
    category: Category | None
    produced_ports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Expansion:
    parent: str
    status: Literal["pending", "closed", "failed", "overflow"]
    members: frozenset[str]


@dataclass(frozen=True, slots=True)
class ReferenceState:
    entries: frozenset[Entry]
    expansions: frozenset[Expansion]
    outputs: frozenset[str]
    events_applied: int
    complete: bool


class Rejected(ValueError):
    def __init__(self, *codes: Code) -> None:
        pool = frozenset(codes)
        self.code = next(code for code in ERROR_ORDER if code in pool)
        super().__init__(self.code)


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def canonical_bytes(cases: Sequence[Object]) -> bytes:
    return _canonical(cast(Json, list(cases))) + b"\n"


def initial_capacity(reservation_count: int, map_expander_count: int = 0) -> tuple[int, int]:
    if isinstance(reservation_count, bool) or isinstance(map_expander_count, bool):
        raise TypeError("invalid capacity type")
    if reservation_count < 0 or map_expander_count < 0 or map_expander_count > reservation_count:
        raise ValueError("invalid capacity value")
    return reservation_count, 3 * reservation_count + map_expander_count


def completion_reserve(
    *, absent: int = 0, unstarted_or_ready: int = 0, running_ordinary: int = 0, open_map_expanders: int = 0
) -> int:
    values = absent, unstarted_or_ready, running_ordinary, open_map_expanders
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("invalid reserve type")
    if any(value < 0 for value in values):
        raise ValueError("invalid reserve value")
    return 3 * absent + 2 * unstarted_or_ready + running_ordinary + open_map_expanders


def _obj(value: Json, keys: set[str]) -> Object:
    if not isinstance(value, dict):
        raise Rejected("invalid_type")
    if set(value) != keys:
        raise Rejected("missing")
    return value


def _list(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise Rejected("invalid_type")
    return value


def _strs(value: Json) -> tuple[str, ...]:
    values = _list(value)
    if any(not isinstance(item, str) for item in values):
        raise Rejected("invalid_type")
    return tuple(cast(str, item) for item in values)


def _int(value: Json) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Rejected("invalid_type")
    if value < 0:
        raise Rejected("invalid_value")
    return value


def _closed(value: Json, values: Sequence[str]) -> str:
    if not isinstance(value, str):
        raise Rejected("invalid_type")
    if value not in values:
        raise Rejected("invalid_value")
    return value


def parse_declaration(value: Mapping[str, Json]) -> Declaration:
    """Parse the JSON boundary once into immutable closed facts."""
    raw = _obj(
        dict(value),
        {
            "invocation",
            "seeds",
            "required",
            "edges",
            "outcomes",
            "choices",
            "subgraphs",
            "aggregates",
            "limits",
            "static",
        },
    )
    seeds: list[Seed] = []
    for item in _list(raw["seeds"]):
        seed = _obj(item, {"key", "template", "invocation", "parent", "iteration", "role"})
        parent, iteration = seed["parent"], seed["iteration"]
        if parent is not None and not isinstance(parent, str):
            raise Rejected("invalid_type")
        seeds.append(
            Seed(
                _closed(seed["key"], ACTIVATIONS),
                _closed(seed["template"], TEMPLATES),
                _closed(seed["invocation"], INVOCATIONS),
                parent,
                None if iteration is None else _int(iteration),
                cast(
                    Role,
                    _closed(
                        seed["role"],
                        ("ordinary", "subgraph", "map_expander", "map_member", "join", "loop_starter", "loop_member"),
                    ),
                ),
            )
        )
    choices: list[Choice] = []
    for item in _list(raw["choices"]):
        choice = _obj(item, {"selector", "branches"})
        branches = tuple(
            (_closed(branch["outcome"], ("ok", "fail")), _strs(branch["members"]))
            for value_branch in _list(choice["branches"])
            for branch in [_obj(value_branch, {"outcome", "members"})]
        )
        choices.append(Choice(_closed(choice["selector"], ACTIVATIONS), branches))
    subgraphs = tuple(
        Subgraph(_closed(item["parent"], ACTIVATIONS), _strs(item["roots"]), _closed(item["sink"], ACTIVATIONS))
        for value_item in _list(raw["subgraphs"])
        for item in [_obj(value_item, {"parent", "roots", "sink"})]
    )
    outcomes: list[Outcome] = []
    for item in _list(raw["outcomes"]):
        outcome = _obj(item, {"template", "name", "category", "produced_ports"})
        name = outcome["name"]
        if not isinstance(name, str):
            raise Rejected("invalid_type")
        if not name:
            raise Rejected("invalid_value")
        ports = _strs(outcome["produced_ports"])
        outcomes.append(
            Outcome(
                _closed(outcome["template"], TEMPLATES),
                name,
                cast(Category, _closed(outcome["category"], CATEGORIES)),
                ports,
            )
        )
    aggregates: list[Aggregate] = []
    for item in _list(raw["aggregates"]):
        if not isinstance(item, dict):
            raise Rejected("invalid_type")
        kind = _closed(item.get("kind"), ("map", "loop"))
        if kind == "map":
            aggregate = _obj(
                item,
                {"parent", "members", "join", "bound", "kind", "expansion_outcomes", "accepted_categories"},
            )
            aggregates.append(
                MapAggregate(
                    _closed(aggregate["parent"], ACTIVATIONS),
                    _strs(aggregate["members"]),
                    _closed(aggregate["join"], ACTIVATIONS),
                    _int(aggregate["bound"]),
                    _strs(aggregate["expansion_outcomes"]),
                    tuple(
                        cast(Category, _closed(category, CATEGORIES))
                        for category in _strs(aggregate["accepted_categories"])
                    ),
                )
            )
        else:
            aggregate = _obj(
                item,
                {
                    "starter",
                    "members",
                    "join",
                    "bound",
                    "kind",
                    "enter_outcomes",
                    "bypass_outcomes",
                    "continue_outcomes",
                    "exit_outcomes",
                    "initial",
                    "carried",
                },
            )
            if not isinstance(aggregate["initial"], bool) or not isinstance(aggregate["carried"], bool):
                raise Rejected("invalid_type")
            aggregates.append(
                LoopAggregate(
                    _closed(aggregate["starter"], ACTIVATIONS),
                    _strs(aggregate["members"]),
                    _closed(aggregate["join"], ACTIVATIONS),
                    _int(aggregate["bound"]),
                    _strs(aggregate["enter_outcomes"]),
                    _strs(aggregate["bypass_outcomes"]),
                    _strs(aggregate["continue_outcomes"]),
                    _strs(aggregate["exit_outcomes"]),
                    aggregate["initial"],
                    aggregate["carried"],
                )
            )
    limits = _obj(raw["limits"], {"max_events", "max_entries", "max_parent_depth"})
    static = _obj(raw["static"], {"groups", "edges", "compatible"})
    if not isinstance(static["compatible"], bool):
        raise Rejected("invalid_type")
    edges = tuple(_strs(edge) for edge in _list(raw["edges"]))
    if any(len(edge) != 2 for edge in edges):
        raise Rejected("invalid_value")
    return Declaration(
        _closed(raw["invocation"], INVOCATIONS),
        tuple(seeds),
        _strs(raw["required"]),
        cast(tuple[tuple[str, str], ...], edges),
        tuple(outcomes),
        tuple(choices),
        subgraphs,
        tuple(aggregates),
        Limits(_int(limits["max_events"]), _int(limits["max_entries"]), _int(limits["max_parent_depth"])),
        tuple(_strs(group) for group in _list(static["groups"])),
        tuple(cast(tuple[str, str], _strs(edge)) for edge in _list(static["edges"])),
        static["compatible"],
    )


def parse_event(value: Mapping[str, Json]) -> Event:
    raw = dict(value)
    kind = raw.get("kind")
    if not isinstance(kind, str):
        raise Rejected("invalid_type")
    if kind == "initialize":
        _obj(raw, {"kind"})
        return Initialize()
    if kind == "select":
        item = _obj(raw, {"kind", "keys", "invocation"})
        return Select(_strs(item["keys"]), _closed(item["invocation"], INVOCATIONS))
    if kind == "start":
        item = _obj(raw, {"kind", "key", "invocation"})
        return Start(_closed(item["key"], ACTIVATIONS), _closed(item["invocation"], INVOCATIONS))
    if kind.startswith("terminal_"):
        item = _obj(raw, {"kind", "key", "invocation", "outcome"})
        category = cast(Category, _closed(kind.removeprefix("terminal_"), CATEGORIES))
        outcome = item["outcome"]
        if outcome is not None and not isinstance(outcome, str):
            raise Rejected("invalid_type")
        if outcome is None and category == "success":
            raise Rejected("invalid_value")
        if isinstance(outcome, str) and not outcome:
            raise Rejected("invalid_value")
        return TerminalEvent(
            _closed(item["key"], ACTIVATIONS), _closed(item["invocation"], INVOCATIONS), category, outcome
        )
    if kind.startswith("close_"):
        item = _obj(raw, {"kind", "key", "invocation"})
        category = kind.removeprefix("close_")
        return Close(
            _closed(item["key"], ACTIVATIONS),
            _closed(item["invocation"], INVOCATIONS),
            cast(Literal["blocked", "inconsistent"], _closed(category, ("blocked", "inconsistent"))),
        )
    if kind in ("membership_open", "membership_close"):
        item = _obj(raw, {"kind", "parent", "invocation", "members"})
        return Membership(
            _closed(item["parent"], ACTIVATIONS),
            _closed(item["invocation"], INVOCATIONS),
            _strs(item["members"]),
            kind == "membership_close",
        )
    if kind == "membership_overflow":
        item = _obj(raw, {"kind", "parent", "invocation", "observed_count"})
        return Overflow(
            _closed(item["parent"], ACTIVATIONS),
            _closed(item["invocation"], INVOCATIONS),
            _int(item["observed_count"]),
        )
    raise Rejected("invalid_value")


def _depth(key: str, seeds: Mapping[str, Seed]) -> int:
    seen: set[str] = set()
    depth = 0
    current: str | None = key
    while current is not None:
        if current in seen:
            raise Rejected("cycle")
        seen.add(current)
        seed = seeds.get(current)
        if seed is None:
            raise Rejected("missing")
        depth += 1
        current = seed.parent
    return depth


def _initialize(declaration: Declaration) -> None:
    seen: dict[str, Seed] = {}
    duplicate = False
    for seed in declaration.seeds:
        duplicate |= seed.key in seen
        seen.setdefault(seed.key, seed)
    foreign = any(seed.invocation != declaration.invocation for seed in declaration.seeds)
    missing = not set(declaration.required) <= seen.keys()
    duplicate |= len(declaration.required) != len(set(declaration.required))
    duplicate |= len(declaration.edges) != len(set(declaration.edges))
    duplicate |= len(declaration.outcomes) != len(
        {(outcome.template, outcome.name) for outcome in declaration.outcomes}
    )
    duplicate |= any(
        len(outcome.produced_ports) != len(set(outcome.produced_ports)) for outcome in declaration.outcomes
    )
    map_count = sum(seed.role == "map_expander" for seed in declaration.seeds)
    limit = (
        declaration.limits.max_entries < len(declaration.seeds)
        or declaration.limits.max_events < 3 * len(declaration.seeds) + map_count
    )
    maximum_depth = 0
    depth_code: Code | None = None
    try:
        maximum_depth = max((_depth(key, seen) for key in seen), default=0)
    except Rejected as error:
        depth_code = error.code
    limit |= declaration.limits.max_parent_depth < maximum_depth
    missing |= depth_code == "missing"
    cycle_from_parents = depth_code == "cycle"
    contradictory = False
    dynamic_overlap = False
    outcome_names = {(outcome.template, outcome.name) for outcome in declaration.outcomes}
    missing |= any(before not in seen or after not in seen for before, after in declaration.edges)
    duplicate |= any(before == after for before, after in declaration.edges)
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
            selector and any((selector.template, outcome) not in outcome_names for outcome, _ in choice.branches)
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
                    (parent_seed.template, outcome) not in outcome_names for outcome in aggregate.expansion_outcomes
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
                    (parent_seed.template, outcome) not in outcome_names
                    for outcome in (*aggregate.enter_outcomes, *aggregate.bypass_outcomes)
                )
            )
            for member in aggregate.members:
                member_seed = seen.get(member)
                missing |= bool(
                    member_seed
                    and any(
                        (member_seed.template, outcome) not in outcome_names
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
                blocked = any(before not in outputs for before in predecessors)
                entries[key] = replace(
                    entry, status="blocked" if blocked else "ready", category="blocked" if blocked else None
                )
                changed = True
        for subgraph in declaration.subgraphs:
            parent, sink = entries.get(subgraph.parent), entries.get(subgraph.sink)
            if parent and sink and parent.status == "running" and sink.status in TERMINAL:
                projected = next(
                    (
                        outcome
                        for outcome in declaration.outcomes
                        if outcome.template == parent.template and outcome.name == sink.outcome
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
                if join.status in ("unstarted", "ready") and all(
                    child and child.status in TERMINAL for child in children
                ):
                    accepted_categories = (
                        aggregate.accepted_categories if isinstance(aggregate, MapAggregate) else ("success",)
                    )
                    accepted = all(child and child.category in accepted_categories for child in children)
                    entries[aggregate.join] = replace(
                        join, status="ready" if accepted else "blocked", category=None if accepted else "blocked"
                    )


def _materialize(key: str, entries: dict[str, Entry], declaration: Declaration) -> None:
    seed = next((item for item in declaration.seeds if item.key == key), None)
    if seed is None:
        raise Rejected("missing")
    if key in entries:
        raise Rejected("duplicate")
    entries[key] = Entry(key, seed.template, "unstarted", None, None, ())


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
            for root in subgraph.roots:
                _materialize(root, entries, declaration)
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
                if outcome.template == seed.template and outcome.name == event.outcome
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
                elif event.outcome in aggregate.enter_outcomes and not aggregate.initial:
                    expansions[event.key] = Expansion(event.key, "failed", frozenset())
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
                elif event.outcome in aggregate.continue_outcomes and not aggregate.carried:
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
        expansions[event.parent] = Expansion(event.parent, "overflow", frozenset())
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


def _state_object(state: ReferenceState, *, digest: bool = True) -> Object:
    result: Object = {
        "complete": state.complete,
        "entries": [
            {
                "activation": item.activation,
                "category": item.category,
                "outcome": item.outcome,
                "produced_ports": list(item.produced_ports),
                "status": item.status,
                "template": item.template,
            }
            for item in sorted(state.entries, key=lambda item: item.activation)
        ],
        "events_applied": state.events_applied,
        "expansions": [
            {"members": sorted(item.members), "parent": item.parent, "status": item.status}
            for item in sorted(state.expansions, key=lambda item: item.parent)
        ],
        "outputs": sorted(state.outputs),
    }
    if digest:
        result["semantic_hash"] = hashlib.sha256(_canonical(cast(Json, _state_object(state, digest=False)))).hexdigest()
    return result


def alpha_normalize_state(value: Mapping[str, Json], *, inverse: bool = False) -> Object:
    """Rename a state and recompute its deterministic whole-state digest."""
    renamed = cast(
        Object, _rename({key: item for key, item in value.items() if key != "semantic_hash"}, inverse=inverse)
    )
    renamed["entries"] = sorted(
        _list(renamed["entries"]),
        key=lambda item: cast(
            str,
            _obj(item, {"activation", "category", "outcome", "produced_ports", "status", "template"})["activation"],
        ),
    )
    renamed["outputs"] = sorted(_strs(renamed["outputs"]))
    expansions = _list(renamed["expansions"])
    for expansion_value in expansions:
        expansion = _obj(expansion_value, {"members", "parent", "status"})
        expansion["members"] = sorted(_strs(expansion["members"]))
    renamed["expansions"] = sorted(
        expansions, key=lambda item: cast(str, _obj(item, {"members", "parent", "status"})["parent"])
    )
    renamed["semantic_hash"] = hashlib.sha256(_canonical(renamed)).hexdigest()
    return renamed


def reduce_trace(declaration: Mapping[str, Json], events: Sequence[Mapping[str, Json]]) -> Object:
    try:
        facts = parse_declaration(declaration)
        if not events:
            raise Rejected("missing")
        first = parse_event(events[0])
        if not isinstance(first, Initialize):
            raise Rejected("missing")
        _initialize(facts)
        entries: dict[str, Entry] = {}
        expansions: dict[str, Expansion] = {}
        outputs: set[str] = set()
        applied = 0
        for raw_event in events[1:]:
            if applied + 1 > facts.limits.max_events:
                raise Rejected("limit_exceeded")
            event = parse_event(raw_event)
            next_entries, next_expansions, next_outputs = dict(entries), dict(expansions), set(outputs)
            _apply(event, next_entries, next_expansions, next_outputs, facts)
            next_applied = applied + 1
            if (
                len(next_entries) > facts.limits.max_entries
                or next_applied + _reserve(next_entries, next_expansions, facts) > facts.limits.max_events
            ):
                raise Rejected("limit_exceeded")
            entries, expansions, outputs, applied = next_entries, next_expansions, next_outputs, next_applied
        state = ReferenceState(
            frozenset(entries.values()),
            frozenset(expansions.values()),
            frozenset(outputs),
            applied,
            _complete(entries, expansions, facts),
        )
        return {"code": None, "state": _state_object(state), "status": "accepted"}
    except Rejected as error:
        return {"code": error.code, "state": None, "status": "rejected"}


def _seed(
    key: str,
    template: str = "N0",
    *,
    parent: str | None = None,
    iteration: int | None = None,
    role: Role = "ordinary",
    invocation: str = "I0",
) -> Object:
    return {
        "invocation": invocation,
        "iteration": iteration,
        "key": key,
        "parent": parent,
        "role": role,
        "template": template,
    }


def _decl(
    seeds: Sequence[Object],
    *,
    required: Sequence[str] = (),
    edges: Sequence[Sequence[str]] = (),
    choices: Sequence[Mapping[str, Json]] = (),
    subgraphs: Sequence[Mapping[str, Json]] = (),
    aggregates: Sequence[Mapping[str, Json]] = (),
    outcomes: Sequence[Mapping[str, Json]] | None = None,
    limits: tuple[int, int, int] | None = None,
    static: Object | None = None,
) -> Object:
    parents = {cast(str, seed["key"]): cast(str | None, seed["parent"]) for seed in seeds}
    depth = 0
    for key in parents:
        count = 0
        current: str | None = key
        while current in parents:
            count += 1
            current = parents[current]
        depth = max(depth, count)
    maps = sum(seed["role"] == "map_expander" for seed in seeds)
    exact = limits or (3 * len(seeds) + maps, len(seeds), depth)
    default_outcomes: tuple[Object, ...] = tuple(
        {
            "category": category,
            "name": name,
            "produced_ports": list(ports),
            "template": template,
        }
        for template in TEMPLATES
        for name, category, ports in (
            ("ok", "success", ("result",)),
            ("fail", "failure", ()),
            ("again", "success", ("carry",)),
            ("stop", "success", ("result",)),
        )
    )
    return {
        "aggregates": [dict(item) for item in aggregates],
        "choices": [dict(item) for item in choices],
        "edges": [list(edge) for edge in edges],
        "invocation": "I0",
        "limits": {"max_entries": exact[1], "max_events": exact[0], "max_parent_depth": exact[2]},
        "outcomes": list(outcomes if outcomes is not None else default_outcomes),
        "required": list(required),
        "seeds": list(seeds),
        "static": static or {"compatible": True, "edges": [], "groups": []},
        "subgraphs": [dict(item) for item in subgraphs],
    }


def _event(kind: str, **fields: Json) -> Object:
    event: Object = {"kind": kind, **fields}
    if kind != "initialize" and "invocation" not in event:
        event["invocation"] = "I0"
    return event


def _select(*keys: str) -> Object:
    return _event("select", keys=list(keys))


def _terminal(key: str, category: Category, outcome: str | None) -> Object:
    return _event(f"terminal_{category}", key=key, outcome=outcome)


def _spare(declaration: Object, amount: int = 1) -> Object:
    result = dict(declaration)
    limits = cast(Object, dict(cast(Object, declaration["limits"])))
    limits["max_events"] = cast(int, limits["max_events"]) + amount
    result["limits"] = limits
    return result


def _ordinary(categories: Sequence[Category], *, keys: Sequence[str] | None = None) -> list[Object]:
    selected = tuple(keys or (f"A{i}" for i in range(len(categories))))
    events = [_event("initialize"), _select(*selected)]
    for key, category in zip(selected, categories, strict=True):
        events += [_event("start", key=key), _terminal(key, category, "ok" if category == "success" else None)]
    return events


def _rename(value: Json, *, inverse: bool = False) -> Json:
    templates, activations = (INVERSE_TEMPLATE, INVERSE_ACTIVATION) if inverse else (RENAME_TEMPLATE, RENAME_ACTIVATION)
    if isinstance(value, list):
        return [_rename(item, inverse=inverse) for item in value]
    if isinstance(value, dict):
        return {
            templates.get(key, activations.get(key, key)): _rename(item, inverse=inverse) for key, item in value.items()
        }
    if isinstance(value, str):
        return templates.get(value, activations.get(value, value))
    return value


def _case(
    family: str,
    coordinate: str,
    name: str,
    declaration: Object,
    events: Sequence[Mapping[str, Json]],
    traces: Sequence[str] = (),
) -> Object:
    event_list = [dict(event) for event in events]
    expected = reduce_trace(declaration, event_list)
    trace_values: list[Json] = []
    for trace in traces:
        if trace == "rename":
            transformed = cast(Object, _rename(declaration))
            transformed_events = cast(list[Object], _rename(cast(Json, event_list)))
        else:
            transformed = declaration
            transformed_events = event_list[:2] + event_list[4:6] + event_list[2:4]
        trace_values.append(
            {
                "declaration": transformed,
                "events": transformed_events,
                "expected": reduce_trace(transformed, transformed_events),
                "name": trace,
            }
        )
    return {
        "case_id": f"{family}/{coordinate}/{name}",
        "declaration": declaration,
        "events": event_list,
        "expected": expected,
        "family": family,
        "mode": "accepted" if expected["status"] == "accepted" else "rejected",
        "traces": trace_values,
    }


def _sequence_cases() -> Iterable[Object]:
    for i, category in enumerate(CATEGORIES):
        declaration = _decl((_seed("A0"),), required=("A0",))
        yield _case("sequence_single", f"{i:03d}", "base", declaration, _ordinary((category,)), ("rename",))
    for i, left in enumerate(CATEGORIES):
        for j, right in enumerate(CATEGORIES):
            declaration = _decl((_seed("A0", "N0"), _seed("A1", "N1")), required=("A0", "A1"), edges=(("A0", "A1"),))
            events = [
                _event("initialize"),
                _select("A0", "A1"),
                _event("start", key="A0"),
                _terminal("A0", left, "ok" if left == "success" else "fail" if left == "failure" else None),
                _event("start", key="A1"),
                _terminal("A1", right, "ok" if right == "success" else "fail" if right == "failure" else None),
            ]
            yield _case("sequence_linked_pair", f"{i:03d}-{j:03d}", "base", declaration, events, ("rename",))
    for i, left in enumerate(CATEGORIES):
        for j, right in enumerate(CATEGORIES):
            declaration = _decl((_seed("A0", "N0"), _seed("A1", "N1")), required=("A0", "A1"))
            events = _ordinary((left, right))
            yield _case(
                "sequence_independent_siblings",
                f"{i:03d}-{j:03d}",
                "base",
                declaration,
                events,
                ("rename", "commute_independent_siblings"),
            )


def _mutation_cases() -> Iterable[Object]:
    declaration = _decl(
        (_seed("A0"),),
        required=("A0",),
        outcomes=({"category": "success", "name": "ok", "produced_ports": ["result"], "template": "N0"},),
    )
    events = (
        [_event("initialize"), _event("start", key="A0")],
        [_event("initialize"), _select("A0"), _terminal("A0", "success", "ok")],
        _ordinary(("success",)) + [_terminal("A0", "success", "ok")],
        [_event("initialize"), _select("A1")],
    )
    names = ("start_before_ready", "terminal_before_start", "duplicate_terminal", "missing_activation")
    for i, (name, trace) in enumerate(zip(names, events, strict=True)):
        facts = _spare(declaration) if name == "duplicate_terminal" else declaration
        yield _case("sequence_mutations", f"{i:03d}", name, facts, trace)
    for i, category in enumerate(CATEGORIES[1:], 4):
        yield _case(
            "sequence_mutations",
            f"{i:03d}",
            f"abnormal_{category}",
            declaration,
            [_event("initialize"), _select("A0"), _event("start", key="A0"), _terminal("A0", category, None)],
        )
    yield _case(
        "sequence_mutations",
        "009",
        "success_without_outcome",
        declaration,
        [_event("initialize"), _select("A0"), _event("start", key="A0"), _terminal("A0", "success", None)],
    )


def _choice_cases() -> Iterable[Object]:
    coordinate = 0
    seeds = (_seed("A0", "N0"), _seed("A1", "N1"), _seed("A2", "N2"))
    for outcome in ("ok", "fail"):
        for order in ("forward", "reverse"):
            branches = [{"members": ["A1"], "outcome": "ok"}, {"members": ["A2"], "outcome": "fail"}]
            if order == "reverse":
                branches.reverse()
            declaration = _decl(seeds, required=("A0",), choices=({"branches": branches, "selector": "A0"},))
            branch = "A1" if outcome == "ok" else "A2"
            category: Category = "success" if outcome == "ok" else "failure"
            events = [
                _event("initialize"),
                _select("A0"),
                _event("start", key="A0"),
                _terminal("A0", category, outcome),
                _event("start", key=branch),
                _terminal(branch, category, outcome),
            ]
            yield _case("choice", f"{coordinate:03d}", f"{outcome}_{order}", declaration, events)
            coordinate += 1
    base = _decl(
        seeds, required=("A0",), choices=({"branches": [{"members": ["A1"], "outcome": "ok"}], "selector": "A0"},)
    )
    special = (
        (
            "foreign_selector",
            _decl(
                (*seeds, _seed("A11", invocation="I1")),
                required=("A0",),
                choices=({"branches": [], "selector": "A11"},),
            ),
            [_event("initialize"), _select("A0")],
        ),
        (
            "unknown_outcome",
            base,
            [_event("initialize"), _select("A0"), _event("start", key="A0"), _terminal("A0", "success", "other")],
        ),
        ("select_both", base, [_event("initialize"), _select("A1", "A2")]),
        (
            "abnormal_selector_failure",
            base,
            [_event("initialize"), _select("A0"), _event("start", key="A0"), _terminal("A0", "failure", None)],
        ),
    )
    for name, declaration, events in special:
        yield _case("choice", f"{coordinate:03d}", name, declaration, events)
        coordinate += 1


def _subgraph_cases() -> Iterable[Object]:
    coordinate = 0
    for size in (1, 2):
        for category in ("success", "failure"):
            keys = tuple(f"A{i + 1}" for i in range(size))
            seeds = [_seed("A0", role="subgraph")] + [_seed(key, "N1", parent="A0") for key in keys]
            declaration = _decl(
                seeds,
                required=("A0",),
                edges=tuple((keys[i], keys[i + 1]) for i in range(size - 1)),
                subgraphs=({"parent": "A0", "roots": list(keys), "sink": keys[-1]},),
            )
            events = [_event("initialize"), _select("A0"), _event("start", key="A0")]
            for index, key in enumerate(keys):
                observed: Category = cast(Category, category) if index + 1 == len(keys) else "success"
                events += [
                    _event("start", key=key),
                    _terminal(key, observed, "ok" if observed == "success" else "fail"),
                ]
            yield _case("subgraph", f"{coordinate:03d}", f"body_{size}_{category}", declaration, events)
            coordinate += 1
    base = _decl(
        (_seed("A0", role="subgraph"), _seed("A1", "N1", parent="A0")),
        required=("A0",),
        subgraphs=({"parent": "A0", "roots": ["A1"], "sink": "A1"},),
    )
    cases = (
        (
            "premature_parent_terminal",
            base,
            [_event("initialize"), _select("A0"), _event("start", key="A0"), _terminal("A0", "success", "ok")],
        ),
        (
            "wrong_parent",
            _decl((_seed("A0", role="subgraph"), _seed("A1", parent="A2")), required=("A0",)),
            [_event("initialize")],
        ),
        (
            "foreign_body_key",
            _decl((_seed("A0", role="subgraph"), _seed("A1", parent="A0", invocation="I1")), required=("A0",)),
            [_event("initialize")],
        ),
        (
            "duplicate_body_key",
            _decl(
                (_seed("A0", role="subgraph"), _seed("A1", parent="A0"), _seed("A1", "N1", parent="A0")),
                required=("A0",),
            ),
            [_event("initialize")],
        ),
    )
    for name, declaration, events in cases:
        yield _case("subgraph", f"{coordinate:03d}", name, declaration, events)
        coordinate += 1
    nested = _decl(
        (_seed("A0", role="subgraph"), _seed("A1", "N1", parent="A0", role="subgraph"), _seed("A2", "N2", parent="A1")),
        required=("A0",),
        subgraphs=({"parent": "A0", "roots": ["A1"], "sink": "A1"}, {"parent": "A1", "roots": ["A2"], "sink": "A2"}),
    )
    yield _case(
        "subgraph",
        f"{coordinate:03d}",
        "nested_body",
        nested,
        [
            _event("initialize"),
            _select("A0"),
            _event("start", key="A0"),
            _event("start", key="A1"),
            _event("start", key="A2"),
            _terminal("A2", "success", "ok"),
        ],
    )
    coordinate += 1
    yield _case(
        "subgraph",
        f"{coordinate:03d}",
        "abnormal_sink_loss",
        base,
        [
            _event("initialize"),
            _select("A0"),
            _event("start", key="A0"),
            _event("start", key="A1"),
            _terminal("A1", "lost", None),
        ],
    )


def _aggregate_decl(
    bound: int, *, kind: Literal["map", "loop"] = "map", initial: bool = True, carried: bool = True
) -> Object:
    count = bound
    parent_role: Role = "loop_starter" if kind == "loop" else "map_expander"
    member_role: Role = "loop_member" if kind == "loop" else "map_member"
    seeds = [_seed("A0", role=parent_role), _seed("A11", "N2", role="join")] + [
        _seed(f"A{i + 1}", "N1", parent="A0", iteration=i if kind == "loop" else None, role=member_role)
        for i in range(count)
    ]
    if kind == "map":
        aggregate: Object = {
            "accepted_categories": ["success"],
            "bound": bound,
            "expansion_outcomes": ["ok"],
            "join": "A11",
            "kind": "map",
            "members": [f"A{i + 1}" for i in range(count)],
            "parent": "A0",
        }
    else:
        aggregate = {
            "bound": bound,
            "bypass_outcomes": ["stop"],
            "carried": carried,
            "continue_outcomes": ["again"],
            "enter_outcomes": ["again"],
            "exit_outcomes": ["stop"],
            "initial": initial,
            "join": "A11",
            "kind": "loop",
            "members": [f"A{i + 1}" for i in range(count)],
            "starter": "A0",
        }
    return _decl(seeds, required=("A0", "A11"), aggregates=(aggregate,))


def _map_events(
    members: Sequence[str],
    categories: Sequence[Category],
    *,
    closed: bool = True,
    parent_category: Category = "success",
    parent_outcome: str | None = "ok",
) -> list[Object]:
    events = [
        _event("initialize"),
        _select("A0", "A11"),
        _event("start", key="A0"),
        _terminal("A0", parent_category, parent_outcome),
        _event("membership_close" if closed else "membership_open", parent="A0", members=list(members)),
    ]
    for member, category in zip(members, categories, strict=False):
        outcome = "ok" if category == "success" else "fail" if category == "failure" else None
        events += [_event("start", key=member), _terminal(member, category, outcome)]
    return events


def _map_cases() -> Iterable[Object]:
    coordinate = 0
    for bound in (0, 1, 2):
        for size in range(bound + 1):
            for assignment in itertools.product(("success", "failure"), repeat=size):
                members = tuple(f"A{i + 1}" for i in range(size))
                yield _case(
                    "map",
                    f"{coordinate:03d}",
                    f"bound_{bound}_size_{size}_{''.join(x[0] for x in assignment) or 'empty'}",
                    _aggregate_decl(bound),
                    _map_events(members, cast(tuple[Category, ...], assignment)),
                )
                coordinate += 1
    declaration = _aggregate_decl(2)
    yield _case(
        "map",
        f"{coordinate:03d}",
        "one_over_3",
        declaration,
        [
            _event("initialize"),
            _select("A0", "A11"),
            _event("start", key="A0"),
            _terminal("A0", "success", "ok"),
            _event("membership_overflow", parent="A0", observed_count=3),
        ],
    )
    coordinate += 1
    failed_empty = [
        _event("initialize"),
        _select("A0", "A11"),
        _event("start", key="A0"),
        _terminal("A0", "failure", "fail"),
    ]
    failed_partial = [
        _event("initialize"),
        _select("A0", "A11"),
        _event("membership_open", parent="A0", members=["A1"]),
        _event("start", key="A1"),
        _terminal("A1", "success", "ok"),
        _event("start", key="A0"),
        _terminal("A0", "failure", "fail"),
    ]
    specials = (
        ("empty_success", _spare(declaration), _map_events((), ())),
        ("failed_empty", declaration, failed_empty),
        ("failed_partial", declaration, failed_partial),
        ("open_partial", declaration, _map_events(("A1",), ("success",), closed=False)),
        ("closed_missing_terminal", declaration, _map_events(("A1",), ())),
        ("duplicate", declaration, _map_events(("A1",), ("success",)) + [_terminal("A1", "success", "ok")]),
        (
            "foreign",
            declaration,
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("membership_close", parent="A0", members=["A1"], invocation="I1"),
            ],
        ),
        (
            "wrong_parent",
            declaration,
            [_event("initialize"), _select("A0", "A11"), _event("membership_close", parent="A1", members=[])],
        ),
        ("closed_grow", declaration, _map_events((), ()) + [_event("membership_close", parent="A0", members=["A1"])]),
        (
            "closed_shrink",
            declaration,
            _map_events(("A1",), ("success",)) + [_event("membership_close", parent="A0", members=[])],
        ),
        ("reopen", declaration, _map_events((), ()) + [_event("membership_open", parent="A0", members=[])]),
        (
            "survivor_only",
            declaration,
            _map_events(("A1",), ("success",), closed=False) + [_event("membership_close", parent="A0", members=[])],
        ),
        (
            "abnormal_expander_failure",
            declaration,
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "failure", None),
            ],
        ),
    )
    for name, facts, events in specials:
        yield _case("map", f"{coordinate:03d}", name, facts, events)
        coordinate += 1


def _join_cases() -> Iterable[Object]:
    coordinate = 0
    declaration = _aggregate_decl(2)
    for count in (0, 1, 2):
        for assignment in itertools.product(("success", "failure", "cancelled", "lost"), repeat=count):
            yield _case(
                "join",
                f"{coordinate:03d}",
                f"children_{count}_{'-'.join(assignment) or 'empty'}",
                declaration,
                _map_events(tuple(f"A{i + 1}" for i in range(count)), cast(tuple[Category, ...], assignment)),
            )
            coordinate += 1
    specials = (
        ("any_match", _spare(declaration), _map_events(("A1", "A2"), ("success", "failure"))),
        (
            "omitted_child",
            declaration,
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "ok"),
                _event("membership_close", parent="A0", members=["A1", "A2"]),
                _event("start", key="A1"),
                _terminal("A1", "success", "ok"),
            ],
        ),
        ("duplicate_terminal", declaration, _map_events(("A1",), ("success",)) + [_terminal("A1", "success", "ok")]),
        (
            "foreign_terminal",
            declaration,
            _map_events((), ()) + [_event("terminal_success", key="A1", outcome="ok", invocation="I1")],
        ),
        ("open_complete_survivors", declaration, _map_events(("A1",), ("success",), closed=False)),
    )
    for name, facts, events in specials:
        yield _case("join", f"{coordinate:03d}", name, facts, events)
        coordinate += 1


def _loop_cases() -> Iterable[Object]:
    coordinate = 0
    for bound in (0, 1, 2):
        declaration = _aggregate_decl(bound, kind="loop")
        yield _case(
            "loop",
            f"{coordinate:03d}",
            f"bypass_bound_{bound}",
            declaration,
            [_event("initialize"), _select("A0", "A11"), _event("start", key="A0"), _terminal("A0", "success", "stop")],
        )
        coordinate += 1
    declaration = _aggregate_decl(0, kind="loop")
    yield _case(
        "loop",
        f"{coordinate:03d}",
        "enter_bound_zero",
        declaration,
        [_event("initialize"), _select("A0", "A11"), _event("start", key="A0"), _terminal("A0", "success", "again")],
    )
    coordinate += 1
    for bound in (1, 2):
        for executed in range(1, bound + 1):
            declaration = _aggregate_decl(bound, kind="loop")
            events = [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "again"),
            ]
            for index in range(executed):
                key = f"A{index + 1}"
                events += [
                    _event("start", key=key),
                    _terminal(key, "success", "stop" if index + 1 == executed else "again"),
                ]
            yield _case("loop", f"{coordinate:03d}", f"bound_{bound}_executed_{executed}_stop", declaration, events)
            coordinate += 1
    declaration = _aggregate_decl(2, kind="loop")
    yield _case(
        "loop",
        f"{coordinate:03d}",
        "one_over_3",
        declaration,
        [
            _event("initialize"),
            _select("A0", "A11"),
            _event("start", key="A0"),
            _terminal("A0", "success", "again"),
            _event("start", key="A1"),
            _terminal("A1", "success", "again"),
            _event("start", key="A2"),
            _terminal("A2", "success", "again"),
        ],
    )
    coordinate += 1
    specials = (
        (
            "missing_initial",
            _aggregate_decl(2, kind="loop", initial=False),
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "again"),
            ],
        ),
        (
            "missing_carried",
            _aggregate_decl(2, kind="loop", carried=False),
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "again"),
                _event("start", key="A1"),
                _terminal("A1", "success", "again"),
            ],
        ),
        ("wrong_iteration", declaration, [_event("initialize"), _select("A2")]),
        ("duplicate_iteration", declaration, [_event("initialize"), _select("A1", "A1")]),
        (
            "foreign_iteration",
            declaration,
            [_event("initialize"), _event("select", keys=["A1"], invocation="I1")],
        ),
        (
            "continue_after_stop",
            declaration,
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "again"),
                _event("start", key="A1"),
                _terminal("A1", "success", "stop"),
                _terminal("A1", "success", "again"),
            ],
        ),
        (
            "terminal_gap",
            declaration,
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "again"),
                _event("start", key="A2"),
                _terminal("A2", "success", "stop"),
            ],
        ),
        (
            "abnormal_member_loss",
            declaration,
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "again"),
                _event("start", key="A1"),
                _terminal("A1", "lost", None),
            ],
        ),
    )
    for name, facts, events in specials:
        yield _case("loop", f"{coordinate:03d}", name, facts, events)
        coordinate += 1


def _nested_cases() -> Iterable[Object]:
    coordinate = 0
    for maps in (0, 1, 2):
        for loops in (0, 1, 2):
            seeds: list[Object] = [_seed("A0", role="map_expander"), _seed("A11", "N2", role="join")]
            subgraphs: list[Object] = []
            aggregates: list[Object] = []
            children: list[str] = []
            coordinates = (("A1", "A2", ("A3", "A4"), "A5"), ("A6", "A7", ("A8", "A9"), "A10"))
            for child, starter, possible_members, loop_join in coordinates[:maps]:
                members = possible_members[:loops]
                children.append(child)
                seeds.extend(
                    (
                        _seed(child, "N1", parent="A0", role="map_member"),
                        _seed(starter, "N0", parent=child, role="loop_starter"),
                        *(
                            _seed(member, "N1", parent=starter, iteration=index, role="loop_member")
                            for index, member in enumerate(members)
                        ),
                        _seed(loop_join, "N2", parent=child, role="join"),
                    )
                )
                subgraphs.append({"parent": child, "roots": [starter, loop_join], "sink": loop_join})
                aggregates.append(
                    {
                        "bound": loops,
                        "bypass_outcomes": ["stop"],
                        "carried": True,
                        "continue_outcomes": ["again"],
                        "enter_outcomes": ["again"],
                        "exit_outcomes": ["stop"],
                        "initial": True,
                        "join": loop_join,
                        "kind": "loop",
                        "members": list(members),
                        "starter": starter,
                    }
                )
            aggregates.insert(
                0,
                {
                    "accepted_categories": ["success"],
                    "bound": maps,
                    "expansion_outcomes": ["ok"],
                    "join": "A11",
                    "kind": "map",
                    "members": children,
                    "parent": "A0",
                },
            )
            declaration = _decl(
                seeds,
                required=("A0", "A11"),
                subgraphs=subgraphs,
                aggregates=aggregates,
            )
            declaration = _spare(declaration, loops)
            events = [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "ok"),
                _event("membership_close", parent="A0", members=children),
            ]
            for child, starter, possible_members, loop_join in coordinates[:maps]:
                members = possible_members[:loops]
                events.extend((_event("start", key=child), _event("start", key=starter)))
                events.append(_terminal(starter, "success", "stop" if loops == 0 else "again"))
                for index, member in enumerate(members):
                    events.extend(
                        (
                            _event("start", key=member),
                            _terminal(member, "success", "stop" if index + 1 == loops else "again"),
                        )
                    )
                events.extend((_event("start", key=loop_join), _terminal(loop_join, "success", "ok")))
            events.extend((_event("start", key="A11"), _terminal("A11", "success", "ok")))
            yield _case(
                "nested_map_loop",
                f"{coordinate:03d}",
                f"map_{maps}_loop_{loops}",
                declaration,
                events,
            )
            coordinate += 1
    yield _case(
        "nested_map_loop",
        f"{coordinate:03d}",
        "map_one_over_3",
        _aggregate_decl(2),
        [
            _event("initialize"),
            _select("A0", "A11"),
            _event("start", key="A0"),
            _terminal("A0", "success", "ok"),
            _event("membership_overflow", parent="A0", observed_count=3),
        ],
    )
    coordinate += 1
    yield _case(
        "nested_map_loop",
        f"{coordinate:03d}",
        "nested_loop_one_over_3",
        _aggregate_decl(2, kind="loop"),
        [
            _event("initialize"),
            _select("A0", "A11"),
            _event("start", key="A0"),
            _terminal("A0", "success", "again"),
            _event("start", key="A1"),
            _terminal("A1", "success", "again"),
            _event("start", key="A2"),
            _terminal("A2", "success", "again"),
        ],
    )


def _precedence_cases() -> Iterable[Object]:
    ordinary = _decl((_seed("A0"),), required=("A0",))
    missing_contradictory = _aggregate_decl(1)
    missing_contradictory["required"] = ["A0", "A11", "A10"]
    malformed_seeds = cast(list[Json], missing_contradictory["seeds"])
    malformed_member = dict(cast(Object, malformed_seeds[2]))
    malformed_member["parent"] = None
    malformed_seeds[2] = malformed_member
    cases = (
        (
            "overflow_type_before_value",
            _aggregate_decl(2),
            [
                _event("initialize"),
                {"invocation": "I0", "kind": "membership_overflow", "observed_count": -1, "parent": 7},
            ],
        ),
        (
            "event_limit_before_foreign",
            _decl((_seed("A0"),), required=("A0",), limits=(3, 1, 1)),
            _ordinary(("success",)) + [_event("start", key="A11", invocation="I1")],
        ),
        (
            "foreign_before_duplicate",
            _decl((_seed("A0", invocation="I1"), _seed("A1"), _seed("A1", "N1"))),
            [_event("initialize")],
        ),
        ("duplicate_before_missing", ordinary, [_event("initialize"), _select("A0", "A0", "A1")]),
        ("missing_before_contradictory", missing_contradictory, [_event("initialize")]),
        (
            "overlap_before_cycle",
            _decl(
                (_seed("A0"), _seed("A1")),
                static={"compatible": True, "edges": [["A0", "A1"], ["A1", "A0"]], "groups": [["A0"], ["A0"]]},
            ),
            [_event("initialize")],
        ),
        (
            "cycle_before_contradictory",
            _decl(
                (_seed("A0"), _seed("A1")),
                static={"compatible": False, "edges": [["A0", "A1"], ["A1", "A0"]], "groups": []},
            ),
            [_event("initialize")],
        ),
    )
    for i, (name, declaration, events) in enumerate(cases):
        yield _case("precedence", f"{i:03d}", name, declaration, events)


def _coverage_cases() -> Iterable[Object]:
    coordinate = 0
    declaration = _aggregate_decl(2)
    for closed in (False, True):
        for count in (0, 1, 2):
            for covered in range(count + 1):
                members = tuple(f"A{i + 1}" for i in range(count))
                yield _case(
                    "terminal_coverage",
                    f"{coordinate:03d}",
                    f"{'closed' if closed else 'open'}_{count}_{covered}",
                    declaration,
                    _map_events(members, tuple("success" for _ in range(covered)), closed=closed),
                )
                coordinate += 1
    specials = (
        ("duplicate", _map_events(("A1",), ("success",)) + [_terminal("A1", "success", "ok")]),
        ("missing", _map_events((), ()) + [_terminal("A1", "success", "ok")]),
        (
            "foreign",
            _map_events((), ()) + [_event("terminal_success", key="A1", outcome="ok", invocation="I1")],
        ),
    )
    for name, events in specials:
        yield _case("terminal_coverage", f"{coordinate:03d}", name, declaration, events)
        coordinate += 1


def generate_cases() -> tuple[Object, ...]:
    cases = tuple(
        itertools.chain(
            _sequence_cases(),
            _mutation_cases(),
            _choice_cases(),
            _subgraph_cases(),
            _map_cases(),
            _join_cases(),
            _loop_cases(),
            _nested_cases(),
            _precedence_cases(),
            _coverage_cases(),
        )
    )
    identifiers = [case["case_id"] for case in cases]
    payloads = [_canonical({key: value for key, value in case.items() if key != "case_id"}) for case in cases]
    if len(identifiers) != len(set(cast(list[str], identifiers))) or len(payloads) != len(set(payloads)):
        raise AssertionError("duplicate finite case")
    return cases


def load_cases(value: Json) -> tuple[Object, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("invalid corpus")
    exact = {"case_id", "declaration", "events", "expected", "family", "mode", "traces"}
    cases: list[Object] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != exact:
            raise ValueError("invalid case shape")
        cases.append(cast(Object, item))
    return tuple(cases)


def _labels(value: Json, universe: Sequence[str]) -> list[str]:
    if isinstance(value, str):
        return [value] if value in universe else []
    if isinstance(value, list):
        return list(itertools.chain.from_iterable(_labels(item, universe) for item in value))
    if isinstance(value, dict):
        return list(itertools.chain.from_iterable(_labels(item, universe) for item in value.values()))
    return []


def counts(cases: Sequence[Object]) -> Object:
    return {
        "case_count": len(cases),
        "event_count": sum(
            len(cast(list[Json], case["events"]))
            + sum(len(cast(list[Json], cast(Object, trace)["events"])) for trace in cast(list[Json], case["traces"]))
            for case in cases
        ),
        "max_activations": max(len(set(_labels(case, ACTIVATIONS))) for case in cases),
        "max_dynamic_depth": 2,
        "max_loop_iterations": 3,
        "max_map_children": 3,
        "max_templates": max(len(set(_labels(case, TEMPLATES))) for case in cases),
        "trace_count": len(cases) + sum(len(cast(list[Json], case["traces"])) for case in cases),
    }


def manifest(cases: Sequence[Object], *, generator_sha256: str, self_test_sha256: str) -> Object:
    return {
        "alphabet": list(ALPHABET),
        "capability": "workflow_activation_v1",
        "contract_sha256": CONTRACT_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(canonical_bytes(cases)).hexdigest(),
        "counts": counts(cases),
        "family_bounds": {
            "activation_universe_size": 12,
            "family_ids": list(FAMILY_IDS),
            "nested_dynamic_depth": 2,
            "one_over": 3,
            "positive_bounds": [0, 1, 2],
            "template_universe_size": 3,
        },
        "generation_provenance": {
            "byte_identical": True,
            "generations": 2,
            "tools": {
                "generator": GENERATOR_VERSION,
                "python": f"{platform.python_implementation()} {platform.python_version()}",
                "self_test": SELF_TEST_VERSION,
            },
        },
        "generator_sha256": generator_sha256,
        "independence": {"kind": "conditional-symmetric-v1", "rule_ids": list(RULE_IDS)},
        "manifest_version": "workflow-activation-reference-v1",
        "packet_id": "R1b",
        "schema_version": 1,
        "self_test_sha256": self_test_sha256,
    }


def main() -> None:
    cases = generate_cases()
    if len(sys.argv) == 1:
        sys.stdout.buffer.write(canonical_bytes(cases))
        return
    if len(sys.argv) != 4 or sys.argv[1] != "--write":
        raise SystemExit("usage: activation_v1.py [--write CORPUS MANIFEST]")
    source, test = Path(__file__), Path(__file__).with_name("test_activation_v1.py")
    Path(sys.argv[2]).write_bytes(canonical_bytes(cases))
    Path(sys.argv[3]).write_bytes(
        _canonical(
            manifest(
                cases,
                generator_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                self_test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
            )
        )
        + b"\n"
    )


if __name__ == "__main__":
    main()
