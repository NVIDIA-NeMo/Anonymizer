# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent finite reference for workflow activation semantics.

The reducer and generator in this module use only neutral strings and standard
library values.  Their exhaustive claim is limited to the twelve frozen
families and finite bounds declared below.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias, cast

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]
Object: TypeAlias = dict[str, Json]
Category: TypeAlias = Literal["success", "failure", "cancelled", "lost", "blocked", "inconsistent"]

CONTRACT_SHA256 = "9f58d60ad4ecc25065cc6c74d784cd05ad6fb2a72f183a615122eb981c7b265b"
CORPUS_PATH = "tests/graph_sdk/reference/activation_v1_cases.json"
GENERATOR_VERSION = "workflow-activation-v1-generator-1"
SELF_TEST_VERSION = "workflow-activation-v1-self-test-1"
TEMPLATES = ("N0", "N1", "N2")
INVOCATIONS = ("I0", "I1")
ACTIVATIONS = tuple(f"A{index}" for index in range(12))
CATEGORIES: tuple[Category, ...] = (
    "success",
    "failure",
    "cancelled",
    "lost",
    "blocked",
    "inconsistent",
)
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
ERROR_ORDER = (
    "invalid_type",
    "invalid_value",
    "limit_exceeded",
    "foreign_owner",
    "duplicate",
    "missing",
    "invalid_range",
    "overlap",
    "cycle",
    "contradictory",
)
RENAME_TEMPLATE = {"N0": "N2", "N1": "N0", "N2": "N1"}
RENAME_ACTIVATION = {f"A{index}": f"A{11 - index}" for index in range(12)}


@dataclass(frozen=True, slots=True, order=True)
class Entry:
    """One neutral activation entry."""

    activation: str
    template: str
    status: str
    outcome: str | None
    category: str | None


@dataclass(frozen=True, slots=True)
class ReferenceState:
    """Hashable semantic result of one accepted neutral trace."""

    entries: frozenset[Entry]
    expansions: frozenset[tuple[str, str, frozenset[str]]]
    outputs: frozenset[str]
    complete: bool


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def canonical_bytes(cases: Sequence[Object]) -> bytes:
    """Return canonical corpus bytes, including the final newline."""
    return _canonical(cast(Json, list(cases))) + b"\n"


def _event(kind: str, activation: str | None = None, **fields: Json) -> Object:
    event: Object = {"kind": kind}
    if activation is not None:
        event["activation"] = activation
    event.update(fields)
    return event


def _terminal(activation: str, category: Category, outcome: str | None = None) -> Object:
    return _event(f"terminal_{category}", activation, outcome=outcome if outcome is not None else None)


def _entry_object(entry: Entry) -> Object:
    return {
        "activation": entry.activation,
        "category": entry.category,
        "outcome": entry.outcome,
        "status": entry.status,
        "template": entry.template,
    }


def _state_object(state: ReferenceState) -> Object:
    return {
        "complete": state.complete,
        "entries": [cast(Json, _entry_object(entry)) for entry in sorted(state.entries)],
        "expansions": [
            {"members": sorted(members), "parent": parent, "status": status}
            for parent, status, members in sorted(state.expansions, key=lambda item: item[0])
        ],
        "outputs": sorted(state.outputs),
        "semantic_hash": hashlib.sha256(
            _canonical(cast(Json, sorted((_entry_object(entry) for entry in state.entries), key=_canonical)))
        ).hexdigest(),
    }


def _error(*codes: str) -> Object:
    applicable = set(codes)
    code = next(code for code in ERROR_ORDER if code in applicable)
    return {"code": code, "state": None, "status": "rejected"}


def _accepted(state: ReferenceState, **facts: Json) -> Object:
    result: Object = {"code": None, "state": _state_object(state), "status": "accepted"}
    result.update(facts)
    return result


def initial_capacity(reservation_count: int, map_expander_count: int = 0) -> tuple[int, int]:
    """Return the exact entry and conservative event capacity for initialization."""
    if isinstance(reservation_count, bool) or isinstance(map_expander_count, bool):
        raise TypeError("invalid capacity type")
    if reservation_count < 0 or map_expander_count < 0 or map_expander_count > reservation_count:
        raise ValueError("invalid capacity value")
    return reservation_count, 3 * reservation_count + map_expander_count


def completion_reserve(
    *,
    absent: int = 0,
    unstarted_or_ready: int = 0,
    running_ordinary: int = 0,
    open_map_expanders: int = 0,
) -> int:
    """Compute the normative nonnegative completion reserve."""
    values = (absent, unstarted_or_ready, running_ordinary, open_map_expanders)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("invalid reserve type")
    if any(value < 0 for value in values):
        raise ValueError("invalid reserve value")
    return 3 * absent + 2 * unstarted_or_ready + running_ordinary + open_map_expanders


def _ordinary_state(declaration: Object, events: Sequence[Object]) -> Object:
    templates = cast(dict[str, str], declaration["templates"])
    dependencies = {tuple(cast(list[str], edge)) for edge in cast(list[Json], declaration.get("dependencies", []))}
    entries: dict[str, Entry] = {}
    for event in events:
        kind = cast(str, event["kind"])
        activation = cast(str | None, event.get("activation"))
        if kind == "select":
            assert activation is not None
            if activation not in templates:
                return _error("missing")
            if activation in entries:
                return _error("duplicate")
            ready = not any(after == activation and before not in entries for before, after in dependencies)
            entries[activation] = Entry(
                activation, templates[activation], "ready" if ready else "unstarted", None, None
            )
        elif kind == "start":
            assert activation is not None
            entry = entries.get(activation)
            if entry is None:
                return _error("missing")
            if entry.status != "ready":
                return _error("contradictory")
            entries[activation] = Entry(activation, entry.template, "running", None, None)
        elif kind.startswith("terminal_"):
            assert activation is not None
            entry = entries.get(activation)
            if entry is None:
                return _error("missing")
            if entry.status in CATEGORIES:
                return _error("duplicate")
            if entry.status != "running":
                return _error("contradictory")
            category = kind.removeprefix("terminal_")
            outcome = cast(str | None, event.get("outcome"))
            if outcome is None and category == "success":
                return _error("invalid_value")
            entries[activation] = Entry(activation, entry.template, category, outcome, category)
            for before, after in dependencies:
                successor = entries.get(after)
                if before == activation and successor is not None and successor.status == "unstarted":
                    entries[after] = Entry(after, successor.template, "ready", None, None)
        elif kind.startswith("close_"):
            assert activation is not None
            entry = entries.get(activation)
            if entry is None:
                return _error("missing")
            category = kind.removeprefix("close_")
            entries[activation] = Entry(activation, entry.template, category, None, category)
    terminal = bool(entries) and all(entry.status in CATEGORIES for entry in entries.values())
    outputs = frozenset(entry.activation for entry in entries.values() if entry.outcome is not None)
    return _accepted(ReferenceState(frozenset(entries.values()), frozenset(), outputs, terminal))


def _aggregate_state(declaration: Object) -> Object:
    kind = cast(str, declaration["aggregate"])
    members = tuple(cast(list[str], declaration.get("members", [])))
    categories = tuple(cast(list[str], declaration.get("categories", [])))
    membership = cast(str, declaration.get("membership", "closed"))
    covered = cast(int, declaration.get("covered", len(categories)))
    mutation = cast(str, declaration.get("mutation", "base"))
    if mutation in {"duplicate", "duplicate_terminal"}:
        return _error("duplicate")
    if mutation in {"foreign", "foreign_terminal", "wrong_parent", "foreign_iteration"}:
        return _error("foreign_owner")
    if mutation in {"missing_activation", "omitted_child"}:
        return _error("missing")
    if mutation in {"closed_grow", "closed_shrink", "reopen", "survivor_only", "continue_after_stop"}:
        return _error("contradictory")
    entries = {
        Entry(
            member,
            "N1",
            categories[index] if index < len(categories) else "unstarted",
            "ok" if index < len(categories) else None,
            categories[index] if index < len(categories) else None,
        )
        for index, member in enumerate(members)
    }
    expansion_status = cast(str, declaration.get("expansion_status", "closed" if membership == "closed" else "pending"))
    join_status = "unstarted"
    complete = False
    if kind == "map":
        if expansion_status == "failed":
            join_status = "blocked"
        elif expansion_status == "overflow":
            join_status = "inconsistent"
        elif membership == "closed" and covered == len(members):
            join_status = "ready" if all(category == "success" for category in categories) else "blocked"
        complete = join_status in {"blocked", "inconsistent"} and covered == len(members)
    elif kind == "join":
        if membership == "closed" and covered == len(members):
            join_status = "ready" if all(category == "success" for category in categories) else "blocked"
        complete = False  # A ready or blocked required join still needs its explicit terminal witness.
    else:
        join_status = cast(str, declaration["join_status"])
        expansion_status = cast(str, declaration["expansion_status"])
        complete = join_status in CATEGORIES
    entries.add(Entry("A11", "N2", join_status, None, join_status if join_status in CATEGORIES else None))
    state = ReferenceState(
        frozenset(entries),
        frozenset({("A0", expansion_status, frozenset(members))}),
        frozenset(member for member, category in zip(members, categories, strict=False) if category == "success"),
        complete,
    )
    return _accepted(state, aggregate=expansion_status, join_status=join_status)


def reduce_trace(declaration: Mapping[str, Json], events: Sequence[Mapping[str, Json]]) -> Object:
    """Reduce one neutral declaration and trace to its expected result."""
    owned = cast(Object, dict(declaration))
    trace = [cast(Object, dict(event)) for event in events]
    defects = cast(list[str], owned.get("defects", []))
    if defects:
        return _error(*defects)
    scenario = cast(str, owned["scenario"])
    if scenario in {"sequence", "choice"}:
        return _ordinary_state(owned, trace)
    if scenario in {"map", "join", "loop", "coverage"}:
        return _aggregate_state(owned)
    if scenario == "subgraph":
        mutation = cast(str, owned.get("mutation", "base"))
        if mutation in {"wrong_parent", "foreign_body_key"}:
            return _error("foreign_owner")
        if mutation == "duplicate_body_key":
            return _error("duplicate")
        if mutation == "premature_parent_terminal":
            return _error("contradictory")
        category = cast(str, owned["category"])
        outcome = cast(str | None, owned.get("outcome", "ok" if category == "success" else "fail"))
        depth = cast(int, owned.get("depth", 1))
        body_size = cast(int, owned.get("body_size", 1))
        entries = {Entry("A0", "N0", category, outcome, category)}
        entries.update(Entry(f"A{index + 1}", "N1", category, outcome, category) for index in range(body_size))
        return _accepted(
            ReferenceState(
                frozenset(entries), frozenset(), frozenset() if outcome is None else frozenset({"A0"}), True
            ),
            derived_parent=True,
            depth=depth,
        )
    if scenario == "nested":
        map_children = cast(int, owned["map_children"])
        iterations = cast(int, owned["loop_iterations"])
        observed = cast(str, owned.get("observed", "materialized"))
        count = 2 + map_children * (3 + iterations) if observed == "materialized" else 2
        entries = frozenset(
            Entry(f"A{index}", TEMPLATES[index % 3], "success", "ok", "success") for index in range(count)
        )
        return _accepted(ReferenceState(entries, frozenset(), frozenset(), True), activation_count=count)
    if scenario == "precedence":
        return _error(*cast(list[str], owned["defect_classes"]))
    raise ValueError("unknown neutral activation scenario")


def _rename(value: Json) -> Json:
    if isinstance(value, list):
        return [_rename(item) for item in value]
    if isinstance(value, dict):
        return {RENAME_TEMPLATE.get(key, RENAME_ACTIVATION.get(key, key)): _rename(item) for key, item in value.items()}
    if isinstance(value, str):
        return RENAME_TEMPLATE.get(value, RENAME_ACTIVATION.get(value, value))
    return value


def _trace(name: str, declaration: Object, events: list[Object]) -> Object:
    if name == "rename":
        renamed_declaration = cast(Object, _rename(declaration))
        renamed_events = cast(list[Object], _rename(cast(Json, events)))
        return {"events": renamed_events, "expected": reduce_trace(renamed_declaration, renamed_events), "name": name}
    if name == "commute_independent_siblings":
        reversed_events = events[:1] + events[4:7] + events[1:4]
        return {"events": reversed_events, "expected": reduce_trace(declaration, reversed_events), "name": name}
    raise ValueError("unknown trace")


def _case(
    family: str, coordinate: str, name: str, declaration: Object, events: list[Object], traces: Sequence[str] = ()
) -> Object:
    expected = reduce_trace(declaration, events)
    return {
        "case_id": f"{family}/{coordinate}/{name}",
        "declaration": declaration,
        "events": events,
        "expected": expected,
        "family": family,
        "mode": "accepted" if expected["status"] == "accepted" else "rejected",
        "traces": [_trace(trace, declaration, events) for trace in traces],
    }


def _sequence_events(categories: Sequence[Category]) -> list[Object]:
    events = [_event("initialize")]
    for index, category in enumerate(categories):
        activation = f"A{index}"
        events.extend(
            (
                _event("select", activation),
                _event("start", activation),
                _terminal(activation, category, "ok" if category == "success" else "fail"),
            )
        )
    return events


def _sequence_cases() -> Iterable[Object]:
    for index, category in enumerate(CATEGORIES):
        declaration: Object = {"dependencies": [], "scenario": "sequence", "templates": {"A0": "N0"}}
        events = _sequence_events((category,))
        yield _case("sequence_single", f"{index:03d}", "base", declaration, events, ("rename",))
    for predecessor_index, predecessor in enumerate(CATEGORIES):
        for successor_index, successor in enumerate(CATEGORIES):
            declaration = {
                "dependencies": [["A0", "A1"]],
                "scenario": "sequence",
                "templates": {"A0": "N0", "A1": "N1"},
            }
            events = [
                _event("initialize"),
                _event("select", "A0"),
                _event("select", "A1"),
                _event("start", "A0"),
                _terminal("A0", predecessor, "ok" if predecessor == "success" else "fail"),
                _event("start", "A1"),
                _terminal("A1", successor, "ok" if successor == "success" else "fail"),
            ]
            yield _case(
                "sequence_linked_pair",
                f"{predecessor_index:03d}-{successor_index:03d}",
                "base",
                declaration,
                events,
                ("rename",),
            )
    for left_index, left in enumerate(CATEGORIES):
        for right_index, right in enumerate(CATEGORIES):
            declaration = {"dependencies": [], "scenario": "sequence", "templates": {"A0": "N0", "A1": "N1"}}
            events = _sequence_events((left, right))
            yield _case(
                "sequence_independent_siblings",
                f"{left_index:03d}-{right_index:03d}",
                "base",
                declaration,
                events,
                ("rename", "commute_independent_siblings"),
            )


def _mutation_cases() -> Iterable[Object]:
    mutations = (
        ("start_before_ready", [_event("initialize"), _event("start", "A0")], ["missing"]),
        (
            "terminal_before_start",
            [_event("initialize"), _event("select", "A0"), _terminal("A0", "success", "ok")],
            ["contradictory"],
        ),
        ("duplicate_terminal", _sequence_events(("success",)) + [_terminal("A0", "success", "ok")], ["duplicate"]),
        ("missing_activation", [_event("initialize"), _event("select", "A1")], ["missing"]),
    )
    base: Object = {"dependencies": [], "scenario": "sequence", "templates": {"A0": "N0"}}
    for coordinate, (name, events, defects) in enumerate(mutations):
        declaration = dict(base)
        declaration["defects"] = defects
        declaration["mutation"] = name
        yield _case("sequence_mutations", f"{coordinate:03d}", name, declaration, events)
    for offset, category in enumerate(CATEGORIES[1:]):
        declaration = dict(base)
        declaration["success_only"] = True
        events = [_event("initialize"), _event("select", "A0"), _event("start", "A0"), _terminal("A0", category, None)]
        yield _case("sequence_mutations", f"{offset + 4:03d}", f"abnormal_{category}", declaration, events)
    events = [_event("initialize"), _event("select", "A0"), _event("start", "A0"), _terminal("A0", "success", None)]
    yield _case("sequence_mutations", "009", "success_without_outcome", base, events)


def _choice_cases() -> Iterable[Object]:
    coordinate = 0
    for outcome in ("ok", "fail"):
        for order in ("forward", "reverse"):
            branch_activation = "A1" if outcome == "ok" else "A2"
            declaration: Object = {
                "branch_order": order,
                "dependencies": [],
                "scenario": "choice",
                "selector_outcome": outcome,
                "templates": {"A0": "N0", branch_activation: "N1" if outcome == "ok" else "N2"},
            }
            category: Category = "success" if outcome == "ok" else "failure"
            events = [
                _event("initialize"),
                _event("select", "A0"),
                _event("start", "A0"),
                _terminal("A0", category, outcome),
                _event("select", branch_activation),
                _event("start", branch_activation),
                _terminal(branch_activation, category, outcome),
            ]
            yield _case("choice", f"{coordinate:03d}", f"{outcome}_{order}", declaration, events, ("rename",))
            coordinate += 1
    for name, defects in (
        ("foreign_selector", ["foreign_owner"]),
        ("unknown_outcome", ["invalid_value"]),
        ("select_both", ["overlap"]),
    ):
        declaration = {
            "defects": defects,
            "dependencies": [],
            "mutation": name,
            "scenario": "choice",
            "templates": {"A0": "N0"},
        }
        yield _case("choice", f"{coordinate:03d}", name, declaration, _sequence_events(("success",)))
        coordinate += 1
    declaration = {"dependencies": [], "scenario": "choice", "selector_outcome": None, "templates": {"A0": "N0"}}
    events = [_event("initialize"), _event("select", "A0"), _event("start", "A0"), _terminal("A0", "failure", None)]
    yield _case("choice", f"{coordinate:03d}", "abnormal_selector_failure", declaration, events)


def _subgraph_cases() -> Iterable[Object]:
    coordinate = 0
    for body_size in (1, 2):
        for category in ("success", "failure"):
            declaration: Object = {"body_size": body_size, "category": category, "depth": 1, "scenario": "subgraph"}
            yield _case(
                "subgraph",
                f"{coordinate:03d}",
                f"body_{body_size}_{category}",
                declaration,
                [
                    _event("initialize"),
                    _event("start", "A0"),
                    _terminal("A1", cast(Category, category), "ok" if category == "success" else "fail"),
                ],
            )
            coordinate += 1
    for name in ("premature_parent_terminal", "wrong_parent", "foreign_body_key", "duplicate_body_key"):
        declaration = {"body_size": 1, "category": "success", "mutation": name, "scenario": "subgraph"}
        yield _case("subgraph", f"{coordinate:03d}", name, declaration, [_event("initialize"), _event("start", "A0")])
        coordinate += 1
    declaration = {"body_size": 2, "category": "success", "depth": 2, "scenario": "subgraph"}
    yield _case(
        "subgraph",
        f"{coordinate:03d}",
        "nested_body",
        declaration,
        [_event("initialize"), _event("start", "A0"), _event("start", "A1"), _terminal("A2", "success", "ok")],
    )
    coordinate += 1
    declaration = {"body_size": 1, "category": "lost", "outcome": None, "scenario": "subgraph"}
    yield _case(
        "subgraph",
        f"{coordinate:03d}",
        "abnormal_sink_loss",
        declaration,
        [_event("initialize"), _event("start", "A0"), _terminal("A1", "lost", None)],
    )


def _map_cases() -> Iterable[Object]:
    coordinate = 0
    for bound in (0, 1, 2):
        for size in range(bound + 1):
            for assignment in itertools.product(("success", "failure"), repeat=size):
                declaration: Object = {
                    "aggregate": "map",
                    "bound": bound,
                    "categories": list(assignment),
                    "covered": size,
                    "members": [f"A{index + 1}" for index in range(size)],
                    "membership": "closed",
                    "scenario": "map",
                }
                yield _case(
                    "map",
                    f"{coordinate:03d}",
                    f"bound_{bound}_size_{size}_{''.join(item[0] for item in assignment) or 'empty'}",
                    declaration,
                    [_event("membership_close", "A0", members=cast(Json, declaration["members"]))],
                )
                coordinate += 1
    declaration = {
        "aggregate": "map",
        "bound": 2,
        "categories": [],
        "expansion_status": "overflow",
        "members": [],
        "observed_count": 3,
        "scenario": "map",
    }
    yield _case(
        "map", f"{coordinate:03d}", "one_over_3", declaration, [_event("membership_overflow", "A0", observed_count=3)]
    )
    coordinate += 1
    specials = (
        ("empty_success", {}),
        ("failed_empty", {"expansion_status": "failed"}),
        ("failed_partial", {"categories": ["success"], "expansion_status": "failed", "members": ["A1"]}),
        ("open_partial", {"categories": ["success"], "members": ["A1"], "membership": "open"}),
        ("closed_missing_terminal", {"covered": 0, "members": ["A1"]}),
        ("duplicate", {"mutation": "duplicate"}),
        ("foreign", {"mutation": "foreign"}),
        ("wrong_parent", {"mutation": "wrong_parent"}),
        ("closed_grow", {"mutation": "closed_grow"}),
        ("closed_shrink", {"mutation": "closed_shrink"}),
        ("reopen", {"mutation": "reopen"}),
        ("survivor_only", {"mutation": "survivor_only"}),
        ("abnormal_expander_failure", {"expansion_status": "failed", "outcome": None}),
    )
    for name, changes in specials:
        declaration = {
            "aggregate": "map",
            "bound": 2,
            "categories": [],
            "members": [],
            "membership": "closed",
            "scenario": "map",
        }
        declaration.update(changes)
        events = [
            _event(
                "membership_open" if declaration.get("membership") == "open" else "membership_close",
                "A0",
                members=cast(Json, declaration["members"]),
            )
        ]
        if name == "failed_empty":
            events.append(_event("close_blocked", "A11"))
        elif name == "abnormal_expander_failure":
            events.append(_event("close_inconsistent", "A11"))
        yield _case(
            "map",
            f"{coordinate:03d}",
            name,
            declaration,
            events,
        )
        coordinate += 1


def _join_cases() -> Iterable[Object]:
    coordinate = 0
    for count in (0, 1, 2):
        for assignment in itertools.product(("success", "failure", "cancelled", "lost"), repeat=count):
            declaration: Object = {
                "aggregate": "join",
                "categories": list(assignment),
                "covered": count,
                "members": [f"A{index + 1}" for index in range(count)],
                "membership": "closed",
                "scenario": "join",
            }
            yield _case(
                "join",
                f"{coordinate:03d}",
                f"children_{count}_{'-'.join(assignment) or 'empty'}",
                declaration,
                [_event("membership_close", "A0", members=cast(Json, declaration["members"]))],
            )
            coordinate += 1
    for name, changes in (
        ("any_match", {"categories": ["success", "failure"], "members": ["A1", "A2"]}),
        ("omitted_child", {"mutation": "omitted_child"}),
        ("duplicate_terminal", {"mutation": "duplicate_terminal"}),
        ("foreign_terminal", {"mutation": "foreign_terminal"}),
        ("open_complete_survivors", {"categories": ["success"], "members": ["A1"], "membership": "open"}),
    ):
        declaration = {"aggregate": "join", "categories": [], "members": [], "membership": "closed", "scenario": "join"}
        declaration.update(changes)
        yield _case(
            "join",
            f"{coordinate:03d}",
            name,
            declaration,
            [_event("membership_close", "A0", members=cast(Json, declaration["members"]))],
        )
        coordinate += 1


def _loop_cases() -> Iterable[Object]:
    coordinate = 0
    for bound in (0, 1, 2):
        declaration: Object = {
            "aggregate": "loop",
            "bound": bound,
            "expansion_status": "closed",
            "join_status": "ready",
            "members": [],
            "scenario": "loop",
        }
        yield _case(
            "loop",
            f"{coordinate:03d}",
            f"bypass_bound_{bound}",
            declaration,
            [_event("membership_close", "A0", members=[])],
        )
        coordinate += 1
    declaration = {
        "aggregate": "loop",
        "bound": 0,
        "expansion_status": "overflow",
        "join_status": "inconsistent",
        "members": [],
        "scenario": "loop",
    }
    yield _case(
        "loop",
        f"{coordinate:03d}",
        "enter_bound_zero",
        declaration,
        [_event("terminal_success", "A0", outcome="again"), _event("membership_overflow", "A0", observed_count=1)],
    )
    coordinate += 1
    for bound in (1, 2):
        for executed in range(1, bound + 1):
            members = [f"A{index + 1}" for index in range(executed)]
            declaration = {
                "aggregate": "loop",
                "bound": bound,
                "categories": ["success"] * executed,
                "expansion_status": "closed",
                "join_status": "ready",
                "members": members,
                "scenario": "loop",
            }
            yield _case(
                "loop",
                f"{coordinate:03d}",
                f"bound_{bound}_executed_{executed}_stop",
                declaration,
                [_event("membership_close", "A0", members=cast(Json, members))],
            )
            coordinate += 1
    declaration = {
        "aggregate": "loop",
        "bound": 2,
        "expansion_status": "overflow",
        "join_status": "inconsistent",
        "members": ["A1", "A2"],
        "observed_count": 3,
        "scenario": "loop",
    }
    yield _case(
        "loop", f"{coordinate:03d}", "one_over_3", declaration, [_event("membership_overflow", "A0", observed_count=3)]
    )
    coordinate += 1
    for name, changes in (
        ("missing_initial", {"failure_cause": "initial", "join_status": "blocked", "expansion_status": "failed"}),
        ("missing_carried", {"failure_cause": "carried", "join_status": "blocked", "expansion_status": "failed"}),
        ("wrong_iteration", {"mutation": "wrong_parent"}),
        ("duplicate_iteration", {"mutation": "duplicate"}),
        ("foreign_iteration", {"mutation": "foreign_iteration"}),
        ("continue_after_stop", {"mutation": "continue_after_stop"}),
        ("terminal_gap", {"join_status": "unstarted", "expansion_status": "pending"}),
        (
            "abnormal_member_loss",
            {
                "categories": ["lost"],
                "join_status": "blocked",
                "expansion_status": "failed",
                "members": ["A1"],
                "outcome": None,
            },
        ),
    ):
        declaration = {
            "aggregate": "loop",
            "bound": 2,
            "categories": [],
            "expansion_status": "closed",
            "join_status": "ready",
            "members": [],
            "scenario": "loop",
        }
        declaration.update(changes)
        yield _case(
            "loop",
            f"{coordinate:03d}",
            name,
            declaration,
            [_event("membership_close", "A0", members=cast(Json, declaration["members"]))],
        )
        coordinate += 1


def _nested_cases() -> Iterable[Object]:
    coordinate = 0
    for map_children in (0, 1, 2):
        for iterations in (0, 1, 2):
            declaration: Object = {"loop_iterations": iterations, "map_children": map_children, "scenario": "nested"}
            yield _case(
                "nested_map_loop",
                f"{coordinate:03d}",
                f"map_{map_children}_loop_{iterations}",
                declaration,
                [_event("initialize")],
            )
            coordinate += 1
    for name, axis in (("map_one_over_3", "map"), ("nested_loop_one_over_3", "loop")):
        declaration = {
            "loop_iterations": 3 if axis == "loop" else 0,
            "map_children": 3 if axis == "map" else 1,
            "observed": "overflow",
            "scenario": "nested",
        }
        yield _case(
            "nested_map_loop",
            f"{coordinate:03d}",
            name,
            declaration,
            [_event("membership_overflow", "A0", observed_count=3)],
        )
        coordinate += 1


def _precedence_cases() -> Iterable[Object]:
    cases = (
        ("overflow_type_before_value", ["invalid_type", "invalid_value"], "constructor"),
        ("event_limit_before_foreign", ["limit_exceeded", "foreign_owner"], "transition"),
        ("foreign_before_duplicate", ["foreign_owner", "duplicate"], "initialization"),
        ("duplicate_before_missing", ["duplicate", "missing"], "transition"),
        ("missing_before_contradictory", ["missing", "contradictory"], "initialization"),
        ("overlap_before_cycle", ["overlap", "cycle"], "static_admission"),
        ("cycle_before_contradictory", ["cycle", "contradictory"], "static_admission"),
    )
    for coordinate, (name, defects, boundary) in enumerate(cases):
        declaration: Object = {"boundary": boundary, "defect_classes": defects, "scenario": "precedence"}
        yield _case("precedence", f"{coordinate:03d}", name, declaration, [_event("initialize")])


def _coverage_cases() -> Iterable[Object]:
    coordinate = 0
    for membership in ("open", "closed"):
        for count in (0, 1, 2):
            for covered in range(count + 1):
                declaration: Object = {
                    "aggregate": "join",
                    "categories": ["success"] * covered,
                    "covered": covered,
                    "members": [f"A{index + 1}" for index in range(count)],
                    "membership": membership,
                    "scenario": "coverage",
                }
                yield _case(
                    "terminal_coverage",
                    f"{coordinate:03d}",
                    f"{membership}_{count}_{covered}",
                    declaration,
                    [
                        _event(
                            "membership_open" if membership == "open" else "membership_close",
                            "A0",
                            members=cast(Json, declaration["members"]),
                        )
                    ],
                )
                coordinate += 1
    for name, mutation in (
        ("duplicate", "duplicate_terminal"),
        ("missing", "omitted_child"),
        ("foreign", "foreign_terminal"),
    ):
        declaration = {
            "aggregate": "join",
            "categories": [],
            "members": [],
            "mutation": mutation,
            "scenario": "coverage",
        }
        yield _case(
            "terminal_coverage", f"{coordinate:03d}", name, declaration, [_event("membership_close", "A0", members=[])]
        )
        coordinate += 1


def generate_cases() -> tuple[Object, ...]:
    """Generate the full finite activation grammar in normative family order."""
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
    if len(identifiers) != len(set(cast(list[str], identifiers))):
        raise AssertionError("duplicate case id")
    payloads = [_canonical({key: value for key, value in case.items() if key != "case_id"}) for case in cases]
    if len(payloads) != len(set(payloads)):
        raise AssertionError("duplicate case payload")
    return cases


def load_cases(value: Json) -> tuple[Object, ...]:
    """Validate the shallow frozen-corpus boundary."""
    if not isinstance(value, list) or not value:
        raise ValueError("corpus must be a nonempty array")
    result: list[Object] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "case_id",
            "declaration",
            "events",
            "expected",
            "family",
            "mode",
            "traces",
        }:
            raise ValueError("invalid case shape")
        result.append(cast(Object, item))
    return tuple(result)


def counts(cases: Sequence[Object]) -> Object:
    """Derive manifest counts and observed bounds from generated cases."""
    event_count = sum(
        len(cast(list[Json], case["events"]))
        + sum(len(cast(list[Json], cast(Object, trace)["events"])) for trace in cast(list[Json], case["traces"]))
        for case in cases
    )
    trace_count = len(cases) + sum(len(cast(list[Json], case["traces"])) for case in cases)
    max_templates = max(len(set(_collect_labels(case, "N"))) for case in cases)
    max_activations = max(len(set(_collect_labels(case, "A"))) for case in cases)
    return {
        "case_count": len(cases),
        "event_count": event_count,
        "max_activations": max_activations,
        "max_dynamic_depth": 2,
        "max_loop_iterations": 3,
        "max_map_children": 3,
        "max_templates": max_templates,
        "trace_count": trace_count,
    }


def _collect_labels(value: Json, prefix: str) -> list[str]:
    labels: list[str] = []
    if isinstance(value, str) and value in (TEMPLATES if prefix == "N" else ACTIVATIONS):
        labels.append(value)
    elif isinstance(value, list):
        for item in value:
            labels.extend(_collect_labels(item, prefix))
    elif isinstance(value, dict):
        for item in value.values():
            labels.extend(_collect_labels(item, prefix))
    return labels


def manifest(cases: Sequence[Object], *, generator_sha256: str, self_test_sha256: str) -> Object:
    """Build the exact product manifest from generated bytes."""
    corpus = canonical_bytes(cases)
    return {
        "alphabet": list(ALPHABET),
        "capability": "workflow_activation_v1",
        "contract_sha256": CONTRACT_SHA256,
        "corpus_path": CORPUS_PATH,
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
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
    """Write canonical corpus or print it when invoked directly."""
    cases = generate_cases()
    if len(sys.argv) == 1:
        sys.stdout.buffer.write(canonical_bytes(cases))
        return
    if len(sys.argv) != 4 or sys.argv[1] != "--write":
        raise SystemExit("usage: activation_v1.py [--write CORPUS MANIFEST]")
    corpus_path = Path(sys.argv[2])
    manifest_path = Path(sys.argv[3])
    source_path = Path(__file__)
    test_path = source_path.with_name("test_activation_v1.py")
    corpus_path.write_bytes(canonical_bytes(cases))
    product_manifest = manifest(
        cases,
        generator_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        self_test_sha256=hashlib.sha256(test_path.read_bytes()).hexdigest(),
    )
    manifest_path.write_bytes(_canonical(product_manifest) + b"\n")


if __name__ == "__main__":
    main()
