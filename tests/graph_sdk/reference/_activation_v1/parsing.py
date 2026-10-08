# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent activation_v1 reference: parsing."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, cast

from tests.graph_sdk.reference._activation_v1.model import (
    ACTIVATIONS,
    CATEGORIES,
    INVOCATIONS,
    TEMPLATES,
    Aggregate,
    Category,
    Choice,
    Close,
    Declaration,
    Event,
    Initialize,
    InputDependency,
    Json,
    Limits,
    LoopAggregate,
    MapAggregate,
    Membership,
    Object,
    Outcome,
    Overflow,
    Rejected,
    Role,
    Seed,
    Select,
    Start,
    Subgraph,
    TerminalEvent,
)


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


def _parse_loop_binding(value: Json, source_kind: str) -> tuple[str, str, str] | None:
    if value is None:
        return None
    item = _obj(value, {"source_kind", "source_port", "destination_port"})
    if item["source_kind"] != source_kind:
        raise Rejected("contradictory")
    source_port = item["source_port"]
    destination_port = item["destination_port"]
    if not isinstance(source_port, str) or not isinstance(destination_port, str):
        raise Rejected("invalid_type")
    if not source_port or not destination_port:
        raise Rejected("invalid_value")
    return source_kind, source_port, destination_port


def parse_declaration(value: Mapping[str, Json]) -> Declaration:
    """Parse the JSON boundary once into immutable closed facts."""
    raw = _obj(
        dict(value),
        {
            "invocation",
            "seeds",
            "required",
            "edges",
            "input_dependencies",
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
        seed = _obj(item, {"key", "template", "invocation", "parent", "iteration", "role", "scope"})
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
                tuple(_closed(part, TEMPLATES) for part in _strs(seed["scope"])),
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
        outcome = _obj(item, {"scope", "template", "name", "category", "produced_ports"})
        name = outcome["name"]
        if not isinstance(name, str):
            raise Rejected("invalid_type")
        if not name:
            raise Rejected("invalid_value")
        ports = _strs(outcome["produced_ports"])
        outcomes.append(
            Outcome(
                tuple(_closed(part, TEMPLATES) for part in _strs(outcome["scope"])),
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
                    "initial_binding",
                    "carried_binding",
                    "member_scope",
                    "member_template",
                },
            )

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
                    _parse_loop_binding(aggregate["initial_binding"], "workflow_input"),
                    _parse_loop_binding(aggregate["carried_binding"], "member_output"),
                    tuple(_closed(part, TEMPLATES) for part in _strs(aggregate["member_scope"])),
                    _closed(aggregate["member_template"], TEMPLATES),
                )
            )
    limits = _obj(raw["limits"], {"max_events", "max_entries", "max_parent_depth"})
    static = _obj(raw["static"], {"groups", "edges", "compatible"})
    if not isinstance(static["compatible"], bool):
        raise Rejected("invalid_type")
    edges = tuple(_strs(edge) for edge in _list(raw["edges"]))
    input_dependencies = tuple(
        InputDependency(
            _closed(item["source"], ACTIVATIONS),
            item["source_port"],
            _closed(item["destination"], ACTIVATIONS),
            item["destination_port"],
        )
        for value_item in _list(raw["input_dependencies"])
        for item in [_obj(value_item, {"source", "source_port", "destination", "destination_port"})]
        if isinstance(item["source_port"], str) and isinstance(item["destination_port"], str)
    )
    if len(input_dependencies) != len(_list(raw["input_dependencies"])):
        raise Rejected("invalid_type")
    if any(not item.source_port or not item.destination_port for item in input_dependencies):
        raise Rejected("invalid_value")
    if any(len(edge) != 2 for edge in edges):
        raise Rejected("invalid_value")
    return Declaration(
        _closed(raw["invocation"], INVOCATIONS),
        tuple(seeds),
        _strs(raw["required"]),
        cast(tuple[tuple[str, str], ...], edges),
        input_dependencies,
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


def _raw_depth(key: str, parents: Mapping[str, tuple[str | None, ...]]) -> tuple[int, bool, bool]:
    maximum = 0
    missing = False
    cycle = False
    pending = [(key, frozenset[str]())]
    visited: set[tuple[str, frozenset[str]]] = set()
    while pending:
        current, path = pending.pop()
        state = (current, path)
        if state in visited:
            continue
        visited.add(state)
        if current in path:
            cycle = True
            continue
        options = parents.get(current)
        if options is None:
            missing = True
            continue
        next_path = path | {current}
        for parent in options:
            if parent is None:
                maximum = max(maximum, len(next_path))
            else:
                pending.append((parent, next_path))
    return maximum, missing, cycle
