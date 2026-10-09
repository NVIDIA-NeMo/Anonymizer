# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent activation_v1 reference: builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from tests.graph_sdk.reference._activation_v1.evaluation import (
    admit_dynamic_workflow,
    admit_static_support,
    reduce_trace,
)
from tests.graph_sdk.reference._activation_v1.model import (
    TEMPLATES,
    Boundary,
    Category,
    Json,
    Object,
    Role,
    _rename,
)


def _seed(
    key: str,
    template: str = "N0",
    *,
    parent: str | None = None,
    iteration: int | None = None,
    role: Role = "ordinary",
    invocation: str = "I0",
    scope: Sequence[str] = (),
) -> Object:
    return {
        "invocation": invocation,
        "iteration": iteration,
        "key": key,
        "parent": parent,
        "role": role,
        "scope": list(scope),
        "template": template,
    }


def _decl(
    seeds: Sequence[Object],
    *,
    required: Sequence[str] = (),
    edges: Sequence[Sequence[str]] = (),
    input_dependencies: Sequence[Mapping[str, Json]] | None = None,
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
    declared_scopes = tuple(dict.fromkeys(tuple(cast(list[str], cast(Object, seed)["scope"])) for seed in seeds)) or (
        (),
    )
    default_outcomes: tuple[Object, ...] = tuple(
        {
            "category": category,
            "name": name,
            "produced_ports": list(ports),
            "scope": list(scope),
            "template": template,
        }
        for scope in declared_scopes
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
        "input_dependencies": [
            {
                "destination": edge[1],
                "destination_port": "input",
                "source": edge[0],
                "source_port": "result",
            }
            for edge in edges
        ]
        if input_dependencies is None
        else [dict(dependency) for dependency in input_dependencies],
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


def _restrict_loop_outcomes(declaration: Object) -> Object:
    result = dict(declaration)
    seeds = {
        cast(str, seed["key"]): seed for value in cast(list[Json], result["seeds"]) for seed in [cast(Object, value)]
    }
    allowed: dict[tuple[tuple[str, ...], str], set[str]] = {}
    for value in cast(list[Json], result["aggregates"]):
        aggregate = cast(Object, value)
        if aggregate["kind"] != "loop":
            continue
        starter = cast(Object, seeds[cast(str, aggregate["starter"])])
        starter_identity = (tuple(cast(list[str], starter["scope"])), cast(str, starter["template"]))
        member_identity = (
            tuple(cast(list[str], aggregate["member_scope"])),
            cast(str, aggregate["member_template"]),
        )
        allowed[starter_identity] = set(
            (*cast(list[str], aggregate["enter_outcomes"]), *cast(list[str], aggregate["bypass_outcomes"]))
        )
        allowed[member_identity] = set(
            (*cast(list[str], aggregate["continue_outcomes"]), *cast(list[str], aggregate["exit_outcomes"]))
        )
    result["outcomes"] = [
        outcome
        for value in cast(list[Json], result["outcomes"])
        for outcome in [cast(Object, value)]
        if (tuple(cast(list[str], outcome["scope"])), cast(str, outcome["template"])) not in allowed
        or cast(str, outcome["name"])
        in allowed[(tuple(cast(list[str], outcome["scope"])), cast(str, outcome["template"]))]
    ]
    return result


def _ordinary(categories: Sequence[Category], *, keys: Sequence[str] | None = None) -> list[Object]:
    selected = tuple(keys or (f"A{i}" for i in range(len(categories))))
    events = [_event("initialize"), _select(*selected)]
    for key, category in zip(selected, categories, strict=True):
        events += [_event("start", key=key), _terminal(key, category, "ok" if category == "success" else None)]
    return events


def _case(
    family: str,
    coordinate: str,
    name: str,
    declaration: Object,
    events: Sequence[Mapping[str, Json]],
    traces: Sequence[str] = (),
    boundary: Boundary = "transition",
) -> Object:
    event_list = [dict(event) for event in events]
    actual_boundary: Boundary = (
        "initialization" if boundary == "transition" and event_list == [_event("initialize")] else boundary
    )
    expected = (
        admit_dynamic_workflow(declaration)
        if actual_boundary == "dynamic_admission"
        else admit_static_support(declaration)
        if actual_boundary == "static_admission"
        else reduce_trace(declaration, event_list)
    )
    trace_values: list[Json] = []
    for trace in traces:
        if trace == "rename":
            transformed = cast(Object, _rename(declaration))
            transformed_events = cast(list[Object], _rename(cast(Json, event_list)))
        else:
            transformed = declaration
            transformed_events = event_list[:2] + event_list[4:6] + event_list[2:4] + event_list[6:]
        trace_values.append(
            {
                "boundary": actual_boundary,
                "declaration": transformed,
                "events": transformed_events,
                "expected": reduce_trace(transformed, transformed_events),
                "name": trace,
            }
        )
    return {
        "boundary": actual_boundary,
        "case_id": f"{family}/{coordinate}/{name}",
        "declaration": declaration,
        "events": event_list,
        "expected": expected,
        "family": family,
        "mode": "accepted" if expected["status"] == "accepted" else "rejected",
        "traces": trace_values,
    }
