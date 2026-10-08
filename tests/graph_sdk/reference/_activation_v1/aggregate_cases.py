# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent activation_v1 reference: aggregate cases."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Literal, cast

from tests.graph_sdk.reference._activation_v1.builders import (
    _case,
    _decl,
    _event,
    _ordinary,
    _restrict_loop_outcomes,
    _seed,
    _select,
    _spare,
    _terminal,
)
from tests.graph_sdk.reference._activation_v1.model import (
    Boundary,
    Category,
    Json,
    Object,
    Role,
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
            "carried_binding": (
                {"destination_port": "input", "source_kind": "member_output", "source_port": "carry"}
                if carried
                else None
            ),
            "continue_outcomes": ["again"],
            "enter_outcomes": ["again"],
            "exit_outcomes": ["stop"],
            "initial_binding": (
                {"destination_port": "input", "source_kind": "workflow_input", "source_port": "input"}
                if initial
                else None
            ),
            "join": "A11",
            "kind": "loop",
            "member_scope": [],
            "member_template": "N1",
            "members": [f"A{i + 1}" for i in range(count)],
            "starter": "A0",
        }
    declaration = _decl(seeds, required=("A0", "A11"), aggregates=(aggregate,))
    return _restrict_loop_outcomes(declaration) if kind == "loop" else declaration


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
    for name, facts in (
        ("missing_initial", _aggregate_decl(2, kind="loop", initial=False)),
        ("missing_carried", _aggregate_decl(2, kind="loop", carried=False)),
    ):
        yield _case("loop", f"{coordinate:03d}", name, facts, (), boundary="dynamic_admission")
        coordinate += 1
    duplicate_iteration = dict(declaration)
    duplicate_seeds = list(cast(list[Json], duplicate_iteration["seeds"]))
    duplicate_seeds.append(_seed("A1", "N2", parent="A0", iteration=0, role="loop_member"))
    duplicate_iteration["seeds"] = duplicate_seeds
    duplicate_limits = dict(cast(Object, duplicate_iteration["limits"]))
    duplicate_limits["max_entries"] = cast(int, duplicate_limits["max_entries"]) + 1
    duplicate_limits["max_events"] = cast(int, duplicate_limits["max_events"]) + 3
    duplicate_iteration["limits"] = duplicate_limits
    specials = (
        (
            "wrong_iteration",
            declaration,
            [
                _event("initialize"),
                _select("A0", "A11"),
                _event("start", key="A0"),
                _terminal("A0", "success", "again"),
                _event("start", key="A2"),
            ],
        ),
        ("duplicate_iteration", duplicate_iteration, [_event("initialize")]),
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
    missing_output = _aggregate_decl(2, kind="loop")
    output_specs = [dict(cast(Object, outcome)) for outcome in cast(list[Json], missing_output["outcomes"])]
    for outcome in output_specs:
        if (outcome["scope"], outcome["template"], outcome["name"]) == ([], "N1", "again"):
            outcome["produced_ports"] = []
        elif (outcome["scope"], outcome["template"], outcome["name"]) == ([], "N1", "stop"):
            outcome["produced_ports"] = ["carry", "result"]
    missing_output["outcomes"] = cast(Json, output_specs)
    yield _case(
        "loop",
        f"{coordinate:03d}",
        "missing_carried_output",
        missing_output,
        [
            _event("initialize"),
            _select("A0", "A11"),
            _event("start", key="A0"),
            _terminal("A0", "success", "again"),
            _event("start", key="A1"),
            _terminal("A1", "success", "again"),
        ],
    )


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
                        _seed(starter, "N0", parent=child, role="loop_starter", scope=("N1",)),
                        *(
                            _seed(
                                member,
                                "N1",
                                parent=starter,
                                iteration=index,
                                role="loop_member",
                                scope=("N1",),
                            )
                            for index, member in enumerate(members)
                        ),
                        _seed(loop_join, "N2", parent=child, role="join", scope=("N1",)),
                    )
                )
                subgraphs.append({"parent": child, "roots": [starter, loop_join], "sink": loop_join})
                aggregates.append(
                    {
                        "bound": loops,
                        "bypass_outcomes": ["stop"],
                        "carried_binding": {
                            "destination_port": "input",
                            "source_kind": "member_output",
                            "source_port": "carry",
                        },
                        "continue_outcomes": ["again"],
                        "enter_outcomes": ["again"],
                        "exit_outcomes": ["stop"],
                        "initial_binding": {
                            "destination_port": "input",
                            "source_kind": "workflow_input",
                            "source_port": "input",
                        },
                        "join": loop_join,
                        "kind": "loop",
                        "member_scope": ["N1"],
                        "member_template": "N1",
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
            declaration = _restrict_loop_outcomes(declaration)
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
    duplicate_missing = _decl(
        (_seed("A0", "N0"), _seed("A0", "N1")),
        required=("A0", "A1"),
    )
    missing_contradictory = _aggregate_decl(1)
    missing_contradictory["required"] = ["A0", "A11"]
    malformed_seeds = cast(list[Json], missing_contradictory["seeds"])
    malformed_member = dict(cast(Object, malformed_seeds[2]))
    malformed_member["parent"] = "A11"
    malformed_seeds[:] = [malformed_seeds[1], malformed_member]
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
        ("duplicate_before_missing", duplicate_missing, [_event("initialize")]),
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
        boundary: Boundary = (
            "event_construction"
            if name == "overflow_type_before_value"
            else "static_admission"
            if name in ("overlap_before_cycle", "cycle_before_contradictory")
            else "transition"
        )
        yield _case("precedence", f"{i:03d}", name, declaration, events, boundary=boundary)


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
