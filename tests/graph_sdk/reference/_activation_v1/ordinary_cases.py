# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent activation_v1 reference: ordinary cases."""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from tests.graph_sdk.reference._activation_v1.builders import (
    _case,
    _decl,
    _event,
    _ordinary,
    _seed,
    _select,
    _spare,
    _terminal,
)
from tests.graph_sdk.reference._activation_v1.model import (
    CATEGORIES,
    Category,
    Json,
    Object,
)


def _sequence_cases() -> Iterable[Object]:
    for i, category in enumerate(CATEGORIES):
        declaration = _decl((_seed("A0"),), required=("A0",))
        yield _case("sequence_single", f"{i:03d}", "base", declaration, _ordinary((category,)), ("rename",))
    for i, left in enumerate(CATEGORIES):
        for j, right in enumerate(CATEGORIES):
            declaration = _decl((_seed("A0", "N0"), _seed("A1", "N1")), required=("A0", "A1"), edges=(("A0", "A1"),))
            declaration["outcomes"] = [
                outcome
                for outcome in cast(list[Json], declaration["outcomes"])
                if (cast(Object, outcome)["scope"], cast(Object, outcome)["template"]) != ([], "N0")
                or cast(Object, outcome)["name"] == "ok"
            ]
            events = [
                _event("initialize"),
                _select("A0", "A1"),
                _event("start", key="A0"),
                _terminal("A0", left, "ok" if left == "success" else None),
                _event("start", key="A1"),
                _terminal("A1", right, "ok" if right == "success" else "fail" if right == "failure" else None),
            ]
            yield _case("sequence_linked_pair", f"{i:03d}-{j:03d}", "base", declaration, events, ("rename",))
    for i, left in enumerate(CATEGORIES):
        for j, right in enumerate(CATEGORIES):
            declaration = _decl(
                (_seed("A0", "N0"), _seed("A1", "N1"), _seed("A2", "N2")),
                required=("A0", "A1", "A2"),
                edges=(("A0", "A2"), ("A1", "A2")),
                input_dependencies=(),
            )
            events = _ordinary((left, right))
            events[1] = _select("A0", "A1", "A2")
            events += [
                _event("start", key="A2"),
                _terminal("A2", "success", "ok"),
            ]
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
        outcomes=({"category": "success", "name": "ok", "produced_ports": ["result"], "scope": [], "template": "N0"},),
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
    missing_result = _decl((_seed("A0", "N0"), _seed("A1", "N1")), required=("A0", "A1"), edges=(("A0", "A1"),))
    missing_result["outcomes"] = [
        outcome
        for outcome in cast(list[Json], missing_result["outcomes"])
        if (cast(Object, outcome)["scope"], cast(Object, outcome)["template"]) != ([], "N0")
        or cast(Object, outcome)["name"] in ("ok", "fail")
    ]
    yield _case(
        "sequence_mutations",
        "010",
        "named_missing_result",
        missing_result,
        (),
        boundary="static_admission",
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
        yield _case(
            "choice",
            f"{coordinate:03d}",
            name,
            declaration,
            events,
            boundary="initialization" if name == "foreign_selector" else "transition",
        )
        coordinate += 1


def _subgraph_cases() -> Iterable[Object]:
    coordinate = 0
    for size in (1, 2):
        for category in ("success", "failure"):
            keys = tuple(f"A{i + 1}" for i in range(size))
            seeds = [_seed("A0", role="subgraph")] + [
                _seed(key, f"N{index + 1}", parent="A0", scope=("N0",)) for index, key in enumerate(keys)
            ]
            declaration = _decl(
                seeds,
                required=("A0",),
                edges=tuple((keys[i], keys[i + 1]) for i in range(size - 1)),
                subgraphs=({"parent": "A0", "roots": [keys[0]], "sink": keys[-1]},),
            )
            if size == 2:
                declaration["outcomes"] = [
                    outcome
                    for outcome in cast(list[Json], declaration["outcomes"])
                    if (cast(Object, outcome)["scope"], cast(Object, outcome)["template"]) != (["N0"], "N1")
                    or cast(Object, outcome)["name"] == "ok"
                ]
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
        (_seed("A0", role="subgraph"), _seed("A1", "N1", parent="A0", scope=("N0",))),
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
            _decl(
                (_seed("A0", role="subgraph"), _seed("A1", parent="A2", scope=("N0",))),
                required=("A0",),
            ),
            [_event("initialize")],
        ),
        (
            "foreign_body_key",
            _decl(
                (
                    _seed("A0", role="subgraph"),
                    _seed("A1", parent="A0", invocation="I1", scope=("N0",)),
                ),
                required=("A0",),
            ),
            [_event("initialize")],
        ),
        (
            "duplicate_body_key",
            _decl(
                (
                    _seed("A0", role="subgraph"),
                    _seed("A1", parent="A0", scope=("N0",)),
                    _seed("A1", "N1", parent="A0", scope=("N0",)),
                ),
                required=("A0",),
            ),
            [_event("initialize")],
        ),
    )
    for name, declaration, events in cases:
        yield _case("subgraph", f"{coordinate:03d}", name, declaration, events)
        coordinate += 1
    nested = _decl(
        (
            _seed("A0", role="subgraph"),
            _seed("A1", "N1", parent="A0", role="subgraph", scope=("N0",)),
            _seed("A2", "N2", parent="A1", scope=("N0", "N1")),
        ),
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
