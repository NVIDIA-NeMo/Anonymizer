# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The Edits layer: which edits stand together (``Span.precedes``), the two policies, and applying them."""

from __future__ import annotations

import pickle
from typing import Any

import pytest

from anonymizer.engine.edits import Conflicting, Edit, Edits, OutOfBounds, Prefer, Strict, check, conflicts
from anonymizer.engine.spans import BY_START, BY_START_END, CONTENT_ORDER, OUTER_FIRST, Order, Span


def _e(start: int, end: int, replacement: str, label: str = "") -> Edit:
    return Edit(Span(start, end), replacement, label)


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ((0, 3), (3, 5), False),  # touching
        ((0, 5), (1, 3), True),  # containing
        ((0, 3), (2, 5), True),  # crossing
        ((0, 3), (0, 3), True),  # identical
        ((0, 5), (2, 2), True),  # an insertion strictly inside
        ((0, 3), (0, 0), False),  # an insertion at an edge
        ((0, 3), (3, 3), False),
        ((2, 2), (2, 2), False),  # two insertions at one point
    ],
)
def test_two_edits_conflict_unless_one_precedes_the_other(a: tuple, b: tuple, expected: bool) -> None:
    assert conflicts(Span(*a), Span(*b)) is expected
    assert conflicts(Span(*b), Span(*a)) is expected


def test_the_relation_table_on_abcd() -> None:
    """Insertions at one point apply in input order; an insertion stands at an edge and is refused strictly inside."""
    assert Edits("abcd", [_e(2, 2, "X"), _e(2, 2, "Y")]).apply() == "abXYcd"
    assert Edits("abcd", [_e(2, 2, "Y"), _e(2, 2, "X")]).apply() == "abYXcd"
    assert Edits("abcd", [_e(1, 1, "X"), _e(1, 3, "Z")]).apply() == "aXZd"
    with pytest.raises(Conflicting):
        Edits("abcd", [_e(2, 2, "X"), _e(1, 3, "Z")])
    for bad in (True, 1.0):
        with pytest.raises(OutOfBounds):
            check(4, [(bad, 2)])


def test_edits_refuse_a_range_past_the_end() -> None:
    with pytest.raises(OutOfBounds):
        Edits("abc", [_e(1, 4, "X")])


def test_check_reports_the_first_problem_in_order() -> None:
    with pytest.raises(Conflicting) as caught:
        check(10, [(5, 8), (0, 6)])
    assert caught.value.later == Span(5, 8)
    with pytest.raises(OutOfBounds):
        check(10, [(8, 3)])
    check(10, [(2, 2), (2, 2), (2, 5), (5, 5)])


def test_prefer_keeps_the_first_in_its_order_and_reports_the_rest_in_input_order() -> None:
    text = "Alice Lee met Bob."
    edits, rejected = Edits.admit(
        text, [_e(0, 5, "F", "first_name"), _e(0, 9, "P", "person"), _e(20, 22, "Z", "z")], Prefer(BY_START)
    )
    assert edits.apply() == "F Lee met Bob."
    assert [(r.edit.label, r.reason) for r in rejected] == [("person", "conflict"), ("z", "out_of_bounds")]
    edits, _ = Edits.admit(text, [_e(0, 9, "P"), _e(0, 5, "F")], Prefer(BY_START))
    assert edits.apply() == "P met Bob."
    edits, _ = Edits.admit(text, [_e(0, 9, "P"), _e(0, 5, "F")], Prefer(BY_START_END))
    assert edits.apply() == "F Lee met Bob."


def test_strict_raises_on_a_conflict() -> None:
    with pytest.raises(Conflicting):
        Edits.admit("abcdef", [_e(0, 3, "X"), _e(2, 4, "Y")], Strict())
    edits, rejected = Edits.admit("abcdef", [_e(0, 3, "X"), _e(3, 4, "Y")], Strict())
    assert (edits.apply(), rejected) == ("XYef", ())


def test_pieces_partition_the_source_and_applied_places_each_replacement() -> None:
    edits = Edits("Alice met Bob", [_e(10, 13, "Robert", "name"), _e(0, 5, "Al", "name")])
    assert "".join(edits.source[p.span.slice] for p in edits.pieces()) == "Alice met Bob"
    placed = edits.applied()
    assert placed.source == "Al met Robert"
    assert [placed.source[e.span.slice] for e in placed] == ["Al", "Robert"]


def test_orders_end_with_input_order_and_content_order_is_a_named_value() -> None:
    items = [("person", Span(0, 3)), ("first_name", Span(0, 3))]
    for order in (BY_START, BY_START_END, OUTER_FIRST):
        assert order.sorted(items, span=lambda x: x[1]) == items
    assert CONTENT_ORDER.sorted(items, span=lambda x: x[1], label=lambda x: x[0]) == items[::-1]
    with pytest.raises(ValueError):
        Order(("start",))  # type: ignore[arg-type]


def test_policies_and_edits_pickle() -> None:
    edits = Edits("abcd", [_e(1, 2, "X")])
    assert pickle.loads(pickle.dumps(edits)).apply() == "aXcd"
    assert pickle.loads(pickle.dumps(Prefer(BY_START))) == Prefer(BY_START)


_RELATION_CASES: dict[str, list[tuple[Any, Any, str]]] = {
    "same point": [(2, 2, "x"), (2, 2, "y")],
    "same point, other order": [(2, 2, "y"), (2, 2, "x")],
    "insertion at an edge": [(1, 1, "x"), (1, 3, "z")],
    "insertion inside": [(2, 2, "x"), (1, 3, "z")],
    "adjacent": [(0, 2, "x"), (2, 4, "y")],
    "overlapping": [(0, 3, "x"), (2, 4, "y")],
    "past the end": [(2, 5, "x")],
    "bool bound": [(True, 2, "x")],
    "float bound": [(0, 1.0, "x")],
}


@pytest.mark.parametrize("case", list(_RELATION_CASES))
def test_assemble_regions_and_edits_follow_one_rule(case: str) -> None:
    """``assemble_regions`` keeps its own loop (the graph package imports nothing outside itself); this keeps it and
    the edits layer from drifting apart."""
    from anonymizer.engine.graph_core.authoring import GraphScope, Operation, ProtectionGraph
    from tests.spec_programs._builder import rejected, run

    regions = _RELATION_CASES[case]
    scope = GraphScope()
    datums = {name: scope.datum(name) for name in ("x", "y", "z")}
    result = run(ProtectionGraph(scope, tuple(datums.values())), Operation("op", lambda text: text.upper()))
    try:
        check(4, [(start, end) for start, end, _ in regions])
        expected: str = Edits("abcd", [_e(start, end, name.upper()) for start, end, name in regions]).apply()
    except OutOfBounds:
        expected = "region_out_of_bounds"
    except Conflicting:
        expected = "overlapping_region"
    graph_regions = [(start, end, datums[name]) for start, end, name in regions]
    got = rejected(lambda: result.assemble_regions("abcd", graph_regions))
    if got is None:
        got = result.assemble_regions("abcd", graph_regions)
    assert got == expected


def test_assemble_regions_refuses_a_bool_or_float_offset() -> None:
    from anonymizer.engine.graph_core.authoring import GraphScope, Operation, ProtectionGraph
    from tests.spec_programs._builder import rejected, run

    scope = GraphScope()
    head = scope.datum("abcd")
    result = run(ProtectionGraph(scope, (head,)), Operation("op", lambda text: text.upper()))
    assert result.assemble_regions("abcd", [(0, 2, head)]) == "ABCDcd"
    assert rejected(lambda: result.assemble_regions("abcd", [(True, 2, head)])) == "region_out_of_bounds"
    assert rejected(lambda: result.assemble_regions("abcd", [(0, 2.0, head)])) == "region_out_of_bounds"
