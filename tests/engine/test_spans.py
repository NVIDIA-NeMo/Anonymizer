# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``Span`` and ``Spans``: validation, value semantics, operators and the containment queries."""

from __future__ import annotations

import copy
import dataclasses
import pickle
import random
from itertools import product

import numpy as np
import pytest

from anonymizer.engine.spans import Span, Spans


@pytest.mark.parametrize(("start", "end"), [(-1, 2), (3, 2), (-2, -1)])
def test_a_span_rejects_negative_or_inverted_bounds(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="0 <= start <= end"):
        Span(start, end)


@pytest.mark.parametrize(("start", "end"), [(0.0, 2), ("0", 2), (0, None), (True, 3), (0, False)])
def test_a_span_rejects_non_integer_bounds(start: object, end: object) -> None:
    with pytest.raises(TypeError):
        Span(start, end)  # ty: ignore[invalid-argument-type]


def test_numpy_integer_bounds_become_ints() -> None:
    span = Span(np.int64(1), np.int32(3))  # ty: ignore[invalid-argument-type]
    assert span == Span(1, 3) and type(span.start) is int and type(span.end) is int


def test_a_span_is_an_immutable_hashable_ordered_value() -> None:
    span = Span(2, 5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        span.start = 3  # ty: ignore[invalid-assignment]
    assert span == Span(2, 5) and hash(span) == hash(Span(2, 5)) and len({span, Span(2, 5)}) == 1
    assert sorted([Span(3, 4), Span(2, 9), Span(2, 5)]) == [Span(2, 5), Span(2, 9), Span(3, 4)]
    assert Span(2, 5) < Span(2, 6) < Span(3, 3)
    assert repr(span) == "Span(2, 5)"
    assert (span.length, "abcdefg"[span.slice]) == (3, "cde")


def test_an_empty_span_is_valid_and_covers_no_text() -> None:
    empty = Span(4, 4)
    assert (empty.length, "abcdef"[empty.slice]) == (0, "")
    assert 4 not in empty and empty in Span(0, 9) and empty in empty
    assert empty & Span(0, 9) is None and not empty.overlaps(Span(0, 9))


@pytest.mark.parametrize(
    ("a", "b", "inside", "shared"),
    [
        pytest.param(Span(0, 5), Span(0, 5), True, Span(0, 5), id="identical"),
        pytest.param(Span(0, 9), Span(2, 4), False, Span(2, 4), id="nested"),
        pytest.param(Span(2, 4), Span(0, 9), True, Span(2, 4), id="inside"),
        pytest.param(Span(0, 5), Span(3, 8), False, Span(3, 5), id="overlapping"),
        pytest.param(Span(0, 3), Span(3, 6), False, None, id="touching"),
        pytest.param(Span(0, 2), Span(5, 7), False, None, id="disjoint"),
    ],
)
def test_span_relations(a: Span, b: Span, inside: bool, shared: Span | None) -> None:
    assert (a in b) is inside
    assert a & b == shared == b & a
    assert a.overlaps(b) is (shared is not None)


def test_positions_in_a_span_are_half_open() -> None:
    span = Span(2, 4)
    assert [i in span for i in range(6)] == [False, False, True, True, False, False]
    assert "2" not in span and True not in Span(0, 3)


@pytest.mark.parametrize(
    ("a", "b", "rest"),
    [
        (Span(0, 9), Span(3, 5), [Span(0, 3), Span(5, 9)]),
        (Span(0, 9), Span(0, 5), [Span(5, 9)]),
        (Span(0, 9), Span(5, 12), [Span(0, 5)]),
        (Span(3, 5), Span(0, 9), []),
        (Span(0, 3), Span(3, 6), [Span(0, 3)]),
        (Span(0, 3), Span(7, 9), [Span(0, 3)]),
        (Span(3, 3), Span(0, 9), []),  # an empty span has no characters to keep
        (Span(3, 3), Span(5, 9), []),
    ],
)
def test_span_difference(a: Span, b: Span, rest: list[Span]) -> None:
    assert list(a - b) == rest


def test_shifted() -> None:
    assert Span(2, 5).shifted(3) == Span(5, 8)
    with pytest.raises(ValueError):
        Span(2, 5).shifted(-3)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        (0, 3, Span(0, 3)),
        (2, 6, Span(2, 6)),
        (3, 3, None),
        (4, 2, None),
        (-1, 2, None),
        (4, 7, None),
    ],
)
def test_in_text_keeps_only_non_empty_ranges_of_the_text(start: int, end: int, expected: Span | None) -> None:
    assert Span.in_text(start, end, "abcdef") == expected


def test_cover_is_the_smallest_span_containing_them_all() -> None:
    assert Span.cover([Span(4, 6), Span(1, 3), Span(2, 9), Span(5, 5)]) == Span(1, 9)
    assert Span.cover(iter([Span(3, 3)])) == Span(3, 3)
    with pytest.raises(ValueError, match="at least one"):
        Span.cover([])


def test_spans_is_an_ordered_set() -> None:
    spans = Spans([Span(5, 6), Span(0, 3), Span(5, 6), Span(0, 2)])
    assert list(spans) == [Span(0, 2), Span(0, 3), Span(5, 6)]
    assert (len(spans), spans[0], spans[-1]) == (3, Span(0, 2), Span(5, 6))
    assert spans[1:] == (Span(0, 3), Span(5, 6)) and spans[::-1] == (Span(5, 6), Span(0, 3), Span(0, 2))
    assert Span(0, 3) in spans and Span(1, 3) not in spans and (0, 3) not in spans
    assert spans == Spans(reversed(list(spans))) and hash(spans) == hash(Spans(list(spans)))
    assert spans | Spans([Span(1, 2), Span(0, 2)]) == Spans([Span(0, 2), Span(0, 3), Span(1, 2), Span(5, 6)])
    assert repr(Spans([Span(1, 2)])) == "Spans([Span(1, 2)])" and repr(Spans()) == "Spans([])"
    with pytest.raises(TypeError):
        Spans([(0, 3)])  # ty: ignore[invalid-argument-type]


def test_spans_is_immutable_and_copies_as_a_value() -> None:
    spans = Spans([Span(0, 15), Span(6, 9)])
    assert spans.container_of(Span(10, 12)) == Span(0, 15)
    with pytest.raises(AttributeError, match="immutable"):
        spans._spans = ()  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        spans.extra = 1  # ty: ignore[invalid-assignment]
    with pytest.raises(AttributeError):
        del spans._spans  # ty: ignore[invalid-assignment]
    assert spans.container_of(Span(10, 12)) == Span(0, 15)
    assert pickle.loads(pickle.dumps(spans)) == spans == copy.deepcopy(spans)


def test_merged_joins_overlapping_and_touching_spans_and_drops_empty_ones() -> None:
    spans = Spans([Span(0, 3), Span(3, 5), Span(7, 9), Span(8, 12), Span(9, 10), Span(14, 14), Span(4, 4)])
    assert spans.merged() == Spans([Span(0, 5), Span(7, 12)])
    assert Spans().merged() == Spans()


def test_outermost_drops_nested_and_keeps_partial_overlaps() -> None:
    spans = Spans([Span(0, 15), Span(6, 9), Span(0, 4), Span(10, 20), Span(10, 12), Span(20, 20)])
    assert spans.outermost() == Spans([Span(0, 15), Span(10, 20)])
    ends = [span.end for span in spans.outermost()]
    assert ends == sorted(set(ends))


@pytest.mark.parametrize(
    ("inner", "container"),
    [
        pytest.param(Span(10, 15), Span(0, 15), id="inside-an-outer-span-with-a-nested-one"),
        pytest.param(Span(6, 9), Span(0, 15), id="the-nested-span-itself"),
        pytest.param(Span(21, 26), Span(20, 30), id="in-both-overlapping-the-one-reaching-furthest"),
        pytest.param(Span(24, 28), Span(20, 30), id="in-the-later-of-two-overlapping"),
        pytest.param(Span(17, 28), None, id="across-two-overlapping"),
        pytest.param(Span(30, 31), None, id="touching-the-end"),
        pytest.param(Span(35, 36), None, id="between-spans"),
    ],
)
def test_container_of(inner: Span, container: Span | None) -> None:
    spans = Spans([Span(0, 15), Span(6, 9), Span(16, 26), Span(20, 30), Span(40, 50)])
    assert spans.container_of(inner) == container
    assert Spans().container_of(inner) is None


def test_container_of_matches_a_linear_scan() -> None:
    """Over random overlapping spans, the outermost container is a container whenever one exists."""
    rng = random.Random(20261119)
    for _ in range(400):
        spans = Spans(Span(a, a + rng.randint(0, 12)) for a in (rng.randint(0, 40) for _ in range(rng.randint(0, 8))))
        for start, length in product(range(0, 45, 3), range(1, 6)):
            inner = Span(start, start + length)
            holders = [span for span in spans if inner in span]
            found = spans.container_of(inner)
            assert (found is None) == (not holders)
            assert found is None or (inner in found and not any(found in s and found != s for s in spans))


def test_uncovered() -> None:
    detected = Spans([Span(0, 5), Span(10, 15)])
    matches = [Span(1, 3), Span(4, 7), Span(12, 15), Span(20, 22)]
    assert detected.uncovered(matches) == Spans([Span(4, 7), Span(20, 22)])
