# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Character ranges: ``Span``, one half-open range ``[start, end)``, and ``Spans``, an ordered set of them.

Operators, each with one meaning:

- ``inner in span``: containment. ``inner`` (a ``Span``) lies inside ``span``, or ``inner`` (an ``int``) is a position
  in it.
- ``span in spans``: membership, an exact match of one of the set's spans. For coverage use ``spans.container_of``.
- ``a & b``: the characters two spans share, as a ``Span``, or ``None`` when they share none.
- ``a - b``: the characters of ``a`` outside ``b``, as ``Spans``.
- ``a < b``: sort order by start, then end. It is not the subset test.
- ``spans | other``: the spans of both sets.
- ``len(spans)``, iteration and indexing: the spans in order. A slice is a plain tuple.

``text[span.slice]`` is the text a span covers.
"""

from __future__ import annotations

import builtins
import operator
from bisect import bisect_right
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from operator import attrgetter
from typing import Any, NoReturn, overload

_BOUNDS = attrgetter("start", "end")


@dataclass(frozen=True, order=True, slots=True)
class Span:
    """A half-open character range ``[start, end)`` with ``0 <= start <= end``; empty when ``start == end``.

    Bounds are integers (``bool`` is rejected; NumPy integers are converted).
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        for name in ("start", "end"):
            bound = getattr(self, name)
            if isinstance(bound, bool):
                raise TypeError(f"Span bounds must be integers, got {bound!r}")
            try:
                object.__setattr__(self, name, operator.index(bound))
            except TypeError:
                raise TypeError(f"Span bounds must be integers, got {bound!r}") from None
        if not 0 <= self.start <= self.end:
            raise ValueError(f"Span needs 0 <= start <= end, got start={self.start}, end={self.end}")

    def __repr__(self) -> str:
        return f"Span({self.start}, {self.end})"

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def slice(self) -> builtins.slice:
        """The span as a slice, so ``text[span.slice]`` is the text it covers."""
        return slice(self.start, self.end)

    def __contains__(self, inner: object) -> bool:
        match inner:
            case Span(start, end):
                return self.start <= start and end <= self.end
            case int(position) if not isinstance(position, bool):
                return self.start <= position < self.end
            case _:
                return False

    def __and__(self, other: Span) -> Span | None:
        start, end = max(self.start, other.start), min(self.end, other.end)
        return Span(start, end) if start < end else None

    def __sub__(self, other: Span) -> Spans:
        match self & other:
            case Span(start, end):
                return Spans(part for part in (Span(self.start, start), Span(end, self.end)) if part.length)
            case None if self.length:
                return Spans([self])
            case _:
                return Spans()  # an empty span has no characters to keep

    def overlaps(self, other: Span) -> bool:
        """Whether the two spans share a character (touching spans do not)."""
        return self & other is not None

    def shifted(self, offset: int) -> Span:
        return Span(self.start + offset, self.end + offset)


class Spans(Sequence[Span]):
    """An immutable set of spans, ordered by start and then end; duplicates collapse."""

    __slots__ = ("_outer", "_outer_starts", "_spans")

    _spans: tuple[Span, ...]
    _outer: tuple[Span, ...] | None
    _outer_starts: tuple[int, ...]

    def __init__(self, spans: Iterable[Span] = ()) -> None:
        unique = dict.fromkeys(spans)
        for span in unique:
            if not isinstance(span, Span):
                raise TypeError(f"Spans holds Span values, got {span!r}")
        object.__setattr__(self, "_spans", tuple(sorted(unique, key=_BOUNDS)))
        object.__setattr__(self, "_outer", None)  # outermost spans, computed on first use

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("Spans is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Spans is immutable")

    def __reduce__(self) -> tuple[type[Spans], tuple[tuple[Span, ...]]]:
        return Spans, (self._spans,)

    @overload
    def __getitem__(self, index: int) -> Span: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[Span, ...]: ...
    def __getitem__(self, index: int | slice) -> Span | tuple[Span, ...]:
        return self._spans[index]

    def __len__(self) -> int:
        return len(self._spans)

    def __iter__(self) -> Iterator[Span]:
        return iter(self._spans)

    def __contains__(self, span: object) -> bool:
        if not isinstance(span, Span):
            return False
        i = bisect_right(self._spans, _BOUNDS(span), key=_BOUNDS) - 1
        return i >= 0 and self._spans[i] == span

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Spans) and self._spans == other._spans

    def __hash__(self) -> int:
        return hash(self._spans)

    def __repr__(self) -> str:
        return f"Spans([{', '.join(map(repr, self._spans))}])"

    def __or__(self, other: Spans) -> Spans:
        return Spans((*self, *other))

    def merged(self) -> Spans:
        """The characters the spans cover, as the fewest spans: overlapping and touching spans join, and empty spans,
        which cover nothing, are dropped."""

        def runs() -> Iterator[Span]:
            current: Span | None = None
            for span in (span for span in self if span.length):
                match current:
                    case None:
                        current = span
                    case Span(start, end) if span.start <= end:
                        current = Span(start, max(end, span.end))
                    case _:
                        yield current
                        current = span
            if current is not None:
                yield current

        return Spans(runs())

    def outermost(self) -> Spans:
        """The spans no other span contains. Ordered by start, their ends strictly increase too."""
        return Spans(self._outermost())

    def _outermost(self) -> tuple[Span, ...]:
        """``outermost`` as a tuple, computed once (with the starts ``container_of`` bisects)."""
        outer = self._outer
        if outer is None:

            def uncontained(spans: Iterable[Span]) -> Iterator[Span]:
                furthest = -1
                for span in spans:  # by start, longest first: a later span is inside an earlier one or reaches past it
                    if span.end > furthest:
                        furthest = span.end
                        yield span

            outer = tuple(uncontained(sorted(self._spans, key=lambda span: (span.start, -span.end))))
            # Starts first: a thread that sees ``_outer`` set must also see ``_outer_starts``.
            object.__setattr__(self, "_outer_starts", tuple(span.start for span in outer))
            object.__setattr__(self, "_outer", outer)
        return outer

    def container_of(self, inner: Span) -> Span | None:
        """The outermost span that contains ``inner`` and starts last, which is the one reaching furthest; ``None``
        when no span contains it.

        Outermost spans ordered by start have increasing ends, so the last one starting at or before ``inner`` is the
        only one that can reach past every other candidate. It contains ``inner`` exactly when some span does, however
        the spans overlap.
        """
        outer = self._outermost()
        k = bisect_right(self._outer_starts, inner.start) - 1
        return outer[k] if k >= 0 and inner in outer[k] else None

    def uncovered(self, inner: Iterable[Span]) -> Spans:
        """The spans of ``inner`` that no span of this set contains."""
        return Spans(span for span in inner if self.container_of(span) is None)
