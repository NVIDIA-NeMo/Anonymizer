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
- ``a.precedes(b)``: ``a`` ends at or before ``b`` starts, so the two apply in this order. Two edits conflict unless
  one precedes the other: an insertion (an empty span) stands beside another edit unless it lies strictly inside it.
- ``spans | other``: the spans of both sets.
- ``len(spans)``, iteration and indexing: the spans in order. A slice is a plain tuple.

``text[span.slice]`` is the text a span covers.

Every sort over spans is stable: items whose keys tie keep the order their producer gave. An ``Order`` names a sort
key and ends with ``position``, the input order, so the last tie rule is written. The edits layer uses ``BY_START``
and ``BY_START_END``; ``OUTER_FIRST`` (outer spans first) and ``CONTENT_ORDER`` (a tie on bounds broken by label, for
a caller that wants one output whatever order a producer lists its spans in) are provided for callers that need them.
"""

from __future__ import annotations

import builtins
import operator
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from operator import attrgetter
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NoReturn, Protocol, TypeVar, overload

from anonymizer.config.entity_labels import normalize_entity_label

_BOUNDS = attrgetter("start", "end")
_ItemT = TypeVar("_ItemT")
_RangedT = TypeVar("_RangedT", bound="_Ranged")


@dataclass(frozen=True, order=True, slots=True)
class Span:
    """A half-open character range ``[start, end)`` with ``0 <= start <= end``; empty when ``start == end``.

    Bounds are integers (``bool`` is rejected; NumPy integers are converted).
    """

    start: int
    end: int

    def __post_init__(self) -> None:
        if type(self.start) is not int or type(self.end) is not int:  # plain ints, the common case, need no check
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

    @classmethod
    def in_text(cls, start: int, end: int, text: str) -> Span | None:
        """``Span(start, end)`` when the bounds are a range of ``text`` holding at least one character; ``None`` for an
        empty, inverted or out-of-range pair."""
        return cls(start, end) if 0 <= start < end <= len(text) else None

    @classmethod
    def cover(cls, spans: Iterable[Span]) -> Span:
        """The smallest span containing every one of ``spans``, which must not be empty."""
        items = tuple(spans)
        if not items:
            raise ValueError("Span.cover needs at least one span")
        return cls(min(span.start for span in items), max(span.end for span in items))

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

    def precedes(self, other: Span) -> bool:
        """This span ends at or before ``other`` starts, so the two apply in this order."""
        return self.end <= other.start

    def shifted(self, offset: int) -> Span:
        return Span(self.start + offset, self.end + offset)


def geometry_problem(noun: str, start: int, end: int, value: str | None = None, *, empty: bool = False) -> str:
    """The first rule a record's range breaks, as its validation message, or "": the end does not precede the start
    (``empty``) or the range holds at least one character, and a ``value`` fills it exactly."""
    if empty:
        if end < start:
            return f"{noun} end must not precede its start"
    elif end <= start:
        return f"{noun} must cover a non-empty span"
    if value is not None and len(value) != end - start:
        return f"{noun} value length must match its source bounds"
    return ""


class Positioned:
    """A record with ``start_position`` and ``end_position``: its ``span``, and the one geometry rule for a validator.

    It holds no fields, so a pydantic model or a dataclass keeps its own fields and key order. ``_noun`` names the
    record in messages; ``_empty`` allows an empty range.
    """

    __slots__ = ()
    _noun: ClassVar[str]
    _empty: ClassVar[bool] = False
    if TYPE_CHECKING:
        start_position: int
        end_position: int

    @property
    def span(self) -> Span:
        return Span(self.start_position, self.end_position)

    def _check_geometry(self, value: str | None = None) -> None:
        """Raise ``ValueError`` with ``geometry_problem``'s message when the range breaks the rule."""
        if problem := geometry_problem(self._noun, self.start_position, self.end_position, value, empty=self._empty):
            raise ValueError(problem)


class _Ranged(Protocol):
    @property
    def start_position(self) -> int: ...
    @property
    def end_position(self) -> int: ...


def within(length: int, records: Iterable[_RangedT]) -> Iterator[tuple[_RangedT, Span]]:
    """Each record whose range lies in a text of ``length`` (an empty range included), with its span, in input
    order; a record whose range does not is skipped."""
    return (
        (record, Span(record.start_position, record.end_position))
        for record in records
        if 0 <= record.start_position <= record.end_position <= length
    )


OrderField = Literal["start", "end", "-end", "label", "position"]


@dataclass(frozen=True, slots=True)
class Order:
    """A sort order over spans, as a value: the fields compared in turn, the last always ``position`` (input order).

    ``label`` compares labels by their key (``normalize_entity_label``).
    """

    fields: tuple[OrderField, ...]

    def __post_init__(self) -> None:
        if not self.fields or self.fields[-1] != "position" or "position" in self.fields[:-1]:
            raise ValueError(f"an Order ends with 'position' and names it once, got {self.fields}")
        if unknown := set(self.fields) - {"start", "end", "-end", "label", "position"}:
            raise ValueError(f"unknown Order fields {sorted(unknown)}")

    def key(self, span: Span, label: str, position: int) -> tuple[int | str, ...]:
        values: dict[str, int | str] = {"start": span.start, "end": span.end, "-end": -span.end, "position": position}
        if "label" in self.fields:
            values["label"] = normalize_entity_label(label)
        return tuple(values[name] for name in self.fields)

    def sorted(
        self, items: Iterable[_ItemT], *, span: Callable[[_ItemT], Span], label: Callable[[_ItemT], str] | None = None
    ) -> list[_ItemT]:
        """``items`` in this order; ``label`` is read only when the order names it."""
        indexed = list(enumerate(items))
        read = label or (lambda _item: "")
        indexed.sort(key=lambda pair: self.key(span(pair[1]), read(pair[1]), pair[0]))
        return [item for _, item in indexed]


BY_START = Order(("start", "position"))
"""Earlier start first; spans that start together keep input order."""
BY_START_END = Order(("start", "end", "position"))
"""Earlier, then shorter; identical spans keep input order."""
OUTER_FIRST = Order(("start", "-end", "position"))
"""Earlier, then longer; identical spans keep input order."""
CONTENT_ORDER = Order(("start", "end", "label", "position"))
"""Earlier, then shorter, then by label key: identical spans come out in one order whatever order they came in."""


def group_overlapping_spans(
    ordered_items: Iterable[_ItemT],
    *,
    span: Callable[[_ItemT], Span],
) -> list[tuple[_ItemT, ...]]:
    """Group start-ordered items into connected components of overlapping spans.

    Callers supply items sorted by span start and choose their own tie
    ordering. An item joins the current group when its span starts before
    the group's furthest end, so touching spans remain separate while nested
    and transitively overlapping spans stay together. On items sorted by
    ``BY_START_END`` the groups are exactly the groups of edits that conflict
    (neither ``precedes`` the other), insertions included.
    """
    groups: list[list[_ItemT]] = []
    reach = 0  # the current group's furthest end; unread until the first item starts a group
    for item in ordered_items:
        item_span = span(item)
        if groups and item_span.start < reach:
            groups[-1].append(item)
            reach = max(reach, item_span.end)
        else:
            groups.append([item])
            reach = item_span.end
    return [tuple(group) for group in groups]


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
