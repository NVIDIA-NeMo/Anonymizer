# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Edits on one source text: decide which edits stand, then apply them.

An ``Edit`` replaces a ``Span`` of the source with its replacement text; an empty span is an insertion. ``Edits``
holds its source and a set of edits that can all be applied: each inside the source, sorted by ``BY_START_END``, and
each preceding the next (``Span.precedes``). Two edits conflict unless one precedes the other, so an insertion
stands beside another edit unless it lies strictly inside it, and several insertions at one point apply in input
order.

``Edits(source, edits)`` is strict: it raises ``OutOfBounds`` or ``Conflicting``. ``Edits.admit`` names a policy for
edits that may be out of range or conflict: ``Strict()`` raises on a conflict, ``Prefer(order)`` walks each
conflicting group in ``order`` and keeps every edit that conflicts with none kept so far. Policies are frozen values,
so a record's edits and its policy pickle.

Characters outside every edit pass through unchanged: ``pieces()`` yields gaps and edits that partition the source.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from typing import Literal, NoReturn, SupportsIndex, overload

from anonymizer.engine.spans import BY_START_END, Order, Span, group_overlapping_spans


@dataclass(frozen=True, slots=True)
class Edit:
    """Replace ``span`` of the source with ``replacement``; ``label`` is carried for reporting and never read here."""

    span: Span
    replacement: str
    label: str = ""


@dataclass(frozen=True, slots=True)
class Gap:
    """Source text between edits, kept as it is."""

    span: Span
    text: str


class OutOfBounds(ValueError):
    """A range that is not inside the source (or whose bounds are not integers)."""

    def __init__(self, start: object, end: object, length: int) -> None:
        super().__init__(f"{start}:{end} is not a range of a text of length {length}")
        self.start, self.end = start, end


class Conflicting(ValueError):
    """Two edits that cannot both be applied: neither precedes the other."""

    def __init__(self, earlier: Span, later: Span) -> None:
        super().__init__(f"{earlier} and {later} conflict")
        self.earlier, self.later = earlier, later


def conflicts(a: Span, b: Span) -> bool:
    """Whether two edits on these spans cannot both be applied."""
    return not (a.precedes(b) or b.precedes(a))


@dataclass(frozen=True, slots=True)
class Strict:
    """Any conflict raises ``Conflicting``."""

    kind: Literal["strict"] = "strict"


@dataclass(frozen=True, slots=True)
class Prefer:
    """In each conflicting group, walk the edits in ``order`` and keep each that conflicts with none kept so far."""

    order: Order
    kind: Literal["prefer"] = "prefer"


Policy = Strict | Prefer


@dataclass(frozen=True, slots=True)
class Rejected:
    edit: Edit
    reason: Literal["out_of_bounds", "conflict"]


def check(length: int, bounds: Iterable[tuple[object, object]]) -> None:
    """Raise unless every ``(start, end)`` is a range of a text of ``length`` and, in ``BY_START_END`` order, each
    precedes the next. A bound that is not an integer (``bool`` included) raises ``OutOfBounds`` before any range is
    compared; then the first problem in order raises ``OutOfBounds`` or ``Conflicting``. For a caller that must know
    its ranges are sound before it renders any replacement."""
    pairs: list[tuple[int, int]] = []
    for start, end in bounds:
        first, last = _index(start), _index(end)
        if first is None or last is None:
            raise OutOfBounds(start, end, length)
        pairs.append((first, last))
    previous: Span | None = None
    for start, end in sorted(pairs):
        if not 0 <= start <= end <= length:
            raise OutOfBounds(start, end, length)
        span = Span(start, end)
        if previous is not None and not previous.precedes(span):
            raise Conflicting(previous, span)
        previous = span


def _index(value: object) -> int | None:
    """``value`` as an int when it is an integer other than a ``bool`` (NumPy integers included), else ``None``."""
    if isinstance(value, bool) or not isinstance(value, SupportsIndex):
        return None
    return operator.index(value)


class Edits(Sequence[Edit]):
    """Edits on one source text, which it holds: each inside it, in ``BY_START_END`` order, each preceding the next."""

    __slots__ = ("_edits", "source")

    source: str
    _edits: tuple[Edit, ...]

    def __init__(self, source: str, edits: Iterable[Edit] = ()) -> None:
        ordered = BY_START_END.sorted(edits, span=lambda e: e.span)
        for i, edit in enumerate(ordered):
            if edit.span.end > len(source):
                raise OutOfBounds(edit.span.start, edit.span.end, len(source))
            if i and not ordered[i - 1].span.precedes(edit.span):
                raise Conflicting(ordered[i - 1].span, edit.span)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "_edits", tuple(ordered))

    def __setattr__(self, name: str, value: object) -> NoReturn:
        raise AttributeError("Edits is immutable")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("Edits is immutable")

    def __reduce__(self) -> tuple[type[Edits], tuple[str, tuple[Edit, ...]]]:
        return Edits, (self.source, self._edits)

    @classmethod
    def admit(cls, source: str, edits: Iterable[Edit], policy: Policy) -> tuple[Edits, tuple[Rejected, ...]]:
        """The edits that stand under ``policy``, and the rest with their reason, in input order.

        An edit reaching past the end of ``source`` is rejected (``out_of_bounds``). Among the others, each group of
        conflicting edits goes to ``policy``: ``Strict()`` raises ``Conflicting``, ``Prefer(order)`` keeps some and
        rejects the rest (``conflict``).
        """
        items = list(edits)
        reasons: dict[int, Literal["out_of_bounds", "conflict"]] = {
            i: "out_of_bounds" for i, edit in enumerate(items) if edit.span.end > len(source)
        }
        live = BY_START_END.sorted((i for i in range(len(items)) if i not in reasons), span=lambda i: items[i].span)
        for group in group_overlapping_spans(live, span=lambda i: items[i].span):
            if len(group) == 1:
                continue
            members = sorted(group)  # input order, which every Order ends with
            if isinstance(policy, Strict):
                first, second = BY_START_END.sorted(members, span=lambda i: items[i].span)[:2]
                raise Conflicting(items[first].span, items[second].span)
            kept: list[Span] = []
            for i in policy.order.sorted(members, span=lambda i: items[i].span, label=lambda i: items[i].label):
                if any(conflicts(items[i].span, k) for k in kept):
                    reasons[i] = "conflict"
                else:
                    kept.append(items[i].span)
        standing = cls(source, (items[i] for i in range(len(items)) if i not in reasons))
        return standing, tuple(Rejected(items[i], reasons[i]) for i in sorted(reasons))

    @overload
    def __getitem__(self, index: int) -> Edit: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[Edit, ...]: ...
    def __getitem__(self, index: int | slice) -> Edit | tuple[Edit, ...]:
        return self._edits[index]

    def __len__(self) -> int:
        return len(self._edits)

    def __iter__(self) -> Iterator[Edit]:
        return iter(self._edits)

    def __repr__(self) -> str:
        return f"Edits({self.source!r}, {list(self._edits)!r})"

    def pieces(self) -> Iterator[Gap | Edit]:
        """Gaps and edits in order: the gaps' texts and the edits' spans partition the source."""
        at = 0
        for edit in self._edits:
            yield Gap(Span(at, edit.span.start), self.source[at : edit.span.start])
            yield edit
            at = edit.span.end
        yield Gap(Span(at, len(self.source)), self.source[at:])

    def apply(self) -> str:
        """The source with every edit applied."""
        return "".join(piece.text if isinstance(piece, Gap) else piece.replacement for piece in self.pieces())

    def applied(self) -> Edits:
        """The output as ``Edits`` on it: each edit moved to where its replacement landed."""
        placed: list[Edit] = []
        at = 0
        for piece in self.pieces():
            if isinstance(piece, Gap):
                at += len(piece.text)
            else:
                placed.append(replace(piece, span=Span(at, at + len(piece.replacement))))
                at += len(piece.replacement)
        return Edits(self.apply(), placed)
