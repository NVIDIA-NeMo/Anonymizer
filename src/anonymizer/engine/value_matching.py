# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rule for whether a replaced value still occurs in a text, and how often, shared by Rewrite and Replace."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

from anonymizer.engine.spans import Span, Spans


def any_value_occurs(values: Iterable[str], text: str) -> bool:
    """Whether any of ``values`` occurs in ``text`` under ``_occurs``."""
    return compile_values(values)(text)


def compile_values(values: Iterable[str]) -> Callable[[str], bool]:
    """A test of whether any of ``values`` occurs in a text under ``_occurs``, normalizing and compiling each value
    once so that one set of values can be checked against many texts."""
    matchers: list[tuple[bool, Callable[[str], bool]]] = []  # (matches with its case, test)
    for value in {_normalized(v) for v in values} - {""}:
        cased = _short_code(value)
        key = value if cased else value.casefold()
        if _unspaced(key[0]) and _unspaced(key[-1]):
            if len(key) >= 2:
                matchers.append((cased, lambda text, key=key: key in text))
        else:
            pattern = re.compile(r"(?<![^\W_])" + re.escape(key) + r"(?![^\W_])")
            matchers.append((cased, lambda text, search=pattern.search: search(text) is not None))

    def occurs(text: str) -> bool:
        plain = _normalized(text)
        folded = plain.casefold()
        return any(test(plain if cased else folded) for cased, test in matchers)

    return occurs


@dataclass(frozen=True)
class Leftover:
    """One detected value of a row, as the leftover checks count it under ``_occurs``.

    ``value`` is the value counted, as first given. ``position`` is the entity a cause names (``e<position>``): the
    value's own detected span; else the first of the spans holding its matches, where a match's holder is the
    containing outermost span that starts last (``Spans.container_of``); else ``None`` (``e-``). ``label`` and
    ``span`` are that entity's, or ``None``. ``in_source`` counts the value's matches in the source,
    ``undetected_spans`` are those no detected span contains, and ``output_spans`` are its matches in the output.
    Holds record text: only ``format_cause`` may reach a check's cause.
    """

    position: int | None
    value: str
    label: str | None
    span: Span | None
    in_source: int
    undetected_spans: Spans
    output_spans: Spans

    @property
    def detected(self) -> int:
        """D: the source matches that a detected span contains."""
        return self.in_source - len(self.undetected_spans)

    @property
    def in_output(self) -> int:
        """O: the value's matches in the output."""
        return len(self.output_spans)

    @property
    def over_bound(self) -> bool:
        """The check fails: the output holds the value more often than the source holds it outside the detected
        spans (O > N - D)."""
        return self.in_output > len(self.undetected_spans)

    @property
    def undetected_kept(self) -> bool:
        """The row is flagged: the output holds a value that the source also holds outside the detected spans."""
        return bool(self.undetected_spans) and bool(self.output_spans)


def count_values(
    source: str, entities: Sequence[tuple[Span | None, str]], output: str, values: Iterable[str] | None = None
) -> tuple[Leftover, ...]:
    """Count each distinct normalized value (default: the detected spans' source text) in ``source`` and ``output``,
    ordered by position, values with no position last.

    ``entities`` are the row's detected ``(span, label)`` pairs in position order (sorted by start), each span inside
    ``source``; an entity without a usable span has ``None`` and keeps its position. Every match start counts,
    overlapping matches included, and a match is detected when some detected span contains it.
    """
    usable = {i: span for i, (span, _) in enumerate(entities) if span is not None}
    detected = Spans(usable.values())
    position_of: dict[Span, int] = {}  # the first entity with each span
    own: dict[str, int] = {}  # the first entity whose text is each normalized value
    for i, span in usable.items():
        position_of.setdefault(span, i)
        own.setdefault(_normalized(source[span.slice]), i)
    if values is None:
        values = [source[span.slice] for span in usable.values()]
    given: dict[str, str] = {}  # each normalized value's first given form
    for value in values:
        given.setdefault(_normalized(value), value)
    source_forms, output_forms = _mapped(source), _mapped(output)
    leftovers: list[tuple[str, Leftover]] = []
    for key, matcher in _compiled(given).items():
        form = 0 if matcher.cased else 1
        matches = Spans(matcher.spans(*source_forms[form]))
        position = own.get(key)
        if position is None:
            containers = (detected.container_of(match) for match in matches)
            position = min((position_of[c] for c in containers if c is not None), default=None)
        span, label = (None, None) if position is None else entities[position]
        leftover = Leftover(
            position=position,
            value=given[key],
            label=label,
            span=span,
            in_source=len(matches),
            undetected_spans=detected.uncovered(matches),
            output_spans=Spans(matcher.spans(*output_forms[form])),
        )
        leftovers.append((key, leftover))
    leftovers.sort(key=lambda item: (item[1].position is None, item[1].position or 0, item[0]))
    return tuple(leftover for _, leftover in leftovers)


def format_cause(leftovers: Iterable[Leftover]) -> str:
    """``e<position>:N<n>,D<d>,O<o>`` for each leftover, in the given order, joined by ``;``, with ``e-`` for no
    position: positions and counts only, never record text."""
    return ";".join(
        f"e{'-' if x.position is None else x.position}:N{x.in_source},D{x.detected},O{x.in_output}" for x in leftovers
    )


@dataclass(frozen=True)
class _Matcher:
    """One normalized value as ``_occurs`` matches it: ``key`` (casefolded unless ``cased``) and, unless the value is
    matched as a substring, the ``pattern`` that checks its word boundaries."""

    key: str
    cased: bool
    pattern: re.Pattern[str] | None

    def spans(self, text: str, origin: list[int]) -> Iterator[Span]:
        """Every match in the normalized ``text``, overlapping ones included, as a span of the original text
        (``origin`` maps each character back). ``str.find`` proposes each start; the pattern checks it."""
        pos = text.find(self.key)
        while pos >= 0:
            if self.pattern is None or self.pattern.match(text, pos):
                yield Span(origin[pos], origin[pos + len(self.key) - 1] + 1)
            pos = text.find(self.key, pos + 1)


def _compiled(values: Iterable[str]) -> dict[str, _Matcher]:
    """One matcher per distinct normalized value; a blank value and a one-character unspaced value get none."""
    matchers: dict[str, _Matcher] = {}
    for value in {_normalized(v) for v in values} - {""}:
        cased = _short_code(value)
        key = value if cased else value.casefold()
        if not (_unspaced(key[0]) and _unspaced(key[-1])):
            matchers[value] = _Matcher(key, cased, re.compile(r"(?<![^\W_])" + re.escape(key) + r"(?![^\W_])"))
        elif len(key) >= 2:
            matchers[value] = _Matcher(key, cased, None)
    return matchers


def _normalized(text: str) -> str:
    """NFKD with combining marks dropped and whitespace collapsed."""
    decomposed = unicodedata.normalize("NFKD", text)
    return " ".join("".join(c for c in decomposed if unicodedata.category(c) != "Mn").split())


def _mapped(text: str) -> tuple[tuple[str, list[int]], tuple[str, list[int]]]:
    """``_normalized(text)`` and its casefold, each with the index in ``text`` of every character's source character.

    Built one source character at a time; equal to the whole-string forms except for canonical reordering of
    combining marks that are not ``Mn``.
    """
    plain: list[str] = []
    origin: list[int] = []
    pending = False
    for i, c in enumerate(text):
        for d in unicodedata.normalize("NFKD", c) if not c.isascii() else c:
            if d.isspace():
                pending = bool(plain)
            elif unicodedata.category(d) != "Mn":
                if pending:
                    plain.append(" ")
                    origin.append(i)
                    pending = False
                plain.append(d)
                origin.append(i)
    folded: list[str] = []
    folded_origin: list[int] = []
    for d, i in zip(plain, origin):
        f = d.casefold()
        folded.append(f)
        folded_origin.extend([i] * len(f))
    return ("".join(plain), origin), ("".join(folded), folded_origin)


# Scripts written without spaces between words: Han, Hiragana, Katakana, Thai, Lao, Khmer, Myanmar.
_UNSPACED = (
    (0x0E00, 0x0EFF),
    (0x1000, 0x109F),
    (0x1780, 0x17FF),
    (0x19E0, 0x19FF),
    (0x3040, 0x30FF),
    (0x31F0, 0x31FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xA9E0, 0xA9FF),
    (0xAA60, 0xAA7F),
    (0xF900, 0xFAFF),
    (0x20000, 0x323AF),
)


def _unspaced(char: str) -> bool:
    return any(low <= ord(char) <= high for low, high in _UNSPACED)


def _short_code(value: str) -> bool:
    """A value of at most three characters, all uppercase: matched with its case."""
    return len(value) <= 3 and value.isupper()


def _occurs(value: str, text: str, folded: str) -> bool:
    """Whether ``value`` occurs in ``text`` (both ``_normalized``; ``folded`` is ``text`` casefolded).

    Case-insensitive unless ``_short_code``. A value whose first and last characters are in ``_UNSPACED`` matches as a
    substring when it has two or more characters, and never when it has one; any other value matches where no letter
    or digit directly precedes or follows it.
    """
    if not value:
        return False
    if not _short_code(value):
        value, text = value.casefold(), folded
    if _unspaced(value[0]) and _unspaced(value[-1]):
        return len(value) >= 2 and value in text
    return re.search(r"(?<![^\W_])" + re.escape(value) + r"(?![^\W_])", text) is not None
