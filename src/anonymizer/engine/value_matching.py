# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rule for whether a replaced value still occurs in a text, and how often, shared by Rewrite and Replace."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum

import regex

from anonymizer.engine.schemas import EntitySchema
from anonymizer.engine.spans import Span, Spans


class MatchMode(StrEnum):
    """How ``count_values`` matches a value. ``EXACT``: the value as spelled, whole words; for every check that edits
    text or fails a row, so a common word that folds to a detected name ("may" for ``May``) is left alone. ``FOLDED``:
    case and accents folded (``normalized``, casefold unless ``_short_code``), whole words; for advisories and audits,
    where a variant spelling is flagged, never edited.

    ``EXACT`` does not normalize: spans are sliced from the same text the value came from, so it compares code
    points, keeps whitespace and compatibility forms as spelled ("Mary Jane" does not match "Mary\\nJane", "Alice"
    does not match fullwidth "Ａlice"), and never matches a word that continues with a combining mark ("Le" in NFD
    "Lê"). The folded audit still counts mixed normalization forms.

    Both modes are configurations of one pipeline (``_PIPELINES``). In both, a match never starts on, or ends
    before, a mark that survives the mode's normalization or a spacing mark (``_GLUE_CLASS``). A zero-width joiner or a
    variation selector right after a value blocks it only when it joins an emoji or a mark in the source
    (``_end_edge``): "Alice" matches before a joiner, an emoji value inside a joined sequence does not. A zero-width
    non-joiner, an emoji modifier or a Prepend character next to a value never blocks it. ``FOLDED`` also drops
    invisible format characters (``_DropIgnorables``), so "An\u00ada" matches "Ana"; a dropped zero-width space or word
    joiner still separates two words."""

    EXACT = "exact"
    FOLDED = "folded"


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
    source: str,
    entities: Sequence[tuple[Span | None, str]],
    output: str,
    values: Iterable[str] | None = None,
    *,
    mode: MatchMode,
) -> tuple[Leftover, ...]:
    """Count each distinct value (default: the detected spans' source text), keyed by ``value_key`` under ``mode``, in
    ``source`` and ``output``, ordered by position, values with no position last.

    ``entities`` are the row's detected ``(span, label)`` pairs in position order (sorted by start), each span inside
    ``source``; an entity without a usable span has ``None`` and keeps its position. Every match start counts,
    overlapping matches included, and a match is detected when some detected span contains it.
    """
    pipeline = _PIPELINES[mode]
    usable = {i: span for i, (span, _) in enumerate(entities) if span is not None}
    detected = Spans(usable.values())
    position_of: dict[Span, int] = {}  # the first entity with each span
    own: dict[str, int] = {}  # the first entity whose text is each key
    for i, span in usable.items():
        position_of.setdefault(span, i)
        own.setdefault(pipeline.key(source[span.slice]), i)
    if values is None:
        values = [source[span.slice] for span in usable.values()]
    given: dict[str, str] = {}  # each key's first given form
    for value in values:
        given.setdefault(pipeline.key(value), value)
    matchers = _compiled(given, pipeline)
    if not matchers:
        return ()
    source_views, output_views = pipeline.views(source), pipeline.views(output)
    leftovers: list[tuple[str, Leftover]] = []
    for key, matcher in matchers.items():
        form = 0 if matcher.cased else 1
        matches = Spans(matcher.spans(source_views[form]))
        output_spans = Spans(matcher.spans(output_views[form]))
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
            output_spans=output_spans,
        )
        leftovers.append((key, leftover))
    leftovers.sort(key=lambda item: (item[1].position is None, item[1].position or 0, item[0]))
    return tuple(leftover for _, leftover in leftovers)


def entity_spans(source: str, entities: Iterable[EntitySchema]) -> list[tuple[Span | None, str]]:
    """``entities`` as ``count_values`` takes them: ``(span, label)`` in position order (sorted by start). An entity
    whose offsets are not a non-empty span of ``source`` holding its value has ``None``: it keeps its position but
    holds nothing."""

    def span_of(entity: EntitySchema) -> Span | None:
        span = Span.in_text(entity.start_position, entity.end_position, source)
        return span if span is not None and source[span.slice] == entity.value else None

    return [(span_of(e), e.label) for e in sorted(entities, key=lambda e: e.start_position)]


def format_cause(leftovers: Iterable[Leftover]) -> str:
    """``e<position>:N<n>,D<d>,O<o>`` for each leftover, in the given order, joined by ``;``, with ``e-`` for no
    position: positions and counts only, never record text."""
    return ";".join(
        f"e{'-' if x.position is None else x.position}:N{x.in_source},D{x.detected},O{x.in_output}" for x in leftovers
    )


# --- The matching pipeline ------------------------------------------------------------------------------------------
#
# Like ``tokenizers``: normalizer steps over an aligned string (every character keeps the index of the source
# character it came from), a segmentation step that says where a match may start or end, and a matcher. ``EXACT``
# and ``FOLDED`` are two configurations of the same pipeline (``_PIPELINES``), not branches. Each step has a plain
# form for keys (``text``, C-speed where ``unicodedata`` allows) and an aligned form for the texts searched
# (``aligned``); ``test_value_matching`` checks that the two agree.

_Chars = list[tuple[str, int]]  # each character with the index of its source character


class _Nfkd:
    """NFKD (the identity on ASCII). Aligned: per character, then every run of combining marks put in canonical order
    with the source indexes carried along, which is what NFKD does to the whole string; inside a reordered run the
    indexes do not increase."""

    ascii_identity = True

    @staticmethod
    def text(text: str) -> str:
        return text if text.isascii() else unicodedata.normalize("NFKD", text)

    @staticmethod
    def aligned(chars: _Chars) -> _Chars:
        out = [(d, i) for c, i in chars for d in (unicodedata.normalize("NFKD", c) if not c.isascii() else c)]
        start = 0
        while start < len(out):
            if not unicodedata.combining(out[start][0]):
                start += 1
                continue
            end = start
            while end < len(out) and unicodedata.combining(out[end][0]):
                end += 1
            out[start:end] = sorted(out[start:end], key=lambda pair: unicodedata.combining(pair[0]))
            start = end
        return out


class _DropIgnorables:
    """Drop invisible format characters (Unicode default-ignorable code points, as NFKC_Casefold does: soft hyphen,
    zero-width space, word joiner, bidi marks and isolates, BOM), so "An\u00ada" folds to "Ana". The zero-width joiner
    and non-joiner stay: they decide where a match may end (``_end_edge``) and, in Persian, where a word ends."""

    ascii_identity = True

    @staticmethod
    def text(text: str) -> str:
        return text if text.isascii() else _IGNORABLE.sub("", text)

    @staticmethod
    def aligned(chars: _Chars) -> _Chars:
        return [(c, i) for c, i in chars if c.isascii() or _IGNORABLE.match(c) is None]


_IGNORABLE = regex.compile(r"[\p{Default_Ignorable_Code_Point}--[\u200c\u200d]]", regex.V1)


class _StripMarks:
    """Drop nonspacing marks (``Mn``): accents fold away."""

    ascii_identity = True

    @staticmethod
    def text(text: str) -> str:
        return text if text.isascii() else "".join(c for c in text if unicodedata.category(c) != "Mn")

    @staticmethod
    def aligned(chars: _Chars) -> _Chars:
        return [(c, i) for c, i in chars if c.isascii() or unicodedata.category(c) != "Mn"]


class _CollapseWhitespace:
    """Every run of whitespace as one space, taking the index of the character after it; none at the edges."""

    ascii_identity = False

    @staticmethod
    def text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def aligned(chars: _Chars) -> _Chars:
        out: _Chars = []
        pending = False
        for c, i in chars:
            if c.isspace():
                pending = bool(out)
            else:
                if pending:
                    out.append((" ", i))
                    pending = False
                out.append((c, i))
        return out


def _casefold(chars: _Chars) -> _Chars:
    """``str.casefold`` per character; an expansion ("ß" to "ss") keeps the one source index."""
    return [(f, i) for c, i in chars for f in (c.casefold() if not c.isascii() else c.lower())]


_Step = type[_Nfkd] | type[_DropIgnorables] | type[_StripMarks] | type[_CollapseWhitespace]

# Characters that continue the cluster before them, so a match never starts on one or ends before one: combining
# marks (``\p{M}``, including Myanmar and other spacing marks whose UAX #29 class is Other), the UAX #29 SpacingMark
# class and the zero-width joiner. Other UAX #29 Extend characters (ZWNJ, emoji modifiers, tags) and Prepend characters
# do not glue: a value next to them still matches.
_GLUE_CLASS = r"[\p{M}\p{Grapheme_Cluster_Break=SpacingMark}\u200d]"
_GLUE = regex.compile(_GLUE_CLASS)
_CONTROL = regex.compile(
    r"[\p{Grapheme_Cluster_Break=Control}\p{Grapheme_Cluster_Break=CR}\p{Grapheme_Cluster_Break=LF}]"
)


def _cluster_edge(text: str, k: int) -> bool:
    """The segmentation step, at one position: whether a match may start or end at ``k``. No match starts on, or ends
    before, a ``_GLUE`` character, unless it follows a control (UAX #29 GB4); the end of the text is always an edge."""
    if k >= len(text):
        return True
    if text[k].isascii():  # no ASCII character glues
        return True
    if k > 0 and _CONTROL.match(text, k - 1):
        return True
    return _GLUE.match(text, k) is None


# A zero-width joiner or a text/emoji variation selector (VS15, VS16) right after a match blocks it only when it joins
# the character before it in the source: an emoji (a pictograph, a skin-tone modifier, a regional indicator) or a mark
# such as a virama. After a letter it joins nothing a reader sees, so a name stays matched ("Alice\u200d\U0001f469").
_JOINERS = frozenset("\u200d\ufe0e\ufe0f")
_JOIN_HOST = regex.compile(r"[\p{Extended_Pictographic}\p{Emoji_Modifier}\p{Regional_Indicator}\p{M}]")


def _end_edge(view: _View, k: int) -> bool:
    """Whether a match in ``view`` may end at ``k``: ``_cluster_edge``, except that a ``_JOINERS`` character there blocks
    only after a ``_JOIN_HOST`` character, decided on the source text (``FOLDED`` has stripped the virama before a
    half-form joiner)."""
    text = view.text
    if k < len(text) and text[k] in _JOINERS:
        j = view.origin[k]
        if j > 0 and _JOIN_HOST.match(view.source, j - 1):
            return False
        while k < len(text) and text[k] in _JOINERS:  # past the joiners, the next character decides (a keycap's
            k += 1  # U+20E3 is an enclosing mark, so "1" in "1\ufe0f\u20e3" does not match)
    return _cluster_edge(text, k)


def _word_edges_at_breaks(view: _View, start: int, end: int) -> bool:
    """Whether ``view.text[start:end]`` is a whole word once a dropped zero-width space or word joiner counts as a
    word edge: ``_DropIgnorables`` removes them so a name split by one matches, but where one stood between two words
    it still separates them ("Ana\u200bLee" holds "Ana"). The word joiner is a format character, not a break, in UAX
    #29; ``_BREAKS`` treats it as a separator anyway, which keeps the base behaviour and only adds matches."""
    text = view.text

    def edge(inside: int, neighbour: int) -> bool:
        """No word character at ``neighbour``, or a dropped break in the source between it and ``inside``."""
        if not 0 <= neighbour < len(text) or not _WORD_CHAR.match(text, neighbour):
            return True
        a, b = sorted((view.origin[inside], view.origin[neighbour]))
        return any(c in _BREAKS for c in view.source[a + 1 : b])

    after = end
    while after < len(text) and text[after] in _JOINERS:  # as ``_JOINED_WORD``: a word goes on past a joiner
        after += 1
    return edge(start, start - 1) and edge(end - 1, after)


_BREAKS = frozenset("\u200b\u2060")


@dataclass(frozen=True)
class _View:
    """One normalized form of ``source``: ``text`` and ``origin``, the source index of each of its characters."""

    text: str
    origin: Sequence[int]
    source: str
    breaks: bool = False  # a ``_BREAKS`` character was dropped, so ``_word_edges_at_breaks`` can apply


@dataclass(frozen=True)
class _Pipeline:
    """``normalizers`` applied in order to the text and to each value; ``fold_case`` adds a casefolded view, which a
    value matches in unless it is a ``_short_code``."""

    normalizers: tuple[_Step, ...]
    fold_case: bool

    def normalize(self, text: str) -> str:
        for step in self.normalizers:
            text = step.text(text)
        return text

    def key(self, value: str) -> str:
        """``value_key``: the value normalized, less surrounding whitespace."""
        return self.normalize(value).strip()

    def views(self, text: str) -> tuple[_View, _View]:
        """The cased view and the casefolded one (the cased view again when the pipeline does not fold case)."""
        if not self.normalizers and not self.fold_case:
            view = _View(text, range(len(text)), text)  # nothing to align
            return (view, view)
        chars: _Chars = list(zip(text, range(len(text))))
        ascii_text = text.isascii()
        for step in self.normalizers:
            if not (ascii_text and step.ascii_identity):
                chars = step.aligned(chars)
        breaks = (
            not ascii_text and any(c in _BREAKS for c in text) and any(s is _DropIgnorables for s in self.normalizers)
        )
        cased = _View("".join(c for c, _ in chars), [i for _, i in chars], text, breaks)
        if not self.fold_case:
            return (cased, cased)
        folded = _casefold(chars)
        return (cased, _View("".join(c for c, _ in folded), [i for _, i in folded], text, breaks))


_PIPELINES = {
    MatchMode.EXACT: _Pipeline(normalizers=(), fold_case=False),
    MatchMode.FOLDED: _Pipeline(normalizers=(_Nfkd, _DropIgnorables, _StripMarks, _CollapseWhitespace), fold_case=True),
}

# A word character for the whole-word rule: a letter, a mark or a number (the underscore is a boundary, so "Emma"
# matches in "emma_kovacs"). Marks count, so a word goes on through a combining mark (WB4) and "Le" does not end
# inside NFD "Lê".
_WORD = r"(?:(?![\ufe0e\ufe0f])[\p{L}\p{M}\p{N}])"
_WORD_CHAR = regex.compile(_WORD)
_JOINED_WORD = (
    r"[\u200d\ufe0e\ufe0f]*+[\p{L}\p{M}\p{N}]"  # a word that goes on past a joiner or selector ("Alice\ufe0fb")
)


@dataclass(frozen=True)
class _Matcher:
    """One key as the pipeline matches it: in the cased view when ``cased``, else in the casefolded one; with a
    ``pattern`` checking whole-word edges unless the key is in an unspaced script (a substring match)."""

    key: str
    cased: bool
    pattern: Callable[[str, int], object] | None  # a compiled pattern's ``match``

    def spans(self, view: _View) -> Iterator[Span]:
        """Every match in ``view``, overlapping ones included, as a span of the original text. ``str.find`` proposes
        each start; the cluster edges and the pattern check it."""
        text, size = view.text, len(self.key)
        pos = text.find(self.key)
        while pos >= 0:
            end = pos + size
            if (
                (
                    self.pattern is None
                    or self.pattern(text, pos)
                    or (view.breaks and _word_edges_at_breaks(view, pos, end))
                )
                and _cluster_edge(text, pos)
                and _end_edge(view, end)
            ):
                covered = view.origin[pos:end]  # canonical reordering can move marks across sources
                yield Span(min(covered), max(covered) + 1)
            pos = text.find(self.key, pos + 1)


def _compiled(keys: Iterable[str], pipeline: _Pipeline) -> dict[str, _Matcher]:
    """One matcher per distinct key, each already ``pipeline.key`` (normalizing is not idempotent: a dropped mark can
    leave marks out of canonical order, so a key is normalized once); a blank key and a one-character unspaced key get
    none. The short-code rule picks the cased view; the unspaced rule drops the word-edge pattern."""
    matchers: dict[str, _Matcher] = {}
    for value in set(keys) - {""}:
        cased = not pipeline.fold_case or _short_code(value)
        key = value if cased else value.casefold()
        if not (_unspaced(key[0]) and _unspaced(key[-1])):
            pattern = regex.compile(f"(?<!{_WORD})" + regex.escape(key) + f"(?!{_JOINED_WORD})")
            matchers[value] = _Matcher(key, cased, pattern.match)
        elif len(key) >= 2:
            matchers[value] = _Matcher(key, cased, None)
    return matchers


def value_key(value: str, mode: MatchMode) -> str:
    """The key ``count_values`` groups ``value`` under in ``mode``: a ``Leftover.value`` maps back to its key."""
    return _PIPELINES[mode].key(value)


def normalized(text: str) -> str:
    """NFKD with combining marks dropped and whitespace collapsed: the ``FOLDED`` normalizers."""
    return _PIPELINES[MatchMode.FOLDED].normalize(text)


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
    """Whether ``value`` occurs in ``text`` (both ``normalized``; ``folded`` is ``text`` casefolded): the reference
    rule, written as one search rather than as the pipeline.

    Case-insensitive unless ``_short_code``. No match starts on a ``_GLUE_CLASS`` character that does not follow a
    control, or ends before one, except a ``_JOINERS`` character that does not follow a ``_JOIN_HOST`` character. A
    value whose first and last characters are in ``_UNSPACED`` matches as a substring when it has two or more
    characters, and never when it has one; any other value matches where no letter, mark or number directly precedes
    or follows it.
    """
    if not value:
        return False
    if not _short_code(value):
        value, text = value.casefold(), folded
    glued = {
        m.start()
        for m in regex.finditer(_GLUE_CLASS, text)
        if not (m.start() > 0 and _CONTROL.match(text, m.start() - 1))
    }
    loose = {k for k in glued if text[k] in _JOINERS and (k == 0 or _JOIN_HOST.match(text, k - 1) is None)}
    if _unspaced(value[0]) and _unspaced(value[-1]):
        if len(value) < 2:
            return False
        pattern = regex.compile(regex.escape(value))
    else:
        pattern = regex.compile(f"(?<!{_WORD})" + regex.escape(value) + f"(?!{_JOINED_WORD})")
    return any(
        m.start() not in glued and (m.end() not in glued or m.end() in loose)
        for m in pattern.finditer(text, overlapped=True)
    )
