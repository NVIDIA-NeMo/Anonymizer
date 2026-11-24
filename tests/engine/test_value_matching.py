# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``count_values`` matches as the per-call rule does, and the Replace check compiles each value once per call."""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from anonymizer.engine import value_matching
from anonymizer.engine.graph_core.authoring import GraphScope, Mention
from anonymizer.engine.replace.route import _verify_no_original
from anonymizer.engine.schemas import EntitySchema
from anonymizer.engine.spans import Span, Spans
from anonymizer.engine.value_matching import (
    Leftover,
    MatchMode,
    _mapped,
    _occurs,
    count_values,
    entity_spans,
    format_cause,
)
from anonymizer.engine.value_matching import (
    normalized as _normalized,
)


def _reference_any_value_occurs(values: Iterable[str], text: str) -> bool:
    """The matching rule applied per call: every value normalized and matched on its own."""
    plain = _normalized(text)
    folded = plain.casefold()
    return any(_occurs(_normalized(value), plain, folded) for value in values)


def _counts(
    source: str, spans: list[tuple[int, int]], output: str, values: list[str] | None = None
) -> tuple[Leftover, ...]:
    """``count_values`` over ``spans`` in position order, each labelled ``name``; an empty or out-of-range span is
    unusable but keeps its position."""
    ordered = sorted(spans, key=lambda span: span[0])
    usable = [Span(a, b) if 0 <= a < b <= len(source) else None for a, b in ordered]
    return count_values(source, [(span, "name") for span in usable], output, values, mode=MatchMode.FOLDED)


def _nums(leftovers: tuple[Leftover, ...]) -> list[tuple[int | None, int, int, int]]:
    return [(x.position, x.in_source, x.detected, x.in_output) for x in leftovers]


# Case, short codes, CJK, Thai, accents (precomposed and combining), casefold expansion, fullwidth, digits,
# punctuation and word boundaries; and marks that NFKD keeps (not Mn) but reorders by combining class: musical
# symbols, a precomposed note that decomposes into them, a Hangul tone mark, viramas and a Vietnamese reading mark;
# and Mn marks of class 0 (dropped, so they can leave the marks around them out of canonical order).
_ALPHABET = ["a", "b", "c", "A", "B", "C", " ", "  ", "中", "文", "ก", "é", "e\u0301", "É", "ß", "Ａ", "1", "'",
             "-", "_", ".", ",", "\n", "\U0001d165", "\U0001d16d", "\U0001d16e", "\U0001d160", "\u302e",
             "\u1715", "\ua953", "\U00016ff0", "\u093a", "\ufe0f"]  # fmt: skip


def _string(rng: random.Random, low: int, high: int) -> str:
    return "".join(rng.choice(_ALPHABET) for _ in range(rng.randint(low, high)))


def _corpus(size: int, seed: int = 20261117) -> list[tuple[list[str], str]]:
    rng = random.Random(seed)
    pairs = []
    for _ in range(size):
        text = _string(rng, 0, 14)
        values = [_string(rng, 0, 4) for _ in range(rng.randint(0, 3))]
        if text and rng.random() < 0.5:  # a slice of the text, so that matches are common
            start = rng.randrange(len(text))
            values.append(text[start : start + rng.randint(1, 4)])
        pairs.append((values, text))
    return pairs


@pytest.mark.parametrize(
    ("values", "text", "expected"),
    [
        (["ID"], "my id is", False),  # a short code keeps its case
        (["ID"], "my ID is", True),
        (["alice"], "ALICE wrote", True),
        (["Ali"], "Alice", False),  # no match inside a word
        (["José"], "Jose\u0301 left", True),  # combining accent
        (["中文"], "这是中文书", True),  # unspaced script: substring
        (["中"], "中文", False),  # one unspaced character never matches
        (["a_b"], "xa_b", False),  # underscore is not a boundary
        (["", "  "], "anything", False),
    ],
)
def test_count_values_examples(values: list[str], text: str, expected: bool) -> None:
    """The text is counted the same way as the source and as the output."""
    assert any(c.in_source for c in _counts(text, [], "", values)) is expected
    assert any(c.in_output for c in _counts("", [], text, values)) is expected
    assert _reference_any_value_occurs(values, text) is expected


def test_mapped_normalization_equals_normalized_on_the_corpus() -> None:
    for _, text in _corpus(20_000):
        (plain, origin), (folded, folded_origin) = _mapped(text)
        assert plain == _normalized(text)
        assert folded == plain.casefold()
        assert len(origin) == len(plain) and len(folded_origin) == len(folded)


def test_count_values_finds_a_value_exactly_when_the_presence_rule_does() -> None:
    corpus = _corpus(20_000)
    mismatches = [
        (v, t)
        for v, t in corpus
        if any(c.in_output for c in _counts("", [], t, v)) != _reference_any_value_occurs(v, t)
    ]
    assert mismatches == []
    assert sum(_reference_any_value_occurs(v, t) for v, t in corpus) > 2_000  # matches, not only misses


def test_every_reported_span_is_a_span_of_the_text_holding_the_value() -> None:
    """Spans stay valid where canonical reordering moves a mark ahead of the character it followed."""
    checked = 0
    for values, text in _corpus(20_000):
        for leftover in _counts("", [], text, values):
            key = _normalized(leftover.value).casefold()
            for span in leftover.output_spans:
                assert 0 <= span.start < span.end <= len(text)
                assert key in _normalized(text[span.slice]).casefold()
                checked += 1
    assert checked > 2_000


_MIRA = "Mira\U0001d157\U0001d16d\U0001d16e"  # the last two marks are out of canonical order


@pytest.mark.parametrize(
    ("value", "output"),
    [
        (_MIRA, f"{_MIRA} called."),  # copied verbatim
        (_MIRA, "Mira\U0001d157\U0001d16e\U0001d16d called."),  # copied in canonical order
        ("김\u302e\u1715", "김\u302e\u1715 called."),  # Hangul tone mark then virama
        ("Ana\ua953\U00016ff0", "Ana\ua953\U00016ff0 called."),  # virama then Vietnamese reading mark
        ("Mira\U0001d16d\u093a\u1b44", "Mira\U0001d16d\u093a\u1b44 called."),  # normalizing twice reorders
        ("Ana\u302e\ufe0f\u1715", "Ana\u302e\ufe0f\u1715 called."),
    ],
)
def test_a_value_with_reordered_marks_is_found_where_it_occurs(value: str, output: str) -> None:
    source = f"{value} called."
    leftovers = count_values(source, [(Span(0, len(value)), "first_name")], output, mode=MatchMode.FOLDED)
    assert format_cause(leftovers) == "e0:N1,D1,O1"
    assert leftovers[0].output_spans == Spans([Span(0, len(output) - len(" called."))])


def test_a_value_inside_another_detected_value() -> None:
    source = "May Smith met May."
    spans = [(14, 17), (0, 9)]  # unsorted on purpose: positions follow the start
    kept = _counts(source, spans, "[R] met May.")
    assert _nums(kept) == [(0, 1, 1, 0), (1, 2, 2, 1)]  # "May Smith" at 0, "May" at its own span 1
    assert [c.over_bound for c in kept] == [False, True]
    assert not any(c.over_bound for c in _counts(source, spans, "[R] met [R]."))


def test_a_value_takes_its_own_span_position() -> None:
    """A match of John inside "John Smith" comes first, but John's position is its own span."""
    source = "John Smith called. John said hi."
    counts = _counts(source, [(0, 10), (19, 23)], "[A] called. John said hi.")
    assert _nums(counts) == [(0, 1, 1, 0), (1, 2, 2, 1)]
    assert format_cause(c for c in counts if c.over_bound) == "e1:N2,D2,O1"


def test_a_value_whose_span_cuts_a_word_keeps_its_position() -> None:
    """The value "field" inside "Mayfield" never matches as a word, yet it keeps its span's position."""
    field, may = _counts("Mayfield and May.", [(3, 8), (13, 16)], "MayXYZ and Ann.")
    assert (field.position, field.in_source) == (0, 0)
    assert (may.position, may.in_source, may.detected) == (1, 1, 1)


def test_straddling_and_overlapping_matches() -> None:
    (straddle,) = _counts("ha ha ha", [(3, 8)], "ha [R]", ["ha ha"])
    assert (straddle.in_source, straddle.detected, straddle.in_output) == (2, 1, 0)  # "ha ha" at 0 straddles the span
    (count,) = _counts("Al Al", [(0, 2), (3, 5)], "Al", ["Al"])
    assert (count.position, count.in_source, count.detected, count.in_output) == (0, 2, 2, 1)


def test_a_match_inside_an_outer_span_with_a_nested_one_is_detected() -> None:
    """Overlapping spans: a match inside the outer span counts as detected, though a later span starts before it."""
    source, spans = "John Q. Smith met Ann.", [(0, 13), (5, 7)]
    smith, john = (_counts(source, spans, "Smith met [A].", [value])[0] for value in ("Smith", "John Q. Smith"))
    assert (smith.position, smith.in_source, smith.detected, smith.in_output, smith.over_bound) == (0, 1, 1, 1, True)
    assert (john.position, john.in_source, john.detected) == (0, 1, 1)


def test_a_match_held_by_overlapping_spans_takes_the_one_reaching_furthest() -> None:
    (count,) = _counts("aaa bbb ccc ddd eee fff ggg hhh", [(0, 12), (3, 30)], "", ["bbb"])
    assert (count.position, count.in_source, count.detected) == (1, 1, 1)


def test_unusable_spans_still_count_toward_positions() -> None:
    (count,) = _counts("Call Alice.", [(5, 10), (0, 0)], "Call Alice.")  # (0, 0) is empty: index 0, unused
    assert _nums((count,)) == [(1, 1, 1, 1)]


def test_the_cause_holds_positions_and_counts_never_value_text() -> None:
    canary = "Zyxwvutsrq"
    source = f"{canary} met {canary}. Bob left."
    spans = [(0, len(canary)), (source.index("Bob"), source.index("Bob") + 3)]
    counts = _counts(source, spans, f"{canary}, {canary} and Bob.")
    cause = format_cause(c for c in counts if c.over_bound)
    assert cause == "e0:N2,D1,O2;e1:N1,D1,O1"
    assert canary.casefold() not in cause.casefold()


def test_equal_counts_give_one_entry_each() -> None:
    source = "Alice met Bob."
    counts = _counts(source, [(0, 5), (10, 13)], source)
    assert format_cause(c for c in counts if c.over_bound) == "e0:N1,D1,O1;e1:N1,D1,O1"


def test_a_value_no_span_holds_has_no_position() -> None:
    assert format_cause(_counts("Alice met Bob.", [], "Bob", ["Bob"])) == "e-:N1,D0,O1"


def test_count_values_reports_where_the_value_occurs() -> None:
    source = "Call Alice. ALICE will answer."
    (count,) = _counts(source, [(5, 10)], "Alice and alice.")
    assert (count.value, count.label, count.span) == ("Alice", "name", Span(5, 10))
    assert count.undetected_spans == Spans([Span(12, 17)])
    assert count.output_spans == Spans([Span(0, 5), Span(10, 15)])
    assert (count.in_source, count.detected, count.in_output, count.over_bound, count.undetected_kept) == (
        2,
        1,
        2,
        True,
        True,
    )


@pytest.mark.parametrize(
    ("output", "over_bound", "undetected_kept"),
    [("Sara and Sara.", False, False), ("Alice and Sara.", False, True), ("Alice and Alice.", True, True)],
)
def test_the_bound_and_the_kept_undetected_value(output: str, over_bound: bool, undetected_kept: bool) -> None:
    """Alice is detected once and missed once: the output may hold her once, and holding her at all keeps the miss."""
    (count,) = _counts("Call Alice. Alice will answer.", [(5, 10)], output)
    assert (count.over_bound, count.undetected_kept) == (over_bound, undetected_kept)


@pytest.mark.parametrize("n", [5, 60])
def test_no_original_check_compiles_each_value_once(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """``n`` distinct names, each replaced: the check builds each name's pattern once."""
    calls = {"escape": 0, "compile": 0}

    def counted(name: str) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            calls[name] += 1
            return getattr(re, name)(*args, **kwargs)

        return call

    shim = SimpleNamespace(**{k: getattr(re, k) for k in dir(re) if not k.startswith("__")})
    shim.escape, shim.compile = counted("escape"), counted("compile")
    monkeypatch.setattr(value_matching, "re", shim)

    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    names = [(words[i % 8] + words[i // 8 % 8] + words[i // 64 % 8]).capitalize() for i in range(n)]
    source, spans = "", []
    for name in names:
        start = len(source) + len("Customer ")
        source += f"Customer {name} called today. "
        spans.append((start, start + len(name)))
    scope = GraphScope()
    target = scope.datum(source)
    mentions = tuple(Mention(target, a, b, "first_name") for a, b in spans)
    text = source
    for j, (a, b) in reversed(list(enumerate(spans))):
        text = text[:a] + f"P{j}" + text[b:]

    assessment = _verify_no_original(text, source, mentions, target)

    assert (assessment.satisfied, assessment.reason) == (True, "")
    assert calls == {"escape": n, "compile": n}


@pytest.mark.parametrize(
    ("source", "span", "values"),
    [("John Smith called.", (0, 10), ["John", "Smith"]), ("田中太郎さん", (0, 4), ["田中", "太郎"])],
)
def test_values_named_by_a_containing_span_keep_their_own_value(
    source: str, span: tuple[int, int], values: list[str]
) -> None:
    """Values without a span of their own are named by the span holding them, yet each record keeps its value."""
    leftovers = _counts(source, [span], "", values)
    assert [(x.position, x.value, x.span) for x in leftovers] == [(0, value, Span(*span)) for value in sorted(values)]
    assert [source[x.span.slice] for x in leftovers if x.span] == [source[span[0] : span[1]]] * len(values)


def test_a_value_keeps_the_form_it_was_given() -> None:
    (count,) = _counts("José met Jose.", [(0, 4)], "", ["José", "Jose"])
    assert (count.value, count.in_source) == ("José", 2)


def test_entity_spans_keep_every_position_and_only_usable_spans() -> None:
    source = "Ann met Bob."
    raw = [("Bob", 8, 11), ("Ann", 0, 3), ("Bo", 8, 11), ("x", 4, 99), ("", 5, 5), ("Ann", -3, 0)]
    entities = [
        EntitySchema(value=v, label=f"l{i}", start_position=a, end_position=b) for i, (v, a, b) in enumerate(raw)
    ]
    assert entity_spans(source, entities) == [
        (None, "l5"),  # negative start
        (Span(0, 3), "l1"),
        (None, "l3"),  # out of range
        (None, "l4"),  # empty
        (Span(8, 11), "l0"),
        (None, "l2"),  # the slice is not the value
    ]


_FUZZ_POOLS = (
    list("abcAB01_'-., "),
    list("aIİıßſKσςΣǅﬃŉΐ"),
    [chr(c) for c in range(0x0300, 0x0370)],
    [
        "\U0001d165",
        "\U0001d166",
        "\U0001d16d",
        "\U0001d16e",
        "\U0001d172",
        "\u302e",
        "\u302f",
        "\u1715",
        "\u1734",
        "\u1b44",
        "\ua953",
        "\ua9c0",
        "\U000111c0",
        "\U00016ff0",
        "\U00016ff1",
    ],
    ["\u093a", "\ufe0f", "\u200d", "\u0903"],
    ["\U0001d15e", "\U0001d160", "\U0001d1bb", "é", "ǖ", "ṩ"],
    [" ", "\n", "\u00a0", "\u3000", "\u200b"],
    list("김한가中文かカ"),
    list("Ａｂ①㎏ﷺ¼™ﬁ²"),
)


def _fuzz_string(rng: random.Random, low: int, high: int) -> str:
    return "".join(rng.choice(rng.choice(_FUZZ_POOLS)) for _ in range(rng.randint(low, high)))


def test_matching_holds_on_a_wide_unicode_corpus() -> None:
    """Over marks of every kind, casefold expansions, compatibility forms and spaces: the map is the normalized text,
    matching agrees with the presence rule, every span is valid and holds the value, and nothing raises."""
    rng = random.Random(426)
    found = 0
    for _ in range(2_000):
        value = _fuzz_string(rng, 1, 5)
        text = _fuzz_string(rng, 0, 12)
        if rng.random() < 0.6:
            cut = rng.randint(0, len(text))
            text = text[:cut] + value + text[cut:]
        for t in (text, value):
            (plain, origin), (folded, folded_origin) = _mapped(t)
            assert plain == _normalized(t) and folded == plain.casefold()
            assert all(0 <= i < len(t) for i in origin + folded_origin)
        key = _normalized(value)
        leftovers = count_values("", [], text, [value], mode=MatchMode.FOLDED)
        plain = _normalized(text)
        present = _occurs(key, plain, plain.casefold())
        assert any(x.in_output for x in leftovers) is present
        found += present
        for x in leftovers:
            for span in x.output_spans:
                assert 0 <= span.start < span.end <= len(text)
                held = _normalized(text[span.slice])
                assert (key if value_matching._short_code(key) else key.casefold()) in (
                    held if value_matching._short_code(key) else held.casefold()
                )
    assert found > 300  # matches, not only misses


@pytest.mark.parametrize(
    ("source", "value", "folded", "exact"),
    [
        ("May said she may come.", "May", 2, 1),
        ("Ông Hàn nói hết hạn.", "Hàn", 2, 1),
        ("Alice met Alice.", "Alice", 2, 2),
        ("Mark marked Mark.", "Mark", 2, 2),
    ],
)
def test_exact_mode_counts_only_the_spelling_given(source: str, value: str, folded: int, exact: int) -> None:
    """``EXACT`` matches the value as spelled, whole words; ``FOLDED`` folds case and accents."""
    for mode, expected in ((MatchMode.FOLDED, folded), (MatchMode.EXACT, exact)):
        [leftover] = count_values(source, [], "", [value], mode=mode)
        assert leftover.in_source == expected, mode


@pytest.mark.parametrize(
    ("source", "value", "exact"),
    [
        ("สมชายบอกให้สมชายมา", "สมชาย", 2),
        ("สมชายไปที่สมชาย", "สมชาย", 2),
        ("सीता और राम सीता से मिले", "सीता", 2),
        ("राम ने राम को देखा", "राम", 2),
        (unicodedata.normalize("NFD", "Ông Le nói về Lê."), "Le", 1),
        (unicodedata.normalize("NFD", "éLe met Le."), "Le", 1),
        ("สมกินข้าว สมก", "สมก", 1),
    ],
    ids=[
        "thai-1",
        "thai-2",
        "devanagari-sita",
        "devanagari-ram",
        "nfd-mark-after",
        "nfd-mark-before",
        "unspaced-ends-in-cluster",
    ],
)
def test_exact_mode_never_splits_a_cluster(source: str, value: str, exact: int) -> None:
    """``EXACT`` rejects a match that ends before a combining mark or starts on one, and a whole-word match right after
    a mark; an unspaced match after a mark that belongs to the previous cluster still counts."""
    [leftover] = count_values(source, [], "", [value], mode=MatchMode.EXACT)
    assert leftover.in_source == exact


@pytest.mark.parametrize("mode", list(MatchMode))
def test_count_values_maps_each_text_once_for_every_value(monkeypatch: pytest.MonkeyPatch, mode: MatchMode) -> None:
    """Every value reads the same offset map of each text it searches, so a long row with many values is mapped once
    per text, not once per value."""
    origins: dict[str, set[int]] = {}
    spans = value_matching._Matcher.spans

    def recorded(self: Any, text: str, origin: Any) -> Any:
        origins.setdefault(text, set()).add(id(origin))
        return spans(self, text, origin)

    monkeypatch.setattr(value_matching._Matcher, "spans", recorded)
    names = ["Alice", "Bruno", "Chen", "Dara", "Emeka"]
    source = " ".join(f"{name} met {name}." for name in names)
    output = source.replace("Alice", "[NAME]")
    entities = [(Span(source.index(name), source.index(name) + len(name)), "first_name") for name in names]
    leftovers = count_values(source, entities, output, mode=mode)

    assert [x.in_source for x in leftovers] == [2] * len(names)
    assert len(origins) == 2  # one view of the source and one of the output (folded mode searches casefolded text)
    assert all(len(ids) == 1 for ids in origins.values())
