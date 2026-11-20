# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``compile_values`` matches as the per-call rule did, and the Replace check compiles each value once per call."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Iterable
from types import SimpleNamespace
from typing import Any

import pytest

from anonymizer.engine import value_matching
from anonymizer.engine.graph_core.authoring import GraphScope, Mention
from anonymizer.engine.replace.route import _verify_no_original
from anonymizer.engine.value_matching import _normalized, _occurs, any_value_occurs, compile_values


def _reference_any_value_occurs(values: Iterable[str], text: str) -> bool:
    """``any_value_occurs`` as it was before ``compile_values``: every value normalized and matched per call."""
    plain = _normalized(text)
    folded = plain.casefold()
    return any(_occurs(_normalized(value), plain, folded) for value in values)


# Case, short codes, CJK, Thai, accents (precomposed and combining), casefold expansion, fullwidth, digits,
# punctuation and word boundaries.
_ALPHABET = ["a", "b", "c", "A", "B", "C", " ", "  ", "中", "文", "ก", "é", "e\u0301", "É", "ß", "Ａ", "1", "'",
             "-", "_", ".", ",", "\n"]  # fmt: skip


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


def test_compile_values_matches_the_reference_on_a_seeded_corpus() -> None:
    corpus = _corpus(20_000)
    mismatches = [(v, t) for v, t in corpus if compile_values(v)(t) != _reference_any_value_occurs(v, t)]
    assert mismatches == []
    assert sum(any_value_occurs(v, t) for v, t in corpus) > 2_000  # the corpus exercises matches, not only misses


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
def test_compile_values_examples(values: list[str], text: str, expected: bool) -> None:
    matcher = compile_values(values)
    assert matcher(text) is expected
    assert matcher(text) is _reference_any_value_occurs(values, text)


@pytest.mark.parametrize("n", [5, 60])
def test_no_original_check_compiles_each_value_once(monkeypatch: pytest.MonkeyPatch, n: int) -> None:
    """``n`` distinct names, each replaced, leave ``n + 1`` kept parts; each name's pattern is built once."""
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
    replacements = [{"original": name, "label": "first_name", "synthetic": f"P{j}"} for j, name in enumerate(names)]
    rmap = {"replacements": replacements}
    text = source
    for j, (a, b) in reversed(list(enumerate(spans))):
        text = text[:a] + f"P{j}" + text[b:]

    assessment = _verify_no_original(text, source, mentions, rmap, target)

    assert (assessment.satisfied, assessment.reason) == (True, "")
    assert calls == {"escape": n, "compile": n}
