# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The rule for whether a replaced value still occurs in a text, shared by Rewrite and Replace."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable


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


def _normalized(text: str) -> str:
    """NFKD with combining marks dropped and whitespace collapsed."""
    decomposed = unicodedata.normalize("NFKD", text)
    return " ".join("".join(c for c in decomposed if unicodedata.category(c) != "Mn").split())


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
