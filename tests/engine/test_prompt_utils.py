# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from anonymizer.engine.prompt_utils import substitute_placeholders

# ---------------------------------------------------------------------------
# Basic substitution
# ---------------------------------------------------------------------------


def test_basic_substitution() -> None:
    assert substitute_placeholders("Hello <<NAME>>!", {"<<NAME>>": "Alice"}) == "Hello Alice!"


def test_multiple_placeholders() -> None:
    result = substitute_placeholders("<<A>> and <<B>>", {"<<A>>": "X", "<<B>>": "Y"})
    assert result == "X and Y"


def test_no_cross_contamination() -> None:
    result = substitute_placeholders(
        "text: <<TEXT>>, score: <<SCORE>>",
        {"<<TEXT>>": "contains <<SCORE>> literally", "<<SCORE>>": "0.9"},
    )
    assert result == "text: contains <<SCORE>> literally, score: 0.9"


def test_repeated_placeholder() -> None:
    result = substitute_placeholders("<<X>> and <<X>>", {"<<X>>": "Y"})
    assert result == "Y and Y"


def test_empty_replacement_value() -> None:
    result = substitute_placeholders("a<<X>>b", {"<<X>>": ""})
    assert result == "ab"


def test_empty_replacements_no_placeholders() -> None:
    assert substitute_placeholders("no placeholders", {}) == "no placeholders"


# ---------------------------------------------------------------------------
# Strict mode (default)
# ---------------------------------------------------------------------------


def test_strict_raises_on_malformed_keys() -> None:
    with pytest.raises(ValueError, match="must use <<\\.\\.\\.>> format"):
        substitute_placeholders("some text", {"TEXT": "value"})


def test_strict_raises_on_unresolved_placeholders() -> None:
    with pytest.raises(ValueError, match="Unresolved placeholders"):
        substitute_placeholders("<<A>> and <<B>>", {"<<A>>": "X"})


def test_strict_raises_on_unresolved_with_empty_replacements() -> None:
    with pytest.raises(ValueError, match="Unresolved placeholders"):
        substitute_placeholders("<<A>> is here", {})


def test_strict_warns_on_unused_keys(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="anonymizer.prompt_utils"):
        result = substitute_placeholders("just <<A>>", {"<<A>>": "X", "<<B>>": "unused"})
    assert result == "just X"
    assert "<<B>>" in caplog.text


# ---------------------------------------------------------------------------
# Non-strict mode
# ---------------------------------------------------------------------------


def test_non_strict_leaves_unresolved() -> None:
    result = substitute_placeholders("<<A>> and <<B>>", {"<<A>>": "X"}, strict=False)
    assert result == "X and <<B>>"


def test_non_strict_allows_malformed_keys() -> None:
    result = substitute_placeholders("hello world", {"world": "earth"}, strict=False)
    assert result == "hello earth"


def test_non_strict_empty_replacements() -> None:
    assert substitute_placeholders("<<A>>", {}, strict=False) == "<<A>>"
