# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validated text and Markdown rendering primitives for W&B reports."""

from __future__ import annotations

import json
import re
import unicodedata
from html import escape as escape_html
from typing import Any
from urllib.parse import urlsplit


def scalar_text(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return str(value)
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def plain_text(value: str) -> str:
    if any(
        unicodedata.category(character) == "Cf"
        or (unicodedata.category(character) == "Cc" and character not in "\t\n\r")
        for character in value
    ):
        raise ValueError("W&B display text contains unsafe control characters")
    return " ".join(value.split())


def validate_output_url(value: str | None) -> str | None:
    if value is None:
        return None
    if any(unicodedata.category(character) in {"Cc", "Cf"} for character in value):
        raise ValueError("W&B output URL contains unsafe control characters")
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("W&B output URL must be a credential-free HTTP(S) URL")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("W&B output URL must use HTTPS unless it targets loopback")
    return value


def validate_output_text(value: str | None) -> str | None:
    if value is not None:
        plain_text(value)
    return value


def escape_link_label(value: str) -> str:
    text = escape_html(plain_text(value), quote=False)
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def escape_markdown_text(value: str) -> str:
    text = escape_html(plain_text(value), quote=False)
    for character in ("\\", "`", "*", "_", "{", "}", "[", "]", "(", ")", "#", "+", "-", ".", "!", "|"):
        text = text.replace(character, f"\\{character}")
    return text


def escape_heading(value: str) -> str:
    return escape_markdown_text(value)


def code_span(value: Any) -> str:
    text = escape_html(scalar_text(value), quote=False).replace("|", "&#124;")
    text = plain_text(text)
    fence = "`" * (max((len(match) for match in re.findall(r"`+", text)), default=0) + 1)
    if "`" in text:
        return f"{fence} {text} {fence}"
    return f"`{text}`"


def table_code(value: Any) -> str:
    return code_span(value)


def escape_list_text(value: Any) -> str:
    return escape_html(plain_text(scalar_text(value)), quote=False).replace("|", "&#124;")


__all__ = [
    "code_span",
    "escape_heading",
    "escape_link_label",
    "escape_list_text",
    "escape_markdown_text",
    "plain_text",
    "scalar_text",
    "table_code",
    "validate_output_text",
    "validate_output_url",
]
