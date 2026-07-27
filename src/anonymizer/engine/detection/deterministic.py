# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable

from anonymizer.engine.detection.postprocess import EntitySpan, resolve_overlaps

DETERMINISTIC_ENTITY_LABELS: frozenset[str] = frozenset(
    {
        "credit_debit_card",
        "email",
        "ipv4",
        "ipv6",
        "mac_address",
        "url",
    }
)

_EMAIL_RE = re.compile(
    r"(?<![A-Za-z0-9._%+\-])"
    r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@"
    r"(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+"
    r"[A-Za-z]{2,63}"
    r"(?![A-Za-z0-9._%+\-])"
)
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>'\"]+")
_IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_IPV6_TOKEN_RE = re.compile(r"(?<![0-9A-Fa-f:.])(?:[0-9A-Fa-f]{0,4}:){2,}[0-9A-Fa-f:.%]+(?![0-9A-Fa-f:.])")
_MAC_RE = re.compile(r"(?i)(?<![0-9A-F])(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}(?![0-9A-F])")
_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")

_TRAILING_URL_PUNCTUATION = ".,;:!?)]}"


def detect_deterministic_entities(text: str, *, labels: Iterable[str] | None = None) -> list[EntitySpan]:
    """Detect high-confidence structured entities in raw text.

    This first deterministic pole intentionally stays narrow: it handles
    strongly structured labels with parser or checksum validation and leaves
    context-sensitive identifiers to GLiNER plus LLM validation.
    """

    requested = set(labels) if labels is not None else set(DETERMINISTIC_ENTITY_LABELS)
    enabled = requested & DETERMINISTIC_ENTITY_LABELS
    if not enabled:
        return []

    spans: list[EntitySpan] = []
    if "email" in enabled:
        spans.extend(_email_spans(text))
    if "url" in enabled:
        spans.extend(_url_spans(text))
    if "ipv4" in enabled:
        spans.extend(_ipv4_spans(text))
    if "ipv6" in enabled:
        spans.extend(_ipv6_spans(text))
    if "mac_address" in enabled:
        spans.extend(_mac_spans(text))
    if "credit_debit_card" in enabled:
        spans.extend(_card_spans(text))
    return resolve_overlaps(spans)


def _email_spans(text: str) -> list[EntitySpan]:
    return [_span(text=text, label="email", start=match.start(), end=match.end()) for match in _EMAIL_RE.finditer(text)]


def _url_spans(text: str) -> list[EntitySpan]:
    spans = []
    for match in _URL_RE.finditer(text):
        start = match.start()
        end = _trim_trailing(text, start, match.end(), _TRAILING_URL_PUNCTUATION)
        if end > start:
            spans.append(_span(text=text, label="url", start=start, end=end))
    return spans


def _ipv4_spans(text: str) -> list[EntitySpan]:
    spans = []
    for match in _IPV4_RE.finditer(text):
        value = match.group(0)
        try:
            ipaddress.IPv4Address(value)
        except ValueError:
            continue
        spans.append(_span(text=text, label="ipv4", start=match.start(), end=match.end()))
    return spans


def _ipv6_spans(text: str) -> list[EntitySpan]:
    spans = []
    for match in _IPV6_TOKEN_RE.finditer(text):
        start = match.start()
        end = _trim_trailing(text, start, match.end(), ".;!,)]}")
        value = text[start:end]
        if value.count(":") < 2:
            continue
        try:
            ipaddress.IPv6Address(value.split("%", 1)[0])
        except ValueError:
            continue
        spans.append(_span(text=text, label="ipv6", start=start, end=end))
    return spans


def _mac_spans(text: str) -> list[EntitySpan]:
    return [
        _span(text=text, label="mac_address", start=match.start(), end=match.end()) for match in _MAC_RE.finditer(text)
    ]


def _card_spans(text: str) -> list[EntitySpan]:
    spans = []
    for match in _CARD_RE.finditer(text):
        digits = _digits_only(match.group(0))
        if not 13 <= len(digits) <= 19:
            continue
        if len(set(digits)) == 1:
            continue
        if not _passes_luhn(digits):
            continue
        end = _trim_trailing(text, match.start(), match.end(), " -")
        spans.append(_span(text=text, label="credit_debit_card", start=match.start(), end=end))
    return spans


def _span(*, text: str, label: str, start: int, end: int) -> EntitySpan:
    return EntitySpan(
        entity_id=f"{label}_{start}_{end}",
        value=text[start:end],
        label=label,
        start_position=start,
        end_position=end,
        score=1.0,
        source="deterministic",
    )


def _trim_trailing(text: str, start: int, end: int, chars: str) -> int:
    while end > start and text[end - 1] in chars:
        end -= 1
    return end


def _digits_only(value: str) -> str:
    return "".join(char for char in value if char.isdigit())


def _passes_luhn(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0
