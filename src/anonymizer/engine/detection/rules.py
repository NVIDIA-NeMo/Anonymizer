# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from anonymizer.engine.detection.postprocess import EntitySpan, resolve_overlaps

_RULE_SCORE = 1.0
_RULE_SOURCE = "rule"
_RELIGIOUS_BELIEF_TERMS = (
    "agnostic",
    "atheist",
    "baptist",
    "buddhist",
    "catholic",
    "christian",
    "hindu",
    "jewish",
    "mormon",
    "muslim",
    "protestant",
    "secular",
)
_RELIGIOUS_BELIEF_RE = "|".join(re.escape(term) for term in _RELIGIOUS_BELIEF_TERMS)
_COOKIE_PAIR_RE = r"[A-Za-z][A-Za-z0-9_-]*=[^;'\s\"\r\n]+"
_COOKIE_VALUE_RE = rf"({_COOKIE_PAIR_RE}(?:;\s*{_COOKIE_PAIR_RE})*)"
_STRUCTURED_ID_VALUE_RE = (
    r"(?:[A-Za-z][A-Za-z0-9]{1,20}[-_][A-Za-z0-9][A-Za-z0-9_-]{5,}|"
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})"
)


@dataclass(frozen=True)
class _RulePattern:
    label: str
    pattern: re.Pattern[str]
    group: int = 0


_RULES: tuple[_RulePattern, ...] = (
    _RulePattern(
        label="api_key",
        pattern=re.compile(r"sk-(?:test|ant-api03|proj|prod)-[A-Za-z0-9_-]{16,}"),
    ),
    _RulePattern(label="api_key", pattern=re.compile(r"ghp_[A-Za-z0-9_]{20,}")),
    _RulePattern(label="api_key", pattern=re.compile(r"hf_[A-Za-z0-9]{20,}")),
    _RulePattern(label="api_key", pattern=re.compile(r"pat-[A-Za-z0-9_-]{20,}")),
    _RulePattern(label="api_key", pattern=re.compile(r"xoxb-[A-Za-z0-9-]{20,}")),
    _RulePattern(label="api_key", pattern=re.compile(r"AIza[A-Za-z0-9_-]{20,}")),
    _RulePattern(label="api_key", pattern=re.compile(r"ya29\.[A-Za-z0-9_-]{20,}")),
    _RulePattern(label="api_key", pattern=re.compile(r"AKIA[A-Z0-9]{16,}")),
    _RulePattern(
        label="api_key",
        pattern=re.compile(
            r"\b(?:api[_-]?key|token|auth[_-]?token|session[_-]?id|aws_access_key_id|access_key_id)="
            r"([^\s;'\"\\]{8,})",
            flags=re.IGNORECASE,
        ),
        group=1,
    ),
    _RulePattern(
        label="api_key",
        pattern=re.compile(r"Authorization:\s*Bearer\s+([A-Za-z0-9._-]{16,})", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="http_cookie",
        pattern=re.compile(rf"\bCookie:\s*{_COOKIE_VALUE_RE}", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="http_cookie",
        pattern=re.compile(rf"\bcookie\s*=\s*{_COOKIE_VALUE_RE}", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="pin",
        pattern=re.compile(r"(?<![A-Za-z0-9_-])['\"]?pin['\"]?\s*[:=]\s*['\"]?(\d{4,8})['\"]?\b", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="unique_id",
        pattern=re.compile(
            rf"\b(?:trace[-_]?id|request[-_]?id|req[-_]?id|order[-_]?id|tenant[-_]?id|unique[-_]?id)"
            rf"(?:\s*[:=]\s*|\s+)['\"]?({_STRUCTURED_ID_VALUE_RE})['\"]?",
            flags=re.IGNORECASE,
        ),
        group=1,
    ),
    _RulePattern(
        label="unique_id",
        pattern=re.compile(rf"--tenant\s+['\"]?({_STRUCTURED_ID_VALUE_RE})['\"]?", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="user_name",
        pattern=re.compile(
            r"(?<![A-Za-z0-9_-])['\"]?(?:user(?:_?name)?|username|login|account)['\"]?"
            r"\s*[:=]\s*['\"]?([A-Za-z][A-Za-z0-9._-]{2,31})['\"]?\b",
            flags=re.IGNORECASE,
        ),
        group=1,
    ),
    _RulePattern(
        label="user_name",
        pattern=re.compile(r"\buser\s+((?=[A-Za-z0-9._-]*\d)[A-Za-z][A-Za-z0-9._-]{2,31})\b"),
        group=1,
    ),
    _RulePattern(
        label="user_name",
        pattern=re.compile(
            r"\baz\s+login\s+--service-principal\b[^\r\n]*?\s-u\s+([A-Za-z][A-Za-z0-9._-]{2,31})\b",
            flags=re.IGNORECASE,
        ),
        group=1,
    ),
    _RulePattern(
        label="password",
        pattern=re.compile(r"\bPassword:\s*([^\s'\"\\]+)", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="password",
        pattern=re.compile(r"\s-p\s+([^\s\\]+)", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="password",
        pattern=re.compile(r"\bAWS_SECRET_ACCESS_KEY=([^\s;'\"\\]+)", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="password",
        pattern=re.compile(r"\bDJANGO_SECRET=([^\s;'\"\\]+)", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="password",
        pattern=re.compile(r"\becho\s+(['\"])([^'\"\r\n]{4,})\1\s*\|\s*sudo\s+-S\b", flags=re.IGNORECASE),
        group=2,
    ),
    _RulePattern(
        label="password",
        pattern=re.compile(r"postgres://[^:\s'\"\\]+:([^@\s'\"\\]+)@", flags=re.IGNORECASE),
        group=1,
    ),
    _RulePattern(
        label="url",
        pattern=re.compile(
            r"\b(?:postgres(?:ql)?|mysql|mariadb|mongodb(?:\+srv)?|redis|rediss)://[^\s;'\"<>]+",
            flags=re.IGNORECASE,
        ),
    ),
    _RulePattern(
        label="email",
        pattern=re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\w-])"),
    ),
    _RulePattern(label="url", pattern=re.compile(r"https?://[^\s;'\"<>]+")),
    _RulePattern(
        label="date_of_birth",
        pattern=re.compile(
            r"\b(?:born|date\s+of\s+birth|dob)\s*(?:[:=-]|\bin\b|\bon\b)?\s*"
            r"(\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
            flags=re.IGNORECASE,
        ),
        group=1,
    ),
    _RulePattern(
        label="religious_belief",
        pattern=re.compile(
            rf"\b(?:describes?\s+(?:himself|herself|themself|themselves)\s+as|"
            rf"identif(?:y|ies)\s+as|raised\s+in\s+the|practicing)\s+"
            rf"(?:a|an|the)?\s*({_RELIGIOUS_BELIEF_RE})\b",
            flags=re.IGNORECASE,
        ),
        group=1,
    ),
    _RulePattern(
        label="street_address",
        pattern=re.compile(
            r"\b(?:lives?\s+at|living\s+at|house\s+on|home\s+on)\s+"
            r"([A-Z0-9][A-Za-z0-9.\s-]{1,60}?\b"
            r"(?:Street|St\.?|Avenue|Ave\.?|Road|Rd\.?|Drive|Dr\.?|Trail|Boulevard|Blvd\.?|Lane|Ln\.?|Court|Ct\.?))",
        ),
        group=1,
    ),
    _RulePattern(
        label="organization_name",
        pattern=re.compile(
            r"\b(?:at|from|with|joining|joined)\s+"
            r"([A-Z][A-Za-z0-9&.'\u2019 -]{2,90}?\b"
            r"(?:Center|Hospital|Clinic|University|College|Institute|Bank|Builders|Construction|Woodworks|Health))"
            r"\b",
        ),
        group=1,
    ),
)

SUPPORTED_RULE_LABELS = frozenset(rule.label for rule in _RULES)
STRUCTURED_RULE_FAST_LANE_LABELS = frozenset(
    {
        "api_key",
        "email",
        "http_cookie",
        "password",
        "pin",
        "unique_id",
        "url",
        "user_name",
    }
)


def detect_high_confidence_entities(text: str, labels: Iterable[str] | None = None) -> list[EntitySpan]:
    """Detect deterministic high-confidence PII and secret spans in raw text.

    These rules intentionally cover narrow, high-signal command/log and prose
    patterns. They are suitable as a local seed detector or benchmark probe,
    not as a complete replacement for model-backed contextual detection.
    """
    allowed_labels = set(labels) if labels is not None else None
    spans: list[EntitySpan] = []

    for rule in _RULES:
        if allowed_labels is not None and rule.label not in allowed_labels:
            continue
        for match in rule.pattern.finditer(text):
            start, end = match.span(rule.group)
            if start < 0 or end <= start:
                continue
            value = text[start:end]
            value, end = _trim_rule_value(label=rule.label, value=value, end=end)
            if not value:
                continue
            spans.append(
                EntitySpan(
                    entity_id=_build_rule_entity_id(label=rule.label, start=start, end=end),
                    value=value,
                    label=rule.label,
                    start_position=start,
                    end_position=end,
                    score=_RULE_SCORE,
                    source=_RULE_SOURCE,
                )
            )

    return resolve_overlaps(_deduplicate(spans))


def _trim_rule_value(*, label: str, value: str, end: int) -> tuple[str, int]:
    if label != "http_cookie":
        return value, end
    trimmed = value.rstrip(".,")
    return trimmed, end - (len(value) - len(trimmed))


def _deduplicate(entities: list[EntitySpan]) -> list[EntitySpan]:
    seen: set[tuple[str, int, int]] = set()
    deduplicated: list[EntitySpan] = []
    for entity in entities:
        key = (entity.label, entity.start_position, entity.end_position)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(entity)
    return deduplicated


def _build_rule_entity_id(*, label: str, start: int, end: int) -> str:
    return f"{label}_{start}_{end}"
