# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ipaddress
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import entry_points
from typing import Any
from urllib.parse import urlsplit

import regex
from pydantic import BaseModel

from anonymizer.config.regex import (
    BuiltinRegex,
    RegexCandidate,
    RegexRule,
    RegexValidationResult,
    RegexValidatorCallable,
)
from anonymizer.engine.detection.postprocess import EntitySpan, merge_entity_sources

DEFAULT_REGEX_TIMEOUT_SECONDS = 0.05
DEFAULT_MAX_MATCHES_PER_RULE = 1000
REGEX_VALIDATOR_ENTRYPOINT_GROUP = "nemo_anonymizer.regex_validators"
_REGEX_SCORE = 1.0
_CONTEXT_WINDOW = 64
_URL_TRAILING_PUNCTUATION = ".,;:!?>'\"。，、；：！？"
_URL_DELIMITER_PAIRS = {
    ")": "(",
    "]": "[",
    "}": "{",
    "）": "（",
    "】": "【",
    "》": "《",
    "」": "「",
    "』": "『",
}


class ResolvedRegexRule(BaseModel):
    """Serializable rule consumed by the DataDesigner regex column."""

    rule_id: str
    label: str
    pattern: str
    validator_id: str | None = None
    validate_with_llm: bool = True
    source: str


@dataclass(frozen=True)
class RegexDetectionResult:
    """Regex candidates partitioned by contextual validation route."""

    llm_entities: list[EntitySpan]
    accepted_entities: list[EntitySpan]


_LOCAL_VALIDATORS: dict[str, RegexValidatorCallable] = {}


def resolve_regex_rules(
    *,
    labels: Iterable[str],
    builtin_regexes: bool,
    rules: list[BuiltinRegex | RegexRule],
) -> list[ResolvedRegexRule]:
    """Resolve active built-in and custom rules into a serializable form."""
    active_labels = set(labels)
    resolved: list[ResolvedRegexRule] = []
    custom_rules = [rule for rule in rules if isinstance(rule, RegexRule)]
    builtin_settings = {rule.label: rule for rule in rules if isinstance(rule, BuiltinRegex)}
    for rule in custom_rules:
        if not rule.enabled:
            continue
        validator_id = _register_or_resolve_validator(rule.validator)
        resolved.append(
            ResolvedRegexRule(
                rule_id=_build_user_rule_id(rule),
                label=rule.label,
                pattern=rule.pattern,
                validator_id=validator_id,
                validate_with_llm=rule.validate_with_llm,
                source="regex_user",
            )
        )

    if builtin_regexes:
        for rule in _BUILTIN_RULES:
            if rule.label not in active_labels:
                continue
            override = builtin_settings.get(rule.label)
            if override is not None and not override.enabled:
                continue
            resolved.append(
                rule.model_copy(
                    update={"validate_with_llm": (override.validate_with_llm if override is not None else True)}
                )
            )
    return resolved


def _build_user_rule_id(rule: RegexRule) -> str:
    digest = sha256(rule.pattern.encode("utf-8")).hexdigest()[:12]
    return f"user:{rule.label}:v1:{digest}"


def detect_regex_entities(
    text: str,
    *,
    rules: list[ResolvedRegexRule],
    timeout_seconds: float = DEFAULT_REGEX_TIMEOUT_SECONDS,
    max_matches_per_rule: int = DEFAULT_MAX_MATCHES_PER_RULE,
) -> RegexDetectionResult:
    """Match, locally validate, and route regex entity candidates."""
    llm_entities: list[EntitySpan] = []
    accepted_entities: list[EntitySpan] = []

    for rule in rules:
        pattern = _compile_pattern(rule.pattern)
        match_count = 0
        try:
            matches = pattern.finditer(text, timeout=timeout_seconds)
            for match in matches:
                start, end = match.span()
                if end <= start:
                    continue
                match_count += 1
                if match_count > max_matches_per_rule:
                    raise RuntimeError(
                        f"Regex rule {rule.rule_id!r} exceeded the maximum of "
                        f"{max_matches_per_rule} matches for one record."
                    )
                if rule.label == "url":
                    trimmed = _trim_url_trailing_punctuation(text[start:end])
                    end = start + len(trimmed)
                    if end <= start:
                        continue
                value = text[start:end]
                if not _passes_validator(
                    text=text,
                    rule=rule,
                    value=value,
                    start=start,
                    end=end,
                    groups={key: val or "" for key, val in match.groupdict().items()},
                ):
                    continue
                entity = EntitySpan(
                    entity_id=f"{rule.label}_{start}_{end}",
                    value=value,
                    label=rule.label,
                    start_position=start,
                    end_position=end,
                    score=_REGEX_SCORE,
                    source=f"{rule.source}:{rule.rule_id}",
                )
                if rule.validate_with_llm:
                    llm_entities.append(entity)
                else:
                    accepted_entities.append(entity)
        except TimeoutError as exc:
            raise RuntimeError(f"Regex rule {rule.rule_id!r} timed out after {timeout_seconds} seconds.") from exc

    return RegexDetectionResult(
        llm_entities=_merge_regex_sources(llm_entities),
        accepted_entities=_merge_regex_sources(accepted_entities),
    )


@lru_cache(maxsize=256)
def _compile_pattern(pattern: str) -> Any:
    return regex.compile(pattern)


def _merge_regex_sources(entities: list[EntitySpan]) -> list[EntitySpan]:
    users = [entity for entity in entities if entity.source.startswith("regex_user:")]
    builtins = [entity for entity in entities if entity.source.startswith("regex_builtin:")]
    return merge_entity_sources(_deduplicate(users), _deduplicate(builtins))


def _trim_url_trailing_punctuation(value: str) -> str:
    """Remove prose punctuation while preserving balanced URL delimiters."""
    trimmed = value.rstrip(_URL_TRAILING_PUNCTUATION)
    while trimmed:
        closing = trimmed[-1]
        opening = _URL_DELIMITER_PAIRS.get(closing)
        if opening is None or trimmed.count(closing) <= trimmed.count(opening):
            break
        trimmed = trimmed[:-1].rstrip(_URL_TRAILING_PUNCTUATION)
    return trimmed


def _passes_validator(
    *,
    text: str,
    rule: ResolvedRegexRule,
    value: str,
    start: int,
    end: int,
    groups: dict[str, str],
) -> bool:
    if rule.validator_id is None:
        return True
    validator = _resolve_validator(rule.validator_id)
    before = max(0, start - _CONTEXT_WINDOW)
    after = min(len(text), end + _CONTEXT_WINDOW)
    candidate = RegexCandidate(
        value=value,
        start=start,
        end=end,
        groups=groups,
        context=text[before:after],
        rule_id=rule.rule_id,
    )
    try:
        result = validator(candidate)
    except Exception as exc:
        raise RuntimeError(
            f"Regex validator {rule.validator_id!r} failed for rule {rule.rule_id!r} with {type(exc).__name__}."
        ) from exc
    if isinstance(result, bool):
        return result
    if isinstance(result, RegexValidationResult):
        return result.valid
    raise TypeError(f"Regex validator {rule.validator_id!r} returned unsupported type {type(result)!r}.")


def _register_or_resolve_validator(
    validator: RegexValidatorCallable | str | None,
) -> str | None:
    if validator is None:
        return None
    if isinstance(validator, str):
        _resolve_validator(validator)
        return validator
    module = getattr(validator, "__module__", "")
    qualified_name = getattr(validator, "__qualname__", "")
    validator_id = f"{module}:{qualified_name}"
    existing = _LOCAL_VALIDATORS.get(validator_id)
    if existing is not None and existing is not validator:
        raise ValueError(f"Duplicate regex validator registration for {validator_id!r}.")
    _LOCAL_VALIDATORS[validator_id] = validator
    return validator_id


def _resolve_validator(validator_id: str) -> RegexValidatorCallable:
    validator = _VALIDATORS.get(validator_id) or _LOCAL_VALIDATORS.get(validator_id)
    if validator is not None:
        return validator
    for entry_point in entry_points(group=REGEX_VALIDATOR_ENTRYPOINT_GROUP):
        if entry_point.name != validator_id:
            continue
        loaded = entry_point.load()
        if not callable(loaded):
            raise TypeError(f"Regex validator entry point {validator_id!r} is not callable.")
        _LOCAL_VALIDATORS[validator_id] = loaded
        return loaded
    raise ValueError(f"Unknown regex validator {validator_id!r}.")


def validate_exportable_regex_rules(rules: list[BuiltinRegex | RegexRule] | None) -> None:
    """Require installed validator names for portable workflow exports."""
    callable_labels = sorted(
        {rule.label for rule in rules or [] if isinstance(rule, RegexRule) and callable(rule.validator)}
    )
    if callable_labels:
        raise ValueError(
            "Exported detection workflows require registered validator names for regex rules "
            f"{callable_labels!r}; install validators through the "
            f"{REGEX_VALIDATOR_ENTRYPOINT_GROUP!r} entry-point group and pass their names."
        )


def _validate_credit_card(candidate: RegexCandidate) -> bool:
    digits = "".join(char for char in candidate.value if char in "0123456789")
    if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
        return False
    total = 0
    parity = len(digits) % 2
    for index, character in enumerate(digits):
        digit = int(character)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _validate_email(candidate: RegexCandidate) -> bool:
    value = candidate.value
    if len(value) > 254 or value.count("@") != 1:
        return False
    local, domain = value.rsplit("@", 1)
    if not local or len(local.encode("utf-8")) > 64:
        return False
    if local.startswith(".") or local.endswith(".") or ".." in local:
        return False
    try:
        encoded_domain = domain.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if len(encoded_domain) > 253 or "." not in encoded_domain:
        return False
    labels = encoded_domain.split(".")
    return all(label and len(label) <= 63 and not label.startswith("-") and not label.endswith("-") for label in labels)


def _validate_ipv4(candidate: RegexCandidate) -> bool:
    try:
        ipaddress.IPv4Address(candidate.value)
    except ValueError:
        return False
    return True


def _validate_ipv6(candidate: RegexCandidate) -> bool:
    try:
        ipaddress.IPv6Address(candidate.value.strip("[]"))
    except ValueError:
        return False
    return True


_MAC_VALIDATION_RE = regex.compile(
    r"^(?:[0-9A-Fa-f]{2}([:-]))(?:[0-9A-Fa-f]{2}\1){4}[0-9A-Fa-f]{2}$"
    r"|^(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}$"
)


def _validate_mac(candidate: RegexCandidate) -> bool:
    return _MAC_VALIDATION_RE.fullmatch(candidate.value) is not None


def _validate_url(candidate: RegexCandidate) -> bool:
    target = candidate.value if not candidate.value.startswith("www.") else f"https://{candidate.value}"
    try:
        parsed = urlsplit(target)
        port = parsed.port
    except ValueError:
        return False
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return False
    if port is not None and not 1 <= port <= 65535:
        return False
    try:
        host = parsed.hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if "." in host:
        return True
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


_VALIDATORS: dict[str, RegexValidatorCallable] = {
    "nemo.credit-card-luhn.v1": _validate_credit_card,
    "nemo.email.v1": _validate_email,
    "nemo.ipv4.v1": _validate_ipv4,
    "nemo.ipv6.v1": _validate_ipv6,
    "nemo.mac-address.v1": _validate_mac,
    "nemo.url.v1": _validate_url,
}

_BUILTIN_RULES: tuple[ResolvedRegexRule, ...] = (
    ResolvedRegexRule(
        rule_id="nemo.credit-debit-card.v1",
        label="credit_debit_card",
        pattern=r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![0-9])",
        validator_id="nemo.credit-card-luhn.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo.email.v1",
        label="email",
        pattern=(
            r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
            r"[\p{L}\p{N}!#$%&'*+/=?^_`{|}~-]+(?:\.[\p{L}\p{N}!#$%&'*+/=?^_`{|}~-]+)*@"
            r"[\p{L}\p{N}](?:[\p{L}\p{N}-]{0,61}[\p{L}\p{N}])?"
            r"(?:\.[\p{L}\p{N}](?:[\p{L}\p{N}-]{0,61}[\p{L}\p{N}])?)+"
            r"(?![A-Za-z0-9_-])"
        ),
        validator_id="nemo.email.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo.ipv4.v1",
        label="ipv4",
        pattern=r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])",
        validator_id="nemo.ipv4.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo.ipv6.v1",
        label="ipv6",
        pattern=r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])",
        validator_id="nemo.ipv6.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo.mac-address.v1",
        label="mac_address",
        pattern=(
            r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])"
            r"|(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}(?![0-9A-Fa-f])"
        ),
        validator_id="nemo.mac-address.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo.url.v1",
        label="url",
        pattern=r"(?i)(?<![A-Za-z0-9_])(?:https?://|www\.)[^\s<>\"']+",
        validator_id="nemo.url.v1",
        source="regex_builtin",
    ),
)


def _deduplicate(entities: list[EntitySpan]) -> list[EntitySpan]:
    seen: set[tuple[str, int, int]] = set()
    result: list[EntitySpan] = []
    for entity in entities:
        key = (entity.label, entity.start_position, entity.end_position)
        if key in seen:
            continue
        seen.add(key)
        result.append(entity)
    return result
