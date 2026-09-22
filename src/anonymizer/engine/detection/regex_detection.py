# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import ipaddress
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import entry_points
from typing import Any
from unicodedata import normalize
from urllib.parse import urlsplit
from weakref import WeakValueDictionary

import regex
from pydantic import BaseModel, ConfigDict, Field

from anonymizer.config.regex import (
    BuiltinRegex,
    RegexCandidate,
    RegexRule,
    RegexValidationResult,
    RegexValidatorCallable,
)
from anonymizer.engine.detection.postprocess import EntitySpan, coalesce_exact_entity_candidates

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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    rule_id: str
    label: str
    pattern: str
    validator_id: str | None = None
    local_validator: RegexValidatorCallable | None = Field(default=None, exclude=True, repr=False)
    validate_with_llm: bool = True
    source: str


@dataclass(frozen=True)
class RegexDetectionResult:
    """Regex candidates partitioned by contextual validation route."""

    llm_entities: list[EntitySpan]
    accepted_entities: list[EntitySpan]
    validation_trace: list[RegexValidationTrace]


@dataclass(frozen=True)
class RegexValidationTrace:
    """PII-minimized trace metadata for one local-validator decision."""

    rule_id: str
    validator_id: str
    label: str
    start: int
    end: int
    valid: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, str | int | bool | None]:
        return {
            "rule_id": self.rule_id,
            "validator_id": self.validator_id,
            "label": self.label,
            "start": self.start,
            "end": self.end,
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class _LocalValidationOutcome:
    valid: bool
    reason: str | None = None


_LOCAL_VALIDATORS: WeakValueDictionary[str, RegexValidatorCallable] = WeakValueDictionary()
_ENTRY_POINT_VALIDATORS: dict[str, RegexValidatorCallable] = {}


def resolve_regex_rules(
    *,
    labels: Iterable[str],
    builtin_regexes: bool,
    rules: list[BuiltinRegex | RegexRule],
) -> list[ResolvedRegexRule]:
    """Resolve active built-in and custom rules into a serializable form."""
    active_labels = set(labels)
    regex_only_labels = resolve_regex_only_labels(
        labels=active_labels,
        builtin_regexes=builtin_regexes,
        rules=rules,
    )
    resolved: list[ResolvedRegexRule] = []
    custom_rules = [rule for rule in rules if isinstance(rule, RegexRule)]
    builtin_settings = {rule.label: rule for rule in rules if isinstance(rule, BuiltinRegex)}
    for rule in custom_rules:
        if not rule.enabled or rule.label not in active_labels:
            continue
        validator_id = _register_or_resolve_validator(rule.validator)
        resolved.append(
            ResolvedRegexRule(
                rule_id=_build_user_rule_id(rule),
                label=rule.label,
                pattern=rule.pattern,
                validator_id=validator_id,
                local_validator=rule.validator if callable(rule.validator) else None,
                validate_with_llm=rule.validate_with_llm and rule.label not in regex_only_labels,
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
                    update={
                        "validate_with_llm": (
                            False
                            if rule.label in regex_only_labels
                            else (override.validate_with_llm if override is not None else True)
                        )
                    }
                )
            )
    return resolved


def resolve_regex_only_labels(
    *,
    labels: Iterable[str],
    builtin_regexes: bool,
    rules: list[BuiltinRegex | RegexRule],
) -> set[str]:
    """Return active labels whose enabled regex configuration is authoritative."""
    active_labels = set(labels)
    return {
        rule.label
        for rule in rules
        if rule.enabled
        and rule.regex_only
        and rule.label in active_labels
        and (builtin_regexes or isinstance(rule, RegexRule))
    }


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
    validation_trace: list[RegexValidationTrace] = []

    for rule in rules:
        pattern = _compile_pattern(rule.pattern)
        match_count = 0
        try:
            matches = pattern.finditer(text, timeout=timeout_seconds)
            for match in matches:
                start, end = match.span()
                if end <= start:
                    raise RuntimeError(
                        f"Regex rule {rule.rule_id!r} produced a zero-width match at offset {start}. "
                        "Regex rules must consume at least one character."
                    )
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
                validation = _run_local_validator(
                    text=text,
                    rule=rule,
                    value=value,
                    start=start,
                    end=end,
                    groups={key: val or "" for key, val in match.groupdict().items()},
                )
                if rule.validator_id is not None:
                    validation_trace.append(
                        RegexValidationTrace(
                            rule_id=rule.rule_id,
                            validator_id=rule.validator_id,
                            label=rule.label,
                            start=start,
                            end=end,
                            valid=validation.valid,
                            reason=validation.reason,
                        )
                    )
                if not validation.valid:
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
        llm_entities=_coalesce_regex_sources(llm_entities),
        accepted_entities=_coalesce_regex_sources(accepted_entities),
        validation_trace=validation_trace,
    )


@lru_cache(maxsize=256)
def _compile_pattern(pattern: str) -> Any:
    return regex.compile(pattern)


def _coalesce_regex_sources(entities: list[EntitySpan]) -> list[EntitySpan]:
    users = [entity for entity in entities if entity.source.startswith("regex_user:")]
    builtins = [entity for entity in entities if entity.source.startswith("regex_builtin:")]
    return coalesce_exact_entity_candidates(users, builtins)


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


def _run_local_validator(
    *,
    text: str,
    rule: ResolvedRegexRule,
    value: str,
    start: int,
    end: int,
    groups: dict[str, str],
) -> _LocalValidationOutcome:
    if rule.validator_id is None:
        return _LocalValidationOutcome(valid=True)
    validator = rule.local_validator
    if validator is None:
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
    if inspect.isawaitable(result):
        close = getattr(result, "close", None)
        if callable(close):
            close()
        raise TypeError(
            f"Regex validator {rule.validator_id!r} returned an awaitable. Regex validators must be synchronous."
        )
    if isinstance(result, bool):
        return _LocalValidationOutcome(valid=result)
    if isinstance(result, RegexValidationResult):
        return _LocalValidationOutcome(valid=result.valid, reason=result.reason)
    raise TypeError(f"Regex validator {rule.validator_id!r} returned unsupported type {type(result)!r}.")


def _register_or_resolve_validator(
    validator: RegexValidatorCallable | str | None,
) -> str | None:
    if validator is None:
        return None
    if isinstance(validator, str):
        return validator
    module = getattr(validator, "__module__", "")
    qualified_name = getattr(validator, "__qualname__", "")
    validator_id = f"local:{module}:{qualified_name}:{id(validator):x}"
    try:
        _LOCAL_VALIDATORS[validator_id] = validator
    except TypeError:
        # Some callable instances do not support weak references. The resolved
        # in-process rule still owns those validators for its full lifetime.
        pass
    return validator_id


def _resolve_validator(validator_id: str) -> RegexValidatorCallable:
    validator = (
        _VALIDATORS.get(validator_id)
        or _LOCAL_VALIDATORS.get(validator_id)
        or _ENTRY_POINT_VALIDATORS.get(validator_id)
    )
    if validator is not None:
        return validator
    matches = [
        entry_point
        for entry_point in entry_points(group=REGEX_VALIDATOR_ENTRYPOINT_GROUP)
        if entry_point.name == validator_id
    ]
    if len(matches) > 1:
        providers = sorted({_entry_point_provider(entry_point) for entry_point in matches})
        raise ValueError(
            f"Regex validator {validator_id!r} is registered by multiple packages: {', '.join(providers)}. "
            "Validator entry-point names must be unique."
        )
    if matches:
        loaded = matches[0].load()
        if not callable(loaded):
            raise TypeError(f"Regex validator entry point {validator_id!r} is not callable.")
        call_method = getattr(loaded, "__call__", None)
        if inspect.iscoroutinefunction(loaded) or inspect.iscoroutinefunction(call_method):
            raise TypeError(
                f"Regex validator entry point {validator_id!r} is asynchronous. Regex validators must be synchronous."
            )
        _ENTRY_POINT_VALIDATORS[validator_id] = loaded
        return loaded
    raise ValueError(f"Unknown regex validator {validator_id!r}.")


def _entry_point_provider(entry_point: Any) -> str:
    """Return a useful distribution name for an entry-point conflict."""
    distribution = getattr(entry_point, "dist", None)
    name = getattr(distribution, "name", None)
    if isinstance(name, str) and name:
        return name
    metadata = getattr(distribution, "metadata", None)
    if metadata is not None:
        metadata_name = metadata.get("Name")
        if isinstance(metadata_name, str) and metadata_name:
            return metadata_name
    return str(getattr(entry_point, "value", "unknown package"))


def validate_exportable_regex_rules(rules: list[BuiltinRegex | RegexRule] | None) -> None:
    """Require named validators for enabled rules in portable workflow exports."""
    callable_labels = sorted(
        {
            rule.label
            for rule in rules or []
            if isinstance(rule, RegexRule) and rule.enabled and callable(rule.validator)
        }
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
    domain = normalize("NFC", domain)
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
_DNS_LABEL_RE = regex.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_IPV4_SHAPED_HOST_RE = regex.compile(r"^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$")


def _validate_mac(candidate: RegexCandidate) -> bool:
    return _MAC_VALIDATION_RE.fullmatch(candidate.value) is not None


def _validate_url(candidate: RegexCandidate) -> bool:
    target = candidate.value if not candidate.value.lower().startswith("www.") else f"https://{candidate.value}"
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
        host = normalize("NFC", parsed.hostname).encode("idna").decode("ascii")
    except UnicodeError:
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return True

    if ":" in host or _IPV4_SHAPED_HOST_RE.fullmatch(host) is not None:
        return False
    if len(host) > 253 or "." not in host:
        return False
    return all(_DNS_LABEL_RE.fullmatch(label) is not None for label in host.split("."))


_VALIDATORS: dict[str, RegexValidatorCallable] = {
    "nemo-anonymizer.credit-card-luhn.v1": _validate_credit_card,
    "nemo-anonymizer.email.v1": _validate_email,
    "nemo-anonymizer.ipv4.v1": _validate_ipv4,
    "nemo-anonymizer.ipv6.v1": _validate_ipv6,
    "nemo-anonymizer.mac-address.v1": _validate_mac,
    "nemo-anonymizer.url.v1": _validate_url,
}

_BUILTIN_RULES: tuple[ResolvedRegexRule, ...] = (
    ResolvedRegexRule(
        rule_id="nemo-anonymizer.credit-debit-card.v1",
        label="credit_debit_card",
        pattern=r"(?<![0-9])(?:[0-9][ -]?){12,18}[0-9](?![0-9])",
        validator_id="nemo-anonymizer.credit-card-luhn.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo-anonymizer.email.v1",
        label="email",
        pattern=(
            r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
            r"[\p{L}\p{N}!#$%&'*+/=?^_`{|}~-]+(?:\.[\p{L}\p{N}!#$%&'*+/=?^_`{|}~-]+)*@"
            r"[\p{L}\p{N}](?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?"
            r"(?:\.[\p{L}\p{N}](?:[\p{L}\p{N}\p{M}-]{0,61}[\p{L}\p{N}\p{M}])?)+"
            r"(?![A-Za-z0-9_-])"
        ),
        validator_id="nemo-anonymizer.email.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo-anonymizer.ipv4.v1",
        label="ipv4",
        pattern=r"(?<![0-9.])(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9]|\.[0-9])",
        validator_id="nemo-anonymizer.ipv4.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo-anonymizer.ipv6.v1",
        label="ipv6",
        pattern=(
            r"(?<![0-9A-Fa-f:])(?:"
            r"(?:[0-9A-Fa-f]{0,4}:){2,6}(?:[0-9]{1,3}\.){3}[0-9]{1,3}"
            r"|(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}"
            r")(?![0-9A-Fa-f:]|\.[0-9])"
        ),
        validator_id="nemo-anonymizer.ipv6.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo-anonymizer.mac-address.v1",
        label="mac_address",
        pattern=(
            r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}(?![0-9A-Fa-f])"
            r"|(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}(?![0-9A-Fa-f])"
        ),
        validator_id="nemo-anonymizer.mac-address.v1",
        source="regex_builtin",
    ),
    ResolvedRegexRule(
        rule_id="nemo-anonymizer.url.v1",
        label="url",
        pattern=r"(?i)(?<![A-Za-z0-9_])(?:https?://|www\.)[^\s<>\"']+",
        validator_id="nemo-anonymizer.url.v1",
        source="regex_builtin",
    ),
)
