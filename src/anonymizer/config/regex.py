# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

import regex
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

MAX_REGEX_PATTERN_LENGTH = 4096
BUILTIN_REGEX_LABELS: tuple[str, ...] = (
    "credit_debit_card",
    "email",
    "ipv4",
    "ipv6",
    "mac_address",
    "url",
)


@dataclass(frozen=True)
class RegexCandidate:
    """A regex match supplied to a user-defined local validator."""

    value: str
    start: int
    end: int
    groups: Mapping[str, str]
    context: str
    rule_id: str


@dataclass(frozen=True)
class RegexValidationResult:
    """Result returned by a user-defined local regex validator."""

    valid: bool
    reason: str | None = None
    normalized_value: str | None = None


RegexValidatorReturn: TypeAlias = bool | RegexValidationResult
RegexValidatorCallable: TypeAlias = Callable[[RegexCandidate], RegexValidatorReturn]


class BuiltinRegex(BaseModel):
    """Configuration for one recognizer from the built-in regex registry."""

    label: str
    enabled: bool = True
    validate_with_llm: bool = True

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if cleaned not in BUILTIN_REGEX_LABELS:
            raise ValueError(
                f"Unsupported built-in regex label {cleaned!r}. Supported labels are {list(BUILTIN_REGEX_LABELS)!r}."
            )
        return cleaned


class RegexRule(BaseModel):
    """A user-defined regex rule for producing entity candidates."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    pattern: str
    validator: RegexValidatorCallable | str | None = None
    enabled: bool = True
    validate_with_llm: bool = True

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        cleaned = value.strip().lower()
        if not cleaned:
            raise ValueError("Regex rule label must not be empty.")
        return cleaned

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, value: str) -> str:
        if not value:
            raise ValueError("Regex rule pattern must not be empty.")
        if len(value) > MAX_REGEX_PATTERN_LENGTH:
            raise ValueError(
                f"Regex rule pattern length {len(value)} exceeds the maximum of {MAX_REGEX_PATTERN_LENGTH}."
            )
        try:
            compiled = regex.compile(value)
        except regex.error as exc:
            raise ValueError(f"Invalid regex pattern {value!r}: {exc}") from exc
        match = compiled.search("")
        if match is not None and match.start() == match.end():
            raise ValueError(f"Regex pattern must not match an empty string: {value!r}")
        return value

    @field_validator("validator")
    @classmethod
    def validate_validator(cls, value: Any) -> RegexValidatorCallable | str | None:
        if value is None or callable(value):
            return value
        if isinstance(value, str) and value.strip():
            return value.strip()
        raise ValueError("Regex rule validator must be a callable or non-empty registered name.")

    @field_serializer("validator")
    def serialize_validator(self, value: RegexValidatorCallable | str | None) -> str | None:
        if value is None or isinstance(value, str):
            return value
        module = getattr(value, "__module__", "")
        qualified_name = getattr(value, "__qualname__", "")
        if not module or not qualified_name or "<locals>" in qualified_name or qualified_name == "<lambda>":
            raise ValueError("Custom regex validator must be a top-level named function to serialize.")
        return f"{module}:{qualified_name}"
