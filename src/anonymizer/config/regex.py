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
            regex.compile(value)
        except regex.error as exc:
            raise ValueError(f"Invalid regex pattern {value!r}: {exc}") from exc
        if _minimum_match_width(value) == 0:
            raise ValueError(f"Regex pattern must not produce zero-width matches: {value!r}")
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
    def serialize_validator(
        self,
        value: RegexValidatorCallable | str | None,
        info: Any,
    ) -> RegexValidatorCallable | str | None:
        if value is None or isinstance(value, str):
            return value
        if info.mode == "json":
            raise ValueError(
                "Direct callable regex validators are in-process only and cannot be serialized to JSON. "
                "Install the validator through the 'nemo_anonymizer.regex_validators' entry-point group "
                "and pass its registered name instead."
            )
        return value


def _minimum_match_width(pattern: str) -> int:
    """Return a conservative structural minimum consumed width.

    ``regex`` does not expose width metadata on its public compiled-pattern
    object. Its compiler uses ``_regex_core`` for the same parse immediately
    before bytecode generation, so inspect that representation to reject any
    branch that can succeed without consuming text. Unknown constructs are
    conservatively assigned zero width. Keep this helper isolated and
    regression-tested because the module is an implementation API of our
    required regex dependency.
    """
    from regex import _regex_core

    global_flags = 0
    while True:
        source = _regex_core.Source(pattern)
        info = _regex_core.Info(global_flags, _regex_core.UNICODE, {})
        setattr(info, "guess_encoding", _regex_core.UNICODE)
        source.ignore_space = bool(info.flags & _regex_core.VERBOSE)
        try:
            parsed = _regex_core._parse_pattern(source, info)
            return _minimum_node_width(parsed, _regex_core)
        except _regex_core._UnscopedFlagSet:
            global_flags = info.global_flags


def _minimum_node_width(node: Any, core: Any) -> int:
    """Return the minimum characters consumed by a parsed regex node."""
    if isinstance(node, (core.ZeroWidthBase, core.LookAround)):
        return 0
    if isinstance(node, core.Sequence):
        width = 0
        for item in node.items:
            if isinstance(item, core.Keep):
                width = 0
            else:
                width += _minimum_node_width(item, core)
        return width
    if isinstance(node, core.Branch):
        return min((_minimum_node_width(branch, core) for branch in node.branches), default=0)
    if isinstance(node, core.GreedyRepeat):
        return node.min_count * _minimum_node_width(node.subpattern, core)
    if isinstance(node, (core.Group, core.Atomic)):
        return _minimum_node_width(node.subpattern, core)
    if isinstance(node, (core.Conditional, core.LookAroundConditional)):
        return min(_minimum_node_width(node.yes_item, core), _minimum_node_width(node.no_item, core))
    if isinstance(node, core.String):
        return len(node.characters)
    if isinstance(node, core.Grapheme):
        return 1
    if isinstance(node, (core.Any, core.Character, core.Property, core.Range, core.SetBase)):
        return 0 if getattr(node, "zerowidth", False) else 1
    return 0
