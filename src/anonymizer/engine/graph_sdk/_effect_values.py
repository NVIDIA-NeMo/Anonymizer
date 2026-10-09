# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared closed values for process-local graph effects."""

from __future__ import annotations

from enum import Enum
from typing import Any, Never, Self, SupportsIndex


class EffectCode(str, Enum):
    """Closed rejection codes for request, binding, and execution boundaries."""

    INVALID_TYPE = "invalid_type"
    INVALID_VALUE = "invalid_value"
    LIMIT_EXCEEDED = "limit_exceeded"
    FOREIGN_OWNER = "foreign_owner"
    DUPLICATE = "duplicate"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    CONTRADICTORY = "contradictory"
    CROSS_POLICY = "cross_policy"
    REPLAY_FORBIDDEN = "replay_forbidden"
    MISSING_PREDECESSOR = "missing_predecessor"
    INVALID_RETRY = "invalid_retry"
    INVALID_CORRECTION = "invalid_correction"
    INVALID_FAILOVER = "invalid_failover"
    INVALID_SETTLEMENT = "invalid_settlement"
    INVALID_USAGE = "invalid_usage"
    REQUEST_CAUSALITY = "request_causality"
    PENDING_LIMIT = "pending_limit"
    CAPABILITY_LIMIT = "capability_limit"
    IMPLEMENTATION_COUNT = "implementation_count"
    CHANGED_FAILOVER_POLICY = "changed_failover_policy"
    FOREIGN_SOURCE = "foreign_source"
    UNSOLICITED_SOURCE = "unsolicited_source"
    EXTRA = "extra"


class EffectRejected(ValueError):
    """Content-free rejection at a graph effect boundary."""

    __slots__ = ("code",)

    def __init__(self, code: EffectCode) -> None:
        self.code = code
        super().__init__(code.value)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


class PrivateValue:
    """Content-free, nonserializable immutable-value behavior."""

    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def __copy__(self) -> PrivateValue:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> PrivateValue:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Never:
        del protocol
        raise TypeError("graph effect value serialization is not supported")


_IDENTITY_KEY = object()


class OpaqueIdentity(PrivateValue):
    """Fresh process-local identity with optional immutable owner fields."""

    __slots__ = ("_token",)
    _token: object

    def __new__(cls, key: object, **owners: object) -> OpaqueIdentity:
        del owners
        if key is not _IDENTITY_KEY:
            raise TypeError("effect identities must be created by their factory")
        return super().__new__(cls)

    def __init__(self, key: object, **owners: object) -> None:
        del key, owners
        object.__setattr__(self, "_token", object())

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("effect identities are immutable")

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and isinstance(other, OpaqueIdentity) and self._token is other._token

    def __hash__(self) -> int:
        return hash(self._token)

    @classmethod
    def _new(cls, **owners: object) -> Self:
        value = cls(_IDENTITY_KEY, **owners)
        for name, owner in owners.items():
            object.__setattr__(value, name, owner)
        return value


def reject(code: EffectCode) -> Never:
    raise EffectRejected(code) from None


def require_count(value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        reject(EffectCode.INVALID_TYPE)
    if value < (1 if positive else 0):
        reject(EffectCode.INVALID_VALUE)
    return value


def require_text(value: object, *, empty: bool = False) -> str:
    if not isinstance(value, str):
        reject(EffectCode.INVALID_TYPE)
    if not empty and not value:
        reject(EffectCode.INVALID_VALUE)
    return value


def require_literal(value: object, allowed: frozenset[str]) -> str:
    if not isinstance(value, str):
        reject(EffectCode.INVALID_TYPE)
    if value not in allowed:
        reject(EffectCode.INVALID_VALUE)
    return value


def require_instance(value: object, expected: type[Any] | tuple[type[Any], ...]) -> None:
    if not isinstance(value, expected):
        reject(EffectCode.INVALID_TYPE)
