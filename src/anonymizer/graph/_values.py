# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable identity and reference values for protection graphs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Never, SupportsIndex


class ValidationCode(str, Enum):
    """Closed validation outcomes for private graph boundaries."""

    INVALID_TYPE = "invalid_type"
    INVALID_VALUE = "invalid_value"
    LIMIT_EXCEEDED = "limit_exceeded"
    FOREIGN_OWNER = "foreign_owner"
    DUPLICATE = "duplicate"
    MISSING = "missing"
    INVALID_RANGE = "invalid_range"
    OVERLAP = "overlap"
    CYCLE = "cycle"
    CONTRADICTORY = "contradictory"
    UNSUPPORTED = "unsupported"


class ContractViolation(ValueError):
    """A content-free rejection at a private graph boundary."""

    __slots__ = ("code",)

    def __init__(self, code: ValidationCode) -> None:
        self.code = code
        super().__init__(code.value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code.value!r})"


class _PrivateRepr:
    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"


_FACTORY_KEY = object()
_PICKLE_ERROR = "graph identity serialization is not supported"


class _OpaqueIdentity(_PrivateRepr):
    __slots__ = ("_token",)
    _token: object

    def __new__(cls, factory_key: object, **owner: object) -> _OpaqueIdentity:
        del owner
        if factory_key is not _FACTORY_KEY:
            raise TypeError("opaque identities must be created by their factory")
        return super().__new__(cls)

    def __init__(self, factory_key: object) -> None:
        del factory_key
        object.__setattr__(self, "_token", object())

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("opaque identities are immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OpaqueIdentity) and type(self) is type(other) and self._token is other._token

    def __hash__(self) -> int:
        return hash(self._token)

    def __copy__(self) -> _OpaqueIdentity:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _OpaqueIdentity:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Never:
        del protocol
        raise TypeError(_PICKLE_ERROR)


class GraphId(_OpaqueIdentity):
    """Opaque process-local graph owner."""

    __slots__ = ()

    @classmethod
    def new(cls) -> GraphId:
        """Create a fresh graph owner."""
        return cls(_FACTORY_KEY)


class DatumId(_OpaqueIdentity):
    """Opaque process-local datum identity owned by a graph."""

    __slots__ = ("graph",)
    graph: GraphId

    def __init__(self, factory_key: object, *, graph: GraphId) -> None:
        if not isinstance(graph, GraphId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        super().__init__(factory_key)
        object.__setattr__(self, "graph", graph)

    @classmethod
    def new(cls, *, graph: GraphId) -> DatumId:
        """Create a fresh datum identity for ``graph``."""
        return cls(_FACTORY_KEY, graph=graph)


class PlanId(_OpaqueIdentity):
    """Opaque process-local plan owner."""

    __slots__ = ()

    @classmethod
    def new(cls) -> PlanId:
        """Create a fresh plan owner."""
        return cls(_FACTORY_KEY)


class InvocationId(_OpaqueIdentity):
    """Opaque process-local invocation identity owned by a plan."""

    __slots__ = ("plan",)
    plan: PlanId

    def __init__(self, factory_key: object, *, plan: PlanId) -> None:
        if not isinstance(plan, PlanId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        super().__init__(factory_key)
        object.__setattr__(self, "plan", plan)

    @classmethod
    def new(cls, *, plan: PlanId) -> InvocationId:
        """Create a fresh invocation identity for ``plan``."""
        return cls(_FACTORY_KEY, plan=plan)


@dataclass(frozen=True, slots=True, kw_only=True, eq=False, repr=False)
class ActivationKey(_PrivateRepr):
    """Structural identity for one activation occurrence."""

    invocation: InvocationId
    occurrence: int
    parent: ActivationKey | None
    iteration: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, InvocationId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if isinstance(self.occurrence, bool) or not isinstance(self.occurrence, int):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.parent is not None and not isinstance(self.parent, ActivationKey):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.iteration is not None and (isinstance(self.iteration, bool) or not isinstance(self.iteration, int)):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.occurrence < 0 or (self.iteration is not None and self.iteration < 0):
            raise ContractViolation(ValidationCode.INVALID_VALUE)
        if self.parent is not None and self.parent.invocation != self.invocation:
            raise ContractViolation(ValidationCode.FOREIGN_OWNER)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ActivationKey):
            return NotImplemented
        left: ActivationKey | None = self
        right: ActivationKey | None = other
        while left is not None and right is not None:
            if (
                left.invocation != right.invocation
                or left.occurrence != right.occurrence
                or left.iteration != right.iteration
            ):
                return False
            left = left.parent
            right = right.parent
        return left is None and right is None

    def __hash__(self) -> int:
        parts: list[tuple[InvocationId, int, int | None]] = []
        current: ActivationKey | None = self
        while current is not None:
            parts.append((current.invocation, current.occurrence, current.iteration))
            current = current.parent
        return hash(tuple(parts))

    def __copy__(self) -> ActivationKey:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> ActivationKey:
        del memo
        return self


class TaskAttemptId(_OpaqueIdentity):
    """Opaque semantic attempt identity owned by an activation."""

    __slots__ = ("activation",)
    activation: ActivationKey

    def __init__(self, factory_key: object, *, activation: ActivationKey) -> None:
        if not isinstance(activation, ActivationKey):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        super().__init__(factory_key)
        object.__setattr__(self, "activation", activation)

    @classmethod
    def new(cls, *, activation: ActivationKey) -> TaskAttemptId:
        """Create a fresh attempt identity for ``activation``."""
        return cls(_FACTORY_KEY, activation=activation)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ArtifactRef(_PrivateRepr):
    """Structural reference to a versioned invocation artifact."""

    invocation: InvocationId
    key: int
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.invocation, InvocationId):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if isinstance(self.key, bool) or not isinstance(self.key, int):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise ContractViolation(ValidationCode.INVALID_TYPE)
        if self.key < 0 or self.version <= 0:
            raise ContractViolation(ValidationCode.INVALID_VALUE)
