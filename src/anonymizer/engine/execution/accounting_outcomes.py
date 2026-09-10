# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Closed terminal outcome algebra for phase-4 accounting."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeAlias, TypeVar, final

from anonymizer.engine.execution.accounting_plan import (
    _AtomicGroupKey,
    _CompiledDependency,
    _StageId,
    _TaskKey,
)
from anonymizer.engine.execution.graph import _DatumId

T = TypeVar("T")


class _PrivateTerminalValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private terminal accounting values are not serializable")


class _CauseCode(str, Enum):
    KNOWN_FAILURE = "known_failure"
    VERIFICATION_FAILED = "verification_failed"
    RELEASE_PREDICATE_FAILED = "release_predicate_failed"
    CANCELLATION = "cancellation"
    STOP_ACKNOWLEDGED = "stop_acknowledged"
    TRANSPORT_LOST = "transport_lost"
    MISSING = "missing"
    DUPLICATE = "duplicate"
    UNKNOWN = "unknown"
    FOREIGN = "foreign"
    STALE = "stale"
    SWAPPED = "swapped"
    CONTRADICTORY = "contradictory"
    PLAN_MISMATCH = "plan_mismatch"
    PREREQUISITE = "prerequisite"
    RESULT_CONSTRUCTION_FAILED = "result_construction_failed"
    CLEANUP_FAILED = "cleanup_failed"
    CLEANUP_UNCONFIRMED = "cleanup_unconfirmed"


@final
@dataclass(frozen=True, slots=True, repr=False)
class _TerminalCause(_PrivateTerminalValue):
    code: _CauseCode


_CAUSE_PRECEDENCE = {code: ordinal for ordinal, code in enumerate(_CauseCode)}


@final
@dataclass(frozen=True, slots=True, repr=False)
class _CauseSet(_PrivateTerminalValue):
    items: tuple[_TerminalCause, ...] = ()

    def __post_init__(self) -> None:
        canonical = tuple(
            _TerminalCause(code)
            for code in sorted({cause.code for cause in self.items}, key=_CAUSE_PRECEDENCE.__getitem__)
        )
        object.__setattr__(self, "items", canonical)

    def __or__(self, other: _CauseSet) -> _CauseSet:
        if not isinstance(other, _CauseSet):
            return NotImplemented
        return _CauseSet((*self.items, *other.items))

    def __iter__(self) -> Iterator[_TerminalCause]:
        return iter(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _TaskSucceeded(_PrivateTerminalValue, Generic[T]):
    task: _TaskKey
    candidate: T


@final
@dataclass(frozen=True, slots=True, repr=False)
class _TaskFailed(_PrivateTerminalValue):
    task: _TaskKey
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _TaskCancelled(_PrivateTerminalValue):
    task: _TaskKey
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _TaskLost(_PrivateTerminalValue):
    task: _TaskKey
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _TaskBlocked(_PrivateTerminalValue):
    task: _TaskKey
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _TaskInconsistent(_PrivateTerminalValue):
    task: _TaskKey
    causes: _CauseSet


_TaskOutcome: TypeAlias = (
    _TaskSucceeded[T] | _TaskFailed | _TaskCancelled | _TaskLost | _TaskBlocked | _TaskInconsistent
)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DatumQualified(_PrivateTerminalValue, Generic[T]):
    datum_id: _DatumId
    candidate: T


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DatumFailed(_PrivateTerminalValue):
    datum_id: _DatumId
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DatumCancelled(_PrivateTerminalValue):
    datum_id: _DatumId
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DatumLost(_PrivateTerminalValue):
    datum_id: _DatumId
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DatumBlocked(_PrivateTerminalValue):
    datum_id: _DatumId
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DatumInconsistent(_PrivateTerminalValue):
    datum_id: _DatumId
    causes: _CauseSet


_DatumOutcome: TypeAlias = (
    _DatumQualified[T] | _DatumFailed | _DatumCancelled | _DatumLost | _DatumBlocked | _DatumInconsistent
)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DependencySatisfied(_PrivateTerminalValue):
    dependency: _CompiledDependency


@final
@dataclass(frozen=True, slots=True, repr=False)
class _DependencyUnsatisfied(_PrivateTerminalValue):
    dependency: _CompiledDependency
    causes: _CauseSet


_DependencyOutcome: TypeAlias = _DependencySatisfied | _DependencyUnsatisfied


@final
@dataclass(frozen=True, slots=True, repr=False)
class _StageSucceeded(_PrivateTerminalValue):
    stage: _StageId


@final
@dataclass(frozen=True, slots=True, repr=False)
class _StageFailed(_PrivateTerminalValue):
    stage: _StageId
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _StageCancelled(_PrivateTerminalValue):
    stage: _StageId
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _StageLost(_PrivateTerminalValue):
    stage: _StageId
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _StageBlocked(_PrivateTerminalValue):
    stage: _StageId
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _StageInconsistent(_PrivateTerminalValue):
    stage: _StageId
    causes: _CauseSet


_StageOutcome: TypeAlias = (
    _StageSucceeded | _StageFailed | _StageCancelled | _StageLost | _StageBlocked | _StageInconsistent
)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _GroupReleased(_PrivateTerminalValue, Generic[T]):
    group: _AtomicGroupKey
    outputs: tuple[tuple[_DatumId, T], ...]


@final
@dataclass(frozen=True, slots=True, repr=False)
class _GroupWithheld(_PrivateTerminalValue):
    group: _AtomicGroupKey
    causes: _CauseSet


_GroupOutcome: TypeAlias = _GroupReleased[T] | _GroupWithheld


@final
@dataclass(frozen=True, slots=True, repr=False)
class _InvocationCompleted(_PrivateTerminalValue, Generic[T]):
    groups: tuple[_GroupOutcome[T], ...]


@final
@dataclass(frozen=True, slots=True, repr=False)
class _InvocationFailed(_PrivateTerminalValue):
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _InvocationCancelled(_PrivateTerminalValue):
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _InvocationLost(_PrivateTerminalValue):
    causes: _CauseSet


@final
@dataclass(frozen=True, slots=True, repr=False)
class _InvocationInconsistent(_PrivateTerminalValue):
    causes: _CauseSet


_InvocationOutcome: TypeAlias = (
    _InvocationCompleted[T] | _InvocationFailed | _InvocationCancelled | _InvocationLost | _InvocationInconsistent
)


@final
@dataclass(frozen=True, slots=True, repr=False)
class _AccountingResult(_PrivateTerminalValue, Generic[T]):
    tasks: tuple[_TaskOutcome[T], ...]
    datums: tuple[_DatumOutcome[T], ...]
    dependencies: tuple[_DependencyOutcome, ...]
    stages: tuple[_StageOutcome, ...]
    groups: tuple[_GroupOutcome[T], ...]
    invocation: _InvocationOutcome[T]
