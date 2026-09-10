# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Invocation-private, one-shot ownership for Phase 7 scope planning.

This deliberately is not an accounting ledger: Phase 4 remains the only
authority which releases work.  The small state machine here only prevents a
planner invocation from producing more than one effect for an admitted scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import Generic, TypeVar

T = TypeVar("T")


class _PlannerState(str, Enum):
    PENDING = "pending"
    PLANNED = "planned"
    ABORTED = "aborted"
    POISONED = "poisoned"


@dataclass(frozen=True, slots=True, repr=False)
class _PlannerSnapshot(Generic[T]):
    state: _PlannerState
    value: T | None = None
    trusted_stop: bool = False


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _Reservation:
    """Identity-only capability; equality must never confer authority."""


@dataclass(slots=True)
class _Entry(Generic[T]):
    reservation: _Reservation | None = None
    snapshot: _PlannerSnapshot[T] | None = None
    evidence: object | None = None
    dispatched: bool = False


class _PlannerLedger(Generic[T]):
    """Closed, non-serializable per-backend ledger keyed by scope capability."""

    __slots__ = ("__entries", "__closed", "__lock")

    def __init__(self) -> None:
        self.__entries: dict[object, _Entry[T]] = {}
        self.__closed = False
        self.__lock = RLock()

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 7 planner ledger is not serializable")

    def __copy__(self) -> _PlannerLedger[T]:
        raise TypeError("private Phase 7 planner ledger cannot escape")

    __deepcopy__ = __copy__

    def reserve(
        self, scope: object, evidence: object | None = None
    ) -> tuple[_Reservation | None, _PlannerSnapshot[T] | None]:
        with self.__lock:
            entry = self.__entries.get(scope)
            if entry is None:
                self.__active()
                entry = _Entry()
                self.__entries[scope] = entry
            if entry.snapshot is not None:
                if entry.evidence != evidence:
                    # Do not disclose a previously accepted private result to
                    # stale or foreign terminal evidence.
                    return None, _PlannerSnapshot(_PlannerState.POISONED)
                return None, entry.snapshot
            self.__active()
            if entry.reservation is not None:
                if entry.evidence != evidence:
                    # Conflicting evidence while a candidate is still
                    # unaccepted is ambiguous and closes this scope.
                    entry.snapshot = _PlannerSnapshot(_PlannerState.POISONED)
                    entry.reservation = None
                    return None, entry.snapshot
                return None, _PlannerSnapshot(_PlannerState.PENDING)
            reservation = _Reservation()
            entry.reservation = reservation
            entry.evidence = evidence
            return reservation, None

    def current(self, scope: object) -> _PlannerSnapshot[T] | None:
        with self.__lock:
            entry = self.__entries.get(scope)
            return None if entry is None else entry.snapshot

    def owns(self, scope: object, reservation: object) -> bool:
        with self.__lock:
            self.__active()
            entry = self.__entries.get(scope)
            return entry is not None and entry.reservation is reservation and entry.snapshot is None

    def mark_dispatched(self, scope: object, reservation: object) -> bool:
        with self.__lock:
            if self.__closed:
                return False
            entry = self.__entries.get(scope)
            if entry is None or entry.reservation is not reservation or entry.snapshot is not None:
                return False
            entry.dispatched = True
            return True

    def terminal(self, scope: object, reservation: object, snapshot: _PlannerSnapshot[T]) -> bool:
        with self.__lock:
            if self.__closed:
                return False
            entry = self.__entries.get(scope)
            if entry is None or entry.reservation is not reservation or entry.snapshot is not None:
                return False
            entry.snapshot = snapshot
            entry.reservation = None
            return True

    def cancel(self, scope: object, *, trusted_stop: bool, value: T | None = None) -> _PlannerSnapshot[T] | None:
        with self.__lock:
            self.__active()
            entry = self.__entries.get(scope)
            if entry is None:
                return None
            if entry.snapshot is not None:
                return entry.snapshot
            if entry.reservation is None:
                return None
            state = _PlannerState.ABORTED if not entry.dispatched or trusted_stop else _PlannerState.POISONED
            entry.snapshot = _PlannerSnapshot(state, value, entry.dispatched and trusted_stop)
            entry.reservation = None
            return entry.snapshot

    def close(self) -> None:
        with self.__lock:
            if self.__closed:
                return
            for entry in self.__entries.values():
                if entry.snapshot is None:
                    entry.snapshot = _PlannerSnapshot(_PlannerState.POISONED)
                    entry.reservation = None
            self.__closed = True

    def discard_values(self) -> None:
        """Drop every private bundle after lifecycle cleanup has been verified."""
        with self.__lock:
            if not self.__closed:
                raise RuntimeError("private Phase 7 planner ledger is still active")
            for entry in self.__entries.values():
                if entry.snapshot is not None:
                    entry.snapshot = _PlannerSnapshot(entry.snapshot.state)
                entry.evidence = None

    def cleanup_observation(self) -> tuple[int, int, bool] | None:
        """Return content-free, post-retirement facts from this sealed ledger.

        ``None`` means that closure/retirement was not completed, so callers
        must embargo rather than manufacture a zero-reference attestation.
        """
        with self.__lock:
            if not self.__closed:
                return None
            active = sum(entry.reservation is not None for entry in self.__entries.values())
            provisional = sum(
                entry.snapshot is not None and entry.snapshot.value is not None for entry in self.__entries.values()
            )
            observable = any(entry.evidence is not None for entry in self.__entries.values())
            return active, provisional, observable

    def __active(self) -> None:
        if self.__closed:
            raise RuntimeError("private Phase 7 planner ledger is closed")
