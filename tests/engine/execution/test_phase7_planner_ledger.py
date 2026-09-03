# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the bounded private Phase 7 planner ledger."""

from __future__ import annotations

import copy
import pickle

import pytest

from anonymizer.engine.execution.phase7_planner_ledger import (
    _PlannerLedger,
    _PlannerSnapshot,
    _PlannerState,
    _Reservation,
)


def test_reservation_is_identity_only_and_terminal_publication_is_one_shot() -> None:
    ledger: _PlannerLedger[str] = _PlannerLedger()
    scope = object()
    reservation, replay = ledger.reserve(scope)

    assert isinstance(reservation, _Reservation)
    assert replay is None
    assert not ledger.owns(scope, _Reservation())
    assert ledger.terminal(scope, reservation, _PlannerSnapshot(_PlannerState.PLANNED, "bundle"))
    assert not ledger.terminal(scope, reservation, _PlannerSnapshot(_PlannerState.PLANNED, "other"))

    replay_reservation, replay = ledger.reserve(scope)
    assert replay_reservation is None
    assert replay == _PlannerSnapshot(_PlannerState.PLANNED, "bundle")


def test_close_preserves_accepted_publication_for_exact_replay_and_rejects_new_work() -> None:
    ledger: _PlannerLedger[str] = _PlannerLedger()
    scope = object()
    reservation, _ = ledger.reserve(scope, ("dispatch",))
    assert isinstance(reservation, _Reservation)
    assert ledger.terminal(scope, reservation, _PlannerSnapshot(_PlannerState.PLANNED, "bundle"))

    ledger.close()

    replay_reservation, replay = ledger.reserve(scope, ("dispatch",))
    assert replay_reservation is None
    assert replay == _PlannerSnapshot(_PlannerState.PLANNED, "bundle")
    with pytest.raises(RuntimeError):
        ledger.reserve(object(), ("new-dispatch",))
    assert not ledger.terminal(scope, reservation, _PlannerSnapshot(_PlannerState.PLANNED, "other"))


def test_close_poison_active_reservations_and_rejects_later_access_or_escape() -> None:
    ledger: _PlannerLedger[str] = _PlannerLedger()
    scope = object()
    reservation, _ = ledger.reserve(scope)
    assert isinstance(reservation, _Reservation)

    with pytest.raises(TypeError):
        copy.copy(ledger)
    with pytest.raises(TypeError):
        pickle.dumps(ledger)

    ledger.close()

    assert ledger.current(scope) == _PlannerSnapshot(_PlannerState.POISONED)
    replay_reservation, replay = ledger.reserve(scope)
    assert replay_reservation is None
    assert replay == _PlannerSnapshot(_PlannerState.POISONED)
    assert not ledger.terminal(scope, reservation, _PlannerSnapshot(_PlannerState.PLANNED, "bundle"))
