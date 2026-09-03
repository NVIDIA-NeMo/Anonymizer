# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sealed Phase 6/7 predecessor authority for the private Phase 8 profile."""

from __future__ import annotations

from dataclasses import dataclass, field

from anonymizer.engine.execution.phase6_plan import _is_admitted_phase6_plan, _Phase6Plan
from anonymizer.engine.execution.phase6_runtime import (
    _is_admitted_phase6_execution,
    _is_admitted_substitute_handoff,
    _Phase6Execution,
)
from anonymizer.engine.execution.phase7_runtime import _Phase7Execution


class _PrivatePhase8SuccessorValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private Phase 8 successor authority is not serializable")


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8SuccessorHandoff(_PrivatePhase8SuccessorValue):
    """Opaque, invocation-local authority produced only by the Phase 7 owner."""

    phase6_plan: _Phase6Plan
    phase6_execution: _Phase6Execution
    phase7_execution: _Phase7Execution
    _seal: object = field(compare=False, default=None)
    _snapshot: tuple[int, ...] = field(compare=False, default=())


_SUCCESSOR_SEAL = object()


def _seal_phase8_successor(
    phase6_plan: _Phase6Plan,
    phase6_execution: _Phase6Execution,
    phase7_execution: _Phase7Execution,
) -> _Phase8SuccessorHandoff | None:
    """Seal the exact predecessor objects while their owning service holds them."""
    if not _valid_predecessor(phase6_plan, phase6_execution, phase7_execution):
        return None
    return _Phase8SuccessorHandoff(
        phase6_plan,
        phase6_execution,
        phase7_execution,
        _SUCCESSOR_SEAL,
        (id(phase6_plan), id(phase6_execution), id(phase7_execution)),
    )


def _is_admitted_phase8_successor(value: object) -> bool:
    if not isinstance(value, _Phase8SuccessorHandoff) or value._seal is not _SUCCESSOR_SEAL:
        return False
    if value._snapshot != (id(value.phase6_plan), id(value.phase6_execution), id(value.phase7_execution)):
        return False
    return _valid_predecessor(value.phase6_plan, value.phase6_execution, value.phase7_execution)


def _valid_predecessor(
    phase6_plan: object,
    phase6_execution: object,
    phase7_execution: object,
) -> bool:
    return (
        isinstance(phase6_plan, _Phase6Plan)
        and _is_admitted_phase6_plan(phase6_plan)
        and isinstance(phase6_execution, _Phase6Execution)
        and _is_admitted_phase6_execution(phase6_execution, phase6_plan)
        and isinstance(phase7_execution, _Phase7Execution)
        and isinstance(phase7_execution.released, tuple)
        and phase7_execution.cleanup.verified
        and not phase7_execution.phase4.global_embargo
        and isinstance(phase6_execution.handoffs, tuple)
        and all(_is_admitted_substitute_handoff(item, phase6_plan) for item in phase6_execution.handoffs)
    )
