# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded whole-group Phase 8 operation state machine."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from anonymizer.engine.execution.phase8_contract import _load_phase8_contract
from anonymizer.engine.execution.phase8_validation import _Phase8Metric, _validate_complete_revisions


@dataclass(frozen=True, slots=True)
class _Phase8GroupOutcome:
    state: str
    revisions: dict[object, str] | None
    repair_iterations: int


@dataclass(frozen=True, slots=True, repr=False)
class _Phase8LifecycleExecution:
    """Content-free terminal receipt for the private grouped lifecycle.

    Candidate revisions exist only while Phase 4 is reducing the invocation.
    The receipt intentionally carries terminal states and released cells, never
    a Phase 7 baseline, a provisional revision, or an operation workframe.
    """

    released: tuple[tuple[object, str], ...]
    terminal_group_states: tuple[str, ...]
    global_embargo: bool
    cleanup_verified: bool


def _run_group_operation(
    members: tuple[object, ...],
    baselines: dict[object, str],
    *,
    analyze: Callable[[], tuple[bool, bool]],
    rewrite: Callable[[dict[object, str]], dict[object, str]],
    evaluate: Callable[[dict[object, str]], _Phase8Metric],
    repair: Callable[[dict[object, str], int], dict[object, str]],
    max_repairs: int,
) -> _Phase8GroupOutcome:
    """Run every stage against one exact group, never adopting a subset."""
    contract = _load_phase8_contract()
    limits = dict(getattr(contract, "limits", ()))
    if (
        max_repairs < 0
        or max_repairs > limits.get("max_repair_iterations", 0)
        or not _validate_complete_revisions(members, baselines)
    ):
        return _Phase8GroupOutcome("failed", None, 0)
    zero_obligations, zero_route_guards = analyze()
    if zero_obligations:
        return (
            _Phase8GroupOutcome("succeeded", dict(baselines), 0)
            if zero_route_guards
            else _Phase8GroupOutcome("failed", None, 0)
        )
    current = rewrite(dict(baselines))
    if not _validate_complete_revisions(members, current):
        return _Phase8GroupOutcome("failed", None, 0)
    for round_number in range(max_repairs + 1):
        metric = evaluate(current)
        if not metric.needs_repair:
            return _Phase8GroupOutcome("succeeded", current, round_number)
        if round_number == max_repairs:
            return _Phase8GroupOutcome("failed", None, round_number)
        current = repair(current, round_number + 1)
        if not _validate_complete_revisions(members, current):
            return _Phase8GroupOutcome("failed", None, round_number + 1)
    return _Phase8GroupOutcome("failed", None, max_repairs)
