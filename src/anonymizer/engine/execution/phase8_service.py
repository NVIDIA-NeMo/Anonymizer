# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private-only composition seam for the Phase 8 grouped profile."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from anonymizer.engine.execution.phase7_application import _AppliedDatum
from anonymizer.engine.execution.phase7_runtime import _Phase7Execution
from anonymizer.engine.execution.phase8_runtime import _Phase8LifecycleExecution, _run_group_operation
from anonymizer.engine.execution.phase8_validation import _Phase8Metric


class _Phase8GroupedRewriteProtectionService:
    """Deliberately not wired into the public Rewrite selector."""

    def run_group(
        self,
        members: tuple[object, ...],
        baselines: dict[object, str],
        *,
        analyze: Callable[[], tuple[bool, bool]],
        rewrite: Callable[[dict[object, str]], dict[object, str]],
        evaluate: Callable[[dict[object, str]], _Phase8Metric],
        repair: Callable[[dict[object, str], int], dict[object, str]],
        max_repairs: int,
    ) -> tuple[tuple[object, str], ...] | None:
        """Return sealed keyed candidates only after a whole group succeeds."""
        outcome = _run_group_operation(
            members,
            baselines,
            analyze=analyze,
            rewrite=rewrite,
            evaluate=evaluate,
            repair=repair,
            max_repairs=max_repairs,
        )
        if outcome.state != "succeeded" or outcome.revisions is None:
            return None
        return tuple((member, outcome.revisions[member]) for member in members)

    def run_lifecycle(
        self,
        *,
        groups: tuple[tuple[object, ...], ...],
        atomic_groups: tuple[tuple[object, ...], ...],
        dependencies: tuple[tuple[object, object], ...],
        phase7_released: tuple[tuple[object, str], ...],
        phase7_cleanup_verified: bool,
        phase7_global_embargo: bool,
        operations: tuple[_GroupOperation, ...],
    ) -> _Phase8LifecycleExecution:
        """Consume only a clean released Phase 7 handoff and reduce via Phase 4.

        This is deliberately a private graph-service seam.  It has no row or
        public result representation: each operation receives the entire
        declared group and its exact released baselines, then Phase 4 withholds
        atomic/dependent cells before any candidate is returned.
        """
        early = _lifecycle_preflight(
            groups,
            atomic_groups,
            dependencies,
            operations,
            phase7_released,
            phase7_cleanup_verified,
            phase7_global_embargo,
        )
        if isinstance(early, _Phase8LifecycleExecution):
            return early
        baselines = early
        candidates, states = _run_operations(groups, operations, baselines)

        qualified = _phase4_qualified(groups, atomic_groups, dependencies, states)
        released = tuple(
            (member, candidates[member]) for members in groups for member in members if member in qualified
        )
        # First cleanup attestation: no candidate-bearing ledger or baseline
        # index survives the Phase 4 reduction.  Clear before constructing the
        # terminal result, then retain only the copied release cells.
        candidates.clear()
        baselines.clear()
        cleanup_verified = not candidates and not baselines
        if not cleanup_verified:
            return _terminal((), tuple(states), True, False)
        # Second attestation is represented by construction: ``released`` is
        # the only value copied past the reduction and every member is unique.
        if len({member for member, _ in released}) != len(released):
            return _terminal((), tuple(states), True, False)
        return _terminal(released, tuple(states), False, True)

    def run_from_phase7_execution(
        self,
        *,
        groups: tuple[tuple[object, ...], ...],
        atomic_groups: tuple[tuple[object, ...], ...],
        dependencies: tuple[tuple[object, object], ...],
        phase7: _Phase7Execution,
        operations: tuple[_GroupOperation, ...],
    ) -> _Phase8LifecycleExecution:
        """Import exactly the Phase 7 release-qualified baseline handoff.

        This adapter intentionally accepts no provisional Phase 7 material.
        A malformed released cell, unverified Phase 7 cleanup, or Phase 7
        embargo is terminal before a Phase 8 operation can be called.
        """
        released = phase7.released
        if not all(isinstance(value, _AppliedDatum) for value in released):
            return _terminal((), tuple("inconsistent" for _ in groups), True, False)
        return self.run_lifecycle(
            groups=groups,
            atomic_groups=atomic_groups,
            dependencies=dependencies,
            phase7_released=tuple((value.datum_id, value.output) for value in released),
            phase7_cleanup_verified=phase7.cleanup.verified,
            phase7_global_embargo=phase7.phase4.global_embargo,
            operations=operations,
        )


_GroupOperation = Callable[[tuple[object, ...], dict[object, str]], tuple[tuple[object, str], ...] | None]


def _terminal(
    released: tuple[tuple[object, str], ...], states: tuple[str, ...], global_embargo: bool, cleanup_verified: bool
) -> _Phase8LifecycleExecution:
    return _Phase8LifecycleExecution(released, states, global_embargo, cleanup_verified)


def _lifecycle_preflight(
    groups: tuple[tuple[object, ...], ...],
    atomic_groups: tuple[tuple[object, ...], ...],
    dependencies: tuple[tuple[object, object], ...],
    operations: tuple[_GroupOperation, ...],
    phase7_released: tuple[tuple[object, str], ...],
    phase7_cleanup_verified: bool,
    phase7_global_embargo: bool,
) -> dict[object, str] | _Phase8LifecycleExecution:
    if not _valid_declarations(groups, atomic_groups, dependencies, operations):
        return _terminal((), (), True, False)
    if not phase7_cleanup_verified or phase7_global_embargo:
        return _terminal((), tuple("blocked" for _ in groups), True, False)
    baselines = _exact_baseline_index(phase7_released)
    return baselines if baselines is not None else _terminal((), tuple("inconsistent" for _ in groups), True, False)


def _run_operations(
    groups: tuple[tuple[object, ...], ...], operations: tuple[_GroupOperation, ...], baselines: dict[object, str]
) -> tuple[dict[object, str], list[str]]:
    candidates: dict[object, str] = {}
    states: list[str] = []
    for members, operation in zip(groups, operations, strict=True):
        group_baselines = {member: baselines[member] for member in members if member in baselines}
        if len(group_baselines) != len(members):
            states.append("blocked")
            continue
        try:
            result = operation(members, group_baselines)
        except BaseException:
            states.append("failed")
            continue
        if not _complete_candidate(members, result):
            states.append("failed")
            continue
        candidates.update(cast(tuple[tuple[object, str], ...], result))
        states.append("succeeded")
    return candidates, states


def _valid_declarations(
    groups: tuple[tuple[object, ...], ...],
    atomic_groups: tuple[tuple[object, ...], ...],
    dependencies: tuple[tuple[object, object], ...],
    operations: tuple[_GroupOperation, ...],
) -> bool:
    if len(groups) == 0 or len(groups) != len(operations):
        return False
    members = tuple(member for group in groups for member in group)
    if not members or len(set(members)) != len(members) or any(not group for group in groups):
        return False
    if any(not atomic or not set(atomic) <= set(members) for atomic in atomic_groups):
        return False
    if set(member for atomic in atomic_groups for member in atomic) != set(members):
        return False
    if any(sum(member in atomic for atomic in atomic_groups) != 1 for member in members):
        return False
    return all(
        isinstance(edge, tuple) and len(edge) == 2 and set(edge) <= set(members) for edge in dependencies
    ) and all(callable(operation) for operation in operations)


def _exact_baseline_index(values: object) -> dict[object, str] | None:
    if not isinstance(values, tuple) or any(not isinstance(item, tuple) or len(item) != 2 for item in values):
        return None
    try:
        result = {member: text for member, text in values}
    except TypeError:
        return None
    return result if len(result) == len(values) and all(isinstance(text, str) for _, text in values) else None


def _complete_candidate(members: tuple[object, ...], result: object) -> bool:
    return (
        isinstance(result, tuple)
        and len(result) == len(members)
        and all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str) for item in result)
        and {item[0] for item in result} == set(members)
        and len({item[0] for item in result}) == len(members)
    )


def _phase4_qualified(
    groups: tuple[tuple[object, ...], ...],
    atomic_groups: tuple[tuple[object, ...], ...],
    dependencies: tuple[tuple[object, object], ...],
    states: list[str],
) -> set[object]:
    eligible = {member for group, state in zip(groups, states, strict=True) if state == "succeeded" for member in group}
    while True:
        next_eligible = {
            member
            for member in eligible
            if all(set(atomic) <= eligible for atomic in atomic_groups if member in atomic)
            and all(prerequisite in eligible for prerequisite, dependent in dependencies if dependent == member)
        }
        if next_eligible == eligible:
            return eligible
        eligible = next_eligible
