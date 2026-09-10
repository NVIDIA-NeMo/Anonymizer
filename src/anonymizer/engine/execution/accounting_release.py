# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure dependency and atomic-group release qualification."""

from __future__ import annotations

from dataclasses import dataclass

from anonymizer.engine.execution.accounting_plan import _AccountingPlan, _AtomicGroupKey
from anonymizer.engine.execution.graph import _DatumId


@dataclass(frozen=True, slots=True, repr=False)
class _ReleaseDecision:
    release_eligible: frozenset[_DatumId]
    released_groups: frozenset[_AtomicGroupKey]
    withheld_groups: frozenset[_AtomicGroupKey]

    def __repr__(self) -> str:
        return "<private release decision>"


def _qualify_release(plan: _AccountingPlan, locally_qualified: frozenset[_DatumId]) -> _ReleaseDecision:
    """Return the least fixed point of group and dependency withholding."""
    eligible = locally_qualified & frozenset(datum.id for datum in plan.datums)
    while True:
        propagated = _propagate_once(plan, eligible)
        if propagated == eligible:
            break
        eligible = propagated
    released = frozenset(group.key for group in plan.atomic_groups if frozenset(group.members).issubset(eligible))
    all_groups = frozenset(group.key for group in plan.atomic_groups)
    return _ReleaseDecision(eligible, released, all_groups - released)


def _propagate_once(plan: _AccountingPlan, eligible: frozenset[_DatumId]) -> frozenset[_DatumId]:
    group_withheld = frozenset(
        member
        for group in plan.atomic_groups
        if not frozenset(group.members).issubset(eligible)
        for member in group.members
    )
    after_groups = eligible - group_withheld
    dependency_withheld = frozenset(
        dependency.dependent for dependency in plan.dependencies if dependency.prerequisite not in after_groups
    )
    return after_groups - dependency_withheld
