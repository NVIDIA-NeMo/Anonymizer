# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class Support:
    reference: str
    node: str
    outcome: str
    promise: str
    coverage: frozenset[str]


@dataclass(frozen=True)
class Evidence:
    reference: str
    node: str
    outcome: str
    promise: str
    coverage: frozenset[str]
    consumed: frozenset[str]
    observed_revision: int = 1
    current_revision: int = 1


@dataclass(frozen=True)
class Decision:
    key: str
    supports: tuple[Support, ...]


@dataclass(frozen=True)
class Target:
    name: str
    required: frozenset[str]
    evidence: frozenset[str]
    prerequisites: frozenset[str] = frozenset()
    atomic: frozenset[str] = frozenset()


_SEAL = object()


@dataclass(frozen=True, init=False)
class Bundle:
    targets: tuple[Target, ...]
    decisions: tuple[Decision, ...]
    verified: tuple[Evidence, ...]

    def __init__(
        self,
        targets: tuple[Target, ...],
        decisions: tuple[Decision, ...],
        verified: tuple[Evidence, ...],
        *,
        _seal: object,
    ) -> None:
        if _seal is not _SEAL:
            raise ValueError("factory provenance")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "verified", verified)


def _fixture_bundle(
    targets: tuple[Target, ...],
    decisions: tuple[Decision, ...],
    verified: tuple[Evidence, ...],
) -> Bundle:
    """Finite fixture authority; not an adapter for caller-provided acceptance flags."""
    known = {e.reference: e for e in verified}
    if len(known) != len(verified) or len({t.name for t in targets}) != len(targets):
        raise ValueError("duplicate finite identity")
    for target in targets:
        if not target.evidence <= known.keys():
            raise ValueError("foreign evidence")
        if any(not known[ref].consumed <= target.required for ref in target.evidence):
            raise ValueError("accepted evidence consumption must be required")
    return Bundle(targets, decisions, verified, _seal=_SEAL)


def subsets(names: frozenset[str]) -> tuple[frozenset[str], ...]:
    ordered = sorted(names)
    return tuple(frozenset(part) for size in range(len(ordered) + 1) for part in combinations(ordered, size))


def feasible(bundle: Bundle, survivors: frozenset[str], conflicts: frozenset[str]) -> bool:
    """Declarative admissible subset, without simulating a commit algorithm."""
    if type(bundle) is not Bundle:
        raise TypeError("factory bundle required")
    selected = tuple(t for t in bundle.targets if t.name in survivors)
    if survivors != frozenset(t.name for t in selected):
        return False
    required = frozenset(k for t in selected for k in t.required)
    if required & conflicts:
        return False
    if any(not (t.prerequisites | t.atomic) <= survivors for t in selected):
        return False
    decisions = {d.key: d for d in bundle.decisions}
    for key in required:
        if key not in decisions or not decisions[key].supports:
            return False
        for support in decisions[key].supports:
            if not any(
                key in target.required
                and evidence.reference in target.evidence
                and evidence.observed_revision == evidence.current_revision
                and evidence.node == support.node
                and evidence.outcome == support.outcome
                and evidence.promise == support.promise
                and support.coverage <= evidence.coverage
                and key in evidence.consumed
                and evidence.consumed <= required
                for target in selected
                for evidence in bundle.verified
            ):
                return False
    return True


def greatest(bundle: Bundle, conflicts: frozenset[str] = frozenset()) -> frozenset[str]:
    """Enumerate all tiny feasible subsets; union is the greatest solution."""
    if type(bundle) is not Bundle:
        raise TypeError("factory bundle required")
    names = frozenset(t.name for t in bundle.targets)
    return frozenset().union(*(s for s in subsets(names) if feasible(bundle, s, conflicts)))


def deletion_rounds(bundle: Bundle, conflicts: frozenset[str]) -> tuple[frozenset[str], ...]:
    """Independent descending formulation used only to check the subset oracle."""
    current = frozenset(t.name for t in bundle.targets)
    history = [current]
    while True:
        required = frozenset(k for t in bundle.targets if t.name in current for k in t.required)
        supported = frozenset(
            d.key
            for d in bundle.decisions
            if d.supports
            and all(
                any(
                    t.name in current
                    and d.key in t.required
                    and e.reference in t.evidence
                    and e.observed_revision == e.current_revision
                    and (e.node, e.outcome, e.promise) == (s.node, s.outcome, s.promise)
                    and s.coverage <= e.coverage
                    and d.key in e.consumed
                    and e.consumed <= required
                    for t in bundle.targets
                    for e in bundle.verified
                )
                for s in d.supports
            )
        )
        following = frozenset(
            t.name
            for t in bundle.targets
            if t.name in current
            and not t.required & conflicts
            and t.required <= supported
            and (t.prerequisites | t.atomic) <= current
        )
        if following == current:
            return tuple(history)
        history.append(following)
        current = following


@dataclass(frozen=True)
class Entry:
    key: str
    revision: int
    value: str
    supports: tuple[str, ...]


def key_conflict(
    change: str,
    prior: int | None,
    current: Entry | None,
    value: str,
    supports: tuple[str, ...],
) -> bool:
    if change == "create":
        return current is not None
    if current is None or prior != current.revision:
        return True
    return change == "reuse" and (current.value != value or current.supports != supports)


def status(
    released: frozenset[str],
    targets: frozenset[str],
    changed: bool,
    conflicts: frozenset[str],
) -> str:
    if conflicts and not released:
        return "conflicted"
    if released and released != targets:
        return "partial"
    return "committed" if changed else "no_change"


def groups(
    targets: tuple[str, ...], explicit: tuple[tuple[str, ...], ...] | None = None
) -> tuple[tuple[str, ...], ...]:
    result = tuple((target,) for target in targets) if explicit is None else explicit
    flat = tuple(target for group in result for target in group)
    if any(not group for group in result) or len(flat) != len(set(flat)) or set(flat) != set(targets):
        raise ValueError("complete partition required")
    return result


def protected_allowed(
    required: frozenset[str],
    scalar: bool,
    execution: str,
    qualification: str,
    receipt: tuple[str, str, frozenset[str]] | None,
    target: str,
) -> bool:
    if not scalar or (required and receipt is None):
        return False
    return receipt is None or (receipt[:2] == (execution, qualification) and target in receipt[2])
