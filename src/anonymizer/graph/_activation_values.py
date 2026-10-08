# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Activation state and event values."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Never, SupportsIndex, TypeAlias

from anonymizer.graph._activation_topology import _context, _node, _reject, _scope_for
from anonymizer.graph._values import ActivationKey, InvocationId, ValidationCode
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    NodeId,
    OutcomeClass,
    SubgraphNode,
)

ActivationStatus: TypeAlias = Literal[
    "unstarted", "ready", "running", "success", "failure", "cancelled", "lost", "blocked", "inconsistent"
]


ExpansionStatus: TypeAlias = Literal["pending", "closed", "failed", "overflow"]


_TERMINAL = frozenset(("success", "failure", "cancelled", "lost", "blocked", "inconsistent"))


_CATEGORIES = _TERMINAL


_PICKLE_ERROR = "activation value serialization is not supported"


class _PrivateValue:
    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def __copy__(self) -> _PrivateValue:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _PrivateValue:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Never:
        del protocol
        raise TypeError(_PICKLE_ERROR)


def _integer(value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(ValidationCode.INVALID_TYPE)
    if value < (1 if positive else 0):
        _reject(ValidationCode.INVALID_VALUE)
    return value


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActivationSeed(_PrivateValue):
    template: NodeId
    activation: ActivationKey

    def __post_init__(self) -> None:
        if not isinstance(self.template, NodeId) or not isinstance(self.activation, ActivationKey):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActivationEntry(_PrivateValue):
    template: NodeId
    activation: ActivationKey
    status: ActivationStatus
    outcome: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.template, NodeId) or not isinstance(self.activation, ActivationKey):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.status, str) or (self.outcome is not None and not isinstance(self.outcome, str)):
            _reject(ValidationCode.INVALID_TYPE)
        if self.status not in {"unstarted", "ready", "running", *_TERMINAL} or self.outcome == "":
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExpansionEntry(_PrivateValue):
    parent: ActivationKey
    members: frozenset[ActivationKey]
    status: ExpansionStatus

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent, ActivationKey)
            or not isinstance(self.members, frozenset)
            or any(not isinstance(member, ActivationKey) for member in self.members)
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.status, str):
            _reject(ValidationCode.INVALID_TYPE)
        if self.status not in {"pending", "closed", "failed", "overflow"}:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActivationLimits(_PrivateValue):
    max_events: int
    max_entries: int
    max_parent_depth: int

    def __post_init__(self) -> None:
        _integer(self.max_events)
        _integer(self.max_entries)
        _integer(self.max_parent_depth)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActivationState(_PrivateValue):
    workflow: AdmittedActivationWorkflow
    invocation: InvocationId
    entries: frozenset[ActivationEntry]
    expansions: frozenset[ExpansionEntry]
    reservations: frozenset[ActivationSeed]
    events_applied: int
    limits: ActivationLimits

    def __post_init__(self) -> None:
        if not isinstance(self.workflow, AdmittedActivationWorkflow) or not isinstance(self.invocation, InvocationId):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.entries, frozenset) or any(
            not isinstance(item, ActivationEntry) for item in self.entries
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.expansions, frozenset) or any(
            not isinstance(item, ExpansionEntry) for item in self.expansions
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.reservations, frozenset) or any(
            not isinstance(item, ActivationSeed) for item in self.reservations
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.limits, ActivationLimits):
            _reject(ValidationCode.INVALID_TYPE)
        _integer(self.events_applied)
        entry_keys = [item.activation for item in self.entries]
        expansion_keys = [item.parent for item in self.expansions]
        reservation_keys = [item.activation for item in self.reservations]
        if (
            len(entry_keys) != len(set(entry_keys))
            or len(expansion_keys) != len(set(expansion_keys))
            or len(reservation_keys) != len(set(reservation_keys))
        ):
            _reject(ValidationCode.DUPLICATE)
        if any(key.invocation != self.invocation for key in (*entry_keys, *expansion_keys, *reservation_keys)):
            _reject(ValidationCode.FOREIGN_OWNER)
        reservations = _seed_map(self.reservations)
        if any(
            item.activation not in reservations or reservations[item.activation].template != item.template
            for item in self.entries
        ) or any(item.parent not in reservations for item in self.expansions):
            _reject(ValidationCode.MISSING)
        if (
            self.events_applied > self.limits.max_events
            or len(self.entries) > self.limits.max_entries
            or len(self.entries) > self.workflow.activation_upper_bound
        ):
            _reject(ValidationCode.LIMIT_EXCEEDED)

    @property
    def complete(self) -> bool:
        """Whether every selected obligation is terminal and every expansion is settled."""
        entries = _entry_map(self.entries)
        if not entries:
            return False
        reservations = _seed_map(self.reservations)
        required: set[ActivationKey] = set()
        for key, seed in reservations.items():
            scope = _scope_for(self.workflow, seed.template)
            choice_member = any(
                seed.template in branch.members for choice in scope.workflow.choices for branch in choice.branches
            )
            dynamic_member = any(seed.template == item.member for item in (*scope.maps, *scope.loops))
            if key.parent is None and not choice_member and not dynamic_member:
                required.add(key)
            if (
                key.parent is not None
                and key.parent in entries
                and isinstance(_node(self.workflow, entries[key.parent].template), SubgraphNode)
            ):
                if not choice_member and not dynamic_member:
                    required.add(key)
        for entry in entries.values():
            if entry.outcome is None:
                continue
            scope = _scope_for(self.workflow, entry.template)
            context = _context(reservations[entry.activation], scope)
            for choice in scope.workflow.choices:
                if choice.selector == entry.template:
                    required.update(
                        seed.activation
                        for seed in reservations.values()
                        for branch in choice.branches
                        if entry.outcome in branch.outcomes and seed.template in branch.members
                        if _context(seed, scope) == context
                    )
        for expansion in self.expansions:
            required.update(expansion.members)
        return (
            required <= entries.keys()
            and all(entries[key].status in _TERMINAL for key in required)
            and all(
                expansion.status != "pending"
                and all(member in entries and entries[member].status in _TERMINAL for member in expansion.members)
                for expansion in self.expansions
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Select(_PrivateValue):
    seeds: frozenset[ActivationSeed]

    def __post_init__(self) -> None:
        if not isinstance(self.seeds, frozenset) or any(not isinstance(seed, ActivationSeed) for seed in self.seeds):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Start(_PrivateValue):
    activation: ActivationKey

    def __post_init__(self) -> None:
        if not isinstance(self.activation, ActivationKey):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObserveTerminal(_PrivateValue):
    activation: ActivationKey
    outcome: str | None
    category: OutcomeClass

    def __post_init__(self) -> None:
        if not isinstance(self.activation, ActivationKey) or (
            self.outcome is not None and not isinstance(self.outcome, str)
        ):
            _reject(ValidationCode.INVALID_TYPE)
        if not isinstance(self.category, str):
            _reject(ValidationCode.INVALID_TYPE)
        if (
            self.category not in _CATEGORIES
            or self.outcome == ""
            or (self.outcome is None and self.category == "success")
        ):
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CloseUnstarted(_PrivateValue):
    activation: ActivationKey
    category: Literal["blocked", "inconsistent"]

    def __post_init__(self) -> None:
        if not isinstance(self.activation, ActivationKey) or not isinstance(self.category, str):
            _reject(ValidationCode.INVALID_TYPE)
        if self.category not in {"blocked", "inconsistent"}:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObserveMembership(_PrivateValue):
    parent: ActivationKey
    members: frozenset[ActivationKey]
    closed: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.parent, ActivationKey)
            or not isinstance(self.members, frozenset)
            or any(not isinstance(item, ActivationKey) for item in self.members)
            or not isinstance(self.closed, bool)
        ):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObserveOverflow(_PrivateValue):
    parent: ActivationKey
    observed_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.parent, ActivationKey):
            _reject(ValidationCode.INVALID_TYPE)
        _integer(self.observed_count)


ActivationEvent: TypeAlias = Select | Start | ObserveTerminal | CloseUnstarted | ObserveMembership | ObserveOverflow


def _entry_map(items: frozenset[ActivationEntry]) -> dict[ActivationKey, ActivationEntry]:
    return {item.activation: item for item in items}


def _expansion_map(items: frozenset[ExpansionEntry]) -> dict[ActivationKey, ExpansionEntry]:
    return {item.parent: item for item in items}


def _seed_map(items: frozenset[ActivationSeed]) -> dict[ActivationKey, ActivationSeed]:
    return {item.activation: item for item in items}
