# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite declarative history predicates, independent of production code."""

from __future__ import annotations

import itertools
from dataclasses import dataclass


def has_total_order(nodes: tuple[int, ...], edges: tuple[tuple[int, int], ...]) -> bool:
    """A tiny graph is acyclic exactly when some complete ordering respects it."""
    return all(a in nodes and b in nodes for a, b in edges) and any(
        all(order.index(a) < order.index(b) for a, b in edges) for order in itertools.permutations(nodes)
    )


def matches_window(source: str, derived: str, start: int, end: int) -> bool:
    """Offsets count Unicode codepoints, and empty read windows are legal."""
    return 0 <= start <= end <= len(source) and source[start:end] == derived


def has_disjoint_outputs(windows: tuple[tuple[int, int, int], ...]) -> bool:
    """Root identity, rather than equal content, determines overlapping writes."""
    return all(start <= end for _, start, end in windows) and all(
        root_a != root_b or max(start_a, start_b) >= min(end_a, end_b)
        for (root_a, start_a, end_a), (root_b, start_b, end_b) in itertools.combinations(windows, 2)
    )


def is_partition(targets: frozenset[int], groups: tuple[frozenset[int], ...]) -> bool:
    """Explicit groups are independently valid; omitted members are singleton."""
    return all(group and group <= targets for group in groups) and all(
        a == b or not a & b for a, b in itertools.combinations(groups, 2)
    )


@dataclass(frozen=True)
class OccurrenceEvent:
    """Exact occurrence is (template, parent, key/version or iteration)."""

    occurrence: str
    kind: str
    category: str = ""


def is_legal_activation(
    selected: frozenset[str],
    edges: tuple[tuple[str, str], ...],
    events: tuple[OccurrenceEvent, ...],
    complete: bool,
) -> bool:
    """Check selected ordinary occurrence readiness and immutable terminals."""
    if any(a not in selected or b not in selected for a, b in edges):
        return False
    for index, event in enumerate(events):
        if event.occurrence not in selected:
            return False
        prior = events[:index]
        own = tuple(item for item in prior if item.occurrence == event.occurrence)
        if event.kind == "start":
            if own or any(
                not any(
                    item.occurrence == a and item.kind == "terminal" and item.category == "success" for item in prior
                )
                for a, b in edges
                if b == event.occurrence
            ):
                return False
        elif event.kind == "terminal":
            if len(own) != 1 or own[0].kind != "start" or event.category not in ("success", "failure", "lost"):
                return False
        elif event.kind == "close_unstarted":
            if own or event.category not in ("blocked", "inconsistent"):
                return False
        else:
            return False
    return not complete or all(
        any(event.occurrence == item and event.kind in ("terminal", "close_unstarted") for event in events)
        for item in selected
    )


def map_join(expansion: str, closed: bool, members: frozenset[str], terminals: tuple[tuple[str, str], ...]) -> str:
    """A join requires exact sealed membership and every accepted terminal."""
    if expansion == "failed":
        return "blocked"
    if expansion == "overflow":
        return "inconsistent"
    if expansion != "success" or not closed:
        return "pending"
    if len({member for member, _ in terminals}) != len(terminals) or any(
        member not in members for member, _ in terminals
    ):
        return "invalid"
    if {member for member, _ in terminals} != members:
        return "pending"
    return "success" if all(category == "success" for _, category in terminals) else "blocked"


def final_loop(outcomes: tuple[str, ...], bound: int, entered: bool) -> tuple[str, int | None]:
    """Select only the actual final exit; bypass differs from failed expansion."""
    if not entered:
        return ("success", None) if not outcomes else ("invalid", None)
    if len(outcomes) > bound:
        return "invalid", None
    if not outcomes:
        return ("inconsistent", None) if bound == 0 else ("pending", None)
    if any(item != "continue" for item in outcomes[:-1]):
        return "invalid", None
    last = outcomes[-1]
    if last == "exit":
        return "success", len(outcomes) - 1
    if last == "failed":
        return "blocked", None
    if last == "continue":
        return ("inconsistent", None) if len(outcomes) == bound else ("pending", None)
    return "invalid", None


@dataclass(frozen=True)
class RequestEvent:
    """Neutral physical request event; None usage means unknown, never zero."""

    request: int
    kind: str
    associations: tuple[int, ...] = ()
    value: str = ""
    usage: int | None = None
    stopped: bool = False


def is_legal_requests(events: tuple[RequestEvent, ...], budget: int) -> bool:
    """Check causal reservation and every prefix's live physical occupancy."""
    for index, event in enumerate(events):
        prior = events[:index]
        own = tuple(item for item in prior if item.request == event.request)
        reserved = {item.request for item in prior if item.kind == "reserve"}
        dispatched = {item.request for item in prior if item.kind == "dispatch"}
        cancelled = {item.request for item in prior if item.kind == "cancel_reserved"}
        live = reserved - dispatched - cancelled
        if event.kind == "reserve":
            if own or not event.associations or len(set(event.associations)) != len(event.associations):
                return False
            if len(dispatched) + len(live) >= budget:
                return False
        elif event.kind == "dispatch":
            if event.request not in live:
                return False
        elif event.kind == "cancel_reserved":
            if event.request not in live:
                return False
        elif event.kind in ("terminal", "settlement", "result"):
            if event.request not in dispatched:
                return False
            if event.kind == "settlement" and event.usage is not None and event.usage < 0:
                return False
        else:
            return False
    return True


def terminal_projection(events: tuple[RequestEvent, ...], request: int) -> tuple[str | None, int | None, bool]:
    """First semantic terminal and first independent settlement survive conflicts."""
    own = tuple(item for item in events if item.request == request)
    association_sets = [item.associations for item in own if item.kind == "reserve"]
    expected = set(association_sets[0]) if association_sets else set()
    terminal_events = [item for item in own if item.kind in ("terminal", "result", "cancel_reserved")]
    first = terminal_events[0] if terminal_events else None
    terminal = None if first is None else first.value
    if first is not None and first.kind == "cancel_reserved":
        terminal = "cancelled"
    if first is not None and first.kind == "result":
        terminal = (
            "success"
            if set(first.associations) == expected and len(first.associations) == len(expected)
            else "inconsistent"
        )
    settlements = [item for item in own if item.kind == "settlement"]
    return terminal, settlements[0].usage if settlements else None, settlements[0].stopped if settlements else False


def may_reissue(
    owner: str, replay: str, failure: str, purpose: str, attempts: int, bound: int, next_alternate: bool
) -> bool:
    """Eligibility excludes budget/reservation, which every new request still needs."""
    replay_ok = replay == "idempotent" or (replay == "before_acceptance" and failure == "rejected_before_acceptance")
    if owner != "executor" or attempts >= bound or not replay_ok:
        return False
    if purpose == "retry":
        return failure in ("retryable", "transport_unknown", "rejected_before_acceptance")
    if purpose == "correction":
        return failure == "malformed_response"
    return purpose == "failover" and next_alternate and failure in ("permanent", "implementation_exception")


def may_close(owned: bool, local_in_flight: bool, remote_outstanding: bool, independent_detachment: bool) -> bool:
    """Caller-owned and locally live resources never close."""
    return owned and not local_in_flight and (not remote_outstanding or independent_detachment)
