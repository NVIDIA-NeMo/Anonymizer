# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TypeAlias

PURPOSES = (
    "initial",
    "initial_binding",
    "adaptive_retrieval",
    "retry",
    "correction",
    "repair",
    "failover",
)
BOUNDARIES = (
    "preparation",
    "context_admission",
    "execution_admission",
    "initial_binding",
    "execution",
    "activation",
    "operation",
    "adaptive_retrieval",
    "physical_request",
    "decision_wait",
    "request_reconciliation",
    "cleanup",
    "qualification",
    "continuity_commit",
    "projection",
    "release",
)
OUTCOMES = (
    "success",
    "failure",
    "cancelled",
    "lost",
    "blocked",
    "inconsistent",
    "partial",
    "unknown",
)
REASONS = (
    "invalid_input",
    "limit_exceeded",
    "unavailable",
    "provider_failure",
    "malformed_response",
    "cancelled",
    "lost",
    "deadline",
    "budget_stopped",
    "request_limit_stopped",
    "missing_evidence",
    "invalid_evidence",
    "continuity_conflict",
    "cleanup_failed",
    "cleanup_unknown",
    "observer_failed",
    "internal_failure",
)
JSON: TypeAlias = str | int | bool | None | list["JSON"] | dict[str, "JSON"]


@dataclass(frozen=True)
class Estimate:
    knowledge: str
    value: int | None


@dataclass(frozen=True)
class Dispatch:
    identity: int
    purpose: str
    usage: tuple[int, int] | None


@dataclass(frozen=True)
class Scope:
    # None means absent owning receipt. An empty receipt proves zero for that phase.
    dispatches: tuple[Dispatch, ...] | None
    possible_purposes: tuple[str, ...] = PURPOSES


@dataclass(frozen=True)
class Accounting:
    purposes: tuple[Estimate, ...]
    total: Estimate
    input_units: Estimate
    output_units: Estimate
    unknown_requests: Estimate


def account(scopes: tuple[Scope, ...]) -> Accounting:
    unknown = Estimate("unknown", None)
    missing = frozenset(p for scope in scopes if scope.dispatches is None for p in scope.possible_purposes)
    dispatches = tuple(d for s in scopes for d in (s.dispatches or ()))
    if len({d.identity for d in dispatches}) != len(dispatches):
        raise ValueError("duplicate physical ownership")
    if any(d.purpose not in PURPOSES for d in dispatches):
        raise ValueError("purpose")
    for dispatch in dispatches:
        if dispatch.usage is not None:
            for value in dispatch.usage:
                count(value)
    totals = tuple(
        unknown if purpose in missing else Estimate("exact", sum(d.purpose == purpose for d in dispatches))
        for purpose in PURPOSES
    )
    unknown_count = sum(d.usage is None for d in dispatches)
    units = tuple(
        Estimate("exact", sum(d.usage[index] for d in dispatches if d.usage is not None))
        if unknown_count == 0 and not missing
        else unknown
        for index in (0, 1)
    )
    return Accounting(
        totals,
        unknown if missing else Estimate("exact", len(dispatches)),
        units[0],
        units[1],
        unknown if missing else Estimate("exact", unknown_count),
    )


def explain(
    upper: int,
    unconditional: int | None,
    conditional: int | None,
    controls: tuple[tuple[bool, bool], ...] | None,
) -> tuple[Estimate, Estimate, Estimate, str]:
    count(upper)
    estimates = tuple(
        Estimate("unknown", None) if n is None else Estimate("exact", count(n)) for n in (unconditional, conditional)
    )
    support = "unknown" if controls is None else ("supported" if all(a and b for a, b in controls) else "unsupported")
    return estimates[0], estimates[1], Estimate("known_upper", upper), support


@dataclass(frozen=True)
class Pair:
    ordinal: int
    parent: int | None
    boundary: str
    terminal_ordinal: int
    terminal_parent: int | None
    terminal_boundary: str
    outcome: str
    reasons: tuple[str, ...]
    start_ns: int
    end_ns: int
    start_present: bool = True
    terminal_present: bool = True


@dataclass(frozen=True)
class Limits:
    pairs: int
    events: int
    nodes: int
    depth: int
    encoded_bytes: int
    duration: int = 100
    count: int = 100


@dataclass(frozen=True)
class Need:
    pairs: int
    events: int
    nodes: int
    depth: int
    encoded_bytes: int
    envelope: Limits | None = None


@dataclass(frozen=True)
class Phase:
    need: Need
    pairs: tuple[Pair, ...]
    owner: int = 0


@dataclass(frozen=True)
class PublicPair:
    ordinal: int
    parent: int | None
    boundary: str
    outcome: str
    reasons: tuple[str, ...]
    duration_ns: int


@dataclass(frozen=True)
class Receipt:
    pairs: tuple[PublicPair, ...]
    delivery: str
    delivered_pairs: int
    schema_version: int = 1


@dataclass(frozen=True)
class Failure:
    code: str
    schema_version: int = 1


Close: TypeAlias = Receipt | Failure


def count(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("count")
    return value


def pair_error(pair: Pair, earlier: frozenset[int], limits: Limits) -> str | None:
    if (
        type(pair.terminal_ordinal) is not int
        or pair.terminal_ordinal < 0
        or pair.terminal_ordinal > limits.count
        or (
            pair.terminal_parent is not None
            and (
                type(pair.terminal_parent) is not int or pair.terminal_parent < 0 or pair.terminal_parent > limits.count
            )
        )
        or type(pair.ordinal) is not int
        or pair.ordinal < 0
        or pair.ordinal > limits.count
        or (pair.parent is not None and (type(pair.parent) is not int or pair.parent < 0 or pair.parent > limits.count))
        or type(pair.start_ns) is not int
        or type(pair.end_ns) is not int
        or not pair.start_present
        or not pair.terminal_present
        or pair.ordinal in earlier
        or pair.ordinal != len(earlier)
        or (pair.parent is not None and (pair.parent not in earlier or pair.parent >= pair.ordinal))
        or (pair.ordinal, pair.parent, pair.boundary)
        != (pair.terminal_ordinal, pair.terminal_parent, pair.terminal_boundary)
        or pair.boundary not in BOUNDARIES
        or pair.outcome not in OUTCOMES
        or len(pair.reasons) != len(set(pair.reasons))
        or not set(pair.reasons) <= set(REASONS)
    ):
        return "producer_contract"
    if pair.end_ns < pair.start_ns:
        return "clock_contract"
    if pair.end_ns - pair.start_ns > limits.duration:
        return "capacity_contract"
    return None


def pair_tree(pair: PublicPair) -> JSON:
    correlation: JSON = {"ordinal": pair.ordinal, "parent_ordinal": pair.parent}
    return {
        "start": {"correlation": correlation, "boundary": pair.boundary},
        "terminal": {
            "correlation": correlation,
            "boundary": pair.boundary,
            "outcome": pair.outcome,
            "reasons": [r for r in REASONS if r in pair.reasons],
            "duration_ns": pair.duration_ns,
            "measurement": None,
        },
    }


def tree(value: Close) -> JSON:
    if type(value) is Failure:
        if value.code not in {
            "capacity_contract",
            "clock_contract",
            "producer_contract",
        }:
            raise ValueError("failure code")
        return {"schema_version": value.schema_version, "code": value.code}
    if type(value) is not Receipt:
        raise TypeError("closed value")
    if value.delivery not in {"not_configured", "complete", "failed"}:
        raise ValueError("delivery")
    return {
        "schema_version": value.schema_version,
        "pairs": [pair_tree(p) for p in value.pairs],
        "delivery": value.delivery,
        "delivered_pairs": value.delivered_pairs,
    }


def shape(value: JSON, depth: int = 1) -> tuple[int, int]:
    children = list(value.values()) if isinstance(value, dict) else value if isinstance(value, list) else []
    nested = [shape(child, depth + 1) for child in children]
    return 1 + sum(n for n, _ in nested), max([depth] + [d for _, d in nested])


def bytes_for(value: JSON) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def encode(value: Close, limits: Limits) -> bytes:
    if type(value) not in {Receipt, Failure}:
        raise TypeError("closed value")
    if type(value.schema_version) is not int or value.schema_version != 1:
        raise ValueError("version")
    if isinstance(value, Receipt):
        if len(value.pairs) > limits.pairs or len(value.pairs) * 2 > limits.events:
            raise ValueError("capacity")
        if count(value.delivered_pairs) > min(len(value.pairs), limits.count):
            raise ValueError("delivery count")
        if value.delivery == "not_configured" and value.delivered_pairs != 0:
            raise ValueError("delivery contradiction")
        if value.delivery == "complete" and value.delivered_pairs != len(value.pairs):
            raise ValueError("delivery contradiction")
        for index, pair in enumerate(value.pairs):
            fact = Pair(
                pair.ordinal,
                pair.parent,
                pair.boundary,
                pair.ordinal,
                pair.parent,
                pair.boundary,
                pair.outcome,
                pair.reasons,
                0,
                pair.duration_ns,
            )
            if pair_error(fact, frozenset(range(index)), limits) is not None:
                raise ValueError("pair contract")
    encoded_tree = tree(value)
    nodes, depth = shape(encoded_tree)
    if nodes > limits.nodes or depth > limits.depth:
        raise ValueError("tree capacity")
    raw = bytes_for(encoded_tree)
    if len(raw) > limits.encoded_bytes:
        raise ValueError("byte capacity")
    return raw


def fits(used: Need, addition: Need, limits: Limits) -> bool:
    return (
        used.pairs + addition.pairs <= limits.pairs
        and used.pairs + addition.pairs <= limits.count
        and used.events + addition.events <= limits.events
        and used.nodes + addition.nodes <= limits.nodes
        and max(used.depth, addition.depth) <= limits.depth
        and used.encoded_bytes + addition.encoded_bytes <= limits.encoded_bytes
    )


def requirements(pairs: int, limits: Limits, logical_depth: int = 1) -> Need:
    """Independent closed-schema worst case, including optional measurement."""
    max_count, max_duration = limits.count, limits.duration
    measurement: JSON = {
        "role": "assessment",
        "route": "external",
        "input_count": max_count,
        "output_count": max_count,
        "artifact_count": max_count,
        "artifact_bytes": max_count,
        "request_count": max_count,
    }
    correlation: JSON = {
        "ordinal": max_count,
        "parent_ordinal": max_count if len(str(max_count)) >= 4 else None,
    }
    pair: JSON = {
        "start": {"correlation": correlation, "boundary": "request_reconciliation"},
        "terminal": {
            "correlation": correlation,
            "boundary": "request_reconciliation",
            "outcome": "inconsistent",
            "reasons": list(REASONS),
            "duration_ns": max_duration,
            "measurement": measurement,
        },
    }
    envelope: JSON = {
        "schema_version": 1,
        "pairs": [pair] * pairs,
        "delivery": "not_configured",
        "delivered_pairs": max_count,
    }
    nodes, depth = shape(envelope)
    return Need(
        pairs,
        2 * pairs,
        nodes,
        max(depth, logical_depth),
        len(bytes_for(envelope)),
        limits,
    )


def forecast(
    *,
    fixed: int,
    slots: int,
    adaptive: int,
    requests: int,
    waits: int,
    cleanup: int,
    limits: Limits,
    logical_depth: int = 1,
) -> Need:
    """Compiled neutral ceilings, never actual observed event counts."""
    pairs = fixed + 2 * slots + adaptive + requests + waits + cleanup
    return requirements(pairs, limits, logical_depth)


def observe(
    phases: tuple[Phase, ...],
    limits: Limits,
    sink_failure_at: int | None = None,
    *,
    sink_configured: bool = False,
) -> tuple[Close, tuple[PublicPair, ...]]:
    """Reduce finite completed phase facts; no callbacks or semantic executor."""
    if not fits(Need(0, 0, 0, 0, 0), requirements(1, limits), limits):
        return Failure("capacity_contract"), ()
    logical_depths: dict[int, int] = {}
    retained: list[PublicPair] = []
    delivered: list[PublicPair] = []
    used = Need(0, 0, 0, 0, 0)
    delivery = "complete" if sink_configured else "not_configured"
    for phase in phases:
        if phase.need.envelope != limits:
            return Failure("capacity_contract"), tuple(delivered)
        if phase.need.events != 2 * phase.need.pairs or not fits(used, phase.need, limits):
            return Failure("capacity_contract"), tuple(delivered)
        used = Need(
            used.pairs + phase.need.pairs,
            used.events + phase.need.events,
            used.nodes + phase.need.nodes,
            max(used.depth, phase.need.depth),
            used.encoded_bytes + phase.need.encoded_bytes,
        )
        if phase.owner != 0:
            return Failure("producer_contract"), tuple(delivered)
        if len(phase.pairs) > phase.need.pairs:
            return Failure("capacity_contract"), tuple(delivered)
        for pair in phase.pairs:
            error = pair_error(pair, frozenset(p.ordinal for p in retained), limits)
            if error is not None:
                return Failure(error), tuple(delivered)
            logical_depth = 1 if pair.parent is None else logical_depths[pair.parent] + 1
            if logical_depth > min(phase.need.depth, limits.depth):
                return Failure("capacity_contract"), tuple(delivered)
            logical_depths[pair.ordinal] = logical_depth
            public = PublicPair(
                pair.ordinal,
                pair.parent,
                pair.boundary,
                pair.outcome,
                pair.reasons,
                pair.end_ns - pair.start_ns,
            )
            retained.append(public)
            if sink_configured and delivery != "failed":
                if sink_failure_at == len(delivered):
                    delivery = "failed"
                else:
                    delivered.append(public)
    return Receipt(tuple(retained), delivery, len(delivered)), tuple(delivered)


def owner_chain(
    expected: tuple[object, ...],
    supplied: tuple[object, ...],
    cached_close: Close,
    supplied_close: Close,
) -> bool:
    return (
        cached_close is supplied_close
        and len(expected) == len(supplied)
        and all(a is b and a is not None for a, b in zip(expected, supplied, strict=True))
    )


def qualification(statuses: tuple[str, ...], verified: int, withheld: int) -> tuple[int, ...]:
    if not set(statuses) <= {"met", "unmet", "unknown", "not_assessed"}:
        raise ValueError("qualification")
    return (
        sum(s != "not_assessed" for s in statuses),
        statuses.count("met"),
        statuses.count("unmet"),
        statuses.count("unknown"),
        count(verified),
        count(withheld),
    )
