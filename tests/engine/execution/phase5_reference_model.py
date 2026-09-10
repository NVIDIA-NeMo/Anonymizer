# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent bounded oracle for Phase 5 context framing and release.

This module deliberately imports no Anonymizer production code, pandas,
DataDesigner, or Phase 4 ledger implementation.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from enum import Enum
from hashlib import sha256


class ReferenceAdmission(str, Enum):
    ADMITTED = "admitted"
    STRUCTURAL = "structural"
    LIMIT = "limit"
    CONTRACT = "contract"
    PREFLIGHT_CAPABILITY = "preflight_capability"


class ReferenceInvocation(str, Enum):
    NOT_OPENED = "not_opened"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"
    INCONSISTENT = "inconsistent"


class ReferenceEventKind(str, Enum):
    BINDING_CONSTRUCTION = "binding_construction"
    BINDING_COMMITMENT = "binding_commitment"
    BINDING_CONSUMPTION = "binding_consumption"
    BINDING_CORRUPTION = "binding_corruption"
    TASK_DISPATCH = "task_dispatch"
    TASK_TERMINAL = "task_terminal"
    TASK_CORRUPTION = "task_corruption"
    CANCELLATION = "cancellation"
    TRUSTED_STOP = "trusted_stop"
    TRANSPORT_LOSS = "transport_loss"
    CLEANUP_PRIMARY = "cleanup_primary"
    CLEANUP_COMPETING = "cleanup_competing"
    PUBLICATION = "publication"
    TEARDOWN = "teardown"


@dataclass(frozen=True)
class ReferenceEvent:
    kind: ReferenceEventKind
    subject: str = "invocation"
    outcome: str = "accepted"


@dataclass(frozen=True)
class ReferenceLimits:
    datum_bytes: int
    id_bytes: int
    members: int
    context_bytes: int
    references: int
    expanded_bytes: int


@dataclass(frozen=True)
class ReferenceScope:
    target: str
    context: tuple[str, ...]


@dataclass(frozen=True)
class ReferenceCase:
    case_id: str
    targets: tuple[tuple[str, str], ...]
    context_only: tuple[tuple[str, str], ...]
    scopes: tuple[ReferenceScope, ...]
    limits: ReferenceLimits
    events: tuple[ReferenceEvent, ...]
    preflight_capability: str = "compatible"
    runtime_capability: str = "compatible"
    relation: str = "bounded_context"
    profile: str = "target-context-v1"
    schema: str = "context-workframe-v1"
    ordering: str = "declared"
    allow_target_as_context: bool = True
    dependencies: tuple[tuple[str, str], ...] = ()
    groups: tuple[tuple[str, ...], ...] = ()
    order_class: str = "declared"
    cycle_class: str = "none"
    payload_class: str = "multibyte"
    limit_class: str = "all:exact"
    schedule_class: str = "success"


@dataclass(frozen=True)
class ReferenceResult:
    admission: ReferenceAdmission
    invocation: ReferenceInvocation
    reason: str
    binding_count: int
    task_outcomes: tuple[tuple[str, str, str], ...]
    private_succeeded: tuple[str, ...]
    private_inconsistent: tuple[str, ...]
    released: tuple[str, ...]
    cleanup: str
    published: bool
    teardown: str
    event_count: int
    event_max: int


def evaluate(case: ReferenceCase) -> ReferenceResult:
    """Reduce an explicit event schedule without production inputs."""
    targets = tuple(identifier for identifier, _text in case.targets)
    bindings = _binding_keys(case.scopes)
    event_max = 4 * len(bindings) + 3 * len(targets) + 7
    _validate_schedule(case.events, bindings, targets, event_max)
    admission = _admit(case)
    if admission is not ReferenceAdmission.ADMITTED:
        return _not_opened(admission, len(bindings), len(case.events), event_max, "admission_rejected")
    if case.runtime_capability != "compatible":
        return _not_opened(admission, len(bindings), len(case.events), event_max, "runtime_capability")
    if _is_pre_dispatch_cancellation(case.events):
        task_outcomes = tuple((target, "cancelled", "cancellation") for target in targets)
        return ReferenceResult(
            admission,
            ReferenceInvocation.CANCELLED,
            "cancellation",
            len(bindings),
            task_outcomes,
            (),
            (),
            (),
            "not_entered",
            False,
            "not_entered",
            len(case.events),
            event_max,
        )
    binding_states, global_binding_fault = _binding_outcomes(case.events, bindings)
    task_outcomes = _task_outcomes(case.events, targets, binding_states)
    if global_binding_fault:
        task_outcomes = tuple((target, "inconsistent", "contradictory") for target in targets)
    task_corruption = next(
        (event.outcome for event in case.events if event.kind is ReferenceEventKind.TASK_CORRUPTION),
        None,
    )
    if task_corruption is not None:
        task_outcomes = tuple((target, "inconsistent", task_corruption) for target in targets)
    cleanup = _cleanup_outcome(case.events)
    invocation, reason = _invocation_outcome(case.events, task_outcomes, global_binding_fault, cleanup)
    published = _event_outcome(case.events, ReferenceEventKind.PUBLICATION) == "accepted"
    released = _released(case, task_outcomes) if invocation is ReferenceInvocation.COMPLETED and published else ()
    private_succeeded = tuple(target for target, state, _reason in task_outcomes if state == "succeeded")
    private_inconsistent = tuple(target for target, state, _reason in task_outcomes if state == "inconsistent")
    return ReferenceResult(
        admission,
        invocation,
        reason,
        len(bindings),
        task_outcomes,
        private_succeeded,
        private_inconsistent,
        released,
        cleanup,
        published,
        _event_outcome(case.events, ReferenceEventKind.TEARDOWN),
        len(case.events),
        event_max,
    )


def reference_cases() -> Iterator[ReferenceCase]:
    """Yield the frozen finite Phase 5 semantic-class envelope."""
    for target_count in range(1, 5):
        for context_count in range(4):
            targets = tuple((f"t{index}", f"target-{index}") for index in range(target_count))
            contexts = tuple((f"c{index}", "é" if index == 0 else f"context-{index}") for index in range(context_count))
            base_scopes = _base_scopes(targets, contexts)
            yield from _case_family(targets, contexts, base_scopes, "declared", "none")
            if target_count > 1:
                yield _make_case(
                    f"{target_count}t-{context_count}c-scope-declaration-reversed",
                    targets,
                    contexts,
                    tuple(reversed(base_scopes)),
                    "scope_declaration_reversed",
                    "none",
                )
            if context_count > 1:
                reversed_scopes = (
                    replace(base_scopes[0], context=tuple(reversed(base_scopes[0].context))),
                    *base_scopes[1:],
                )
                yield from _case_family(targets, contexts, reversed_scopes, "reversed", "none")
            if 2 <= target_count <= 3:
                for cycle_index, cyclic in enumerate(_cycle_scopes(targets, contexts)):
                    cycle_class = f"{target_count}-target-{cycle_index}"
                    yield from _case_family(targets, contexts, cyclic, "declared", cycle_class)
    yield from _invalid_semantic_cases()


def corpus_manifest() -> dict[str, object]:
    digest = sha256()
    counts: dict[str, dict[str, int]] = {
        "cardinality": {},
        "context_order": {},
        "context_cycle": {},
        "payload_class": {},
        "limit_class": {},
        "schedule_class": {},
        "t_b_emax": {},
    }
    graph_keys: set[tuple[object, ...]] = set()
    trace_count = 0
    actual_event_count = 0
    for case in reference_cases():
        result = evaluate(case)
        digest.update(canonical_case(case, result))
        trace_count += 1
        actual_event_count += result.event_count
        _increment(counts["cardinality"], f"{len(case.targets)}t-{len(case.context_only)}c")
        _increment(counts["context_order"], case.order_class)
        _increment(counts["context_cycle"], case.cycle_class)
        _increment(counts["payload_class"], case.payload_class)
        _increment(counts["limit_class"], case.limit_class)
        _increment(counts["schedule_class"], case.schedule_class)
        _increment(counts["t_b_emax"], f"T{len(case.targets)}-B{result.binding_count}-E{result.event_max}")
        graph_keys.add((case.targets, case.context_only, case.scopes, case.order_class, case.cycle_class))
    return {
        "generator_version": "phase5-reference-v2",
        "model_version": "target-context-event-workframe-v2",
        "graph_count": len(graph_keys),
        "canonical_trace_count": trace_count,
        "actual_event_count": actual_event_count,
        "counts": counts,
        "ceiling_fields": ["datum_bytes", "id_bytes", "members", "context_bytes", "references", "expanded_bytes"],
        "ceiling_domain": ["zero", "exact", "exact_plus_one"],
        "payload_domain": ["empty", "one_byte", "multibyte", "exact_limit", "one_over_limit"],
        "event_bound": "4B+3T+7",
        "sha256": digest.hexdigest(),
    }


def canonical_case(case: ReferenceCase, result: ReferenceResult) -> bytes:
    payload = {"case": asdict(case), "result": asdict(result)}
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def schedule_for(
    case: ReferenceCase,
    *,
    binding_evidence: str = "exact",
    cleanup: str = "verified",
    task_schedule: str = "success",
    teardown: str = "accepted",
) -> tuple[ReferenceEvent, ...]:
    """Build one bounded canonical representative from independent event classes."""
    bindings = _binding_keys(case.scopes)
    if task_schedule == "cancel_pre_dispatch":
        return (ReferenceEvent(ReferenceEventKind.CANCELLATION),)
    events = _binding_events(bindings, binding_evidence)
    events += _task_events(tuple(identifier for identifier, _text in case.targets), task_schedule)
    events += _cleanup_events(cleanup)
    events += (
        ReferenceEvent(ReferenceEventKind.PUBLICATION),
        ReferenceEvent(ReferenceEventKind.TEARDOWN, outcome=teardown),
    )
    return events


def _case_family(
    targets: tuple[tuple[str, str], ...],
    contexts: tuple[tuple[str, str], ...],
    scopes: tuple[ReferenceScope, ...],
    order_class: str,
    cycle_class: str,
) -> Iterator[ReferenceCase]:
    stem = f"{len(targets)}t-{len(contexts)}c-{order_class}-{cycle_class}"
    base = _make_case(stem, targets, contexts, scopes, order_class, cycle_class)
    yield base
    yield from _payload_cases(base)
    yield from _ceiling_cases(base)
    for state in ("missing", "incompatible", "weakened", "retention_enabled", "profile", "schema", "ordering"):
        yield _reschedule(replace(base, case_id=f"{stem}-preflight-{state}", preflight_capability=state))
        yield _reschedule(replace(base, case_id=f"{stem}-runtime-{state}", runtime_capability=state))
    if _binding_keys(scopes):
        for evidence in ("missing", "duplicate", "wrong_ordinal", "foreign", "cross_target", "contradictory"):
            yield _reschedule(replace(base, case_id=f"{stem}-binding-{evidence}"), binding_evidence=evidence)
    for cleanup in ("failed", "missing", "duplicate", "foreign", "incompatible", "contradictory"):
        yield _reschedule(replace(base, case_id=f"{stem}-cleanup-{cleanup}"), cleanup=cleanup)
    for task_schedule in ("cancel_pre_dispatch", "trusted_stop", "transport_loss", "terminal_then_cancel"):
        yield _reschedule(replace(base, case_id=f"{stem}-{task_schedule}"), task_schedule=task_schedule)
    yield _reschedule(replace(base, case_id=f"{stem}-teardown-failed"), teardown="failed")
    for terminal_fault in (
        "missing",
        "duplicate",
        "foreign",
        "stale",
        "cross_target",
        "plan_mismatch",
        "contradictory",
    ):
        yield _reschedule(
            replace(base, case_id=f"{stem}-terminal-{terminal_fault}"),
            task_schedule=f"terminal_{terminal_fault}",
        )
    if len(targets) > 1:
        dependency = ((targets[0][0], targets[1][0]),)
        overlap = "overlap" if cycle_class != "none" else "disjoint"
        yield _reschedule(
            replace(
                base,
                case_id=f"{stem}-dependency-{overlap}",
                dependencies=dependency,
                schedule_class=f"dependency_{overlap}",
            )
        )


def _make_case(
    case_id: str,
    targets: tuple[tuple[str, str], ...],
    contexts: tuple[tuple[str, str], ...],
    scopes: tuple[ReferenceScope, ...],
    order_class: str,
    cycle_class: str,
) -> ReferenceCase:
    groups = tuple((identifier,) for identifier, _text in targets)
    seed = ReferenceCase(
        case_id,
        targets,
        contexts,
        scopes,
        _exact_limits(targets, contexts, scopes),
        (),
        groups=groups,
        order_class=order_class,
        cycle_class=cycle_class,
    )
    return _reschedule(seed)


def _reschedule(
    case: ReferenceCase,
    *,
    binding_evidence: str = "exact",
    cleanup: str = "verified",
    task_schedule: str = "success",
    teardown: str = "accepted",
) -> ReferenceCase:
    schedule_class = ":".join((binding_evidence, cleanup, task_schedule, teardown))
    return replace(
        case,
        events=schedule_for(
            case, binding_evidence=binding_evidence, cleanup=cleanup, task_schedule=task_schedule, teardown=teardown
        ),
        schedule_class=schedule_class,
    )


def _payload_cases(base: ReferenceCase) -> Iterator[ReferenceCase]:
    if not base.context_only:
        return
    for payload_class, payload in (("empty", ""), ("one_byte", "x"), ("exact_limit", "x" * 8)):
        contexts = tuple((identifier, payload) for identifier, _text in base.context_only)
        scopes = base.scopes
        candidate = replace(
            base,
            case_id=f"{base.case_id}-payload-{payload_class}",
            context_only=contexts,
            limits=_exact_limits(base.targets, contexts, scopes),
            payload_class=payload_class,
        )
        yield _reschedule(candidate)
    contexts = tuple((identifier, "x" * 9) for identifier, _text in base.context_only)
    exact = _exact_limits(base.targets, contexts, base.scopes)
    one_over = replace(exact, context_bytes=max(0, exact.context_bytes - 1))
    candidate = replace(
        base,
        case_id=f"{base.case_id}-payload-one_over_limit",
        context_only=contexts,
        limits=one_over,
        payload_class="one_over_limit",
    )
    yield _reschedule(candidate)


def _ceiling_cases(base: ReferenceCase) -> Iterator[ReferenceCase]:
    exact = _exact_limits(base.targets, base.context_only, base.scopes)
    for field in asdict(exact):
        for ceiling, value in (
            ("zero", 0),
            ("exact", getattr(exact, field)),
            ("exact_plus_one", getattr(exact, field) + 1),
        ):
            limits = replace(exact, **{field: value})
            candidate = replace(
                base, case_id=f"{base.case_id}-{field}-{ceiling}", limits=limits, limit_class=f"{field}:{ceiling}"
            )
            yield _reschedule(candidate)


def _invalid_semantic_cases() -> Iterator[ReferenceCase]:
    targets = (("t0", "target"), ("t1", "peer"))
    contexts = (("c0", "context"),)
    valid = _make_case(
        "invalid-base", targets, contexts, (ReferenceScope("t0", ("c0",)), ReferenceScope("t1", ())), "declared", "none"
    )
    mutations = (
        ("missing_scope", {"scopes": valid.scopes[:1]}),
        ("duplicate_scope", {"scopes": (*valid.scopes, valid.scopes[0])}),
        ("unknown_target", {"scopes": (ReferenceScope("unknown", ("c0",)), valid.scopes[1])}),
        ("unknown_context", {"scopes": (ReferenceScope("t0", ("unknown",)), valid.scopes[1])}),
        ("self_context", {"scopes": (ReferenceScope("t0", ("t0", "c0")), valid.scopes[1])}),
        ("duplicate_member", {"scopes": (ReferenceScope("t0", ("c0", "c0")), valid.scopes[1])}),
        ("orphan", {"scopes": (ReferenceScope("t0", ()), valid.scopes[1])}),
        (
            "target_disabled",
            {"scopes": (ReferenceScope("t0", ("c0", "t1")), valid.scopes[1]), "allow_target_as_context": False},
        ),
        ("relation", {"relation": "wildcard"}),
        ("profile", {"profile": "future"}),
        ("schema", {"schema": "future"}),
        ("ordering", {"ordering": "implicit"}),
    )
    for name, changes in mutations:
        candidate = replace(valid, case_id=f"invalid-{name}", **changes)
        yield _reschedule(candidate)


def _base_scopes(
    targets: tuple[tuple[str, str], ...],
    contexts: tuple[tuple[str, str], ...],
) -> tuple[ReferenceScope, ...]:
    context_ids = tuple(identifier for identifier, _text in contexts)
    return tuple(
        ReferenceScope(target, context_ids if index == 0 else ()) for index, (target, _text) in enumerate(targets)
    )


def _cycle_scopes(
    targets: tuple[tuple[str, str], ...],
    contexts: tuple[tuple[str, str], ...],
) -> Iterator[tuple[ReferenceScope, ...]]:
    target_ids = tuple(identifier for identifier, _text in targets)
    context_ids = tuple(identifier for identifier, _text in contexts)
    directions = (1,) if len(targets) == 2 else (1, -1)
    for direction in directions:
        scopes: list[ReferenceScope] = []
        remaining = list(context_ids)
        for index, target in enumerate(target_ids):
            capacity = 2
            owned = tuple(remaining[:capacity])
            del remaining[:capacity]
            scopes.append(ReferenceScope(target, (*owned, target_ids[(index + direction) % len(target_ids)])))
        if remaining:
            scopes[0] = replace(scopes[0], context=(*scopes[0].context, *remaining))
        yield tuple(scopes)


def _binding_keys(scopes: tuple[ReferenceScope, ...]) -> tuple[tuple[str, str, int], ...]:
    return tuple((scope.target, member, ordinal) for scope in scopes for ordinal, member in enumerate(scope.context))


def _binding_events(
    bindings: tuple[tuple[str, str, int], ...],
    evidence: str,
) -> tuple[ReferenceEvent, ...]:
    events: list[ReferenceEvent] = []
    for index, (owner, _member, ordinal) in enumerate(bindings):
        subject = f"{owner}:{ordinal}"
        events.append(ReferenceEvent(ReferenceEventKind.BINDING_CONSTRUCTION, subject))
        events.append(ReferenceEvent(ReferenceEventKind.BINDING_COMMITMENT, subject))
        if not (index == 0 and evidence == "missing"):
            events.append(ReferenceEvent(ReferenceEventKind.BINDING_CONSUMPTION, subject))
    if bindings and evidence not in {"exact", "missing"}:
        owner, _member, ordinal = bindings[0]
        events.append(ReferenceEvent(ReferenceEventKind.BINDING_CORRUPTION, f"{owner}:{ordinal}", evidence))
    return tuple(events)


def _task_events(targets: tuple[str, ...], schedule: str) -> tuple[ReferenceEvent, ...]:
    events = [ReferenceEvent(ReferenceEventKind.TASK_DISPATCH, target) for target in targets]
    if schedule == "trusted_stop":
        events.extend(
            (ReferenceEvent(ReferenceEventKind.CANCELLATION), ReferenceEvent(ReferenceEventKind.TRUSTED_STOP))
        )
    elif schedule == "transport_loss":
        events.extend(
            (ReferenceEvent(ReferenceEventKind.CANCELLATION), ReferenceEvent(ReferenceEventKind.TRANSPORT_LOSS))
        )
    elif schedule == "terminal_missing":
        events.extend(ReferenceEvent(ReferenceEventKind.TASK_TERMINAL, target, "success") for target in targets[1:])
    else:
        events.extend(ReferenceEvent(ReferenceEventKind.TASK_TERMINAL, target, "success") for target in targets)
        if schedule == "terminal_then_cancel":
            events.append(ReferenceEvent(ReferenceEventKind.CANCELLATION))
        elif schedule.startswith("terminal_"):
            events.append(
                ReferenceEvent(
                    ReferenceEventKind.TASK_CORRUPTION,
                    targets[0],
                    schedule.removeprefix("terminal_"),
                )
            )
    return tuple(events)


def _cleanup_events(cleanup: str) -> tuple[ReferenceEvent, ...]:
    primary = ReferenceEvent(ReferenceEventKind.CLEANUP_PRIMARY, outcome=cleanup)
    if cleanup in {"duplicate", "contradictory"}:
        return (primary, ReferenceEvent(ReferenceEventKind.CLEANUP_COMPETING, outcome=cleanup))
    return (primary,)


def _validate_schedule(
    events: tuple[ReferenceEvent, ...],
    bindings: tuple[tuple[str, str, int], ...],
    targets: tuple[str, ...],
    event_max: int,
) -> None:
    if len(events) > event_max:
        raise AssertionError("reference trace exceeded the deterministic Phase 5 bound")
    binding_slots = {kind for kind in ReferenceEventKind if kind.value.startswith("binding_")}
    task_slots = {
        ReferenceEventKind.TASK_DISPATCH,
        ReferenceEventKind.TASK_TERMINAL,
        ReferenceEventKind.TASK_CORRUPTION,
    }
    invocation_slots = set(ReferenceEventKind) - binding_slots - task_slots
    if sum(event.kind in binding_slots for event in events) > 4 * len(bindings):
        raise AssertionError("reference binding event cardinality exceeded")
    if sum(event.kind in task_slots for event in events) > 3 * len(targets):
        raise AssertionError("reference task event cardinality exceeded")
    if sum(event.kind in invocation_slots for event in events) > 7:
        raise AssertionError("reference invocation event cardinality exceeded")
    _validate_lifecycle_order(events)


def _validate_lifecycle_order(events: tuple[ReferenceEvent, ...]) -> None:
    if _is_pre_dispatch_cancellation(events):
        return
    positions = {id(event): index for index, event in enumerate(events)}
    for subject in {event.subject for event in events if event.kind.value.startswith("binding_")}:
        construction = _first_position(events, ReferenceEventKind.BINDING_CONSTRUCTION, subject)
        commitment = _first_position(events, ReferenceEventKind.BINDING_COMMITMENT, subject)
        consumption = _first_position(events, ReferenceEventKind.BINDING_CONSUMPTION, subject)
        if construction is None or commitment is None or construction >= commitment:
            raise AssertionError("binding construction must precede commitment")
        if consumption is not None and commitment >= consumption:
            raise AssertionError("binding commitment must precede consumption")
    binding_end = max(
        (positions[id(event)] for event in events if event.kind.value.startswith("binding_")),
        default=-1,
    )
    for event in events:
        if event.kind is ReferenceEventKind.TASK_DISPATCH and positions[id(event)] <= binding_end:
            raise AssertionError("dispatch must follow binding construction")
        if event.kind is ReferenceEventKind.TASK_TERMINAL:
            dispatch = _first_position(events, ReferenceEventKind.TASK_DISPATCH, event.subject)
            if dispatch is None or dispatch >= positions[id(event)]:
                raise AssertionError("task terminal must follow dispatch")
    cleanup_positions = [
        positions[id(event)]
        for event in events
        if event.kind in {ReferenceEventKind.CLEANUP_PRIMARY, ReferenceEventKind.CLEANUP_COMPETING}
    ]
    pre_cleanup = [
        positions[id(event)]
        for event in events
        if event.kind
        not in {
            ReferenceEventKind.CLEANUP_PRIMARY,
            ReferenceEventKind.CLEANUP_COMPETING,
            ReferenceEventKind.PUBLICATION,
            ReferenceEventKind.TEARDOWN,
        }
    ]
    publication = _first_position(events, ReferenceEventKind.PUBLICATION)
    teardown = _first_position(events, ReferenceEventKind.TEARDOWN)
    if not cleanup_positions or (pre_cleanup and min(cleanup_positions) <= max(pre_cleanup)):
        raise AssertionError("cleanup must follow terminal evidence")
    if publication is None or publication <= max(cleanup_positions):
        raise AssertionError("publication must follow cleanup")
    if teardown is None or teardown <= publication:
        raise AssertionError("teardown must follow publication")


def _first_position(
    events: tuple[ReferenceEvent, ...],
    kind: ReferenceEventKind,
    subject: str | None = None,
) -> int | None:
    return next(
        (
            index
            for index, event in enumerate(events)
            if event.kind is kind and (subject is None or event.subject == subject)
        ),
        None,
    )


def _is_pre_dispatch_cancellation(events: tuple[ReferenceEvent, ...]) -> bool:
    return any(event.kind is ReferenceEventKind.CANCELLATION for event in events) and not any(
        event.kind is ReferenceEventKind.TASK_DISPATCH for event in events
    )


def _admit(case: ReferenceCase) -> ReferenceAdmission:
    targets = tuple(identifier for identifier, _text in case.targets)
    contexts = tuple(identifier for identifier, _text in case.context_only)
    text = dict((*case.targets, *case.context_only))
    scope_targets = tuple(scope.target for scope in case.scopes)
    if set(scope_targets) != set(targets) or len(scope_targets) != len(set(scope_targets)):
        return ReferenceAdmission.STRUCTURAL
    referenced_context: set[str] = set()
    for scope in case.scopes:
        if (
            scope.target not in targets
            or scope.target in scope.context
            or len(scope.context) != len(set(scope.context))
        ):
            return ReferenceAdmission.STRUCTURAL
        if any(member not in text for member in scope.context):
            return ReferenceAdmission.STRUCTURAL
        if not case.allow_target_as_context and any(member in targets for member in scope.context):
            return ReferenceAdmission.STRUCTURAL
        referenced_context.update(member for member in scope.context if member in contexts)
    if referenced_context != set(contexts):
        return ReferenceAdmission.STRUCTURAL
    if (case.relation, case.profile, case.schema, case.ordering) != (
        "bounded_context",
        "target-context-v1",
        "context-workframe-v1",
        "declared",
    ):
        return ReferenceAdmission.CONTRACT
    if _exceeds(_exact_limits(case.targets, case.context_only, case.scopes), case.limits):
        return ReferenceAdmission.LIMIT
    return (
        ReferenceAdmission.ADMITTED
        if case.preflight_capability == "compatible"
        else ReferenceAdmission.PREFLIGHT_CAPABILITY
    )


def _exact_limits(
    targets: tuple[tuple[str, str], ...],
    contexts: tuple[tuple[str, str], ...],
    scopes: tuple[ReferenceScope, ...],
) -> ReferenceLimits:
    values = (*targets, *contexts)
    text = dict(values)
    scope_bytes = tuple(
        sum(len(text[member].encode()) for member in scope.context if member in text) for scope in scopes
    )
    return ReferenceLimits(
        max((len(value.encode()) for _identifier, value in values), default=0),
        max((len(identifier.encode()) for identifier, _value in values), default=0),
        max((len(scope.context) for scope in scopes), default=0),
        max(scope_bytes, default=0),
        sum(len(scope.context) for scope in scopes),
        sum(len(value.encode()) for _identifier, value in targets) + sum(scope_bytes),
    )


def _exceeds(actual: ReferenceLimits, ceiling: ReferenceLimits) -> bool:
    return any(
        actual_value > ceiling_value
        for actual_value, ceiling_value in zip(asdict(actual).values(), asdict(ceiling).values(), strict=True)
    )


def _binding_outcomes(
    events: tuple[ReferenceEvent, ...],
    bindings: tuple[tuple[str, str, int], ...],
) -> tuple[dict[str, tuple[str, str]], bool]:
    states: dict[str, tuple[str, str]] = {}
    global_fault = False
    for owner, _member, ordinal in bindings:
        subject = f"{owner}:{ordinal}"
        kinds = {event.kind for event in events if event.subject == subject}
        if ReferenceEventKind.BINDING_CONSTRUCTION not in kinds or ReferenceEventKind.BINDING_COMMITMENT not in kinds:
            states[subject] = ("failed", "construction_missing")
        elif ReferenceEventKind.BINDING_CONSUMPTION not in kinds:
            states[subject] = ("inconsistent", "missing")
        else:
            states[subject] = ("available", "none")
    for event in events:
        if event.kind is not ReferenceEventKind.BINDING_CORRUPTION:
            continue
        if event.outcome in {"foreign", "cross_target", "contradictory"}:
            global_fault = True
        elif event.subject in states:
            reason = "contradictory" if event.outcome == "wrong_ordinal" else event.outcome
            states[event.subject] = ("inconsistent", reason)
    return states, global_fault


def _task_outcomes(
    events: tuple[ReferenceEvent, ...],
    targets: tuple[str, ...],
    binding_states: dict[str, tuple[str, str]],
) -> tuple[tuple[str, str, str], ...]:
    outcomes: list[tuple[str, str, str]] = []
    cancellation = any(event.kind is ReferenceEventKind.CANCELLATION for event in events)
    trusted_stop = any(event.kind is ReferenceEventKind.TRUSTED_STOP for event in events)
    transport_loss = any(event.kind is ReferenceEventKind.TRANSPORT_LOSS for event in events)
    for target in targets:
        owned_faults = tuple(
            value
            for subject, value in binding_states.items()
            if subject.startswith(f"{target}:") and value[0] != "available"
        )
        if owned_faults:
            state, reason = owned_faults[0]
        elif not any(event.kind is ReferenceEventKind.TASK_DISPATCH and event.subject == target for event in events):
            state, reason = ("cancelled", "cancellation") if cancellation else ("failed", "not_dispatched")
        elif any(event.kind is ReferenceEventKind.TASK_TERMINAL and event.subject == target for event in events):
            terminal = next(
                event for event in events if event.kind is ReferenceEventKind.TASK_TERMINAL and event.subject == target
            )
            state, reason = ("succeeded", "none") if terminal.outcome == "success" else ("failed", terminal.outcome)
        elif trusted_stop:
            state, reason = "cancelled", "stop_acknowledged"
        elif transport_loss:
            state, reason = "lost", "transport_lost"
        else:
            state, reason = "inconsistent", "terminal_missing"
        outcomes.append((target, state, reason))
    return tuple(outcomes)


def _cleanup_outcome(events: tuple[ReferenceEvent, ...]) -> str:
    cleanup = tuple(
        event
        for event in events
        if event.kind in {ReferenceEventKind.CLEANUP_PRIMARY, ReferenceEventKind.CLEANUP_COMPETING}
    )
    if len(cleanup) != 1:
        return "unconfirmed"
    outcome = cleanup[0].outcome
    if outcome == "verified":
        return "verified"
    if outcome == "failed":
        return "failed"
    return "unconfirmed"


def _invocation_outcome(
    events: tuple[ReferenceEvent, ...],
    tasks: tuple[tuple[str, str, str], ...],
    global_binding_fault: bool,
    cleanup: str,
) -> tuple[ReferenceInvocation, str]:
    if global_binding_fault:
        return ReferenceInvocation.INCONSISTENT, "contradictory"
    if cleanup == "failed":
        return ReferenceInvocation.FAILED, "cleanup_failed"
    if cleanup != "verified":
        return ReferenceInvocation.INCONSISTENT, "cleanup_unconfirmed"
    states = {state for _target, state, _reason in tasks}
    if "lost" in states:
        return ReferenceInvocation.LOST, "transport_lost"
    if states == {"cancelled"}:
        return ReferenceInvocation.CANCELLED, "cancellation"
    if any(event.kind is ReferenceEventKind.CANCELLATION for event in events):
        return ReferenceInvocation.CANCELLED, "cancellation"
    if any(event.kind is ReferenceEventKind.TASK_CORRUPTION for event in events):
        return ReferenceInvocation.INCONSISTENT, "terminal_attribution_invalid"
    return ReferenceInvocation.COMPLETED, "none"


def _released(case: ReferenceCase, tasks: tuple[tuple[str, str, str], ...]) -> tuple[str, ...]:
    eligible = {target for target, state, _reason in tasks if state == "succeeded"}
    changed = True
    while changed:
        changed = False
        for prerequisite, dependent in case.dependencies:
            if prerequisite not in eligible and dependent in eligible:
                eligible.remove(dependent)
                changed = True
        for group in case.groups:
            if not set(group).issubset(eligible) and eligible.intersection(group):
                eligible.difference_update(group)
                changed = True
    return tuple(target for target, _text in case.targets if target in eligible)


def _event_outcome(events: tuple[ReferenceEvent, ...], kind: ReferenceEventKind) -> str:
    matches = tuple(event.outcome for event in events if event.kind is kind)
    return matches[0] if len(matches) == 1 else "missing" if not matches else "contradictory"


def _not_opened(
    admission: ReferenceAdmission,
    binding_count: int,
    event_count: int,
    event_max: int,
    reason: str,
) -> ReferenceResult:
    return ReferenceResult(
        admission,
        ReferenceInvocation.NOT_OPENED,
        reason,
        binding_count,
        (),
        (),
        (),
        (),
        "not_entered",
        False,
        "not_entered",
        event_count,
        event_max,
    )


def _increment(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1
