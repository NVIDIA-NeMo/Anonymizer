# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Phase 6 oracle with no production, pandas, or DataDesigner imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import TypeAlias

REFERENCE_MODEL_VERSION = "phase6-reference/v3"
GENERATOR_VERSION = "phase6-finite-envelope/v3"
SYMBOLIC_ALPHABET = ("A", "é", "😀", "e\u0301", " ")
EVENT_ALPHABET = (
    "dispatch",
    "terminal",
    "candidate_decision",
    "evidence",
    "finalize",
    "patch",
    "transform",
    "verify",
    "group_verify",
    "release",
    "cancel",
    "trusted_stop",
    "loss",
    "cleanup",
    "immutable_accept",
    "teardown",
)
INDEPENDENCE_RELATION = (
    ("candidate_decision", "evidence"),
    ("patch", "group_verify"),
    ("terminal", "terminal"),
)
ORDERED_RACES = (
    ("dispatch", "terminal"),
    ("cancel", "terminal"),
    ("verify", "release"),
    ("finalize", "release"),
    ("teardown", "immutable_accept"),
)
_REPLACEMENT = "[REDACTED]"
_CandidateKey: TypeAlias = tuple[int, int, int, str, str, str, str | None]


class ReferenceEventKind(str, Enum):
    DISPATCH = "dispatch"
    TERMINAL = "terminal"
    CANDIDATE_DECISION = "candidate_decision"
    EVIDENCE = "evidence"
    FINALIZE = "finalize"
    PATCH = "patch"
    TRANSFORM = "transform"
    VERIFY = "verify"
    GROUP_VERIFY = "group_verify"
    RELEASE = "release"
    CANCEL = "cancel"
    TRUSTED_STOP = "trusted_stop"
    LOSS = "loss"
    CLEANUP = "cleanup"
    IMMUTABLE_ACCEPT = "immutable_accept"
    TEARDOWN = "teardown"


@dataclass(frozen=True, slots=True)
class ReferenceEvent:
    kind: ReferenceEventKind
    subject: str = "invocation"
    outcome: str = "accepted"


@dataclass(frozen=True, slots=True)
class ReferenceScheduleResult:
    invocation: str
    task_terminal: str
    task_outcomes: tuple[tuple[str, str], ...]
    cancellation: str
    finalized: bool
    verified: bool
    cleanup: str
    teardown: str
    immutable_result: bool
    release: str
    released_subjects: tuple[str, ...]
    event_count: int


def default_schedule() -> tuple[ReferenceEvent, ...]:
    """Return the canonical bounded successful Phase 6 schedule."""
    return tuple(
        ReferenceEvent(kind)
        for kind in (
            ReferenceEventKind.DISPATCH,
            ReferenceEventKind.CANDIDATE_DECISION,
            ReferenceEventKind.EVIDENCE,
            ReferenceEventKind.FINALIZE,
            ReferenceEventKind.PATCH,
            ReferenceEventKind.TRANSFORM,
            ReferenceEventKind.VERIFY,
            ReferenceEventKind.GROUP_VERIFY,
            ReferenceEventKind.TERMINAL,
            ReferenceEventKind.CLEANUP,
            ReferenceEventKind.IMMUTABLE_ACCEPT,
            ReferenceEventKind.TEARDOWN,
            ReferenceEventKind.RELEASE,
        )
    )


@dataclass(frozen=True, slots=True)
class ReferenceCandidate:
    target: int
    start: int
    end: int
    source_slice: str
    label: str
    decision: str = "keep"
    reclassified_label: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceEvidence:
    kind: str
    left_candidate: int
    right_candidate: int


@dataclass(frozen=True, slots=True)
class ReferenceCase:
    name: str
    texts: tuple[str, ...]
    candidates: tuple[ReferenceCandidate, ...]
    evidence: tuple[ReferenceEvidence, ...] = ()
    dependencies: tuple[tuple[int, int], ...] = ()
    groups: tuple[tuple[int, ...], ...] = ()
    group_passes: tuple[bool, ...] = ()
    returned: tuple[str, ...] | None = None
    events: tuple[ReferenceEvent, ...] = field(default_factory=default_schedule)
    schedule_class: str = "success"


@dataclass(frozen=True, slots=True)
class ReferenceMention:
    candidate_indexes: tuple[int, ...]
    target: int
    start: int
    end: int
    source_slice: str
    label: str


@dataclass(frozen=True, slots=True)
class ReferenceResult:
    rejection: str | None
    mentions: tuple[ReferenceMention, ...]
    clusters: tuple[tuple[int, ...], ...]
    outputs: tuple[str, ...]
    released_groups: tuple[int, ...]
    schedule: ReferenceScheduleResult
    event_count: int
    max_event_count: int


def reduce_reference(case: ReferenceCase) -> ReferenceResult:
    """Derive the complete bounded outcome without production decisions as inputs."""
    groups = case.groups or tuple((index,) for index in range(len(case.texts)))
    group_passes = case.group_passes or tuple(True for _group in groups)
    base_bound = _event_bound(case, groups)
    schedule = _reduce_schedule(case.events, base_bound)
    rejected = _validate_declaration(case, groups, group_passes)
    if rejected is not None:
        return _rejected_result(rejected, schedule, base_bound)
    finalized = _finalize(case)
    if isinstance(finalized, str):
        return _rejected_result(finalized, schedule, base_bound)
    mentions, mention_by_candidate = finalized
    clustered = _cluster(mentions, mention_by_candidate, case.evidence)
    if isinstance(clustered, str):
        return _rejected_result(clustered, schedule, base_bound, mentions=mentions)
    outputs = _redact(case.texts, mentions)
    if isinstance(outputs, str):
        return _rejected_result(outputs, schedule, base_bound, mentions=mentions, clusters=clustered)
    if case.returned is not None and case.returned != outputs:
        return _rejected_result(
            "release_predicate_failed",
            schedule,
            base_bound,
            mentions=mentions,
            clusters=clustered,
            outputs=outputs,
        )
    eligible = set(range(len(case.texts)))
    for group, passed in zip(groups, group_passes, strict=True):
        if not passed:
            eligible.difference_update(group)
    scheduled_subjects = {
        int(subject.removeprefix("target-"))
        for subject, outcome in schedule.task_outcomes
        if subject.startswith("target-") and outcome != "succeeded"
    }
    eligible.difference_update(scheduled_subjects)
    while True:
        before = set(eligible)
        for group in groups:
            if not set(group).issubset(eligible):
                eligible.difference_update(group)
        for prerequisite, dependent in case.dependencies:
            if prerequisite not in eligible:
                eligible.discard(dependent)
        if eligible == before:
            break
    released = (
        tuple(index for index, group in enumerate(groups) if set(group).issubset(eligible))
        if schedule.release == "accepted"
        else ()
    )
    return ReferenceResult(
        None,
        mentions,
        clustered,
        outputs,
        released,
        schedule,
        schedule.event_count,
        base_bound,
    )


def finite_reference_cases() -> tuple[ReferenceCase, ...]:
    return (*_alphabet_cases(), *_anchor_cases(), *_evidence_cases(), *_fault_cases())


def _alphabet_cases() -> tuple[ReferenceCase, ...]:
    cases: list[ReferenceCase] = []
    for index, text in enumerate(SYMBOLIC_ALPHABET):
        cases.append(ReferenceCase(f"alphabet-{index}-empty", (text,), ()))
        cases.append(
            ReferenceCase(
                f"alphabet-{index}-whole",
                (text,),
                (ReferenceCandidate(0, 0, len(text), text, "label"),),
            )
        )
    return tuple(cases)


def _anchor_cases() -> tuple[ReferenceCase, ...]:
    return (
        ReferenceCase("repeated-first", ("A A",), (ReferenceCandidate(0, 0, 1, "A", "name"),)),
        ReferenceCase(
            "repeated-both",
            ("A A",),
            (
                ReferenceCandidate(0, 0, 1, "A", "name"),
                ReferenceCandidate(0, 2, 3, "A", "name"),
            ),
        ),
        ReferenceCase(
            "adjacent-unicode",
            ("A😀",),
            (
                ReferenceCandidate(0, 0, 1, "A", "name"),
                ReferenceCandidate(0, 1, 2, "😀", "symbol", "reclass", "emoji"),
            ),
        ),
    )


def _fault_cases() -> tuple[ReferenceCase, ...]:
    candidates = (
        ReferenceCandidate(0, 0, 1, "A", "name"),
        ReferenceCandidate(0, 2, 3, "B", "name"),
        ReferenceCandidate(0, 4, 5, "C", "name"),
    )
    return (
        ReferenceCase(
            "transitive-contradiction",
            ("A B C",),
            candidates,
            (
                ReferenceEvidence("same_subject", 0, 1),
                ReferenceEvidence("same_subject", 1, 2),
                ReferenceEvidence("distinct_subject", 0, 2),
            ),
        ),
        ReferenceCase("invalid-offset", ("A",), (ReferenceCandidate(0, -1, 1, "A", "name"),)),
        ReferenceCase("slice-mismatch", ("Alice",), (ReferenceCandidate(0, 0, 5, "Mallory", "name"),)),
        ReferenceCase(
            "missing-decision",
            ("Alice",),
            (ReferenceCandidate(0, 0, 5, "Alice", "name", "missing"),),
        ),
        ReferenceCase(
            "group-propagation",
            ("A", "B"),
            (),
            dependencies=((0, 1),),
            groups=((0,), (1,)),
            group_passes=(False, True),
        ),
    )


def reference_manifest() -> dict[str, object]:
    cases = finite_reference_cases()
    races = ordered_race_schedules()
    lifecycle = lifecycle_reference_cases()
    payload: list[dict[str, object]] = [
        {"case": _case_payload(case), "result": asdict(reduce_reference(case))} for case in cases
    ]
    payload.extend(
        {
            "race": name,
            "orientation": orientation,
            "events": [asdict(event) for event in events],
            "result": asdict(reduce_reference(ReferenceCase(f"{name}-{orientation}", ("A",), (), events=events))),
        }
        for name, first, second in races
        for orientation, events in (("first", first), ("second", second))
    )
    payload.extend({"lifecycle": _case_payload(case), "result": asdict(reduce_reference(case))} for case in lifecycle)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    race_event_count = sum(len(events) for _name, first, second in races for events in (first, second))
    lifecycle_event_count = sum(len(case.events) for case in lifecycle)
    schedule_class_counts: dict[str, int] = {"success": len(cases)}
    for name, _first, _second in races:
        schedule_class_counts[name] = 2
    for case in lifecycle:
        schedule_class_counts[case.schedule_class] = schedule_class_counts.get(case.schedule_class, 0) + 1
    return {
        "alphabet": list(SYMBOLIC_ALPHABET),
        "actual_event_count": sum(len(case.events) for case in cases) + race_event_count + lifecycle_event_count,
        "canonical_trace_count": len(cases) + 2 * len(races) + len(lifecycle),
        "case_count": len(cases),
        "digest": digest,
        "event_alphabet": list(EVENT_ALPHABET),
        "generator_version": GENERATOR_VERSION,
        "independence_relation": [list(pair) for pair in INDEPENDENCE_RELATION],
        "lifecycle_trace_count": len(lifecycle),
        "max_event_count": max(reduce_reference(case).max_event_count for case in (*cases, *lifecycle)),
        "ordered_races": [list(pair) for pair in ORDERED_RACES],
        "race_trace_count": 2 * len(races) + len(lifecycle),
        "reference_model_version": REFERENCE_MODEL_VERSION,
        "schedule_class_counts": schedule_class_counts,
    }


def canonical_schedule(events: tuple[ReferenceEvent, ...]) -> tuple[ReferenceEvent, ...]:
    """Canonicalize only adjacent events declared to commute."""
    relation = {frozenset(pair) for pair in INDEPENDENCE_RELATION}
    rank = {value: index for index, value in enumerate(EVENT_ALPHABET)}
    canonical = list(events)
    changed = True
    while changed:
        changed = False
        for index in range(len(canonical) - 1):
            left = canonical[index]
            right = canonical[index + 1]
            if frozenset((left.kind.value, right.kind.value)) in relation and (rank[left.kind.value], left.subject) > (
                rank[right.kind.value],
                right.subject,
            ):
                canonical[index], canonical[index + 1] = right, left
                changed = True
    return tuple(canonical)


def ordered_race_schedules() -> tuple[tuple[str, tuple[ReferenceEvent, ...], tuple[ReferenceEvent, ...]], ...]:
    """Return both orientations of each release-critical ordered race."""
    event = ReferenceEvent
    kind = ReferenceEventKind
    return (
        (
            "dispatch-terminal",
            (event(kind.DISPATCH), event(kind.TERMINAL)),
            (event(kind.TERMINAL), event(kind.DISPATCH)),
        ),
        (
            "cancel-terminal",
            (event(kind.DISPATCH), event(kind.CANCEL), event(kind.TERMINAL)),
            (event(kind.DISPATCH), event(kind.TERMINAL), event(kind.CANCEL)),
        ),
        (
            "verify-release",
            (
                event(kind.DISPATCH),
                event(kind.TERMINAL),
                event(kind.FINALIZE),
                event(kind.CLEANUP),
                event(kind.TEARDOWN),
                event(kind.VERIFY),
                event(kind.IMMUTABLE_ACCEPT),
                event(kind.RELEASE),
            ),
            (
                event(kind.DISPATCH),
                event(kind.TERMINAL),
                event(kind.FINALIZE),
                event(kind.CLEANUP),
                event(kind.TEARDOWN),
                event(kind.RELEASE),
                event(kind.VERIFY),
                event(kind.IMMUTABLE_ACCEPT),
            ),
        ),
        (
            "finalize-release",
            (
                event(kind.DISPATCH),
                event(kind.TERMINAL),
                event(kind.VERIFY),
                event(kind.CLEANUP),
                event(kind.TEARDOWN),
                event(kind.FINALIZE),
                event(kind.IMMUTABLE_ACCEPT),
                event(kind.RELEASE),
            ),
            (
                event(kind.DISPATCH),
                event(kind.TERMINAL),
                event(kind.VERIFY),
                event(kind.CLEANUP),
                event(kind.TEARDOWN),
                event(kind.RELEASE),
                event(kind.FINALIZE),
                event(kind.IMMUTABLE_ACCEPT),
            ),
        ),
        (
            "teardown-acceptance",
            (
                event(kind.DISPATCH),
                event(kind.TERMINAL),
                event(kind.FINALIZE),
                event(kind.VERIFY),
                event(kind.CLEANUP),
                event(kind.TEARDOWN, outcome="failed"),
                event(kind.IMMUTABLE_ACCEPT),
                event(kind.RELEASE),
            ),
            (
                event(kind.DISPATCH),
                event(kind.TERMINAL),
                event(kind.FINALIZE),
                event(kind.VERIFY),
                event(kind.CLEANUP),
                event(kind.IMMUTABLE_ACCEPT),
                event(kind.TEARDOWN, outcome="failed"),
                event(kind.RELEASE),
            ),
        ),
    )


def lifecycle_reference_cases() -> tuple[ReferenceCase, ...]:
    """Return the frozen lifecycle schedules required by the Phase 6 design."""
    event = ReferenceEvent
    kind = ReferenceEventKind

    def tail(*, teardown: str = "accepted", cleanup: str = "accepted") -> tuple[ReferenceEvent, ...]:
        return (
            event(kind.FINALIZE),
            event(kind.VERIFY),
            event(kind.CLEANUP, outcome=cleanup),
            event(kind.IMMUTABLE_ACCEPT),
            event(kind.TEARDOWN, outcome=teardown),
            event(kind.RELEASE),
        )

    target = "target-0"
    return (
        ReferenceCase(
            "cancel-before-dispatch",
            ("A",),
            (),
            events=(event(kind.CANCEL), event(kind.DISPATCH, target), event(kind.TERMINAL, target), *tail()),
            schedule_class="cancel-dispatch",
        ),
        ReferenceCase(
            "cancel-after-dispatch",
            ("A",),
            (),
            events=(event(kind.DISPATCH, target), event(kind.CANCEL), event(kind.TERMINAL, target), *tail()),
            schedule_class="cancel-dispatch",
        ),
        ReferenceCase(
            "late-candidate-after-cancel",
            ("A",),
            (),
            events=(event(kind.DISPATCH, target), event(kind.CANCEL), event(kind.CANDIDATE_DECISION, target), *tail()),
            schedule_class="late-candidate-after-cancel",
        ),
        ReferenceCase(
            "late-evidence-after-cancel",
            ("A",),
            (),
            events=(event(kind.DISPATCH, target), event(kind.CANCEL), event(kind.EVIDENCE, target), *tail()),
            schedule_class="late-evidence-after-cancel",
        ),
        ReferenceCase(
            "late-candidate-after-loss",
            ("A",),
            (),
            events=(event(kind.DISPATCH, target), event(kind.LOSS), event(kind.CANDIDATE_DECISION, target), *tail()),
            schedule_class="late-candidate-after-loss",
        ),
        ReferenceCase(
            "late-evidence-after-loss",
            ("A",),
            (),
            events=(event(kind.DISPATCH, target), event(kind.LOSS), event(kind.EVIDENCE, target), *tail()),
            schedule_class="late-evidence-after-loss",
        ),
        ReferenceCase(
            "duplicate-resolver-completion",
            ("A",),
            (),
            events=(event(kind.DISPATCH, target), event(kind.EVIDENCE, target, "duplicate"), *tail()),
            schedule_class="duplicate-resolver-completion",
        ),
        ReferenceCase(
            "patch-before-contradictory-record",
            ("A",),
            (),
            events=(
                event(kind.DISPATCH, target),
                event(kind.PATCH, target),
                event(kind.TERMINAL, target, "contradictory"),
                *tail(),
            ),
            schedule_class="contradictory-record-patch",
        ),
        ReferenceCase(
            "contradictory-record-before-patch",
            ("A",),
            (),
            events=(
                event(kind.DISPATCH, target),
                event(kind.TERMINAL, target, "contradictory"),
                event(kind.PATCH, target),
                *tail(),
            ),
            schedule_class="contradictory-record-patch",
        ),
        ReferenceCase(
            "local-failure-independent-success",
            ("A", "B"),
            (),
            groups=((0,), (1,)),
            events=(
                event(kind.DISPATCH, "target-0"),
                event(kind.TERMINAL, "target-0", "failed"),
                event(kind.DISPATCH, "target-1"),
                event(kind.TERMINAL, "target-1"),
                *tail(),
            ),
            schedule_class="local-failure-independent-success",
        ),
        ReferenceCase(
            "cancel-after-verification",
            ("A",),
            (),
            events=(
                event(kind.DISPATCH, target),
                event(kind.TERMINAL, target),
                event(kind.FINALIZE),
                event(kind.VERIFY),
                event(kind.CANCEL),
                event(kind.CLEANUP),
                event(kind.IMMUTABLE_ACCEPT),
                event(kind.TEARDOWN),
                event(kind.RELEASE),
            ),
            schedule_class="cancel-after-verification",
        ),
        ReferenceCase(
            "cleanup-failure",
            ("A",),
            (),
            events=(event(kind.DISPATCH, target), event(kind.TERMINAL, target), *tail(cleanup="unconfirmed")),
            schedule_class="cleanup-failure",
        ),
        ReferenceCase(
            "teardown-failure-after-acceptance",
            ("A",),
            (),
            events=(event(kind.DISPATCH, target), event(kind.TERMINAL, target), *tail(teardown="failed")),
            schedule_class="teardown-failure-after-acceptance",
        ),
    )


def schedule_reference_cases() -> tuple[ReferenceCase, ...]:
    """Return every frozen schedule that must be compared with production accounting."""
    ordered = tuple(
        ReferenceCase(
            f"{name}-{orientation}",
            ("A",),
            (),
            events=tuple(
                replace(event, subject="target-0") if event.subject == "invocation" else event for event in events
            ),
            schedule_class=name,
        )
        for name, first, second in ordered_race_schedules()
        for orientation, events in (("first", first), ("second", second))
    )
    return (*ordered, *lifecycle_reference_cases())


def _evidence_cases() -> tuple[ReferenceCase, ...]:
    candidates = (
        ReferenceCandidate(0, 0, 5, "Alice", "name"),
        ReferenceCandidate(1, 0, 2, "Al", "alias"),
    )
    return tuple(
        ReferenceCase(f"evidence-{kind}", ("Alice", "Al"), candidates, evidence)
        for kind, evidence in (
            ("none", ()),
            ("same", (ReferenceEvidence("same_subject", 0, 1),)),
            ("distinct", (ReferenceEvidence("distinct_subject", 0, 1),)),
        )
    )


def _validate_declaration(
    case: ReferenceCase,
    groups: tuple[tuple[int, ...], ...],
    group_passes: tuple[bool, ...],
) -> str | None:
    target_indexes = set(range(len(case.texts)))
    members = tuple(member for group in groups for member in group)
    if (
        not case.texts
        or set(members) != target_indexes
        or len(members) != len(set(members))
        or len(groups) != len(group_passes)
        or any(type(passed) is not bool for passed in group_passes)
    ):
        return "malformed_graph"
    if any(
        prerequisite not in target_indexes or dependent not in target_indexes or prerequisite == dependent
        for prerequisite, dependent in case.dependencies
    ):
        return "malformed_dependency"
    return None


def _finalize(
    case: ReferenceCase,
) -> tuple[tuple[ReferenceMention, ...], dict[int, int]] | str:
    indexed = _index_candidates(case)
    if isinstance(indexed, str):
        return indexed
    accepted = _materialize_mentions(indexed)
    rejection = _validate_mention_spans(accepted)
    if rejection is not None:
        return rejection
    mention_by_candidate = {
        candidate_index: mention_index
        for mention_index, mention in enumerate(accepted)
        for candidate_index in mention.candidate_indexes
    }
    return accepted, mention_by_candidate


def _index_candidates(case: ReferenceCase) -> dict[_CandidateKey, list[int]] | str:
    exact: dict[_CandidateKey, list[int]] = {}
    for index, candidate in enumerate(case.candidates):
        if (
            candidate.target not in range(len(case.texts))
            or type(candidate.start) is not int
            or type(candidate.end) is not int
            or candidate.start < 0
            or candidate.end <= candidate.start
            or candidate.end > len(case.texts[candidate.target])
        ):
            return "invalid_offset"
        if case.texts[candidate.target][candidate.start : candidate.end] != candidate.source_slice:
            return "source_slice_mismatch"
        if not candidate.label:
            return "invalid_label"
        if candidate.decision not in {"keep", "reclass", "drop"}:
            return "missing_decision"
        if candidate.decision == "reclass" and not candidate.reclassified_label:
            return "invalid_label"
        key = (
            candidate.target,
            candidate.start,
            candidate.end,
            candidate.source_slice,
            candidate.label,
            candidate.decision,
            candidate.reclassified_label,
        )
        exact.setdefault(key, []).append(index)
    return exact


def _materialize_mentions(exact: dict[_CandidateKey, list[int]]) -> tuple[ReferenceMention, ...]:
    accepted: list[ReferenceMention] = []
    for key, indexes in exact.items():
        target, start, end, source_slice, label, decision, reclassified = key
        if decision == "drop":
            continue
        accepted.append(
            ReferenceMention(
                tuple(indexes),
                target,
                start,
                end,
                source_slice,
                reclassified if decision == "reclass" and reclassified is not None else label,
            )
        )
    return tuple(sorted(accepted, key=lambda mention: (mention.target, mention.start, mention.end)))


def _validate_mention_spans(accepted: tuple[ReferenceMention, ...]) -> str | None:
    previous_end: dict[int, int] = {}
    by_span: set[tuple[int, int, int]] = set()
    for mention in accepted:
        span = (mention.target, mention.start, mention.end)
        if span in by_span:
            return "contradictory_candidate"
        if mention.start < previous_end.get(mention.target, 0):
            return "overlap"
        by_span.add(span)
        previous_end[mention.target] = mention.end
    return None


def _cluster(
    mentions: tuple[ReferenceMention, ...],
    mention_by_candidate: dict[int, int],
    evidence: tuple[ReferenceEvidence, ...],
) -> tuple[tuple[int, ...], ...] | str:
    parents = list(range(len(mentions)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    normalized = _normalize_evidence(mention_by_candidate, evidence)
    if isinstance(normalized, str):
        return normalized
    for kind, left, right in sorted(normalized):
        if kind != "same_subject":
            continue
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)
    if any(find(left) == find(right) for kind, left, right in normalized if kind == "distinct_subject"):
        return "evidence_contradiction"
    components: dict[int, list[int]] = {}
    for index in range(len(mentions)):
        components.setdefault(find(index), []).append(index)
    return tuple(tuple(component) for _root, component in sorted(components.items()))


def _normalize_evidence(
    mention_by_candidate: dict[int, int],
    evidence: tuple[ReferenceEvidence, ...],
) -> set[tuple[str, int, int]] | str:
    normalized: set[tuple[str, int, int]] = set()
    for item in evidence:
        if (
            item.kind not in {"same_subject", "distinct_subject"}
            or item.left_candidate not in mention_by_candidate
            or item.right_candidate not in mention_by_candidate
        ):
            return "invalid_evidence"
        left = mention_by_candidate[item.left_candidate]
        right = mention_by_candidate[item.right_candidate]
        if left == right:
            return "invalid_evidence"
        edge = (item.kind, min(left, right), max(left, right))
        if edge in normalized:
            return "invalid_evidence"
        if ("same_subject" if item.kind == "distinct_subject" else "distinct_subject", edge[1], edge[2]) in normalized:
            return "evidence_contradiction"
        normalized.add(edge)
    return normalized


def _redact(texts: tuple[str, ...], mentions: tuple[ReferenceMention, ...]) -> tuple[str, ...] | str:
    by_target: list[list[ReferenceMention]] = [[] for _text in texts]
    for mention in mentions:
        if mention.source_slice in _REPLACEMENT:
            return "invalid_patch"
        by_target[mention.target].append(mention)
    outputs: list[str] = []
    for text, target_mentions in zip(texts, by_target, strict=True):
        cursor = 0
        parts: list[str] = []
        for mention in target_mentions:
            parts.extend((text[cursor : mention.start], _REPLACEMENT))
            cursor = mention.end
        parts.append(text[cursor:])
        outputs.append("".join(parts))
    return tuple(outputs)


def _reduce_schedule(events: tuple[ReferenceEvent, ...], event_bound: int) -> ReferenceScheduleResult:
    if not events:
        raise AssertionError("schedule must contain at least one event")
    if len(events) > event_bound:
        raise AssertionError("schedule exceeds computed event bound")
    if any(not isinstance(event, ReferenceEvent) for event in events):
        raise AssertionError("schedule contains an invalid event")

    subjects = tuple(
        dict.fromkeys(
            event.subject
            for event in events
            if event.subject != "invocation"
            and event.kind
            in {
                ReferenceEventKind.DISPATCH,
                ReferenceEventKind.TERMINAL,
                ReferenceEventKind.CANDIDATE_DECISION,
                ReferenceEventKind.EVIDENCE,
            }
        )
    ) or ("invocation",)
    dispatched: set[str] = set()
    task_outcomes = {subject: "missing" for subject in subjects}
    finalized = False
    verified = False
    cancelled = False
    cancellation = "none"
    cleanup = "unconfirmed"
    teardown = "unconfirmed"
    immutable_result = False
    release = "not_requested"
    release_observed = False
    released_subjects: tuple[str, ...] = ()
    lost = False
    stopped = False
    inconsistent = False
    for event in events:
        subject = event.subject if event.subject != "invocation" else subjects[0]
        match event.kind:
            case ReferenceEventKind.DISPATCH:
                if not cancelled and not lost and task_outcomes.get(subject) == "missing":
                    dispatched.add(subject)
            case ReferenceEventKind.TERMINAL:
                if event.outcome == "contradictory":
                    inconsistent = True
                    task_outcomes[subject] = "inconsistent"
                elif subject in dispatched and task_outcomes.get(subject) == "missing" and not lost and not stopped:
                    task_outcomes[subject] = "succeeded" if event.outcome == "accepted" else "failed"
                else:
                    task_outcomes.setdefault(subject, "rejected")
            case ReferenceEventKind.CANCEL:
                cancellation = (
                    "before_dispatch"
                    if not dispatched
                    else "after_terminal"
                    if any(outcome not in {"missing", "cancelled"} for outcome in task_outcomes.values())
                    else "after_dispatch"
                )
                cancelled = True
                for planned in subjects:
                    if task_outcomes[planned] == "missing":
                        task_outcomes[planned] = "cancelled"
            case ReferenceEventKind.TRUSTED_STOP:
                stopped = True
            case ReferenceEventKind.LOSS:
                lost = True
                for active in dispatched:
                    if task_outcomes[active] == "missing":
                        task_outcomes[active] = "lost"
            case ReferenceEventKind.CANDIDATE_DECISION | ReferenceEventKind.EVIDENCE:
                if event.outcome == "duplicate":
                    inconsistent = True
                    task_outcomes[subject] = "inconsistent"
            case ReferenceEventKind.FINALIZE:
                finalized = True
            case ReferenceEventKind.VERIFY:
                verified = True
            case ReferenceEventKind.CLEANUP:
                cleanup = event.outcome
            case ReferenceEventKind.IMMUTABLE_ACCEPT:
                immutable_result = (
                    event.outcome == "accepted"
                    and any(outcome == "succeeded" for outcome in task_outcomes.values())
                    and finalized
                    and verified
                    and cleanup == "accepted"
                    and teardown != "failed"
                    and not cancelled
                    and not lost
                    and not stopped
                    and not inconsistent
                    and not release_observed
                )
                if immutable_result:
                    released_subjects = tuple(
                        subject for subject, outcome in task_outcomes.items() if outcome == "succeeded"
                    )
            case ReferenceEventKind.TEARDOWN:
                teardown = event.outcome
            case ReferenceEventKind.RELEASE:
                release_observed = True
                release = "accepted" if immutable_result else "withheld"
            case _:
                pass

    for subject in subjects:
        if task_outcomes[subject] == "missing":
            if subject in dispatched:
                task_outcomes[subject] = "lost"
                lost = True
            else:
                task_outcomes[subject] = "blocked"
    terminal_states = tuple(task_outcomes.values())
    task_terminal = terminal_states[0] if len(set(terminal_states)) == 1 else "mixed"
    if immutable_result:
        invocation = "completed"
    elif cleanup == "unconfirmed":
        invocation = "inconsistent"
    elif teardown == "failed":
        invocation = "failed"
    elif cleanup == "failed":
        invocation = "failed"
    elif inconsistent:
        invocation = "inconsistent"
    elif lost:
        invocation = "lost"
    elif cancelled:
        invocation = "cancelled"
    elif stopped:
        invocation = "stopped"
    else:
        invocation = "completed"
    return ReferenceScheduleResult(
        invocation,
        task_terminal,
        tuple(task_outcomes.items()),
        cancellation,
        finalized,
        verified,
        cleanup,
        teardown,
        immutable_result,
        release,
        released_subjects,
        len(events),
    )


def _rejected_result(
    rejection: str,
    schedule: ReferenceScheduleResult,
    event_bound: int,
    *,
    mentions: tuple[ReferenceMention, ...] = (),
    clusters: tuple[tuple[int, ...], ...] = (),
    outputs: tuple[str, ...] = (),
) -> ReferenceResult:
    failed_schedule = replace(schedule, release="withheld")
    return ReferenceResult(
        rejection,
        mentions,
        clusters,
        outputs,
        (),
        failed_schedule,
        failed_schedule.event_count,
        event_bound,
    )


def _event_bound(case: ReferenceCase, groups: tuple[tuple[int, ...], ...]) -> int:
    task_events = len(case.texts) * 8 * 2
    return (
        task_events
        + len(case.candidates)
        + len(case.evidence)
        + len(case.candidates)
        + len(case.texts)
        + len(groups)
        + 4
    )


def _case_payload(case: ReferenceCase) -> dict[str, object]:
    return asdict(case)
