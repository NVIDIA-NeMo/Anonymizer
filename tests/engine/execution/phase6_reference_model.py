# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure Phase 6 oracle with no production, pandas, or DataDesigner imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import TypeAlias

REFERENCE_MODEL_VERSION = "phase6-reference/v1"
GENERATOR_VERSION = "phase6-finite-envelope/v1"
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
    ("teardown", "release"),
)
_REPLACEMENT = "[REDACTED]"
_CandidateKey: TypeAlias = tuple[int, int, int, str, str, str, str | None]


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
    max_event_count: int


def reduce_reference(case: ReferenceCase) -> ReferenceResult:
    """Derive the complete bounded outcome without production decisions as inputs."""
    groups = case.groups or tuple((index,) for index in range(len(case.texts)))
    group_passes = case.group_passes or tuple(True for _group in groups)
    base_bound = _event_bound(case, groups)
    rejected = _validate_declaration(case, groups, group_passes)
    if rejected is not None:
        return ReferenceResult(rejected, (), (), (), (), base_bound)
    finalized = _finalize(case)
    if isinstance(finalized, str):
        return ReferenceResult(finalized, (), (), (), (), base_bound)
    mentions, mention_by_candidate = finalized
    clustered = _cluster(mentions, mention_by_candidate, case.evidence)
    if isinstance(clustered, str):
        return ReferenceResult(clustered, mentions, (), (), (), base_bound)
    outputs = _redact(case.texts, mentions)
    if isinstance(outputs, str):
        return ReferenceResult(outputs, mentions, clustered, (), (), base_bound)
    if case.returned is not None and case.returned != outputs:
        return ReferenceResult("release_predicate_failed", mentions, clustered, outputs, (), base_bound)
    eligible = set(range(len(case.texts)))
    for group, passed in zip(groups, group_passes, strict=True):
        if not passed:
            eligible.difference_update(group)
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
    released = tuple(index for index, group in enumerate(groups) if set(group).issubset(eligible))
    return ReferenceResult(None, mentions, clustered, outputs, released, base_bound)


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
    payload = [_case_payload(case) for case in cases]
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "alphabet": list(SYMBOLIC_ALPHABET),
        "canonical_trace_count": len(cases),
        "case_count": len(cases),
        "digest": digest,
        "event_alphabet": list(EVENT_ALPHABET),
        "generator_version": GENERATOR_VERSION,
        "independence_relation": [list(pair) for pair in INDEPENDENCE_RELATION],
        "max_event_count": max(reduce_reference(case).max_event_count for case in cases),
        "ordered_races": [list(pair) for pair in ORDERED_RACES],
        "reference_model_version": REFERENCE_MODEL_VERSION,
    }


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
