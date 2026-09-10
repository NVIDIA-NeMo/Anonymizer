# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict target-anchored mention finalization for the private graph profile."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum

from anonymizer.engine.execution.graph import _DatumId


class _PrivateMentionValue:
    def __repr__(self) -> str:
        return f"<private {type(self).__name__.strip('_').replace('_', ' ').lower()}>"

    def __reduce__(self) -> str | tuple[object, ...]:
        raise TypeError("private mention values are not serializable")


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _MentionTargetToken(_PrivateMentionValue):
    """Executor-issued identity for one current target."""


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _CandidateToken(_PrivateMentionValue):
    """Executor-issued identity for one provisional candidate lineage."""


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class _MentionId(_PrivateMentionValue):
    """Executor-issued graph-scoped mention identity."""


class _MentionProvenance(str, Enum):
    SPAN_DETECTOR = "span_detector"
    EXACT_AUGMENTER = "exact_augmenter"


class _ValidationDecisionKind(str, Enum):
    KEEP = "keep"
    RECLASS = "reclass"
    DROP = "drop"


class _MentionRejectionCode(str, Enum):
    UNKNOWN_TARGET = "unknown_target"
    INVALID_OFFSET = "invalid_offset"
    SOURCE_SLICE_MISMATCH = "source_slice_mismatch"
    UNSUPPORTED_PROVENANCE = "unsupported_provenance"
    MISSING_DECISION = "missing_decision"
    DUPLICATE_DECISION = "duplicate_decision"
    OVERLAP = "overlap"
    FOREIGN_TOKEN = "foreign_token"
    STALE_TOKEN = "stale_token"
    CONTRADICTORY_CANDIDATE = "contradictory_candidate"


@dataclass(frozen=True, slots=True, repr=False)
class _MentionTarget(_PrivateMentionValue):
    token: _MentionTargetToken
    datum_id: _DatumId
    text: str


@dataclass(frozen=True, slots=True, repr=False)
class _ProvisionalCandidate(_PrivateMentionValue):
    token: _CandidateToken
    target_token: _MentionTargetToken
    start: int
    end: int
    source_slice: str
    detector_label: str
    provenance: _MentionProvenance


@dataclass(frozen=True, slots=True, repr=False)
class _ValidationDecision(_PrivateMentionValue):
    candidate_token: _CandidateToken
    kind: _ValidationDecisionKind
    proposed_label: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class _AnchoredMention(_PrivateMentionValue):
    id: _MentionId
    target_datum_id: _DatumId
    start: int
    end: int
    source_slice: str
    detector_label: str
    provenance: _MentionProvenance


@dataclass(frozen=True, slots=True, repr=False)
class _DetectedGraph(_PrivateMentionValue):
    targets: tuple[_MentionTarget, ...]
    mentions: tuple[_AnchoredMention, ...]


@dataclass(frozen=True, slots=True, repr=False)
class _MentionRejected(_PrivateMentionValue):
    code: _MentionRejectionCode
    owner: _MentionTargetToken | None = None


@dataclass(frozen=True, slots=True, repr=False)
class _MentionLimits(_PrivateMentionValue):
    max_candidates_per_target: int
    max_mentions_per_target: int
    max_label_bytes: int
    max_source_slice_bytes: int


def _finalize_mentions(
    targets: tuple[_MentionTarget, ...],
    candidates: tuple[_ProvisionalCandidate, ...],
    decisions: tuple[_ValidationDecision, ...],
    *,
    limits: _MentionLimits,
) -> _DetectedGraph | _MentionRejected:
    target_by_token = _validate_targets(targets)
    if target_by_token is None or not _valid_limits(limits):
        return _MentionRejected(_MentionRejectionCode.UNKNOWN_TARGET)
    normalized = _normalize_candidates(candidates, target_by_token, limits)
    if isinstance(normalized, _MentionRejected):
        return normalized
    decision_by_token = _index_decisions(decisions, normalized)
    if isinstance(decision_by_token, _MentionRejected):
        return decision_by_token
    finalized = _apply_decisions(normalized, decision_by_token, limits)
    if isinstance(finalized, _MentionRejected):
        return finalized
    return _build_detected_graph(targets, target_by_token, finalized, limits)


def _validate_targets(targets: object) -> dict[_MentionTargetToken, _MentionTarget] | None:
    if (
        not isinstance(targets, tuple)
        or not targets
        or not all(isinstance(target, _MentionTarget) for target in targets)
    ):
        return None
    target_by_token: dict[_MentionTargetToken, _MentionTarget] = {}
    datum_ids: set[_DatumId] = set()
    for target in targets:
        if (
            not isinstance(target.token, _MentionTargetToken)
            or not isinstance(target.datum_id, _DatumId)
            or not isinstance(target.text, str)
            or target.token in target_by_token
            or target.datum_id in datum_ids
        ):
            return None
        target_by_token[target.token] = target
        datum_ids.add(target.datum_id)
    return target_by_token


def _normalize_candidates(
    candidates: object,
    target_by_token: dict[_MentionTargetToken, _MentionTarget],
    limits: _MentionLimits,
) -> tuple[_ProvisionalCandidate, ...] | _MentionRejected:
    if not isinstance(candidates, tuple) or not all(isinstance(item, _ProvisionalCandidate) for item in candidates):
        return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE)
    by_token: dict[_CandidateToken, _ProvisionalCandidate] = {}
    for candidate in candidates:
        target = target_by_token.get(candidate.target_token)
        if target is None:
            return _MentionRejected(_MentionRejectionCode.UNKNOWN_TARGET)
        prior = by_token.get(candidate.token)
        if prior is not None:
            if prior != candidate:
                return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE, target.token)
            continue
        rejection = _validate_candidate(candidate, target, limits)
        if rejection is not None:
            return rejection
        by_token[candidate.token] = candidate
    counts = Counter(candidate.target_token for candidate in by_token.values())
    if any(count > limits.max_candidates_per_target for count in counts.values()):
        return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE)
    return tuple(by_token.values())


def _validate_candidate(
    candidate: _ProvisionalCandidate,
    target: _MentionTarget,
    limits: _MentionLimits,
) -> _MentionRejected | None:
    owner = target.token
    if not isinstance(candidate.token, _CandidateToken):
        return _MentionRejected(_MentionRejectionCode.STALE_TOKEN, owner)
    if type(candidate.start) is not int or type(candidate.end) is not int:
        return _MentionRejected(_MentionRejectionCode.INVALID_OFFSET, owner)
    if candidate.start < 0 or candidate.end <= candidate.start or candidate.end > len(target.text):
        return _MentionRejected(_MentionRejectionCode.INVALID_OFFSET, owner)
    if (
        not isinstance(candidate.source_slice, str)
        or target.text[candidate.start : candidate.end] != candidate.source_slice
    ):
        return _MentionRejected(_MentionRejectionCode.SOURCE_SLICE_MISMATCH, owner)
    if not isinstance(candidate.provenance, _MentionProvenance):
        return _MentionRejected(_MentionRejectionCode.UNSUPPORTED_PROVENANCE, owner)
    if not _valid_bounded_text(candidate.detector_label, limits.max_label_bytes):
        return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE, owner)
    if not _valid_bounded_text(candidate.source_slice, limits.max_source_slice_bytes):
        return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE, owner)
    return None


def _index_decisions(
    decisions: object,
    candidates: tuple[_ProvisionalCandidate, ...],
) -> dict[_CandidateToken, _ValidationDecision] | _MentionRejected:
    if not isinstance(decisions, tuple) or not all(isinstance(item, _ValidationDecision) for item in decisions):
        return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE)
    candidate_by_token = {candidate.token: candidate for candidate in candidates}
    decision_by_token: dict[_CandidateToken, _ValidationDecision] = {}
    for decision in decisions:
        candidate = candidate_by_token.get(decision.candidate_token)
        if candidate is None:
            return _MentionRejected(_MentionRejectionCode.FOREIGN_TOKEN)
        if decision.candidate_token in decision_by_token:
            return _MentionRejected(_MentionRejectionCode.DUPLICATE_DECISION, candidate.target_token)
        decision_by_token[decision.candidate_token] = decision
    missing = next((candidate for candidate in candidates if candidate.token not in decision_by_token), None)
    if missing is not None:
        return _MentionRejected(_MentionRejectionCode.MISSING_DECISION, missing.target_token)
    return decision_by_token


def _apply_decisions(
    candidates: tuple[_ProvisionalCandidate, ...],
    decisions: dict[_CandidateToken, _ValidationDecision],
    limits: _MentionLimits,
) -> tuple[_ProvisionalCandidate, ...] | _MentionRejected:
    finalized: list[_ProvisionalCandidate] = []
    for candidate in candidates:
        decision = decisions[candidate.token]
        if not isinstance(decision.kind, _ValidationDecisionKind):
            return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE, candidate.target_token)
        if decision.kind is _ValidationDecisionKind.RECLASS:
            if not _valid_bounded_text(decision.proposed_label, limits.max_label_bytes):
                return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE, candidate.target_token)
            finalized.append(replace(candidate, detector_label=decision.proposed_label))
        elif decision.proposed_label not in {None, ""}:
            return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE, candidate.target_token)
        elif decision.kind is _ValidationDecisionKind.KEEP:
            finalized.append(candidate)
    return tuple(finalized)


def _build_detected_graph(
    targets: tuple[_MentionTarget, ...],
    target_by_token: dict[_MentionTargetToken, _MentionTarget],
    candidates: tuple[_ProvisionalCandidate, ...],
    limits: _MentionLimits,
) -> _DetectedGraph | _MentionRejected:
    target_position = {target.token: index for index, target in enumerate(targets)}
    ordered = sorted(candidates, key=lambda item: (target_position[item.target_token], item.start, item.end))
    by_span: dict[tuple[_MentionTargetToken, int, int], _ProvisionalCandidate] = {}
    previous_end: dict[_MentionTargetToken, int] = {}
    mention_counts: Counter[_MentionTargetToken] = Counter()
    mentions: list[_AnchoredMention] = []
    for candidate in ordered:
        span = (candidate.target_token, candidate.start, candidate.end)
        if span in by_span:
            return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE, candidate.target_token)
        if candidate.start < previous_end.get(candidate.target_token, 0):
            return _MentionRejected(_MentionRejectionCode.OVERLAP, candidate.target_token)
        by_span[span] = candidate
        previous_end[candidate.target_token] = candidate.end
        mention_counts[candidate.target_token] += 1
        if mention_counts[candidate.target_token] > limits.max_mentions_per_target:
            return _MentionRejected(_MentionRejectionCode.CONTRADICTORY_CANDIDATE, candidate.target_token)
        mentions.append(
            _AnchoredMention(
                _MentionId(),
                target_by_token[candidate.target_token].datum_id,
                candidate.start,
                candidate.end,
                candidate.source_slice,
                candidate.detector_label,
                candidate.provenance,
            )
        )
    return _DetectedGraph(targets, tuple(mentions))


def _valid_limits(limits: object) -> bool:
    return isinstance(limits, _MentionLimits) and all(
        type(value) is int and value > 0
        for value in (
            limits.max_candidates_per_target,
            limits.max_mentions_per_target,
            limits.max_label_bytes,
            limits.max_source_slice_bytes,
        )
    )


def _valid_bounded_text(value: object, limit: int) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        return len(value.encode("utf-8")) <= limit
    except UnicodeEncodeError:
        return False
