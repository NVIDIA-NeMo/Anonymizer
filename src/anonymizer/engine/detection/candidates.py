# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from anonymizer.engine.constants import SOURCE_KIND_AUGMENTER
from anonymizer.engine.schemas import (
    AcceptedOrigin,
    AdmissionResults,
    AdmissionStatus,
    AdmittedCandidate,
    DetectionResult,
    NativeScore,
    ObservationRef,
    SourceTextRef,
)

# Source only breaks ties between candidates with identical source geometry.
# Regex roles are included now so PR #265 can reuse this policy unchanged.
_SOURCE_PRIORITY = {
    "regex_user": 0,
    "regex_builtin": 1,
    "initial_detector": 2,
    SOURCE_KIND_AUGMENTER: 3,
}
_UNKNOWN_SOURCE_PRIORITY = len(_SOURCE_PRIORITY)


def normalize_candidate_label(label: str) -> str:
    """Return the canonical identity form used for candidate labels."""
    return label.strip().casefold()


def materialize_admitted_candidates(
    *,
    canonical_text: str,
    detection_result: DetectionResult,
    admission_results: AdmissionResults,
) -> tuple[AdmittedCandidate, ...]:
    """Turn eligible terminal admissions into source-bound candidates.

    Admission is authoritative: rejected observations are omitted, accepted
    observations contribute provenance, and fallback protection contributes
    lineage without being promoted to accepted authority.
    """
    observations_by_ref = {observation.ref: observation for observation in detection_result.observations}
    admissions_by_ref = {result.observation_ref: result for result in admission_results.results}
    if set(admissions_by_ref) != set(observations_by_ref):
        raise ValueError("admission results must cover every observation exactly once")

    candidates: list[AdmittedCandidate] = []
    for observation in detection_result.observations:
        result = admissions_by_ref[observation.ref]
        if result.route is not observation.admission_route:
            raise ValueError("admission route does not match its observation")
        if result.status is AdmissionStatus.rejected:
            continue
        if canonical_text[observation.start_position : observation.end_position] != observation.value:
            raise ValueError("observation value does not match canonical source offsets")
        label = normalize_candidate_label(result.effective_label or "")
        if not label:
            raise ValueError("retained admission result has no usable effective label")
        observation_ref = observation.ref
        is_explicit = result.status is AdmissionStatus.accepted
        origins = (
            (
                AcceptedOrigin(
                    observation_ref=observation_ref,
                    source_kind=observation.source_kind,
                    source_instance=observation.source_instance,
                    route=observation.admission_route,
                ),
            )
            if is_explicit
            else ()
        )
        scores = (
            (
                NativeScore(
                    observation_ref=observation_ref,
                    source_kind=observation.source_kind,
                    source_instance=observation.source_instance,
                    score=observation.native_score,
                ),
            )
            if is_explicit and observation.native_score is not None
            else ()
        )
        candidates.append(
            AdmittedCandidate(
                source_text_ref=observation.source_text_ref,
                value=observation.value,
                label=label,
                start_position=observation.start_position,
                end_position=observation.end_position,
                lineage=(observation_ref,),
                accepted_origins=origins,
                native_scores=scores,
                has_fallback_protection=result.status is AdmissionStatus.fallback_protected,
            )
        )
    return tuple(candidates)


def coalesce_exact_candidates(candidates: Iterable[AdmittedCandidate]) -> tuple[AdmittedCandidate, ...]:
    """Combine admitted claims with the same source, normalized label, and bounds."""
    grouped: dict[tuple[SourceTextRef, str, int, int], list[AdmittedCandidate]] = defaultdict(list)
    for candidate in candidates:
        normalized_label = normalize_candidate_label(candidate.label)
        if not normalized_label:
            raise ValueError("candidate label must not be empty after normalization")
        grouped[
            (
                candidate.source_text_ref,
                normalized_label,
                candidate.start_position,
                candidate.end_position,
            )
        ].append(candidate)

    coalesced: list[AdmittedCandidate] = []
    for (source_ref, label, start, end), equivalents in grouped.items():
        values = {candidate.value for candidate in equivalents}
        if len(values) != 1:
            raise ValueError("exact candidate identities disagree on source value")
        coalesced.append(
            AdmittedCandidate(
                source_text_ref=source_ref,
                value=values.pop(),
                label=label,
                start_position=start,
                end_position=end,
                lineage=tuple(sorted({ref for candidate in equivalents for ref in candidate.lineage}, key=_ref_key)),
                accepted_origins=tuple(
                    sorted(
                        {origin for candidate in equivalents for origin in candidate.accepted_origins},
                        key=_origin_key,
                    )
                ),
                native_scores=tuple(
                    sorted(
                        {score for candidate in equivalents for score in candidate.native_scores},
                        key=_score_key,
                    )
                ),
                has_fallback_protection=any(candidate.has_fallback_protection for candidate in equivalents),
            )
        )
    return tuple(sorted(coalesced, key=_publication_key))


def reconcile_candidates(candidates: Iterable[AdmittedCandidate]) -> tuple[AdmittedCandidate, ...]:
    """Select one deterministic non-overlapping candidate set per source text."""
    by_source: dict[SourceTextRef, list[AdmittedCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_source[candidate.source_text_ref].append(candidate)

    reconciled: list[AdmittedCandidate] = []
    for source_ref in sorted(by_source, key=_source_ref_key):
        ordered = _rank_candidates(by_source[source_ref])
        accepted: list[AdmittedCandidate] = []
        for candidate in ordered:
            if any(_overlaps(candidate, existing) for existing in accepted):
                continue
            accepted.append(candidate)
        reconciled.extend(sorted(accepted, key=_publication_key))
    return tuple(reconciled)


def representative_source(candidate: AdmittedCandidate) -> str:
    """Return the highest-priority accepted source for the legacy projection."""
    if not candidate.accepted_origins:
        return "fallback"
    return min(
        candidate.accepted_origins,
        key=lambda origin: (
            _SOURCE_PRIORITY.get(origin.source_kind, _UNKNOWN_SOURCE_PRIORITY),
            origin.source_kind,
            origin.source_instance,
            _ref_key(origin.observation_ref),
        ),
    ).source_kind


def _rank_candidates(candidates: list[AdmittedCandidate]) -> list[AdmittedCandidate]:
    by_geometry: dict[tuple[int, int], list[AdmittedCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_geometry[(candidate.start_position, candidate.end_position)].append(candidate)

    identical_rank: dict[int, int] = {}
    for equivalents in by_geometry.values():
        for rank, candidate in enumerate(_rank_identical_geometry(equivalents)):
            identical_rank[id(candidate)] = rank
    return sorted(
        candidates,
        key=lambda candidate: (
            -(candidate.end_position - candidate.start_position),
            candidate.start_position,
            candidate.end_position,
            identical_rank[id(candidate)],
        ),
    )


def _rank_identical_geometry(candidates: list[AdmittedCandidate]) -> list[AdmittedCandidate]:
    """Rank equal-bound alternatives without comparing uncalibrated scores."""
    score_rank: dict[int, float] = {id(candidate): 0.0 for candidate in candidates}
    by_admission_and_priority: dict[tuple[bool, int], list[AdmittedCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_admission_and_priority[(candidate.has_explicit_acceptance, _effective_source_priority(candidate))].append(
            candidate
        )
    for tied in by_admission_and_priority.values():
        domains = {_effective_calibration_domain(candidate) for candidate in tied}
        if len(domains) != 1 or None in domains:
            continue
        for candidate in tied:
            score = _effective_native_score(candidate)
            if score is not None:
                score_rank[id(candidate)] = -score
    return sorted(
        candidates,
        key=lambda candidate: (
            not candidate.has_explicit_acceptance,
            _effective_source_priority(candidate),
            score_rank[id(candidate)],
            normalize_candidate_label(candidate.label),
            tuple(_ref_key(ref) for ref in candidate.lineage),
        ),
    )


def _effective_source_priority(candidate: AdmittedCandidate) -> int:
    if not candidate.accepted_origins:
        return _UNKNOWN_SOURCE_PRIORITY
    return min(
        _SOURCE_PRIORITY.get(origin.source_kind, _UNKNOWN_SOURCE_PRIORITY) for origin in candidate.accepted_origins
    )


def _effective_calibration_domain(candidate: AdmittedCandidate) -> tuple[str, str] | None:
    if not candidate.accepted_origins:
        return None
    priority = _effective_source_priority(candidate)
    domains = {
        (origin.source_kind, origin.source_instance)
        for origin in candidate.accepted_origins
        if _SOURCE_PRIORITY.get(origin.source_kind, _UNKNOWN_SOURCE_PRIORITY) == priority
    }
    return next(iter(domains)) if len(domains) == 1 else None


def _effective_native_score(candidate: AdmittedCandidate) -> float | None:
    domain = _effective_calibration_domain(candidate)
    if domain is None:
        return None
    scores = [score.score for score in candidate.native_scores if (score.source_kind, score.source_instance) == domain]
    return max(scores) if scores else None


def _overlaps(left: AdmittedCandidate, right: AdmittedCandidate) -> bool:
    return left.start_position < right.end_position and right.start_position < left.end_position


def _source_ref_key(source_ref: SourceTextRef) -> tuple[str, str]:
    return source_ref.record_id, source_ref.field_id


def _ref_key(observation_ref: ObservationRef) -> tuple[str, str]:
    return observation_ref.detection_result_id, observation_ref.observation_id


def _origin_key(origin: AcceptedOrigin) -> tuple[int, str, str, tuple[str, str]]:
    return (
        _SOURCE_PRIORITY.get(origin.source_kind, _UNKNOWN_SOURCE_PRIORITY),
        origin.source_kind,
        origin.source_instance,
        _ref_key(origin.observation_ref),
    )


def _score_key(score: NativeScore) -> tuple[str, str, tuple[str, str], float]:
    return score.source_kind, score.source_instance, _ref_key(score.observation_ref), -score.score


def _publication_key(
    candidate: AdmittedCandidate,
) -> tuple[tuple[str, str], int, int, str, tuple[tuple[str, str], ...]]:
    return (
        _source_ref_key(candidate.source_text_ref),
        candidate.start_position,
        candidate.end_position,
        normalize_candidate_label(candidate.label),
        tuple(_ref_key(ref) for ref in candidate.lineage),
    )
