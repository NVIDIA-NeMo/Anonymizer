# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import permutations

import pytest

from anonymizer.engine.detection.admission import apply_validation_admission
from anonymizer.engine.detection.candidates import (
    coalesce_exact_candidates,
    materialize_admitted_candidates,
    reconcile_candidates,
    representative_source,
)
from anonymizer.engine.schemas import (
    AcceptedOrigin,
    AdmissionReason,
    AdmissionResult,
    AdmissionResults,
    AdmissionRoute,
    AdmissionStatus,
    AdmittedCandidate,
    DetectionResult,
    EffectiveLabelPolicy,
    NativeScore,
    Observation,
    ObservationRef,
    RawValidationDecisionSchema,
    SourceTextRef,
    SourceView,
    ValidationChoice,
    ValidationInventoryEntry,
    ValidationRequestDecisionSet,
    ValidationRequestDecisionSets,
    ValidationRequestInventory,
)


def _source_ref() -> SourceTextRef:
    return SourceTextRef(record_id="record-1", field_id="text")


def _observation(
    *,
    result_id: str,
    observation_id: str,
    value: str,
    label: str,
    start: int,
    end: int,
    source_kind: str = "initial_detector",
    source_instance: str = "gliner",
    route: AdmissionRoute = AdmissionRoute.llm_validation,
    score: float | None = 0.8,
) -> Observation:
    return Observation(
        detection_result_id=result_id,
        observation_id=observation_id,
        source_text_ref=_source_ref(),
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        source_kind=source_kind,
        source_instance=source_instance,
        admission_route=route,
        native_score=score,
    )


def _result(result_id: str, text: str, observations: tuple[Observation, ...]) -> DetectionResult:
    first = observations[0]
    return DetectionResult(
        detection_result_id=result_id,
        source_view=SourceView(source_text_ref=_source_ref(), start_position=0, end_position=len(text)),
        source_kind=first.source_kind,
        source_instance=first.source_instance,
        observations=observations,
    )


def _admission(
    observation: Observation,
    *,
    status: AdmissionStatus = AdmissionStatus.accepted,
    label: str | None = None,
) -> AdmissionResult:
    if status is AdmissionStatus.accepted:
        reason = (
            AdmissionReason.validator_keep
            if observation.admission_route is AdmissionRoute.llm_validation
            else AdmissionReason.direct_acceptance
        )
    elif status is AdmissionStatus.fallback_protected:
        reason = AdmissionReason.missing_decision_fallback
    else:
        reason = AdmissionReason.validator_drop
    return AdmissionResult(
        observation_ref=ObservationRef(
            detection_result_id=observation.detection_result_id,
            observation_id=observation.observation_id,
        ),
        route=observation.admission_route,
        status=status,
        effective_label=None if status is AdmissionStatus.rejected else (label or observation.label),
        reason=reason,
    )


def _candidate(
    *,
    result_id: str,
    observation_id: str,
    value: str,
    label: str,
    start: int,
    end: int,
    source_kind: str = "initial_detector",
    source_instance: str = "gliner",
    explicit: bool = True,
    score: float | None = 0.8,
) -> AdmittedCandidate:
    ref = ObservationRef(detection_result_id=result_id, observation_id=observation_id)
    origin = AcceptedOrigin(
        observation_ref=ref,
        source_kind=source_kind,
        source_instance=source_instance,
        route=AdmissionRoute.direct_acceptance,
    )
    native_scores = (
        (
            NativeScore(
                observation_ref=ref,
                source_kind=source_kind,
                source_instance=source_instance,
                score=score,
            ),
        )
        if explicit and score is not None
        else ()
    )
    return AdmittedCandidate(
        source_text_ref=_source_ref(),
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        lineage=(ref,),
        accepted_origins=(origin,) if explicit else (),
        native_scores=native_scores,
        has_fallback_protection=not explicit,
    )


def test_materialization_uses_offsets_and_does_not_promote_fallback_provenance() -> None:
    text = "Alice called"
    accepted = _observation(result_id="r1", observation_id="o1", value="Alice", label="FIRST_NAME", start=0, end=5)
    fallback = _observation(result_id="r1", observation_id="o2", value="called", label="verb", start=6, end=12)
    result = _result("r1", text, (accepted, fallback))

    candidates = materialize_admitted_candidates(
        canonical_text=text,
        detection_result=result,
        admission_results=AdmissionResults(
            results=(
                _admission(accepted, label=" first_name "),
                _admission(fallback, status=AdmissionStatus.fallback_protected),
            )
        ),
    )

    assert candidates[0].label == "first_name"
    assert candidates[0].accepted_origins
    assert candidates[1].accepted_origins == ()
    assert candidates[1].has_fallback_protection


def test_materialization_rejects_noncanonical_value() -> None:
    observation = _observation(result_id="r1", observation_id="o1", value="ALICE", label="first_name", start=0, end=5)
    result = _result("r1", "Alice", (observation,))

    with pytest.raises(ValueError, match="canonical source offsets"):
        materialize_admitted_candidates(
            canonical_text="Alice",
            detection_result=result,
            admission_results=AdmissionResults(results=(_admission(observation),)),
        )


def test_exact_identity_coalesces_normalized_labels_and_preserves_origins() -> None:
    detector = _candidate(
        result_id="r1",
        observation_id="o1",
        value="TOKEN",
        label=" token ",
        start=0,
        end=5,
    )
    regex = _candidate(
        result_id="r2",
        observation_id="o1",
        value="TOKEN",
        label="TOKEN",
        start=0,
        end=5,
        source_kind="regex_user",
        source_instance="custom-token",
    )

    result = coalesce_exact_candidates((regex, detector))

    assert len(result) == 1
    assert result[0].label == "token"
    assert {origin.source_kind for origin in result[0].accepted_origins} == {
        "initial_detector",
        "regex_user",
    }
    assert len(result[0].lineage) == 2
    assert representative_source(result[0]) == "regex_user"


def test_longer_coverage_wins_before_source_priority() -> None:
    narrow_regex = _candidate(
        result_id="r1",
        observation_id="o1",
        value="Smith",
        label="last_name",
        start=5,
        end=10,
        source_kind="regex_user",
        source_instance="custom-name",
    )
    broad_augmenter = _candidate(
        result_id="r2",
        observation_id="o1",
        value="john.smith@example.com",
        label="email",
        start=0,
        end=22,
        source_kind="augmenter",
        source_instance="augmenter",
    )

    assert reconcile_candidates((narrow_regex, broad_augmenter)) == (broad_augmenter,)


def test_ineligible_longer_observation_cannot_suppress_allowed_shorter_observation() -> None:
    text = "Alice Smith"
    longer = _observation(
        result_id="r1",
        observation_id="o1",
        value=text,
        label="full_name",
        start=0,
        end=11,
    )
    shorter = _observation(
        result_id="r1",
        observation_id="o2",
        value="Alice",
        label="first_name",
        start=0,
        end=5,
    )
    result = _result("r1", text, (longer, shorter))
    inventory = ValidationRequestInventory(
        request_id="validation-request-1",
        source_text_ref=_source_ref(),
        context_view=result.source_view,
        source_context=text,
        effective_label_policy=EffectiveLabelPolicy(
            allowed_labels=("full_name", "first_name"),
            excluded_labels=("full_name",),
        ),
        entries=tuple(
            ValidationInventoryEntry(
                alias=f"c{index}",
                conflict_group_id="g1",
                observation_ref=ObservationRef(
                    detection_result_id=observation.detection_result_id,
                    observation_id=observation.observation_id,
                ),
                value=observation.value,
                label=observation.label,
                start_position=observation.start_position,
                end_position=observation.end_position,
            )
            for index, observation in enumerate((longer, shorter), start=1)
        ),
    )
    admissions = apply_validation_admission(
        effective_label_policy=inventory.effective_label_policy,
        detection_result=result,
        inventories=(inventory,),
        decision_sets=ValidationRequestDecisionSets(
            responses=(
                ValidationRequestDecisionSet(
                    request_id=inventory.request_id,
                    decisions=(
                        RawValidationDecisionSchema(id="c1", decision=ValidationChoice.keep),
                        RawValidationDecisionSchema(id="c2", decision=ValidationChoice.keep),
                    ),
                ),
            )
        ),
    )

    assert admissions[0].status is AdmissionStatus.rejected
    candidates = materialize_admitted_candidates(
        canonical_text=text,
        detection_result=result,
        admission_results=AdmissionResults(results=admissions),
    )
    assert reconcile_candidates(candidates) == (candidates[0],)
    assert candidates[0].value == "Alice"


def test_identical_bounds_prefer_explicit_then_source_priority() -> None:
    fallback = _candidate(
        result_id="r1",
        observation_id="o1",
        value="May",
        label="first_name",
        start=0,
        end=3,
        explicit=False,
    )
    augmenter = _candidate(
        result_id="r2",
        observation_id="o1",
        value="May",
        label="city",
        start=0,
        end=3,
        source_kind="augmenter",
        source_instance="augmenter",
    )
    detector = _candidate(
        result_id="r3",
        observation_id="o1",
        value="May",
        label="date",
        start=0,
        end=3,
    )

    assert reconcile_candidates((fallback, augmenter)) == (augmenter,)
    assert reconcile_candidates((augmenter, detector)) == (detector,)


def test_native_score_only_ranks_one_calibration_domain() -> None:
    low_score = _candidate(
        result_id="r1",
        observation_id="o1",
        value="May",
        label="city",
        start=0,
        end=3,
        score=0.4,
    )
    high_score = _candidate(
        result_id="r1",
        observation_id="o2",
        value="May",
        label="first_name",
        start=0,
        end=3,
        score=0.9,
    )
    other_domain = _candidate(
        result_id="r2",
        observation_id="o1",
        value="May",
        label="company",
        start=0,
        end=3,
        source_instance="other-detector",
        score=1.0,
    )

    assert reconcile_candidates((low_score, high_score)) == (high_score,)
    assert reconcile_candidates((low_score, other_domain))[0].label == "city"


def test_reconciliation_is_order_independent_and_keeps_touching_spans() -> None:
    left = _candidate(result_id="r1", observation_id="o1", value="Alice", label="first_name", start=0, end=5)
    right = _candidate(result_id="r1", observation_id="o2", value="Smith", label="last_name", start=5, end=10)
    bridge = _candidate(
        result_id="r2",
        observation_id="o1",
        value="ce Sm",
        label="other",
        start=3,
        end=8,
        source_kind="augmenter",
        source_instance="augmenter",
    )

    expected = (left, right)
    for ordering in permutations((left, right, bridge)):
        assert reconcile_candidates(ordering) == expected
