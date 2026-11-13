# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anonymizer.engine.schemas import (
    AcceptedOrigin,
    AdmissionReason,
    AdmissionResult,
    AdmissionRoute,
    AdmissionStatus,
    AdmittedCandidate,
    DetectionResult,
    EffectiveLabelPolicy,
    NativeScore,
    Observation,
    ObservationRef,
    SourceTextRef,
    SourceView,
    ValidationInventoryEntry,
    ValidationRequestInventory,
)


def _source_text_ref(*, record_id: str = "record-1") -> SourceTextRef:
    return SourceTextRef(record_id=record_id, field_id="text")


def _source_view(*, record_id: str = "record-1", end_position: int = 20) -> SourceView:
    return SourceView(
        source_text_ref=_source_text_ref(record_id=record_id),
        start_position=0,
        end_position=end_position,
    )


def _observation(
    *,
    detection_result_id: str = "r1",
    observation_id: str = "o1",
    source_text_ref: SourceTextRef | None = None,
    source_kind: str = "initial_detector",
    source_instance: str = "gliner-pii-detector",
    start_position: int = 0,
    end_position: int = 5,
    native_score: float | None = 0.8,
) -> Observation:
    return Observation(
        detection_result_id=detection_result_id,
        observation_id=observation_id,
        source_text_ref=source_text_ref or _source_text_ref(),
        value="Alice",
        label="first_name",
        start_position=start_position,
        end_position=end_position,
        source_kind=source_kind,
        source_instance=source_instance,
        admission_route=AdmissionRoute.llm_validation,
        native_score=native_score,
    )


def _observation_ref() -> ObservationRef:
    return ObservationRef(detection_result_id="r1", observation_id="o1")


@pytest.mark.parametrize("allowed", [None, [], ["first_name", "email"]])
def test_effective_label_policy_copies_configured_lists(allowed: list[str] | None) -> None:
    excluded = ["email"]
    policy = EffectiveLabelPolicy.from_labels(allowed_labels=allowed, excluded_labels=excluded)

    assert policy.allowed_labels == (tuple(allowed) if allowed is not None else None)
    assert policy.excluded_labels == ("email",)
    excluded.append("first_name")
    if allowed is not None:
        allowed.append("city")
        assert "city" not in (policy.allowed_labels or ())
    assert policy.excluded_labels == ("email",)


def test_observation_ref_is_a_hashable_property_not_a_serialized_field() -> None:
    observation = _observation()
    equivalent_ref = _observation_ref()

    assert observation.ref == equivalent_ref
    assert {observation.ref: observation}[equivalent_ref] is observation
    assert observation.ref != _observation(detection_result_id="r2").ref
    assert "ref" not in observation.model_dump()
    assert "ref" not in Observation.model_json_schema()["properties"]


def test_lifecycle_models_are_frozen_and_reject_extra_fields() -> None:
    source_ref = _source_text_ref()

    with pytest.raises(ValidationError, match="frozen"):
        setattr(source_ref, "record_id", "record-2")

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        SourceTextRef.model_validate({"record_id": "record-1", "field_id": "text", "unexpected": True})


def test_source_view_accepts_empty_view_and_rejects_reversed_geometry() -> None:
    empty_view = SourceView(
        source_text_ref=_source_text_ref(),
        start_position=7,
        end_position=7,
    )
    assert empty_view.start_position == empty_view.end_position

    with pytest.raises(ValidationError, match="source view end must not precede its start"):
        SourceView(
            source_text_ref=_source_text_ref(),
            start_position=8,
            end_position=7,
        )


def test_observation_requires_non_empty_geometry() -> None:
    with pytest.raises(ValidationError, match="observation must cover a non-empty span"):
        _observation(start_position=5, end_position=5)

    with pytest.raises(ValidationError, match="observation value length must match"):
        _observation(end_position=6)


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")])
def test_native_score_models_reject_non_finite_values(invalid_score: float) -> None:
    with pytest.raises(ValidationError, match="finite number"):
        _observation(native_score=invalid_score)

    with pytest.raises(ValidationError, match="finite number"):
        NativeScore(
            observation_ref=_observation_ref(),
            source_kind="initial_detector",
            source_instance="gliner-pii-detector",
            score=invalid_score,
        )


def test_detection_result_allows_successful_empty_result() -> None:
    result = DetectionResult(
        detection_result_id="r1",
        source_view=_source_view(),
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
    )

    assert result.observations == ()


def test_detection_result_enforces_owned_unique_observations() -> None:
    observation = _observation()
    common = {
        "detection_result_id": "r1",
        "source_view": _source_view(),
        "source_kind": "initial_detector",
        "source_instance": "gliner-pii-detector",
        "observations": (observation,),
    }
    cases = [
        ({"observations": (observation, observation)}, "observation IDs must be unique"),
        ({"detection_result_id": "r2"}, "observation belongs to another result"),
        ({"source_view": _source_view(record_id="record-2")}, "observation belongs to another source text"),
        ({"source_kind": "augmenter"}, "observation belongs to another component"),
        (
            {"observations": (_observation(start_position=16, end_position=21),)},
            "observation falls outside its source view",
        ),
    ]
    for changes, error in cases:
        with pytest.raises(ValidationError, match=error):
            DetectionResult.model_validate({**common, **changes})


def test_detection_result_round_trip_preserves_tuple_state() -> None:
    result = DetectionResult(
        detection_result_id="r1",
        source_view=_source_view(),
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
        observations=(_observation(),),
    )

    restored = DetectionResult.model_validate(result.model_dump(mode="json"))

    assert restored == result
    assert isinstance(restored.observations, tuple)


def test_validation_inventory_binds_unique_aliases_to_context() -> None:
    source_ref = _source_text_ref()
    entry = ValidationInventoryEntry(
        alias="c1",
        conflict_group_id="g1",
        observation_ref=_observation_ref(),
        value="Alice",
        label="first_name",
        start_position=0,
        end_position=5,
    )
    inventory = ValidationRequestInventory(
        request_id="validation-1",
        source_text_ref=source_ref,
        context_view=SourceView(
            source_text_ref=source_ref,
            start_position=0,
            end_position=5,
        ),
        source_context="Alice",
        effective_label_policy=EffectiveLabelPolicy(allowed_labels=("first_name",)),
        entries=(entry,),
    )

    assert inventory.entries == (entry,)

    with pytest.raises(ValidationError, match="request-local aliases must be unique"):
        ValidationRequestInventory.model_validate({**inventory.model_dump(mode="json"), "entries": [entry, entry]})

    duplicate_observation = entry.model_copy(update={"alias": "c2"})
    with pytest.raises(ValidationError, match="request-local observation references must be unique"):
        ValidationRequestInventory.model_validate(
            {**inventory.model_dump(mode="json"), "entries": [entry, duplicate_observation]}
        )


def test_validation_inventory_entry_validates_its_own_geometry() -> None:
    for start, end, error in [
        (5, 5, "inventory entry must cover a non-empty span"),
        (0, 6, "inventory entry value length must match"),
    ]:
        with pytest.raises(ValidationError, match=error):
            ValidationInventoryEntry(
                alias="c1",
                observation_ref=_observation_ref(),
                value="Alice",
                label="first_name",
                start_position=start,
                end_position=end,
            )


def test_validation_inventory_requires_matching_source_context() -> None:
    source_ref = _source_text_ref()
    entry = ValidationInventoryEntry(
        alias="c1",
        observation_ref=_observation_ref(),
        value="Alice",
        label="first_name",
        start_position=0,
        end_position=5,
    )
    common = {
        "request_id": "validation-1",
        "source_text_ref": source_ref,
        "effective_label_policy": EffectiveLabelPolicy(),
        "entries": (entry,),
    }

    with pytest.raises(ValidationError, match="context view belongs to another source text"):
        ValidationRequestInventory(
            **common,
            context_view=SourceView(
                source_text_ref=_source_text_ref(record_id="record-2"), start_position=0, end_position=5
            ),
            source_context="Alice",
        )

    with pytest.raises(ValidationError, match="context text does not match its source view"):
        ValidationRequestInventory(
            **common,
            context_view=SourceView(source_text_ref=source_ref, start_position=0, end_position=5),
            source_context="Alice!",
        )

    with pytest.raises(ValidationError, match="inventory entry falls outside its context"):
        ValidationRequestInventory(
            **common,
            context_view=SourceView(source_text_ref=source_ref, start_position=1, end_position=5),
            source_context="lice",
        )


@pytest.mark.parametrize(
    ("alias", "conflict_group_id"),
    [("c0", None), ("c01", None), ("x1", None), ("c1", "g0")],
)
def test_validation_inventory_entry_rejects_invalid_aliases(
    alias: str,
    conflict_group_id: str | None,
) -> None:
    with pytest.raises(ValidationError, match="String should match pattern"):
        ValidationInventoryEntry(
            alias=alias,
            conflict_group_id=conflict_group_id,
            observation_ref=_observation_ref(),
            value="Alice",
            label="first_name",
            start_position=0,
            end_position=5,
        )


@pytest.mark.parametrize(
    ("route", "status", "effective_label", "reason"),
    [
        (
            AdmissionRoute.llm_validation,
            AdmissionStatus.accepted,
            "first_name",
            AdmissionReason.validator_keep,
        ),
        (
            AdmissionRoute.llm_validation,
            AdmissionStatus.fallback_protected,
            "first_name",
            AdmissionReason.missing_decision_fallback,
        ),
        (
            AdmissionRoute.direct_acceptance,
            AdmissionStatus.accepted,
            "first_name",
            AdmissionReason.augmenter_output,
        ),
    ],
)
def test_admission_result_accepts_consistent_terminal_state(
    route: AdmissionRoute,
    status: AdmissionStatus,
    effective_label: str,
    reason: AdmissionReason,
) -> None:
    result = AdmissionResult(
        observation_ref=_observation_ref(),
        route=route,
        status=status,
        effective_label=effective_label,
        reason=reason,
    )

    assert result.status is status


@pytest.mark.parametrize(
    ("route", "status", "effective_label", "reason", "error"),
    [
        (
            AdmissionRoute.llm_validation,
            AdmissionStatus.rejected,
            "first_name",
            AdmissionReason.validator_drop,
            "a rejected result has no effective label",
        ),
        (
            AdmissionRoute.llm_validation,
            AdmissionStatus.accepted,
            None,
            AdmissionReason.validator_keep,
            "a retained result requires an effective label",
        ),
        (
            AdmissionRoute.direct_acceptance,
            AdmissionStatus.accepted,
            "first_name",
            AdmissionReason.validator_keep,
            "validator reason requires LLM validation",
        ),
        (
            AdmissionRoute.llm_validation,
            AdmissionStatus.fallback_protected,
            "first_name",
            AdmissionReason.validator_keep,
            "admission reason does not match its status",
        ),
        (
            AdmissionRoute.llm_validation,
            AdmissionStatus.accepted,
            "first_name",
            AdmissionReason.augmenter_output,
            "direct reason requires direct acceptance",
        ),
    ],
)
def test_admission_result_rejects_inconsistent_terminal_state(
    route: AdmissionRoute,
    status: AdmissionStatus,
    effective_label: str | None,
    reason: AdmissionReason,
    error: str,
) -> None:
    with pytest.raises(ValidationError, match=error):
        AdmissionResult(
            observation_ref=_observation_ref(),
            route=route,
            status=status,
            effective_label=effective_label,
            reason=reason,
        )


def test_admitted_candidate_requires_matching_acceptance_provenance() -> None:
    observation_ref = _observation_ref()
    origin = AcceptedOrigin(
        observation_ref=observation_ref,
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
        route=AdmissionRoute.llm_validation,
    )
    candidate = AdmittedCandidate(
        source_text_ref=_source_text_ref(),
        value="Alice",
        label="first_name",
        start_position=0,
        end_position=5,
        lineage=(observation_ref,),
        accepted_origins=(origin,),
        has_fallback_protection=False,
    )

    restored = AdmittedCandidate.model_validate(candidate.model_dump(mode="json"))
    assert restored == candidate
    assert isinstance(restored.lineage, tuple)
    assert isinstance(restored.accepted_origins, tuple)
    assert restored.has_explicit_acceptance

    unrelated_ref = ObservationRef(detection_result_id="r1", observation_id="o2")
    unrelated_score = NativeScore(
        observation_ref=unrelated_ref,
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
        score=0.8,
    )
    for changes, error in [
        (
            {"accepted_origins": (origin.model_copy(update={"observation_ref": unrelated_ref}),)},
            "accepted origins must reference candidate lineage",
        ),
        (
            {"lineage": (observation_ref, unrelated_ref), "native_scores": (unrelated_score,)},
            "native scores must reference accepted origins",
        ),
    ]:
        with pytest.raises(ValidationError, match=error):
            AdmittedCandidate.model_validate({**candidate.model_dump(), **changes})


def test_admitted_candidate_requires_matching_value_length_and_protection_state() -> None:
    for end, fallback, error in [
        (6, True, "candidate value length must match"),
        (5, False, "candidate requires an admitted protection state"),
    ]:
        with pytest.raises(ValidationError, match=error):
            AdmittedCandidate(
                source_text_ref=_source_text_ref(),
                value="Alice",
                label="first_name",
                start_position=0,
                end_position=end,
                lineage=(_observation_ref(),),
                has_fallback_protection=fallback,
            )


def test_admitted_candidate_allows_fallback_without_accepted_origin() -> None:
    candidate = AdmittedCandidate(
        source_text_ref=_source_text_ref(),
        value="Alice",
        label="first_name",
        start_position=0,
        end_position=5,
        lineage=(_observation_ref(),),
        has_fallback_protection=True,
    )

    assert candidate.accepted_origins == ()
