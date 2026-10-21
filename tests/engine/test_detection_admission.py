# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import pytest

from anonymizer.engine.detection.admission import apply_validation_admission, find_duplicate_assigned_aliases
from anonymizer.engine.schemas import (
    AdmissionReason,
    AdmissionResults,
    AdmissionRoute,
    AdmissionStatus,
    DetectionResult,
    EffectiveLabelPolicy,
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
    observation_id: str,
    value: str,
    label: str,
    start: int,
    end: int,
    *,
    route: AdmissionRoute = AdmissionRoute.llm_validation,
) -> Observation:
    return Observation(
        detection_result_id="r1",
        observation_id=observation_id,
        source_text_ref=_source_ref(),
        value=value,
        label=label,
        start_position=start,
        end_position=end,
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
        admission_route=route,
        native_score=0.9,
    )


def _detection_result(text: str, observations: tuple[Observation, ...]) -> DetectionResult:
    return DetectionResult(
        detection_result_id="r1",
        source_view=SourceView(
            source_text_ref=_source_ref(),
            start_position=0,
            end_position=len(text),
        ),
        source_kind="initial_detector",
        source_instance="gliner-pii-detector",
        observations=observations,
    )


def _inventory(text: str, observations: tuple[Observation, ...]) -> ValidationRequestInventory:
    return ValidationRequestInventory(
        request_id="validation-request-1",
        source_text_ref=_source_ref(),
        context_view=SourceView(
            source_text_ref=_source_ref(),
            start_position=0,
            end_position=len(text),
        ),
        source_context=text,
        effective_label_policy=EffectiveLabelPolicy(),
        entries=tuple(
            ValidationInventoryEntry(
                alias=f"c{index}",
                observation_ref=ObservationRef(
                    detection_result_id=observation.detection_result_id,
                    observation_id=observation.observation_id,
                ),
                value=observation.value,
                label=observation.label,
                start_position=observation.start_position,
                end_position=observation.end_position,
            )
            for index, observation in enumerate(observations, start=1)
        ),
    )


def test_validation_decisions_produce_one_terminal_admission_per_observation() -> None:
    text = "Alice 123 Seattle Carol May"
    observations = (
        _observation("o1", "Alice", "first_name", 0, 5),
        _observation("o2", "123", "account_number", 6, 9),
        _observation("o3", "Seattle", "country", 10, 17),
        _observation("o4", "Carol", "first_name", 18, 23),
        _observation("o5", "May", "first_name", 24, 27),
    )
    inventory = _inventory(text, observations)
    decisions = ValidationRequestDecisionSets(
        responses=(
            ValidationRequestDecisionSet(
                request_id=inventory.request_id,
                decisions=(
                    RawValidationDecisionSchema(id="c1", decision=ValidationChoice.keep),
                    RawValidationDecisionSchema(id="c2", decision=ValidationChoice.drop),
                    RawValidationDecisionSchema(
                        id="c3",
                        decision=ValidationChoice.reclass,
                        proposed_label="city",
                    ),
                    RawValidationDecisionSchema(
                        id="c4",
                        decision=ValidationChoice.reclass,
                        proposed_label="   ",
                    ),
                    RawValidationDecisionSchema(id="wrong-request-alias", decision=ValidationChoice.drop),
                ),
            ),
        )
    )

    results = apply_validation_admission(
        detection_result=_detection_result(text, observations),
        inventories=(inventory,),
        decision_sets=decisions,
    )

    assert [(item.status, item.effective_label, item.reason) for item in results] == [
        (AdmissionStatus.accepted, "first_name", AdmissionReason.validator_keep),
        (AdmissionStatus.rejected, None, AdmissionReason.validator_drop),
        (AdmissionStatus.accepted, "city", AdmissionReason.validator_reclass),
        (
            AdmissionStatus.fallback_protected,
            "first_name",
            AdmissionReason.unusable_reclass_fallback,
        ),
        (
            AdmissionStatus.fallback_protected,
            "first_name",
            AdmissionReason.missing_decision_fallback,
        ),
    ]


def test_inventory_reference_to_another_detection_result_is_rejected() -> None:
    text = "Alice"
    observations = (_observation("o1", "Alice", "first_name", 0, 5),)
    inventory = _inventory(text, observations)
    entry = inventory.entries[0].model_copy(
        update={"observation_ref": ObservationRef(detection_result_id="r2", observation_id="o1")}
    )
    inventory = inventory.model_copy(update={"entries": (entry,)})
    decisions = ValidationRequestDecisionSets(
        responses=(ValidationRequestDecisionSet(request_id=inventory.request_id, decisions=()),)
    )

    with pytest.raises(ValueError, match="unknown observation"):
        apply_validation_admission(
            detection_result=_detection_result(text, observations),
            inventories=(inventory,),
            decision_sets=decisions,
        )


def test_duplicate_assigned_alias_is_rejected_at_admission_boundary() -> None:
    text = "Alice"
    observations = (_observation("o1", "Alice", "first_name", 0, 5),)
    inventory = _inventory(text, observations)
    decisions = ValidationRequestDecisionSets(
        responses=(
            ValidationRequestDecisionSet(
                request_id=inventory.request_id,
                decisions=(
                    RawValidationDecisionSchema(id="c1", decision=ValidationChoice.keep),
                    RawValidationDecisionSchema(id="c1", decision=ValidationChoice.drop),
                ),
            ),
        )
    )

    with pytest.raises(ValueError, match="duplicate assigned aliases"):
        apply_validation_admission(
            detection_result=_detection_result(text, observations),
            inventories=(inventory,),
            decision_sets=decisions,
        )


def test_direct_acceptance_does_not_require_a_validation_request() -> None:
    text = "TOKEN-123"
    observation = _observation(
        "o1",
        text,
        "api_key",
        0,
        len(text),
        route=AdmissionRoute.direct_acceptance,
    )

    results = apply_validation_admission(
        detection_result=_detection_result(text, (observation,)),
        inventories=(),
        decision_sets=ValidationRequestDecisionSets(),
    )

    assert results[0].status is AdmissionStatus.accepted
    assert results[0].reason is AdmissionReason.direct_acceptance


@pytest.mark.parametrize(
    ("aliases", "expected"),
    [
        (("c1", "c2"), []),
        (("c2", "c1", "c2", "c1"), ["c1", "c2"]),
        (("unknown", "unknown", "c1"), []),
    ],
)
def test_duplicate_alias_detection_only_counts_assigned_aliases(aliases: tuple[str, ...], expected: list[str]) -> None:
    assert (
        find_duplicate_assigned_aliases(
            decisions=tuple(RawValidationDecisionSchema(id=alias, decision=ValidationChoice.keep) for alias in aliases),
            assigned_aliases=("c1", "c2"),
        )
        == expected
    )


def test_unknown_aliases_are_ignored_and_logged_without_their_contents(caplog: pytest.LogCaptureFixture) -> None:
    text = "Alice"
    observations = (_observation("o1", text, "first_name", 0, len(text)),)
    inventory = _inventory(text, observations)
    unknown_alias = "sensitive-model-output"
    decisions = ValidationRequestDecisionSets(
        responses=(
            ValidationRequestDecisionSet(
                request_id=inventory.request_id,
                decisions=(
                    RawValidationDecisionSchema(id="c1", decision=ValidationChoice.keep),
                    RawValidationDecisionSchema(id=unknown_alias, decision=ValidationChoice.drop),
                    RawValidationDecisionSchema(id=unknown_alias, decision=ValidationChoice.drop),
                ),
            ),
        )
    )

    with caplog.at_level(logging.WARNING, logger="anonymizer.detection.admission"):
        results = apply_validation_admission(
            detection_result=_detection_result(text, observations),
            inventories=(inventory,),
            decision_sets=decisions,
        )

    assert len(results) == 1
    assert results[0].status is AdmissionStatus.accepted
    assert "ignored 2 entries with unknown aliases" in caplog.text
    assert inventory.request_id in caplog.text
    assert unknown_alias not in caplog.text
    assert text not in caplog.text
    assert len(decisions.responses[0].decisions) == 3


def test_admission_results_reject_equivalent_observation_refs() -> None:
    observation = _observation("o1", "Alice", "first_name", 0, 5, route=AdmissionRoute.direct_acceptance)
    result = apply_validation_admission(
        detection_result=_detection_result("Alice", (observation,)),
        inventories=(),
        decision_sets=ValidationRequestDecisionSets(),
    )[0]

    with pytest.raises(ValueError, match="each observation at most once"):
        AdmissionResults(results=(result, result.model_copy(deep=True)))
