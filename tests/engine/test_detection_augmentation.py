# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from anonymizer.engine.constants import (
    AUGMENTATION_DETECTION_RESULT_ID,
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTATION_ADMISSION_RESULTS,
    COL_AUGMENTATION_ATTEMPTS,
    COL_AUGMENTATION_DETECTION_RESULT,
    COL_AUGMENTED_CANDIDATES,
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_RECONCILED_CANDIDATES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TEXT,
    RECORD_ID_COLUMN,
    SEED_DETECTION_RESULT_ID,
)
from anonymizer.engine.detection.augmentation import (
    _find_exact_occurrences,
    _ground_augmentation_items,
    _repair_message,
    augment_row,
    augment_row_async,
)
from anonymizer.engine.detection.custom_columns import finalize_entities, merge_and_build_candidates
from anonymizer.engine.detection.detection_workflow import _get_augment_prompt
from anonymizer.engine.detection.postprocess import EntitySpan, TagNotation, build_conflict_aware_tagged_text
from anonymizer.engine.schemas import (
    AcceptedOrigin,
    AdmissionResults,
    AdmissionRoute,
    AdmissionStatus,
    AdmittedCandidate,
    AdmittedCandidates,
    AugmentationAttemptOutcome,
    AugmentationAttempts,
    AugmentationAttemptTrace,
    AugmentationGroundingFailure,
    AugmentationGroundingFailureCategory,
    AugmentedEntitiesSchema,
    DetectionResult,
    ObservationRef,
    SourceTextRef,
    SourceView,
)


@pytest.mark.parametrize("category", list(AugmentationGroundingFailureCategory))
def test_grounding_failure_categories_round_trip_as_strings(category: AugmentationGroundingFailureCategory) -> None:
    trace = AugmentationAttemptTrace(
        attempt_index=0,
        outcome=AugmentationAttemptOutcome.grounding_failure,
        failures=(AugmentationGroundingFailure(item_index=1, category=category),),
    )
    assert trace.model_dump(mode="json") == {
        "attempt_index": 0,
        "outcome": "grounding_failure",
        "failures": [{"item_index": 1, "category": category.value}],
    }
    restored = AugmentationAttemptTrace.model_validate_json(trace.model_dump_json())
    assert restored.failures[0].category is category


def test_grounding_failure_categories_are_a_closed_set() -> None:
    schema = AugmentationAttemptTrace.model_json_schema()
    assert schema["$defs"]["AugmentationGroundingFailureCategory"]["enum"] == [
        "non_verbatim_value",
        "missing_context",
        "non_verbatim_context",
        "zero_contextual_matches",
    ]
    assert set(schema["properties"]) == {"attempt_index", "outcome", "failures"}
    with pytest.raises(ValidationError, match="failures.0.category"):
        AugmentationAttemptTrace.model_validate(
            {"attempt_index": 0, "outcome": "grounding_failure", "failures": [{"item_index": 1, "category": "typo"}]}
        )


def test_grounding_rejects_contexts_that_match_different_occurrences() -> None:
    text = "A May X B May Y"
    result = _ground_augmentation_items(
        canonical_text=text,
        source_view=_view(text),
        output=_output({"value": "May", "context_before": "A ", "context_after": " Y", "label": "first_name"}),
    )
    assert result.failures[0].category is AugmentationGroundingFailureCategory.zero_contextual_matches
    assert result.occurrences == ()
    assert result.grounded_entities.entities == []
    repair = _repair_message(
        output=_output({"value": "May", "context_before": "A ", "context_after": " Y", "label": "first_name"}),
        failures=result.failures,
    )
    assert json.loads(repair.split("\n")[-1])[0]["failure_categories"] == ["zero_contextual_matches"]


class _FakeFacade:
    def __init__(self, responses: list[AugmentedEntitiesSchema]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kwargs: Any) -> tuple[AugmentedEntitiesSchema, list[Any]]:
        self.calls.append(kwargs)
        return self.responses.pop(0), []

    async def agenerate(self, **kwargs: Any) -> tuple[AugmentedEntitiesSchema, list[Any]]:
        return self.generate(**kwargs)


def _view(text: str) -> SourceView:
    return SourceView(
        source_text_ref=SourceTextRef(record_id="record-1", field_id=COL_TEXT),
        start_position=0,
        end_position=len(text),
    )


def _output(*entities: dict[str, str]) -> AugmentedEntitiesSchema:
    return AugmentedEntitiesSchema.model_validate({"entities": list(entities)})


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    ("bounds", "has_groups"),
    [([], False), ([(0, 5)], False), ([(0, 5), (5, 11)], False), ([(0, 5), (3, 11)], True), ([(0, 11), (6, 11)], True)],
)
def test_group_prompt_guidance_depends_on_seed_spans_not_literal_markers(
    use_async: bool, bounds: list[tuple[int, int]], has_groups: bool
) -> None:
    text = "Santa Clara mentions the literal NAGROUP_g99."
    seeds = [
        EntitySpan(f"s{index}", text[start:end], "city", start, end, 0.9, "detector")
        for index, (start, end) in enumerate(bounds)
    ]
    facade = _FakeFacade([_output()])
    row: dict[str, Any] = {
        RECORD_ID_COLUMN: "record-1",
        COL_TEXT: text,
        COL_INITIAL_TAGGED_TEXT: build_conflict_aware_tagged_text(text, seeds),
        COL_SEED_ENTITIES_JSON: json.dumps([seed.as_dict() for seed in seeds]),
        COL_TAG_NOTATION: TagNotation.xml.value,
    }
    kwargs: dict[str, Any] = {
        "row": row,
        "prompt_template": _get_augment_prompt(data_summary=None, labels=["city", "first_name", "last_name"]),
        "model_alias": "augmenter-model",
        "facade": facade,
        "allowed_entity_labels": None,
        "excluded_entity_labels": [],
    }
    result = asyncio.run(augment_row_async(**kwargs)) if use_async else augment_row(**kwargs)
    prompt = facade.calls[0]["prompt"]
    assert ("A tag named NAGROUP_gN" in prompt) is has_groups
    assert ("never return it as a label" in prompt) is has_groups
    assert "has_overlap_groups" not in result
    assert "NAGROUP_g99" in prompt  # Literal source content is still present, but does not enable the rule.


def test_exact_occurrences_include_overlapping_matches() -> None:
    assert _find_exact_occurrences("aaa", "aa") == (0, 1)


@pytest.mark.parametrize("value", [" Alice", "Alice ", " Alice ", "\tAlice\n"])
def test_value_edge_whitespace_becomes_context_not_part_of_the_span(value: str) -> None:
    text = f"Ask {value} now"
    output = _output({"value": value, "context_before": "Ask ", "context_after": " now", "label": "first_name"})
    facade = _FakeFacade([output])
    row = augment_row(
        row={RECORD_ID_COLUMN: "record-1", COL_TEXT: text},
        prompt_template="Source: {{ __nemo_anonymizer_text_input__ }}",
        model_alias="augmenter-model",
        facade=facade,
        allowed_entity_labels=["first_name"],
        excluded_entity_labels=[],
    )
    assert len(facade.calls) == 1
    observations = DetectionResult.model_validate(row[COL_AUGMENTATION_DETECTION_RESULT]).observations
    start = text.index("Alice")
    assert [(item.value, item.start_position, item.end_position) for item in observations] == [
        ("Alice", start, start + 5)
    ]
    grounded = AugmentedEntitiesSchema.model_validate(row[COL_AUGMENTED_ENTITIES]).entities[0]
    assert grounded.value == "Alice"
    assert grounded.context_before == text[:start]
    assert grounded.context_after == text[start + 5 :]
    assert output.entities[0].value == value


def test_whitespace_only_value_is_an_ungroundable_item() -> None:
    text = "Ask   now"
    result = _ground_augmentation_items(
        canonical_text=text,
        source_view=_view(text),
        output=_output({"value": "   ", "context_before": "Ask", "context_after": "now", "label": "first_name"}),
    )
    assert result.grounded_entities.entities == []
    assert result.occurrences == ()
    assert result.failures == (
        AugmentationGroundingFailure(
            item_index=1,
            category=AugmentationGroundingFailureCategory.non_verbatim_value,
        ),
    )


@pytest.mark.parametrize("excluded", [[], ["first_name"]])
def test_sync_and_async_augmentation_share_preparation_and_results(excluded: list[str]) -> None:
    text = "Ask May now"
    output = _output({"value": "May", "context_before": "Ask ", "context_after": " now", "label": "first_name"})
    sync_facade = _FakeFacade([output])
    async_facade = _FakeFacade([output])
    kwargs: dict[str, Any] = {
        "prompt_template": "Source: {{ __nemo_anonymizer_text_input__ }}",
        "model_alias": "augmenter-model",
        "allowed_entity_labels": ["first_name"],
        "excluded_entity_labels": excluded,
        "system_prompt": "Augmentation rules",
    }

    sync_result = augment_row(row={RECORD_ID_COLUMN: "record-1", COL_TEXT: text}, facade=sync_facade, **kwargs)
    async_result = asyncio.run(
        augment_row_async(row={RECORD_ID_COLUMN: "record-1", COL_TEXT: text}, facade=async_facade, **kwargs)
    )

    assert sync_result == async_result
    assert len(sync_facade.calls) == len(async_facade.calls) == 1
    for field in ("prompt", "system_prompt", "purpose"):
        assert sync_facade.calls[0][field] == async_facade.calls[0][field]
    assert "Source: Ask May now" in sync_facade.calls[0]["prompt"]
    result = DetectionResult.model_validate(sync_result[COL_AUGMENTATION_DETECTION_RESULT])
    assert result.source_view == _view(text)
    assert result.detection_result_id == AUGMENTATION_DETECTION_RESULT_ID
    assert AUGMENTATION_DETECTION_RESULT_ID != SEED_DETECTION_RESULT_ID
    admissions = AdmissionResults.model_validate(sync_result[COL_AUGMENTATION_ADMISSION_RESULTS])
    expected = AdmissionStatus.rejected if excluded else AdmissionStatus.accepted
    assert admissions.results[0].status is expected


def test_grounding_materializes_every_exact_contextual_occurrence() -> None:
    text = "She told May to stop laughing. They told May to stop laughing again."
    output = _output(
        {
            "value": "May",
            "context_before": "told ",
            "context_after": " to stop laughing",
            "label": "first_name",
        }
    )

    result = _ground_augmentation_items(
        canonical_text=text,
        source_view=_view(text),
        output=output,
    )

    assert result.failures == ()
    assert [
        (text[item.start_position : item.end_position], item.start_position, item.end_position)
        for item in result.occurrences
    ] == [
        ("May", 9, 12),
        ("May", 41, 44),
    ]


def test_empty_context_is_valid_only_when_value_is_the_entire_view() -> None:
    result = _ground_augmentation_items(
        canonical_text="May",
        source_view=_view("May"),
        output=_output({"value": "May", "context_before": "", "context_after": "", "label": "first_name"}),
    )
    assert result.failures == ()
    assert result.grounded_entities.entities[0].value == "May"
    assert [(item.start_position, item.end_position) for item in result.occurrences] == [(0, 3)]

    result = _ground_augmentation_items(
        canonical_text="Ask May now",
        source_view=_view("Ask May now"),
        output=_output({"value": "May", "context_before": "", "context_after": "", "label": "first_name"}),
    )
    assert result.occurrences == ()
    assert result.grounded_entities.entities == []
    assert [failure.category for failure in result.failures] == ["missing_context"]


@pytest.mark.parametrize(
    ("entity", "category"),
    [
        (
            {"value": "MAY", "context_before": "Ask ", "context_after": " now", "label": "first_name"},
            "non_verbatim_value",
        ),
        (
            {"value": "May", "context_before": "Tell ", "context_after": " now", "label": "first_name"},
            "non_verbatim_context",
        ),
        (
            {"value": "May", "context_before": "Ask ", "context_after": " later", "label": "first_name"},
            "non_verbatim_context",
        ),
    ],
)
def test_grounding_rejects_nonverbatim_output(entity: dict[str, str], category: str) -> None:
    text = "Ask May now"
    result = _ground_augmentation_items(canonical_text=text, source_view=_view(text), output=_output(entity))
    assert result.occurrences == ()
    assert result.grounded_entities.entities == []
    assert [failure.category for failure in result.failures] == [category]


def test_augmentation_retries_failed_items_once_and_publishes_lifecycle_state() -> None:
    bad = _output({"value": "MAY", "context_before": "Ask ", "context_after": " now", "label": "first_name"})
    good = _output({"value": "May", "context_before": "Ask ", "context_after": " now", "label": "first_name"})
    facade = _FakeFacade([bad, good])
    row: dict[str, Any] = {RECORD_ID_COLUMN: "record-1", COL_TEXT: "Ask May now"}

    result = augment_row(
        row=row,
        prompt_template="Source: {{ __nemo_anonymizer_text_input__ }}",
        model_alias="augmenter-model",
        facade=facade,
        allowed_entity_labels=["first_name"],
        excluded_entity_labels=[],
    )

    assert len(facade.calls) == 2
    repair_prompt = facade.calls[1]["prompt"]
    correction = repair_prompt[repair_prompt.index("Correction required.") :]
    assert "non_verbatim_value" in correction
    assert "MAY" in correction
    assert result[COL_AUGMENTED_ENTITIES] == good.model_dump(mode="json")
    assert DetectionResult.model_validate(result[COL_AUGMENTATION_DETECTION_RESULT]).observations[0].value == "May"
    admissions = AdmissionResults.model_validate(result[COL_AUGMENTATION_ADMISSION_RESULTS])
    assert admissions.results[0].status is AdmissionStatus.accepted
    assert len(AdmittedCandidates.model_validate(result[COL_AUGMENTED_CANDIDATES]).candidates) == 1
    attempts = AugmentationAttempts.model_validate(result[COL_AUGMENTATION_ATTEMPTS])
    assert [
        (item.outcome.value, tuple(failure.category for failure in item.failures)) for item in attempts.attempts
    ] == [
        ("grounding_failure", ("non_verbatim_value",)),
        ("success", ()),
    ]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("retry_result", ["corrected", "invalid", "omitted", "partial"])
def test_targeted_retry_preserves_grounded_items_and_records_only_safe_diagnostics(
    use_async: bool, retry_result: str, caplog: pytest.LogCaptureFixture
) -> None:
    text = "Ask Alice now. Ask Bob later. Ask Carol soon."
    alice = {"value": "Alice", "context_before": "Ask ", "context_after": " now.", "label": "first_name"}
    bob = {"value": "Bob", "context_before": "Ask ", "context_after": " later.", "label": "first_name"}
    carol = {"value": "Carol", "context_before": "Ask ", "context_after": " soon.", "label": "first_name"}
    bad_bob = {**bob, "value": "BOB"}
    bad_carol = {**carol, "value": "CAROL"}
    corrections = {
        "corrected": _output(bob, carol),
        "invalid": _output(bad_carol, bad_bob),
        "omitted": _output(),
        "partial": _output(carol, bad_bob),
    }
    facade = _FakeFacade([_output(alice, bad_bob, bad_carol), corrections[retry_result]])
    seed_state = {"entities": [{"value": "Ask", "label": "organization", "start_position": 0, "end_position": 3}]}
    row: dict[str, Any] = {
        RECORD_ID_COLUMN: "record-1",
        COL_TEXT: text,
        COL_SEED_ENTITIES_JSON: json.dumps(seed_state["entities"]),
    }
    kwargs: dict[str, Any] = {
        "row": row,
        "prompt_template": "Source: {{ __nemo_anonymizer_text_input__ }}",
        "model_alias": "augmenter-model",
        "facade": facade,
        "allowed_entity_labels": ["first_name"],
        "excluded_entity_labels": [],
    }
    result = asyncio.run(augment_row_async(**kwargs)) if use_async else augment_row(**kwargs)
    assert json.loads(result[COL_SEED_ENTITIES_JSON]) == seed_state["entities"]
    assert len(facade.calls) == 2
    repair_prompt = facade.calls[1]["prompt"]
    correction = repair_prompt[repair_prompt.index("Correction required.") :]
    # Inspect only repair data: the original prompt necessarily contains the source text.
    repair_items, _ = json.JSONDecoder().raw_decode(correction.split("\n", 1)[1])
    assert repair_items == [
        {**bad_bob, "failure_categories": ["non_verbatim_value"]},
        {**bad_carol, "failure_categories": ["non_verbatim_value"]},
    ]
    expected = {"corrected": [alice, bob, carol], "invalid": [alice], "omitted": [alice], "partial": [alice, carol]}
    assert result[COL_AUGMENTED_ENTITIES] == _output(*expected[retry_result]).model_dump(mode="json")
    detection = DetectionResult.model_validate(result[COL_AUGMENTATION_DETECTION_RESULT])
    assert [item.value for item in detection.observations] == [item["value"] for item in expected[retry_result]]
    assert [item.observation_id for item in detection.observations] == [
        f"o{i}" for i in range(1, len(expected[retry_result]) + 1)
    ]
    assert all(
        item.status is AdmissionStatus.accepted
        for item in AdmissionResults.model_validate(result[COL_AUGMENTATION_ADMISSION_RESULTS]).results
    )
    trace = result[COL_AUGMENTATION_ATTEMPTS]
    assert trace["attempts"][0]["outcome"] == "partial_grounding"
    assert trace["attempts"][0]["failures"] == [
        {"item_index": 2, "category": "non_verbatim_value"},
        {"item_index": 3, "category": "non_verbatim_value"},
    ]
    if retry_result in {"invalid", "partial"}:
        assert trace["attempts"][1]["outcome"] == (
            "partial_grounding" if retry_result == "partial" else "grounding_failure"
        )
        assert trace["attempts"][1]["failures"] == [
            {"item_index": index, "category": "non_verbatim_value"}
            for index in ([1, 2] if retry_result == "invalid" else [2])
        ]
    else:
        assert trace["attempts"][1]["outcome"] == "success"
        assert trace["attempts"][1]["failures"] == []
    for value in ("Alice", "Bob", "Carol", "BOB", "CAROL"):
        assert value not in json.dumps(trace)
        assert value not in caplog.text


@pytest.mark.parametrize("use_async", [False, True])
def test_all_grounding_failures_return_empty_augmentation(use_async: bool) -> None:
    invalid = _output({"value": "MAY", "context_before": "Ask ", "context_after": " now", "label": "first_name"})
    kwargs: dict[str, Any] = {
        "row": {RECORD_ID_COLUMN: "record-1", COL_TEXT: "Ask May now"},
        "prompt_template": "Source: {{ __nemo_anonymizer_text_input__ }}",
        "model_alias": "augmenter-model",
        "facade": _FakeFacade([invalid, invalid]),
        "allowed_entity_labels": ["first_name"],
        "excluded_entity_labels": [],
    }
    result = asyncio.run(augment_row_async(**kwargs)) if use_async else augment_row(**kwargs)
    assert result[COL_AUGMENTED_ENTITIES] == {"entities": []}
    assert DetectionResult.model_validate(result[COL_AUGMENTATION_DETECTION_RESULT]).observations == ()
    assert [attempt["outcome"] for attempt in result[COL_AUGMENTATION_ATTEMPTS]["attempts"]] == [
        "grounding_failure",
        "grounding_failure",
    ]


def test_ungroundable_augmentation_preserves_admitted_gliner_candidates_through_merge() -> None:
    text = "Ask May now"
    ref = ObservationRef(detection_result_id=SEED_DETECTION_RESULT_ID, observation_id="o1")
    seed = AdmittedCandidate(
        source_text_ref=_view(text).source_text_ref,
        value="May",
        label="first_name",
        start_position=4,
        end_position=7,
        lineage=(ref,),
        accepted_origins=(
            AcceptedOrigin(
                observation_ref=ref,
                source_kind="initial_detector",
                source_instance="gliner",
                route=AdmissionRoute.llm_validation,
            ),
        ),
        has_fallback_protection=False,
    )
    row: dict[str, Any] = {
        RECORD_ID_COLUMN: "record-1",
        COL_TEXT: text,
        COL_ADMITTED_CANDIDATES: AdmittedCandidates(candidates=(seed,)).model_dump(mode="json"),
    }
    invalid = _output({"value": "MAY", "context_before": "Ask ", "context_after": " now", "label": "first_name"})
    augment_row(
        row=row,
        prompt_template="Source: {{ __nemo_anonymizer_text_input__ }}",
        model_alias="augmenter-model",
        facade=_FakeFacade([invalid, invalid]),
        allowed_entity_labels=["first_name"],
        excluded_entity_labels=[],
    )
    merge_and_build_candidates(row)
    assert AdmittedCandidates.model_validate(row[COL_RECONCILED_CANDIDATES]).candidates == (seed,)
    assert [
        (item["value"], item["start_position"], item["end_position"]) for item in row[COL_MERGED_ENTITIES]["entities"]
    ] == [("May", 4, 7)]


def test_corrections_receive_ids_after_merging_in_source_order() -> None:
    text = "Ask Alice now. Ask Bob later."
    alice = {"value": "Alice", "context_before": "Ask ", "context_after": " now.", "label": "first_name"}
    bob = {"value": "Bob", "context_before": "Ask ", "context_after": " later.", "label": "first_name"}
    row = augment_row(
        row={RECORD_ID_COLUMN: "record-1", COL_TEXT: text},
        prompt_template="Source: {{ __nemo_anonymizer_text_input__ }}",
        model_alias="augmenter-model",
        facade=_FakeFacade([_output(bob, {**alice, "value": "ALICE"}), _output(alice)]),
        allowed_entity_labels=["first_name"],
        excluded_entity_labels=[],
    )
    observations = DetectionResult.model_validate(row[COL_AUGMENTATION_DETECTION_RESULT]).observations
    assert [(item.observation_id, item.value) for item in observations] == [("o1", "Alice"), ("o2", "Bob")]


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("failure_kind", ["transport", "schema"])
@pytest.mark.parametrize("on_retry", [False, True])
def test_real_augmentation_failures_still_raise(use_async: bool, failure_kind: str, on_retry: bool) -> None:
    class FailingFacade(_FakeFacade):
        def generate(self, **kwargs: Any) -> tuple[AugmentedEntitiesSchema, list[Any]]:
            if self.responses:
                return super().generate(**kwargs)
            if failure_kind == "transport":
                raise ConnectionError("transport unavailable")
            return _output({"value": "missing required fields"}), []

    invalid = _output({"value": "MAY", "context_before": "Ask ", "context_after": " now", "label": "first_name"})
    kwargs: dict[str, Any] = {
        "row": {RECORD_ID_COLUMN: "record-1", COL_TEXT: "Ask May now"},
        "prompt_template": "Source: {{ __nemo_anonymizer_text_input__ }}",
        "model_alias": "augmenter-model",
        "facade": FailingFacade([invalid] if on_retry else []),
        "allowed_entity_labels": ["first_name"],
        "excluded_entity_labels": [],
    }
    with pytest.raises(ConnectionError if failure_kind == "transport" else ValidationError):
        if use_async:
            asyncio.run(augment_row_async(**kwargs))
        else:
            augment_row(**kwargs)


def test_out_of_scope_augmentation_label_is_rejected_without_retry() -> None:
    output = _output({"value": "May", "context_before": "Ask ", "context_after": " now", "label": "month"})
    facade = _FakeFacade([output])
    row: dict[str, Any] = {RECORD_ID_COLUMN: "record-1", COL_TEXT: "Ask May now"}

    result = augment_row(
        row=row,
        prompt_template="Source: {{ __nemo_anonymizer_text_input__ }}",
        model_alias="augmenter-model",
        facade=facade,
        allowed_entity_labels=["first_name"],
        excluded_entity_labels=[],
    )

    assert len(facade.calls) == 1
    admissions = AdmissionResults.model_validate(result[COL_AUGMENTATION_ADMISSION_RESULTS])
    assert admissions.results[0].status is AdmissionStatus.rejected
    assert AdmittedCandidates.model_validate(result[COL_AUGMENTED_CANDIDATES]).candidates == ()


def test_merge_consumes_grounded_augmentation_candidates() -> None:
    output = _output({"value": "May", "context_before": "Ask ", "context_after": " now", "label": "first_name"})
    row: dict[str, Any] = {
        RECORD_ID_COLUMN: "record-1",
        COL_TEXT: "Ask May now",
        COL_ADMITTED_CANDIDATES: {"candidates": []},
    }
    augment_row(
        row=row,
        prompt_template="Source: {{ __nemo_anonymizer_text_input__ }}",
        model_alias="augmenter-model",
        facade=_FakeFacade([output]),
        allowed_entity_labels=["first_name"],
        excluded_entity_labels=[],
    )

    merge_and_build_candidates(row)

    assert [
        (item["value"], item["start_position"], item["source"]) for item in row[COL_MERGED_ENTITIES]["entities"]
    ] == [("May", 4, "augmenter")]
    assert len(AdmittedCandidates.model_validate(row[COL_RECONCILED_CANDIDATES]).candidates) == 1


def test_finalization_does_not_propagate_equal_values_or_split_full_names() -> None:
    text = "John Smith arrived. Later, John left."
    output = _output(
        {
            "value": "John Smith",
            "context_before": "",
            "context_after": " arrived.",
            "label": "full_name",
        }
    )
    row: dict[str, Any] = {
        RECORD_ID_COLUMN: "record-1",
        COL_TEXT: text,
        COL_ADMITTED_CANDIDATES: {"candidates": []},
    }
    augment_row(
        row=row,
        prompt_template="Source: {{ __nemo_anonymizer_text_input__ }}",
        model_alias="augmenter-model",
        facade=_FakeFacade([output]),
        allowed_entity_labels=["full_name", "first_name", "last_name"],
        excluded_entity_labels=[],
    )
    merge_and_build_candidates(row)

    finalize_entities(row)

    assert [
        (item["value"], item["label"], item["start_position"], item["end_position"])
        for item in row[COL_DETECTED_ENTITIES]["entities"]
    ] == [("John Smith", "full_name", 0, 10)]
