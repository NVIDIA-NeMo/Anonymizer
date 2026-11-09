# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from anonymizer.config.regex import BuiltinRegex, RegexRule
from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTATION_ADMISSION_RESULTS,
    COL_AUGMENTED_CANDIDATES,
    COL_DETECTED_ENTITIES,
    COL_RAW_DETECTED,
    COL_REGEX_DETECTION_RESULTS,
    COL_SEED_DETECTION_RESULT,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    COL_VALIDATION_SKELETON,
    RECORD_ID_COLUMN,
    _jinja,
)
from anonymizer.engine.detection.augmentation import augment_row, augment_row_async
from anonymizer.engine.detection.backend import load_seed_detection_results
from anonymizer.engine.detection.chunked_validation import prepare_validation_requests
from anonymizer.engine.detection.custom_columns import (
    admit_validation_decisions,
    apply_validation_to_seed_entities,
    finalize_entities,
    merge_and_build_candidates,
    parse_detected_entities,
)
from anonymizer.engine.detection.postprocess import TagNotation
from anonymizer.engine.detection.regex_detection import (
    build_regex_detection_results,
    detect_regex_entities,
    resolve_regex_rules,
)
from anonymizer.engine.schemas import (
    AdmissionReason,
    AdmissionResults,
    AdmissionStatus,
    AdmittedCandidates,
    AugmentedEntitiesSchema,
    DetectionResult,
    DetectionResultBatch,
    EffectiveLabelPolicy,
    RawValidationDecisionSchema,
    SourceTextRef,
    ValidationRequestDecisionSet,
    ValidationRequestDecisionSets,
)


def _row(
    text: str,
    rules: list[RegexRule | BuiltinRegex],
    *,
    detector: list[dict[str, Any]] | None = None,
    constrained: list[str] | None = None,
    builtins: bool = False,
) -> dict[str, Any]:
    labels = sorted({rule.label for rule in rules})
    matches = detect_regex_entities(
        text, rules=resolve_regex_rules(labels=labels, builtin_regexes=builtins, rules=[*rules])
    )
    row = {
        COL_TEXT: text,
        RECORD_ID_COLUMN: "synthetic-record",
        COL_RAW_DETECTED: json.dumps({"entities": detector or []}),
        COL_REGEX_DETECTION_RESULTS: [
            result.model_dump(mode="json")
            for result in build_regex_detection_results(
                text=text,
                source_text_ref=SourceTextRef(record_id="synthetic-record", field_id=COL_TEXT),
                matches=matches,
            )
        ],
    }
    return parse_detected_entities(
        row, detection_result_id="r1", source_instance="gliner", regex_constrained_entity_labels=constrained
    )


def _admit(
    row: dict[str, Any],
    decisions: list[dict[str, Any]],
    *,
    allowed: list[str] | None = None,
    excluded: list[str] | None = None,
    constrained: list[str] | None = None,
) -> dict[str, Any]:
    requests = prepare_validation_requests(
        detection_result=load_seed_detection_results(row),
        canonical_text=row[COL_TEXT],
        max_entities_per_call=100,
        excerpt_window_chars=100,
        single_chunk_full_text=True,
        notation=TagNotation.xml,
        prompt_template=_jinja(COL_VALIDATION_SKELETON),
        effective_label_policy=EffectiveLabelPolicy.from_labels(allowed_labels=allowed, excluded_labels=excluded),
    )
    row[COL_VALIDATION_INVENTORIES] = [request.inventory.model_dump(mode="json") for request in requests]
    row[COL_VALIDATION_DECISIONS] = ValidationRequestDecisionSets(
        responses=tuple(
            ValidationRequestDecisionSet(
                request_id=request.inventory.request_id,
                decisions=tuple(RawValidationDecisionSchema.model_validate(decision) for decision in decisions),
            )
            for request in requests
        )
    ).model_dump(mode="json")
    admit_validation_decisions(
        row, allowed_entity_labels=allowed, excluded_entity_labels=excluded, regex_constrained_entity_labels=constrained
    )
    return apply_validation_to_seed_entities(row)


def _finalize(row: dict[str, Any], *, constrained: list[str] | None = None) -> list[dict[str, Any]]:
    row.setdefault(COL_AUGMENTED_CANDIDATES, AdmittedCandidates().model_dump(mode="json"))
    merge_and_build_candidates(row)
    finalize_entities(row)
    return row[COL_DETECTED_ENTITIES]["entities"]


@pytest.mark.parametrize("decision", ["keep", "drop", None])
def test_direct_regex_and_detector_claims_are_admitted_independently(decision: str | None) -> None:
    row = _row(
        "allow:ABC deny:ABC",
        [RegexRule(label="token", pattern=r"(?<=allow:)ABC", validate_matches_with_llm=False)],
        detector=[{"text": "ABC", "label": "token", "start": 6, "end": 9, "score": 0.8}],
    )
    _admit(row, [] if decision is None else [{"id": "c1", "decision": decision}])
    inventories = row[COL_VALIDATION_INVENTORIES]
    assert len(inventories) == 1
    assert len(inventories[0]["entries"]) == 1
    assert inventories[0]["entries"][0]["observation_ref"]["detection_result_id"] == "r1"
    candidates = AdmittedCandidates.model_validate(row[COL_ADMITTED_CANDIDATES]).candidates
    assert len(candidates) == 1
    assert {origin.source_kind for origin in candidates[0].accepted_origins} == (
        {"regex_user", "initial_detector"} if decision == "keep" else {"regex_user"}
    )
    assert [(entity["start_position"], entity["end_position"]) for entity in _finalize(row)] == [(6, 9)]


@pytest.mark.parametrize(
    ("label", "value"),
    [
        ("credit_debit_card", "4111 1111 1111 1111"),
        ("email", "synthetic@example.org"),
        ("ipv4", "192.0.2.10"),
        ("ipv6", "2001:db8::1"),
        ("mac_address", "00:1A:2B:3C:4D:5E"),
        ("url", "https://example.org/private"),
    ],
)
@pytest.mark.parametrize("validate_matches", [False, True])
@pytest.mark.parametrize("decision", ["keep", "drop"])
@pytest.mark.parametrize("excluded", [False, True])
def test_builtin_validation_and_exclusion_admission(
    label: str, value: str, validate_matches: bool, decision: str, excluded: bool
) -> None:
    text = f"Start {value} end"
    row = _row(
        text,
        [BuiltinRegex(label=label, validate_matches_with_llm=validate_matches, detect_additional_matches=False)],
        builtins=True,
        constrained=[label],
    )
    _admit(
        row,
        [{"id": "c1", "decision": decision}],
        allowed=[label],
        excluded=[label] if excluded else [],
        constrained=[label],
    )
    entities = _finalize(row)
    expected = not excluded and (not validate_matches or decision == "keep")
    assert [(e["value"], e["label"], e["start_position"], e["end_position"]) for e in entities] == (
        [(value, label, len("Start "), len("Start ") + len(value))] if expected else []
    )
    inventories = row[COL_VALIDATION_INVENTORIES]
    # The helper injects matches before policy filtering; admission must still
    # reject excluded labels, even if a validator returned "keep" for them.
    assert sum(len(inventory["entries"]) for inventory in inventories) == int(validate_matches)


@pytest.mark.parametrize(("allowed", "excluded"), [(None, ["token"]), (["organization"], []), ([], [])])
def test_direct_regex_policy_is_enforced_without_validation_inventories(
    allowed: list[str] | None, excluded: list[str]
) -> None:
    row = _row("ABC", [RegexRule(label="token", pattern="ABC", validate_matches_with_llm=False)])
    _admit(row, [], allowed=allowed, excluded=excluded)
    assert row[COL_VALIDATION_INVENTORIES] == []
    assert _finalize(row) == []
    assert row[COL_ADMISSION_RESULTS]["results"][0]["reason"] == "ineligible_label"


def test_overlapping_rules_keep_independent_decisions_until_reconciliation() -> None:
    row = _row(
        "John Doe", [RegexRule(label="full_name", pattern="John Doe"), RegexRule(label="first_name", pattern="John")]
    )
    _admit(row, [{"id": "c1", "decision": "keep"}, {"id": "c2", "decision": "drop"}])
    entries = row[COL_VALIDATION_INVENTORIES][0]["entries"]
    assert [entry["value"] for entry in entries] == ["John", "John Doe"]
    assert {entry["conflict_group_id"] for entry in entries} == {"g1"}
    assert [(entity["value"], entity["label"]) for entity in _finalize(row)] == [("John", "first_name")]


def test_duplicate_rules_are_not_coalesced_before_their_decisions() -> None:
    row = _row("ABC", [RegexRule(label="token", pattern="ABC"), RegexRule(label="token", pattern="A.C")])
    batch = load_seed_detection_results(row)
    assert len(batch.observations) == 2
    assert len({observation.ref for observation in batch.observations}) == 2
    _admit(row, [{"id": "c1", "decision": "drop"}, {"id": "c2", "decision": "keep"}])
    candidate = AdmittedCandidates.model_validate(row[COL_ADMITTED_CANDIDATES]).candidates[0]
    assert len(candidate.lineage) == len(candidate.accepted_origins) == 1
    assert len(_finalize(row)) == 1


@pytest.mark.parametrize("regex_decision", ["keep", "drop", "reclass", None])
def test_constrained_reclassification_requires_accepted_unchanged_regex_evidence(regex_decision: str | None) -> None:
    row = _row(
        "ABC",
        [RegexRule(label="token", pattern="ABC", detect_additional_matches=False)],
        detector=[{"text": "ABC", "label": "organization", "start": 0, "end": 3}],
        constrained=["token"],
    )
    # Equal-span alternatives are ordered by label: organization, then token.
    decisions: list[dict[str, Any]] = [{"id": "c1", "decision": "reclass", "proposed_label": "token"}]
    if regex_decision is not None:
        decisions.append({"id": "c2", "decision": regex_decision, "proposed_label": "organization"})
    _admit(row, decisions, constrained=["token"])
    admissions = AdmissionResults.model_validate(row[COL_ADMISSION_RESULTS]).results
    detector_admission = next(result for result in admissions if result.observation_ref.detection_result_id == "r1")
    assert detector_admission.status is (
        AdmissionStatus.accepted if regex_decision == "keep" else AdmissionStatus.rejected
    )
    if regex_decision != "keep":
        assert detector_admission.reason is AdmissionReason.missing_regex_evidence
    if regex_decision is None:
        regex_admission = next(result for result in admissions if result.observation_ref.detection_result_id != "r1")
        assert regex_admission.status is AdmissionStatus.fallback_protected
    _finalize(row, constrained=["token"])


def test_regex_constrained_gate_runs_before_overlap_selection() -> None:
    row = _row(
        "ABC DEF",
        [RegexRule(label="token", pattern="ABC", detect_additional_matches=False, validate_matches_with_llm=False)],
        detector=[
            {"text": "ABC DEF", "label": "organization", "start": 0, "end": 7},
            {"text": "DEF", "label": "first_name", "start": 4, "end": 7},
        ],
        constrained=["token"],
    )
    _admit(
        row,
        [{"id": "c1", "decision": "reclass", "proposed_label": "token"}, {"id": "c2", "decision": "keep"}],
        constrained=["token"],
    )
    assert [(entity["value"], entity["label"]) for entity in _finalize(row, constrained=["token"])] == [
        ("ABC", "token"),
        ("DEF", "first_name"),
    ]


def test_accepted_direct_regex_uses_existing_same_span_source_priority() -> None:
    row = _row(
        "ABC",
        [RegexRule(label="token", pattern="ABC", validate_matches_with_llm=False)],
        detector=[{"text": "ABC", "label": "organization", "start": 0, "end": 3, "score": 0.99}],
    )
    _admit(row, [{"id": "c1", "decision": "keep"}])
    assert [(entity["value"], entity["label"], entity["source"]) for entity in _finalize(row)] == [
        ("ABC", "token", "regex_user")
    ]
    assert "NAGROUP" not in row[COL_TAGGED_TEXT]


def test_detection_batch_rejects_duplicate_result_ids_and_mixed_sources() -> None:
    row = _row("ABC", [])
    result = DetectionResult.model_validate(row[COL_SEED_DETECTION_RESULT])
    with pytest.raises(ValidationError, match="IDs must be unique"):
        DetectionResultBatch(results=(result, result))
    other = result.model_copy(
        update={
            "detection_result_id": "other",
            "source_view": result.source_view.model_copy(
                update={"source_text_ref": SourceTextRef(record_id="another-record", field_id=COL_TEXT)}
            ),
        }
    )
    with pytest.raises(ValidationError, match="same source view"):
        DetectionResultBatch(results=(result, other))


class _Augmenter:
    def __init__(self, entities: list[dict[str, str]]) -> None:
        self.entities = entities
        self.calls = 0

    def generate(self, **kwargs: Any) -> tuple[AugmentedEntitiesSchema, list[Any]]:
        self.calls += 1
        return AugmentedEntitiesSchema.model_validate({"entities": self.entities}), []

    async def agenerate(self, **kwargs: Any) -> tuple[AugmentedEntitiesSchema, list[Any]]:
        return self.generate(**kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("validate_matches", [False, True])
def test_augmentation_cannot_bypass_regex_only_scope_and_records_rejection(
    asynchronous: bool, validate_matches: bool
) -> None:
    row = _row(
        "allow:ABC deny:ABC",
        [
            RegexRule(
                label="token",
                pattern=r"(?<=allow:)ABC",
                validate_matches_with_llm=validate_matches,
                detect_additional_matches=False,
            )
        ],
        constrained=["token"],
    )
    _admit(row, [{"id": "c1", "decision": "keep"}] if validate_matches else [], constrained=["token"])
    model = _Augmenter(
        [
            {
                "value": "ABC",
                "label": "token",
                "context_before": "allow:",
                "context_after": " deny:ABC",
                "reason": "identifier",
            },
            {"value": "ABC", "label": "token", "context_before": "deny:", "context_after": "", "reason": "identifier"},
        ]
    )
    kwargs: dict[str, Any] = {
        "row": row,
        "prompt_template": "Find identifiers.",
        "model_alias": "test-model",
        "facade": model,
        "allowed_entity_labels": None,
        "excluded_entity_labels": [],
        "regex_constrained_entity_labels": ["token"],
    }
    if asynchronous:
        asyncio.run(augment_row_async(**kwargs))
    else:
        augment_row(**kwargs)
    admissions = AdmissionResults.model_validate(row[COL_AUGMENTATION_ADMISSION_RESULTS]).results
    assert [result.status for result in admissions] == [AdmissionStatus.accepted, AdmissionStatus.rejected]
    assert admissions[1].reason is AdmissionReason.missing_regex_evidence
    assert [(entity["start_position"], entity["end_position"]) for entity in _finalize(row)] == [(6, 9)]
    assert model.calls == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_disabled_augmentation_still_publishes_empty_lifecycle(asynchronous: bool) -> None:
    row = _row("ABC", [RegexRule(label="token", pattern="ABC", validate_matches_with_llm=False)])
    _admit(row, [])
    model = _Augmenter([])
    kwargs: dict[str, Any] = {
        "row": row,
        "prompt_template": "unused",
        "model_alias": "test-model",
        "facade": model,
        "allowed_entity_labels": ["token"],
        "excluded_entity_labels": [],
        "enabled": False,
    }
    if asynchronous:
        asyncio.run(augment_row_async(**kwargs))
    else:
        augment_row(**kwargs)
    assert model.calls == 0
    assert row[COL_AUGMENTED_CANDIDATES] == {"candidates": []}
    assert [entity["value"] for entity in _finalize(row)] == ["ABC"]
