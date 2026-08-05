# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_ENTITY_COVERAGE,
    COL_ENTITY_COVERAGE_JUDGE,
    COL_MISSED_ENTITIES,
    COL_TEXT,
)
from anonymizer.engine.evaluation.entity_coverage_judge import (
    _FINAL_ENTITIES_FOR_COVERAGE_COL,
    EntityCoverageWorkflow,
    _coverage_prompt,
    _filter_out_of_scope_entities,
    _find_missed_candidates,
    _is_candidate_value_covered,
    _parse_candidate_entities,
)
from anonymizer.engine.evaluation.judge_base import JudgeResult
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN


def test_coverage_prompt_omits_data_summary_context_when_summary_absent() -> None:
    without_summary = _coverage_prompt(entity_labels=None, strict_entity_protection=False)
    with_blank_summary = _coverage_prompt(
        entity_labels=None,
        strict_entity_protection=False,
        data_summary="   ",
    )

    assert without_summary == with_blank_summary
    assert "<data_summary_context>" not in without_summary


def test_coverage_prompt_includes_data_summary_as_interpretive_context() -> None:
    prompt = _coverage_prompt(
        entity_labels=["first_name"],
        strict_entity_protection=False,
        data_summary="Customer support transcripts.",
    )

    assert "<data_summary_context>\nCustomer support transcripts.\n" in prompt
    assert "Use this context only to interpret literal values and their semantic types." in prompt
    assert "Do not infer or invent entities that are absent from the original text." in prompt


def test_find_missed_candidates_removes_covered_subspans_and_composites() -> None:
    detected = [
        {"value": "Mstr Marzella", "label": "givenname"},
        {"value": "Nawabganj", "label": "city"},
        {"value": "382210", "label": "zipcode"},
        {"value": "44 Dunsfold Drive", "label": "street"},
        {"value": "Chihuahuan Desert", "label": "location"},
        {"value": "Annex Building", "label": "place_name"},
    ]
    candidates = [
        {"value": "Mstr", "label": "title"},  # subspan of a single final
        {"value": "Nawabganj - 382210", "label": "city"},  # composite of two whole finals
        {"value": "44", "label": "buildingnum"},  # short subspan
        # "Chihuahuan Desert Festival" adds the content token "festival" on top of the
        # detected "Chihuahuan Desert" — a distinct candidate, so it remains missed.
        {"value": "Chihuahuan Desert Festival", "label": "event"},
        {"value": "m", "label": "sex"},  # short token, not covered
        {"value": "Ann", "label": "first_name"},  # partial token of "Annex", not covered
        {"value": "uncovered value", "label": "unique_id"},
    ]

    assert _find_missed_candidates(candidates, detected) == [
        {"value": "Chihuahuan Desert Festival", "label": "event"},
        {"value": "m", "label": "sex"},
        {"value": "Ann", "label": "first_name"},
        {"value": "uncovered value", "label": "unique_id"},
    ]


@pytest.mark.parametrize(
    ("candidate_value", "final_values"),
    [
        ("Mstr", ["Mstr Marzella"]),  # subspan of a single final entity
        ("the Nawabganj", ["Nawabganj"]),  # grammatical stopword ignored
        ("44", ["44 Dunsfold Drive"]),  # short numeric subspan
        ("White House", ["White House Road"]),  # contiguous, in-order multi-token subspan
        ("Nawabganj - 382210", ["Nawabganj", "382210"]),  # composite of whole finals
        ("Nawabganj", ["Nawabganj", "382210"]),  # exact match against one final
        ("José", ["José García"]),  # accented subspan (Unicode tokenizer)
        ("Zürich", ["Zürich"]),  # accented exact match
    ],
)
def test_is_candidate_value_covered_true(candidate_value: str, final_values: list[str]) -> None:
    assert _is_candidate_value_covered(candidate_value, final_values) is True


@pytest.mark.parametrize(
    ("candidate_value", "final_values"),
    [
        # Cross-entity: pieces come from unrelated final entities -> a distinct missed candidate.
        ("John Smith", ["John Doe", "Jane Smith"]),
        # Reverse / non-contiguous order within a SINGLE final entity: shared tokens are
        # NOT enough — order and adjacency are required, so a reversed span is not covered.
        ("John Doe", ["Doe John"]),
        ("Ann Lee", ["Lee Ann Boulevard"]),
        ("John Doe", ["Doe John Memorial Highway"]),
        # Content descriptor is NOT ignored: a named event is a distinct candidate.
        ("Davos Summit", ["Davos"]),
        ("Chihuahuan Desert Festival", ["Chihuahuan Desert"]),
        # Partial-token substrings must NOT count as covered (no raw substring matching).
        ("Ann", ["Annex Building"]),
        ("Sara", ["Sarah Connor"]),
        ("ana", ["Banana Republic"]),
        # Short-token safeguard: a single letter is not covered by a longer token it prefixes.
        ("m", ["Mstr Marzella"]),
        # Nothing in common.
        ("uncovered value", ["Mstr Marzella", "Nawabganj"]),
        # No final entities -> nothing can be covered.
        ("Alice", []),
        # "at" must not be stripped — old stopword removal reduced "AT&T" to ["t"],
        # which was then incorrectly matched as a subspan of "T-Mobile" → ["t", "mobile"].
        ("AT&T", ["T-Mobile"]),
        # "of" is load-bearing in org names; stripping it would create a false subspan match.
        ("Bank of America", ["Bank"]),
    ],
)
def test_is_candidate_value_covered_false(candidate_value: str, final_values: list[str]) -> None:
    assert _is_candidate_value_covered(candidate_value, final_values) is False


def test_find_missed_candidates_keeps_cross_entity_reconstruction() -> None:
    """A candidate whose tokens span unrelated final entities remains missed."""
    detected = [
        {"value": "John Doe", "label": "first_name"},
        {"value": "Jane Smith", "label": "first_name"},
    ]
    candidates = [{"value": "John Smith", "label": "first_name"}]

    assert _find_missed_candidates(candidates, detected) == candidates


def test_find_missed_candidates_passthrough_on_no_final_entities() -> None:
    candidates = [{"value": "Alice", "label": "first_name"}]

    assert _find_missed_candidates(candidates, []) == candidates
    assert _find_missed_candidates(candidates, None) == candidates


def test_parse_candidate_entities_accepts_pydantic_model() -> None:
    from anonymizer.engine.evaluation.entity_coverage_judge import CandidateEntity, EntityCoverageSchema

    raw = EntityCoverageSchema(
        candidate_entities=[
            CandidateEntity(value="Alice", label="givenname", reasoning="The given name was not detected.")
        ]
    )
    assert _parse_candidate_entities(raw) == [
        {
            "value": "Alice",
            "label": "givenname",
            "reasoning": "The given name was not detected.",
        }
    ]


def test_parse_candidate_entities_accepts_dict() -> None:
    raw = {
        "candidate_entities": [
            {"value": "Alice", "label": "givenname", "reasoning": "The given name was not detected."}
        ]
    }
    assert _parse_candidate_entities(raw) == [
        {
            "value": "Alice",
            "label": "givenname",
            "reasoning": "The given name was not detected.",
        }
    ]


def test_parse_candidate_entities_returns_none_for_string_input() -> None:
    # With LLMStructuredColumnConfig, raw input is never a plain string.
    # Strings are treated as malformed and return None.
    assert _parse_candidate_entities('{"candidate_entities": []}') is None


def test_parse_candidate_entities_returns_none_for_none() -> None:
    assert _parse_candidate_entities(None) is None


def test_parse_candidate_entities_returns_none_for_empty_dict() -> None:
    # {} is a valid JSON object but omits candidate_entities entirely.
    # This must be treated as an unavailable score, not "no candidates found",
    # so that an incomplete structured response can't silently produce perfect coverage.
    assert _parse_candidate_entities({}) is None


def test_coverage_prompt_extracts_independently_before_deterministic_filtering() -> None:
    prompt = _coverage_prompt(entity_labels=["sex", "title"], strict_entity_protection=False)

    assert "Work independently from the anonymizer" in prompt
    assert "deterministic postprocessing step" in prompt
    assert "<anonymizer_final_entities>" not in prompt
    assert "_final_entities_for_coverage_judge" not in prompt


def test_coverage_prompt_requires_systematic_structured_text_scan() -> None:
    prompt = _coverage_prompt(entity_labels=["sex", "title"], strict_entity_protection=False)

    assert "salutations, signatures" in prompt
    assert "Tables, bullets, forms" in prompt
    assert "Short or single-token values" in prompt
    assert "Honorifics attached to person names" in prompt


def _stub_evaluate_selection() -> EvaluateModelSelection:
    return EvaluateModelSelection(
        entity_coverage_judge="nemotron-super",
        detection_validity_judge="gpt-oss-120b",
        replace_type_fidelity_judge="gpt-oss-120b",
        replace_relational_consistency_judge="gpt-oss-120b",
        replace_attribute_fidelity_judge="gpt-oss-120b",
        rewrite_judge="nemotron-30b-thinking",
    )


def test_column_config_builds_prompt_and_resolves_model() -> None:
    """Smoke-guard the real prompt-build path: a broken ``_coverage_prompt`` signature
    (e.g. a stray required parameter) must fail loudly here, since ``run_non_critical``
    swallows exceptions downstream and would otherwise mask it as ``entity_coverage=None``.
    """
    workflow = EntityCoverageWorkflow(
        adapter=Mock(),
        entity_labels=["sex", "title"],
        strict_entity_protection=True,
        data_summary="Customer support transcripts.",
    )

    config = workflow.column_config(_stub_evaluate_selection())

    assert config.model_alias == "nemotron-super"
    # Prompt built without error and carries the instance-specific context.
    assert isinstance(config.prompt, str) and config.prompt
    assert "Customer support transcripts." in config.prompt  # data_summary threaded in
    assert "sex, title" in config.prompt  # entity_labels scope threaded in


def test_postprocess_coverage_uses_judge_anchored_recall() -> None:
    """Coverage score is n_covered / n_candidates — recall over the judge's candidate set.

    False positives in the anonymizer's detections do not affect the score;
    only the judge's candidates determine the numerator and denominator.
    """
    workflow = EntityCoverageWorkflow(adapter=Mock())

    # Judge found 5 candidates; 4 are in final_entities (covered), 1 is not (missed).
    judge_candidates = [
        {"value": "Alice", "label": "first_name", "reasoning": "name"},
        {"value": "Smith", "label": "last_name", "reasoning": "name"},
        {"value": "alice@example.com", "label": "email", "reasoning": "email"},
        {"value": "555-1234", "label": "phone_number", "reasoning": "phone"},
        {"value": "MissedEntity", "label": "org", "reasoning": "org"},  # not in final_entities
    ]
    # Anonymizer detected 10 entities — 4 overlap with judge, 6 are false positives.
    final_entities = [
        {"value": "Alice", "label": "first_name"},
        {"value": "Smith", "label": "last_name"},
        {"value": "alice@example.com", "label": "email"},
        {"value": "555-1234", "label": "phone_number"},
        {"value": "FP1", "label": "location"},
        {"value": "FP2", "label": "location"},
        {"value": "FP3", "label": "location"},
        {"value": "FP4", "label": "location"},
        {"value": "FP5", "label": "location"},
        {"value": "FP6", "label": "location"},
    ]
    text = "Alice Smith alice@example.com 555-1234 MissedEntity"

    df = pd.DataFrame(
        {
            COL_TEXT: [text],
            COL_ENTITY_COVERAGE_JUDGE: [{"candidate_entities": judge_candidates}],
            _FINAL_ENTITIES_FOR_COVERAGE_COL: [final_entities],
        }
    )

    result = workflow.postprocess(df)

    # n_candidates=5, n_covered=4, n_missed=1 → 4/5 = 0.8
    assert result[COL_ENTITY_COVERAGE].iloc[0] == pytest.approx(0.8)
    assert len(result[COL_MISSED_ENTITIES].iloc[0]) == 1
    assert result[COL_MISSED_ENTITIES].iloc[0][0]["value"] == "MissedEntity"


def test_postprocess_coverage_is_1_when_judge_finds_no_candidates() -> None:
    """Coverage is 1.0 when the judge finds no PII candidates — nothing to miss."""
    workflow = EntityCoverageWorkflow(adapter=Mock())
    df = pd.DataFrame(
        {
            COL_TEXT: ["No PII here."],
            COL_ENTITY_COVERAGE_JUDGE: [{"candidate_entities": []}],
            _FINAL_ENTITIES_FOR_COVERAGE_COL: [[]],
        }
    )
    result = workflow.postprocess(df)
    assert result[COL_ENTITY_COVERAGE].iloc[0] == 1.0
    assert result[COL_MISSED_ENTITIES].iloc[0] == []


def test_build_prompt_stub_fails_loudly_instead_of_ignoring_instance_configuration() -> None:
    workflow = EntityCoverageWorkflow(adapter=Mock(), entity_labels=["first_name"])

    with pytest.raises(NotImplementedError, match="builds its prompt in column_config"):
        workflow._build_prompt()


def test_run_non_critical_preserves_successful_rows_when_adapter_drops_one() -> None:
    adapter = Mock()
    adapter._attach_record_ids.side_effect = lambda dataframe: dataframe.assign(
        **{RECORD_ID_COLUMN: ["row-0", "row-1"]}
    )
    workflow = EntityCoverageWorkflow(adapter=adapter)
    failed_record = Mock()
    workflow.evaluate = Mock(
        return_value=JudgeResult(
            dataframe=pd.DataFrame(
                {
                    RECORD_ID_COLUMN: ["row-0"],
                    COL_ENTITY_COVERAGE_JUDGE: [{"candidate_entities": []}],
                    COL_ENTITY_COVERAGE: [1.0],
                    COL_MISSED_ENTITIES: [[]],
                }
            ),
            failed_records=[failed_record],
        )
    )

    result, failed_records = workflow.run_non_critical(
        pd.DataFrame({"input_value": ["scored", "dropped"]}),
        model_configs=[],
        selected_models=_stub_evaluate_selection(),
    )

    assert result["input_value"].tolist() == ["scored", "dropped"]
    assert result.loc[0, COL_ENTITY_COVERAGE] == 1.0
    assert result.loc[1, COL_ENTITY_COVERAGE] is None
    assert result[COL_MISSED_ENTITIES].tolist() == [[], []]
    assert RECORD_ID_COLUMN not in result.columns
    assert failed_records == [failed_record]


def test_filter_out_of_scope_entities_drops_out_of_scope_label() -> None:
    entities = [
        {"value": "Alice", "label": "first_name", "reasoning": "..."},
        {"value": "555-1234", "label": "phone_number", "reasoning": "..."},
    ]
    result = _filter_out_of_scope_entities(entities, entity_labels=["first_name", "email"])
    assert result == [{"value": "Alice", "label": "first_name", "reasoning": "..."}]


def test_filter_out_of_scope_entities_drops_empty_label() -> None:
    entities = [
        {"value": "Alice", "label": "", "reasoning": "..."},
        {"value": "bob@example.com", "label": "email", "reasoning": "..."},
    ]
    result = _filter_out_of_scope_entities(entities, entity_labels=None)
    assert result == [{"value": "bob@example.com", "label": "email", "reasoning": "..."}]


def test_filter_out_of_scope_entities_allows_all_when_no_scope_configured() -> None:
    entities = [
        {"value": "Alice", "label": "first_name", "reasoning": "..."},
        {"value": "555-1234", "label": "phone_number", "reasoning": "..."},
    ]
    result = _filter_out_of_scope_entities(entities, entity_labels=None)
    assert result == entities


def test_filter_out_of_scope_entities_is_case_insensitive() -> None:
    entities = [{"value": "Alice", "label": "First_Name", "reasoning": "..."}]
    result = _filter_out_of_scope_entities(entities, entity_labels=["first_name"])
    assert result == entities
