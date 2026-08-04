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
    COL_LEAKED_ENTITIES,
)
from anonymizer.engine.evaluation.entity_coverage_judge import (
    EntityCoverageWorkflow,
    _coverage_prompt,
    _filter_covered_leaked_entities,
    _filter_out_of_scope_entities,
    _is_leaked_value_covered,
    _parse_leaked_entities,
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


def test_filter_covered_leaked_entities_removes_subspans_and_composites() -> None:
    detected = [
        {"value": "Mstr Marzella", "label": "givenname"},
        {"value": "Nawabganj", "label": "city"},
        {"value": "382210", "label": "zipcode"},
        {"value": "44 Dunsfold Drive", "label": "street"},
        {"value": "Chihuahuan Desert", "label": "location"},
        {"value": "Annex Building", "label": "place_name"},
    ]
    leaked = [
        {"value": "Mstr", "label": "title"},  # subspan of a single final
        {"value": "Nawabganj - 382210", "label": "city"},  # composite of two whole finals
        {"value": "44", "label": "buildingnum"},  # short subspan
        # "Chihuahuan Desert Festival" adds the content token "festival" on top of the
        # detected "Chihuahuan Desert" — a named event, so it is a real leak (NOT covered).
        {"value": "Chihuahuan Desert Festival", "label": "event"},
        {"value": "m", "label": "sex"},  # short token, not covered
        {"value": "Ann", "label": "first_name"},  # partial token of "Annex", not covered
        {"value": "uncovered value", "label": "unique_id"},
    ]

    assert _filter_covered_leaked_entities(leaked, detected) == [
        {"value": "Chihuahuan Desert Festival", "label": "event"},
        {"value": "m", "label": "sex"},
        {"value": "Ann", "label": "first_name"},
        {"value": "uncovered value", "label": "unique_id"},
    ]


@pytest.mark.parametrize(
    ("leaked_value", "final_values"),
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
def test_is_leaked_value_covered_true(leaked_value: str, final_values: list[str]) -> None:
    assert _is_leaked_value_covered(leaked_value, final_values) is True


@pytest.mark.parametrize(
    ("leaked_value", "final_values"),
    [
        # Cross-entity: pieces come from unrelated final entities -> a real, distinct leak.
        ("John Smith", ["John Doe", "Jane Smith"]),
        # Reverse / non-contiguous order within a SINGLE final entity: shared tokens are
        # NOT enough — order and adjacency are required, so a reversed span is not covered.
        ("John Doe", ["Doe John"]),
        ("Ann Lee", ["Lee Ann Boulevard"]),
        ("John Doe", ["Doe John Memorial Highway"]),
        # Content descriptor is NOT ignored: a named event is a distinct leak.
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
def test_is_leaked_value_covered_false(leaked_value: str, final_values: list[str]) -> None:
    assert _is_leaked_value_covered(leaked_value, final_values) is False


def test_filter_covered_leaked_entities_keeps_cross_entity_reconstruction() -> None:
    """A leak whose tokens are spread across unrelated final entities is a real leak."""
    detected = [
        {"value": "John Doe", "label": "first_name"},
        {"value": "Jane Smith", "label": "first_name"},
    ]
    leaked = [{"value": "John Smith", "label": "first_name"}]

    assert _filter_covered_leaked_entities(leaked, detected) == leaked


def test_filter_covered_leaked_entities_passthrough_on_no_final_entities() -> None:
    leaked = [{"value": "Alice", "label": "first_name"}]

    assert _filter_covered_leaked_entities(leaked, []) == leaked
    assert _filter_covered_leaked_entities(leaked, None) == leaked


def test_parse_leaked_entities_accepts_pydantic_model() -> None:
    from anonymizer.engine.evaluation.entity_coverage_judge import EntityCoverageSchema, LeakedEntity

    raw = EntityCoverageSchema(
        leaked_entities=[LeakedEntity(value="Alice", label="givenname", reasoning="The given name was not detected.")]
    )
    assert _parse_leaked_entities(raw) == [
        {
            "value": "Alice",
            "label": "givenname",
            "reasoning": "The given name was not detected.",
        }
    ]


def test_parse_leaked_entities_accepts_dict() -> None:
    raw = {
        "leaked_entities": [{"value": "Alice", "label": "givenname", "reasoning": "The given name was not detected."}]
    }
    assert _parse_leaked_entities(raw) == [
        {
            "value": "Alice",
            "label": "givenname",
            "reasoning": "The given name was not detected.",
        }
    ]


def test_parse_leaked_entities_returns_none_for_string_input() -> None:
    # With LLMStructuredColumnConfig, raw input is never a plain string.
    # Strings are treated as malformed and return None.
    assert _parse_leaked_entities('{"leaked_entities": []}') is None


def test_parse_leaked_entities_returns_none_for_none() -> None:
    assert _parse_leaked_entities(None) is None


def test_parse_leaked_entities_returns_none_for_empty_dict() -> None:
    # {} is a valid JSON object but omits leaked_entities entirely.
    # This must be treated as an unavailable score, not "no leaks found",
    # so that an incomplete structured response can't silently produce perfect coverage.
    assert _parse_leaked_entities({}) is None


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
                    COL_ENTITY_COVERAGE_JUDGE: [{"leaked_entities": []}],
                    COL_ENTITY_COVERAGE: [1.0],
                    COL_LEAKED_ENTITIES: [[]],
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
    assert result[COL_LEAKED_ENTITIES].tolist() == [[], []]
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
