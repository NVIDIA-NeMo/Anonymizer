# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_FULL_REWRITE,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    _jinja,
)
from anonymizer.engine.rewrite.rewrite_generation import (
    RewriteGenerationWorkflow,
    _extract_rewritten_text,
    _filter_replacement_map_for_prompt,
    _format_rewrite_disposition_block,
    _get_rewrite_prompt,
)
from anonymizer.engine.schemas import EntityReplacementMapSchema, RewriteOutputSchema

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def privacy_goal() -> PrivacyGoal:
    return PrivacyGoal(
        protect="All direct identifiers including names, locations, and contact details",
        preserve="Career trajectory, skills, and professional context in abstract terms",
    )


@pytest.fixture
def stub_sensitivity_disposition() -> dict:
    return {
        "sensitivity_disposition": [
            {
                "id": 1,
                "source": "tagged",
                "category": "direct_identifier",
                "sensitivity": "high",
                "entity_label": "first_name",
                "entity_value": "Alice",
                "needs_protection": True,
                "protection_reason": "Full name uniquely identifies the subject",
                "protection_method_suggestion": "replace",
                "combined_risk_level": "high",
            }
        ]
    }


@pytest.fixture
def stub_replacement_map() -> dict:
    return {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maria"}]}


# ---------------------------------------------------------------------------
# Tests: _format_rewrite_disposition_block
# ---------------------------------------------------------------------------


def test_format_rewrite_disposition_block_filters_to_protected_only(
    stub_sensitivity_disposition: dict,
) -> None:
    row = {COL_SENSITIVITY_DISPOSITION: stub_sensitivity_disposition}
    result = _format_rewrite_disposition_block(row)
    block = result[COL_REWRITE_DISPOSITION_BLOCK]
    assert len(block) == 1
    assert block[0]["entity_value"] == "Alice"


def test_format_rewrite_disposition_block_excludes_unprotected_entities() -> None:
    disposition = {
        "sensitivity_disposition": [
            {
                "id": 1,
                "source": "tagged",
                "category": "quasi_identifier",
                "sensitivity": "low",
                "entity_label": "city",
                "entity_value": "Portland",
                "needs_protection": False,
                "protection_reason": "Not identifying alone",
                "protection_method_suggestion": "leave_as_is",
                "combined_risk_level": "low",
            }
        ]
    }
    row = {COL_SENSITIVITY_DISPOSITION: disposition}
    result = _format_rewrite_disposition_block(row)
    assert result[COL_REWRITE_DISPOSITION_BLOCK] == []


def test_format_rewrite_disposition_block_serializes_required_fields(
    stub_sensitivity_disposition: dict,
) -> None:
    row = {COL_SENSITIVITY_DISPOSITION: stub_sensitivity_disposition}
    result = _format_rewrite_disposition_block(row)
    block = result[COL_REWRITE_DISPOSITION_BLOCK]
    entry = block[0]
    assert set(entry.keys()) == {
        "entity_label",
        "entity_value",
        "sensitivity",
        "protection_method_suggestion",
        "protection_reason",
    }


def test_format_rewrite_disposition_block_empty_when_no_protected_entities() -> None:
    disposition = {
        "sensitivity_disposition": [
            {
                "id": 1,
                "source": "tagged",
                "category": "quasi_identifier",
                "sensitivity": "low",
                "entity_label": "city",
                "entity_value": "Portland",
                "needs_protection": False,
                "protection_reason": "Not identifying alone",
                "protection_method_suggestion": "leave_as_is",
                "combined_risk_level": "low",
            }
        ]
    }
    row = {COL_SENSITIVITY_DISPOSITION: disposition}
    result = _format_rewrite_disposition_block(row)
    assert result[COL_REWRITE_DISPOSITION_BLOCK] == []


# ---------------------------------------------------------------------------
# Tests: _filter_replacement_map_for_prompt
# ---------------------------------------------------------------------------


def test_filter_replacement_map_keeps_only_replace_method_entities(
    stub_sensitivity_disposition: dict,
    stub_replacement_map: dict,
) -> None:
    disposition_block = [
        {
            "entity_label": "first_name",
            "entity_value": "Alice",
            "sensitivity": "high",
            "protection_method_suggestion": "replace",
            "protection_reason": "Direct identifier",
        }
    ]
    row = {
        COL_REPLACEMENT_MAP: stub_replacement_map,
        COL_REWRITE_DISPOSITION_BLOCK: disposition_block,
    }
    result = _filter_replacement_map_for_prompt(row)
    filtered = result[COL_REPLACEMENT_MAP_FOR_PROMPT]
    assert len(filtered["replacements"]) == 1
    assert filtered["replacements"][0]["original"] == "Alice"


def test_filter_replacement_map_empty_when_no_replace_method() -> None:
    disposition_block = [
        {
            "entity_label": "city",
            "entity_value": "Portland",
            "sensitivity": "low",
            "protection_method_suggestion": "generalize",
            "protection_reason": "Quasi-identifier",
        }
    ]
    row = {
        COL_REPLACEMENT_MAP: {"replacements": [{"original": "Portland", "label": "city", "synthetic": "Seattle"}]},
        COL_REWRITE_DISPOSITION_BLOCK: disposition_block,
    }
    result = _filter_replacement_map_for_prompt(row)
    assert result[COL_REPLACEMENT_MAP_FOR_PROMPT]["replacements"] == []


def test_filter_replacement_map_accepts_schema_instance(
    stub_replacement_map: dict,
) -> None:
    disposition_block = [
        {
            "entity_label": "first_name",
            "entity_value": "Alice",
            "sensitivity": "high",
            "protection_method_suggestion": "replace",
            "protection_reason": "Direct identifier",
        }
    ]
    schema = EntityReplacementMapSchema.model_validate(stub_replacement_map)
    row = {
        COL_REPLACEMENT_MAP: schema,
        COL_REWRITE_DISPOSITION_BLOCK: disposition_block,
    }
    result = _filter_replacement_map_for_prompt(row)
    assert len(result[COL_REPLACEMENT_MAP_FOR_PROMPT]["replacements"]) == 1


# ---------------------------------------------------------------------------
# Tests: _extract_rewritten_text
# ---------------------------------------------------------------------------


def test_extract_rewritten_text_from_pydantic_instance() -> None:
    row = {COL_FULL_REWRITE: RewriteOutputSchema(rewritten_text="Maria works at TechCorp")}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] == "Maria works at TechCorp"


def test_extract_rewritten_text_from_dict_payload() -> None:
    row = {COL_FULL_REWRITE: {"rewritten_text": "Maria works at TechCorp"}}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] == "Maria works at TechCorp"


def test_extract_rewritten_text_returns_none_on_failure() -> None:
    row = {COL_FULL_REWRITE: "not-a-valid-payload"}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] is None


def test_extract_rewritten_text_returns_none_on_missing_key() -> None:
    row = {COL_FULL_REWRITE: {"wrong_key": "value"}}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] is None


def test_extract_rewritten_text_returns_none_on_blank_output() -> None:
    row = {COL_FULL_REWRITE: {"rewritten_text": "   "}}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] is None


# ---------------------------------------------------------------------------
# Tests: _get_rewrite_prompt
# ---------------------------------------------------------------------------


def test_get_rewrite_prompt_contains_privacy_goal(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal)
    assert "PROTECT" in prompt
    assert "PRESERVE" in prompt


def test_get_rewrite_prompt_uses_xml_section_headers(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal)
    for tag in ["privacy_goal", "instructions", "input", "sensitivity_disposition", "output_requirements"]:
        assert f"<{tag}>" in prompt
        assert f"</{tag}>" in prompt


def test_get_rewrite_prompt_injects_data_context_when_provided(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal, data_summary="Biographical profiles")
    assert "Biographical profiles" in prompt
    assert "<data_context>" in prompt


def test_get_rewrite_prompt_no_data_context_when_none(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal, data_summary=None)
    assert "<data_context>" not in prompt


def test_get_rewrite_prompt_references_required_columns(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal)
    assert _jinja(COL_TAGGED_TEXT) in prompt
    assert COL_TAG_NOTATION in prompt
    assert COL_REWRITE_DISPOSITION_BLOCK in prompt
    assert _jinja(COL_REPLACEMENT_MAP_FOR_PROMPT) in prompt


# ---------------------------------------------------------------------------
# Tests: RewriteGenerationWorkflow.columns()
# ---------------------------------------------------------------------------


def test_columns_returns_four_configs(
    stub_rewrite_model_selection: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
) -> None:
    workflow = RewriteGenerationWorkflow()
    cols = workflow.columns(selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal)
    assert len(cols) == 4


def test_columns_has_llm_config_with_rewriter_alias(
    stub_rewrite_model_selection: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
) -> None:
    workflow = RewriteGenerationWorkflow()
    cols = workflow.columns(selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal)
    llm_cols = [c for c in cols if isinstance(c, LLMStructuredColumnConfig)]
    assert len(llm_cols) == 1
    assert llm_cols[0].name == COL_FULL_REWRITE


def test_columns_full_rewrite_uses_rewrite_output_schema(
    stub_rewrite_model_selection: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
) -> None:
    workflow = RewriteGenerationWorkflow()
    cols = workflow.columns(selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal)
    full_rewrite_col = next(c for c in cols if c.name == COL_FULL_REWRITE)
    assert full_rewrite_col.output_format == RewriteOutputSchema.model_json_schema()


def test_columns_includes_custom_configs_for_disposition_and_text_extraction(
    stub_rewrite_model_selection: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
) -> None:
    workflow = RewriteGenerationWorkflow()
    cols = workflow.columns(selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal)
    custom_names = {c.name for c in cols if isinstance(c, CustomColumnConfig)}
    assert COL_REWRITE_DISPOSITION_BLOCK in custom_names
    assert COL_REPLACEMENT_MAP_FOR_PROMPT in custom_names
    assert COL_REWRITTEN_TEXT in custom_names
