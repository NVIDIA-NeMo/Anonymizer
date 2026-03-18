# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_FULL_REWRITE,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.engine.rewrite.rewrite_generation import (
    RewriteGenerationWorkflow,
    _extract_rewritten_text,
    _filter_replacement_map_for_prompt,
    _format_rewrite_disposition_block,
    _get_rewrite_prompt,
    _has_entities,
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
def stub_entities_by_value_with_entities() -> dict:
    return {
        "entities_by_value": [
            {"value": "Alice", "labels": ["first_name"]},
            {"value": "Acme Corp", "labels": ["company_name"]},
        ]
    }


@pytest.fixture
def stub_entities_by_value_empty() -> dict:
    return {"entities_by_value": []}


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


@pytest.fixture
def stub_df_with_entities(
    stub_entities_by_value_with_entities: dict,
    stub_sensitivity_disposition: dict,
    stub_replacement_map: dict,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme Corp"],
            COL_TAGGED_TEXT: ["⟦Alice|first_name⟧ works at ⟦Acme Corp|company_name⟧"],
            COL_ENTITIES_BY_VALUE: [stub_entities_by_value_with_entities],
            COL_SENSITIVITY_DISPOSITION: [stub_sensitivity_disposition],
            COL_REPLACEMENT_MAP: [stub_replacement_map],
        }
    )


@pytest.fixture
def stub_df_no_entities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["The sky is blue"],
            COL_TAGGED_TEXT: ["The sky is blue"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}],
            COL_SENSITIVITY_DISPOSITION: [{"sensitivity_disposition": []}],
            COL_REPLACEMENT_MAP: [{"replacements": []}],
        }
    )


# ---------------------------------------------------------------------------
# Tests: _has_entities
# ---------------------------------------------------------------------------


def test_has_entities_returns_true_when_entities_present(
    stub_entities_by_value_with_entities: dict,
) -> None:
    assert _has_entities(stub_entities_by_value_with_entities) is True


def test_has_entities_returns_false_when_empty(stub_entities_by_value_empty: dict) -> None:
    assert _has_entities(stub_entities_by_value_empty) is False


def test_has_entities_returns_false_for_none() -> None:
    assert _has_entities(None) is False


def test_has_entities_returns_false_for_invalid_data() -> None:
    assert _has_entities("not-a-dict") is False


# ---------------------------------------------------------------------------
# Tests: _format_rewrite_disposition_block
# ---------------------------------------------------------------------------


def test_format_rewrite_disposition_block_filters_to_protected_only(
    stub_sensitivity_disposition: dict,
) -> None:
    row: dict = {COL_SENSITIVITY_DISPOSITION: stub_sensitivity_disposition}
    result = _format_rewrite_disposition_block(row)
    block = result[COL_REWRITE_DISPOSITION_BLOCK]
    assert len(block) == 1
    assert block[0]["entity_label"] == "first_name"
    assert block[0]["entity_value"] == "Alice"


def test_format_rewrite_disposition_block_excludes_unprotected_entities() -> None:
    disposition = {
        "sensitivity_disposition": [
            {
                "id": 1,
                "source": "tagged",
                "category": "direct_identifier",
                "sensitivity": "high",
                "entity_label": "first_name",
                "entity_value": "Alice",
                "needs_protection": True,
                "protection_reason": "Direct identifier",
                "protection_method_suggestion": "replace",
                "combined_risk_level": "high",
            },
            {
                "id": 2,
                "source": "tagged",
                "category": "quasi_identifier",
                "sensitivity": "low",
                "entity_label": "city",
                "entity_value": "Portland",
                "needs_protection": False,
                "protection_reason": "Low risk, generic city name",
                "protection_method_suggestion": "left_as_is",
                "combined_risk_level": "low",
            },
        ]
    }
    row: dict = {COL_SENSITIVITY_DISPOSITION: disposition}
    result = _format_rewrite_disposition_block(row)
    block = result[COL_REWRITE_DISPOSITION_BLOCK]
    assert len(block) == 1
    assert block[0]["entity_label"] == "first_name"


def test_format_rewrite_disposition_block_serializes_required_fields(
    stub_sensitivity_disposition: dict,
) -> None:
    row: dict = {COL_SENSITIVITY_DISPOSITION: stub_sensitivity_disposition}
    result = _format_rewrite_disposition_block(row)
    entry = result[COL_REWRITE_DISPOSITION_BLOCK][0]
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
                "protection_reason": "Low risk, generic city name",
                "protection_method_suggestion": "left_as_is",
                "combined_risk_level": "low",
            }
        ]
    }
    row: dict = {COL_SENSITIVITY_DISPOSITION: disposition}
    result = _format_rewrite_disposition_block(row)
    assert result[COL_REWRITE_DISPOSITION_BLOCK] == []


# ---------------------------------------------------------------------------
# Tests: _filter_replacement_map_for_prompt
# ---------------------------------------------------------------------------


def test_filter_replacement_map_keeps_only_replace_method_entities() -> None:
    row: dict = {
        COL_REPLACEMENT_MAP: {
            "replacements": [
                {"original": "Alice", "label": "first_name", "synthetic": "Maria"},
                {"original": "Portland", "label": "city", "synthetic": "Austin"},
            ]
        },
        COL_REWRITE_DISPOSITION_BLOCK: [
            {
                "entity_label": "first_name",
                "entity_value": "Alice",
                "sensitivity": "high",
                "protection_method_suggestion": "replace",
                "protection_reason": "Direct identifier",
            },
            {
                "entity_label": "city",
                "entity_value": "Portland",
                "sensitivity": "medium",
                "protection_method_suggestion": "generalize",
                "protection_reason": "Quasi-identifier, generalize instead",
            },
        ],
    }
    result = _filter_replacement_map_for_prompt(row)
    filtered = result[COL_REPLACEMENT_MAP_FOR_PROMPT]
    assert len(filtered["replacements"]) == 1
    assert filtered["replacements"][0]["original"] == "Alice"


def test_filter_replacement_map_empty_when_no_replace_method() -> None:
    row: dict = {
        COL_REPLACEMENT_MAP: {
            "replacements": [
                {"original": "Alice", "label": "first_name", "synthetic": "Maria"},
            ]
        },
        COL_REWRITE_DISPOSITION_BLOCK: [
            {
                "entity_label": "first_name",
                "entity_value": "Alice",
                "sensitivity": "high",
                "protection_method_suggestion": "generalize",
                "protection_reason": "Generalize for privacy",
            },
        ],
    }
    result = _filter_replacement_map_for_prompt(row)
    assert result[COL_REPLACEMENT_MAP_FOR_PROMPT] == {"replacements": []}


def test_filter_replacement_map_accepts_schema_instance() -> None:
    row: dict = {
        COL_REPLACEMENT_MAP: EntityReplacementMapSchema(
            replacements=[
                {"original": "Alice", "label": "first_name", "synthetic": "Maria"},
                {"original": "Portland", "label": "city", "synthetic": "Austin"},
            ]
        ),
        COL_REWRITE_DISPOSITION_BLOCK: [
            {
                "entity_label": "first_name",
                "entity_value": "Alice",
                "sensitivity": "high",
                "protection_method_suggestion": "replace",
                "protection_reason": "Direct identifier",
            }
        ],
    }
    result = _filter_replacement_map_for_prompt(row)
    assert result[COL_REPLACEMENT_MAP_FOR_PROMPT] == {
        "replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maria"}]
    }


# ---------------------------------------------------------------------------
# Tests: _extract_rewritten_text
# ---------------------------------------------------------------------------


def test_extract_rewritten_text_from_pydantic_instance() -> None:
    schema_instance = RewriteOutputSchema(rewritten_text="Maria works at TechCorp")
    row: dict = {COL_FULL_REWRITE: schema_instance, COL_TEXT: "original"}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] == "Maria works at TechCorp"


def test_extract_rewritten_text_from_dict_payload() -> None:
    row: dict = {COL_FULL_REWRITE: {"rewritten_text": "Maria works at TechCorp"}, COL_TEXT: "original"}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] == "Maria works at TechCorp"


def test_extract_rewritten_text_returns_none_on_failure() -> None:
    row: dict = {COL_FULL_REWRITE: None, COL_TEXT: "original text"}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] is None


def test_extract_rewritten_text_returns_none_on_missing_key() -> None:
    row: dict = {COL_FULL_REWRITE: {"wrong_key": "value"}, COL_TEXT: "original text"}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] is None


def test_extract_rewritten_text_returns_none_on_blank_output() -> None:
    row: dict = {COL_FULL_REWRITE: {"rewritten_text": "   "}, COL_TEXT: "original text"}
    result = _extract_rewritten_text(row)
    assert result[COL_REWRITTEN_TEXT] is None


# ---------------------------------------------------------------------------
# Tests: _get_rewrite_prompt
# ---------------------------------------------------------------------------


def test_get_rewrite_prompt_contains_privacy_goal(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal)
    assert "PROTECT:" in prompt
    assert "PRESERVE:" in prompt


def test_get_rewrite_prompt_uses_xml_section_headers(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal)
    assert "<instructions>" in prompt
    assert "<privacy_goal>" in prompt
    assert "<input>" in prompt
    assert "<sensitivity_disposition>" in prompt
    assert "<replacement_map>" in prompt  # inside Jinja conditional
    assert "<output_requirements>" in prompt


def test_get_rewrite_prompt_injects_data_context_when_provided(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal, data_summary="Biography dataset from healthcare domain")
    assert "<data_context>" in prompt
    assert "Dataset description:" in prompt
    assert "Biography dataset from healthcare domain" in prompt


def test_get_rewrite_prompt_no_data_context_when_none(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal, data_summary=None)
    assert "<data_context>" not in prompt


def test_get_rewrite_prompt_references_required_columns(privacy_goal: PrivacyGoal) -> None:
    prompt = _get_rewrite_prompt(privacy_goal)
    assert COL_TAG_NOTATION in prompt
    assert _jinja(COL_TAGGED_TEXT) in prompt
    assert COL_REWRITE_DISPOSITION_BLOCK in prompt
    assert _jinja(COL_REPLACEMENT_MAP_FOR_PROMPT) in prompt


# ---------------------------------------------------------------------------
# Tests: RewriteGenerationWorkflow.columns()
# ---------------------------------------------------------------------------


def test_columns_returns_four_configs(
    stub_rewrite_model_selection: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
) -> None:
    workflow = RewriteGenerationWorkflow(adapter=Mock())
    cols = workflow.columns(selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal)
    assert len(cols) == 4


def test_columns_has_llm_config_with_rewriter_alias(
    stub_rewrite_model_selection: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
) -> None:
    workflow = RewriteGenerationWorkflow(adapter=Mock())
    cols = workflow.columns(selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal)
    llm_cols = [c for c in cols if isinstance(c, LLMStructuredColumnConfig)]
    assert len(llm_cols) == 1
    assert llm_cols[0].model_alias == stub_rewrite_model_selection.rewriter


def test_columns_full_rewrite_uses_rewrite_output_schema(
    stub_rewrite_model_selection: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
) -> None:
    workflow = RewriteGenerationWorkflow(adapter=Mock())
    cols = workflow.columns(selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal)
    llm_col = next(c for c in cols if isinstance(c, LLMStructuredColumnConfig))
    assert llm_col.name == COL_FULL_REWRITE
    assert llm_col.output_format["title"] == "RewriteOutputSchema"
    assert "rewritten_text" in llm_col.output_format["properties"]


def test_columns_includes_custom_configs_for_disposition_and_text_extraction(
    stub_rewrite_model_selection: RewriteModelSelection,
    privacy_goal: PrivacyGoal,
) -> None:
    workflow = RewriteGenerationWorkflow(adapter=Mock())
    cols = workflow.columns(selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal)
    custom_names = {c.name for c in cols if isinstance(c, CustomColumnConfig)}
    assert COL_REWRITE_DISPOSITION_BLOCK in custom_names
    assert COL_REPLACEMENT_MAP_FOR_PROMPT in custom_names
    assert COL_REWRITTEN_TEXT in custom_names


# ---------------------------------------------------------------------------
# Tests: RewriteGenerationWorkflow.run() — fast path
# ---------------------------------------------------------------------------


def test_run_passthrough_rows_get_original_text_without_llm_call(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection,
    stub_df_no_entities: pd.DataFrame,
    privacy_goal: PrivacyGoal,
) -> None:
    adapter = Mock()
    workflow = RewriteGenerationWorkflow(adapter=adapter)

    result = workflow.run(
        stub_df_no_entities,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=privacy_goal,
    )

    adapter.run_workflow.assert_not_called()
    assert COL_REWRITTEN_TEXT in result.dataframe.columns
    assert result.dataframe[COL_REWRITTEN_TEXT].iloc[0] == "The sky is blue"


def test_run_output_length_equals_input_length_when_all_passthrough(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection,
    privacy_goal: PrivacyGoal,
) -> None:
    df = pd.DataFrame(
        {
            COL_TEXT: ["text one", "text two", "text three"],
            COL_TAGGED_TEXT: ["text one", "text two", "text three"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}] * 3,
            COL_SENSITIVITY_DISPOSITION: [{"sensitivity_disposition": []}] * 3,
            COL_REPLACEMENT_MAP: [{"replacements": []}] * 3,
        }
    )
    adapter = Mock()
    workflow = RewriteGenerationWorkflow(adapter=adapter)

    result = workflow.run(
        df,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=privacy_goal,
    )

    assert len(result.dataframe) == 3


# ---------------------------------------------------------------------------
# Tests: RewriteGenerationWorkflow.run() — entity rows
# ---------------------------------------------------------------------------


def test_run_output_length_equals_input_length_mixed(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection,
    privacy_goal: PrivacyGoal,
    stub_sensitivity_disposition: dict,
    stub_replacement_map: dict,
    stub_entities_by_value_with_entities: dict,
) -> None:
    df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works here", "no entities here"],
            COL_TAGGED_TEXT: [
                "⟦Alice|first_name⟧ works here",
                "no entities here",
            ],
            COL_ENTITIES_BY_VALUE: [
                stub_entities_by_value_with_entities,
                {"entities_by_value": []},
            ],
            COL_SENSITIVITY_DISPOSITION: [stub_sensitivity_disposition, {"sensitivity_disposition": []}],
            COL_REPLACEMENT_MAP: [stub_replacement_map, {"replacements": []}],
        }
    )

    replace_result_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works here"],
            COL_TAGGED_TEXT: ["⟦Alice|first_name⟧ works here"],
            COL_ENTITIES_BY_VALUE: [stub_entities_by_value_with_entities],
            COL_SENSITIVITY_DISPOSITION: [stub_sensitivity_disposition],
            COL_REPLACEMENT_MAP: [stub_replacement_map],
            "_row_order": [0],
        }
    )

    rewrite_result_df = replace_result_df.copy()
    rewrite_result_df[COL_FULL_REWRITE] = [RewriteOutputSchema(rewritten_text="Maria works here")]
    rewrite_result_df[COL_REWRITTEN_TEXT] = ["Maria works here"]

    adapter = Mock()
    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=replace_result_df, failed_records=[]),
        WorkflowRunResult(dataframe=rewrite_result_df, failed_records=[]),
    ]

    workflow = RewriteGenerationWorkflow(adapter=adapter)
    result = workflow.run(
        df,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=privacy_goal,
    )

    assert len(result.dataframe) == 2


def test_run_preserves_original_row_order_with_mixed_entity_and_passthrough_rows(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection,
    privacy_goal: PrivacyGoal,
    stub_sensitivity_disposition: dict,
    stub_replacement_map: dict,
    stub_entities_by_value_with_entities: dict,
) -> None:
    df = pd.DataFrame(
        {
            COL_TEXT: ["first passthrough", "Alice works here", "last passthrough"],
            COL_TAGGED_TEXT: ["first passthrough", "⟦Alice|first_name⟧ works here", "last passthrough"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": []},
                stub_entities_by_value_with_entities,
                {"entities_by_value": []},
            ],
            COL_SENSITIVITY_DISPOSITION: [
                {"sensitivity_disposition": []},
                stub_sensitivity_disposition,
                {"sensitivity_disposition": []},
            ],
            COL_REPLACEMENT_MAP: [{"replacements": []}, stub_replacement_map, {"replacements": []}],
        }
    )

    replace_result_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works here"],
            COL_TAGGED_TEXT: ["⟦Alice|first_name⟧ works here"],
            COL_ENTITIES_BY_VALUE: [stub_entities_by_value_with_entities],
            COL_SENSITIVITY_DISPOSITION: [stub_sensitivity_disposition],
            COL_REPLACEMENT_MAP: [stub_replacement_map],
            "_row_order": [1],
        }
    )

    rewrite_result_df = replace_result_df.copy()
    rewrite_result_df[COL_FULL_REWRITE] = [{"rewritten_text": "Maria works here"}]
    rewrite_result_df[COL_REWRITTEN_TEXT] = ["Maria works here"]

    adapter = Mock()
    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=replace_result_df, failed_records=[]),
        WorkflowRunResult(dataframe=rewrite_result_df, failed_records=[]),
    ]

    workflow = RewriteGenerationWorkflow(adapter=adapter)
    result = workflow.run(
        df,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=privacy_goal,
    )

    assert result.dataframe[COL_TEXT].tolist() == ["first passthrough", "Alice works here", "last passthrough"]
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == [
        "first passthrough",
        "Maria works here",
        "last passthrough",
    ]


def test_run_propagates_dataframe_attrs(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection,
    privacy_goal: PrivacyGoal,
    stub_df_no_entities: pd.DataFrame,
) -> None:
    stub_df_no_entities.attrs["original_text_column"] = "bio"
    adapter = Mock()
    workflow = RewriteGenerationWorkflow(adapter=adapter)

    result = workflow.run(
        stub_df_no_entities,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=privacy_goal,
    )

    assert result.dataframe.attrs.get("original_text_column") == "bio"


def test_run_collects_failed_records_from_replace_and_rewrite(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection,
    privacy_goal: PrivacyGoal,
    stub_df_with_entities: pd.DataFrame,
    stub_replacement_map: dict,
    stub_entities_by_value_with_entities: dict,
    stub_sensitivity_disposition: dict,
) -> None:
    replace_result_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme Corp"],
            COL_TAGGED_TEXT: ["⟦Alice|first_name⟧ works at ⟦Acme Corp|company_name⟧"],
            COL_ENTITIES_BY_VALUE: [stub_entities_by_value_with_entities],
            COL_SENSITIVITY_DISPOSITION: [stub_sensitivity_disposition],
            COL_REPLACEMENT_MAP: [stub_replacement_map],
            "_row_order": [0],
        }
    )

    rewrite_result_df = replace_result_df.copy()
    rewrite_result_df[COL_REWRITTEN_TEXT] = ["Maria works at TechCorp"]

    failed_replace = FailedRecord(record_id="abc", step="replace-map-generation", reason="timeout")
    failed_rewrite = FailedRecord(record_id="def", step="rewrite-generation", reason="timeout")

    adapter = Mock()
    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=replace_result_df, failed_records=[failed_replace]),
        WorkflowRunResult(dataframe=rewrite_result_df, failed_records=[failed_rewrite]),
    ]

    workflow = RewriteGenerationWorkflow(adapter=adapter)
    result = workflow.run(
        stub_df_with_entities,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=privacy_goal,
    )

    assert failed_replace in result.failed_records
    assert failed_rewrite in result.failed_records
