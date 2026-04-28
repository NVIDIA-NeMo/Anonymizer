# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.ndd.adapter import WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow


def test_generate_map_only_returns_replacement_map(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Workflow operates on plain DataFrames; pipeline metadata is the orchestrator's job."""
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                "text": ["Alice works at Acme"],
                COL_REPLACEMENT_MAP: [{"replacements": []}],
                "_anonymizer_row_order": [0],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            "text": ["Alice works at Acme"],
            "_entities_by_value": [[{"value": "Alice", "labels": ["first_name"]}]],
            "tagged_text": ["<<SENSITIVE:first_name>>Alice<</SENSITIVE:first_name>> works at Acme"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert COL_REPLACEMENT_MAP in result.dataframe.columns


def test_generate_map_only_skips_llm_for_rows_without_entities(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["CT guided biopsy revealed a fatty mass which was diagnosed as lipoma."],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}],
            "tagged_text": ["CT guided biopsy revealed a fatty mass which was diagnosed as lipoma."],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    adapter.run_workflow.assert_not_called()
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {"replacements": []}


def test_generate_map_only_filters_hallucinated_replacements(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
                "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
                "_anonymizer_row_order": [0],
                COL_REPLACEMENT_MAP: [
                    {
                        "replacements": [
                            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                            {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
                        ]
                    }
                ],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
            "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0] == {
        "replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]
    }


def test_generate_map_only_preserves_original_anonymizer_row_order_with_mixed_rows(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
                "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
                "_anonymizer_row_order": [1],
                COL_REPLACEMENT_MAP: [
                    {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]}
                ],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            COL_TEXT: ["No entities here", "Alice works at Acme", "Still no entities"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": []},
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {"entities_by_value": []},
            ],
            "tagged_text": [
                "No entities here",
                "<<PII:first_name>>Alice<</PII:first_name>> works at Acme",
                "Still no entities",
            ],
        }
    )

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert result.dataframe[COL_TEXT].tolist() == ["No entities here", "Alice works at Acme", "Still no entities"]
    assert result.dataframe[COL_REPLACEMENT_MAP].tolist() == [
        {"replacements": []},
        {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]},
        {"replacements": []},
    ]
