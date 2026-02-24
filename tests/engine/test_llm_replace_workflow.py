# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import COL_REPLACEMENT_MAP
from anonymizer.engine.ndd.adapter import WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow


def test_generate_map_only_preserves_input_attrs(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=pd.DataFrame(
            {
                "text": ["Alice works at Acme"],
                COL_REPLACEMENT_MAP: [{"replacements": []}],
            }
        ),
        failed_records=[],
    )
    workflow = LlmReplaceWorkflow(adapter=adapter)

    input_df = pd.DataFrame(
        {
            "text": ["Alice works at Acme"],
            "_entities_by_value": [[{"value": "Alice", "labels": ["first_name"]}]],
            "tagged_text": ["<<PII:first_name>>Alice<</PII:first_name>> works at Acme"],
        }
    )
    input_df.attrs["original_text_column"] = "bio"

    result = workflow.generate_map_only(
        input_df,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert result.dataframe.attrs["original_text_column"] == "bio"
