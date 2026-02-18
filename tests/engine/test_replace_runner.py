# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import HashReplace, LLMReplace, RedactReplace
from anonymizer.engine.detection.constants import COL_DETECTED_ENTITIES, COL_TEXT
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult
from anonymizer.engine.replace.replace_runner import ReplaceRunner
from anonymizer.engine.replace.strategies import apply_replacement_map


def test_local_replace_runner_uses_strategy_directly(
    stub_dataframe_with_entities: pd.DataFrame,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplaceRunner()
    output_df, failures = runner.run(
        stub_dataframe_with_entities,
        replace_strategy=RedactReplace(),
        model_configs=stub_model_configs,
        model_providers=None,
        selected_models=stub_replace_model_selection,
    )
    assert failures == []
    assert "replaced_text" in output_df.columns
    assert "[REDACTED_FIRST_NAME]" in output_df["replaced_text"].iloc[0]


def test_local_replace_runner_with_custom_redact_template(
    stub_dataframe_with_entities: pd.DataFrame,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplaceRunner()
    output_df, failures = runner.run(
        stub_dataframe_with_entities,
        replace_strategy=RedactReplace(redact_template="***"),
        model_configs=stub_model_configs,
        model_providers=None,
        selected_models=stub_replace_model_selection,
    )
    assert failures == []
    assert output_df["replaced_text"].iloc[0] == "*** works at ***"


def test_llm_replace_runner_applies_generated_map(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    stub_entities: list[dict],
) -> None:
    llm_workflow = Mock()
    llm_workflow.generate_map_only.return_value = LlmReplaceResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_DETECTED_ENTITIES: [stub_entities],
                "_replacement_map": [
                    {
                        "replacements": [
                            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                            {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
                        ]
                    }
                ],
            }
        ),
        failed_records=[FailedRecord(record_id="r1", step="replace-map-generation", reason="none")],
    )
    runner = ReplaceRunner(llm_workflow=llm_workflow)
    output_df, failures = runner.run(
        pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_DETECTED_ENTITIES: [[]]}),
        replace_strategy=LLMReplace(),
        model_configs=stub_model_configs,
        model_providers=None,
        selected_models=stub_replace_model_selection,
    )
    assert llm_workflow.generate_map_only.call_count == 1
    assert output_df["replaced_text"].iloc[0] == "Maya works at NovaCorp"
    assert len(failures) == 1


def test_llm_replace_without_workflow_raises(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplaceRunner()
    with pytest.raises(ValueError, match="llm_workflow"):
        runner.run(
            pd.DataFrame({COL_TEXT: ["Alice"], COL_DETECTED_ENTITIES: [[]]}),
            replace_strategy=LLMReplace(),
            model_configs=stub_model_configs,
            model_providers=None,
            selected_models=stub_replace_model_selection,
        )


def test_apply_replacement_map_handles_string_map() -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["abc Alice xyz"],
            COL_DETECTED_ENTITIES: [
                [{"value": "Alice", "label": "first_name", "start_position": 4, "end_position": 9}]
            ],
            "_replacement_map": ['{"replacements":[{"original":"Alice","label":"first_name","synthetic":"Elena"}]}'],
        }
    )
    output_df = apply_replacement_map(dataframe)
    assert output_df["replaced_text"].iloc[0] == "abc Elena xyz"


def test_apply_replacement_map_handles_numpy_array_entities() -> None:
    """Entities may come back as numpy arrays after parquet round-trip through DataDesigner."""
    entities = np.array(
        [
            {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
            {"value": "Acme", "label": "organization", "start_position": 15, "end_position": 19},
        ],
        dtype=object,
    )
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_DETECTED_ENTITIES: [entities],
            "_replacement_map": [
                {
                    "replacements": [
                        {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                        {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
                    ]
                }
            ],
        }
    )
    output_df = apply_replacement_map(dataframe)
    assert output_df["replaced_text"].iloc[0] == "Maya works at NovaCorp"


def test_hash_replace_strategy_executes(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplaceRunner()
    input_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_DETECTED_ENTITIES: [
                [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]
            ],
        }
    )
    output_df, failures = runner.run(
        input_df,
        replace_strategy=HashReplace(),
        model_configs=stub_model_configs,
        model_providers=None,
        selected_models=stub_replace_model_selection,
    )
    assert failures == []
    assert output_df["replaced_text"].iloc[0] == "<HASH_FIRST_NAME_3bc51062973c>"
