# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import Hash, Redact, Substitute
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.replace.strategies import apply_replacement_map
from anonymizer.engine.schemas import EntitiesSchema


def _build_entities_payload(payload_kind: str) -> object:
    entities_list = [
        {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
        {"value": "Acme", "label": "organization", "start_position": 15, "end_position": 19},
    ]
    if payload_kind == "dict_wrapper":
        return {"entities": entities_list}
    if payload_kind == "numpy_wrapped_dict_wrapper":
        return {"entities": np.array(entities_list, dtype=object)}
    if payload_kind == "entities_schema":
        return EntitiesSchema(entities=entities_list)
    raise ValueError(f"Unsupported payload kind: {payload_kind}")


def test_local_replace_runner_uses_strategy_directly(
    stub_dataframe_with_entities: pd.DataFrame,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplacementWorkflow()
    result = runner.run(
        stub_dataframe_with_entities,
        replace_method=Redact(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    assert result.failed_records == []
    assert COL_REPLACED_TEXT in result.dataframe.columns
    assert "[REDACTED_FIRST_NAME]" in result.dataframe[COL_REPLACED_TEXT].iloc[0]


def test_local_replace_runner_with_custom_format_template(
    stub_dataframe_with_entities: pd.DataFrame,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplacementWorkflow()
    result = runner.run(
        stub_dataframe_with_entities,
        replace_method=Redact(format_template="***"),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    assert result.failed_records == []
    assert result.dataframe[COL_REPLACED_TEXT].iloc[0] == "*** works at ***"


def test_substitute_runner_applies_generated_map(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    stub_entities: list[dict],
) -> None:
    llm_workflow = Mock()
    llm_workflow.generate_map_only.return_value = LlmReplaceResult(
        dataframe=pd.DataFrame(
            {
                COL_TEXT: ["Alice works at Acme"],
                COL_FINAL_ENTITIES: [{"entities": stub_entities}],
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
        failed_records=[FailedRecord(record_id="r1", step="replace-map-generation", reason="none")],
    )
    runner = ReplacementWorkflow(llm_workflow=llm_workflow)
    result = runner.run(
        pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        replace_method=Substitute(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    assert llm_workflow.generate_map_only.call_count == 1
    assert result.dataframe[COL_REPLACED_TEXT].iloc[0] == "Maya works at NovaCorp"
    assert len(result.failed_records) == 1


def test_substitute_without_workflow_raises(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplacementWorkflow()
    with pytest.raises(ValueError, match="llm_workflow"):
        runner.run(
            pd.DataFrame({COL_TEXT: ["Alice"], COL_FINAL_ENTITIES: [{"entities": []}]}),
            replace_method=Substitute(),
            model_configs=stub_model_configs,
            selected_models=stub_replace_model_selection,
        )


def test_apply_replacement_map_handles_string_map() -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["abc Alice xyz"],
            COL_FINAL_ENTITIES: [
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 4, "end_position": 9}]}
            ],
            COL_REPLACEMENT_MAP: ['{"replacements":[{"original":"Alice","label":"first_name","synthetic":"Elena"}]}'],
        }
    )
    output_df = apply_replacement_map(dataframe)
    assert output_df[COL_REPLACED_TEXT].iloc[0] == "abc Elena xyz"


@pytest.mark.parametrize(
    "payload_kind",
    ["dict_wrapper", "numpy_wrapped_dict_wrapper", "entities_schema"],
)
def test_apply_replacement_map_handles_entity_payload_shapes(payload_kind: str) -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_FINAL_ENTITIES: [_build_entities_payload(payload_kind)],
            COL_REPLACEMENT_MAP: [
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
    assert output_df[COL_REPLACED_TEXT].iloc[0] == "Maya works at NovaCorp"


def test_hash_strategy_executes(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplacementWorkflow()
    input_df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_FINAL_ENTITIES: [
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
            ],
        }
    )
    result = runner.run(
        input_df,
        replace_method=Hash(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    assert result.failed_records == []
    assert result.dataframe[COL_REPLACED_TEXT].iloc[0] == "<HASH_FIRST_NAME_3bc51062973c>"


def test_redact_none_replaces_all(
    stub_dataframe_with_entities: pd.DataFrame,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    runner = ReplacementWorkflow()
    result = runner.run(
        stub_dataframe_with_entities,
        replace_method=Redact(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    replaced = result.dataframe[COL_REPLACED_TEXT].iloc[0]
    assert "Alice" not in replaced
    assert "Acme" not in replaced


# --- detection → replace integration ---


def test_detection_output_flows_through_redact(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Smoke test: dict-wrapped final_entities from detection must work with Redact."""
    detection_output = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_FINAL_ENTITIES: [
                {
                    "entities": [
                        {
                            "id": "first_name_0_5",
                            "value": "Alice",
                            "label": "first_name",
                            "start_position": 0,
                            "end_position": 5,
                            "score": 1.0,
                            "source": "detector",
                        },
                        {
                            "id": "org_15_19",
                            "value": "Acme",
                            "label": "organization",
                            "start_position": 15,
                            "end_position": 19,
                            "score": 0.9,
                            "source": "augmenter",
                        },
                    ]
                }
            ],
        }
    )
    runner = ReplacementWorkflow()
    result = runner.run(
        detection_output,
        replace_method=Redact(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    replaced = result.dataframe[COL_REPLACED_TEXT].iloc[0]
    assert "Alice" not in replaced
    assert "Acme" not in replaced
    assert "[REDACTED_FIRST_NAME]" in replaced
    assert "[REDACTED_ORGANIZATION]" in replaced


def test_detection_output_with_numpy_entities_flows_through_redact(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Regression: parquet round-trip produces {"entities": numpy_array}."""
    detection_output = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_FINAL_ENTITIES: [
                {
                    "entities": np.array(
                        [
                            {
                                "id": "first_name_0_5",
                                "value": "Alice",
                                "label": "first_name",
                                "start_position": 0,
                                "end_position": 5,
                            },
                            {
                                "id": "org_15_19",
                                "value": "Acme",
                                "label": "organization",
                                "start_position": 15,
                                "end_position": 19,
                            },
                        ],
                        dtype=object,
                    )
                }
            ],
        }
    )
    runner = ReplacementWorkflow()
    result = runner.run(
        detection_output,
        replace_method=Redact(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    replaced = result.dataframe[COL_REPLACED_TEXT].iloc[0]
    assert "Alice" not in replaced
    assert "[REDACTED_FIRST_NAME]" in replaced
