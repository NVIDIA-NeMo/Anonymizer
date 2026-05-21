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
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_JUDGE,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_RELATIONAL_CONSISTENCY_JUDGE,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TEXT,
    COL_TYPE_FIDELITY_JUDGE,
    COL_TYPE_FIDELITY_VALID,
)
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.engine.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.replace.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult
from anonymizer.engine.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.replace.strategies import apply_replacement_map
from anonymizer.engine.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
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


def test_substitute_runner_uses_merged_dd_workflow_for_judges(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    stub_entities: list[dict],
) -> None:
    """When an adapter is wired, all 4 judges run as columns of a SINGLE DD workflow
    call (DataDesigner parallelizes the columns internally — no Python threads)."""

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
                COL_ENTITIES_BY_VALUE: [
                    {
                        "entities_by_value": [
                            {"value": "Alice", "labels": ["first_name"]},
                            {"value": "Acme", "labels": ["organization"]},
                        ]
                    }
                ],
            }
        ),
        failed_records=[],
    )

    judge_defaults = {
        COL_DETECTION_JUDGE: {"all_valid": True, "invalid_entities": []},
        COL_TYPE_FIDELITY_JUDGE: {"all_valid": True, "invalid_replacements": []},
        COL_RELATIONAL_CONSISTENCY_JUDGE: {"all_consistent": True, "relations": []},
        COL_ATTRIBUTE_FIDELITY_JUDGE: {"all_valid": True, "entities": []},
    }

    def fake_run_workflow(df: pd.DataFrame, *, columns, **_: object) -> WorkflowRunResult:
        out = df.copy()
        for column in columns:
            out[column.name] = [judge_defaults[column.name]] * len(out)
        return WorkflowRunResult(dataframe=out, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = fake_run_workflow

    runner = ReplacementWorkflow(
        llm_workflow=llm_workflow,
        detection_judge=DetectionJudgeWorkflow(adapter=adapter),
        type_fidelity_judge=TypeFidelityJudgeWorkflow(adapter=adapter),
        relational_consistency_judge=RelationalConsistencyJudgeWorkflow(adapter=adapter),
        attribute_fidelity_judge=AttributeFidelityJudgeWorkflow(adapter=adapter),
        adapter=adapter,
    )

    result = runner.run(
        pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        replace_method=Substitute(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    # Exactly ONE adapter call for the judges step (proves merge, not 4 separate workflows).
    assert adapter.run_workflow.call_count == 1
    call_columns = adapter.run_workflow.call_args.kwargs["columns"]
    assert {c.name for c in call_columns} == set(judge_defaults)

    # And each judge's VALID column ended up on the result, with True (default payload above).
    for col in (
        COL_DETECTION_VALID,
        COL_TYPE_FIDELITY_VALID,
        COL_RELATIONAL_CONSISTENCY_VALID,
        COL_ATTRIBUTE_FIDELITY_VALID,
    ):
        assert col in result.dataframe.columns, f"missing column: {col}"
        assert bool(result.dataframe[col].iloc[0]) is True


def test_substitute_runner_skips_all_judges_when_evaluation_disabled(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    stub_entities: list[dict],
) -> None:
    """run_replace_evaluation=False short-circuits before any LLM judge runs."""
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
        failed_records=[],
    )
    detection_judge = Mock()
    type_fidelity_judge = Mock()
    relational_judge = Mock()
    attribute_judge = Mock()
    runner = ReplacementWorkflow(
        llm_workflow=llm_workflow,
        detection_judge=detection_judge,
        type_fidelity_judge=type_fidelity_judge,
        relational_consistency_judge=relational_judge,
        attribute_fidelity_judge=attribute_judge,
    )

    result = runner.run(
        pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        replace_method=Substitute(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
        run_replace_evaluation=False,
    )

    detection_judge.evaluate.assert_not_called()
    type_fidelity_judge.evaluate.assert_not_called()
    relational_judge.evaluate.assert_not_called()
    attribute_judge.evaluate.assert_not_called()
    for col in (
        COL_DETECTION_VALID,
        COL_TYPE_FIDELITY_VALID,
        COL_ATTRIBUTE_FIDELITY_VALID,
        COL_RELATIONAL_CONSISTENCY_VALID,
    ):
        assert col not in result.dataframe.columns
    assert result.dataframe[COL_REPLACED_TEXT].iloc[0] == "Maya works at NovaCorp"


def test_redact_runner_skips_detection_judge_when_evaluation_disabled(
    stub_dataframe_with_entities: pd.DataFrame,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    """Non-Substitute paths also honour run_replace_evaluation=False."""
    detection_judge = Mock()
    runner = ReplacementWorkflow(detection_judge=detection_judge)

    result = runner.run(
        stub_dataframe_with_entities,
        replace_method=Redact(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
        run_replace_evaluation=False,
    )

    detection_judge.evaluate.assert_not_called()
    assert COL_DETECTION_VALID not in result.dataframe.columns


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
