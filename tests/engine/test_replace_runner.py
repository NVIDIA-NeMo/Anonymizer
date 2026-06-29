# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import EvaluateModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Hash, Redact, Substitute
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_JUDGE,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_COVERAGE,
    COL_ENTITY_COVERAGE_JUDGE,
    COL_FINAL_ENTITIES,
    COL_RELATIONAL_CONSISTENCY_JUDGE,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TEXT,
    COL_TYPE_FIDELITY_JUDGE,
    COL_TYPE_FIDELITY_VALID,
)
from anonymizer.engine.evaluation.detection_judge import DetectionJudgeWorkflow
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgeWorkflow
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgeWorkflow
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgeWorkflow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, WorkflowRunResult
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


def test_evaluate_uses_merged_dd_workflow_for_judges(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    """``evaluate()`` runs all 4 judges as columns of a SINGLE DD workflow call
    (DataDesigner parallelizes the columns internally — no Python threads)."""

    # Trace-shaped input: simulates a dataframe returned by a prior ``run()``.
    saved_trace = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_FINAL_ENTITIES: [{"entities": []}],
            COL_REPLACED_TEXT: ["Maya works at NovaCorp"],
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
    )

    judge_defaults = {
        COL_ENTITY_COVERAGE_JUDGE: {"leaked_entities": []},
        COL_TYPE_FIDELITY_JUDGE: {"all_valid": True, "invalid_replacements": []},
        COL_RELATIONAL_CONSISTENCY_JUDGE: {"all_consistent": True, "relations": []},
        COL_ATTRIBUTE_FIDELITY_JUDGE: {"all_valid": True, "entities": []},
    }

    def fake_run_workflow(df: pd.DataFrame, *, columns, **_: object) -> WorkflowRunResult:
        out = df.copy()
        for column in columns:
            out[column.name] = [judge_defaults[column.name]] * len(out)
        return WorkflowRunResult(dataframe=out, failed_records=[])

    def fake_attach_ids(df: pd.DataFrame) -> pd.DataFrame:
        if RECORD_ID_COLUMN in df.columns:
            return df.copy()
        out = df.copy()
        out[RECORD_ID_COLUMN] = [f"id-{i}" for i in range(len(out))]
        return out

    adapter = Mock()
    adapter.run_workflow.side_effect = fake_run_workflow
    adapter._attach_record_ids.side_effect = fake_attach_ids

    runner = ReplacementWorkflow(
        detection_judge=DetectionJudgeWorkflow(adapter=adapter),
        type_fidelity_judge=TypeFidelityJudgeWorkflow(adapter=adapter),
        relational_consistency_judge=RelationalConsistencyJudgeWorkflow(adapter=adapter),
        attribute_fidelity_judge=AttributeFidelityJudgeWorkflow(adapter=adapter),
        adapter=adapter,
    )

    result = runner.evaluate(
        saved_trace,
        replace_method=Substitute(),
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
    )

    # Exactly ONE adapter call for the judges step (proves merge, not 4 separate workflows).
    assert adapter.run_workflow.call_count == 1
    call_columns = adapter.run_workflow.call_args.kwargs["columns"]
    assert {c.name for c in call_columns} == set(judge_defaults)

    # And each judge's output column ended up on the result.
    # Entity coverage is a float (1.0 = full coverage); replace judges use bool True.
    assert COL_ENTITY_COVERAGE in result.dataframe.columns
    assert result.dataframe[COL_ENTITY_COVERAGE].iloc[0] == 1.0
    for col in (
        COL_TYPE_FIDELITY_VALID,
        COL_RELATIONAL_CONSISTENCY_VALID,
        COL_ATTRIBUTE_FIDELITY_VALID,
    ):
        assert col in result.dataframe.columns, f"missing column: {col}"
        assert bool(result.dataframe[col].iloc[0]) is True


def test_evaluate_preserves_all_rows_when_llm_drops_some(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    """Evaluation is non-critical: rows the LLM drops (parse error, timeout,
    etc.) must still appear in the result with *_valid=None ("Unavailable"),
    not vanish from a previously successful preview/run.
    """
    saved_trace = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme", "Bob works at Globex"],
            COL_FINAL_ENTITIES: [{"entities": []}, {"entities": []}],
            COL_REPLACED_TEXT: ["Maya works at NovaCorp", "Carl works at Initech"],
            COL_REPLACEMENT_MAP: [
                {
                    "replacements": [
                        {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                        {"original": "Acme", "label": "organization", "synthetic": "NovaCorp"},
                    ]
                },
                {
                    "replacements": [
                        {"original": "Bob", "label": "first_name", "synthetic": "Carl"},
                        {"original": "Globex", "label": "organization", "synthetic": "Initech"},
                    ]
                },
            ],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {"entities_by_value": [{"value": "Bob", "labels": ["first_name"]}]},
            ],
        }
    )

    judge_payload = {
        COL_ENTITY_COVERAGE_JUDGE: {"leaked_entities": []},
        COL_TYPE_FIDELITY_JUDGE: {"all_valid": True, "invalid_replacements": []},
        COL_RELATIONAL_CONSISTENCY_JUDGE: {"all_consistent": True, "relations": []},
        COL_ATTRIBUTE_FIDELITY_JUDGE: {"all_valid": True, "entities": []},
    }

    def fake_attach_ids(df: pd.DataFrame) -> pd.DataFrame:
        if RECORD_ID_COLUMN in df.columns:
            return df.copy()
        out = df.copy()
        out[RECORD_ID_COLUMN] = [f"id-{i}" for i in range(len(out))]
        return out

    def fake_run_workflow(df: pd.DataFrame, *, columns, **_: object) -> WorkflowRunResult:
        # Simulate the LLM successfully judging only the first row;
        # the second row got dropped during the workflow.
        kept = df.iloc[:1].copy()
        for column in columns:
            kept[column.name] = [judge_payload[column.name]] * len(kept)
        dropped = FailedRecord(record_id="id-1", step="replace-judges", reason="parse error")
        return WorkflowRunResult(dataframe=kept, failed_records=[dropped])

    adapter = Mock()
    adapter._attach_record_ids.side_effect = fake_attach_ids
    adapter.run_workflow.side_effect = fake_run_workflow

    runner = ReplacementWorkflow(
        detection_judge=DetectionJudgeWorkflow(adapter=adapter),
        type_fidelity_judge=TypeFidelityJudgeWorkflow(adapter=adapter),
        relational_consistency_judge=RelationalConsistencyJudgeWorkflow(adapter=adapter),
        attribute_fidelity_judge=AttributeFidelityJudgeWorkflow(adapter=adapter),
        adapter=adapter,
    )
    result = runner.evaluate(
        saved_trace,
        replace_method=Substitute(),
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
    )

    # Row count is preserved end-to-end.
    assert len(result.dataframe) == 2
    # First row got a real verdict (entity_coverage is a float, replace judges are bool).
    assert result.dataframe[COL_ENTITY_COVERAGE].iloc[0] == 1.0
    assert bool(result.dataframe[COL_TYPE_FIDELITY_VALID].iloc[0]) is True
    # Second row (LLM-dropped) is surfaced as Unavailable, not dropped.
    # Float column: None becomes NaN in pandas, so use pd.isna instead of `is None`.
    assert pd.isna(result.dataframe[COL_ENTITY_COVERAGE].iloc[1])
    assert result.dataframe[COL_TYPE_FIDELITY_VALID].iloc[1] is None
    assert result.dataframe[COL_RELATIONAL_CONSISTENCY_VALID].iloc[1] is None
    assert result.dataframe[COL_ATTRIBUTE_FIDELITY_VALID].iloc[1] is None
    # The drop is still visible via failed_records for downstream observability.
    assert len(result.failed_records) == 1
    assert result.failed_records[0].record_id == "id-1"


def test_runner_does_not_invoke_judges(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    stub_entities: list[dict],
) -> None:
    """``ReplacementWorkflow.run()`` only does the replace step — never the judges.

    The judges live behind a separate ``evaluate()`` call.
    """
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
    adapter = Mock()
    runner = ReplacementWorkflow(
        llm_workflow=llm_workflow,
        detection_judge=detection_judge,
        type_fidelity_judge=type_fidelity_judge,
        relational_consistency_judge=relational_judge,
        attribute_fidelity_judge=attribute_judge,
        adapter=adapter,
    )

    result = runner.run(
        pd.DataFrame({COL_TEXT: ["Alice works at Acme"], COL_FINAL_ENTITIES: [{"entities": []}]}),
        replace_method=Substitute(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    detection_judge.evaluate.assert_not_called()
    type_fidelity_judge.evaluate.assert_not_called()
    relational_judge.evaluate.assert_not_called()
    attribute_judge.evaluate.assert_not_called()
    adapter.run_workflow.assert_not_called()
    for col in (
        COL_DETECTION_VALID,
        COL_TYPE_FIDELITY_VALID,
        COL_ATTRIBUTE_FIDELITY_VALID,
        COL_RELATIONAL_CONSISTENCY_VALID,
    ):
        assert col not in result.dataframe.columns
    assert result.dataframe[COL_REPLACED_TEXT].iloc[0] == "Maya works at NovaCorp"


def test_evaluate_raises_on_missing_required_columns(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    """``evaluate()`` rejects dataframes lacking the columns the judges need,
    with a message that hints at the trace_dataframe workflow."""
    runner = ReplacementWorkflow(
        detection_judge=DetectionJudgeWorkflow(adapter=Mock()),
        type_fidelity_judge=TypeFidelityJudgeWorkflow(adapter=Mock()),
        relational_consistency_judge=RelationalConsistencyJudgeWorkflow(adapter=Mock()),
        attribute_fidelity_judge=AttributeFidelityJudgeWorkflow(adapter=Mock()),
        adapter=Mock(),
    )
    bare_df = pd.DataFrame({COL_TEXT: ["Alice"]})  # missing _entities_by_value and _replacement_map
    with pytest.raises(ValueError, match="trace_dataframe"):
        runner.evaluate(
            bare_df,
            replace_method=Substitute(),
            model_configs=stub_model_configs,
            selected_models=stub_evaluate_model_selection,
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
