# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from data_designer.config import SkipConfig, custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import ModelSelection, ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_FULL_REWRITE,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_REPAIR_ITERATIONS,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_INITIAL,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.rewrite.combined_workflow import (
    CombinedRewriteGraph,
    CombinedRewriteWorkflow,
    EvaluationState,
    RepairState,
)
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.measurement import MeasurementCollector, measurement_session

_PRIVACY_GOAL = PrivacyGoal(
    protect="All direct identifiers including names, locations, and contact details",
    preserve="Career trajectory, skills, and professional context in abstract terms",
)
_REPAIRS_NEEDED = "repairs_needed"
_REPLACE_PATCH = "anonymizer.engine.rewrite.rewrite_workflow.LlmReplaceWorkflow"


@custom_column_generator(required_columns=[_REPAIRS_NEEDED])
def _initial_rewrite(row: dict[str, Any]) -> dict[str, Any]:
    row[COL_REWRITTEN_TEXT_INITIAL] = "rewrite-0"
    return row


def _deterministic_evaluation_column(state: EvaluationState) -> CustomColumnConfig:
    side_effect_columns = [
        state.privacy_reanswer,
        state.quality_compare,
        state.utility_score,
        state.leakage_mass,
        state.weighted_leakage_rate,
        state.any_high_leaked,
        state.needs_repair,
    ]

    @custom_column_generator(
        required_columns=[_REPAIRS_NEEDED, state.rewritten_text],
        side_effect_columns=side_effect_columns,
    )
    def evaluate(row: dict[str, Any]) -> dict[str, Any]:
        needs_repair = int(row[_REPAIRS_NEEDED]) > state.iteration
        items: list[Any] | np.ndarray = [] if state.iteration == 0 else np.array([], dtype=object)
        row[state.quality_reanswer] = {"answers": items}
        row[state.privacy_reanswer] = {"answers": items}
        row[state.quality_compare] = {"per_item": items}
        row[state.utility_score] = 0.5 if needs_repair else 1.0
        row[state.leakage_mass] = 1.0 if needs_repair else 0.0
        row[state.weighted_leakage_rate] = 1.0 if needs_repair else 0.0
        row[state.any_high_leaked] = needs_repair
        row[state.needs_repair] = needs_repair
        return row

    return CustomColumnConfig(name=state.quality_reanswer, generator_function=evaluate)


def _deterministic_repair_column(previous: EvaluationState, state: RepairState) -> CustomColumnConfig:
    @custom_column_generator(required_columns=[previous.needs_repair])
    def repair(row: dict[str, Any]) -> dict[str, Any]:
        row[state.rewritten_text] = f"rewrite-{state.iteration + 1}"
        return row

    return CustomColumnConfig(
        name=state.rewritten_text,
        generator_function=repair,
        skip=SkipConfig(when=f"{{{{ not {previous.needs_repair} }}}}"),
    )


def _deterministic_columns(graph: CombinedRewriteGraph) -> list[ColumnConfigT]:
    columns: list[ColumnConfigT] = [
        CustomColumnConfig(
            name=graph.evaluation_states[0].rewritten_text,
            generator_function=_initial_rewrite,
        ),
        _deterministic_evaluation_column(graph.evaluation_states[0]),
    ]
    for previous, repair, current in zip(
        graph.evaluation_states,
        graph.repair_states,
        graph.evaluation_states[1:],
    ):
        columns.extend(
            [
                _deterministic_repair_column(previous, repair),
                _deterministic_evaluation_column(current),
            ]
        )
    columns.append(graph.columns[-1])
    return columns


@pytest.mark.parametrize("max_repair_iterations", [0, 1, 3])
def test_graph_unrolls_conditional_repairs(
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    max_repair_iterations: int,
) -> None:
    graph = CombinedRewriteWorkflow(adapter=Mock()).build_graph(
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=max_repair_iterations),
    )

    assert len(graph.evaluation_states) == max_repair_iterations + 1
    assert len(graph.repair_states) == max_repair_iterations
    by_name = {column.name: column for column in graph.columns}
    for previous, repair in zip(graph.evaluation_states, graph.repair_states):
        condition = by_name[repair.leaked_items].skip
        assert condition is not None
        assert previous.needs_repair in condition.columns


def test_finalizer_selects_last_executed_iteration(
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    graph = CombinedRewriteWorkflow(adapter=Mock()).build_graph(
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
    )
    initial, repaired, skipped = graph.evaluation_states
    row = {
        initial.rewritten_text: "Initial rewrite",
        initial.quality_reanswer: {"answers": []},
        initial.privacy_reanswer: {"answers": []},
        initial.quality_compare: {"per_item": []},
        initial.utility_score: 0.7,
        initial.leakage_mass: 2.0,
        initial.weighted_leakage_rate: 0.8,
        initial.any_high_leaked: True,
        initial.needs_repair: True,
        repaired.rewritten_text: "Repaired rewrite",
        repaired.quality_reanswer: {"answers": []},
        repaired.privacy_reanswer: {"answers": []},
        repaired.quality_compare: {"per_item": []},
        repaired.utility_score: 0.9,
        repaired.leakage_mass: 0.1,
        repaired.weighted_leakage_rate: 0.05,
        repaired.any_high_leaked: False,
        repaired.needs_repair: False,
        skipped.needs_repair: None,
    }
    finalizer = graph.columns[-1]
    assert isinstance(finalizer, CustomColumnConfig)

    result = finalizer.generator_function(row, finalizer.generator_params)

    assert result[COL_REWRITTEN_TEXT] == "Repaired rewrite"
    assert result[COL_UTILITY_SCORE] == 0.9
    assert result[COL_LEAKAGE_MASS] == 0.1
    assert result[COL_WEIGHTED_LEAKAGE_RATE] == 0.05
    assert result[COL_ANY_HIGH_LEAKED] is False
    assert result[COL_NEEDS_REPAIR] is False
    assert result[COL_REPAIR_ITERATIONS] == 1
    assert result[COL_NEEDS_HUMAN_REVIEW] is False


def test_combined_graph_preserves_malformed_rewrite_handling(
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    graph = CombinedRewriteWorkflow(adapter=Mock()).build_graph(
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=0),
    )
    rewrite_column = next(column for column in graph.columns if column.name == COL_REWRITTEN_TEXT_INITIAL)
    assert isinstance(rewrite_column, CustomColumnConfig)

    result = rewrite_column.generator_function({COL_FULL_REWRITE: "not-a-valid-payload"})

    assert result[COL_REWRITTEN_TEXT_INITIAL] is None


def test_conditional_repairs_execute_independently_per_row(
    tmp_path: Path,
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    data_designer = DataDesigner(artifact_path=tmp_path / "artifacts", auto_configure_logging=False)
    adapter = NddAdapter(data_designer)
    graph = CombinedRewriteWorkflow(adapter).build_graph(
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
    )
    columns = _deterministic_columns(graph)
    collector = MeasurementCollector(record_hash_key="test-key")

    with measurement_session(collector):
        result = adapter.run_workflow(
            pd.DataFrame({_REPAIRS_NEEDED: [0, 1, 2, 3]}),
            model_configs=[],
            columns=columns,
            workflow_name="rewrite-combined",
            preview_num_records=4,
        )

    assert result.failed_records == []
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == [
        "rewrite-0",
        "rewrite-1",
        "rewrite-2",
        "rewrite-2",
    ]
    assert result.dataframe[COL_REPAIR_ITERATIONS].tolist() == [0, 1, 2, 2]
    assert result.dataframe[COL_NEEDS_REPAIR].tolist() == [False, False, False, True]
    assert result.dataframe[COL_NEEDS_HUMAN_REVIEW].tolist() == [False, False, False, True]
    workflow_records = [record for record in collector.records if record["record_type"] == "ndd_workflow"]
    assert len(workflow_records) == 1
    assert workflow_records[0]["workflow_name"] == "rewrite-combined"
    assert workflow_records[0]["input_row_count"] == 4
    assert workflow_records[0]["output_row_count"] == 4
    assert workflow_records[0]["column_count"] == len(columns)


def test_run_executes_one_data_designer_workflow(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
        }
    )
    output = dataframe.copy()
    output[COL_REWRITTEN_TEXT] = "Maria works at a company"
    output[COL_UTILITY_SCORE] = 0.9
    output[COL_LEAKAGE_MASS] = 0.1
    output[COL_WEIGHTED_LEAKAGE_RATE] = 0.05
    output[COL_ANY_HIGH_LEAKED] = False
    output[COL_NEEDS_REPAIR] = False
    output[COL_REPAIR_ITERATIONS] = 1
    output[COL_NEEDS_HUMAN_REVIEW] = False
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(dataframe=output, failed_records=[])

    collector = MeasurementCollector(record_hash_key="test-key")
    with measurement_session(collector):
        result = CombinedRewriteWorkflow(adapter=adapter).run(
            dataframe,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=2),
        )

    adapter.run_workflow.assert_called_once()
    assert adapter.run_workflow.call_args.kwargs["workflow_name"] == "rewrite-combined"
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == ["Maria works at a company"]
    stage_records = [record for record in collector.records if record["record_type"] == "stage"]
    assert len(stage_records) == 1
    assert stage_records[0]["stage"] == "CombinedRewriteWorkflow.run"
    assert stage_records[0]["input_row_count"] == 1
    assert stage_records[0]["output_row_count"] == 1
    assert stage_records[0]["failed_record_count"] == 0


def test_run_reports_combined_failure_and_drops_only_failed_entity_row(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["entity-ok", "passthrough", "entity-failed"],
            COL_TEXT: ["Alice works at Acme", "No entities", "Bob works at Beta"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {"entities_by_value": []},
                {"entities_by_value": [{"value": "Bob", "labels": ["first_name"]}]},
            ],
        }
    )
    output = dataframe.iloc[[0]].copy()
    output[COL_REWRITTEN_TEXT] = "Person works at Company"
    output[COL_UTILITY_SCORE] = 0.9
    output[COL_LEAKAGE_MASS] = 0.0
    output[COL_WEIGHTED_LEAKAGE_RATE] = 0.0
    output[COL_ANY_HIGH_LEAKED] = False
    output[COL_NEEDS_REPAIR] = False
    output[COL_REPAIR_ITERATIONS] = 0
    output[COL_NEEDS_HUMAN_REVIEW] = False
    failed = FailedRecord(
        record_id="entity-failed",
        step="rewrite-combined",
        reason="Record missing from workflow output",
    )
    adapter = Mock()
    adapter.run_workflow.return_value = WorkflowRunResult(dataframe=output, failed_records=[failed])

    result = CombinedRewriteWorkflow(adapter=adapter).run(
        dataframe,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
    )

    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["entity-ok", "passthrough"]
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == ["Person works at Company", "No entities"]
    assert result.failed_records == [failed]


def test_combined_and_legacy_paths_return_equivalent_repaired_result(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
        }
    )
    pipeline = dataframe.copy()
    pipeline[COL_REWRITTEN_TEXT] = "Initial rewrite"
    evaluation_before = pd.DataFrame(
        {
            COL_NEEDS_REPAIR: [True],
            COL_UTILITY_SCORE: [0.5],
            COL_LEAKAGE_MASS: [1.0],
            COL_WEIGHTED_LEAKAGE_RATE: [1.0],
            COL_ANY_HIGH_LEAKED: [True],
        }
    )
    repair = pd.DataFrame({COL_REWRITTEN_TEXT_NEXT: ["Repaired rewrite"]})
    evaluation_after = pd.DataFrame(
        {
            COL_NEEDS_REPAIR: [False],
            COL_UTILITY_SCORE: [0.9],
            COL_LEAKAGE_MASS: [0.0],
            COL_WEIGHTED_LEAKAGE_RATE: [0.0],
            COL_ANY_HIGH_LEAKED: [False],
        }
    )
    legacy_adapter = Mock()
    legacy_adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=pipeline, failed_records=[]),
        WorkflowRunResult(dataframe=evaluation_before, failed_records=[]),
        WorkflowRunResult(dataframe=repair, failed_records=[]),
        WorkflowRunResult(dataframe=evaluation_after, failed_records=[]),
    ]
    with patch(_REPLACE_PATCH) as replace_workflow:
        replace_workflow.return_value.generate_map_only.return_value = WorkflowRunResult(
            dataframe=dataframe.copy(),
            failed_records=[],
        )
        legacy_result = RewriteWorkflow(adapter=legacy_adapter).run(
            dataframe,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=2),
        )

    combined_output = dataframe.copy()
    combined_output[COL_REWRITTEN_TEXT] = "Repaired rewrite"
    combined_output[COL_UTILITY_SCORE] = 0.9
    combined_output[COL_LEAKAGE_MASS] = 0.0
    combined_output[COL_WEIGHTED_LEAKAGE_RATE] = 0.0
    combined_output[COL_ANY_HIGH_LEAKED] = False
    combined_output[COL_NEEDS_REPAIR] = False
    combined_output[COL_REPAIR_ITERATIONS] = 1
    combined_output[COL_NEEDS_HUMAN_REVIEW] = False
    combined_adapter = Mock()
    combined_adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=combined_output,
        failed_records=[],
    )
    combined_result = CombinedRewriteWorkflow(adapter=combined_adapter).run(
        dataframe,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
    )

    output_columns = [
        COL_REWRITTEN_TEXT,
        COL_UTILITY_SCORE,
        COL_LEAKAGE_MASS,
        COL_WEIGHTED_LEAKAGE_RATE,
        COL_ANY_HIGH_LEAKED,
        COL_NEEDS_REPAIR,
        COL_REPAIR_ITERATIONS,
        COL_NEEDS_HUMAN_REVIEW,
    ]
    pd.testing.assert_frame_equal(
        legacy_result.dataframe[output_columns],
        combined_result.dataframe[output_columns],
        check_dtype=False,
    )
    assert legacy_result.failed_records == combined_result.failed_records == []


@pytest.mark.parametrize(
    ("repairs_needed", "expected_counts"),
    [
        ([0] * 62 + [1, 2], {0: 62, 1: 1, 2: 1}),
        ([2] * 62 + [0, 1], {0: 1, 1: 1, 2: 62}),
    ],
    ids=["mostly-skipped", "mostly-repaired"],
)
def test_conditional_graph_handles_larger_mixed_batches(
    tmp_path: Path,
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    repairs_needed: list[int],
    expected_counts: dict[int, int],
) -> None:
    data_designer = DataDesigner(artifact_path=tmp_path / "artifacts", auto_configure_logging=False)
    adapter = NddAdapter(data_designer)
    graph = CombinedRewriteWorkflow(adapter).build_graph(
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
    )

    result = adapter.run_workflow(
        pd.DataFrame({_REPAIRS_NEEDED: repairs_needed}),
        model_configs=[],
        columns=_deterministic_columns(graph),
        workflow_name="rewrite-combined-scale-test",
        preview_num_records=len(repairs_needed),
    )

    assert result.failed_records == []
    assert result.dataframe[COL_REPAIR_ITERATIONS].value_counts().to_dict() == expected_counts
    assert len(result.dataframe) == len(repairs_needed)


def test_run_preserves_mixed_row_order_and_passthrough_defaults(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme", "No entities", "Bob works at Beta"],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
                {"entities_by_value": []},
                {"entities_by_value": [{"value": "Bob", "labels": ["first_name"]}]},
            ],
        }
    )
    adapter = Mock()

    def run_workflow(entity_rows: pd.DataFrame, **_: Any) -> WorkflowRunResult:
        output = entity_rows.copy()
        output[COL_REWRITTEN_TEXT] = ["Person works at Company", "Worker works at Business"]
        output[COL_UTILITY_SCORE] = 0.9
        output[COL_LEAKAGE_MASS] = 0.0
        output[COL_WEIGHTED_LEAKAGE_RATE] = 0.0
        output[COL_ANY_HIGH_LEAKED] = False
        output[COL_NEEDS_REPAIR] = False
        output[COL_REPAIR_ITERATIONS] = 0
        output[COL_NEEDS_HUMAN_REVIEW] = False
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter.run_workflow.side_effect = run_workflow

    result = CombinedRewriteWorkflow(adapter=adapter).run(
        dataframe,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
    )

    assert result.dataframe[COL_TEXT].tolist() == dataframe[COL_TEXT].tolist()
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == [
        "Person works at Company",
        "No entities",
        "Worker works at Business",
    ]
    assert result.dataframe[COL_UTILITY_SCORE].tolist() == [0.9, 1.0, 0.9]
    assert result.dataframe[COL_REPAIR_ITERATIONS].tolist() == [0, 0, 0]
    assert adapter.run_workflow.call_args.args[0][COL_TEXT].tolist() == [
        "Alice works at Acme",
        "Bob works at Beta",
    ]


def test_run_skips_data_designer_when_no_rows_have_entities(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["No entities", "Still no entities"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}, {"entities_by_value": []}],
        }
    )
    adapter = Mock()

    result = CombinedRewriteWorkflow(adapter=adapter).run(
        dataframe,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=2),
    )

    adapter.run_workflow.assert_not_called()
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == dataframe[COL_TEXT].tolist()
    assert result.dataframe[COL_UTILITY_SCORE].tolist() == [1.0, 1.0]
    assert result.dataframe[COL_REPAIR_ITERATIONS].tolist() == [0, 0]


@pytest.mark.parametrize("max_repair_iterations", [0, 2, 10])
def test_graph_validates_with_data_designer(
    tmp_path: Path,
    stub_slim_model_selection: ModelSelection,
    max_repair_iterations: int,
) -> None:
    provider = ModelProvider(
        name="stub",
        endpoint="http://stub.invalid/v1",
        provider_type="openai",
        api_key="EMPTY",
    )
    data_designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=[provider],
        auto_configure_logging=False,
    )
    adapter = NddAdapter(data_designer)
    graph = CombinedRewriteWorkflow(adapter).build_graph(
        selected_models=stub_slim_model_selection.rewrite,
        replace_model_selection=stub_slim_model_selection.replace,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=max_repair_iterations),
    )
    dataframe = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_TAGGED_TEXT: ["[[Alice|first_name]] works at Acme"],
            COL_TAG_NOTATION: ["bracket"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}],
            COL_LATENT_ENTITIES: [{"latent_entities": []}],
        }
    )
    model_configs = [ModelConfig(alias="known", model="stub-model", provider="stub")]
    builder = adapter.build_config(
        dataframe,
        model_configs=model_configs,
        columns=graph.columns,
        seed_path=tmp_path / "seed.parquet",
    )

    data_designer.validate(builder)
