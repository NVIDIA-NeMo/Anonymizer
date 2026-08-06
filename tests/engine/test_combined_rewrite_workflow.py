# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import ModelSelection, ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_REPAIR_ITERATIONS,
    COL_REWRITTEN_TEXT,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.ndd.adapter import NddAdapter, WorkflowRunResult
from anonymizer.engine.rewrite.combined_workflow import CombinedRewriteWorkflow

_PRIVACY_GOAL = PrivacyGoal(
    protect="All direct identifiers including names, locations, and contact details",
    preserve="Career trajectory, skills, and professional context in abstract terms",
)


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


def test_graph_validates_with_data_designer(
    tmp_path: Path,
    stub_slim_model_selection: ModelSelection,
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
        evaluation=EvaluationCriteria(max_repair_iterations=2),
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
