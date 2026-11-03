# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.models.errors import ModelTimeoutError

from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_LEAKED_PRIVACY_ITEMS,
    COL_PRIVACY_QA,
    COL_QUALITY_QA,
    COL_QUALITY_QA_REANSWER,
    COL_REWRITE_BASELINE_TEXT,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.evaluate import EvaluateWorkflow
from anonymizer.engine.rewrite.repair import RepairWorkflow


@pytest.fixture(params=[False, True], ids=["standalone", "combined"])
def model_columns(
    request: pytest.FixtureRequest,
    model_row: dict[str, Any],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> list[CustomColumnConfig]:
    adapter = Mock()
    privacy_goal = PrivacyGoal(protect="Personal names and locations", preserve="Professional meaning and context")
    evaluation = EvaluationCriteria(max_repair_iterations=1)
    if request.param:
        graph = CombinedRewriteWorkflow(adapter).build_graph(
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=privacy_goal,
            evaluation=evaluation,
        )
        initial = graph.evaluation_states[0]
        repair = graph.repair_states[0]
        model_row.update(
            {
                initial.rewritten_text: model_row[COL_REWRITTEN_TEXT],
                initial.quality_reanswer: model_row[COL_QUALITY_QA_REANSWER],
                initial.leakage_mass: model_row[COL_LEAKAGE_MASS],
                initial.any_high_leaked: model_row[COL_ANY_HIGH_LEAKED],
                initial.utility_score: model_row[COL_UTILITY_SCORE],
                repair.leaked_items: model_row[COL_LEAKED_PRIVACY_ITEMS],
            }
        )
        names = [
            initial.quality_reanswer,
            initial.privacy_reanswer,
            initial.quality_compare,
            repair.rewritten_text,
        ]
        by_name = {column.name: column for column in graph.columns}
        columns = [by_name[name] for name in names]
    else:
        columns = [
            *EvaluateWorkflow(adapter).columns(selected_models=stub_rewrite_model_selection, evaluation=evaluation)[:3],
            RepairWorkflow(adapter).columns(
                selected_models=stub_rewrite_model_selection, privacy_goal=privacy_goal, effective_threshold=1.0
            )[-1],
        ]
    assert all(isinstance(column, CustomColumnConfig) for column in columns)
    return cast(list[CustomColumnConfig], columns)


@pytest.fixture
def model_row() -> dict[str, Any]:
    return {
        COL_QUALITY_QA: {
            "items": [
                {
                    "id": 1,
                    "aspect": "occupation",
                    "importance": "important",
                    "question": "What is the job?",
                    "reference_answer": "Engineer",
                }
            ]
        },
        COL_PRIVACY_QA: {
            "items": [
                {
                    "id": 1,
                    "question": "Is Alice named?",
                    "sensitivity": "high",
                    "entity_label": "first_name",
                    "entity_value": "Alice",
                    "category": "direct_identifier",
                }
            ]
        },
        COL_QUALITY_QA_REANSWER: {"answers": [{"id": 1, "answer": "Engineer"}]},
        COL_TEXT: "Alice is an engineer.",
        COL_REWRITE_BASELINE_TEXT: "Maria is an engineer.",
        COL_REWRITTEN_TEXT: "Alice is an engineer.",
        COL_LEAKED_PRIVACY_ITEMS: "Alice remains named.",
        COL_LEAKAGE_MASS: 1.0,
        COL_ANY_HIGH_LEAKED: True,
        COL_UTILITY_SCORE: 1.0,
    }


@pytest.mark.parametrize(
    ("index", "response"),
    [
        (0, {"answers": [{"id": 1, "answer": "Engineer"}]}),
        (1, {"answers": [{"id": 1, "answer": "yes", "confidence": 1.0, "reason": "Named", "evidence": ["Alice"]}]}),
        (2, {"per_item": [{"id": 1, "score": 1.0, "reason": "Matches"}]}),
        (3, {"rewritten_text": "An engineer works here."}),
    ],
    ids=["quality-reanswer", "privacy-reanswer", "quality-compare", "repair"],
)
def test_model_columns_await_async_facade(
    model_columns: list[CustomColumnConfig], model_row: dict[str, Any], index: int, response: dict[str, Any]
) -> None:
    column = model_columns[index]
    row = {name: model_row[name] for name in column.required_columns}

    async def agenerate(*, parser: Any, **_kwargs: Any) -> tuple[Any, list]:
        await asyncio.sleep(0)
        return parser(json.dumps(response)), []

    facade = Mock(
        generate=Mock(side_effect=AssertionError("Sync model call")), agenerate=AsyncMock(side_effect=agenerate)
    )
    resources = Mock()
    resources.model_registry.get_model.return_value = facade
    generator = CustomColumnGenerator(column, resource_provider=resources)

    result = asyncio.run(generator.agenerate(row))

    assert result[column.name] == (response["rewritten_text"] if index == 3 else response)
    facade.generate.assert_not_called()
    facade.agenerate.assert_awaited_once()
    assert facade.agenerate.call_args.kwargs["max_correction_steps"] == 3


@pytest.mark.parametrize("error", [ModelTimeoutError("Timeout"), asyncio.CancelledError()])
@pytest.mark.parametrize("index", range(4), ids=["quality-reanswer", "privacy-reanswer", "quality-compare", "repair"])
def test_model_failure_propagates(
    model_columns: list[CustomColumnConfig], model_row: dict[str, Any], index: int, error: BaseException
) -> None:
    column = model_columns[index]
    facade = Mock(agenerate=AsyncMock(side_effect=error))
    resources = Mock()
    resources.model_registry.get_model.return_value = facade
    generator = CustomColumnGenerator(column, resource_provider=resources)
    row = {name: model_row[name] for name in column.required_columns}

    with pytest.raises(type(error)):
        asyncio.run(generator.agenerate(row))
