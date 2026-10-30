# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.models import ModelConfig
from data_designer.interface.data_designer import DataDesigner

from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_VALID,
    COL_DOMAIN,
    COL_ENTITIES_BY_VALUE,
    COL_JUDGE_EVALUATION,
    COL_LEAKAGE_MASS,
    COL_LEAKED_PRIVACY_ITEMS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REPAIR_ITERATIONS,
    COL_RETAINED_PROTECTED_ENTITIES,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.rewrite.repair import RepairWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow, _detection_valid_fraction, _join_new_columns

_REPLACE_PATCH = "anonymizer.engine.rewrite.rewrite_workflow.LlmReplaceWorkflow"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PRIVACY_GOAL = PrivacyGoal(
    protect="All direct identifiers including names, locations, and contact details",
    preserve="Career trajectory, skills, and professional context in abstract terms",
)

_EVALUATION = EvaluationCriteria()


def test_join_new_columns_preserves_surviving_target_order() -> None:
    target = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["turn-2", "dropped", "turn-1"],
            "existing": [2, 0, 1],
        }
    )
    source = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["turn-1", "turn-2"],
            "new": [10, 20],
        }
    )

    result = _join_new_columns(target, source)

    assert result[RECORD_ID_COLUMN].tolist() == ["turn-2", "turn-1"]
    assert result["existing"].tolist() == [2, 1]
    assert result["new"].tolist() == [20, 10]


@pytest.fixture
def stub_entities_by_value_with_entities() -> dict:
    return {
        "entities_by_value": [
            {"value": "Alice", "labels": ["first_name"]},
        ]
    }


@pytest.fixture
def stub_df_no_entities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["The sky is blue", "Water is wet"],
            COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}, {"entities_by_value": []}],
        }
    )


@pytest.fixture
def stub_df_with_entities(stub_entities_by_value_with_entities: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme Corp"],
            COL_ENTITIES_BY_VALUE: [stub_entities_by_value_with_entities],
        }
    )


@pytest.fixture
def stub_df_two_entities(stub_entities_by_value_with_entities: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["Alice works here", "Bob works there"],
            COL_ENTITIES_BY_VALUE: [
                stub_entities_by_value_with_entities,
                stub_entities_by_value_with_entities,
            ],
        }
    )


@pytest.fixture
def stub_df_mixed(stub_entities_by_value_with_entities: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_TEXT: ["Alice works here", "The sky is blue"],
            COL_ENTITIES_BY_VALUE: [
                stub_entities_by_value_with_entities,
                {"entities_by_value": []},
            ],
        }
    )


@pytest.fixture
def stub_pre_gen_df(stub_df_with_entities: pd.DataFrame) -> pd.DataFrame:
    df = stub_df_with_entities.copy()
    df[COL_DOMAIN] = "BIOGRAPHY_PROFILE"
    df["_anonymizer_row_order"] = [0]
    df[RECORD_ID_COLUMN] = ["rec-0"]
    return df


@pytest.fixture
def stub_replace_df(stub_df_with_entities: pd.DataFrame) -> pd.DataFrame:
    """Entity rows after replace-map generation (adds COL_REPLACEMENT_MAP)."""
    df = stub_df_with_entities.copy()
    df["_replacement_map"] = [{"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maria"}]}]
    return df


@pytest.fixture
def stub_pipeline_df(stub_pre_gen_df: pd.DataFrame) -> pd.DataFrame:
    """Result of the combined rewrite-pipeline adapter call."""
    df = stub_pre_gen_df.copy()
    df[COL_REWRITTEN_TEXT] = "Maria works"
    df[COL_REPAIR_ITERATIONS] = 0
    return df


@pytest.fixture
def stub_eval_df(stub_pipeline_df: pd.DataFrame) -> pd.DataFrame:
    df = stub_pipeline_df.copy()
    df[COL_NEEDS_REPAIR] = False
    df[COL_UTILITY_SCORE] = 0.9
    df[COL_LEAKAGE_MASS] = 0.1
    df[COL_ANY_HIGH_LEAKED] = False
    return df


@pytest.fixture
def stub_judge_df(stub_eval_df: pd.DataFrame) -> pd.DataFrame:
    """Fixture used by evaluate() tests — only judge-produced columns here."""
    df = stub_eval_df.copy()
    df[COL_JUDGE_EVALUATION] = None
    return df


def _standard_side_effect(
    pipeline_df: pd.DataFrame,
    eval_df: pd.DataFrame,
) -> list[WorkflowRunResult]:
    """Happy-path adapter side_effect for run(): pipeline then evaluate.

    The final-judge no longer runs inside run() — it only runs via evaluate().
    """
    return [
        WorkflowRunResult(dataframe=pipeline_df, failed_records=[]),
        WorkflowRunResult(dataframe=eval_df, failed_records=[]),
    ]


def _mock_replace(mock_cls: Mock, replace_df: pd.DataFrame) -> None:
    """Configure a patched LlmReplaceWorkflow class to return replace_df."""
    mock_cls.return_value.generate_map_only.return_value = Mock(dataframe=replace_df, failed_records=[])


# ---------------------------------------------------------------------------
# Tests: fast path
# ---------------------------------------------------------------------------


def test_no_entities_skips_all_workflows(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_no_entities: pd.DataFrame,
) -> None:
    adapter = Mock()
    wf = RewriteWorkflow(adapter=adapter)

    result = wf.run(
        stub_df_no_entities,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=_EVALUATION,
    )

    adapter.run_workflow.assert_not_called()
    assert len(result.dataframe) == 2
    assert result.failed_records == []


def test_passthrough_defaults_populated(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_no_entities: pd.DataFrame,
) -> None:
    adapter = Mock()
    wf = RewriteWorkflow(adapter=adapter)

    result = wf.run(
        stub_df_no_entities,
        model_configs=stub_model_configs,
        selected_models=stub_rewrite_model_selection,
        replace_model_selection=stub_replace_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=_EVALUATION,
    )

    df = result.dataframe
    assert df[COL_REWRITTEN_TEXT].tolist() == ["The sky is blue", "Water is wet"]
    assert df[COL_UTILITY_SCORE].tolist() == [1.0, 1.0]
    assert df[COL_LEAKAGE_MASS].tolist() == [0.0, 0.0]
    assert df[COL_WEIGHTED_LEAKAGE_RATE].tolist() == [0.0, 0.0]
    assert df[COL_ANY_HIGH_LEAKED].tolist() == [False, False]
    assert df[COL_NEEDS_HUMAN_REVIEW].tolist() == [False, False]
    assert df[COL_REPAIR_ITERATIONS].tolist() == [0, 0]
    assert COL_JUDGE_EVALUATION not in df.columns


def test_has_entities_returns_true_when_present(stub_entities_by_value_with_entities: dict) -> None:
    from anonymizer.engine.rewrite.rewrite_workflow import _has_entities

    assert _has_entities(stub_entities_by_value_with_entities) is True


def test_has_entities_returns_false_when_empty() -> None:
    from anonymizer.engine.rewrite.rewrite_workflow import _has_entities

    assert _has_entities({"entities_by_value": []}) is False


def test_has_entities_returns_false_for_none() -> None:
    from anonymizer.engine.rewrite.rewrite_workflow import _has_entities

    assert _has_entities(None) is False


# ---------------------------------------------------------------------------
# Tests: full pipeline call order
# ---------------------------------------------------------------------------


def test_calls_sub_workflows_in_order(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_with_entities: pd.DataFrame,
    stub_replace_df: pd.DataFrame,
    stub_pipeline_df: pd.DataFrame,
    stub_eval_df: pd.DataFrame,
) -> None:
    adapter = Mock()
    adapter.run_workflow.side_effect = _standard_side_effect(stub_pipeline_df, stub_eval_df)

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, stub_replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        result = wf.run(
            stub_df_with_entities,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=_EVALUATION,
        )

    workflow_names = [call.kwargs["workflow_name"] for call in adapter.run_workflow.call_args_list]
    assert workflow_names[0] == "rewrite-pipeline"
    assert workflow_names[1].startswith("rewrite-evaluate")
    assert "rewrite-final-judge" not in workflow_names

    assert len(result.dataframe) == 1


# ---------------------------------------------------------------------------
# Tests: failed records
# ---------------------------------------------------------------------------


def test_failed_records_accumulated_across_steps(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_with_entities: pd.DataFrame,
    stub_replace_df: pd.DataFrame,
    stub_pipeline_df: pd.DataFrame,
    stub_eval_df: pd.DataFrame,
) -> None:
    failed_pipeline = FailedRecord(record_id="a", step="rewrite-pipeline", reason="timeout")
    failed_eval = FailedRecord(record_id="b", step="rewrite-evaluate-0", reason="timeout")

    adapter = Mock()
    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=stub_pipeline_df, failed_records=[failed_pipeline]),
        WorkflowRunResult(dataframe=stub_eval_df, failed_records=[failed_eval]),
    ]

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        mock_replace_cls.return_value.generate_map_only.return_value = Mock(
            dataframe=stub_replace_df,
            failed_records=[FailedRecord(record_id="d", step="replace-map-generation", reason="timeout")],
        )
        wf = RewriteWorkflow(adapter=adapter)
        result = wf.run(
            stub_df_with_entities,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=_EVALUATION,
        )

    record_ids = {f.record_id for f in result.failed_records}
    assert record_ids == {"a", "b", "d"}


# ---------------------------------------------------------------------------
# Tests: final judge failure tolerance
# ---------------------------------------------------------------------------


def test_judge_failure_does_not_propagate(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection,
    stub_eval_df: pd.DataFrame,
) -> None:
    """evaluate() holistic judge failure is non-fatal; rows get COL_JUDGE_EVALUATION=None."""
    adapter = Mock()

    wf = RewriteWorkflow(adapter=adapter)
    # Mock detection judge to return successfully
    wf._detection_judge_wf = Mock()
    wf._detection_judge_wf.evaluate.return_value = Mock(dataframe=stub_eval_df.copy(), failed_records=[])
    # Make the holistic judge adapter call raise
    adapter.run_workflow.side_effect = RuntimeError("Judge LLM unavailable")

    result = wf.evaluate(
        stub_eval_df,
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_PRIVACY_GOAL,
    )

    assert len(result.dataframe) == 1
    assert result.dataframe[COL_JUDGE_EVALUATION].iloc[0] is None


def test_judge_partial_row_loss_preserves_all_rows(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection,
    stub_df_two_entities: pd.DataFrame,
) -> None:
    """evaluate() judge drops 1 of 2 rows — surviving row gets scores, missing row gets None."""
    df = stub_df_two_entities.copy()
    df["_anonymizer_record_id"] = ["rec-0", "rec-1"]

    # Build a run()-style result dataframe with all required columns
    run_result_df = df.copy()
    run_result_df[COL_REWRITTEN_TEXT] = ["Maria works here", "Rob works there"]
    run_result_df[COL_NEEDS_REPAIR] = False
    run_result_df[COL_UTILITY_SCORE] = [0.9, 0.8]
    run_result_df[COL_LEAKAGE_MASS] = [0.1, 0.2]
    run_result_df[COL_ANY_HIGH_LEAKED] = False
    run_result_df[COL_NEEDS_HUMAN_REVIEW] = False
    run_result_df[COL_REPAIR_ITERATIONS] = 0

    adapter = Mock()

    # detection judge returns both rows
    det_df = run_result_df.copy()
    # holistic judge returns only first row
    judge_df = run_result_df.iloc[[0]].copy().reset_index(drop=True)
    judge_df[COL_JUDGE_EVALUATION] = [
        {"privacy": {"score": "high"}, "quality": {"score": "high"}, "style": {"score": "medium"}}
    ]

    wf = RewriteWorkflow(adapter=adapter)
    wf._detection_judge_wf = Mock()
    wf._detection_judge_wf.evaluate.return_value = Mock(dataframe=det_df, failed_records=[])
    adapter.run_workflow.return_value = WorkflowRunResult(
        dataframe=judge_df,
        failed_records=[FailedRecord(record_id="rec-1", step="rewrite-final-judge", reason="timeout")],
    )

    result = wf.evaluate(
        run_result_df,
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_PRIVACY_GOAL,
    )

    assert len(result.dataframe) == 2
    assert result.dataframe[COL_JUDGE_EVALUATION].iloc[0] is not None
    assert result.dataframe[COL_JUDGE_EVALUATION].iloc[1] is None


def test_detection_judge_partial_row_loss_preserves_all_rows(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection,
    stub_df_two_entities: pd.DataFrame,
) -> None:
    """evaluate() detection judge drops 1 of 2 rows — surviving row gets scores, missing row gets None."""
    df = stub_df_two_entities.copy()
    df["_anonymizer_record_id"] = ["rec-0", "rec-1"]

    run_result_df = df.copy()
    run_result_df[COL_REWRITTEN_TEXT] = ["Maria works here", "Rob works there"]
    run_result_df[COL_NEEDS_REPAIR] = False
    run_result_df[COL_UTILITY_SCORE] = [0.9, 0.8]
    run_result_df[COL_LEAKAGE_MASS] = [0.1, 0.2]
    run_result_df[COL_ANY_HIGH_LEAKED] = False
    run_result_df[COL_NEEDS_HUMAN_REVIEW] = False
    run_result_df[COL_REPAIR_ITERATIONS] = 0

    # detection judge returns only first row
    det_df = run_result_df.iloc[[0]].copy().reset_index(drop=True)
    det_df[COL_DETECTION_VALID] = True
    det_df[COL_DETECTION_INVALID_ENTITIES] = [[]]

    # holistic judge returns both rows (no additional drop)
    judge_df = run_result_df.copy()
    judge_df[COL_JUDGE_EVALUATION] = [None, None]

    adapter = Mock()
    wf = RewriteWorkflow(adapter=adapter)
    wf._detection_judge_wf = Mock()
    wf._detection_judge_wf.evaluate.return_value = Mock(
        dataframe=det_df,
        failed_records=[FailedRecord(record_id="rec-1", step="rewrite-detection-judge", reason="timeout")],
    )
    adapter.run_workflow.return_value = WorkflowRunResult(dataframe=judge_df, failed_records=[])

    result = wf.evaluate(
        run_result_df,
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        compute_detection_validity=True,
    )

    assert len(result.dataframe) == 2
    # surviving row gets a computed detection_valid fraction
    assert not pd.isna(result.dataframe[COL_DETECTION_VALID].iloc[0])
    # dropped row defaults to null instead of being removed
    assert pd.isna(result.dataframe[COL_DETECTION_VALID].iloc[1])
    assert result.dataframe[COL_DETECTION_INVALID_ENTITIES].iloc[1] is None


# ---------------------------------------------------------------------------
# Tests: evaluate-repair loop
# ---------------------------------------------------------------------------


def test_repair_loop_exits_early_when_no_rows_need_repair(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_with_entities: pd.DataFrame,
    stub_replace_df: pd.DataFrame,
    stub_pipeline_df: pd.DataFrame,
    stub_eval_df: pd.DataFrame,
) -> None:
    adapter = Mock()
    adapter.run_workflow.side_effect = _standard_side_effect(stub_pipeline_df, stub_eval_df)

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, stub_replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        wf.run(
            stub_df_with_entities,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=3),
        )

    workflow_names = [call.kwargs["workflow_name"] for call in adapter.run_workflow.call_args_list]
    repair_calls = [n for n in workflow_names if "repair" in n]
    assert repair_calls == []


def test_repair_loop_runs_up_to_max_iterations(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_with_entities: pd.DataFrame,
) -> None:
    adapter = Mock()
    max_iters = 2

    pre_gen_df = stub_df_with_entities.copy()
    pre_gen_df[COL_DOMAIN] = "BIOGRAPHY_PROFILE"
    pre_gen_df["_anonymizer_row_order"] = [0]

    rewrite_gen_df = pre_gen_df.copy()
    rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works"
    rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

    eval_needs_repair = rewrite_gen_df.copy()
    eval_needs_repair[COL_NEEDS_REPAIR] = True
    eval_needs_repair[COL_UTILITY_SCORE] = 0.9
    eval_needs_repair[COL_LEAKAGE_MASS] = 2.0
    eval_needs_repair[COL_ANY_HIGH_LEAKED] = True

    repaired_df = eval_needs_repair.copy()
    repaired_df[COL_REWRITTEN_TEXT_NEXT] = "Repaired text"

    replace_df = stub_df_with_entities.copy()
    replace_df["_replacement_map"] = [{"replacements": []}]

    adapter.run_workflow.side_effect = [
        # pipeline (domain + disposition + QA + rewrite)
        WorkflowRunResult(dataframe=rewrite_gen_df, failed_records=[]),
        # evaluate-0 -> needs repair
        WorkflowRunResult(dataframe=eval_needs_repair, failed_records=[]),
        # repair-0
        WorkflowRunResult(dataframe=repaired_df, failed_records=[]),
        # evaluate-1 -> still needs repair
        WorkflowRunResult(dataframe=eval_needs_repair, failed_records=[]),
        # repair-1
        WorkflowRunResult(dataframe=repaired_df, failed_records=[]),
        # evaluate-final (after loop exhaustion)
        WorkflowRunResult(dataframe=eval_needs_repair, failed_records=[]),
    ]

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        wf.run(
            stub_df_with_entities,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=max_iters),
        )

    workflow_names = [call.kwargs["workflow_name"] for call in adapter.run_workflow.call_args_list]
    repair_calls = [n for n in workflow_names if "repair" in n]
    assert len(repair_calls) == max_iters


_PRIVACY_QA = {"items": []}


class _ScriptedAdapter:
    """Adapter-shaped stub: each step returns its seed plus only the columns that step produces.

    ``repairs_needed`` maps a record id to the number of repairs after which it passes evaluation.
    Repair ``N`` returns ``"<record id> repaired N"`` so every iteration's output is distinct.
    """

    def __init__(self, repairs_needed: dict[str, int]) -> None:
        self.repairs_needed = repairs_needed
        self.seeds: dict[str, pd.DataFrame] = {}

    def run_workflow(self, seed: pd.DataFrame, *, workflow_name: str, **_: object) -> WorkflowRunResult:
        self.seeds[workflow_name] = seed.copy()
        output = seed.copy()
        record_ids = output[RECORD_ID_COLUMN].astype(str)
        iteration = int(workflow_name.rsplit("-", 1)[1]) if workflow_name[-1].isdigit() else 0
        if workflow_name == "rewrite-pipeline":
            output[COL_REWRITTEN_TEXT] = [f"{record_id} initial" for record_id in record_ids]
            output[COL_PRIVACY_QA] = [dict(_PRIVACY_QA) for _ in record_ids]
        elif workflow_name.startswith("rewrite-evaluate-"):
            needs_repair = [iteration < self.repairs_needed[record_id] for record_id in record_ids]
            output[COL_PRIVACY_QA_REANSWER] = [{"answers": []} for _ in record_ids]
            output[COL_UTILITY_SCORE] = 0.9
            output[COL_LEAKAGE_MASS] = [2.0 if flag else 0.0 for flag in needs_repair]
            output[COL_WEIGHTED_LEAKAGE_RATE] = 0.0
            output[COL_ANY_HIGH_LEAKED] = False
            output[COL_RETAINED_PROTECTED_ENTITIES] = [[] for _ in record_ids]
            output[COL_NEEDS_REPAIR] = needs_repair
        elif workflow_name.startswith("rewrite-repair-"):
            output[COL_LEAKED_PRIVACY_ITEMS] = f"leaked items {iteration}"
            output[COL_REWRITTEN_TEXT_NEXT] = [f"{record_id} repaired {iteration}" for record_id in record_ids]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    @property
    def workflow_names(self) -> list[str]:
        return list(self.seeds)


def _run_scripted(
    adapter: _ScriptedAdapter,
    *,
    max_repair_iterations: int,
    model_configs: list[ModelConfig],
    rewrite_selection: RewriteModelSelection,
    replace_selection: ReplaceModelSelection,
) -> pd.DataFrame:
    record_ids = list(adapter.repairs_needed)
    dataframe = pd.DataFrame(
        {
            RECORD_ID_COLUMN: record_ids,
            COL_TEXT: [f"Alice works at Acme ({record_id})" for record_id in record_ids],
            COL_ENTITIES_BY_VALUE: [
                {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]} for _ in record_ids
            ],
        }
    )
    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, dataframe.copy())
        result = RewriteWorkflow(adapter=Mock(wraps=adapter)).run(
            dataframe,
            model_configs=model_configs,
            selected_models=rewrite_selection,
            replace_model_selection=replace_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=max_repair_iterations),
        )
    return result.dataframe


def _by_id(dataframe: pd.DataFrame, column: str) -> dict[str, object]:
    return dict(zip(dataframe[RECORD_ID_COLUMN].astype(str), dataframe[column]))


def test_repair_loop_keeps_each_iterations_repaired_text(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = _ScriptedAdapter({"rec": 99})

    result = _run_scripted(
        adapter,
        max_repair_iterations=3,
        model_configs=stub_model_configs,
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    row = result.iloc[0]
    assert row[COL_REWRITTEN_TEXT] == "rec repaired 2"
    assert row[COL_REWRITTEN_TEXT_NEXT] == "rec repaired 2"
    assert row[COL_LEAKED_PRIVACY_ITEMS] == "leaked items 2"
    assert row[COL_REPAIR_ITERATIONS] == 3
    evaluated = [adapter.seeds[f"rewrite-evaluate-{n}"][COL_REWRITTEN_TEXT].tolist() for n in (1, 2, 3)]
    assert evaluated == [["rec repaired 0"], ["rec repaired 1"], ["rec repaired 2"]]
    repaired = [adapter.seeds[f"rewrite-repair-{n}"][COL_REWRITTEN_TEXT].tolist() for n in (0, 1, 2)]
    assert repaired == [["rec initial"], ["rec repaired 0"], ["rec repaired 1"]]


def test_repair_join_keeps_seed_columns(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = _ScriptedAdapter({"rec": 99})

    result = _run_scripted(
        adapter,
        max_repair_iterations=2,
        model_configs=stub_model_configs,
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    # The repair seed carries the privacy QA as a JSON string; the frame must keep its own dict.
    assert isinstance(adapter.seeds["rewrite-repair-1"][COL_PRIVACY_QA].iloc[0], str)
    assert result[COL_PRIVACY_QA].iloc[0] == _PRIVACY_QA


def test_repair_loop_mixed_rows_keep_their_last_repair(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = _ScriptedAdapter({"thrice": 99, "passing": 0, "once": 1, "twice": 2})

    result = _run_scripted(
        adapter,
        max_repair_iterations=3,
        model_configs=stub_model_configs,
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    assert result[RECORD_ID_COLUMN].tolist() == ["thrice", "passing", "once", "twice"]
    assert _by_id(result, COL_REWRITTEN_TEXT) == {
        "thrice": "thrice repaired 2",
        "passing": "passing initial",
        "once": "once repaired 0",
        "twice": "twice repaired 1",
    }
    assert _by_id(result, COL_REPAIR_ITERATIONS) == {"thrice": 3, "passing": 0, "once": 1, "twice": 2}
    assert adapter.seeds["rewrite-repair-1"][RECORD_ID_COLUMN].tolist() == ["thrice", "twice"]
    assert adapter.seeds["rewrite-repair-2"][RECORD_ID_COLUMN].tolist() == ["thrice"]


def test_repair_loop_stops_after_successful_repair(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    adapter = _ScriptedAdapter({"rec": 2})

    result = _run_scripted(
        adapter,
        max_repair_iterations=3,
        model_configs=stub_model_configs,
        rewrite_selection=stub_rewrite_model_selection,
        replace_selection=stub_replace_model_selection,
    )

    assert adapter.workflow_names == [
        "rewrite-pipeline",
        "rewrite-evaluate-0",
        "rewrite-repair-0",
        "rewrite-evaluate-1",
        "rewrite-repair-1",
        "rewrite-evaluate-2",
    ]
    assert result[COL_REWRITTEN_TEXT].iloc[0] == "rec repaired 1"
    assert result[COL_REPAIR_ITERATIONS].iloc[0] == 2
    assert not bool(result[COL_NEEDS_REPAIR].iloc[0])


_REPAIRS_NEEDED = "repairs_needed"


@custom_column_generator(
    required_columns=[COL_REWRITTEN_TEXT, _REPAIRS_NEEDED],
    side_effect_columns=[
        COL_PRIVACY_QA_REANSWER,
        COL_QUALITY_QA_COMPARE,
        COL_UTILITY_SCORE,
        COL_LEAKAGE_MASS,
        COL_WEIGHTED_LEAKAGE_RATE,
        COL_ANY_HIGH_LEAKED,
        COL_RETAINED_PROTECTED_ENTITIES,
        COL_NEEDS_REPAIR,
    ],
)
def _deterministic_evaluation(row: dict[str, Any]) -> dict[str, Any]:
    needs_repair = int(str(row[COL_REWRITTEN_TEXT]).rsplit("-", 1)[1]) < int(row[_REPAIRS_NEEDED])
    row[COL_QUALITY_QA_REANSWER] = {"answers": []}
    row[COL_PRIVACY_QA_REANSWER] = {"answers": []}
    row[COL_QUALITY_QA_COMPARE] = {"per_item": []}
    row[COL_UTILITY_SCORE] = 0.9
    row[COL_LEAKAGE_MASS] = 2.0 if needs_repair else 0.0
    row[COL_WEIGHTED_LEAKAGE_RATE] = 0.0
    row[COL_ANY_HIGH_LEAKED] = False
    row[COL_RETAINED_PROTECTED_ENTITIES] = []
    row[COL_NEEDS_REPAIR] = needs_repair
    return row


@custom_column_generator(required_columns=[COL_LEAKED_PRIVACY_ITEMS, COL_REWRITTEN_TEXT])
def _deterministic_repair(row: dict[str, Any]) -> dict[str, Any]:
    stem, version = str(row[COL_REWRITTEN_TEXT]).rsplit("-", 1)
    row[COL_REWRITTEN_TEXT_NEXT] = f"{stem}-{int(version) + 1}"
    return row


def test_repair_loop_through_real_adapter_keeps_each_repair(
    tmp_path: Path,
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    adapter = NddAdapter(DataDesigner(artifact_path=tmp_path / "artifacts", auto_configure_logging=False))
    workflow = RewriteWorkflow(adapter)
    workflow._evaluate_wf = Mock(
        columns=Mock(
            return_value=[
                CustomColumnConfig(name=COL_QUALITY_QA_REANSWER, generator_function=_deterministic_evaluation)
            ]
        )
    )
    leaked_items = RepairWorkflow(adapter).columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        effective_threshold=1.0,
    )[0]
    workflow._repair_wf = Mock(
        columns=Mock(
            return_value=[
                leaked_items,
                CustomColumnConfig(name=COL_REWRITTEN_TEXT_NEXT, generator_function=_deterministic_repair),
            ]
        )
    )
    dataframe = pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["twice", "exhausted"],
            COL_TEXT: ["Alice works at Acme", "Bob works at Beta"],
            COL_REWRITTEN_TEXT: ["twice-0", "exhausted-0"],
            COL_PRIVACY_QA: [dict(_PRIVACY_QA), dict(_PRIVACY_QA)],
            _REPAIRS_NEEDED: [2, 5],
        }
    )

    result, failed = workflow._run_evaluate_repair_loop(
        dataframe,
        model_configs=[],
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        evaluation=EvaluationCriteria(max_repair_iterations=3),
        preview_num_records=None,
    )

    assert failed == []
    assert _by_id(result, COL_REWRITTEN_TEXT) == {"twice": "twice-2", "exhausted": "exhausted-3"}
    assert _by_id(result, COL_REPAIR_ITERATIONS) == {"twice": 2, "exhausted": 3}
    assert _by_id(result, COL_PRIVACY_QA) == {"twice": _PRIVACY_QA, "exhausted": _PRIVACY_QA}


def test_replacement_unavailable_can_be_repaired(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_with_entities: pd.DataFrame,
) -> None:
    pipeline_df = stub_df_with_entities.copy()
    pipeline_df["_anonymizer_row_order"] = [0]
    pipeline_df[COL_REWRITE_REPLACEMENT_READY] = False
    pipeline_df[COL_REWRITTEN_TEXT] = "draft requiring protection"

    initial_eval = pipeline_df.copy()
    initial_eval[COL_NEEDS_REPAIR] = True
    initial_eval[COL_UTILITY_SCORE] = 0.0
    initial_eval[COL_LEAKAGE_MASS] = 1.0
    initial_eval[COL_ANY_HIGH_LEAKED] = False

    repaired = initial_eval.copy()
    repaired[COL_REWRITTEN_TEXT_NEXT] = "apparently safe repaired text"

    passing_eval = repaired.copy()
    passing_eval[COL_REWRITTEN_TEXT] = "apparently safe repaired text"
    passing_eval[COL_NEEDS_REPAIR] = False
    passing_eval[COL_UTILITY_SCORE] = 1.0
    passing_eval[COL_LEAKAGE_MASS] = 0.0

    replace_df = stub_df_with_entities.copy()
    replace_df["_replacement_map"] = [{"replacements": []}]
    adapter = Mock()
    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=pipeline_df, failed_records=[]),
        WorkflowRunResult(dataframe=initial_eval, failed_records=[]),
        WorkflowRunResult(dataframe=repaired, failed_records=[]),
        WorkflowRunResult(dataframe=passing_eval, failed_records=[]),
    ]

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, replace_df)
        result = RewriteWorkflow(adapter=adapter).run(
            stub_df_with_entities,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=1),
        )

    row = result.dataframe.iloc[0]
    assert row[COL_REWRITE_REPLACEMENT_READY] is False or not bool(row[COL_REWRITE_REPLACEMENT_READY])
    assert row[COL_REWRITTEN_TEXT] == "apparently safe repaired text"
    assert row[COL_REPAIR_ITERATIONS] == 1
    assert bool(row[COL_NEEDS_HUMAN_REVIEW]) is True
    workflow_names = [call.kwargs["workflow_name"] for call in adapter.run_workflow.call_args_list]
    assert "rewrite-repair-0" in workflow_names
    assert "rewrite-evaluate-1" in workflow_names


def test_only_failing_rows_sent_to_repair(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_two_entities: pd.DataFrame,
) -> None:
    df = stub_df_two_entities

    adapter = Mock()

    pre_gen_df = df.copy()
    pre_gen_df[COL_DOMAIN] = "BIOGRAPHY_PROFILE"
    pre_gen_df["_anonymizer_row_order"] = [0, 1]

    rewrite_gen_df = pre_gen_df.copy()
    rewrite_gen_df[COL_REWRITTEN_TEXT] = ["Maria works here", "Rob works there"]
    rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

    eval_df = rewrite_gen_df.copy()
    eval_df[COL_NEEDS_REPAIR] = [True, False]
    eval_df[COL_UTILITY_SCORE] = [0.9, 0.9]
    eval_df[COL_LEAKAGE_MASS] = [2.0, 0.1]
    eval_df[COL_ANY_HIGH_LEAKED] = [True, False]
    eval_df[COL_PRIVACY_QA_REANSWER] = [
        {"answers": np.array([{"id": 0}], dtype=object)},
        {"answers": []},
    ]

    failing_row = eval_df[eval_df[COL_NEEDS_REPAIR]].copy()
    repaired_row = failing_row.copy()
    repaired_row[COL_REWRITTEN_TEXT_NEXT] = "Repaired Maria"

    eval_after_repair = failing_row.copy()
    eval_after_repair[COL_NEEDS_REPAIR] = False
    eval_after_repair[COL_UTILITY_SCORE] = 0.9
    eval_after_repair[COL_LEAKAGE_MASS] = 0.1
    eval_after_repair[COL_ANY_HIGH_LEAKED] = False

    replace_df = df.copy()
    replace_df["_replacement_map"] = [{"replacements": []}, {"replacements": []}]

    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=rewrite_gen_df, failed_records=[]),
        WorkflowRunResult(dataframe=eval_df, failed_records=[]),
        WorkflowRunResult(dataframe=repaired_row, failed_records=[]),
        WorkflowRunResult(dataframe=eval_after_repair, failed_records=[]),
    ]

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        wf.run(
            df,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=2),
        )

    repair_calls = [
        call for call in adapter.run_workflow.call_args_list if "repair" in call.kwargs.get("workflow_name", "")
    ]
    assert len(repair_calls) == 1
    repair_input_df = repair_calls[0].args[0]
    assert len(repair_input_df) == 1
    assert json.loads(repair_input_df[COL_PRIVACY_QA_REANSWER].iloc[0]) == {"answers": [{"id": 0}]}


def test_repair_iterations_tracked_per_row(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_with_entities: pd.DataFrame,
) -> None:
    adapter = Mock()

    pre_gen_df = stub_df_with_entities.copy()
    pre_gen_df[COL_DOMAIN] = "BIOGRAPHY_PROFILE"
    pre_gen_df["_anonymizer_row_order"] = [0]

    rewrite_gen_df = pre_gen_df.copy()
    rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works"
    rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

    eval_needs_repair = rewrite_gen_df.copy()
    eval_needs_repair[COL_NEEDS_REPAIR] = True
    eval_needs_repair[COL_UTILITY_SCORE] = 0.9
    eval_needs_repair[COL_LEAKAGE_MASS] = 2.0
    eval_needs_repair[COL_ANY_HIGH_LEAKED] = True

    repaired_df = eval_needs_repair.copy()
    repaired_df[COL_REWRITTEN_TEXT_NEXT] = "Repaired text"

    eval_pass = rewrite_gen_df.copy()
    eval_pass[COL_NEEDS_REPAIR] = False
    eval_pass[COL_UTILITY_SCORE] = 0.9
    eval_pass[COL_LEAKAGE_MASS] = 0.1
    eval_pass[COL_ANY_HIGH_LEAKED] = False
    eval_pass[COL_REPAIR_ITERATIONS] = 1

    replace_df = stub_df_with_entities.copy()
    replace_df["_replacement_map"] = [{"replacements": []}]

    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=rewrite_gen_df, failed_records=[]),
        WorkflowRunResult(dataframe=eval_needs_repair, failed_records=[]),
        WorkflowRunResult(dataframe=repaired_df, failed_records=[]),
        WorkflowRunResult(dataframe=eval_pass, failed_records=[]),
    ]

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        result = wf.run(
            stub_df_with_entities,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=3),
        )

    assert result.dataframe[COL_REPAIR_ITERATIONS].iloc[0] == 1


def test_zero_max_repair_iterations_still_evaluates(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_with_entities: pd.DataFrame,
    stub_replace_df: pd.DataFrame,
    stub_pipeline_df: pd.DataFrame,
    stub_eval_df: pd.DataFrame,
) -> None:
    adapter = Mock()
    adapter.run_workflow.side_effect = _standard_side_effect(stub_pipeline_df, stub_eval_df)

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, stub_replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        result = wf.run(
            stub_df_with_entities,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=0),
        )

    workflow_names = [call.kwargs["workflow_name"] for call in adapter.run_workflow.call_args_list]
    eval_calls = [n for n in workflow_names if "evaluate" in n]
    repair_calls = [n for n in workflow_names if "repair" in n]
    assert len(eval_calls) == 1
    assert repair_calls == []
    assert COL_UTILITY_SCORE in result.dataframe.columns
    assert COL_LEAKAGE_MASS in result.dataframe.columns
    assert COL_ANY_HIGH_LEAKED in result.dataframe.columns


def test_evaluate_dropping_rows_degrades_gracefully(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_two_entities: pd.DataFrame,
) -> None:
    """When evaluate drops a row, the surviving rows still complete the pipeline."""
    df = stub_df_two_entities

    adapter = Mock()

    pre_gen_df = df.copy()
    pre_gen_df[COL_DOMAIN] = "BIOGRAPHY_PROFILE"
    pre_gen_df["_anonymizer_row_order"] = [0, 1]
    pre_gen_df["_anonymizer_record_id"] = ["rec-0", "rec-1"]

    rewrite_gen_df = pre_gen_df.copy()
    rewrite_gen_df[COL_REWRITTEN_TEXT] = ["Maria works here", "Rob works there"]
    rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

    eval_df = rewrite_gen_df.iloc[[0]].copy().reset_index(drop=True)
    eval_df[COL_NEEDS_REPAIR] = False
    eval_df[COL_UTILITY_SCORE] = 0.9
    eval_df[COL_LEAKAGE_MASS] = 0.1
    eval_df[COL_ANY_HIGH_LEAKED] = False

    replace_df = df.copy()
    replace_df["_replacement_map"] = [{"replacements": []}, {"replacements": []}]

    eval_failed = FailedRecord(record_id="rec-1", step="rewrite-evaluate-0", reason="LLM timeout")

    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=rewrite_gen_df, failed_records=[]),
        WorkflowRunResult(dataframe=eval_df, failed_records=[eval_failed]),
    ]

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        result = wf.run(
            df,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=1),
        )

    assert len(result.dataframe) == 1
    assert result.dataframe[COL_REWRITTEN_TEXT].iloc[0] == "Maria works here"
    assert any(f.record_id == "rec-1" for f in result.failed_records)


def test_repair_dropping_rows_degrades_gracefully(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_two_entities: pd.DataFrame,
) -> None:
    """When repair drops a row, the surviving row completes and the failed row is tracked."""
    df = stub_df_two_entities

    adapter = Mock()

    pre_gen_df = df.copy()
    pre_gen_df[COL_DOMAIN] = "BIOGRAPHY_PROFILE"
    pre_gen_df["_anonymizer_row_order"] = [0, 1]
    pre_gen_df["_anonymizer_record_id"] = ["rec-0", "rec-1"]

    rewrite_gen_df = pre_gen_df.copy()
    rewrite_gen_df[COL_REWRITTEN_TEXT] = ["Maria works here", "Rob works there"]
    rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

    # Initial evaluate: both rows need repair
    eval_df = rewrite_gen_df.copy()
    eval_df[COL_NEEDS_REPAIR] = True
    eval_df[COL_UTILITY_SCORE] = [0.9, 0.8]
    eval_df[COL_LEAKAGE_MASS] = [2.0, 2.0]
    eval_df[COL_ANY_HIGH_LEAKED] = True

    # Repair returns only row 0 (rec-1 dropped during repair)
    repaired_df = eval_df.iloc[[0]].copy().reset_index(drop=True)
    repaired_df[COL_REWRITTEN_TEXT_NEXT] = "Repaired Maria"

    repair_failed = FailedRecord(record_id="rec-1", step="rewrite-repair-0", reason="LLM timeout")

    # Re-evaluate after repair: 1 row passes
    eval_after = repaired_df.copy()
    eval_after[COL_NEEDS_REPAIR] = False
    eval_after[COL_UTILITY_SCORE] = 0.9
    eval_after[COL_LEAKAGE_MASS] = 0.1
    eval_after[COL_ANY_HIGH_LEAKED] = False

    replace_df = df.copy()
    replace_df["_replacement_map"] = [{"replacements": []}, {"replacements": []}]

    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=rewrite_gen_df, failed_records=[]),
        # initial evaluate
        WorkflowRunResult(dataframe=eval_df, failed_records=[]),
        # repair (drops rec-1)
        WorkflowRunResult(dataframe=repaired_df, failed_records=[repair_failed]),
        # re-evaluate
        WorkflowRunResult(dataframe=eval_after, failed_records=[]),
    ]

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        result = wf.run(
            df,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=EvaluationCriteria(max_repair_iterations=1),
        )

    assert len(result.dataframe) == 1
    assert result.dataframe[COL_REWRITTEN_TEXT].iloc[0] == "Repaired Maria"
    assert any(f.record_id == "rec-1" for f in result.failed_records)


# ---------------------------------------------------------------------------
# Tests: mixed rows (entity + passthrough)
# ---------------------------------------------------------------------------


def test_passthrough_rows_get_defaults(
    stub_model_configs: list[ModelConfig],
    stub_rewrite_model_selection: RewriteModelSelection,
    stub_replace_model_selection: ReplaceModelSelection,
    stub_df_mixed: pd.DataFrame,
    stub_entities_by_value_with_entities: dict,
) -> None:
    adapter = Mock()

    entity_df = stub_df_mixed[
        stub_df_mixed[COL_ENTITIES_BY_VALUE].apply(lambda x: len(x.get("entities_by_value", [])) > 0)
    ].copy()

    pre_gen_df = entity_df.copy()
    pre_gen_df[COL_DOMAIN] = "BIOGRAPHY_PROFILE"
    pre_gen_df["_anonymizer_row_order"] = [0]

    rewrite_gen_df = pre_gen_df.copy()
    rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works here"
    rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

    eval_df = rewrite_gen_df.copy()
    eval_df[COL_NEEDS_REPAIR] = False
    eval_df[COL_UTILITY_SCORE] = 0.85
    eval_df[COL_LEAKAGE_MASS] = 0.3
    eval_df[COL_ANY_HIGH_LEAKED] = False

    replace_df = entity_df.copy()
    replace_df["_replacement_map"] = [{"replacements": []}]

    adapter.run_workflow.side_effect = [
        WorkflowRunResult(dataframe=rewrite_gen_df, failed_records=[]),
        WorkflowRunResult(dataframe=eval_df, failed_records=[]),
    ]

    with patch(_REPLACE_PATCH) as mock_replace_cls:
        _mock_replace(mock_replace_cls, replace_df)
        wf = RewriteWorkflow(adapter=adapter)
        result = wf.run(
            stub_df_mixed,
            model_configs=stub_model_configs,
            selected_models=stub_rewrite_model_selection,
            replace_model_selection=stub_replace_model_selection,
            privacy_goal=_PRIVACY_GOAL,
            evaluation=_EVALUATION,
        )

    df = result.dataframe
    assert len(df) == 2
    assert df[COL_REWRITTEN_TEXT].iloc[0] == "Maria works here"
    assert df[COL_REWRITTEN_TEXT].iloc[1] == "The sky is blue"
    assert df[COL_UTILITY_SCORE].iloc[1] == 1.0
    assert df[COL_LEAKAGE_MASS].iloc[1] == 0.0
    assert not df[COL_NEEDS_HUMAN_REVIEW].iloc[1]


# ---------------------------------------------------------------------------
# Tests: evaluate() — happy path and column presence
# ---------------------------------------------------------------------------


def test_evaluate_produces_judge_evaluation_column(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection,
    stub_eval_df: pd.DataFrame,
) -> None:
    adapter = Mock()
    wf = RewriteWorkflow(adapter=adapter)

    det_df = stub_eval_df.copy()
    det_df[COL_DETECTION_VALID] = True
    wf._detection_judge_wf = Mock()
    wf._detection_judge_wf.evaluate.return_value = Mock(dataframe=det_df, failed_records=[])

    judge_df = stub_eval_df.copy()
    judge_df[COL_JUDGE_EVALUATION] = [
        {"privacy": {"score": "high"}, "quality": {"score": "high"}, "style": {"score": "medium"}}
    ]
    adapter.run_workflow.return_value = WorkflowRunResult(dataframe=judge_df, failed_records=[])

    result = wf.evaluate(
        stub_eval_df,
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_PRIVACY_GOAL,
    )

    assert COL_JUDGE_EVALUATION in result.dataframe.columns
    assert result.dataframe[COL_JUDGE_EVALUATION].iloc[0] is not None


def test_evaluate_produces_detection_valid_column(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection,
    stub_eval_df: pd.DataFrame,
) -> None:
    adapter = Mock()
    wf = RewriteWorkflow(adapter=adapter)

    det_df = stub_eval_df.copy()
    det_df[COL_DETECTION_VALID] = True
    wf._detection_judge_wf = Mock()
    wf._detection_judge_wf.evaluate.return_value = Mock(dataframe=det_df, failed_records=[])

    judge_df = stub_eval_df.copy()
    judge_df[COL_JUDGE_EVALUATION] = [None]
    adapter.run_workflow.return_value = WorkflowRunResult(dataframe=judge_df, failed_records=[])

    result = wf.evaluate(
        stub_eval_df,
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        compute_detection_validity=True,
    )

    assert COL_DETECTION_VALID in result.dataframe.columns


# ---------------------------------------------------------------------------
# Tests: evaluate() — passthrough rows
# ---------------------------------------------------------------------------


def test_evaluate_skips_passthrough_rows(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection,
    stub_eval_df: pd.DataFrame,
) -> None:
    """evaluate() must only send entity rows to the judges, not passthrough rows."""
    passthrough_row = stub_eval_df.iloc[0].to_dict()
    passthrough_row[COL_ENTITIES_BY_VALUE] = {"entities_by_value": []}
    mixed_df = pd.concat([stub_eval_df, pd.DataFrame([passthrough_row])], ignore_index=True)

    adapter = Mock()
    wf = RewriteWorkflow(adapter=adapter)

    det_df = stub_eval_df.copy()
    det_df[COL_DETECTION_VALID] = True
    wf._detection_judge_wf = Mock()
    wf._detection_judge_wf.evaluate.return_value = Mock(dataframe=det_df, failed_records=[])

    judge_df = stub_eval_df.copy()
    judge_df[COL_JUDGE_EVALUATION] = [None]
    adapter.run_workflow.return_value = WorkflowRunResult(dataframe=judge_df, failed_records=[])

    result = wf.evaluate(
        mixed_df,
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        compute_detection_validity=True,
    )

    detection_call_df = wf._detection_judge_wf.evaluate.call_args.args[0]
    assert len(detection_call_df) == 1
    assert len(result.dataframe) == 2


def test_evaluate_passthrough_rows_get_none_judge_defaults(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection,
    stub_eval_df: pd.DataFrame,
) -> None:
    """Passthrough rows must have COL_JUDGE_EVALUATION=None and COL_DETECTION_VALID=1.0 (trivially valid)."""
    passthrough_row = stub_eval_df.iloc[0].to_dict()
    passthrough_row[COL_ENTITIES_BY_VALUE] = {"entities_by_value": []}
    mixed_df = pd.concat([stub_eval_df, pd.DataFrame([passthrough_row])], ignore_index=True)

    adapter = Mock()
    wf = RewriteWorkflow(adapter=adapter)

    det_df = stub_eval_df.copy()
    det_df[COL_DETECTION_VALID] = True
    wf._detection_judge_wf = Mock()
    wf._detection_judge_wf.evaluate.return_value = Mock(dataframe=det_df, failed_records=[])

    judge_df = stub_eval_df.copy()
    judge_df[COL_JUDGE_EVALUATION] = [None]
    adapter.run_workflow.return_value = WorkflowRunResult(dataframe=judge_df, failed_records=[])

    result = wf.evaluate(
        mixed_df,
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_PRIVACY_GOAL,
        compute_detection_validity=True,
    )

    passthrough_result = result.dataframe[
        result.dataframe[COL_ENTITIES_BY_VALUE].apply(lambda x: len(x.get("entities_by_value", [])) == 0)
    ]
    assert passthrough_result[COL_JUDGE_EVALUATION].iloc[0] is None
    assert passthrough_result[COL_DETECTION_VALID].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# Tests: needs_human_review not overwritten by evaluate()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: _detection_valid_fraction
# ---------------------------------------------------------------------------


def test_detection_valid_fraction_returns_none_when_valid_is_none() -> None:
    row = pd.Series({COL_DETECTION_VALID: None, COL_DETECTION_INVALID_ENTITIES: [], COL_ENTITIES_BY_VALUE: {}})
    assert _detection_valid_fraction(row) is None


def test_detection_valid_fraction_returns_1_when_all_valid() -> None:
    row = pd.Series(
        {
            COL_DETECTION_VALID: True,
            COL_DETECTION_INVALID_ENTITIES: [],
            COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
        }
    )
    assert _detection_valid_fraction(row) == 1.0


def test_detection_valid_fraction_computes_correct_fraction() -> None:
    entities = {"entities_by_value": [{"value": "Alice", "labels": ["first_name", "full_name"]}]}
    row = pd.Series(
        {
            COL_DETECTION_VALID: False,
            COL_DETECTION_INVALID_ENTITIES: [{"value": "Alice", "label": "full_name", "reasoning": "wrong label"}],
            COL_ENTITIES_BY_VALUE: entities,
        }
    )
    result = _detection_valid_fraction(row)
    assert result == pytest.approx(0.5)


def test_detection_valid_fraction_returns_none_on_parse_failure() -> None:
    row = pd.Series(
        {
            COL_DETECTION_VALID: False,
            COL_DETECTION_INVALID_ENTITIES: [{"value": "x", "label": "y", "reasoning": "z"}],
            COL_ENTITIES_BY_VALUE: "not a valid schema payload <<<",
        }
    )
    assert _detection_valid_fraction(row) is None


def test_detection_valid_fraction_returns_none_when_total_is_zero_and_valid_false() -> None:
    """valid=False with an empty entity list — judge flagged invalid but no entities found."""
    row = pd.Series(
        {
            COL_DETECTION_VALID: False,
            COL_DETECTION_INVALID_ENTITIES: [],
            COL_ENTITIES_BY_VALUE: {"entities_by_value": []},
        }
    )
    assert _detection_valid_fraction(row) is None


def test_detection_valid_fraction_returns_none_when_valid_false_and_invalid_entities_empty_but_total_nonzero() -> None:
    """valid=False with empty invalid_entities but entities exist — would spuriously yield 1.0 without the fix."""
    row = pd.Series(
        {
            COL_DETECTION_VALID: False,
            COL_DETECTION_INVALID_ENTITIES: [],
            COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]},
        }
    )
    assert _detection_valid_fraction(row) is None


def test_run_needs_human_review_not_overwritten_by_evaluate(
    stub_model_configs: list[ModelConfig],
    stub_evaluate_model_selection,
    stub_eval_df: pd.DataFrame,
) -> None:
    """COL_NEEDS_HUMAN_REVIEW set during run() must not be modified by evaluate()."""
    run_df = stub_eval_df.copy()
    run_df[COL_NEEDS_HUMAN_REVIEW] = True

    adapter = Mock()
    wf = RewriteWorkflow(adapter=adapter)

    det_df = run_df.copy()
    det_df[COL_DETECTION_VALID] = True
    wf._detection_judge_wf = Mock()
    wf._detection_judge_wf.evaluate.return_value = Mock(dataframe=det_df, failed_records=[])

    judge_df = run_df.copy()
    judge_df[COL_JUDGE_EVALUATION] = [{"privacy": {"score": "high"}}]
    adapter.run_workflow.return_value = WorkflowRunResult(dataframe=judge_df, failed_records=[])

    result = wf.evaluate(
        run_df,
        model_configs=stub_model_configs,
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_PRIVACY_GOAL,
    )

    assert bool(result.dataframe[COL_NEEDS_HUMAN_REVIEW].iloc[0]) is True
