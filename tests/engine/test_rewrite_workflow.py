# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_DOMAIN,
    COL_ENTITIES_BY_VALUE,
    COL_JUDGE_EVALUATION,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_REPAIR_ITERATIONS,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.engine.rewrite.rewrite_generation import RewriteGenerationResult
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PRIVACY_GOAL = PrivacyGoal(
    protect="All direct identifiers including names, locations, and contact details",
    preserve="Career trajectory, skills, and professional context in abstract terms",
)

_EVALUATION = EvaluationCriteria()


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


def _make_workflow_result(
    df: pd.DataFrame,
    *,
    failed_records: list[FailedRecord] | None = None,
    add_needs_repair: bool = False,
    needs_repair_value: bool = False,
    add_eval_cols: bool = False,
    add_judge_cols: bool = False,
    add_rewrite_cols: bool = False,
) -> WorkflowRunResult:
    """Build a WorkflowRunResult with optional extra columns."""
    result_df = df.copy()
    if add_needs_repair:
        result_df[COL_NEEDS_REPAIR] = needs_repair_value
    if add_eval_cols:
        result_df[COL_UTILITY_SCORE] = 0.9
        result_df[COL_LEAKAGE_MASS] = 0.1
        result_df[COL_ANY_HIGH_LEAKED] = False
    if add_judge_cols:
        result_df[COL_JUDGE_EVALUATION] = None
        result_df[COL_NEEDS_HUMAN_REVIEW] = False
    if add_rewrite_cols:
        result_df[COL_REWRITTEN_TEXT] = "Rewritten text"
        result_df[COL_DOMAIN] = "BIOGRAPHY"
    return WorkflowRunResult(dataframe=result_df, failed_records=failed_records or [])


# ---------------------------------------------------------------------------
# Tests: fast path
# ---------------------------------------------------------------------------


class TestFastPath:
    def test_no_entities_skips_all_workflows(
        self,
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
        self,
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
        assert df[COL_ANY_HIGH_LEAKED].tolist() == [False, False]
        assert df[COL_NEEDS_HUMAN_REVIEW].tolist() == [False, False]
        assert df[COL_JUDGE_EVALUATION].tolist() == [None, None]
        assert df[COL_REPAIR_ITERATIONS].tolist() == [0, 0]

    def test_attrs_propagated_on_fast_path(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_df_no_entities: pd.DataFrame,
    ) -> None:
        stub_df_no_entities.attrs["original_text_column"] = "bio"
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

        assert result.dataframe.attrs.get("original_text_column") == "bio"


# ---------------------------------------------------------------------------
# Tests: full pipeline call order
# ---------------------------------------------------------------------------


class TestCallOrder:
    def test_calls_sub_workflows_in_order(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_df_with_entities: pd.DataFrame,
    ) -> None:
        adapter = Mock()

        pre_gen_df = stub_df_with_entities.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
        pre_gen_df["_anonymizer_row_order"] = [0]

        rewrite_gen_df = pre_gen_df.copy()
        rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works at TechCorp"
        rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

        eval_df = rewrite_gen_df.copy()
        eval_df[COL_NEEDS_REPAIR] = False
        eval_df[COL_UTILITY_SCORE] = 0.9
        eval_df[COL_LEAKAGE_MASS] = 0.1
        eval_df[COL_ANY_HIGH_LEAKED] = False

        judge_df = eval_df.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = False

        adapter.run_workflow.side_effect = [
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
            # evaluate (iteration 0)
            WorkflowRunResult(dataframe=eval_df, failed_records=[]),
            # judge
            WorkflowRunResult(dataframe=judge_df, failed_records=[]),
        ]

        with patch.object(
            RewriteWorkflow,
            "_run_evaluate_repair_loop",
            wraps=None,
        ) as _:
            pass

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
            result = wf.run(
                stub_df_with_entities,
                model_configs=stub_model_configs,
                selected_models=stub_rewrite_model_selection,
                replace_model_selection=stub_replace_model_selection,
                privacy_goal=_PRIVACY_GOAL,
                evaluation=_EVALUATION,
            )

        workflow_names = [call.kwargs["workflow_name"] for call in adapter.run_workflow.call_args_list]
        assert workflow_names[0] == "rewrite-pre-generation"
        assert workflow_names[1].startswith("rewrite-evaluate")
        assert workflow_names[-1] == "rewrite-final-judge"

        assert len(result.dataframe) == 1
        assert result.dataframe[COL_REWRITTEN_TEXT].iloc[0] == "Maria works at TechCorp"


# ---------------------------------------------------------------------------
# Tests: failed records
# ---------------------------------------------------------------------------


class TestFailedRecords:
    def test_accumulated_across_steps(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_df_with_entities: pd.DataFrame,
    ) -> None:
        adapter = Mock()

        pre_gen_df = stub_df_with_entities.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
        pre_gen_df["_anonymizer_row_order"] = [0]

        rewrite_gen_df = pre_gen_df.copy()
        rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works at TechCorp"
        rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

        eval_df = rewrite_gen_df.copy()
        eval_df[COL_NEEDS_REPAIR] = False
        eval_df[COL_UTILITY_SCORE] = 0.9
        eval_df[COL_LEAKAGE_MASS] = 0.1
        eval_df[COL_ANY_HIGH_LEAKED] = False

        judge_df = eval_df.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = False

        failed_pre_gen = FailedRecord(record_id="a", step="rewrite-pre-generation", reason="timeout")
        failed_eval = FailedRecord(record_id="b", step="rewrite-evaluate-0", reason="timeout")
        failed_judge = FailedRecord(record_id="c", step="rewrite-final-judge", reason="timeout")

        adapter.run_workflow.side_effect = [
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[failed_pre_gen]),
            WorkflowRunResult(dataframe=eval_df, failed_records=[failed_eval]),
            WorkflowRunResult(dataframe=judge_df, failed_records=[failed_judge]),
        ]

        rewrite_gen_result = RewriteGenerationResult(
            dataframe=rewrite_gen_df,
            failed_records=[FailedRecord(record_id="d", step="rewrite-generation", reason="timeout")],
        )

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
            result = wf.run(
                stub_df_with_entities,
                model_configs=stub_model_configs,
                selected_models=stub_rewrite_model_selection,
                replace_model_selection=stub_replace_model_selection,
                privacy_goal=_PRIVACY_GOAL,
                evaluation=_EVALUATION,
            )

        record_ids = {f.record_id for f in result.failed_records}
        assert record_ids == {"a", "b", "c", "d"}


# ---------------------------------------------------------------------------
# Tests: attrs propagation
# ---------------------------------------------------------------------------


class TestAttrs:
    def test_attrs_propagated_to_final_output(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_df_with_entities: pd.DataFrame,
    ) -> None:
        stub_df_with_entities.attrs["original_text_column"] = "bio"
        adapter = Mock()

        pre_gen_df = stub_df_with_entities.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
        pre_gen_df["_anonymizer_row_order"] = [0]

        rewrite_gen_df = pre_gen_df.copy()
        rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works"
        rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

        eval_df = rewrite_gen_df.copy()
        eval_df[COL_NEEDS_REPAIR] = False
        eval_df[COL_UTILITY_SCORE] = 0.9
        eval_df[COL_LEAKAGE_MASS] = 0.1
        eval_df[COL_ANY_HIGH_LEAKED] = False

        judge_df = eval_df.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = False

        adapter.run_workflow.side_effect = [
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
            WorkflowRunResult(dataframe=eval_df, failed_records=[]),
            WorkflowRunResult(dataframe=judge_df, failed_records=[]),
        ]

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
            result = wf.run(
                stub_df_with_entities,
                model_configs=stub_model_configs,
                selected_models=stub_rewrite_model_selection,
                replace_model_selection=stub_replace_model_selection,
                privacy_goal=_PRIVACY_GOAL,
                evaluation=_EVALUATION,
            )

        assert result.dataframe.attrs.get("original_text_column") == "bio"


# ---------------------------------------------------------------------------
# Tests: final judge failure tolerance
# ---------------------------------------------------------------------------


class TestJudgeFailure:
    def test_judge_failure_does_not_propagate(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_df_with_entities: pd.DataFrame,
    ) -> None:
        adapter = Mock()

        pre_gen_df = stub_df_with_entities.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
        pre_gen_df["_anonymizer_row_order"] = [0]

        rewrite_gen_df = pre_gen_df.copy()
        rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works"
        rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

        eval_df = rewrite_gen_df.copy()
        eval_df[COL_NEEDS_REPAIR] = False
        eval_df[COL_UTILITY_SCORE] = 0.9
        eval_df[COL_LEAKAGE_MASS] = 0.1
        eval_df[COL_ANY_HIGH_LEAKED] = False

        adapter.run_workflow.side_effect = [
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
            WorkflowRunResult(dataframe=eval_df, failed_records=[]),
            RuntimeError("Judge LLM unavailable"),
        ]

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
            result = wf.run(
                stub_df_with_entities,
                model_configs=stub_model_configs,
                selected_models=stub_rewrite_model_selection,
                replace_model_selection=stub_replace_model_selection,
                privacy_goal=_PRIVACY_GOAL,
                evaluation=_EVALUATION,
            )

        assert len(result.dataframe) == 1
        assert result.dataframe[COL_NEEDS_HUMAN_REVIEW].iloc[0]
        assert result.dataframe[COL_JUDGE_EVALUATION].iloc[0] is None


# ---------------------------------------------------------------------------
# Tests: evaluate-repair loop
# ---------------------------------------------------------------------------


class TestRepairLoop:
    def test_exits_early_when_no_rows_need_repair(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_df_with_entities: pd.DataFrame,
    ) -> None:
        adapter = Mock()

        pre_gen_df = stub_df_with_entities.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
        pre_gen_df["_anonymizer_row_order"] = [0]

        rewrite_gen_df = pre_gen_df.copy()
        rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works"
        rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

        eval_df = rewrite_gen_df.copy()
        eval_df[COL_NEEDS_REPAIR] = False
        eval_df[COL_UTILITY_SCORE] = 0.9
        eval_df[COL_LEAKAGE_MASS] = 0.1
        eval_df[COL_ANY_HIGH_LEAKED] = False

        judge_df = eval_df.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = False

        adapter.run_workflow.side_effect = [
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
            # evaluate-0: no repair needed
            WorkflowRunResult(dataframe=eval_df, failed_records=[]),
            # judge
            WorkflowRunResult(dataframe=judge_df, failed_records=[]),
        ]

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
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

    def test_runs_up_to_max_iterations(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_df_with_entities: pd.DataFrame,
    ) -> None:
        adapter = Mock()
        max_iters = 2

        pre_gen_df = stub_df_with_entities.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
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
        repaired_df[COL_REWRITTEN_TEXT] = "Repaired text"

        judge_df = repaired_df.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = True

        adapter.run_workflow.side_effect = [
            # pre-gen
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
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
            # judge
            WorkflowRunResult(dataframe=judge_df, failed_records=[]),
        ]

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
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

    def test_only_failing_rows_sent_to_repair(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_entities_by_value_with_entities: dict,
    ) -> None:
        df = pd.DataFrame(
            {
                COL_TEXT: ["Alice works here", "Bob works there"],
                COL_ENTITIES_BY_VALUE: [
                    stub_entities_by_value_with_entities,
                    stub_entities_by_value_with_entities,
                ],
            }
        )

        adapter = Mock()

        pre_gen_df = df.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
        pre_gen_df["_anonymizer_row_order"] = [0, 1]

        rewrite_gen_df = pre_gen_df.copy()
        rewrite_gen_df[COL_REWRITTEN_TEXT] = ["Maria works here", "Rob works there"]
        rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

        eval_df = rewrite_gen_df.copy()
        eval_df[COL_NEEDS_REPAIR] = [True, False]
        eval_df[COL_UTILITY_SCORE] = [0.9, 0.9]
        eval_df[COL_LEAKAGE_MASS] = [2.0, 0.1]
        eval_df[COL_ANY_HIGH_LEAKED] = [True, False]

        failing_row = eval_df[eval_df[COL_NEEDS_REPAIR]].copy()
        repaired_row = failing_row.copy()
        repaired_row[COL_REWRITTEN_TEXT] = "Repaired Maria"

        eval_after_repair = rewrite_gen_df.copy()
        eval_after_repair[COL_NEEDS_REPAIR] = False
        eval_after_repair[COL_UTILITY_SCORE] = 0.9
        eval_after_repair[COL_LEAKAGE_MASS] = 0.1
        eval_after_repair[COL_ANY_HIGH_LEAKED] = False

        judge_df = eval_after_repair.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = False

        adapter.run_workflow.side_effect = [
            # pre-gen
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
            # evaluate-0
            WorkflowRunResult(dataframe=eval_df, failed_records=[]),
            # repair-0 (should get 1 row)
            WorkflowRunResult(dataframe=repaired_row, failed_records=[]),
            # evaluate-1 (all pass)
            WorkflowRunResult(dataframe=eval_after_repair, failed_records=[]),
            # judge
            WorkflowRunResult(dataframe=judge_df, failed_records=[]),
        ]

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
            wf.run(
                df,
                model_configs=stub_model_configs,
                selected_models=stub_rewrite_model_selection,
                replace_model_selection=stub_replace_model_selection,
                privacy_goal=_PRIVACY_GOAL,
                evaluation=EvaluationCriteria(max_repair_iterations=2),
            )

        # The repair call should have received only 1 row
        repair_calls = [
            call for call in adapter.run_workflow.call_args_list if "repair" in call.kwargs.get("workflow_name", "")
        ]
        assert len(repair_calls) == 1
        repair_input_df = repair_calls[0].args[0]
        assert len(repair_input_df) == 1

    def test_repair_iterations_tracked_per_row(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_df_with_entities: pd.DataFrame,
    ) -> None:
        adapter = Mock()

        pre_gen_df = stub_df_with_entities.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
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
        repaired_df[COL_REWRITTEN_TEXT] = "Repaired text"

        eval_pass = rewrite_gen_df.copy()
        eval_pass[COL_NEEDS_REPAIR] = False
        eval_pass[COL_UTILITY_SCORE] = 0.9
        eval_pass[COL_LEAKAGE_MASS] = 0.1
        eval_pass[COL_ANY_HIGH_LEAKED] = False
        eval_pass[COL_REPAIR_ITERATIONS] = 1

        judge_df = eval_pass.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = False

        adapter.run_workflow.side_effect = [
            # pre-gen
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
            # evaluate-0 -> needs repair
            WorkflowRunResult(dataframe=eval_needs_repair, failed_records=[]),
            # repair-0
            WorkflowRunResult(dataframe=repaired_df, failed_records=[]),
            # evaluate-1 -> passes
            WorkflowRunResult(dataframe=eval_pass, failed_records=[]),
            # judge
            WorkflowRunResult(dataframe=judge_df, failed_records=[]),
        ]

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
            result = wf.run(
                stub_df_with_entities,
                model_configs=stub_model_configs,
                selected_models=stub_rewrite_model_selection,
                replace_model_selection=stub_replace_model_selection,
                privacy_goal=_PRIVACY_GOAL,
                evaluation=EvaluationCriteria(max_repair_iterations=3),
            )

        assert result.dataframe[COL_REPAIR_ITERATIONS].iloc[0] == 1

    def test_evaluate_dropping_rows_degrades_gracefully(
        self,
        stub_model_configs: list[ModelConfig],
        stub_rewrite_model_selection: RewriteModelSelection,
        stub_replace_model_selection: ReplaceModelSelection,
        stub_entities_by_value_with_entities: dict,
    ) -> None:
        """When evaluate drops a row, the surviving rows still complete the pipeline."""
        df = pd.DataFrame(
            {
                COL_TEXT: ["Alice works here", "Bob works there"],
                COL_ENTITIES_BY_VALUE: [
                    stub_entities_by_value_with_entities,
                    stub_entities_by_value_with_entities,
                ],
            }
        )

        adapter = Mock()

        pre_gen_df = df.copy()
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
        pre_gen_df["_anonymizer_row_order"] = [0, 1]
        pre_gen_df["_anonymizer_record_id"] = ["rec-0", "rec-1"]

        rewrite_gen_df = pre_gen_df.copy()
        rewrite_gen_df[COL_REWRITTEN_TEXT] = ["Maria works here", "Rob works there"]
        rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

        # Evaluate returns only 1 of 2 rows (rec-1 dropped)
        eval_df = rewrite_gen_df.iloc[[0]].copy().reset_index(drop=True)
        eval_df[COL_NEEDS_REPAIR] = False
        eval_df[COL_UTILITY_SCORE] = 0.9
        eval_df[COL_LEAKAGE_MASS] = 0.1
        eval_df[COL_ANY_HIGH_LEAKED] = False

        judge_df = eval_df.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = False

        eval_failed = FailedRecord(record_id="rec-1", step="rewrite-evaluate-0", reason="LLM timeout")

        adapter.run_workflow.side_effect = [
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
            WorkflowRunResult(dataframe=eval_df, failed_records=[eval_failed]),
            WorkflowRunResult(dataframe=judge_df, failed_records=[]),
        ]

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
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


# ---------------------------------------------------------------------------
# Tests: mixed rows (entity + passthrough)
# ---------------------------------------------------------------------------


class TestMixedRows:
    def test_passthrough_rows_get_defaults(
        self,
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
        pre_gen_df[COL_DOMAIN] = "BIOGRAPHY"
        pre_gen_df["_anonymizer_row_order"] = [0]

        rewrite_gen_df = pre_gen_df.copy()
        rewrite_gen_df[COL_REWRITTEN_TEXT] = "Maria works here"
        rewrite_gen_df[COL_REPAIR_ITERATIONS] = 0

        eval_df = rewrite_gen_df.copy()
        eval_df[COL_NEEDS_REPAIR] = False
        eval_df[COL_UTILITY_SCORE] = 0.85
        eval_df[COL_LEAKAGE_MASS] = 0.3
        eval_df[COL_ANY_HIGH_LEAKED] = False

        judge_df = eval_df.copy()
        judge_df[COL_JUDGE_EVALUATION] = None
        judge_df[COL_NEEDS_HUMAN_REVIEW] = False

        adapter.run_workflow.side_effect = [
            WorkflowRunResult(dataframe=pre_gen_df, failed_records=[]),
            WorkflowRunResult(dataframe=eval_df, failed_records=[]),
            WorkflowRunResult(dataframe=judge_df, failed_records=[]),
        ]

        rewrite_gen_result = RewriteGenerationResult(dataframe=rewrite_gen_df, failed_records=[])

        wf = RewriteWorkflow(adapter=adapter)
        with patch.object(wf._rewrite_gen_wf, "run", return_value=rewrite_gen_result):
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
        # Row 0 = entity row (rewritten)
        assert df[COL_REWRITTEN_TEXT].iloc[0] == "Maria works here"
        # Row 1 = passthrough row (original text)
        assert df[COL_REWRITTEN_TEXT].iloc[1] == "The sky is blue"
        assert df[COL_UTILITY_SCORE].iloc[1] == 1.0
        assert df[COL_LEAKAGE_MASS].iloc[1] == 0.0
        assert not df[COL_NEEDS_HUMAN_REVIEW].iloc[1]
