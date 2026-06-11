# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import LLMJudgeColumnConfig

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_JUDGE_EVALUATION,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
)
from anonymizer.engine.rewrite.final_judge import (
    FinalJudgeWorkflow,
    PRIVACY_RUBRIC,
    QUALITY_RUBRIC,
    STYLE_RUBRIC,
    _judge_prompt,
)

_STUB_PRIVACY_GOAL = PrivacyGoal(
    protect="Direct identifiers, quasi-identifier combinations, and latent inferences",
    preserve="General utility, content quality, and semantic meaning of the original text",
)


# ---------------------------------------------------------------------------
# Tests: _judge_prompt
# ---------------------------------------------------------------------------


def test_judge_prompt_contains_privacy_goal() -> None:
    prompt = _judge_prompt(_STUB_PRIVACY_GOAL)
    assert "PROTECT:" in prompt
    assert "PRESERVE:" in prompt


def test_judge_prompt_uses_xml_sections() -> None:
    prompt = _judge_prompt(_STUB_PRIVACY_GOAL)
    assert "<privacy_goal>" in prompt
    assert "</privacy_goal>" in prompt
    assert "<original_text>" in prompt
    assert "</original_text>" in prompt
    assert "<rewritten_text>" in prompt
    assert "</rewritten_text>" in prompt
    assert "<task>" in prompt
    assert "</task>" in prompt


def test_judge_prompt_references_required_columns() -> None:
    prompt = _judge_prompt(_STUB_PRIVACY_GOAL)
    assert COL_TEXT in prompt
    assert COL_REWRITTEN_TEXT in prompt


# ---------------------------------------------------------------------------
# Tests: FinalJudgeWorkflow.columns()
# ---------------------------------------------------------------------------


def test_columns_returns_one_config(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    assert len(cols) == 1


def test_judge_column_uses_rewrite_judge_alias(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    judge_cols = [c for c in cols if isinstance(c, LLMJudgeColumnConfig)]
    assert len(judge_cols) == 1
    assert judge_cols[0].model_alias == stub_evaluate_model_selection.rewrite_judge


def test_judge_column_has_three_rubrics(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    judge_col = next(c for c in cols if isinstance(c, LLMJudgeColumnConfig))
    assert judge_col.name == COL_JUDGE_EVALUATION
    score_names = {s.name for s in judge_col.scores}
    assert score_names == {"privacy", "quality", "style"}


def test_judge_rubrics_use_categorical_scores(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    judge_col = next(c for c in cols if isinstance(c, LLMJudgeColumnConfig))
    for score in judge_col.scores:
        assert "low" in score.options
        assert "medium" in score.options
        assert "high" in score.options


def test_rubric_names_match_constants() -> None:
    assert PRIVACY_RUBRIC.name == "privacy"
    assert QUALITY_RUBRIC.name == "quality"
    assert STYLE_RUBRIC.name == "style"
