# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import CustomColumnConfig, LLMJudgeColumnConfig
import pytest

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_JUDGE_EVALUATION,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.rewrite.final_judge import (
    FinalJudgeWorkflow,
    HumanReviewParams,
    _determine_needs_human_review,
    _judge_prompt,
)

_STUB_PRIVACY_GOAL = PrivacyGoal(
    protect="Direct identifiers, quasi-identifier combinations, and latent inferences",
    preserve="General utility, content quality, and semantic meaning of the original text",
)

_STUB_EVALUATION = EvaluationCriteria()


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


def test_columns_returns_two_configs(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
        evaluation=_STUB_EVALUATION,
    )
    assert len(cols) == 2


def test_judge_column_uses_judge_alias(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
        evaluation=_STUB_EVALUATION,
    )
    judge_cols = [c for c in cols if isinstance(c, LLMJudgeColumnConfig)]
    assert len(judge_cols) == 1
    assert judge_cols[0].model_alias == stub_rewrite_model_selection.judge


def test_judge_column_has_three_rubrics(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
        evaluation=_STUB_EVALUATION,
    )
    judge_col = next(c for c in cols if isinstance(c, LLMJudgeColumnConfig))
    assert judge_col.name == COL_JUDGE_EVALUATION
    score_names = {s.name for s in judge_col.scores}
    assert score_names == {"privacy", "quality", "naturalness"}
    for score in judge_col.scores:
        assert 1 in score.options
        assert 10 in score.options


def test_needs_human_review_column_present(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
        evaluation=_STUB_EVALUATION,
    )
    custom_cols = [c for c in cols if isinstance(c, CustomColumnConfig)]
    assert len(custom_cols) == 1
    assert custom_cols[0].name == COL_NEEDS_HUMAN_REVIEW


def test_needs_human_review_column_uses_evaluation_thresholds(
    stub_rewrite_model_selection: RewriteModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    evaluation = EvaluationCriteria(flag_utility_below=0.6, flag_leakage_mass_above=1.5)
    cols = wf.columns(
        selected_models=stub_rewrite_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
        evaluation=evaluation,
    )
    custom_col = next(c for c in cols if isinstance(c, CustomColumnConfig))
    params = HumanReviewParams.model_validate(custom_col.generator_params)
    assert params.flag_utility_below == 0.6
    assert params.flag_leakage_mass_above == 1.5


# ---------------------------------------------------------------------------
# Tests: _determine_needs_human_review
# ---------------------------------------------------------------------------


def _make_row(
    rewritten_text: str | None = "some rewritten text",
    utility_score: float = 0.8,
    leakage_mass: float = 0.5,
    any_high_leaked: bool = False,
) -> dict:
    return {
        COL_REWRITTEN_TEXT: rewritten_text,
        COL_UTILITY_SCORE: utility_score,
        COL_LEAKAGE_MASS: leakage_mass,
        COL_ANY_HIGH_LEAKED: any_high_leaked,
    }


def test_needs_human_review_flags_none_rewrite() -> None:
    row = _make_row(rewritten_text=None)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    result = _determine_needs_human_review(row, generator_params=params)
    assert result[COL_NEEDS_HUMAN_REVIEW] is True


def test_needs_human_review_flags_low_utility() -> None:
    row = _make_row(utility_score=0.3)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    result = _determine_needs_human_review(row, generator_params=params)
    assert result[COL_NEEDS_HUMAN_REVIEW] is True


def test_needs_human_review_flags_high_leakage() -> None:
    row = _make_row(leakage_mass=3.0)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    result = _determine_needs_human_review(row, generator_params=params)
    assert result[COL_NEEDS_HUMAN_REVIEW] is True


def test_needs_human_review_flags_any_high_leaked() -> None:
    row = _make_row(any_high_leaked=True)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    result = _determine_needs_human_review(row, generator_params=params)
    assert result[COL_NEEDS_HUMAN_REVIEW] is True


def test_needs_human_review_false_when_all_good() -> None:
    row = _make_row(utility_score=0.8, leakage_mass=0.5, any_high_leaked=False)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    result = _determine_needs_human_review(row, generator_params=params)
    assert result[COL_NEEDS_HUMAN_REVIEW] is False


def test_needs_human_review_none_thresholds_skip_checks() -> None:
    row = _make_row(utility_score=0.1, leakage_mass=10.0)
    params = HumanReviewParams(flag_utility_below=None, flag_leakage_mass_above=None)
    result = _determine_needs_human_review(row, generator_params=params)
    assert result[COL_NEEDS_HUMAN_REVIEW] is False


def test_needs_human_review_exact_threshold_utility() -> None:
    row = _make_row(utility_score=0.50)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    result = _determine_needs_human_review(row, generator_params=params)
    assert result[COL_NEEDS_HUMAN_REVIEW] is False


def test_needs_human_review_exact_threshold_leakage() -> None:
    row = _make_row(leakage_mass=2.0)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    result = _determine_needs_human_review(row, generator_params=params)
    assert result[COL_NEEDS_HUMAN_REVIEW] is False


def test_needs_human_review_raises_on_invalid_utility_score() -> None:
    row = _make_row(utility_score=None)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    with pytest.raises(TypeError):
        _determine_needs_human_review(row, generator_params=params)


def test_needs_human_review_raises_on_invalid_leakage_mass() -> None:
    row = _make_row(leakage_mass=None)
    params = HumanReviewParams(flag_utility_below=0.50, flag_leakage_mass_above=2.0)
    with pytest.raises(TypeError):
        _determine_needs_human_review(row, generator_params=params)
