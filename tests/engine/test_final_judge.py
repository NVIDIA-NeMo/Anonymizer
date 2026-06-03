# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import CustomColumnConfig

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_JUDGE_EVALUATION,
)
from anonymizer.engine.rewrite.final_judge import (
    PRIVACY_RUBRIC,
    QUALITY_RUBRIC,
    STYLE_RUBRIC,
    FinalJudgeWorkflow,
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


def test_judge_prompt_has_window_placeholders_and_scales() -> None:
    prompt = _judge_prompt(_STUB_PRIVACY_GOAL)
    # Per-window slices are injected via these Jinja placeholders (not DD column refs).
    assert "{{ original_text }}" in prompt
    assert "{{ rewritten_text }}" in prompt
    # The categorical rubric scales must be embedded in the prompt for the direct model call.
    for name in ("privacy", "quality", "style"):
        assert name in prompt
    assert "<output_format>" in prompt


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
    judge_col = next(c for c in cols if c.name == COL_JUDGE_EVALUATION)
    assert isinstance(judge_col, CustomColumnConfig)
    assert judge_col.generator_params.alias == stub_evaluate_model_selection.rewrite_judge


def test_judge_column_is_windowed_generator_with_three_rubrics(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    wf = FinalJudgeWorkflow()
    cols = wf.columns(
        selected_models=stub_evaluate_model_selection,
        privacy_goal=_STUB_PRIVACY_GOAL,
    )
    judge_col = next(c for c in cols if c.name == COL_JUDGE_EVALUATION)
    assert isinstance(judge_col, CustomColumnConfig)
    template = judge_col.generator_params.prompt_template
    for name in ("privacy", "quality", "style"):
        assert name in template
    assert judge_col.generator_params.max_render_chars > 0


def test_judge_rubrics_use_categorical_scores() -> None:
    for score in (PRIVACY_RUBRIC, QUALITY_RUBRIC, STYLE_RUBRIC):
        assert "low" in score.options
        assert "medium" in score.options
        assert "high" in score.options


def test_rubric_names_match_constants() -> None:
    assert PRIVACY_RUBRIC.name == "privacy"
    assert QUALITY_RUBRIC.name == "quality"
    assert STYLE_RUBRIC.name == "style"


def test_judge_prompt_references_style_not_naturalness() -> None:
    prompt = _judge_prompt(_STUB_PRIVACY_GOAL)
    assert "style" in prompt.lower()
    assert "naturalness" not in prompt.lower()


def test_judge_prompt_references_categorical_scale() -> None:
    prompt = _judge_prompt(_STUB_PRIVACY_GOAL)
    assert "high" in prompt
    assert "medium" in prompt
    assert "low" in prompt
