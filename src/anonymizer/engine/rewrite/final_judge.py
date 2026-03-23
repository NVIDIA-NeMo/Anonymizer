# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMJudgeColumnConfig, Score
from data_designer.config.column_types import ColumnConfigT
from pydantic import BaseModel

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
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias

logger = logging.getLogger("anonymizer.rewrite.final_judge")


# ---------------------------------------------------------------------------
# Generator params
# ---------------------------------------------------------------------------


class HumanReviewParams(BaseModel):
    flag_utility_below: float | None
    flag_leakage_mass_above: float | None


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _judge_prompt(privacy_goal: PrivacyGoal) -> str:
    prompt = """You are an expert judge evaluating privacy-preserving text rewrites.

<privacy_goal>
<<PRIVACY_GOAL>>
</privacy_goal>

<original_text>
<<COL_TEXT>>
</original_text>

<rewritten_text>
<<COL_REWRITTEN_TEXT>>
</rewritten_text>

<task>
Evaluate independently whether the rewrite achieves the privacy goal while maintaining quality and naturalness.
</task>"""
    return (
        prompt.replace("<<PRIVACY_GOAL>>", privacy_goal.to_prompt_string())
        .replace("<<COL_TEXT>>", _jinja(COL_TEXT))
        .replace("<<COL_REWRITTEN_TEXT>>", _jinja(COL_REWRITTEN_TEXT))
    )


# ---------------------------------------------------------------------------
# Custom column generators
# ---------------------------------------------------------------------------


@custom_column_generator(
    required_columns=[COL_REWRITTEN_TEXT, COL_UTILITY_SCORE, COL_LEAKAGE_MASS, COL_ANY_HIGH_LEAKED],
)
def _determine_needs_human_review(row: dict[str, Any], generator_params: HumanReviewParams) -> dict[str, Any]:
    """Flag rows that need manual review based on objective metrics."""
    if row.get(COL_REWRITTEN_TEXT) is None:
        row[COL_NEEDS_HUMAN_REVIEW] = True
        return row

    if row[COL_ANY_HIGH_LEAKED]:
        row[COL_NEEDS_HUMAN_REVIEW] = True
        return row

    if generator_params.flag_utility_below is not None:
        if float(row[COL_UTILITY_SCORE]) < generator_params.flag_utility_below:
            row[COL_NEEDS_HUMAN_REVIEW] = True
            return row

    if generator_params.flag_leakage_mass_above is not None:
        if float(row[COL_LEAKAGE_MASS]) > generator_params.flag_leakage_mass_above:
            row[COL_NEEDS_HUMAN_REVIEW] = True
            return row

    row[COL_NEEDS_HUMAN_REVIEW] = False
    return row


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


PRIVACY_RUBRIC = Score(
    name="privacy",
    description="Privacy protection effectiveness - how well are identifiable entities obscured?",
    options={
        "10": "Perfect privacy - complete anonymization, impossible to re-identify",
        "9": "Excellent privacy - virtually impossible to re-identify",
        "8": "Strong privacy - very difficult to re-identify",
        "7": "Good privacy - only low-sensitivity context remains",
        "6": "Fair privacy - minor quasi-identifiers remain",
        "5": "Acceptable - some low-sensitivity leaks",
        "4": "Moderate leaks - mostly medium/low sensitivity information",
        "3": "Significant leaks - some high-sensitivity information remains",
        "2": "Major privacy issues - multiple high-sensitivity entities exposed",
        "1": "Severe privacy leaks - direct identifiers easily recoverable",
    },
)

QUALITY_RUBRIC = Score(
    name="quality",
    description="Content quality and utility preservation - how well is important information retained?",
    options={
        "10": "Perfect - complete information preservation",
        "9": "Excellent - nearly all information retained",
        "8": "Strong - content well preserved",
        "7": "Good - minor details lost",
        "6": "Fair - most information preserved",
        "5": "Acceptable - some information loss but usable",
        "4": "Below average - important details missing",
        "3": "Poor - significant information loss",
        "2": "Very poor - major content degradation",
        "1": "Unusable - critical information lost",
    },
)

NATURALNESS_RUBRIC = Score(
    name="naturalness",
    description="Writing quality and naturalness - does it read like the original text?",
    options={
        "10": "Perfect - indistinguishable from human-written",
        "9": "Very natural - fluent writing",
        "8": "Natural - reads well",
        "7": "Good - minor awkwardness",
        "6": "Fair - mostly natural",
        "5": "Acceptable - readable but not smooth",
        "4": "Below average - some unnatural phrases",
        "3": "Awkward - noticeably artificial",
        "2": "Very awkward - difficult to read",
        "1": "Unreadable - severely unnatural",
    },
)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class FinalJudgeWorkflow:
    """Holistic LLM judge evaluation of privacy, quality, and naturalness.

    Produces ``COL_JUDGE_EVALUATION`` (informational only -- not used for
    automated decisions) and ``COL_NEEDS_HUMAN_REVIEW`` (based on objective
    metrics from the evaluate step).
    """

    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        evaluation: EvaluationCriteria,
    ) -> list[ColumnConfigT]:
        judge_alias = resolve_model_alias("judge", selected_models)

        return [
            LLMJudgeColumnConfig(
                name=COL_JUDGE_EVALUATION,
                prompt=_judge_prompt(privacy_goal),
                model_alias=judge_alias,
                scores=[PRIVACY_RUBRIC, QUALITY_RUBRIC, NATURALNESS_RUBRIC],
            ),
            CustomColumnConfig(
                name=COL_NEEDS_HUMAN_REVIEW,
                generator_function=_determine_needs_human_review,
                generator_params=HumanReviewParams(
                    flag_utility_below=evaluation.flag_utility_below,
                    flag_leakage_mass_above=evaluation.flag_leakage_mass_above,
                ),
            ),
        ]
