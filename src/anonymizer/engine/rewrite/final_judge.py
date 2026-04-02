# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
from anonymizer.engine.rewrite.parsers import render_template

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
Evaluate the rewritten text on three separate dimensions:
1. privacy protection,
2. content quality / meaning preservation,
3. naturalness of writing.

Score each dimension independently.
</task>

<privacy_scoring_instructions>
- Judge privacy relative to the original text, not in isolation.
- The privacy score must reflect the likelihood that a realistic attacker could discover or match the rewritten text back to the original person, case, listing, or record using available information.
- Focus on residual linkage risk after the rewrite, not on the mere presence of detail.

- When evaluating identifiers, explicitly compare rewritten values to original values.
- Only treat an identifier as leaked if the SAME original value, a near-direct equivalent, or an obvious reversible transformation remains.
- If no original identifiers remain, do NOT treat identifiers present in the rewritten text as leaked.
- The presence of the same value alone does not imply a privacy leak unless the value is inherently identifying or uniquely distinguishing.

- Distinguish between:
  * direct identifiers: information that uniquely identifies the protected subject on its own (not other people mentioned)
  * quasi-identifiers: information that contributes to identification only in combination with other details

- Treat preserved direct identifiers as high risk, but assess both:
  * their strength (how uniquely identifying they are), and
  * their role (whether they directly identify the subject or are secondary/relational, such as names of relatives)
- Secondary or relational identifiers and distinctive combinations increase linkage risk but typically require additional context. They should not alone imply easy re-identification and should only result in the lowest scores (1–2) when they provide a short, reliable, and low-ambiguity path to identifying the protected subject.

- Treat contextual attributes (e.g., organizations, locations, roles, affiliations) as quasi-identifiers unless they uniquely identify the individual on their own.

- Evaluate linkage risk based on how much the remaining information narrows the set of plausible matches.
  * Do not infer high risk from specificity alone.
  * Only penalize quasi-identifiers when they significantly reduce the candidate set or form a distinctive or rare combination.

- Distinguish between levels of linkage risk:
  * plausible linkage (requires effort, auxiliary information, or narrowing across many candidates) → scores 5–6
  * strong but indirect signals (significantly narrows candidates but still requires effort) → scores 3–4
  * likely or easy re-identification (directly identifying or easily traceable with minimal effort) → scores 1–2

- A small number of weak or secondary identifiers (e.g., common names of relatives) should increase risk but should not by themselves result in low scores unless they substantially narrow the candidate set.

- Apply the stated privacy goal. Do not require unnecessary over-redaction if the rewrite adequately achieves the intended level of protection.
</privacy_scoring_instructions>

<quality_scoring_instructions>
    - Judge quality based on how well the rewritten text preserves the important meaning, facts,
      and utility of the original while pursuing the privacy goal.
    - Focus on preservation of substantive content, not verbatim wording.
    - Do NOT penalize the rewrite for changing or removing details when those changes are necessary
      for privacy and the important meaning is still preserved.
    - Reward rewrites that preserve the main events, relationships, roles, chronology, and
      conclusions of the original.
    - Penalize rewrites that omit important facts, distort the meaning, introduce contradictions,
      or materially change the core message.
    - Minor loss of detail should reduce the score only slightly; major loss of important
      information should reduce it substantially.
    - Judge quality independently from privacy and naturalness. A rewrite can be privacy-protective
      but still low quality if it loses important meaning, and it can be high quality even if
      some surface details are changed.
</quality_scoring_instructions>

<naturalness_scoring_instructions>
    - Judge naturalness based on whether the rewritten text reads as fluent, coherent, and human-written.
    - Focus on readability, grammatical correctness, clarity, and smooth phrasing.
    - Do NOT penalize the rewrite merely for using different wording, sentence structure, or
      level of specificity from the original.
    - Reward rewrites that sound natural and internally consistent.
    - Penalize awkward phrasing, repetitive wording, broken grammar, incoherence, unnatural
      insertions, placeholder-like language, or text that feels machine-generated.
    - Judge naturalness independently from privacy and quality. A rewrite can be natural even if it
      changes content, and it can preserve content while still sounding awkward.
</naturalness_scoring_instructions>
"""
    return render_template(
        prompt,
        {
            "<<PRIVACY_GOAL>>": privacy_goal.to_prompt_string(),
            "<<COL_TEXT>>": _jinja(COL_TEXT),
            "<<COL_REWRITTEN_TEXT>>": _jinja(COL_REWRITTEN_TEXT),
        },
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
    description="Privacy protection effectiveness - how well the rewrite reduces the risk of linking the text back to the original person, case, listing, or record while preserving the stated privacy goal.",
    options={
        10: "Excellent privacy protection. Original direct identifiers are removed or safely transformed, and remaining details are too generic to support realistic linkage.",
        9: "Very strong privacy protection. No meaningful original direct identifiers remain, and only weak, common contextual details are left.",
        8: "Strong privacy protection. Original direct identifiers are removed; some quasi-identifiers remain, but linkage risk is low.",
        7: "Good privacy protection. Original direct identifiers are removed; some contextual specifics remain, but re-identification would be difficult.",
        6: "Moderate privacy protection. Original direct identifiers are removed, but a distinctive factual bundle or several quasi-identifiers still create noticeable linkage risk.",
        5: "Mixed privacy protection. No obvious original direct identifiers remain, but the rewrite preserves enough distinctive detail that linkage is plausible.",
        4: "Weak privacy protection. Important identifying or highly distinctive details remain, making linkage fairly feasible.",
        3: "Poor privacy protection. The rewrite leaves strong identifying clues or a highly unique fact pattern that makes re-identification likely.",
        2: "Very poor privacy protection. One or more original direct identifiers or near-direct equivalents remain, or the text is easily linkable to the original.",
        1: "Severe privacy failure. Original direct identifiers are plainly exposed or the person, case, listing, or record is essentially still directly identifiable.",
    },
)

QUALITY_RUBRIC = Score(
    name="quality",
    description="Content quality and utility preservation - how well is important information retained?",
    options={
        10: "Perfect - complete information preservation",
        9: "Excellent - nearly all information retained",
        8: "Strong - content well preserved",
        7: "Good - minor details lost",
        6: "Fair - most information preserved",
        5: "Acceptable - some information loss but usable",
        4: "Below average - important details missing",
        3: "Poor - significant information loss",
        2: "Very poor - major content degradation",
        1: "Unusable - critical information lost",
    },
)

NATURALNESS_RUBRIC = Score(
    name="naturalness",
    description="Writing quality and naturalness - does it read like the original text?",
    options={
        10: "Perfect - indistinguishable from human-written",
        9: "Very natural - fluent writing",
        8: "Natural - reads well",
        7: "Good - minor awkwardness",
        6: "Fair - mostly natural",
        5: "Acceptable - readable but not smooth",
        4: "Below average - some unnatural phrases",
        3: "Awkward - noticeably artificial",
        2: "Very awkward - difficult to read",
        1: "Unreadable - severely unnatural",
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
