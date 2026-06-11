# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import LLMJudgeColumnConfig, Score
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_JUDGE_EVALUATION,
    COL_REWRITTEN_TEXT,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders

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
3. style of writing.

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
- Secondary or relational identifiers and distinctive combinations increase linkage risk but typically require additional context. They should not alone imply easy re-identification and should only result in a `low` score when they provide a short, reliable, and low-ambiguity path to identifying the protected subject.

- Treat contextual attributes (e.g., organizations, locations, roles, affiliations) as quasi-identifiers unless they uniquely identify the individual on their own.

- Evaluate linkage risk based on how much the remaining information narrows the set of plausible matches.
  * Do not infer high risk from specificity alone.
  * Only penalize quasi-identifiers when they significantly reduce the candidate set or form a distinctive or rare combination.

- Distinguish between levels of linkage risk:
  * high  — original direct identifiers removed; remaining details create low linkage risk
  * medium — no obvious direct identifiers, but a distinctive quasi-identifier bundle creates noticeable linkage risk
  * low   — one or more direct identifiers or near-equivalents remain, or the record is easily or near-certainly linkable

- A small number of weak or secondary identifiers (e.g., common names of relatives) should increase risk but should not by themselves result in a `low` score unless they substantially narrow the candidate set.

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
    - Judge quality independently from privacy and style. A rewrite can be privacy-protective
      but still low quality if it loses important meaning, and it can be high quality even if
      some surface details are changed.
    - Score as:
      * high   — important meaning, facts, and structure fully preserved
      * medium — most content preserved; minor details lost or slightly distorted
      * low    — material loss of important information, contradictions, or distorted core meaning
</quality_scoring_instructions>

<style_scoring_instructions>
    - Judge style based on whether the rewritten text reads as fluent, coherent, and human-written.
    - Focus on readability, grammatical correctness, clarity, and smooth phrasing.
    - Do NOT penalize the rewrite merely for using different wording, sentence structure, or
      level of specificity from the original.
    - Reward rewrites that sound natural and internally consistent.
    - Penalize awkward phrasing, repetitive wording, broken grammar, incoherence, unnatural
      insertions, placeholder-like language, or text that feels machine-generated.
    - Judge style independently from privacy and quality. A rewrite can be natural even if it
      changes content, and it can preserve content while still sounding awkward.
    - Score as:
      * high   — fluent, coherent, human-written prose
      * medium — mostly readable; isolated awkward phrasing or stiff transitions
      * low    — noticeably unnatural; broken grammar, placeholder-like language, or machine-generated feel
</style_scoring_instructions>
"""
    return substitute_placeholders(
        prompt,
        {
            "<<PRIVACY_GOAL>>": privacy_goal.to_prompt_string(),
            "<<COL_TEXT>>": _jinja(COL_TEXT),
            "<<COL_REWRITTEN_TEXT>>": _jinja(COL_REWRITTEN_TEXT),
        },
    )


# ---------------------------------------------------------------------------
# Rubrics
# ---------------------------------------------------------------------------

PRIVACY_RUBRIC = Score(
    name="privacy",
    description="Privacy protection — how well the rewrite removes linkage risk to the original record.",
    options={
        "high": "Original direct identifiers removed; remaining quasi-identifiers create low linkage risk.",
        "medium": "No obvious direct identifiers remain, but a distinctive quasi-identifier bundle creates noticeable linkage risk.",
        "low": "The record is easily or near-certainly linkable back to the original: key direct identifiers remain, or enough identifying detail survives that re-identification requires minimal effort regardless of how many entities were successfully transformed.",
    },
)

QUALITY_RUBRIC = Score(
    name="quality",
    description="Content quality — how well important meaning, facts, and structure are preserved.",
    options={
        "high": "Important meaning, facts, and structure fully preserved.",
        "medium": "Most content preserved; minor details lost or slightly distorted.",
        "low": "Material loss of important information, contradictions, or distorted core meaning.",
    },
)

STYLE_RUBRIC = Score(
    name="style",
    description="Writing style — does the rewritten text read as fluent, coherent, human-written prose?",
    options={
        "high": "Reads as fluent, coherent, human-written prose.",
        "medium": "Mostly readable; isolated awkward phrasing or stiff transitions.",
        "low": "Noticeably unnatural; broken grammar, placeholder-like language, or machine-generated feel.",
    },
)


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class FinalJudgeWorkflow:
    """Holistic LLM judge evaluation of privacy, quality, and style.

    Produces ``COL_JUDGE_EVALUATION`` only — informational, not used for any
    automated decisions. ``COL_NEEDS_HUMAN_REVIEW`` is computed separately in
    the evaluate-repair loop based on objective metrics.
    """

    def columns(
        self,
        *,
        selected_models: EvaluateModelSelection,
        privacy_goal: PrivacyGoal,
    ) -> list[ColumnConfigT]:
        judge_alias = resolve_model_alias("rewrite_judge", selected_models)

        return [
            LLMJudgeColumnConfig(
                name=COL_JUDGE_EVALUATION,
                prompt=_judge_prompt(privacy_goal),
                model_alias=judge_alias,
                scores=[PRIVACY_RUBRIC, QUALITY_RUBRIC, STYLE_RUBRIC],
            ),
        ]
