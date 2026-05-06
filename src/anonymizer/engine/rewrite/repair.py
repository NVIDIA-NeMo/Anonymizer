# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from pydantic import BaseModel

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_LEAKED_PRIVACY_ITEMS,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.rewrite.parsers import (
    parse_privacy_answers,
    parse_privacy_qa,
)
from anonymizer.engine.schemas.rewrite import (
    PrivacyAnswer,
    PrivacyAnswerItemSchema,
    PrivacyQAPairsSchema,
    RewriteOutputSchema,
)

logger = logging.getLogger("anonymizer.rewrite.repair")


# ---------------------------------------------------------------------------
# Generator params
# ---------------------------------------------------------------------------


class RepairParams(BaseModel):
    privacy_goal_str: str
    max_privacy_leak: float


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _leaked_items_text(
    privacy_answers: list[PrivacyAnswerItemSchema],
    privacy_qa: PrivacyQAPairsSchema,
) -> str:
    """Format leaked privacy items for the repair prompt."""
    qa_lookup: dict[int, Any] = {item.id: item for item in privacy_qa.items}
    lines = []
    for answer in privacy_answers:
        if answer.answer == PrivacyAnswer.yes:
            item = qa_lookup.get(answer.id)
            if item:
                evidence_str = ""
                if answer.evidence:
                    quoted = "; ".join(f'"{e}"' for e in answer.evidence)
                    evidence_str = f"\n  Evidence: {quoted}"
                lines.append(
                    f'- [{item.sensitivity.upper()}] {item.entity_label}: "{item.entity_value}" -- {item.question} '
                    f"(confidence_leakage_occurred: {answer.confidence:.2f}; reason: {answer.reason})"
                    f"{evidence_str}"
                )
    return "\n".join(lines)


def _render_repair_prompt(row: dict[str, Any], params: RepairParams) -> str:
    """Build the repair prompt from row values (no Jinja2)."""
    prompt = """You are helping to rewrite text for privacy protection.

<privacy_goal>
<<PRIVACY_GOAL>>
</privacy_goal>

<original_text>
<<ORIGINAL_TEXT>>
</original_text>

<previous_rewrite>
<<REWRITTEN_TEXT>>
</previous_rewrite>

<privacy_issues>
The following questions about sensitive information can still be answered from the rewrite.
Leakage mass: <<LEAKAGE_MASS>> (target: < <<MAX_PRIVACY_LEAK>>)<<HIGH_WARN>>
<<LEAKED_ITEMS>>
</privacy_issues>

<adversarial_goal>
Assume a motivated reader is actively trying to infer these attributes.
Your rewrite is successful ONLY if the attribute can no longer be reliably inferred.
</adversarial_goal>

<inference_rules>
To prevent inference, you MUST remove or weaken all evidence enabling it.

Use one or more of the following:
- Generalize or remove specific details
- Break combinations of clues (not just individual phrases)
- Replace concrete descriptions with vague or abstract ones
- Introduce ambiguity so multiple interpretations are plausible
- Replace implication-causing statements with alternative explanations that do NOT uniquely support the attribute
- Remove or rewrite statements that imply a condition through patterns or context
- Disrupt causal or narrative chains that lead to inference

If multiple clues point to the same attribute, you MUST break the combination.
</inference_rules>

<critical_warnings>
- DO NOT simply rephrase the same evidence
- DO NOT preserve combinations of clues that reveal the attribute
- If a reasonable reader can still guess the attribute, it is NOT fixed
</critical_warnings>

<success_criteria>
A privacy issue is FIXED only if:
- A reasonable reader cannot confidently infer the attribute
- Multiple plausible interpretations exist
- The original inference is no longer the most likely explanation

If the attribute is still the most natural conclusion, the issue is NOT fixed.
</success_criteria>

<task>
Fix the privacy leaks by suppressing inference.

Before rewriting, internally identify all leakage patterns and how they arise (phrases, combinations, narrative implications). Use this to guide your rewrite. Do NOT output this analysis.

You MUST do the following for EACH leaked item:
- Identify all phrases, clues, and narrative patterns enabling the inference
- Remove, weaken, or rewrite them so the attribute is no longer the most likely conclusion
- Break combinations of clues, not just individual phrases

Requirements:
- Rewrite as much as needed; do not preserve partially leaking sentences
- Any phrase or pattern contributing to a leaked attribute MUST be modified or removed
- Even subtle or indirect contributors MUST be neutralized
- Treat common life-pattern signals (e.g., routine, schedule, aging, daily activities) as leakage ONLY when they contribute
- Fix shared patterns across multiple leaks
- Do not alter content that does not contribute to leakage
- Ensure multiple plausible interpretations remain

Before finalizing:
- Ask: "What would a motivated reader guess?"
- If the leaked attribute is still the most likely guess, revise again

Maintain overall coherence, consistency, and naturalness (utility score: <<UTILITY_SCORE>>).

Provide ONLY the rewritten text.
</task>
"""
    replacements = {
        "<<PRIVACY_GOAL>>": params.privacy_goal_str,
        "<<MAX_PRIVACY_LEAK>>": str(params.max_privacy_leak),
        "<<ORIGINAL_TEXT>>": str(row.get(COL_TEXT, "")),
        "<<REWRITTEN_TEXT>>": str(row.get(COL_REWRITTEN_TEXT, "")),
        "<<LEAKAGE_MASS>>": str(row.get(COL_LEAKAGE_MASS, 0.0)),
        "<<HIGH_WARN>>": "\nWARNING: HIGH-SENSITIVITY LEAK DETECTED - must be fixed!"
        if bool(row.get(COL_ANY_HIGH_LEAKED, False))
        else "",
        "<<LEAKED_ITEMS>>": str(row.get(COL_LEAKED_PRIVACY_ITEMS, "")),
        "<<UTILITY_SCORE>>": str(row.get(COL_UTILITY_SCORE, 0.0)),
    }
    return substitute_placeholders(prompt, replacements)


# ---------------------------------------------------------------------------
# Custom column generators
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_PRIVACY_QA_REANSWER, COL_PRIVACY_QA])
def _inject_leaked_items_column(row: dict[str, Any]) -> dict[str, Any]:
    """Format leaked privacy items into a text block for the repair prompt."""
    privacy_answers = parse_privacy_answers(row.get(COL_PRIVACY_QA_REANSWER))
    privacy_qa = parse_privacy_qa(row.get(COL_PRIVACY_QA))
    row[COL_LEAKED_PRIVACY_ITEMS] = _leaked_items_text(privacy_answers, privacy_qa)
    return row


def _make_repair_column(repairer_alias: str) -> Any:
    """Factory that creates a repair column generator bound to a resolved model alias."""

    @custom_column_generator(
        required_columns=[
            COL_LEAKED_PRIVACY_ITEMS,
            COL_REWRITTEN_TEXT,
            COL_TEXT,
            COL_LEAKAGE_MASS,
            COL_ANY_HIGH_LEAKED,
            COL_UTILITY_SCORE,
        ],
        model_aliases=[repairer_alias],
    )
    def _repair_column(row: dict[str, Any], generator_params: RepairParams, models: dict) -> dict[str, Any]:
        recipe = PydanticResponseRecipe(data_type=RewriteOutputSchema)
        prompt = recipe.apply_recipe_to_user_prompt(_render_repair_prompt(row, generator_params))
        result, _ = models[repairer_alias].generate(prompt=prompt, parser=recipe.parse, max_correction_steps=3)
        row[COL_REWRITTEN_TEXT_NEXT] = result.rewritten_text
        return row

    return _repair_column


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RepairWorkflow:
    """Repair rewritten text that failed privacy evaluation.

    The orchestrator is responsible for filtering to rows where
    _needs_repair=True before calling this workflow. Every row
    processed here receives an unconditional repair LLM call.
    """

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        effective_threshold: float,
    ) -> list[ColumnConfigT]:
        repairer_alias = resolve_model_alias("repairer", selected_models)

        return [
            # Step 1 -- Format leaked items for repair prompt (pure Python)
            CustomColumnConfig(
                name=COL_LEAKED_PRIVACY_ITEMS,
                generator_function=_inject_leaked_items_column,
            ),
            # Step 2 -- Repair rewritten text (LLM via custom column)
            CustomColumnConfig(
                name=COL_REWRITTEN_TEXT_NEXT,
                generator_function=_make_repair_column(repairer_alias),
                generator_params=RepairParams(
                    privacy_goal_str=privacy_goal.to_prompt_string(),
                    max_privacy_leak=effective_threshold,
                ),
            ),
        ]
