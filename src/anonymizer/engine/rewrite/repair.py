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
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITTEN_TEXT,
    COL_REWRITTEN_TEXT_NEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.rewrite.parsers import (
    field,
    normalize_payload,
    parse_privacy_answers,
    parse_privacy_qa,
    parse_sensitivity_disposition,
    render_template,
)
from anonymizer.engine.schemas.rewrite import (
    EntityDispositionSchema,
    PrivacyAnswer,
    PrivacyAnswerItemSchema,
    PrivacyQAPairsSchema,
    RewriteOutputSchema,
)

logger = logging.getLogger("anonymizer.rewrite.repair")


_F_NEEDS_PROTECTION = field(EntityDispositionSchema, "needs_protection")
_F_ENTITY_LABEL = field(EntityDispositionSchema, "entity_label")
_F_ENTITY_VALUE = field(EntityDispositionSchema, "entity_value")
_F_PROTECTION_METHOD = field(EntityDispositionSchema, "protection_method_suggestion")
_F_PROTECTION_REASON = field(EntityDispositionSchema, "protection_reason")


def _replacement_map_is_empty(raw_map: Any) -> bool:
    """Return True when the replacement map is absent or has no replacements."""
    normalized = normalize_payload(raw_map)
    if normalized is None:
        return True
    if not isinstance(normalized, dict):
        return False
    replacements = normalized.get("replacements")
    return isinstance(replacements, list) and len(replacements) == 0


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
                lines.append(
                    f'- [{item.sensitivity.upper()}] {item.entity_label}: "{item.entity_value}" -- {item.question} '
                    f"(confidence_leakage_occurred: {answer.confidence:.2f}; reason: {answer.reason})"
                )
    return "\n".join(lines)


def _format_protection_block(row: dict[str, Any]) -> str:
    """Format the protection decisions section of the repair prompt."""
    disposition = parse_sensitivity_disposition(row.get(COL_SENSITIVITY_DISPOSITION, {}))
    lines = [
        f'- {e.entity_label}: "{e.entity_value}" -> {e.protection_method_suggestion}\n  Reason: {e.protection_reason}'
        for e in disposition.sensitivity_disposition
        if e.needs_protection
    ]
    return "\n".join(lines)


def _render_repair_prompt(row: dict[str, Any], params: RepairParams) -> str:
    """Build the repair prompt from row values (no Jinja2)."""
    disposition = parse_sensitivity_disposition(row.get(COL_SENSITIVITY_DISPOSITION, {}))
    has_replace_entities = any(
        e.protection_method_suggestion == "replace" and e.needs_protection for e in disposition.sensitivity_disposition
    )
    raw_map = row.get(COL_REPLACEMENT_MAP_FOR_PROMPT)
    map_is_empty = _replacement_map_is_empty(raw_map)
    if has_replace_entities and map_is_empty:
        logger.warning(
            "Repair prompt has entities requiring replacement but COL_REPLACEMENT_MAP_FOR_PROMPT is empty; "
            "the LLM will have no synthetic values to use."
        )

    prompt = """You are helping to rewrite text for privacy protection.

<privacy_goal>
<<PRIVACY_GOAL>>
</privacy_goal>

<protection_decisions>
<<PROTECTION_BLOCK>>
</protection_decisions>

<original_text>
<<ORIGINAL_TEXT>>
</original_text>

<previous_rewrite>
<<REWRITTEN_TEXT>>
</previous_rewrite>

<replacement_map>
<<REPLACEMENT_MAP>>
</replacement_map>

<privacy_issues>
The following questions about sensitive information can still be answered from the rewrite.
Leakage mass: <<LEAKAGE_MASS>> (target: < <<MAX_PRIVACY_LEAK>>)<<HIGH_WARN>>
<<LEAKED_ITEMS>>
</privacy_issues>

<task>
Fix the privacy leaks by following the protection decisions above:
1. For "replace" - use synthetic values from the replacement map
2. For "generalize" - use broader categories
3. For "remove" - omit the detail entirely
4. For "suppress_inference" - modify text so the attribute cannot be reliably inferred

If privacy issues remain, you may override earlier "leave_as_is" decisions when needed to satisfy privacy goals.
You may modify surrounding context beyond the explicit entity span to break inferential leakage.
Modify the text such that latent attributes cannot be reliably inferred by a motivated reader.
This may involve reducing specificity, removing or weakening key details, breaking causal or identifying linkages,
or introducing ambiguity, while preserving overall narrative coherence.

Maintain content quality (utility score: <<UTILITY_SCORE>>), consistency, and naturalness.

Provide ONLY the rewritten text. Do not include explanations, comments, or markdown formatting.
</task>
"""
    replacements = {
        "<<PRIVACY_GOAL>>": params.privacy_goal_str,
        "<<MAX_PRIVACY_LEAK>>": str(params.max_privacy_leak),
        "<<PROTECTION_BLOCK>>": _format_protection_block(row),
        "<<ORIGINAL_TEXT>>": str(row.get(COL_TEXT, "")),
        "<<REWRITTEN_TEXT>>": str(row.get(COL_REWRITTEN_TEXT, "")),
        "<<REPLACEMENT_MAP>>": str(row.get(COL_REPLACEMENT_MAP_FOR_PROMPT, "")),
        "<<LEAKAGE_MASS>>": str(row.get(COL_LEAKAGE_MASS, 0.0)),
        "<<HIGH_WARN>>": "\nWARNING: HIGH-SENSITIVITY LEAK DETECTED - must be fixed!"
        if bool(row.get(COL_ANY_HIGH_LEAKED, False))
        else "",
        "<<LEAKED_ITEMS>>": str(row.get(COL_LEAKED_PRIVACY_ITEMS, "")),
        "<<UTILITY_SCORE>>": str(row.get(COL_UTILITY_SCORE, 0.0)),
    }
    return render_template(prompt, replacements)


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
            COL_SENSITIVITY_DISPOSITION,
            COL_TEXT,
            COL_REPLACEMENT_MAP_FOR_PROMPT,
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
