# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from pydantic import BaseModel

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
    COL_WEIGHTED_LEAKAGE_RATE,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_REANSWER,
    COL_QUALITY_QA,
    COL_QUALITY_QA_COMPARE,
    COL_QUALITY_QA_REANSWER,
    COL_REWRITTEN_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.rewrite.parsers import (
    parse_privacy_answers,
    parse_privacy_qa,
    parse_quality_answers,
    parse_quality_compare,
    parse_quality_qa,
    render_template,
)
from anonymizer.engine.schemas.rewrite import (
    PrivacyAnswer,
    PrivacyAnswerItemSchema,
    PrivacyAnswersSchema,
    PrivacyQAPairsSchema,
    QACompareResultsSchema,
    QualityAnswersSchema,
    SensitivityLevel,
)

logger = logging.getLogger("anonymizer.rewrite.evaluate")

# ---------------------------------------------------------------------------
# Generator params
# ---------------------------------------------------------------------------


class MetricsParams(BaseModel):
    sensitivity_weights: dict[str, float]


class RepairNeedsParams(BaseModel):
    auto_repair_privacy: bool
    repair_any_high_leak: bool
    effective_threshold: float


# ---------------------------------------------------------------------------
# Python-rendered prompts
# ---------------------------------------------------------------------------


def _render_quality_reanswer_prompt(row: dict[str, Any]) -> str:
    qa = parse_quality_qa(row.get(COL_QUALITY_QA, {}))
    skeleton = [{"id": item.id, "question": item.question, "answer": ""} for item in qa.items]

    prompt = """You are taking a reading comprehension exam. You will answer each question about the text.

<rules>
- If the text does not state the answer, use "unknown"
- Keep answers concise and factual
- Do not invent details
- You MUST provide an answer for EVERY item in the template below
</rules>

<text>
<<REWRITTEN_TEXT>>
</text>

<task>
Fill in the "answer" field for each item. Do not add or remove items.
</task>
<answer_template>
<<SKELETON>>
</answer_template>
"""
    return render_template(
        prompt,
        {
            "<<SKELETON>>": json.dumps({"answers": skeleton}, indent=2),
            "<<REWRITTEN_TEXT>>": str(row.get(COL_REWRITTEN_TEXT, "")),
        },
    )


def _render_privacy_reanswer_prompt(row: dict[str, Any]) -> str:
    qa = parse_privacy_qa(row.get(COL_PRIVACY_QA))
    skeleton = [
        {"id": item.id, "question": item.question, "answer": "", "confidence": 1.0, "reason": ""} for item in qa.items
    ]

    prompt = """You are a privacy auditor. Read the text and answer each question with "yes" or "no".

<rules>
- Answer "yes" if the specific entity value can be identified or reasonably inferred from the text
- Answer "no" if it cannot
- You must commit to "yes" or "no". Do not hedge.
- Set confidence from 0.0 to 1.0 reflecting how strongly the answer is supported
- Provide a brief reason grounded in the text
- You MUST provide answer, confidence, and reason for EVERY item in the template below
</rules>

<text>
<<REWRITTEN_TEXT>>
</text>

<task>
Fill in the "answer" ("yes"/"no"), "confidence" (0.0-1.0), and "reason" fields for each item.
Do not add or remove items.
</task>
<answer_template>
<<SKELETON>>
</answer_template>
"""
    return render_template(
        prompt,
        {
            "<<SKELETON>>": json.dumps({"answers": skeleton}, indent=2),
            "<<REWRITTEN_TEXT>>": str(row.get(COL_REWRITTEN_TEXT, "")),
        },
    )


def _render_quality_compare_prompt(row: dict[str, Any]) -> str:
    qa = parse_quality_qa(row.get(COL_QUALITY_QA, {}))
    reanswer = parse_quality_answers(row.get(COL_QUALITY_QA_REANSWER))
    answer_lookup: dict[int, str] = {a.id: a.answer for a in reanswer}

    skeleton = [
        {
            "id": item.id,
            "question": item.question,
            "expected_answer": item.reference_answer,
            "student_answer": answer_lookup.get(item.id, "(no answer)"),
            "score": "",
            "reason": "",
        }
        for item in qa.items
    ]

    prompt = """A student took a reading comprehension exam.

<scoring_rubric>
Compare the student's answer to the reference answer and grade each on a 0.0-1.0 scale:
* 1.0 = Perfect match (semantically equivalent)
* 0.7-0.9 = Mostly correct (minor details missing/different)
* 0.4-0.6 = Partially correct (some key info present)
* 0.1-0.3 = Minimal match (tangentially related)
* 0.0 = No match (wrong, "unknown", or contradictory)
</scoring_rubric>

<grading_rules>
* Grade conservatively
* No credit for invented/contradictory info
* Semantically equivalent answers get full credit
* "unknown" answers get 0.0 score
</grading_rules>

<task>
Fill in the "score" (0.0-1.0) and "reason" fields for each item. Do not add or remove items.
</task>
<answer_template>
<<SKELETON>>
</answer_template>
"""
    return render_template(
        prompt,
        {
            "<<SKELETON>>": json.dumps({"per_item": skeleton}, indent=2),
        },
    )


# ---------------------------------------------------------------------------
# Custom parser builders (closure captures expected_ids per row)
# ---------------------------------------------------------------------------


def _make_quality_answer_parser(
    recipe: PydanticResponseRecipe,
    expected_ids: list[int],
) -> Any:
    def parser(response: str) -> QualityAnswersSchema:
        obj = recipe.parse(response)
        payload = obj.model_dump() if hasattr(obj, "model_dump") else obj
        payload["answers"] = _normalize_answer_items(
            payload.get("answers"),
            expected_ids=expected_ids,
            label="quality answer",
            default_item_factory=lambda item_id: {"id": item_id, "answer": "unknown"},
        )
        return QualityAnswersSchema.model_validate(
            payload,
            context={"expected_ids": expected_ids},
        )

    return parser


def _make_privacy_answer_parser(
    recipe: PydanticResponseRecipe,
    expected_ids: list[int],
) -> Any:
    def parser(response: str) -> PrivacyAnswersSchema:
        obj = recipe.parse(response)
        payload = obj.model_dump() if hasattr(obj, "model_dump") else obj
        payload["answers"] = _normalize_answer_items(
            payload.get("answers"),
            expected_ids=expected_ids,
            label="privacy answer",
            default_item_factory=lambda item_id: {
                "id": item_id,
                "answer": "yes",
                "confidence": 1.0,
                "reason": "Model omitted this item; defaulted to highest-confidence leak.",
            },
        )
        return PrivacyAnswersSchema.model_validate(
            payload,
            context={"expected_ids": expected_ids},
        )

    return parser


def _make_qa_compare_parser(
    recipe: PydanticResponseRecipe,
    expected_ids: list[int],
) -> Any:
    def parser(response: str) -> QACompareResultsSchema:
        obj = recipe.parse(response)
        payload = obj.model_dump() if hasattr(obj, "model_dump") else obj
        payload["per_item"] = _normalize_answer_items(
            payload.get("per_item"),
            expected_ids=expected_ids,
            label="quality compare",
            default_item_factory=lambda item_id: {
                "id": item_id,
                "score": 0.0,
                "reason": "Model omitted comparison; defaulted conservatively.",
            },
        )
        return QACompareResultsSchema.model_validate(
            payload,
            context={"expected_ids": expected_ids},
        )

    return parser


def _normalize_answer_items(
    raw_items: Any,
    *,
    expected_ids: list[int],
    label: str,
    default_item_factory: Any,
) -> list[dict[str, Any]]:
    """Normalize LLM answer payloads to the exact expected ID set."""
    if not isinstance(raw_items, list):
        raise TypeError(f"Expected list of {label}s, got {type(raw_items).__name__}")

    expected_set = set(expected_ids)
    by_id: dict[int, dict[str, Any]] = {}
    duplicate_ids: list[int] = []
    extra_ids: list[int] = []

    for item in raw_items:
        if not isinstance(item, dict):
            raise TypeError(f"Expected dict {label}, got {type(item).__name__}")
        item_id = item.get("id")
        if not isinstance(item_id, int):
            raise TypeError(f"Expected integer id in {label}, got {type(item_id).__name__}")
        if item_id not in expected_set:
            extra_ids.append(item_id)
            continue
        if item_id in by_id:
            duplicate_ids.append(item_id)
            continue
        by_id[item_id] = item

    missing_ids = [item_id for item_id in expected_ids if item_id not in by_id]
    if missing_ids or duplicate_ids or extra_ids:
        logger.warning(
            "Evaluator returned malformed %s set; missing=%s duplicate=%s extra=%s. Applying conservative normalization.",
            label,
            missing_ids,
            sorted(set(duplicate_ids)),
            sorted(set(extra_ids)),
        )

    for item_id in missing_ids:
        by_id[item_id] = default_item_factory(item_id)

    return [by_id[item_id] for item_id in expected_ids]


# ---------------------------------------------------------------------------
# Pure-Python metric helpers (public, testable)
# ---------------------------------------------------------------------------


def compute_utility_score(compare_scores: list[float]) -> float:
    """Mean of per-item QA comparison scores.

    Answer coverage is enforced upstream by the context-validated parser
    on QACompareResultsSchema, so all expected IDs are guaranteed present.
    """
    if not compare_scores:
        return 0.0
    return sum(compare_scores) / len(compare_scores)


def compute_leakage_mass(
    privacy_answers: list[PrivacyAnswerItemSchema],
    privacy_qa: PrivacyQAPairsSchema,
    sensitivity_weights: dict[str, float],
) -> float:
    """Confidence-weighted sum of leaked entities.

    An entity is "leaked" when the privacy re-answer is ``yes``.
    Answer coverage is enforced upstream by the context-validated parser
    on PrivacyAnswersSchema, so all expected IDs are guaranteed present.
    """
    qa_lookup: dict[int, str] = {item.id: item.sensitivity for item in privacy_qa.items}
    total = 0.0
    for answer in privacy_answers:
        if answer.answer == PrivacyAnswer.yes:
            sensitivity = qa_lookup.get(answer.id)
            if sensitivity is None:
                continue
            total += sensitivity_weights[sensitivity] * answer.confidence
    return total


def compute_any_high_leaked(
    privacy_answers: list[PrivacyAnswerItemSchema],
    privacy_qa: PrivacyQAPairsSchema,
) -> bool:
    """Return True if any HIGH-sensitivity entity has answer=yes."""
    qa_lookup: dict[int, str] = {item.id: item.sensitivity for item in privacy_qa.items}
    return any(
        answer.answer == PrivacyAnswer.yes and qa_lookup.get(answer.id) == SensitivityLevel.high
        for answer in privacy_answers
    )


def compute_weighted_leakage_rate(
    leakage_mass: float,
    privacy_qa: PrivacyQAPairsSchema,
    sensitivity_weights: dict[str, float],
) -> float:
    """Normalized leakage in [0, 1] relative to maximum possible leakage mass."""
    max_possible_mass = sum(sensitivity_weights[item.sensitivity] for item in privacy_qa.items)
    if max_possible_mass <= 0.0:
        return 0.0
    return leakage_mass / max_possible_mass


def determine_repair_needs(
    *,
    any_high_leaked: bool,
    leakage_mass: float,
    auto_repair_privacy: bool,
    repair_any_high_leak: bool,
    effective_threshold: float,
) -> bool:
    """Decide whether a single row needs repair."""
    if not auto_repair_privacy:
        return False
    if repair_any_high_leak and any_high_leaked:
        return True
    return leakage_mass > effective_threshold


# ---------------------------------------------------------------------------
# Custom column generator factories
# ---------------------------------------------------------------------------


def _make_quality_reanswer_column(evaluator_alias: str) -> Any:
    @custom_column_generator(
        required_columns=[COL_QUALITY_QA, COL_REWRITTEN_TEXT],
        model_aliases=[evaluator_alias],
    )
    def _quality_reanswer(row: dict[str, Any], generator_params: None, models: dict) -> dict[str, Any]:
        qa = parse_quality_qa(row.get(COL_QUALITY_QA, {}))
        expected_ids = [item.id for item in qa.items]

        recipe = PydanticResponseRecipe(data_type=QualityAnswersSchema)
        prompt = recipe.apply_recipe_to_user_prompt(_render_quality_reanswer_prompt(row))
        parser = _make_quality_answer_parser(recipe, expected_ids)
        result, _ = models[evaluator_alias].generate(prompt=prompt, parser=parser, max_correction_steps=3)
        row[COL_QUALITY_QA_REANSWER] = result.model_dump()
        return row

    return _quality_reanswer


def _make_privacy_reanswer_column(evaluator_alias: str) -> Any:
    @custom_column_generator(
        required_columns=[COL_PRIVACY_QA, COL_REWRITTEN_TEXT],
        model_aliases=[evaluator_alias],
    )
    def _privacy_reanswer(row: dict[str, Any], generator_params: None, models: dict) -> dict[str, Any]:
        qa = parse_privacy_qa(row.get(COL_PRIVACY_QA))
        expected_ids = [item.id for item in qa.items]

        recipe = PydanticResponseRecipe(data_type=PrivacyAnswersSchema)
        prompt = recipe.apply_recipe_to_user_prompt(_render_privacy_reanswer_prompt(row))
        parser = _make_privacy_answer_parser(recipe, expected_ids)
        result, _ = models[evaluator_alias].generate(prompt=prompt, parser=parser, max_correction_steps=3)
        row[COL_PRIVACY_QA_REANSWER] = result.model_dump()
        return row

    return _privacy_reanswer


def _make_quality_compare_column(evaluator_alias: str) -> Any:
    @custom_column_generator(
        required_columns=[COL_QUALITY_QA, COL_QUALITY_QA_REANSWER],
        model_aliases=[evaluator_alias],
    )
    def _quality_compare(row: dict[str, Any], generator_params: None, models: dict) -> dict[str, Any]:
        qa = parse_quality_qa(row.get(COL_QUALITY_QA, {}))
        expected_ids = [item.id for item in qa.items]

        recipe = PydanticResponseRecipe(data_type=QACompareResultsSchema)
        prompt = recipe.apply_recipe_to_user_prompt(_render_quality_compare_prompt(row))
        parser = _make_qa_compare_parser(recipe, expected_ids)
        result, _ = models[evaluator_alias].generate(prompt=prompt, parser=parser, max_correction_steps=3)
        row[COL_QUALITY_QA_COMPARE] = result.model_dump()
        return row

    return _quality_compare


# ---------------------------------------------------------------------------
# Pure-Python custom column generators
# ---------------------------------------------------------------------------


@custom_column_generator(
    required_columns=[COL_QUALITY_QA_COMPARE, COL_PRIVACY_QA_REANSWER, COL_PRIVACY_QA, COL_QUALITY_QA],
    side_effect_columns=[COL_LEAKAGE_MASS, COL_WEIGHTED_LEAKAGE_RATE, COL_ANY_HIGH_LEAKED],
)
def _compute_metrics_columns(row: dict[str, Any], generator_params: MetricsParams) -> dict[str, Any]:
    """Compute utility_score, leakage_mass, weighted_leakage_rate, and any_high_leaked from evaluation outputs."""
    _, compare_scores = parse_quality_compare(row.get(COL_QUALITY_QA_COMPARE))
    privacy_answers = parse_privacy_answers(row.get(COL_PRIVACY_QA_REANSWER))
    privacy_qa = parse_privacy_qa(row.get(COL_PRIVACY_QA))

    row[COL_UTILITY_SCORE] = compute_utility_score(compare_scores)
    leakage_mass = compute_leakage_mass(privacy_answers, privacy_qa, generator_params.sensitivity_weights)
    row[COL_LEAKAGE_MASS] = leakage_mass
    row[COL_WEIGHTED_LEAKAGE_RATE] = compute_weighted_leakage_rate(
        leakage_mass, privacy_qa, generator_params.sensitivity_weights
    )
    row[COL_ANY_HIGH_LEAKED] = compute_any_high_leaked(privacy_answers, privacy_qa)
    return row


@custom_column_generator(required_columns=[COL_ANY_HIGH_LEAKED, COL_LEAKAGE_MASS])
def _determine_repair_needs_column(row: dict[str, Any], generator_params: RepairNeedsParams) -> dict[str, Any]:
    """Set _needs_repair flag based on evaluation criteria."""
    row[COL_NEEDS_REPAIR] = determine_repair_needs(
        any_high_leaked=bool(row.get(COL_ANY_HIGH_LEAKED, False)),
        leakage_mass=float(row.get(COL_LEAKAGE_MASS, 0.0)),
        auto_repair_privacy=generator_params.auto_repair_privacy,
        repair_any_high_leak=generator_params.repair_any_high_leak,
        effective_threshold=generator_params.effective_threshold,
    )
    return row


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class EvaluateWorkflow:
    """Evaluate rewritten text for quality preservation and privacy leakage.

    LLM calls use custom columns (not LLMStructuredColumnConfig) so we can
    build a per-row parser that passes ``context={"expected_ids": [...]}`` to
    Pydantic's ``model_validate``. This enforces that the LLM answers every
    question ID -- DD's correction loop retries with the exact missing IDs
    if the LLM skips any. LLMStructuredColumnConfig does not support passing
    Pydantic validation context, which is why custom columns are required.

    The orchestrator uses the output metrics and COL_NEEDS_REPAIR flag to
    decide whether to run RepairWorkflow on failing rows.
    """

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        evaluation: EvaluationCriteria,
        domain: str | None = None,
    ) -> list[ColumnConfigT]:
        evaluator_alias = resolve_model_alias("evaluator", selected_models)
        effective_threshold = evaluation.get_effective_threshold(domain)

        return [
            # Step 1 -- Quality re-answer (LLM with context-validated parser)
            CustomColumnConfig(
                name=COL_QUALITY_QA_REANSWER,
                generator_function=_make_quality_reanswer_column(evaluator_alias),
            ),
            # Step 2 -- Privacy re-answer (LLM with context-validated parser)
            CustomColumnConfig(
                name=COL_PRIVACY_QA_REANSWER,
                generator_function=_make_privacy_reanswer_column(evaluator_alias),
            ),
            # Step 3 -- Quality comparison (LLM with context-validated parser)
            CustomColumnConfig(
                name=COL_QUALITY_QA_COMPARE,
                generator_function=_make_quality_compare_column(evaluator_alias),
            ),
            # Step 4 -- Compute metrics (pure Python)
            CustomColumnConfig(
                name=COL_UTILITY_SCORE,
                generator_function=_compute_metrics_columns,
                generator_params=MetricsParams(sensitivity_weights=evaluation.sensitivity_weights),
            ),
            # Step 5 -- Determine repair needs (pure Python)
            CustomColumnConfig(
                name=COL_NEEDS_REPAIR,
                generator_function=_determine_repair_needs_column,
                generator_params=RepairNeedsParams(
                    auto_repair_privacy=evaluation.auto_repair_privacy,
                    repair_any_high_leak=evaluation.repair_any_high_leak,
                    effective_threshold=effective_threshold,
                ),
            ),
        ]
