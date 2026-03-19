# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT
from pydantic import BaseModel

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import EvaluationCriteria
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_LEAKAGE_MASS,
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
from anonymizer.engine.schemas.rewrite import (
    PrivacyAnswer,
    PrivacyAnswerItemSchema,
    PrivacyAnswersSchema,
    PrivacyQAPairsSchema,
    QACompareResultsSchema,
    QualityAnswerSchema,
    QualityAnswersSchema,
    QualityQAItemSchema,
    QualityQAPairsSchema,
    SensitivityLevel,
)

_COL_NEEDS_REPAIR = "_needs_repair"
_COL_QA_COMPARISON_BLOCK = "_qa_comparison_block"


# Schema field names validated at import time.
def _field(model: type, name: str) -> str:
    if name not in model.model_fields:
        raise KeyError(f"{model.__name__} has no field '{name}'")
    return name


_F_ITEMS = _field(QualityQAPairsSchema, "items")
_F_ANSWERS = _field(QualityAnswersSchema, "answers")
_F_ID = _field(QualityQAItemSchema, "id")
_F_QUESTION = _field(QualityQAItemSchema, "question")
_F_ANSWER = _field(QualityAnswerSchema, "answer")
_F_REFERENCE_ANSWER = _field(QualityQAItemSchema, "reference_answer")


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
# Prompts (Jinja2 templates resolved by NDD)
# ---------------------------------------------------------------------------


def _quality_reanswer_prompt() -> str:
    prompt = """You are taking a reading comprehension exam. You will answer each question about the text.

<rules>
- If the text does not state the answer, use "unknown"
- Keep answers concise and factual
- Do not invent details
</rules>

<text>
{{ <<COL_REWRITTEN_TEXT>> }}
</text>

<questions>
{% for it in <<COL_QUALITY_QA>>['<<F_ITEMS>>'] %}- {{ it['<<F_ID>>'] }}: {{ it['<<F_QUESTION>>'] }}
{% endfor %}</questions>
"""
    return (
        prompt.replace("<<COL_REWRITTEN_TEXT>>", COL_REWRITTEN_TEXT)
        .replace("<<COL_QUALITY_QA>>", COL_QUALITY_QA)
        .replace("<<F_ITEMS>>", _F_ITEMS)
        .replace("<<F_ID>>", _F_ID)
        .replace("<<F_QUESTION>>", _F_QUESTION)
    )


def _privacy_reanswer_prompt() -> str:
    prompt = """You are a privacy auditor. Read the text and answer each question with "yes" or "no".

<rules>
- Answer "yes" if the specific entity value can be identified or reasonably inferred from the text.
- Answer "no" if it cannot be answered from the text.
- You must commit to "yes" or "no". Do not hedge.
</rules>

<text>
{{ <<COL_REWRITTEN_TEXT>> }}
</text>

<questions>
{% for it in <<COL_PRIVACY_QA>>['<<F_ITEMS>>'] %}- {{ it['<<F_ID>>'] }}: {{ it['<<F_QUESTION>>'] }}
{% endfor %}</questions>
"""
    return (
        prompt.replace("<<COL_REWRITTEN_TEXT>>", COL_REWRITTEN_TEXT)
        .replace("<<COL_PRIVACY_QA>>", COL_PRIVACY_QA)
        .replace("<<F_ITEMS>>", _F_ITEMS)
        .replace("<<F_ID>>", _F_ID)
        .replace("<<F_QUESTION>>", _F_QUESTION)
    )


def _quality_compare_prompt() -> str:
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

<qa_pairs>
{{ <<COL_QA_COMPARISON_BLOCK>> }}
</qa_pairs>
"""
    return prompt.replace("<<COL_QA_COMPARISON_BLOCK>>", _COL_QA_COMPARISON_BLOCK)


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _parse_privacy_answers(raw: Any) -> list[PrivacyAnswerItemSchema]:
    if isinstance(raw, PrivacyAnswersSchema):
        return raw.answers
    if isinstance(raw, dict):
        return PrivacyAnswersSchema.model_validate(raw).answers
    raise TypeError(f"Expected PrivacyAnswersSchema or dict, got {type(raw).__name__}")


def _parse_quality_qa(raw: Any) -> QualityQAPairsSchema:
    if isinstance(raw, QualityQAPairsSchema):
        return raw
    if isinstance(raw, dict):
        return QualityQAPairsSchema.model_validate(raw)
    raise TypeError(f"Expected QualityQAPairsSchema or dict, got {type(raw).__name__}")


def _parse_quality_answers(raw: Any) -> list[QualityAnswerSchema]:
    if isinstance(raw, QualityAnswersSchema):
        return raw.answers
    if isinstance(raw, dict):
        return QualityAnswersSchema.model_validate(raw).answers
    raise TypeError(f"Expected QualityAnswersSchema or dict, got {type(raw).__name__}")


def _parse_quality_compare(raw: Any) -> list[float]:
    if isinstance(raw, QACompareResultsSchema):
        return [item.score for item in raw.per_item]
    if isinstance(raw, dict):
        parsed = QACompareResultsSchema.model_validate(raw)
        return [item.score for item in parsed.per_item]
    raise TypeError(f"Expected QACompareResultsSchema or dict, got {type(raw).__name__}")


def _parse_privacy_qa(raw: Any) -> PrivacyQAPairsSchema:
    if isinstance(raw, PrivacyQAPairsSchema):
        return raw
    if isinstance(raw, dict):
        return PrivacyQAPairsSchema.model_validate(raw)
    raise TypeError(f"Expected PrivacyQAPairsSchema or dict, got {type(raw).__name__}")


# ---------------------------------------------------------------------------
# Pure-Python metric helpers (public, testable)
# ---------------------------------------------------------------------------


def compute_utility_score(compare_scores: list[float]) -> float:
    """Mean of per-item QA comparison scores, or 0.0 if empty."""
    if not compare_scores:
        return 0.0
    return sum(compare_scores) / len(compare_scores)


def compute_leakage_mass(
    privacy_answers: list[PrivacyAnswerItemSchema],
    privacy_qa: PrivacyQAPairsSchema,
    sensitivity_weights: dict[str, float],
) -> float:
    """Weighted sum of leaked entities.

    An entity is "leaked" when the privacy re-answer is ``yes``.
    The weight comes from ``sensitivity_weights[entity.sensitivity]``.
    """
    qa_lookup: dict[int, str] = {item.id: item.sensitivity for item in privacy_qa.items}
    total = 0.0
    for answer in privacy_answers:
        if answer.answer == PrivacyAnswer.yes:
            sensitivity = qa_lookup.get(answer.id, SensitivityLevel.low)
            weight = sensitivity_weights[sensitivity]
            total += weight
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
    return leakage_mass >= effective_threshold


# ---------------------------------------------------------------------------
# Custom column generators
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_QUALITY_QA, COL_QUALITY_QA_REANSWER])
def _pair_qa_for_comparison(row: dict[str, Any]) -> dict[str, Any]:
    """Match quality QA items with their reanswered counterparts by ID for the compare prompt."""
    qa = _parse_quality_qa(row.get(COL_QUALITY_QA, {}))
    reanswer = _parse_quality_answers(row.get(COL_QUALITY_QA_REANSWER))
    answer_lookup: dict[int, str] = {a.id: a.answer for a in reanswer}

    lines = []
    for item in qa.items:
        student = answer_lookup.get(item.id, "(no answer)")
        lines.append(
            f"- question ID: {item.id}\n"
            f"  question: {item.question}\n"
            f"  expected answer: {item.reference_answer}\n"
            f"  student answer: {student}"
        )
    row[_COL_QA_COMPARISON_BLOCK] = "\n".join(lines)
    return row


@custom_column_generator(
    required_columns=[COL_QUALITY_QA_COMPARE, COL_PRIVACY_QA_REANSWER, COL_PRIVACY_QA],
    side_effect_columns=[COL_LEAKAGE_MASS, COL_ANY_HIGH_LEAKED],
)
def _compute_metrics_columns(row: dict[str, Any], generator_params: MetricsParams) -> dict[str, Any]:
    """Compute utility_score, leakage_mass, and any_high_leaked from evaluation outputs."""
    compare_scores = _parse_quality_compare(row.get(COL_QUALITY_QA_COMPARE))
    privacy_answers = _parse_privacy_answers(row.get(COL_PRIVACY_QA_REANSWER))
    privacy_qa = _parse_privacy_qa(row.get(COL_PRIVACY_QA))

    row[COL_UTILITY_SCORE] = compute_utility_score(compare_scores)
    row[COL_LEAKAGE_MASS] = compute_leakage_mass(privacy_answers, privacy_qa, generator_params.sensitivity_weights)
    row[COL_ANY_HIGH_LEAKED] = compute_any_high_leaked(privacy_answers, privacy_qa)
    return row


@custom_column_generator(required_columns=[COL_ANY_HIGH_LEAKED, COL_LEAKAGE_MASS])
def _determine_repair_needs_column(row: dict[str, Any], generator_params: RepairNeedsParams) -> dict[str, Any]:
    """Set _needs_repair flag based on evaluation criteria."""
    row[_COL_NEEDS_REPAIR] = determine_repair_needs(
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

    Produces metric columns (utility_score, leakage_mass, any_high_leaked)
    and a _needs_repair flag. The orchestrator uses these to decide whether
    to run RepairWorkflow on failing rows and whether to iterate.
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
            # Step 1 -- Quality re-answer (LLM)
            LLMStructuredColumnConfig(
                name=COL_QUALITY_QA_REANSWER,
                prompt=_quality_reanswer_prompt(),
                model_alias=evaluator_alias,
                output_format=QualityAnswersSchema,
            ),
            # Step 2 -- Privacy re-answer (LLM)
            LLMStructuredColumnConfig(
                name=COL_PRIVACY_QA_REANSWER,
                prompt=_privacy_reanswer_prompt(),
                model_alias=evaluator_alias,
                output_format=PrivacyAnswersSchema,
            ),
            # Step 3 -- Pair QA items by ID for comparison (pure Python)
            CustomColumnConfig(
                name=_COL_QA_COMPARISON_BLOCK,
                generator_function=_pair_qa_for_comparison,
            ),
            # Step 4 -- Quality comparison (LLM)
            LLMStructuredColumnConfig(
                name=COL_QUALITY_QA_COMPARE,
                prompt=_quality_compare_prompt(),
                model_alias=evaluator_alias,
                output_format=QACompareResultsSchema,
            ),
            # Step 5 -- Compute metrics (pure Python)
            CustomColumnConfig(
                name=COL_UTILITY_SCORE,
                generator_function=_compute_metrics_columns,
                generator_params=MetricsParams(sensitivity_weights=evaluation.sensitivity_weights),
            ),
            # Step 6 -- Determine repair needs (pure Python)
            CustomColumnConfig(
                name=_COL_NEEDS_REPAIR,
                generator_function=_determine_repair_needs_column,
                generator_params=RepairNeedsParams(
                    auto_repair_privacy=evaluation.auto_repair_privacy,
                    repair_any_high_leak=evaluation.repair_any_high_leak,
                    effective_threshold=effective_threshold,
                ),
            ),
        ]
