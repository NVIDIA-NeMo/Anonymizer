# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer Replace privacy questions without converting evaluation failures to leaks."""

from __future__ import annotations

import json
import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from pydantic import BaseModel, Field, ValidationError, model_validator

from anonymizer.engine.constants import COL_PRIVACY_QA, COL_PRIVACY_QA_ERROR, COL_PRIVACY_QA_REANSWER, COL_REPLACED_TEXT
from anonymizer.engine.evaluation.privacy_answers import answer_privacy_qa
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.rewrite.parsers import parse_privacy_qa
from anonymizer.engine.schemas import PrivacyAnswerItemSchema, PrivacyAnswersSchema, PrivacyQAPairsSchema

logger = logging.getLogger("anonymizer.evaluation.replace.privacy_answers")


def render_replace_privacy_answer_prompt(qa: PrivacyQAPairsSchema, text: str) -> str:
    """Audit original values using only Replace output, without priming confidence."""
    skeleton = [
        {"id": item.id, "question": item.question, "answer": "", "confidence": None, "reason": "", "evidence": []}
        for item in qa.items
    ]
    prompt = """You are auditing text that has undergone entity replacement or redaction.

For each question, determine whether the specified original entity value
can still be identified or reasonably inferred from the anonymized text.

<rules>
Evidence and scope
- Use the anonymized text as evidence. The target value supplied in the
  question is a candidate to assess, not evidence that it is recoverable.
- Search the entire text, including narrative passages, structured fields,
  and repeated occurrences.
- Evaluate the target in the semantic role specified by its entity label.
  A matching string used in a clearly different role is not sufficient.
- Respect the subject and relationship described in the text. Do not
  transfer an attribute from one person, object, or relationship to another.

Replacement and redaction
- A placeholder reveals that information was removed and may reveal its
  type. It does not reveal the hidden value.
- A familiar format, pattern, or broad category does not establish the
  missing content needed to recover a specific value.
- A substituted value does not automatically reveal the original value.
- Conversely, a substituted value does not cancel original information
  that remains elsewhere. Consider explicit occurrences and contextual
  clues even when another passage provides a competing value.

Equivalent forms and geographic inference
- Treat normalized, synonymous, abbreviated, or expanded forms as
  equivalent when they preserve the meaning relevant to the requested
  entity label.
- Geographic containment can support inference of a broader location:
  evidence of a narrower place may identify its containing region or
  country. Do not reverse this relationship without additional evidence.

Examples:
- "United States", "United States of America", "U.S.", "USA", "US",
  and "America" when used to mean the country → country = US.
- "western United States" → country = US.
- "California", "Texas", or "New York City" → country = US,
  unless context clearly indicates another meaning.
- "North America" alone does not establish US. Combined with other
  relevant clues, it may make US a well-supported, most likely inference.

- Preserve the requested semantic role: evidence that someone lives in,
  visits, or works in a country does not by itself establish birthplace,
  citizenship, or nationality. Those require evidence connecting the
  location to the attribute being asked about.

Decision
- Answer "yes" when the text contains the target in the relevant role,
  contains a meaning-equivalent form, or provides contextual evidence
  that makes the target a well-supported inference.
- Match the specificity of the target. Evidence for a broad category
  does not automatically establish a particular member or subtype.
- Certainty is not required, but mere compatibility, a generic
  association, or a guess prompted by seeing the target is insufficient.
  Consider whether the evidence meaningfully distinguishes the target
  from plausible alternatives.
- Answer "no" when the text does not provide sufficient evidence to
  recover the specific target value in the requested role.
- Apply the same standard independently to every question.

Answer and confidence
- Return "yes" or "no" for every question.
- Confidence is your confidence that the selected answer is correct.
  It is not the probability that the target is recoverable.
- A well-supported "no" can have high confidence. Do not use zero merely
  because the answer is "no".
- Set confidence from 0.0 to 1.0. Reflect ambiguity and conflicting
  evidence in the confidence and reason.
- Give a brief reason explaining the evidence or why it is insufficient.
- Evidence must contain short, exact quotes copied from the anonymized
  text. Do not paraphrase, reconstruct, or quote the question.
- Use an empty evidence list when there is no relevant passage to quote.
</rules>

<text>
<<ANONYMIZED_TEXT>>
</text>

<task>
Fill in answer, confidence, reason, and evidence for every item.
Preserve all question IDs. Do not add or remove items.
</task>

<answer_template>
<<SKELETON>>
</answer_template>
"""
    return substitute_placeholders(
        prompt,
        {"<<ANONYMIZED_TEXT>>": text, "<<SKELETON>>": json.dumps({"answers": skeleton}, indent=2)},
    )


class CompletePrivacyAnswerSchema(PrivacyAnswerItemSchema):
    """Require explicit answers instead of Rewrite's conservative null defaults."""

    @model_validator(mode="before")
    @classmethod
    def require_complete_fields(cls, value: object) -> object:
        if isinstance(value, dict):
            required = {"id", "answer", "confidence", "reason", "evidence"}
            missing = sorted(key for key in required if value.get(key) is None)
            if missing:
                raise ValueError(f"Privacy answer has missing or null fields: {missing}")
        return value


class CompletePrivacyAnswersSchema(BaseModel):
    """Model response before exact question coverage is validated."""

    answers: list[CompletePrivacyAnswerSchema]


class PrivacyAnswerParams(BaseModel):
    batch_size: int | None = Field(default=5, ge=1, strict=True)


class ReplacePrivacyAnswerWorkflow:
    """Declare privacy answering with a caller-supplied model alias.

    Failure leaves answers unavailable (None), with a diagnostic error column.
    It must not be interpreted as an empty successful evaluation or a leak.
    """

    def columns(self, *, evaluator_alias: str, batch_size: int | None = 5) -> list[ColumnConfigT]:
        return [
            CustomColumnConfig(
                name=COL_PRIVACY_QA_REANSWER,
                generator_function=_make_privacy_answer_column(evaluator_alias),
                storage_encoding="json",
                generator_params=PrivacyAnswerParams(batch_size=batch_size),
            )
        ]


def _make_complete_answer_parser(recipe: PydanticResponseRecipe, expected_ids: list[int]) -> Any:
    def parser(response: str) -> PrivacyAnswersSchema:
        result = recipe.parse(response)
        try:
            return PrivacyAnswersSchema.model_validate(result.model_dump(), context={"expected_ids": expected_ids})
        except ValidationError:
            raise ParserException(
                f"Privacy answers must cover exactly these question IDs once each: {expected_ids}.", source=response
            ) from None

    return parser


def _make_privacy_answer_column(evaluator_alias: str) -> Any:
    @custom_column_generator(
        required_columns=[COL_PRIVACY_QA, COL_REPLACED_TEXT],
        side_effect_columns=[COL_PRIVACY_QA_ERROR],
        model_aliases=[evaluator_alias],
    )
    async def answer(row: dict[str, Any], generator_params: PrivacyAnswerParams | None, models: dict) -> dict[str, Any]:
        qa = parse_privacy_qa(row[COL_PRIVACY_QA])
        row[COL_PRIVACY_QA_ERROR] = None
        if not qa.items:
            row[COL_PRIVACY_QA_REANSWER] = PrivacyAnswersSchema(answers=[]).model_dump(mode="json")
            return row

        recipe = PydanticResponseRecipe(data_type=CompletePrivacyAnswersSchema)
        model = models[evaluator_alias]
        text = row[COL_REPLACED_TEXT]
        if not isinstance(text, str):
            raise ValueError("Replace privacy answering requires replaced text to be a string.")
        try:
            batch_size = (generator_params.batch_size if generator_params else None) or len(qa.items)
            answers: list[PrivacyAnswerItemSchema] = []
            for start in range(0, len(qa.items), batch_size):
                batch = PrivacyQAPairsSchema(items=qa.items[start : start + batch_size])
                parser = _make_complete_answer_parser(recipe, [item.id for item in batch.items])
                result = await answer_privacy_qa(
                    batch,
                    text,
                    model=model,
                    recipe=recipe,
                    parser=parser,
                    prompt=render_replace_privacy_answer_prompt(batch, text),
                )
                answers.extend(result.answers)
            merged = PrivacyAnswersSchema.model_validate(
                {"answers": [answer.model_dump() for answer in answers]},
                context={"expected_ids": [item.id for item in qa.items]},
            )
            by_id = {answer.id: answer for answer in merged.answers}
            result = PrivacyAnswersSchema(answers=[by_id[item.id] for item in qa.items])
        except Exception as exc:
            # This diagnostic judge is non-critical: preserve the record, but never
            # manufacture an answer or expose source text through exception messages.
            logger.warning("Replace privacy QA evaluation unavailable (%s).", type(exc).__name__)
            row[COL_PRIVACY_QA_REANSWER] = None
            row[COL_PRIVACY_QA_ERROR] = "Privacy QA evaluation failed; answers unavailable."
            return row

        row[COL_PRIVACY_QA_REANSWER] = result.model_dump(mode="json")
        return row

    return answer
