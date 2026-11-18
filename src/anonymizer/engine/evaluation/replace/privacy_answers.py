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
    prompt = """You are a privacy auditor. For each question, determine whether the
specified original entity value can still be identified or reasonably
inferred from the anonymized text.

The target value in the question is a candidate to assess, not evidence
that it is recoverable.

<rules>
Search and evidence
- Before answering each question, search the ENTIRE text for evidence.
- Check narrative passages, structured fields, tool calls, tool results,
  and repeated occurrences.
- Do not stop at a redacted occurrence or a weak clue if stronger
  evidence exists elsewhere. Base your answer on the strongest evidence.
- Evaluate the target in the semantic role specified by its entity label.
  A matching string used in a clearly different role is not sufficient.
- Respect the subject and relationship described in the text. Do not
  transfer an attribute from one person, object, or relationship to another.

Decision order
1. Direct recovery:
   If the target value appears anywhere in the relevant semantic role,
   answer "yes". One surviving occurrence is sufficient, even if every
   other occurrence has been redacted or substituted.

2. Equivalent forms:
   If a normalized, synonymous, abbreviated, or expanded form identifies
   the target in the relevant role, answer "yes".

3. Contextual inference:
   If the value is not explicitly present, consider whether the strongest
   contextual evidence makes it a reasonable, best-supported inference.
   Ask: "Using only this text, what value would I infer for this entity?"
   Do not require certainty or an exact spelling match.
   Mere compatibility, a generic association, or a guess prompted only
   by seeing the target in the question is insufficient.

4. No recovery:
   Answer "no" only after checking direct occurrences, equivalent forms,
   and contextual evidence throughout the text and finding insufficient
   support for the target.

Redaction and substitution
- You are assessing whether the original value remains recoverable
  anywhere in the text, not whether a particular redacted span can
  be filled in.
- A placeholder elsewhere NEVER cancels surviving evidence.
- Do not answer "no" merely because a relevant field contains a placeholder
  or substitute. Continue searching the rest of the text.
- A placeholder alone reveals that information was removed and may reveal
  its type. It does not reveal the hidden value.
- A familiar format or broad category alone does not establish a specific
  hidden value.
- A substituted value does not automatically reveal the original value.
  It also does not invalidate original information that survives elsewhere.

Equivalent forms and geography
- Treat normalized, synonymous, abbreviated, and expanded forms as the
  same value when context supports the intended meaning.
- Geographic inference is directional: a city can identify its containing
  state or country. A state or country generally cannot identify a
  particular city.
- For example, an address containing Chicago, USA supports Illinois;
  Illinois alone does not identify Chicago. A redacted or substituted
  state field does not override this evidence.
- Apply this relationship consistently to all locations, not only the
  examples listed.

Examples:
- "United States", "United States of America", "U.S.", "USA", "US",
  and "America" when used to mean the country → country = US.
- "western United States" → country = US.
- "NYC" in a city or address context → city = New York City / New York,
  state = New York / NY, and country = US.
- "LA" in a context that identifies Los Angeles → city = Los Angeles,
  state = California / CA, and country = US.
- "California", "Texas", or "New York City" → country = US,
  unless context clearly indicates otherwise.
- "North America" alone does not establish US. Surrounding evidence may
  make US a reasonable, best-supported inference.

- Preserve the requested relationship: living in, visiting, or working
  in a country does not by itself establish birthplace, citizenship,
  or nationality.
- Match the specificity of the target. A broad category does not
  automatically identify a particular member or subtype.

Answer and confidence
- Apply these rules independently to every question.
- Return "yes" or "no" for every question.
- Confidence is your confidence that the selected answer is correct,
  not the probability that the target is recoverable.
- Set confidence from 0.0 to 1.0. A well-supported "no" may have high
  confidence; do not use zero merely because the answer is "no".
- Give a brief reason grounded in the strongest available evidence.
- For "yes", quote the surviving occurrence or contextual evidence.
- For "no", do not claim the value is absent if it appears in the text.
  If a matching string has a clearly different role, explain that distinction.
- Evidence must contain short, exact quotes copied from the anonymized
  text. Do not paraphrase, reconstruct, or quote the question.
- Use an empty evidence list when no relevant passage is available.
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
