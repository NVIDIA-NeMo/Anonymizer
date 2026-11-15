# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Answer Replace privacy questions without converting evaluation failures to leaks."""

from __future__ import annotations

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
from anonymizer.engine.rewrite.parsers import parse_privacy_qa
from anonymizer.engine.schemas import PrivacyAnswerItemSchema, PrivacyAnswersSchema, PrivacyQAPairsSchema

logger = logging.getLogger("anonymizer.evaluation.replace.privacy_answers")


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
                result = await answer_privacy_qa(batch, text, model=model, recipe=recipe, parser=parser)
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
