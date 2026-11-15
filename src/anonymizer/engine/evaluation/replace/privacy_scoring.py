# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic leakage mass and entity contributions for Replace evaluation."""

from __future__ import annotations

import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT
from pydantic import BaseModel, Field

from anonymizer.config.rewrite import SENSITIVITY_WEIGHTS
from anonymizer.engine.constants import (
    COL_LEAKAGE_MASS,
    COL_PRIVACY_LEAKAGE_CONTRIBUTIONS,
    COL_PRIVACY_QA,
    COL_PRIVACY_QA_ERROR,
    COL_PRIVACY_QA_REANSWER,
    COL_WEIGHTED_LEAKAGE_RATE,
)
from anonymizer.engine.evaluation.replace.privacy_answers import CompletePrivacyAnswersSchema
from anonymizer.engine.rewrite.evaluate import compute_leakage_mass, compute_weighted_leakage_rate
from anonymizer.engine.rewrite.parsers import normalize_payload, parse_privacy_qa
from anonymizer.engine.schemas import (
    EntityCategory,
    PrivacyAnswer,
    PrivacyAnswerItemSchema,
    PrivacyAnswersSchema,
    PrivacyQAPairsSchema,
    SensitivityLevel,
)

logger = logging.getLogger("anonymizer.evaluation.replace.privacy_scoring")


class PrivacyLeakageContributionSchema(PrivacyAnswerItemSchema):
    """One evaluated entity, including zero contributions for negative answers."""

    entity_label: str
    entity_value: str
    category: EntityCategory
    sensitivity: SensitivityLevel
    leakage_contribution: float = Field(ge=0.0)


class PrivacyLeakageScoreSchema(BaseModel):
    """Record mass and its complete entity-level breakdown."""

    leakage_mass: float = Field(ge=0.0)
    weighted_leakage_rate: float = Field(ge=0.0, le=1.0)
    entities: list[PrivacyLeakageContributionSchema]


def compute_privacy_leakage_score(
    privacy_qa: PrivacyQAPairsSchema, privacy_answers: PrivacyAnswersSchema
) -> PrivacyLeakageScoreSchema:
    """Score every target, joining answers by ID rather than response order.

    Requires complete, explicit answers. Low-sensitivity targets count with the
    same weights as Rewrite. Missing or invalid answers raise rather than imply
    zero leakage. An empty question set with empty answers has zero mass.
    """
    expected_ids = [item.id for item in privacy_qa.items]
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Privacy questions must have unique IDs for leakage scoring.")
    complete = CompletePrivacyAnswersSchema.model_validate(privacy_answers.model_dump())
    answers = PrivacyAnswersSchema.model_validate(complete.model_dump(), context={"expected_ids": expected_ids})
    by_id = {answer.id: answer for answer in answers.answers}
    entities = []
    for item in privacy_qa.items:
        answer = by_id[item.id]
        contribution = (
            SENSITIVITY_WEIGHTS[item.sensitivity] * answer.confidence if answer.answer == PrivacyAnswer.yes else 0.0
        )
        entities.append(
            PrivacyLeakageContributionSchema(
                **answer.model_dump(),
                entity_label=item.entity_label,
                entity_value=item.entity_value,
                category=item.category,
                sensitivity=item.sensitivity,
                leakage_contribution=contribution,
            )
        )
    mass = compute_leakage_mass(answers.answers, privacy_qa, dict(SENSITIVITY_WEIGHTS))
    return PrivacyLeakageScoreSchema(
        leakage_mass=mass,
        weighted_leakage_rate=compute_weighted_leakage_rate(mass, privacy_qa, dict(SENSITIVITY_WEIGHTS)),
        entities=entities,
    )


class ReplacePrivacyScoringWorkflow:
    """Declare leakage mass, normalized rate, and entity contributions."""

    def columns(self) -> list[ColumnConfigT]:
        return [CustomColumnConfig(name=COL_LEAKAGE_MASS, generator_function=_compute_privacy_score_column)]


@custom_column_generator(
    required_columns=[COL_PRIVACY_QA, COL_PRIVACY_QA_REANSWER, COL_PRIVACY_QA_ERROR],
    side_effect_columns=[COL_PRIVACY_LEAKAGE_CONTRIBUTIONS, COL_WEIGHTED_LEAKAGE_RATE],
)
def _compute_privacy_score_column(row: dict[str, Any]) -> dict[str, Any]:
    # Reset outputs so re-evaluating an unavailable record cannot retain old scores.
    row[COL_LEAKAGE_MASS] = None
    row[COL_WEIGHTED_LEAKAGE_RATE] = None
    row[COL_PRIVACY_LEAKAGE_CONTRIBUTIONS] = None
    if row.get(COL_PRIVACY_QA_ERROR):
        return row
    try:
        qa = parse_privacy_qa(row[COL_PRIVACY_QA])
        complete = CompletePrivacyAnswersSchema.model_validate(normalize_payload(row[COL_PRIVACY_QA_REANSWER]))
        answers = PrivacyAnswersSchema.model_validate(complete.model_dump())
        score = compute_privacy_leakage_score(qa, answers)
    except (ValueError, TypeError):
        logger.warning("Privacy QA answers or questions are unavailable or invalid; leakage score unavailable.")
        return row
    row[COL_LEAKAGE_MASS] = score.leakage_mass
    row[COL_WEIGHTED_LEAKAGE_RATE] = score.weighted_leakage_rate
    row[COL_PRIVACY_LEAKAGE_CONTRIBUTIONS] = [entity.model_dump(mode="json") for entity in score.entities]
    return row
