# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate privacy questions for every assessed Replace entity without an LLM."""

from __future__ import annotations

from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.engine.constants import COL_ENTITY_SENSITIVITY, COL_PRIVACY_QA
from anonymizer.engine.evaluation.privacy_qa import build_privacy_question
from anonymizer.engine.evaluation.replace.entity_sensitivity import EntitySensitivitiesSchema
from anonymizer.engine.rewrite.parsers import normalize_payload
from anonymizer.engine.schemas import PrivacyQAPairsSchema


def generate_privacy_qa_from_sensitivities(assessments: EntitySensitivitiesSchema) -> PrivacyQAPairsSchema:
    """Create one question per original value/label pair, including low sensitivity.

    IDs follow assessment order, starting at one. Original values, labels,
    categories, and sensitivity levels pass through unchanged.
    """
    return PrivacyQAPairsSchema(
        items=[
            build_privacy_question(
                question_id=idx,
                entity_label=entity.entity_label,
                entity_value=entity.entity_value,
                sensitivity=entity.sensitivity,
                category=entity.category,
                text_kind="replaced",
            )
            for idx, entity in enumerate(assessments.entities, start=1)
        ]
    )


class ReplacePrivacyQAGenerationWorkflow:
    """Declare the deterministic QA column; execution remains behind NddAdapter."""

    def columns(self) -> list[ColumnConfigT]:
        return [
            CustomColumnConfig(
                name=COL_PRIVACY_QA, generator_function=_generate_privacy_qa_column, storage_encoding="json"
            )
        ]


@custom_column_generator(required_columns=[COL_ENTITY_SENSITIVITY])
def _generate_privacy_qa_column(row: dict[str, Any]) -> dict[str, Any]:
    assessments = EntitySensitivitiesSchema.model_validate(normalize_payload(row[COL_ENTITY_SENSITIVITY]))
    row[COL_PRIVACY_QA] = generate_privacy_qa_from_sensitivities(assessments).model_dump(mode="json")
    return row
