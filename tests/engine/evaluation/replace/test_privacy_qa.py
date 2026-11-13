# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
from data_designer.config.column_configs import CustomColumnConfig
from pydantic import ValidationError

from anonymizer.engine.constants import COL_ENTITY_SENSITIVITY, COL_PRIVACY_QA
from anonymizer.engine.evaluation.replace.entity_sensitivity import EntitySensitivitiesSchema
from anonymizer.engine.evaluation.replace.privacy_qa import (
    ReplacePrivacyQAGenerationWorkflow,
    _generate_privacy_qa_column,
    generate_privacy_qa_from_sensitivities,
)
from anonymizer.engine.schemas import PrivacyQAPairsSchema


def _assessments() -> EntitySensitivitiesSchema:
    return EntitySensitivitiesSchema.model_validate(
        {
            "entities": [
                {
                    "entity_value": "Zoë O'Neil",
                    "entity_label": label,
                    "category": "direct_identifier",
                    "sensitivity": "high",
                    "reason": "The original name identifies the individual.",
                }
                for label in ["full_name", "user_name"]
            ]
            + [
                {
                    "entity_value": "USA",
                    "entity_label": "country",
                    "category": "quasi_identifier",
                    "sensitivity": "low",
                    "reason": "The country is a broad location shared by many people.",
                },
                {
                    "entity_value": "Denver",
                    "entity_label": "city",
                    "category": "quasi_identifier",
                    "sensitivity": "medium",
                    "reason": "The city meaningfully narrows identity in this context.",
                },
            ]
        }
    )


def test_questions_cover_all_assessments_and_preserve_targets() -> None:
    assessments = _assessments()
    before = assessments.model_dump()
    qa = generate_privacy_qa_from_sensitivities(assessments)
    assert [item.id for item in qa.items] == [1, 2, 3, 4]
    assert len(qa.items) == len(assessments.entities)
    for item, assessment in zip(qa.items, assessments.entities, strict=True):
        assert item.entity_value == assessment.entity_value
        assert item.entity_label == assessment.entity_label
        assert item.category == assessment.category
        assert item.sensitivity == assessment.sensitivity
        assert item.question == (
            f"Can the {assessment.entity_label} '{assessment.entity_value}' be deduced from the replaced text?"
        )
    assert qa.items[2].sensitivity == "low"
    assert assessments.model_dump() == before


@pytest.mark.parametrize("representation", ["schema", "dict", "json", "array"])
def test_column_accepts_supported_assessment_payloads(representation: str) -> None:
    assessments = _assessments()
    payload: object = assessments
    if representation == "dict":
        payload = assessments.model_dump(mode="json")
    elif representation == "json":
        payload = assessments.model_dump_json()
    elif representation == "array":
        payload = {"entities": np.array(assessments.model_dump(mode="json")["entities"], dtype=object)}
    row = {COL_ENTITY_SENSITIVITY: payload}
    result = _generate_privacy_qa_column(row)
    actual = PrivacyQAPairsSchema.model_validate(result[COL_PRIVACY_QA])
    assert actual == generate_privacy_qa_from_sensitivities(assessments)
    assert result[COL_ENTITY_SENSITIVITY] is payload


def test_empty_assessments_produce_empty_questions() -> None:
    result = _generate_privacy_qa_column({COL_ENTITY_SENSITIVITY: {"entities": []}})
    assert result[COL_PRIVACY_QA] == {"items": []}


@pytest.mark.parametrize("payload", ["not JSON", {}, {"entities": [{}]}])
def test_column_does_not_silently_treat_invalid_assessments_as_empty(payload: object) -> None:
    with pytest.raises(ValidationError):
        _generate_privacy_qa_column({COL_ENTITY_SENSITIVITY: payload})


def test_workflow_declares_only_deterministic_qa_column() -> None:
    columns = ReplacePrivacyQAGenerationWorkflow().columns()
    assert len(columns) == 1
    column = columns[0]
    assert isinstance(column, CustomColumnConfig)
    assert column.name == COL_PRIVACY_QA
    assert column.required_columns == [COL_ENTITY_SENSITIVITY]
    result = column.generator_function({COL_ENTITY_SENSITIVITY: _assessments().model_dump(mode="json")})
    assert result[COL_PRIVACY_QA] == generate_privacy_qa_from_sensitivities(_assessments()).model_dump(mode="json")
