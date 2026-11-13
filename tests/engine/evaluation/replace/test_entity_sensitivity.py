# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from data_designer.config.column_configs import CustomColumnConfig
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from pydantic import ValidationError

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITY_CLASSIFICATION,
    COL_ENTITY_SENSITIVITY,
    COL_REPLACED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.evaluation.replace.entity_sensitivity import (
    EntitySensitivitiesSchema,
    IndexedEntitySensitivitiesSchema,
    ReplaceEntitySensitivityWorkflow,
    SensitivityPromptParams,
    _make_sensitivity_column,
    _make_sensitivity_parser,
    _original_entity_pairs,
    _render_sensitivity_prompt,
)
from anonymizer.engine.evaluation.sensitivity_rules import (
    DIRECT_IDENTIFIER_SENSITIVITY_RULES,
    QUASI_IDENTIFIER_SENSITIVITY_RULES,
    SENSITIVITY_CONTEXT_RULES,
    SENSITIVITY_RISK_SCOPE,
)
from anonymizer.engine.rewrite.sensitivity_disposition import _get_sensitivity_disposition_prompt


def _assessment(value: str = "USA", label: str = "country") -> dict[str, str]:
    return {
        "entity_value": value,
        "entity_label": label,
        "category": "quasi_identifier",
        "sensitivity": "low",
        "reason": "A broad country designation does not meaningfully narrow identity in this document.",
    }


def test_parser_keeps_low_sensitivity_and_requires_exact_coverage() -> None:
    recipe = PydanticResponseRecipe(data_type=IndexedEntitySensitivitiesSchema)
    parser = _make_sensitivity_parser(recipe, [("USA", "country")])
    assessment = {"id": 1, "sensitivity": "low", "reason": "Broad country designation."}
    result = parser(json.dumps({"entities": [assessment]}))
    assert result.entities[0].sensitivity == "low"
    assert result.entities[0].entity_value == "USA"
    for entities in (
        [],
        [assessment, assessment],
        [{**assessment, "id": 2}],
        [{**assessment, "entity_value": "Canada"}],
    ):
        with pytest.raises(ParserException):
            parser(json.dumps({"entities": entities}))


def test_parser_restores_exact_original_text_and_labels_despite_reordered_responses() -> None:
    pairs = [("Zoë\u202fO’Neil", "full_name"), ("Zoë\u202fO’Neil", "user_name"), ("40‑year‑old", "age")]
    recipe = PydanticResponseRecipe(data_type=IndexedEntitySensitivitiesSchema)
    parser = _make_sensitivity_parser(
        recipe, pairs, {"full_name": "direct_identifier", "user_name": "direct_identifier", "age": "quasi_identifier"}
    )
    result = parser(
        json.dumps(
            {
                "entities": [
                    {
                        "id": idx,
                        "sensitivity": "low" if idx == 3 else "high",
                        "reason": f"Reason for ID {idx}",
                    }
                    for idx in [3, 1, 2]
                ]
            }
        )
    )
    assert [(entity.entity_value, entity.entity_label) for entity in result.entities] == pairs
    assert [entity.reason for entity in result.entities] == ["Reason for ID 1", "Reason for ID 2", "Reason for ID 3"]


@pytest.mark.parametrize("item_id", [0, -1, True, "1", 1.5])
def test_response_schema_rejects_invalid_ids(item_id: object) -> None:
    recipe = PydanticResponseRecipe(data_type=IndexedEntitySensitivitiesSchema)
    parser = _make_sensitivity_parser(recipe, [("USA", "country")])
    with pytest.raises(ParserException):
        parser(
            json.dumps(
                {
                    "entities": [
                        {
                            "id": item_id,
                            "sensitivity": "low",
                            "reason": "Broad country designation.",
                        }
                    ]
                }
            )
        )


def test_known_names_cannot_be_downgraded_to_common_quasi_identifiers() -> None:
    recipe = PydanticResponseRecipe(data_type=IndexedEntitySensitivitiesSchema)
    parser = _make_sensitivity_parser(recipe, [("Alice", "first_name")])
    with pytest.raises(ParserException, match="Direct identifiers must have high sensitivity"):
        parser(json.dumps({"entities": [{"id": 1, "sensitivity": "low", "reason": "A common name."}]}))
    result = parser(
        json.dumps(
            {
                "entities": [
                    {
                        "id": 1,
                        "sensitivity": "high",
                        "reason": "A supplied direct identifier.",
                    }
                ]
            }
        )
    )
    assert result.entities[0].category == "direct_identifier"
    assert result.entities[0].sensitivity == "high"


@pytest.mark.parametrize(
    "changes",
    [
        {"category": "latent_identifier"},
        {"category": "direct_identifier", "sensitivity": "medium"},
        {"reason": ""},
        {"protection_method_suggestion": "leave_as_is"},
    ],
)
def test_schema_rejects_invalid_or_out_of_scope_assessments(changes: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        EntitySensitivitiesSchema.model_validate({"entities": [{**_assessment(), **changes}]})


def test_pairs_preserve_multiple_labels_and_deduplicate_exact_targets() -> None:
    raw = {
        "entities_by_value": [
            {"value": "Alice", "labels": ["first_name", "full_name"]},
            {"value": "Alice", "labels": ["first_name"]},
        ]
    }
    assert _original_entity_pairs(raw) == [("Alice", "first_name"), ("Alice", "full_name")]


def test_prompt_uses_original_context_and_shared_rules() -> None:
    row = {
        COL_ENTITY_CLASSIFICATION: {"country": "quasi_identifier"},
        COL_TEXT: "Alice lives in USA.",
        COL_REPLACED_TEXT: "SECRET REPLACED TEXT",
        COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "USA", "labels": ["country"]}]},
    }
    prompt = _render_sensitivity_prompt(row, SensitivityPromptParams(data_summary="Biographies"))
    assert "Alice lives in USA." in prompt
    assert "SECRET REPLACED TEXT" not in prompt
    assert "Biographies" in prompt
    assert '"id": 1' in prompt
    assert "ONLY id, sensitivity, and reason" in prompt
    assert "<domain_context>" not in prompt
    assert "<privacy_goal>" not in prompt
    assert "<input_latent_entities>" not in prompt
    assert "Assign medium unless retaining the information adds no meaningful linkage" in prompt
    assert "Generic quasi-identifiers:" in prompt
    assert "A planned edit is not a reason to rate it low." in prompt
    assert "explicit sensitivity" in prompt
    assert '"category": "quasi_identifier"' in prompt
    goal = PrivacyGoal(protect="Protect identity of subjects", preserve="Preserve useful factual content")
    rewrite_prompt = _get_sensitivity_disposition_prompt(goal)
    for rules in [
        DIRECT_IDENTIFIER_SENSITIVITY_RULES,
        QUASI_IDENTIFIER_SENSITIVITY_RULES,
        SENSITIVITY_CONTEXT_RULES,
        SENSITIVITY_RISK_SCOPE,
    ]:
        assert rules in prompt
        assert rules in rewrite_prompt


def test_columns_reuse_existing_model_roles(stub_rewrite_model_selection: RewriteModelSelection) -> None:
    columns = ReplaceEntitySensitivityWorkflow().columns(selected_models=stub_rewrite_model_selection)
    assert len(columns) == 1
    sensitivity = columns[-1]
    assert isinstance(sensitivity, CustomColumnConfig)
    assert sensitivity.name == COL_ENTITY_SENSITIVITY
    assert COL_REPLACED_TEXT not in sensitivity.required_columns
    assert set(sensitivity.required_columns) == {COL_TEXT, COL_ENTITIES_BY_VALUE, COL_ENTITY_CLASSIFICATION}


def test_sensitivity_generator_calls_model_with_coverage_parser() -> None:
    model = Mock()
    model.agenerate = AsyncMock(return_value=(EntitySensitivitiesSchema(entities=[_assessment()]), None))
    row = {
        COL_ENTITY_CLASSIFICATION: {"country": "quasi_identifier"},
        COL_TEXT: "Resident of USA.",
        COL_ENTITIES_BY_VALUE: {"entities_by_value": [{"value": "USA", "labels": ["country"]}]},
    }
    generator = _make_sensitivity_column("assessor")
    result = asyncio.run(generator(row, SensitivityPromptParams(), {"assessor": model}))
    assert result[COL_ENTITY_SENSITIVITY]["entities"][0]["sensitivity"] == "low"
    parser = model.agenerate.call_args.kwargs["parser"]
    with pytest.raises(ParserException):
        parser('{"entities": []}')


def test_empty_entities_skip_sensitivity_model() -> None:
    generator = _make_sensitivity_column("assessor")
    result = asyncio.run(generator({COL_ENTITIES_BY_VALUE: {"entities_by_value": []}}, SensitivityPromptParams(), {}))
    assert result[COL_ENTITY_SENSITIVITY] == {"entities": []}
