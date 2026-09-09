# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry

from anonymizer.engine.workflow_columns.structured.config import TolerantStructuredColumnConfig
from anonymizer.engine.workflow_columns.structured.impl import (
    TolerantStructuredCellGenerator,
    TolerantStructuredResponseRecipe,
)

_SCHEMA = {
    "type": "object",
    "properties": {"entities": {"type": "array", "items": {"type": "string"}}},
    "required": ["entities"],
    "additionalProperties": False,
}


@pytest.mark.parametrize(
    "response",
    [
        '{"entities": ["Alice"]}',
        '```json\n{"entities": ["Alice"]}\n```',
    ],
)
def test_tolerant_structured_recipe_accepts_bare_and_fenced_json(response: str) -> None:
    recipe = TolerantStructuredResponseRecipe(json_schema=_SCHEMA)

    assert recipe.parse(response) == {"entities": ["Alice"]}


def test_tolerant_structured_recipe_preserves_schema_validation() -> None:
    recipe = TolerantStructuredResponseRecipe(json_schema=_SCHEMA)

    with pytest.raises(ParserException):
        recipe.parse('{"wrong": []}')


def test_tolerant_structured_config_has_plugin_discriminator() -> None:
    config = TolerantStructuredColumnConfig(
        name="entities",
        prompt="Find entities",
        model_alias="nemotron-super",
        output_format=_SCHEMA,
    )

    assert config.column_type == "anonymizer-tolerant-structured"


def test_tolerant_structured_plugin_is_registered() -> None:
    generator = DataDesignerRegistry().column_generators.get_for_config_type(TolerantStructuredColumnConfig)

    assert generator is TolerantStructuredCellGenerator
