# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cached_property
from typing import cast

from data_designer.engine.column_generators.generators.llm_completion import LLMStructuredCellGenerator
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe
from data_designer.engine.processing.gsonschema.validators import JSONSchemaValidationError, validate

from anonymizer.engine.workflow_columns.structured.config import TolerantStructuredColumnConfig


class TolerantStructuredResponseRecipe(StructuredResponseRecipe):
    """Preserve schema validation while admitting provider-native bare JSON."""

    def _build_parser_fn(self) -> Callable[[str], dict]:
        fenced_parser = super()._build_parser_fn()

        def parse(response: str) -> dict:
            try:
                return fenced_parser(response)
            except ParserException as fenced_error:
                try:
                    return validate(json.loads(response), **self._validate_args)
                except (JSONSchemaValidationError, TypeError, ValueError):
                    raise fenced_error from None

        return parse


class TolerantStructuredCellGenerator(LLMStructuredCellGenerator):
    """Run a structured LLM column with the tolerant response recipe."""

    config: TolerantStructuredColumnConfig

    @cached_property
    def response_recipe(self) -> TolerantStructuredResponseRecipe:
        return TolerantStructuredResponseRecipe(json_schema=cast(dict, self.config.output_format))
