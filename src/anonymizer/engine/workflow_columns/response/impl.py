# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
from typing import Any, cast

from data_designer.config.utils.constants import REASONING_CONTENT_COLUMN_POSTFIX, TRACE_COLUMN_POSTFIX
from data_designer.config.utils.storage_schema import compile_json_storage_schema
from data_designer.config.utils.trace_type import TraceType
from data_designer.engine.column_generators.generators.llm_completion import ColumnGeneratorWithModelChatCompletion
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.recipes.response_recipes import StructuredResponseRecipe
from data_designer.engine.processing.utils import deserialize_json_values

from anonymizer.engine.ndd.model_response import (
    ResponseCheckError,
    ask_with_trace,
    log_unavailable,
    no_response,
    row_key,
)
from anonymizer.engine.workflow_columns.response.config import ResponseColumnConfig

logger = logging.getLogger("anonymizer.ndd.response")


class ResponseColumnGenerator(ColumnGeneratorWithModelChatCompletion[ResponseColumnConfig]):
    """DataDesigner's structured generator, except that a response that never parses keeps the row."""

    @functools.cached_property
    def response_recipe(self) -> StructuredResponseRecipe:  # ty: ignore[invalid-mutable-override]
        # DataDesigner picks the recipe by column type; this is the one it gives ``llm-structured``. The config's
        # validator has already turned a Pydantic ``output_format`` into its JSON schema.
        schema = cast(dict[str, Any], self.config.output_format)
        return StructuredResponseRecipe(json_schema=compile_json_storage_schema(schema)[0])

    def generate(self, data: dict) -> dict:
        raise NotImplementedError("ResponseColumnGenerator is async-only; use agenerate().")

    async def agenerate(self, data: dict) -> dict:
        missing = [column for column in self.config.ask_only_if if no_response(data.get(column))]
        if missing:
            logger.debug("%s not asked: %s has no value.", self.config.name, ", ".join(missing))
            return self._unavailable(data)
        kwargs = self._prepare_generation_kwargs(data)
        parse = kwargs.pop("parser")
        check = self.config.response_check
        check_cause: str | None = None  # the last response's check failure, when the check can name it

        def checked(response: str) -> tuple[Any, bool]:
            nonlocal check_cause
            check_cause = None
            value = parse(response)
            if check is not None:
                try:
                    check(value, data)
                except ValueError as exc:
                    check_cause = exc.cause if isinstance(exc, ResponseCheckError) else None
                    raise ParserException(str(exc), source=response) from None
            return value, False

        response, trace = await ask_with_trace(self.model, checked, step=self.config.name, key=row_key(data), **kwargs)
        if response.state == "unavailable":
            log_unavailable(logger, self.config.name, response, cause=check_cause)
            return self._unavailable(data)
        data = self._process_generation_result(data, response.value, trace)
        if self.config.unavailable_marker:
            data[self.config.unavailable_marker] = False
        return data

    def _unavailable(self, data: dict[str, Any]) -> dict[str, Any]:
        data[self.config.name] = None if self.config.unavailable is None else dict(self.config.unavailable)
        if self.config.unavailable_marker:
            data[self.config.unavailable_marker] = True
        if self.config.with_trace != TraceType.NONE:
            data[self.config.name + TRACE_COLUMN_POSTFIX] = []
        if self.config.extract_reasoning_content:
            data[self.config.name + REASONING_CONTENT_COLUMN_POSTFIX] = None
        return data

    def _process_serialized_output(self, serialized_output: str) -> dict | list:
        return deserialize_json_values(serialized_output)
