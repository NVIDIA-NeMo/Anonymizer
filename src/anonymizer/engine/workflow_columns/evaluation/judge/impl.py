# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import cast

from data_designer.engine.column_generators.generators.llm_completion import ColumnGeneratorWithModelChatCompletion
from data_designer.engine.models.parsers.errors import ParserException
from data_designer.engine.models.parsers.parser import LLMResponseParser
from data_designer.engine.models.parsers.postprocessors import (
    StructuredDataBlock,
    deserialize_json_code,
    merge_text_blocks,
)
from data_designer.engine.models.recipes.response_recipes import PydanticResponseRecipe
from data_designer.engine.processing.utils import deserialize_json_values
from pydantic import BaseModel

from anonymizer.engine.evaluation.detection_judge import DetectionJudgmentSchema
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import AttributeFidelityJudgmentSchema
from anonymizer.engine.evaluation.replace.relational_consistency_judge import RelationalConsistencyJudgmentSchema
from anonymizer.engine.evaluation.replace.type_fidelity_judge import TypeFidelityJudgmentSchema
from anonymizer.engine.workflow_columns.evaluation.judge.config import JudgeColumnConfig, JudgeKind

_JUDGE_SCHEMAS: dict[JudgeKind, type[BaseModel]] = {
    JudgeKind.DETECTION: DetectionJudgmentSchema,
    JudgeKind.TYPE_FIDELITY: TypeFidelityJudgmentSchema,
    JudgeKind.ATTRIBUTE_FIDELITY: AttributeFidelityJudgmentSchema,
    JudgeKind.RELATIONAL_CONSISTENCY: RelationalConsistencyJudgmentSchema,
}


class _StrictJudgeResponseRecipe(PydanticResponseRecipe):
    def _build_parser_fn(self) -> Callable[[str], BaseModel]:
        parser = LLMResponseParser(postprocessors=[merge_text_blocks, deserialize_json_code])

        def parse_response(response: str) -> BaseModel:
            try:
                block = cast(StructuredDataBlock, parser.parse(response).filter([StructuredDataBlock]).parsed.pop())
                decoded = block.obj
                return self.data_type.model_validate(decoded, strict=True)
            except IndexError:
                raise ParserException(
                    "No parsable JSON structure within ```json markdown fence.",
                    source=response,
                ) from None
            except Exception as exc:
                raise ParserException(
                    "Response doesn't match requested <response_schema>\n" + str(exc),
                    source=response,
                ) from None

        return parse_response


class JudgeColumnGenerator(ColumnGeneratorWithModelChatCompletion[JudgeColumnConfig]):
    @functools.cached_property
    def response_recipe(self) -> _StrictJudgeResponseRecipe:
        schema = _JUDGE_SCHEMAS[JudgeKind(self.config.judge_kind)]
        return _StrictJudgeResponseRecipe(schema)

    def _process_serialized_output(self, serialized_output: str) -> dict | list:
        return deserialize_json_values(serialized_output)
