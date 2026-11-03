# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)

from anonymizer.engine.detection.chunked_validation import (
    ChunkedValidationParams,
    chunked_validate_row_async,
)
from anonymizer.engine.detection.custom_columns import (
    apply_validation_and_finalize,
    apply_validation_to_seed_entities,
    enrich_validation_decisions,
    merge_and_build_candidates,
    parse_detected_entities,
    prepare_validation_inputs,
)
from anonymizer.engine.workflow_columns.detection.config import (
    ChunkedValidationConfig,
    DetectionTransformConfig,
    DetectionTransformOperation,
)

_TRANSFORMS: dict[DetectionTransformOperation, Callable[[dict[str, Any]], dict[str, Any]]] = {
    DetectionTransformOperation.PARSE_DETECTED_ENTITIES: parse_detected_entities,
    DetectionTransformOperation.PREPARE_VALIDATION_INPUTS: prepare_validation_inputs,
    DetectionTransformOperation.ENRICH_VALIDATION_DECISIONS: enrich_validation_decisions,
    DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES: apply_validation_to_seed_entities,
    DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES: merge_and_build_candidates,
    DetectionTransformOperation.APPLY_VALIDATION_AND_FINALIZE: apply_validation_and_finalize,
}


class DetectionTransformGenerator(ColumnGeneratorCellByCell[DetectionTransformConfig]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        operation = DetectionTransformOperation(self.config.operation)
        if operation == DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES:
            return apply_validation_to_seed_entities(
                data,
                excluded_entity_labels=self.config.excluded_entity_labels,
            )
        if operation == DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES:
            return merge_and_build_candidates(
                data,
                excluded_entity_labels=self.config.excluded_entity_labels,
            )
        if operation == DetectionTransformOperation.APPLY_VALIDATION_AND_FINALIZE:
            return apply_validation_and_finalize(
                data,
                excluded_entity_labels=self.config.excluded_entity_labels,
                allowed_entity_labels=self.config.allowed_entity_labels,
            )
        return _TRANSFORMS[operation](data)


class ChunkedValidationGenerator(ColumnGeneratorWithModelRegistry[ChunkedValidationConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def _params(self, models: dict[str, Any]) -> ChunkedValidationParams:
        return ChunkedValidationParams(
            pool=list(self.config.pool),
            max_entities_per_call=self.config.max_entities_per_call,
            excerpt_window_chars=self.config.excerpt_window_chars,
            max_parallel_chunks=self.config.max_parallel_chunks or _derive_max_parallel_chunks(models),
            single_chunk_full_text=self.config.single_chunk_full_text,
            prompt_template=self.config.prompt_template,
            system_prompt=self.config.system_prompt,
        )

    def generate(self, data: dict[str, Any]) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        raise NotImplementedError("ChunkedValidationGenerator is async-only; use agenerate().")

    async def agenerate(self, data: dict[str, Any]) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        models = {alias: self.get_model(alias) for alias in self.config.pool}
        return await chunked_validate_row_async(data, self._params(models), models)


def _derive_max_parallel_chunks(models: dict[str, Any]) -> int:
    return max(1, sum(max(1, int(getattr(model, "max_parallel_requests", 1) or 1)) for model in models.values()))
