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

from anonymizer.engine.detection.augmentation import augment_row, augment_row_async
from anonymizer.engine.detection.chunked_validation import (
    ChunkedValidationParams,
    chunked_validate_row_async,
)
from anonymizer.engine.detection.custom_columns import (
    admit_validation_decisions,
    apply_validation_to_seed_entities,
    finalize_entities,
    merge_and_build_candidates,
    parse_detected_entities,
    prepare_detector_source,
    prepare_validation_inputs,
)
from anonymizer.engine.workflow_columns.detection.config import (
    AugmentationConfig,
    ChunkedValidationConfig,
    DetectionTransformConfig,
    DetectionTransformOperation,
)

_TRANSFORMS: dict[DetectionTransformOperation, Callable[[dict[str, Any]], dict[str, Any]]] = {
    DetectionTransformOperation.PREPARE_DETECTOR_SOURCE: prepare_detector_source,
    DetectionTransformOperation.PREPARE_VALIDATION_INPUTS: prepare_validation_inputs,
    DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES: apply_validation_to_seed_entities,
    DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES: merge_and_build_candidates,
    DetectionTransformOperation.FINALIZE_ENTITIES: finalize_entities,
}


class DetectionTransformGenerator(ColumnGeneratorCellByCell[DetectionTransformConfig]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        operation = DetectionTransformOperation(self.config.operation)
        if operation == DetectionTransformOperation.PARSE_DETECTED_ENTITIES:
            return parse_detected_entities(
                data,
                detection_result_id=self.config.detection_result_id,
                source_instance=self.config.source_instance,
            )
        if operation == DetectionTransformOperation.ADMIT_VALIDATION_DECISIONS:
            return admit_validation_decisions(
                data,
                allowed_entity_labels=self.config.allowed_entity_labels,
                excluded_entity_labels=self.config.excluded_entity_labels,
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
            max_conversation_restarts=self.resource_provider.run_config.max_conversation_restarts,
            max_correction_steps=self.resource_provider.run_config.max_conversation_correction_steps,
            single_chunk_full_text=self.config.single_chunk_full_text,
            prompt_template=self.config.prompt_template,
            system_prompt=self.config.system_prompt,
            allowed_entity_labels=self.config.allowed_entity_labels,
            excluded_entity_labels=self.config.excluded_entity_labels,
        )

    def generate(self, data: dict[str, Any]) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        raise NotImplementedError("ChunkedValidationGenerator is async-only; use agenerate().")

    async def agenerate(self, data: dict[str, Any]) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        models = {alias: self.get_model(alias) for alias in self.config.pool}
        return await chunked_validate_row_async(data, self._params(models), models)


class AugmentationGenerator(ColumnGeneratorWithModelRegistry[AugmentationConfig]):
    """Generate and ground augmenter output before publishing the column."""

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict[str, Any]) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        model = _AsyncBridgedModelFacade(self.get_model(self.config.model_alias))
        return augment_row(
            row=data,
            prompt_template=self.config.prompt_template,
            model_alias=self.config.model_alias,
            facade=model,
            allowed_entity_labels=self.config.allowed_entity_labels,
            excluded_entity_labels=self.config.excluded_entity_labels,
            system_prompt=self.config.system_prompt,
        )

    async def agenerate(self, data: dict[str, Any]) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        return await augment_row_async(
            row=data,
            prompt_template=self.config.prompt_template,
            model_alias=self.config.model_alias,
            facade=self.get_model(self.config.model_alias),
            allowed_entity_labels=self.config.allowed_entity_labels,
            excluded_entity_labels=self.config.excluded_entity_labels,
            system_prompt=self.config.system_prompt,
        )


def _derive_max_parallel_chunks(models: dict[str, Any]) -> int:
    return max(1, sum(max(1, int(getattr(model, "max_parallel_requests", 1) or 1)) for model in models.values()))
