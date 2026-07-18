# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Callable
from typing import Any

from data_designer.config.column_configs import GenerationStrategy
from data_designer.engine.column_generators.generators.base import (
    ColumnGeneratorCellByCell,
    ColumnGeneratorWithModelRegistry,
)

from anonymizer.engine.detection.chunked_validation import (
    ChunkedValidationParams,
    chunked_validate_row,
    chunked_validate_row_async,
)
from anonymizer.engine.detection.custom_columns import (
    apply_validation_and_finalize,
    apply_validation_to_seed_entities,
    detect_deterministic_entities_with_labels,
    enrich_validation_decisions,
    finalize_deterministic_entities,
    merge_and_build_candidates,
    parse_detected_entities,
    prepare_validation_inputs,
)
from anonymizer.engine.workflow_columns.detection.config import (
    ChunkedValidationConfig,
    DetectionTransformConfig,
    DetectionTransformOperation,
)

_TRANSFORMS: dict[DetectionTransformOperation, Callable[[dict[str, Any], DetectionTransformConfig], dict[str, Any]]] = {
    DetectionTransformOperation.DETECT_DETERMINISTIC_ENTITIES: (
        lambda data, config: detect_deterministic_entities_with_labels(
            data,
            labels=config.deterministic_labels,
        )
    ),
    DetectionTransformOperation.PARSE_DETECTED_ENTITIES: lambda data, _config: parse_detected_entities(data),
    DetectionTransformOperation.PREPARE_VALIDATION_INPUTS: lambda data, _config: prepare_validation_inputs(data),
    DetectionTransformOperation.ENRICH_VALIDATION_DECISIONS: lambda data, _config: enrich_validation_decisions(data),
    DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES: lambda data, _config: (
        apply_validation_to_seed_entities(data)
    ),
    DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES: lambda data, _config: merge_and_build_candidates(data),
    DetectionTransformOperation.APPLY_VALIDATION_AND_FINALIZE: lambda data, _config: apply_validation_and_finalize(
        data
    ),
    DetectionTransformOperation.FINALIZE_DETERMINISTIC_ENTITIES: lambda data, _config: finalize_deterministic_entities(
        data
    ),
}


_BRIDGE_TIMEOUT_FLOOR_S = 60.0


def _compute_bridge_timeout(
    request_timeout: float,
    max_correction_steps: int,
    max_conversation_restarts: int = 0,
) -> float:
    attempts = (1 + max_conversation_restarts) * (1 + max_correction_steps)
    return max(_BRIDGE_TIMEOUT_FLOOR_S, attempts * request_timeout * 1.5)


class _AsyncBridgedModelFacade:
    def __init__(self, facade: Any) -> None:
        self._facade = facade

    def generate(self, *args: Any, **kwargs: Any) -> tuple[Any, list]:
        from data_designer.engine.models.clients.errors import SyncClientUnavailableError

        try:
            return self._facade.generate(*args, **kwargs)
        except SyncClientUnavailableError:
            pass

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("model.generate() cannot be bridged from the running event loop.")

        from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

        timeout_override = kwargs.get("timeout")
        request_timeout = float(timeout_override) if timeout_override is not None else self._facade.request_timeout
        bridge_timeout = _compute_bridge_timeout(
            request_timeout=request_timeout,
            max_correction_steps=int(kwargs.get("max_correction_steps", 0) or 0),
            max_conversation_restarts=int(kwargs.get("max_conversation_restarts", 0) or 0),
        )
        loop = ensure_async_engine_loop()
        future = asyncio.run_coroutine_threadsafe(self._facade.agenerate(*args, **kwargs), loop)
        try:
            return future.result(timeout=bridge_timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            from data_designer.engine.models.errors import ModelTimeoutError

            raise ModelTimeoutError(f"model.generate() bridge timed out after {bridge_timeout:.0f}s") from exc

    def __getattr__(self, name: str) -> Any:
        return getattr(self._facade, name)


class DetectionTransformGenerator(ColumnGeneratorCellByCell[DetectionTransformConfig]):
    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        operation = DetectionTransformOperation(self.config.operation)
        return _TRANSFORMS[operation](data, self.config)


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

    def generate(self, data: dict[str, Any]) -> dict[str, Any]:
        raw_models = {alias: self.get_model(alias) for alias in self.config.pool}
        models = {alias: _AsyncBridgedModelFacade(model) for alias, model in raw_models.items()}
        return chunked_validate_row(data, self._params(raw_models), models)

    async def agenerate(self, data: dict[str, Any]) -> dict[str, Any]:
        models = {alias: self.get_model(alias) for alias in self.config.pool}
        return await chunked_validate_row_async(data, self._params(models), models)


def _derive_max_parallel_chunks(models: dict[str, Any]) -> int:
    return max(1, sum(max(1, int(getattr(model, "max_parallel_requests", 1) or 1)) for model in models.values()))
