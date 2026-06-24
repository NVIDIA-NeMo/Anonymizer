# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import Field

from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_TAGGED_TEXT,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATED_SEED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
)


class DetectionTransformOperation(str, Enum):
    PARSE_DETECTED_ENTITIES = "parse_detected_entities"
    PREPARE_VALIDATION_INPUTS = "prepare_validation_inputs"
    ENRICH_VALIDATION_DECISIONS = "enrich_validation_decisions"
    APPLY_VALIDATION_TO_SEED_ENTITIES = "apply_validation_to_seed_entities"
    MERGE_AND_BUILD_CANDIDATES = "merge_and_build_candidates"
    APPLY_VALIDATION_AND_FINALIZE = "apply_validation_and_finalize"


class DetectionTransformConfig(SingleColumnConfig):
    column_type: Literal["anonymizer-detection-transform"] = "anonymizer-detection-transform"
    operation: DetectionTransformOperation

    _REQUIRED_COLUMNS: ClassVar[dict[DetectionTransformOperation, list[str]]] = {
        DetectionTransformOperation.PARSE_DETECTED_ENTITIES: [COL_TEXT, COL_RAW_DETECTED],
        DetectionTransformOperation.PREPARE_VALIDATION_INPUTS: [COL_TEXT, COL_SEED_ENTITIES],
        DetectionTransformOperation.ENRICH_VALIDATION_DECISIONS: [
            COL_VALIDATION_DECISIONS,
            COL_SEED_VALIDATION_CANDIDATES,
        ],
        DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES: [
            COL_TEXT,
            COL_SEED_ENTITIES,
            COL_VALIDATED_ENTITIES,
        ],
        DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES: [
            COL_TEXT,
            COL_VALIDATED_SEED_ENTITIES,
            COL_AUGMENTED_ENTITIES,
        ],
        DetectionTransformOperation.APPLY_VALIDATION_AND_FINALIZE: [
            COL_TEXT,
            COL_MERGED_ENTITIES,
            COL_VALIDATED_ENTITIES,
        ],
    }
    _SIDE_EFFECT_COLUMNS: ClassVar[dict[DetectionTransformOperation, list[str]]] = {
        DetectionTransformOperation.PARSE_DETECTED_ENTITIES: [COL_TAG_NOTATION],
        DetectionTransformOperation.PREPARE_VALIDATION_INPUTS: [COL_SEED_TAGGED_TEXT],
        DetectionTransformOperation.ENRICH_VALIDATION_DECISIONS: [],
        DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES: [
            COL_INITIAL_TAGGED_TEXT,
            COL_SEED_ENTITIES_JSON,
            COL_VALIDATED_SEED_ENTITIES,
        ],
        DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES: [
            COL_MERGED_TAGGED_TEXT,
            COL_VALIDATION_CANDIDATES,
        ],
        DetectionTransformOperation.APPLY_VALIDATION_AND_FINALIZE: [COL_TAGGED_TEXT],
    }

    @staticmethod
    def get_column_emoji() -> str:
        return "A"

    @property
    def required_columns(self) -> list[str]:
        return self._REQUIRED_COLUMNS[DetectionTransformOperation(self.operation)]

    @property
    def side_effect_columns(self) -> list[str]:
        return self._SIDE_EFFECT_COLUMNS[DetectionTransformOperation(self.operation)]


class ChunkedValidationConfig(SingleColumnConfig):
    column_type: Literal["anonymizer-chunked-validation"] = "anonymizer-chunked-validation"
    pool: list[str] = Field(min_length=1)
    max_entities_per_call: int = Field(gt=0)
    excerpt_window_chars: int = Field(gt=0)
    max_parallel_chunks: int | None = Field(default=None, gt=0)
    single_chunk_full_text: bool = True
    prompt_template: str = Field(repr=False)
    system_prompt: str | None = Field(default=None, repr=False)

    @staticmethod
    def get_column_emoji() -> str:
        return "A"

    @property
    def required_columns(self) -> list[str]:
        return [
            COL_TEXT,
            COL_SEED_ENTITIES,
            COL_SEED_VALIDATION_CANDIDATES,
            COL_TAG_NOTATION,
        ]

    @property
    def side_effect_columns(self) -> list[str]:
        return []

    def get_model_aliases(self) -> list[str]:
        return list(self.pool)
