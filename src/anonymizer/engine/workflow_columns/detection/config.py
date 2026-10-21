# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import ClassVar, Literal

from data_designer.config.base import SingleColumnConfig
from pydantic import Field

from anonymizer.engine.constants import (
    COL_ADMISSION_RESULTS,
    COL_ADMITTED_CANDIDATES,
    COL_AUGMENTATION_ADMISSION_RESULTS,
    COL_AUGMENTATION_ATTEMPTS,
    COL_AUGMENTATION_DETECTION_RESULT,
    COL_AUGMENTED_CANDIDATES,
    COL_INITIAL_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_RECONCILED_CANDIDATES,
    COL_SEED_DETECTION_RESULT,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATION_CANDIDATES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_INVENTORIES,
    RECORD_ID_COLUMN,
    SEED_DETECTION_RESULT_ID,
)


class AugmentationConfig(SingleColumnConfig):
    """Model-backed occurrence-specific augmentation column."""

    column_type: Literal["anonymizer-augmentation"] = "anonymizer-augmentation"
    model_alias: str = Field(min_length=1)
    prompt_template: str = Field(repr=False)
    system_prompt: str | None = Field(default=None, repr=False)
    allowed_entity_labels: list[str] | None = None
    excluded_entity_labels: list[str] = Field(default_factory=list)

    @staticmethod
    def get_column_emoji() -> str:
        return "A"

    @property
    def required_columns(self) -> list[str]:
        return [
            RECORD_ID_COLUMN,
            COL_TEXT,
            COL_INITIAL_TAGGED_TEXT,
            COL_SEED_ENTITIES_JSON,
            COL_TAG_NOTATION,
        ]

    @property
    def side_effect_columns(self) -> list[str]:
        return [
            COL_AUGMENTATION_DETECTION_RESULT,
            COL_AUGMENTATION_ADMISSION_RESULTS,
            COL_AUGMENTATION_ATTEMPTS,
            COL_AUGMENTED_CANDIDATES,
        ]

    def get_model_aliases(self) -> list[str]:
        return [self.model_alias]


class DetectionTransformOperation(str, Enum):
    PARSE_DETECTED_ENTITIES = "parse_detected_entities"
    PREPARE_VALIDATION_INPUTS = "prepare_validation_inputs"
    ADMIT_VALIDATION_DECISIONS = "admit_validation_decisions"
    APPLY_VALIDATION_TO_SEED_ENTITIES = "apply_validation_to_seed_entities"
    MERGE_AND_BUILD_CANDIDATES = "merge_and_build_candidates"
    FINALIZE_ENTITIES = "finalize_entities"


class DetectionTransformConfig(SingleColumnConfig):
    column_type: Literal["anonymizer-detection-transform"] = "anonymizer-detection-transform"
    operation: DetectionTransformOperation
    excluded_entity_labels: list[str] = Field(default_factory=list)
    allowed_entity_labels: list[str] | None = None
    detection_result_id: str = SEED_DETECTION_RESULT_ID
    source_instance: str = "gliner"

    _REQUIRED_COLUMNS: ClassVar[dict[DetectionTransformOperation, list[str]]] = {
        DetectionTransformOperation.PARSE_DETECTED_ENTITIES: [
            RECORD_ID_COLUMN,
            COL_TEXT,
            COL_RAW_DETECTED,
        ],
        DetectionTransformOperation.PREPARE_VALIDATION_INPUTS: [COL_TEXT, COL_SEED_ENTITIES],
        DetectionTransformOperation.ADMIT_VALIDATION_DECISIONS: [
            COL_SEED_DETECTION_RESULT,
            COL_VALIDATION_DECISIONS,
            COL_VALIDATION_INVENTORIES,
        ],
        DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES: [
            COL_TEXT,
            COL_SEED_DETECTION_RESULT,
            COL_ADMISSION_RESULTS,
        ],
        DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES: [
            COL_TEXT,
            COL_ADMITTED_CANDIDATES,
            COL_AUGMENTED_CANDIDATES,
        ],
        DetectionTransformOperation.FINALIZE_ENTITIES: [
            COL_TEXT,
            COL_RECONCILED_CANDIDATES,
        ],
    }
    _SIDE_EFFECT_COLUMNS: ClassVar[dict[DetectionTransformOperation, list[str]]] = {
        DetectionTransformOperation.PARSE_DETECTED_ENTITIES: [
            COL_SEED_DETECTION_RESULT,
            COL_TAG_NOTATION,
        ],
        DetectionTransformOperation.PREPARE_VALIDATION_INPUTS: [],
        DetectionTransformOperation.ADMIT_VALIDATION_DECISIONS: [],
        DetectionTransformOperation.APPLY_VALIDATION_TO_SEED_ENTITIES: [
            COL_ADMITTED_CANDIDATES,
            COL_INITIAL_TAGGED_TEXT,
            COL_SEED_ENTITIES_JSON,
        ],
        DetectionTransformOperation.MERGE_AND_BUILD_CANDIDATES: [
            COL_RECONCILED_CANDIDATES,
            COL_VALIDATION_CANDIDATES,
        ],
        DetectionTransformOperation.FINALIZE_ENTITIES: [COL_TAGGED_TEXT],
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
    allowed_entity_labels: list[str] | None = None
    excluded_entity_labels: list[str] = Field(default_factory=list)

    @staticmethod
    def get_column_emoji() -> str:
        return "A"

    @property
    def required_columns(self) -> list[str]:
        return [
            COL_TEXT,
            COL_SEED_DETECTION_RESULT,
            COL_TAG_NOTATION,
        ]

    @property
    def side_effect_columns(self) -> list[str]:
        return [COL_VALIDATION_INVENTORIES]

    def get_model_aliases(self) -> list[str]:
        return list(self.pool)
