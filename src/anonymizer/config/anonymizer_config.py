# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

from anonymizer.config.replace_strategies import ReplaceStrategy
from anonymizer.config.rewrite import (
    DEFAULT_PRESERVE_TEXT,
    DEFAULT_PROTECT_TEXT,
    EvaluationCriteria,
    PrivacyGoal,
    RewriteParams,
)


class AnonymizerInput(BaseModel):
    """Input source definition for the anonymizer pipeline.

    Format is inferred from file extension (.csv or .parquet).
    """

    source: str
    text_column: str = Field(default="text", min_length=1)
    id_column: str | None = None
    data_summary: str | None = None


class AnonymizerConfig(BaseModel):
    """Primary user-facing config for anonymization behavior."""

    # Basics required for every dataset/workflow
    entity_labels: list[str] | None = None
    locale: str = Field(default="en_US", min_length=2)
    gliner_detection_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Replace configuration
    replace: ReplaceStrategy

    # Rewrite/evaluation configuration
    rewrite: RewriteParams | None = None
    privacy_goal: PrivacyGoal | None = None
    evaluation: EvaluationCriteria = Field(default_factory=EvaluationCriteria)

    @field_validator("entity_labels")
    @classmethod
    def validate_entity_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [label.strip() for label in value if label.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("entity_labels must not contain duplicates")
        return cleaned

    @model_validator(mode="after")
    def validate_rewrite_goal_consistency(self) -> AnonymizerConfig:
        if self.rewrite is None and self.privacy_goal is not None:
            raise ValueError("privacy_goal is only valid when rewrite params are provided")
        if self.rewrite is not None and self.privacy_goal is None:
            self.privacy_goal = PrivacyGoal(
                protect=DEFAULT_PROTECT_TEXT,
                preserve=DEFAULT_PRESERVE_TEXT,
            )
        return self
