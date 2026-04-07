# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import (
    DEFAULT_PRESERVE_TEXT,
    DEFAULT_PROTECT_TEXT,
    EvaluationCriteria,
    PrivacyGoal,
    RiskTolerance,
)

logger = logging.getLogger(__name__)


def is_remote_input_source(value: str) -> bool:
    """Return True when the input source is an HTTP(S) URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def infer_input_source_suffix(value: str) -> str:
    """Infer the lowercase file suffix from a local path or remote URL path."""
    if is_remote_input_source(value):
        return Path(urlparse(value).path).suffix.lower()
    return Path(value).suffix.lower()


class AnonymizerInput(BaseModel):
    """Input source definition for the anonymizer pipeline.

    Format is inferred from the file extension of a local path or HTTP(S) URL.
    """

    source: str = Field(description="Local path or HTTP(S) URL for a .csv or .parquet input file.")
    text_column: str = Field(default="text", min_length=1, description="Column containing the text to anonymize.")
    id_column: str | None = Field(default=None, description="Optional column to use as record identifier.")
    data_summary: str | None = Field(
        default=None, description="Short description of the data. Improves LLM detection accuracy."
    )

    @field_validator("source")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        if is_remote_input_source(value):
            return value
        source = Path(value)
        if not source.exists():
            raise ValueError(f"Input path does not exist: {source}")
        if not source.is_file():
            raise ValueError(f"Input path is not a file: {source}")
        return value


class Detect(BaseModel):
    """Configuration for the entity detection stage."""

    entity_labels: list[str] | None = Field(
        default=None,
        description=(
            "Labels to detect. None uses the built-in default detection label set. "
            "To inspect the default set, use `from anonymizer import DEFAULT_ENTITY_LABELS`."
        ),
    )
    gliner_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="GLiNER detection confidence threshold (0.0-1.0)."
    )

    @field_validator("entity_labels")
    @classmethod
    def validate_entity_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [label.strip().lower() for label in value if label.strip()]
        if not cleaned:
            raise ValueError("entity_labels must not be empty. Use None to detect all default labels.")
        deduped = sorted(set(cleaned))
        if len(deduped) != len(cleaned):
            logger.warning("entity_labels contained duplicates, removed automatically.")
        return deduped


class Rewrite(BaseModel):
    """Configuration for rewrite-mode execution."""

    privacy_goal: PrivacyGoal | None = Field(
        default=None, description="Structured privacy goal. Auto-populated with defaults if not provided."
    )
    instructions: str | None = Field(default=None, description="Additional instructions for the rewrite LLM.")
    risk_tolerance: RiskTolerance = Field(
        default=RiskTolerance.low,
        description="Preset controlling repair thresholds and review flagging.",
    )
    max_repair_iterations: int = Field(
        default=2,
        ge=0,
        description="Maximum repair rounds. Set to 0 to disable repair.",
    )

    @model_validator(mode="after")
    def populate_default_privacy_goal(self) -> Rewrite:
        if self.privacy_goal is None:
            self.privacy_goal = PrivacyGoal(
                protect=DEFAULT_PROTECT_TEXT,
                preserve=DEFAULT_PRESERVE_TEXT,
            )
        return self

    @property
    def evaluation(self) -> EvaluationCriteria:
        """Internal: construct EvaluationCriteria for the engine."""
        return EvaluationCriteria(
            risk_tolerance=self.risk_tolerance,
            max_repair_iterations=self.max_repair_iterations,
        )


class AnonymizerConfig(BaseModel):
    """Primary user-facing config for anonymization behavior."""

    detect: Detect = Field(default_factory=Detect, description="Entity detection configuration.")
    replace: ReplaceMethod | None = Field(
        default=None,
        description="Replacement method (Substitute(), Redact(), Annotate(), or Hash()).",
    )
    rewrite: Rewrite | None = Field(default=None, description="Optional rewrite-mode parameters. ")

    @model_validator(mode="after")
    def validate_exactly_one_mode(self) -> AnonymizerConfig:
        if self.replace is None and self.rewrite is None:
            raise ValueError(
                "Exactly one of replace or rewrite must be provided."
                " Use replace=Redact() for entity replacement, or rewrite=Rewrite() for LLM rewriting."
            )
        if self.replace is not None and self.rewrite is not None:
            raise ValueError(
                "Cannot use both replace and rewrite — choose one mode."
                " Use replace=Redact() for entity replacement, or rewrite=Rewrite() for LLM rewriting."
            )
        return self
