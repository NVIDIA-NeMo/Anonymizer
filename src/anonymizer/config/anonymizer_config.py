# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import re
from enum import Enum
from pathlib import Path
from string import Formatter
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

from anonymizer.config.models import ModelSelection
from anonymizer.config.params import EvaluationCriteria, RewriteParams


class EntityReplaceStrategy(Protocol):
    """Protocol for local deterministic replace strategies."""

    def replace(self, text: str, label: str) -> str:
        """Return a replacement value for one entity."""


class WorkflowReplaceStrategy(Protocol):
    """Protocol for workflow-backed replace strategy markers."""

    kind: str


class PrivacyGoal(BaseModel):
    """Structured privacy and utility goal for rewrite mode."""

    protect: str = Field(min_length=10, max_length=1000)
    preserve: str = Field(min_length=10, max_length=1000)

    @field_validator("protect", "preserve")
    @classmethod
    def validate_min_words(cls, value: str) -> str:
        cleaned = value.strip()
        if len(cleaned.split()) < 3:
            raise ValueError("privacy goal sections must contain at least 3 words")
        return cleaned

    def to_prompt_string(self) -> str:
        """Serialize goal into prompt-ready text."""
        return f"PROTECT: {self.protect}\nPRESERVE: {self.preserve}"


class LabelReplace(BaseModel):
    """Replace each entity with a readable label token."""

    kind: Literal["label"] = "label"
    format_template: str = "<{text}, {label}>"

    @field_validator("format_template")
    @classmethod
    def validate_format_template(cls, value: str) -> str:
        _validate_template(
            value=value,
            required_fields={"text", "label"},
            allowed_fields={"text", "label"},
        )
        return value

    def replace(self, text: str, label: str) -> str:
        return _render_template(
            template=self.format_template,
            text=text,
            label=label,
        )


class RedactReplace(BaseModel):
    """Replace each entity with a configurable redaction template."""

    kind: Literal["redact"] = "redact"
    redact_template: str = "[REDACTED_{label}]"
    normalize_label: bool = True

    @field_validator("redact_template")
    @classmethod
    def validate_redact_template(cls, value: str) -> str:
        _validate_template(
            value=value,
            required_fields=set(),
            allowed_fields={"text", "label"},
        )
        return value

    def replace(self, text: str, label: str) -> str:
        normalized_label = _format_label_for_redaction(label) if self.normalize_label else label
        return _render_template(
            template=self.redact_template,
            text=text,
            label=normalized_label,
        )


class HashReplace(BaseModel):
    """Replace each entity with a deterministic hash token."""

    kind: Literal["hash"] = "hash"
    algorithm: Literal["sha256", "sha1", "md5"] = "sha256"
    digest_length: int = Field(default=12, ge=6, le=64)

    def replace(self, text: str, label: str) -> str:
        digest = hashlib.new(self.algorithm, text.encode("utf-8")).hexdigest()
        return f"<HASH_{label.upper()}_{digest[: self.digest_length]}>"


class LLMReplace(BaseModel):
    """Marker config for LLM-backed replacement workflow execution."""

    kind: Literal["llm"] = "llm"
    model_alias: str = Field(default="text")
    workflow_id: str = Field(default="replace")
    instructions: str | None = None


ReplaceStrategy = Annotated[
    LabelReplace | RedactReplace | HashReplace | LLMReplace,
    Field(discriminator="kind"),
]

LocalReplaceStrategy = Annotated[
    LabelReplace | RedactReplace | HashReplace,
    Field(discriminator="kind"),
]


DEFAULT_PROTECT_TEXT = (
    "Direct identifiers, quasi-identifier combinations, and latent inferences "
    "that could enable re-identification"
)
DEFAULT_PRESERVE_TEXT = "General utility, content quality, and semantic meaning of the original text"


class InputSourceType(str, Enum):
    """Supported input source types."""

    parquet = "parquet"
    csv = "csv"
    json = "json"
    dataframe = "dataframe"


class AnonymizerInput(BaseModel):
    """Input source definition for the anonymizer pipeline."""

    source: str | Path
    source_type: InputSourceType = InputSourceType.parquet
    text_column: str = Field(default="text", min_length=1)
    id_column: str | None = None


class AnonymizerConfig(BaseModel):
    """Primary user-facing config for anonymization behavior."""

    # Basics required for every dataset/workflow
    entity_labels: list[str] | None = None
    locale: str = Field(default="en_US", min_length=2)
    data_summary: str | None = None
    selected_models: ModelSelection = Field(default_factory=ModelSelection)
    detection_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

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
    def validate_rewrite_goal_consistency(self) -> "AnonymizerConfig":
        if self.rewrite is None and self.privacy_goal is not None:
            raise ValueError("privacy_goal is only valid when rewrite params are provided")
        if self.rewrite is not None and self.privacy_goal is None:
            self.privacy_goal = PrivacyGoal(
                protect=DEFAULT_PROTECT_TEXT,
                preserve=DEFAULT_PRESERVE_TEXT,
            )
        return self


def _format_label_for_redaction(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_")
    if cleaned == "":
        return "UNKNOWN"
    return cleaned.upper()


def _validate_template(
    value: str,
    required_fields: set[str],
    allowed_fields: set[str] | None = None,
) -> None:
    fields = {field_name for _, field_name, _, _ in Formatter().parse(value) if field_name}
    allowed = allowed_fields or {"text", "label"}
    unexpected = fields - allowed
    if unexpected:
        invalid = ", ".join(sorted(unexpected))
        raise ValueError(f"template contains unsupported placeholders: {invalid}")
    if not required_fields.issubset(fields):
        missing = ", ".join(sorted(required_fields - fields))
        raise ValueError(f"template is missing required placeholders: {missing}")


def _render_template(template: str, *, text: str, label: str) -> str:
    return template.format(text=text, label=label)
