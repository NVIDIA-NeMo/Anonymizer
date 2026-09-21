# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import (
    DEFAULT_PRESERVE_TEXT,
    DEFAULT_PROTECT_TEXT,
    EvaluationCriteria,
    PrivacyGoal,
    RiskTolerance,
)
from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS
from anonymizer.engine.detection.entity_label_examples import normalize_entity_label_examples
from anonymizer.engine.detection.postprocess import normalize_label

logger = logging.getLogger(__name__)


def is_remote_input_source(value: str) -> bool:
    """Return True when the input source is an HTTP(S) URL."""
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"}


def has_unsupported_url_scheme(value: str) -> bool:
    """Return True when the input looks like a URL but uses an unsupported scheme."""
    parsed = urlparse(value)
    return "://" in value and bool(parsed.scheme) and parsed.scheme not in {"http", "https"}


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
        if has_unsupported_url_scheme(value):
            scheme = urlparse(value).scheme
            raise ValueError(f"Unsupported input URL scheme: {scheme!r}. Use http:// or https:// URLs.")
        source = Path(value)
        if not source.exists():
            raise ValueError(f"Input path does not exist: {source}")
        if not source.is_file():
            raise ValueError(f"Input path is not a file: {source}")
        return value


class Detect(BaseModel):
    """Configuration for the entity detection stage."""

    model_config = ConfigDict(hide_input_in_errors=True)

    entity_labels: list[str] | None = Field(
        default=None,
        description=(
            "Labels to detect. None uses the built-in default detection label set. "
            "To inspect the default set, use `from anonymizer import DEFAULT_ENTITY_LABELS`."
        ),
    )
    entity_label_examples: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Positive detection examples keyed by entity label. Built-in labels retain their default "
            "examples and append these values. When entity_labels is None, custom keys are activated "
            "alongside the default labels; explicit entity_labels remain a strict allowlist."
        ),
    )
    excluded_entity_labels: list[str] | None = Field(
        default=None,
        description=(
            "Entity labels to never detect, even if present in entity_labels or the default set. "
            "Excluded labels are removed before GLiNER and LLM prompts run, and are also filtered "
            "from the final entity output as a safety net. If this entirely overlaps the effective "
            "allowlist (entity_labels if set, otherwise the default label set), leaving an empty "
            "effective detection set, Detect raises a ValueError at config time."
        ),
    )
    gliner_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="GLiNER detection confidence threshold (0.0-1.0)."
    )
    validation_max_entities_per_call: int = Field(
        default=100,
        gt=0,
        description=(
            "Maximum number of candidate entities included in a single validator LLM call. "
            "When a row has more candidates than this, validation is split into chunks that "
            "are dispatched (round-robin) across the validator pool."
        ),
    )
    validation_excerpt_window_chars: int = Field(
        default=500,
        gt=0,
        description=(
            "Number of characters to include before and after a chunk's entity span when "
            "building the text excerpt sent to the validator. Bounds the prompt context the "
            "validator sees per chunk; it is NOT the LLM's context window limit."
        ),
    )

    @field_validator("entity_labels")
    @classmethod
    def validate_entity_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [normalize_label(label) for label in value if normalize_label(label)]
        if not cleaned:
            raise ValueError("entity_labels must not be empty. Use None to detect all default labels.")
        deduped = sorted(set(cleaned))
        if len(deduped) != len(cleaned):
            logger.warning("entity_labels contained duplicates, removed automatically.")
        return deduped

    @field_validator("excluded_entity_labels")
    @classmethod
    def validate_excluded_entity_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [normalize_label(label) for label in value if normalize_label(label)]
        if not cleaned:
            raise ValueError("excluded_entity_labels must not be empty. Use None to disable exclusions.")
        deduped = sorted(set(cleaned))
        if len(deduped) != len(cleaned):
            logger.warning("excluded_entity_labels contained duplicates, removed automatically.")
        return deduped

    @field_validator("entity_label_examples", mode="before")
    @classmethod
    def validate_entity_label_examples(cls, value: Any) -> dict[str, list[str]]:
        if not isinstance(value, dict):
            raise ValueError("entity_label_examples must be a dictionary of label names to lists of examples.")

        normalized, duplicate_keys, duplicate_value_labels = normalize_entity_label_examples(value)
        if duplicate_keys:
            logger.warning(
                "entity_label_examples contained keys that normalize to the same label; merged automatically: %s",
                duplicate_keys,
            )
        if duplicate_value_labels:
            logger.warning(
                "entity_label_examples contained duplicate examples; removed automatically for labels: %s",
                duplicate_value_labels,
            )
        return normalized

    @model_validator(mode="after")
    def validate_entity_label_overlap(self) -> "Detect":
        excluded_set = set(self.excluded_entity_labels or [])
        example_labels = set(self.entity_label_examples)
        excluded_examples = sorted(example_labels & excluded_set)
        if excluded_examples:
            logger.warning(
                "entity_label_examples configured excluded labels; their examples will be ignored: %s",
                excluded_examples,
            )

        active_example_labels = example_labels - excluded_set
        if self.entity_labels is not None:
            entity_labels_set = set(self.entity_labels)
            unknown_examples = sorted(active_example_labels - entity_labels_set)
            if unknown_examples:
                raise ValueError(
                    "entity_label_examples contains labels outside the explicit entity_labels allowlist: "
                    f"{unknown_examples}. Add them to entity_labels, remove their examples, or unset "
                    "entity_labels to activate custom example labels alongside the defaults."
                )
            overlap = sorted(entity_labels_set & excluded_set)
            effective_labels = entity_labels_set - excluded_set
            if not effective_labels:
                raise ValueError(
                    "excluded_entity_labels entirely overlaps entity_labels, leaving an empty effective detection set."
                )
            if overlap:
                logger.warning(
                    "entity_labels and excluded_entity_labels share labels that will never be detected: %s",
                    overlap,
                )
        else:
            effective_labels = (set(DEFAULT_ENTITY_LABELS) | active_example_labels) - excluded_set
            if not effective_labels:
                raise ValueError(
                    "excluded_entity_labels entirely overlaps DEFAULT_ENTITY_LABELS and all automatically "
                    "activated custom example labels, leaving an empty effective detection set."
                )
        return self


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
        default=3,
        ge=0,
        description="Maximum repair rounds. Set to 0 to disable repair.",
    )
    use_combined_graph: bool = Field(
        default=False,
        description="Run rewrite and conditional repair iterations in one Data Designer graph.",
    )
    strict_entity_protection: bool = Field(
        default=False,
        description="If True, requires every entity to receive a protective disposition during sensitivity analysis.",
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
        """Construct `EvaluationCriteria` from this `Rewrite` config for the engine.

        `Rewrite` and `EvaluationCriteria` both carry `max_repair_iterations`.
        This property keeps them in sync: it passes through `self.risk_tolerance`
        and `self.max_repair_iterations`. Leakage thresholds and repair
        parameters are derived from `risk_tolerance` via `_RiskToleranceBundle`
        (see `rewrite.py`).

        Production code that starts from a user-facing `Rewrite` should pass
        `rewrite.evaluation` into the engine — never duplicate the mapping
        manually. Tests and engine-internal callers may construct
        `EvaluationCriteria` directly when they aren't routing through a
        user-facing `Rewrite`.
        """
        return EvaluationCriteria(
            risk_tolerance=self.risk_tolerance,
            max_repair_iterations=self.max_repair_iterations,
        )


class AnonymizerConfig(BaseModel):
    """Primary user-facing config for anonymization behavior."""

    model_config = ConfigDict(hide_input_in_errors=True)

    detect: Detect = Field(default_factory=Detect, description="Entity detection configuration.")
    replace: ReplaceMethod | None = Field(
        default=None,
        description="Replacement method (Substitute(), Redact(), Annotate(), or Hash()).",
    )
    rewrite: Rewrite | None = Field(default=None, description="Optional rewrite-mode parameters. ")
    emit_telemetry: bool = Field(
        default=True,
        description=(
            "Whether to emit anonymous Anonymizer telemetry events. See the Telemetry section "
            "in the README for what is collected and how to opt out at the environment or CLI level."
        ),
    )

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


class EvaluateConfig(BaseModel):
    """Optional knobs for :meth:`Anonymizer.evaluate`.

    Reserved for genuinely evaluation-specific configuration — metric selection,
    per-judge model/prompt overrides, scoring thresholds, etc. The anonymization
    mode is **not** here: it travels on the ``AnonymizerResult`` /
    ``PreviewResult`` produced by ``run()`` / ``preview()`` and is read directly
    by ``evaluate()``, so users don't restate it and can't mis-state it.

    Today this is an empty placeholder; fields will be added as evaluation
    knobs are introduced.
    """

    compute_detection_validity: bool = False
    """Run the tag-precision judge (detection_valid / detection_invalid_entities).

    Disabled by default — intended for internal use during model and threshold
    experiments. When True, adds
    ``detection_valid`` and ``detection_invalid_entities`` columns to the
    evaluate() output alongside ``entity_coverage``.
    """
