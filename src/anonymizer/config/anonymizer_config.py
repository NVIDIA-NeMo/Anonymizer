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

try:
    from data_designer.engine.processing.ginja.environment import MAX_RENDERED_LEN as _NDD_MAX_RENDERED_LEN
except Exception:  # pragma: no cover - fall back if NDD internals move
    _NDD_MAX_RENDERED_LEN = 512_000

# Default per-call render cap for the windowed long-context stages. Kept well below
# NDD's hard render cap so each window stays small enough to map/rewrite within a
# single LLM request — large windows on entity-dense documents otherwise time out.
# Clamped so it never exceeds NDD's cap if that is ever lowered.
_DEFAULT_WINDOW_MAX_RENDER_CHARS = min(128_000, _NDD_MAX_RENDERED_LEN)

# Floor on the per-window character budget; mirrors ``_MIN_WINDOW_CHARS`` in the
# engine's chunked-detection planners. Defined here (rather than imported) to keep
# the user-facing config free of an engine import cycle. Used only to compute the
# effective window size for overlap validation.
_MIN_DETECTION_WINDOW_CHARS = 4_000


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
    detection_window_max_render_chars: int = Field(
        default=_DEFAULT_WINDOW_MAX_RENDER_CHARS,
        gt=0,
        description=(
            "Upper bound on a single rendered prompt (characters) for the windowed "
            "augmentation, latent, substitute-map, and rewrite stages. Documents whose "
            "rendered prompt would exceed this are processed in windows. Defaults to 128 KiB "
            "(131072), kept below NDD's MAX_RENDERED_LEN render cap so each window maps or "
            "rewrites within a single LLM request without timing out on long documents."
        ),
    )
    detection_window_safety_margin_chars: int = Field(
        default=8_000,
        ge=0,
        description=(
            "Headroom subtracted from detection_window_max_render_chars to leave room for "
            "prompt scaffolding and tags when sizing augmentation/latent windows."
        ),
    )
    detection_window_overlap_chars: int = Field(
        default=1_000,
        ge=0,
        description=(
            "Overlap between adjacent augmentation/latent windows so an entity straddling a "
            "window boundary is fully visible in at least one window. Must be smaller than the "
            "effective window size (max_render_chars - safety_margin_chars, floored at 4000); a "
            "larger overlap stalls the planners to one character per step and is rejected at config time."
        ),
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

    @model_validator(mode="after")
    def validate_window_overlap(self) -> Detect:
        """Reject overlaps that would stall the windowed detection planners.

        The augmentation/latent planners advance by ``window - overlap`` characters
        per step. The effective window mirrors the engine's sizing:
        ``max(_MIN_DETECTION_WINDOW_CHARS, max_render - safety_margin)``. When the
        overlap meets or exceeds that, the stride collapses to a single character
        and one long row explodes into tens of thousands of model calls (a 20k-char
        row became 16,001 windows in testing), so require it to be strictly smaller.
        """
        effective_window = max(
            _MIN_DETECTION_WINDOW_CHARS,
            self.detection_window_max_render_chars - self.detection_window_safety_margin_chars,
        )
        if self.detection_window_overlap_chars >= effective_window:
            raise ValueError(
                f"detection_window_overlap_chars ({self.detection_window_overlap_chars}) must be smaller than "
                f"the effective window size ({effective_window} chars = max({_MIN_DETECTION_WINDOW_CHARS}, "
                "detection_window_max_render_chars - detection_window_safety_margin_chars)). A larger overlap "
                "makes the windowed planners advance one character at a time, exploding a single row into "
                "thousands of model calls."
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

    # Intentionally empty for now. New fields land here as evaluation
    # configurability is introduced.
