# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator, model_validator

from anonymizer.config.regex import BuiltinRegex, RegexRule
from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import (
    DEFAULT_PRESERVE_TEXT,
    DEFAULT_PROTECT_TEXT,
    EvaluationCriteria,
    PrivacyGoal,
    RiskTolerance,
)
from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS

logger = logging.getLogger(__name__)


def resolve_effective_detection_labels(
    entity_labels: list[str] | None,
    *,
    regex_rules: list[BuiltinRegex | RegexRule] | None = None,
    excluded_entity_labels: list[str] | set[str] | None = None,
) -> list[str]:
    """Resolve the labels used by detection after custom rules and exclusions.

    An explicit ``entity_labels`` list is authoritative. When it is ``None``,
    detection starts with ``DEFAULT_ENTITY_LABELS`` and appends labels from
    enabled custom ``RegexRule`` entries. Only enabled custom rules extend the
    effective label set: disabled custom rules are inert, and ``BuiltinRegex``
    settings configure existing built-ins without adding labels. Exclusions
    are applied last using normalized label comparisons.

    This helper intentionally returns only the resolved labels. Callers must
    retain the original ``entity_labels is None`` state because it separately
    controls permissive versus strict augmentation.
    """
    labels = list(DEFAULT_ENTITY_LABELS) if entity_labels is None else list(entity_labels)
    if entity_labels is None:
        known = {label.strip().casefold() for label in labels}
        for rule in regex_rules or []:
            if not isinstance(rule, RegexRule) or not rule.enabled:
                continue
            normalized = rule.label.strip().casefold()
            if normalized not in known:
                labels.append(rule.label)
                known.add(normalized)

    excluded = {label.strip().casefold() for label in excluded_entity_labels or []}
    return [label for label in labels if label.strip().casefold() not in excluded]


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
    excluded_entity_labels: list[str] | None = Field(
        default=None,
        description=(
            "Entity labels to never detect, even if present in entity_labels or the default set. "
            "Excluded labels are removed before GLiNER and LLM prompts run, and are also filtered "
            "from the final entity output as a safety net. If this entirely overlaps the effective "
            "allowlist (entity_labels if set, otherwise the default label set plus labels from enabled "
            "custom regex rules), leaving an empty effective detection set, Detect raises a ValueError "
            "at config time."
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
    builtin_regexes: bool = Field(
        default=True,
        description="Run built-in regex recognizers for labels in the effective detection scope.",
    )
    regex_rules: list[BuiltinRegex | RegexRule] = Field(
        default_factory=list,
        description="Per-label built-in settings and user-defined regex candidate rules.",
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
    def validate_regex_rule_scope(self) -> Detect:
        enabled_custom_rules = [rule for rule in self.regex_rules if isinstance(rule, RegexRule) and rule.enabled]
        builtin_rules = [rule for rule in self.regex_rules if isinstance(rule, BuiltinRegex)]
        identities = [(rule.label, rule.pattern) for rule in enabled_custom_rules]
        if len(set(identities)) != len(identities):
            raise ValueError("regex_rules contains duplicate label and pattern pairs.")
        builtin_labels = [rule.label for rule in builtin_rules]
        if len(set(builtin_labels)) != len(builtin_labels):
            raise ValueError("regex_rules contains duplicate built-in labels.")
        if self.entity_labels is not None:
            missing = sorted({rule.label for rule in enabled_custom_rules} - set(self.entity_labels))
            if missing:
                raise ValueError(f"Regex rule labels {missing!r} are missing from explicit entity_labels.")
        return self

    @field_validator("excluded_entity_labels")
    @classmethod
    def validate_excluded_entity_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [label.strip().lower() for label in value if label.strip()]
        if not cleaned:
            raise ValueError("excluded_entity_labels must not be empty. Use None to disable exclusions.")
        deduped = sorted(set(cleaned))
        if len(deduped) != len(cleaned):
            logger.warning("excluded_entity_labels contained duplicates, removed automatically.")
        return deduped

    @model_validator(mode="after")
    def validate_entity_label_overlap(self) -> "Detect":
        if self.excluded_entity_labels is None:
            return self
        excluded_set = set(self.excluded_entity_labels)
        effective_labels = resolve_effective_detection_labels(
            self.entity_labels,
            regex_rules=self.regex_rules,
            excluded_entity_labels=excluded_set,
        )

        if self.entity_labels is not None:
            entity_labels_set = set(self.entity_labels)
            overlap = sorted(entity_labels_set & excluded_set)
            if not overlap:
                return self
            if not effective_labels:
                raise ValueError(
                    "excluded_entity_labels entirely overlaps entity_labels, leaving an empty "
                    f"effective detection set. Overlapping labels: {overlap}. Remove these labels from "
                    "excluded_entity_labels, add other labels to entity_labels, or unset entity_labels "
                    "(use None) to fall back to the default detection set — note excluded_entity_labels "
                    "still applies against it."
                )
            logger.warning(
                "entity_labels and excluded_entity_labels share labels that will never be detected: %s",
                overlap,
            )
            return self

        # entity_labels=None combines defaults with enabled custom regex labels.
        if not effective_labels:
            raise ValueError(
                "excluded_entity_labels entirely overlaps DEFAULT_ENTITY_LABELS and all enabled custom "
                "regex labels, leaving an empty effective detection set. Set entity_labels explicitly to "
                "a non-empty subset of labels you still want detected, add an enabled custom regex rule "
                "with a non-excluded label, or remove some labels from excluded_entity_labels."
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
