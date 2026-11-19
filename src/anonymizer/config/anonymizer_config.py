# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from anonymizer.config.entity_labels import normalize_entity_label
from anonymizer.config.regex import BuiltinRegex, RegexLimits, RegexRule
from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.config.rewrite import (
    EvaluationCriteria,
    PrivacyGoal,
    RiskTolerance,
    default_privacy_goal,
)
from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS
from anonymizer.engine.detection.entity_label_examples import (
    ResolvedEntityLabelConfig,
    normalize_entity_label_examples,
    resolve_entity_label_config,
)

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
        known = {normalize_entity_label(label) for label in labels}
        for rule in regex_rules or []:
            if not isinstance(rule, RegexRule) or not rule.enabled:
                continue
            normalized = normalize_entity_label(rule.label)
            if normalized not in known:
                labels.append(rule.label)
                known.add(normalized)

    excluded = {normalize_entity_label(label) for label in excluded_entity_labels or []}
    return [label for label in labels if normalize_entity_label(label) not in excluded]


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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

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


class TextRecord(BaseModel):
    """One in-memory text record to anonymize."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    id: str = Field(min_length=1, description="Caller-defined record identifier.")
    text: str = Field(description="Text to anonymize.")


class TextRecordsInput(BaseModel):
    """Ordered in-memory text records and their dataset context."""

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    records: list[TextRecord] = Field(min_length=1, description="Text records to anonymize in order.")
    data_summary: str | None = Field(
        default=None, description="Short description of the data. Improves LLM detection accuracy."
    )

    @model_validator(mode="after")
    def validate_record_ids(self) -> TextRecordsInput:
        record_ids = [record.id for record in self.records]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("In-memory record IDs must be unique.")
        return self


class DetectionScope(BaseModel):
    """The detection scope a run used. Finding no entities describes the result under this scope only."""

    model_config = ConfigDict(frozen=True)

    effective_labels: tuple[str, ...] = Field(description="Labels detection looked for, after exclusions.")
    allowed_labels: tuple[str, ...] | None = Field(
        description=(
            "Labels final entities are restricted to (effective_labels when entity_labels is set). "
            "None when entity_labels is unset: final entities may then carry labels outside effective_labels."
        )
    )
    excluded_labels: tuple[str, ...] = Field(description="Configured exclusions (normalized).")
    gliner_threshold: float = Field(description="GLiNER detection confidence threshold.")


class Detect(BaseModel):
    """Configuration for the entity detection stage."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    entity_labels: list[str] | None = Field(
        default=None,
        description=(
            "Labels to detect. None uses the built-in default detection label set. "
            "To inspect the default set, use `from anonymizer import DEFAULT_ENTITY_LABELS`."
        ),
    )
    entity_label_examples: dict[str, list[str]] = Field(
        default_factory=dict,
        repr=False,
        description=(
            "Configured positive detection examples organized by entity label. For default labels, these "
            "values are appended to the built-in examples. Non-default labels must also be declared "
            "in an explicit entity_labels set; examples never activate labels implicitly."
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
            "Target maximum number of candidate entities per validator LLM call. "
            "Overlapping alternatives stay in one chunk; a group larger than this target "
            "is sent alone with a warning. Chunks are dispatched round-robin across the validator pool."
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
    regex_limits: RegexLimits = Field(
        default_factory=RegexLimits,
        description=(
            "Per-record, per-rule validated-match cap and character-scaled timeout, in CPU seconds, "
            "for all regex rules."
        ),
    )

    @property
    def scope(self) -> DetectionScope:
        """The detection scope this configuration resolves to, computed on every read."""
        labels = sorted(self.ordered_labels)
        return DetectionScope(
            effective_labels=tuple(labels),
            allowed_labels=tuple(labels) if self.strict_labels else None,
            excluded_labels=tuple(self.excluded_entity_labels or []),
            gliner_threshold=self.gliner_threshold,
        )

    @property
    def ordered_labels(self) -> tuple[str, ...]:
        """The effective labels in resolution order, computed on every read.

        This is the order sent to the detector and the detection prompts, minus exclusions: the default label order
        followed by the labels of enabled custom regex rules, or `entity_labels`, which validation already sorted.
        `scope.effective_labels` holds the same labels sorted.
        """
        return tuple(
            resolve_effective_detection_labels(
                self.entity_labels,
                regex_rules=self.regex_rules,
                excluded_entity_labels=self.excluded_entity_labels,
            )
        )

    @property
    def strict_labels(self) -> bool:
        """True when final entities are restricted to the effective labels (entity_labels is set)."""
        return self.entity_labels is not None

    @property
    def validator_examples(self) -> dict[str, tuple[str, ...]]:
        """Built-in plus configured examples per effective label, in resolution order."""
        return {label: tuple(examples) for label, examples in self._resolved().validator_examples.items()}

    @property
    def augmenter_examples(self) -> dict[str, tuple[str, ...]]:
        """Configured examples per effective label, in resolution order."""
        return {label: tuple(examples) for label, examples in self._resolved().augmenter_examples.items()}

    def _resolved(self) -> ResolvedEntityLabelConfig:
        return resolve_entity_label_config(
            entity_labels=list(self.ordered_labels),
            excluded_entity_labels=self.excluded_entity_labels,
            entity_label_examples=self.entity_label_examples,
        )

    @field_validator("entity_labels")
    @classmethod
    def validate_entity_labels(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [normalize_entity_label(label) for label in value if normalize_entity_label(label)]
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
        cleaned = [normalize_entity_label(label) for label in value if normalize_entity_label(label)]
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
                "entity_label_examples contained label names that normalize to the same label; "
                "merged automatically (%d labels).",
                len(duplicate_keys),
            )
        if duplicate_value_labels:
            logger.warning(
                "entity_label_examples contained duplicate examples; removed automatically for %d labels.",
                len(duplicate_value_labels),
            )
        return normalized

    @model_validator(mode="after")
    def validate_entity_label_overlap(self) -> Detect:
        excluded_set = set(self.excluded_entity_labels or [])
        example_labels = set(self.entity_label_examples)
        excluded_examples = sorted(example_labels & excluded_set)
        if excluded_examples:
            logger.warning(
                "entity_label_examples configured excluded labels; their examples will be ignored (%d labels).",
                len(excluded_examples),
            )

        active_example_labels = example_labels - excluded_set
        effective_labels = set(
            resolve_effective_detection_labels(
                self.entity_labels,
                regex_rules=self.regex_rules,
                excluded_entity_labels=excluded_set,
            )
        )
        if self.entity_labels is not None:
            entity_labels_set = set(self.entity_labels)
            overlap = sorted(entity_labels_set & excluded_set)
            if overlap:
                logger.warning(
                    "entity_labels and excluded_entity_labels share %d labels that will never be detected.",
                    len(overlap),
                )
        else:
            entity_labels_set = set(DEFAULT_ENTITY_LABELS)

        unknown_examples = sorted(active_example_labels - entity_labels_set)
        if unknown_examples:
            raise ValueError(
                "entity_label_examples contains labels outside the active label set: "
                f"{unknown_examples}. Add every example label to entity_labels "
                "(non-default labels require an explicit entity_labels set) or remove their examples."
            )
        if not effective_labels:
            source = "entity_labels" if self.entity_labels is not None else "DEFAULT_ENTITY_LABELS"
            raise ValueError(
                f"excluded_entity_labels entirely overlaps {source} and all enabled custom regex labels, "
                "leaving an empty effective detection set."
            )
        return self

    @model_validator(mode="after")
    def validate_regex_rule_scope(self) -> Detect:
        enabled_custom_rules = [rule for rule in self.regex_rules if isinstance(rule, RegexRule) and rule.enabled]
        builtin_rules = [rule for rule in self.regex_rules if isinstance(rule, BuiltinRegex)]
        enabled_rules = [
            rule for rule in self.regex_rules if rule.enabled and (self.builtin_regexes or isinstance(rule, RegexRule))
        ]
        identities = [(rule.label, rule.pattern) for rule in enabled_custom_rules]
        if len(set(identities)) != len(identities):
            raise ValueError("regex_rules contains duplicate label and pattern pairs.")
        builtin_labels = [rule.label for rule in builtin_rules]
        if len(set(builtin_labels)) != len(builtin_labels):
            raise ValueError("regex_rules contains duplicate built-in labels.")
        additional_detection_by_label: dict[str, set[bool]] = {}
        for rule in enabled_rules:
            additional_detection_by_label.setdefault(rule.label, set()).add(rule.detect_additional_matches)
        conflicting_labels = sorted(label for label, values in additional_detection_by_label.items() if len(values) > 1)
        if conflicting_labels:
            raise ValueError(
                "Enabled regex rules sharing a label must use the same detect_additional_matches value. "
                f"Conflicting labels: {conflicting_labels!r}."
            )
        if self.entity_labels is not None:
            missing = sorted({rule.label for rule in enabled_custom_rules} - set(self.entity_labels))
            if missing:
                raise ValueError(f"Regex rule labels {missing!r} are missing from explicit entity_labels.")
        return self


INSTRUCTIONS_REMOVED = (
    "Rewrite.instructions was removed in v1; it never reached the rewrite model. "
    "Put the guidance in privacy_goal.protect and privacy_goal.preserve."
)


class Rewrite(BaseModel):
    """Configuration for rewrite-mode execution."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    privacy_goal: PrivacyGoal = Field(
        default_factory=default_privacy_goal,
        description="Structured privacy goal. None or omitted uses the default goal.",
    )
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

    @model_validator(mode="before")
    @classmethod
    def reject_instructions(cls, data: Any) -> Any:
        if isinstance(data, dict) and "instructions" in data:
            if data["instructions"] is not None:
                raise ValueError(INSTRUCTIONS_REMOVED)
            data = {key: value for key, value in data.items() if key != "instructions"}
        return data

    @field_validator("privacy_goal", mode="before")
    @classmethod
    def default_goal_for_none(cls, value: Any) -> Any:
        return default_privacy_goal() if value is None else value

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

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

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

    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)

    compute_detection_validity: bool = False
    """Run the tag-precision judge (detection_valid / detection_invalid_entities).

    Disabled by default — intended for internal use during model and threshold
    experiments. When True, adds
    ``detection_valid`` and ``detection_invalid_entities`` columns to the
    evaluate() output alongside ``entity_coverage``.
    """
