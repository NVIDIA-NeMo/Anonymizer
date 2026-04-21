# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from anonymizer.engine.schemas.shared import _parse_raw_wrapper


class ValidationChoice(str, Enum):
    keep = "keep"
    reclass = "reclass"
    drop = "drop"


class EntitySchema(BaseModel):
    """Canonical entity span used across workflow columns."""

    id: str = Field(default="")
    value: str = Field(default="")
    label: str = Field(default="")
    start_position: int = Field(default=0)
    end_position: int = Field(default=0)
    score: float = Field(default=0.0)
    source: str = Field(default="detector")


class EntitiesSchema(BaseModel):
    """Canonical ``{"entities": [...]}`` wrapper for entity spans."""

    entities: list[EntitySchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> EntitiesSchema:
        return _parse_raw_wrapper(cls, raw, "entities")


class ValidationCandidateSchema(BaseModel):
    id: str = Field(default="")
    value: str = Field(default="")
    label: str = Field(default="")
    context_before: str = Field(default="")
    context_after: str = Field(default="")


class ValidationCandidatesSchema(BaseModel):
    candidates: list[ValidationCandidateSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> ValidationCandidatesSchema:
        return _parse_raw_wrapper(cls, raw, "candidates", fallback_keys=("entities",))


class RawValidationDecisionSchema(BaseModel):
    id: str = Field(default="")
    decision: ValidationChoice | None = None
    proposed_label: str = Field(default="")
    reason: str | None = None


class RawValidationDecisionsSchema(BaseModel):
    decisions: list[RawValidationDecisionSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> RawValidationDecisionsSchema:
        return _parse_raw_wrapper(cls, raw, "decisions")


class ValidatedDecisionSchema(RawValidationDecisionSchema):
    value: str = Field(default="")
    label: str = Field(default="")


class ValidatedDecisionsSchema(BaseModel):
    decisions: list[ValidatedDecisionSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> ValidatedDecisionsSchema:
        return _parse_raw_wrapper(cls, raw, "decisions")


class ValidationSkeletonDecisionSchema(BaseModel):
    id: str = Field(default="")
    value: str = Field(default="")
    label: str = Field(default="")
    decision: ValidationChoice | None = None
    proposed_label: str | None = None
    reason: str | None = None


class ValidationSkeletonSchema(BaseModel):
    decisions: list[ValidationSkeletonDecisionSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> ValidationSkeletonSchema:
        return _parse_raw_wrapper(cls, raw, "decisions")


class ValidationDecisionSchema(BaseModel):
    """Loose wire-contract for per-entity validation decisions from the LLM.

    The strict internal shape has only the three fields the server actually
    consumes: `id`, `decision`, `proposed_label` (+ optional `reason`). The
    previous schema also carried `value` and `label`, but those are
    overridden from the trusted candidate_lookup in
    enrich_validation_decisions — they were pure drift surface.

    Wire-layer looseness addresses classes M/N/O from the bench:
      * `decision` is typed `str` (not ValidationChoice) so DD’s
        jsonschema pre-check cannot reject enum drift; before-validator
        normalizes to a valid enum. Default "keep" means the field is
        NOT in `required`, so omission does not drop the record.
      * `proposed_label` is `str | None` so the emitted JSON Schema is
        `anyOf: [string, null]` — explicit `null` emissions no longer
        fail `type: "string"` at the pre-check.
      * `value`/`label` removed entirely — any int/null drift on those
        fields is now impossible because they’re not in the schema.
    """

    id: str
    decision: str = Field(
        default="keep",
        description='one of: "keep" | "reclass" | "drop"',
    )
    proposed_label: str | None = Field(
        default="",
        description="Correct label when decision is 'reclass', otherwise empty",
    )
    reason: str | None = None

    @field_validator("proposed_label", mode="before")
    @classmethod
    def _coerce_proposed_label(cls, v: object) -> object:
        """Coerce None / non-string to empty string so the strict downstream
        shape is always a string (RawValidationDecisionSchema expects str)."""
        if v is None:
            return ""
        if isinstance(v, (int, float, bool)):
            return str(v)
        return v

    @field_validator("decision", mode="before")
    @classmethod
    def _normalize_decision(cls, v: object) -> str:
        """Coerce drift into a valid ValidationChoice value.

        None / non-string / unknown strings default to 'keep' — the
        conservative choice that preserves detection. A substring match
        catches small-model variants like 'Keep.' or 'DROP!'.
        """
        if v is None or not isinstance(v, str) or not v.strip():
            return "keep"
        cleaned = v.strip().lower()
        if cleaned in {"keep", "reclass", "drop"}:
            return cleaned
        for choice in ("reclass", "drop", "keep"):  # check most-specific first
            if choice in cleaned:
                return choice
        return "keep"


class ValidationDecisionsSchema(BaseModel):
    decisions: list[ValidationDecisionSchema] = Field(default_factory=list)


class AugmentedEntitySchema(BaseModel):
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    reason: str | None = None


class AugmentedEntitiesSchema(BaseModel):
    entities: list[AugmentedEntitySchema] = Field(default_factory=list)


class LatentCategory(str, Enum):
    latent_identifier = "latent_identifier"
    latent_sensitive_attribute = "latent_sensitive_attribute"


class LatentConfidence(str, Enum):
    high = "high"
    medium = "medium"


class LatentEntitySchema(BaseModel):
    category: LatentCategory
    label: str = Field(
        min_length=1,
        description=(
            "General category/class of the inference in snake_case "
            "(e.g., employer, specific_institution, home_location, medication, health_condition)"
        ),
    )
    value: str = Field(
        min_length=1,
        description="Concise inferred value (generalize if not pinned down strongly by evidence)",
    )
    confidence: LatentConfidence
    evidence: list[str] = Field(
        min_length=1,
        max_length=2,
        description="One or two short quotes from the text that support this inference",
    )
    rationale: str = Field(
        min_length=10,
        max_length=150,
        description="One sentence explaining the inference without adding new facts",
    )

    @field_validator("rationale", mode="before")
    @classmethod
    def _cap_rationale(cls, v: object) -> object:
        """Truncate overlong rationales so verbose models (e.g. Nemotron) do not
        fail the maxLength=150 constraint on dense notes. Observed on the
        oncology bench note: a 260-char rationale dropped the whole record."""
        if isinstance(v, str) and len(v) > 150:
            return v[:147].rstrip() + "..."
        return v


class LatentEntitiesSchema(BaseModel):
    latent_entities: list[LatentEntitySchema] = Field(default_factory=list)


class EntityByValueSchema(BaseModel):
    value: str = Field(default="")
    labels: list[str] = Field(default_factory=list)


class EntitiesByValueSchema(BaseModel):
    entities_by_value: list[EntityByValueSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> EntitiesByValueSchema:
        return _parse_raw_wrapper(cls, raw, "entities_by_value", fallback_keys=("entities",))
