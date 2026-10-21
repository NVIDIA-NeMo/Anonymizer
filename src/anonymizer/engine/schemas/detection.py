# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from anonymizer.engine.schemas.shared import _parse_raw_wrapper


class FrozenStrictModel(BaseModel):
    """Base model that forbids mutation and undeclared fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceTextRef(FrozenStrictModel):
    """Execution-scoped identity for one canonical source-text field."""

    record_id: str = Field(min_length=1)
    field_id: str = Field(min_length=1)


class SourceView(FrozenStrictModel):
    """Contiguous canonical source range supplied to a detection component."""

    source_text_ref: SourceTextRef
    start_position: int = Field(ge=0)
    end_position: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> SourceView:
        if self.end_position < self.start_position:
            raise ValueError("source view end must not precede its start")
        return self


class AdmissionRoute(str, Enum):
    llm_validation = "llm_validation"
    direct_acceptance = "direct_acceptance"


class ObservationRef(FrozenStrictModel):
    """Immutable reference to one observation within one detector result."""

    detection_result_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)


class Observation(FrozenStrictModel):
    """One source-bound detector claim before admission."""

    detection_result_id: str = Field(min_length=1)
    observation_id: str = Field(min_length=1)
    source_text_ref: SourceTextRef
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start_position: int = Field(ge=0)
    end_position: int = Field(gt=0)
    source_kind: str = Field(min_length=1)
    source_instance: str = Field(min_length=1)
    admission_route: AdmissionRoute
    native_score: float | None = Field(default=None, allow_inf_nan=False)

    @property
    def ref(self) -> ObservationRef:
        """Return the immutable reference to this observation."""
        return ObservationRef(
            detection_result_id=self.detection_result_id,
            observation_id=self.observation_id,
        )

    @model_validator(mode="after")
    def validate_geometry(self) -> Observation:
        if self.end_position <= self.start_position:
            raise ValueError("observation must cover a non-empty span")
        if len(self.value) != self.end_position - self.start_position:
            raise ValueError("observation value length must match its source bounds")
        return self


class DetectionResult(FrozenStrictModel):
    """Successful result of one detector invocation over one source view."""

    detection_result_id: str = Field(min_length=1)
    source_view: SourceView
    source_kind: str = Field(min_length=1)
    source_instance: str = Field(min_length=1)
    observations: tuple[Observation, ...] = ()

    @model_validator(mode="after")
    def validate_result(self) -> DetectionResult:
        observation_ids = [item.observation_id for item in self.observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation IDs must be unique within a result")
        for observation in self.observations:
            if observation.detection_result_id != self.detection_result_id:
                raise ValueError("observation belongs to another result")
            if observation.source_text_ref != self.source_view.source_text_ref:
                raise ValueError("observation belongs to another source text")
            if observation.source_kind != self.source_kind or observation.source_instance != self.source_instance:
                raise ValueError("observation belongs to another component")
            if (
                observation.start_position < self.source_view.start_position
                or observation.end_position > self.source_view.end_position
            ):
                raise ValueError("observation falls outside its source view")
        return self


class EffectiveLabelPolicy(FrozenStrictModel):
    """Resolved label scope attached to a detection or validation request."""

    allowed_labels: tuple[str, ...] | None = None
    excluded_labels: tuple[str, ...] = ()


class ValidationInventoryEntry(FrozenStrictModel):
    """Authoritative request-local mapping from an alias to an observation."""

    alias: str = Field(pattern=r"^c[1-9][0-9]*$")
    conflict_group_id: str | None = Field(default=None, pattern=r"^g[1-9][0-9]*$")
    observation_ref: ObservationRef
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start_position: int = Field(ge=0)
    end_position: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> ValidationInventoryEntry:
        if self.end_position <= self.start_position:
            raise ValueError("inventory entry must cover a non-empty span")
        if len(self.value) != self.end_position - self.start_position:
            raise ValueError("inventory entry value length must match its source bounds")
        return self


class ValidationRequestInventory(FrozenStrictModel):
    """Immutable authority map for one logical validation request."""

    request_id: str = Field(min_length=1)
    source_text_ref: SourceTextRef
    context_view: SourceView
    source_context: str
    effective_label_policy: EffectiveLabelPolicy
    entries: tuple[ValidationInventoryEntry, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_inventory(self) -> ValidationRequestInventory:
        aliases = [entry.alias for entry in self.entries]
        if len(aliases) != len(set(aliases)):
            raise ValueError("request-local aliases must be unique")
        observation_refs = [entry.observation_ref for entry in self.entries]
        if len(observation_refs) != len(set(observation_refs)):
            raise ValueError("request-local observation references must be unique")
        if self.context_view.source_text_ref != self.source_text_ref:
            raise ValueError("context view belongs to another source text")
        if len(self.source_context) != (self.context_view.end_position - self.context_view.start_position):
            raise ValueError("context text does not match its source view")
        for entry in self.entries:
            if (
                entry.start_position < self.context_view.start_position
                or entry.end_position > self.context_view.end_position
            ):
                raise ValueError("inventory entry falls outside its context")
        return self


class AdmissionStatus(str, Enum):
    """Terminal admission outcome for one observation.

    ``accepted`` means the observation was explicitly admitted, either by a
    validator keep/reclass decision or by a route that bypasses validation.
    ``rejected`` means the observation was not admitted because the validator
    dropped it or its label or route is ineligible. ``fallback_protected`` means
    the validator returned no usable decision (the alias was omitted, or
    reclass had no label), so the exact original span and label are retained
    for privacy. Fallback is not acceptance and grants no accepted provenance.
    """

    accepted = "accepted"
    rejected = "rejected"
    fallback_protected = "fallback_protected"


class AdmissionReason(str, Enum):
    validator_keep = "validator_keep"
    validator_reclass = "validator_reclass"
    validator_drop = "validator_drop"
    missing_decision_fallback = "missing_decision_fallback"
    unusable_reclass_fallback = "unusable_reclass_fallback"
    direct_acceptance = "direct_acceptance"
    augmenter_output = "augmenter_output"
    ineligible_label = "ineligible_label"


class AdmissionResult(FrozenStrictModel):
    """Terminal admission outcome for one observation."""

    observation_ref: ObservationRef
    route: AdmissionRoute
    status: AdmissionStatus
    effective_label: str | None = None
    reason: AdmissionReason

    @model_validator(mode="after")
    def validate_result(self) -> AdmissionResult:
        accepted_reasons = {
            AdmissionReason.validator_keep,
            AdmissionReason.validator_reclass,
            AdmissionReason.direct_acceptance,
            AdmissionReason.augmenter_output,
        }
        rejected_reasons = {
            AdmissionReason.validator_drop,
            AdmissionReason.ineligible_label,
        }
        fallback_reasons = {
            AdmissionReason.missing_decision_fallback,
            AdmissionReason.unusable_reclass_fallback,
        }
        llm_reasons = {
            AdmissionReason.validator_keep,
            AdmissionReason.validator_reclass,
            AdmissionReason.validator_drop,
            *fallback_reasons,
        }
        direct_reasons = {
            AdmissionReason.direct_acceptance,
            AdmissionReason.augmenter_output,
        }
        if self.status is AdmissionStatus.rejected:
            if self.effective_label is not None:
                raise ValueError("a rejected result has no effective label")
            valid_reasons = rejected_reasons
        elif not self.effective_label:
            raise ValueError("a retained result requires an effective label")
        elif self.status is AdmissionStatus.accepted:
            valid_reasons = accepted_reasons
        else:
            valid_reasons = fallback_reasons
        if self.reason not in valid_reasons:
            raise ValueError("admission reason does not match its status")
        if self.reason in llm_reasons and self.route is not AdmissionRoute.llm_validation:
            raise ValueError("validator reason requires LLM validation")
        if self.reason in direct_reasons and self.route is not AdmissionRoute.direct_acceptance:
            raise ValueError("direct reason requires direct acceptance")
        return self


class AcceptedOrigin(FrozenStrictModel):
    """Explicitly accepted origin retained as candidate provenance."""

    observation_ref: ObservationRef
    source_kind: str = Field(min_length=1)
    source_instance: str = Field(min_length=1)
    route: AdmissionRoute


class NativeScore(FrozenStrictModel):
    """Source-native score tied to the observation that produced it."""

    observation_ref: ObservationRef
    source_kind: str = Field(min_length=1)
    source_instance: str = Field(min_length=1)
    score: float = Field(allow_inf_nan=False)


class AdmittedCandidate(FrozenStrictModel):
    """Eligible protection candidate produced after observation admission."""

    source_text_ref: SourceTextRef
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    start_position: int = Field(ge=0)
    end_position: int = Field(gt=0)
    lineage: tuple[ObservationRef, ...] = Field(min_length=1)
    accepted_origins: tuple[AcceptedOrigin, ...] = ()
    native_scores: tuple[NativeScore, ...] = ()
    has_fallback_protection: bool

    @property
    def has_explicit_acceptance(self) -> bool:
        """Whether at least one origin explicitly admitted this candidate."""
        return bool(self.accepted_origins)

    @model_validator(mode="after")
    def validate_candidate(self) -> AdmittedCandidate:
        if self.end_position <= self.start_position:
            raise ValueError("candidate must cover a non-empty span")
        if len(self.value) != self.end_position - self.start_position:
            raise ValueError("candidate value length must match its source bounds")
        if not (self.has_explicit_acceptance or self.has_fallback_protection):
            raise ValueError("candidate requires an admitted protection state")
        lineage = set(self.lineage)
        accepted_origin_refs = {origin.observation_ref for origin in self.accepted_origins}
        if not accepted_origin_refs <= lineage:
            raise ValueError("accepted origins must reference candidate lineage")
        native_score_refs = {score.observation_ref for score in self.native_scores}
        if not native_score_refs <= accepted_origin_refs:
            raise ValueError("native scores must reference accepted origins")
        return self


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

    @field_validator("proposed_label", mode="before")
    @classmethod
    def normalize_null_proposed_label(cls, value: object) -> object:
        """Treat an explicit JSON null like an omitted optional response field."""
        return "" if value is None else value


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
    """Per-entity validation decision from the LLM validator."""

    id: str
    value: str = Field(default="", description="Entity value (echoed from skeleton)")
    label: str = Field(default="", description="Entity label (echoed from skeleton)")
    decision: ValidationChoice
    proposed_label: str = Field(
        default="",
        description="Correct label when decision is 'reclass', otherwise empty",
    )
    reason: str | None = None


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
        min_length=20,
        max_length=250,
        description="One sentence explaining the inference without adding new facts",
    )


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
