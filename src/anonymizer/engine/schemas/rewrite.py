# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the rewrite pipeline.

Each schema group corresponds to one pipeline step:

    Step 1 — Domain classification
        DomainClassificationSchema

    Step 1b — Entity classification (direct vs. quasi-identifier, non-latent entities)
        EntityLabelClassificationSchema, EntityLabelClassificationsSchema

    Step 2 — Sensitivity disposition (document-level protection plan + per-entity dispositions)
        EntityDispositionSchema, SensitivityDispositionSchema

    Step 3a — Meaning unit extraction
            (Meaning units are small, PII-safe semantic units
            extracted from the source text and used to generate
            content-preservation QA.)
        MeaningUnitsSchema

    Step 3b — QA generation
        QualityQAPairsSchema          (LLM — quality questions from meaning units)
        PrivacyQAPairsSchema          (template — one question per entity needing protection)

    Step 4 — Rewrite generation
        RewriteSchema

    Step 5 — Evaluate & repair
        QualityAnswersSchema          (LLM re-answers quality questions on rewritten text)
        PrivacyAnswersSchema          (LLM re-answers privacy questions on rewritten text)
        QACompareResultsSchema        (LLM scores quality answer match)

    Step 6 — Final judge
        Uses LLMJudgeColumnConfig with Score rubrics (no custom schema needed)

Supporting enums: Domain, EntitySource, EntityCategory, SensitivityLevel,
                  ProtectionMethod, PrivacyAnswer
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

logger = logging.getLogger("anonymizer.schemas.rewrite")

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    """Valid domain types for domain classification and meaning unit extraction.

    Adding a value here also requires a matching entry in ``DOMAIN_METADATA``
    (``anonymizer.engine.rewrite.domain_classification``); that module fails
    to import if the two drift.
    """

    BIOGRAPHY_PROFILE = "BIOGRAPHY_PROFILE"
    INSURANCE = "INSURANCE"
    GOVERNMENT_PUBLIC_RECORDS = "GOVERNMENT_PUBLIC_RECORDS"
    NEWS_PUBLIC_AFFAIRS = "NEWS_PUBLIC_AFFAIRS"
    MARKETING_COMMERCIAL = "MARKETING_COMMERCIAL"
    TECHNICAL_SOFTWARE_ENGINEERING = "TECHNICAL_SOFTWARE_ENGINEERING"
    RESEARCH_SCIENTIFIC = "RESEARCH_SCIENTIFIC"
    SECURITY_INFOSEC = "SECURITY_INFOSEC"
    FINANCIAL = "FINANCIAL"
    ECONOMIC_ANALYSIS = "ECONOMIC_ANALYSIS"
    POLICY_REGULATORY = "POLICY_REGULATORY"
    LEGAL = "LEGAL"
    HR_EMPLOYMENT = "HR_EMPLOYMENT"
    BUSINESS_OPERATIONS = "BUSINESS_OPERATIONS"
    MEDICAL_CLINICAL = "MEDICAL_CLINICAL"
    EDUCATION = "EDUCATION"
    CREATIVE_FICTION = "CREATIVE_FICTION"
    ENTERTAINMENT_MEDIA = "ENTERTAINMENT_MEDIA"
    SOCIAL_COMMENTARY = "SOCIAL_COMMENTARY"
    META_TEXT = "META_TEXT"
    OTHER = "OTHER"


class DomainClassificationSchema(BaseModel):
    """LLM output schema for domain classification step."""

    domain: Domain
    domain_confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Entity Classification (direct vs. quasi-identifier, non-latent entities only)
# ---------------------------------------------------------------------------


class EntityLabelClassificationSchema(BaseModel):
    """LLM classification of one entity label not covered by DEFAULT_ENTITY_LABEL_CATEGORY.

    Restricted to direct/quasi — this step never runs on latent entities, so
    ``latent_identifier`` (the third EntityCategory value) is not a valid output here.
    """

    label: str = Field(min_length=1)
    category: Literal["direct_identifier", "quasi_identifier"]


class EntityLabelClassificationsSchema(BaseModel):
    """LLM output schema for the entity classification step.

    One call per row/document: the prompt lists every unmapped label present
    in that row, and this returns one entry per label. Empty when every label
    in the document is already covered by DEFAULT_ENTITY_LABEL_CATEGORY.
    """

    classifications: list[EntityLabelClassificationSchema] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Sensitivity Disposition
# ---------------------------------------------------------------------------


class EntitySource(str, Enum):
    tagged = "tagged"  # from GLiNER + LLM validation (explicit in text)
    latent = "latent"  # from latent entity detection (inferred from context)


class EntityCategory(str, Enum):
    direct_identifier = "direct_identifier"
    quasi_identifier = "quasi_identifier"
    latent_identifier = "latent_identifier"


class SensitivityLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class ProtectionMethod(str, Enum):
    replace = "replace"
    generalize = "generalize"
    remove = "remove"
    suppress_inference = "suppress_inference"
    leave_as_is = "leave_as_is"


class EntityDispositionSchema(BaseModel):
    """Protection decision for one tagged or latent entity in rewrite planning.

    Each instance represents one entry in the sensitivity disposition, not each
    repeated text span where that entity may appear.

    ``sensitivity`` is a protection-policy tier that drives both protection and
    leakage weighting. ``low`` ⇔ ``leave_as_is``; ``medium``/``high`` ⇒ protected.
    """

    model_config = ConfigDict(use_enum_values=True)

    id: int = Field(ge=1)
    source: EntitySource
    category: EntityCategory
    sensitivity: SensitivityLevel
    entity_label: str = Field(min_length=1)
    entity_value: str = Field(min_length=1)
    protection_reason: str = Field(min_length=10, max_length=500)
    protection_method_suggestion: ProtectionMethod

    @property
    def needs_protection(self) -> bool:
        return self.protection_method_suggestion != ProtectionMethod.leave_as_is

    @model_validator(mode="after")
    def _validate_protection_consistency(self) -> EntityDispositionSchema:
        retained = not self.needs_protection
        if self.sensitivity == SensitivityLevel.low and not retained:
            logger.warning(
                "Entity %d (label=%r): sensitivity='low' conflicts with "
                "protection_method_suggestion=%r; promoting sensitivity to 'medium'.",
                self.id,
                self.entity_label,
                self.protection_method_suggestion,
            )
            # Normalize toward more protection: trust the protection intent over the level.
            self.sensitivity = SensitivityLevel.medium.value  # type: ignore[assignment]  # ty: ignore[invalid-assignment]
        elif self.sensitivity != SensitivityLevel.low and retained:
            # No safe normalization exists (the schema cannot pick a method), so fail closed.
            raise ValueError(
                f"Entity {self.id}: sensitivity={self.sensitivity!r} cannot have protection_method_suggestion='leave_as_is'"
            )
        return self


class SensitivityDispositionSchema(BaseModel):
    """Per-entity dispositions with stable sequential IDs and protection consistency.

    The orchestrator skips this step when detection finds no entities.
    """

    # Non-empty by design: the rewrite workflow only runs when entities were detected.
    # The orchestrator is responsible for short-circuiting before this step if detection
    # found nothing, so an empty disposition indicates a pipeline bug, not a valid state.
    sensitivity_disposition: list[EntityDispositionSchema] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ids(self) -> SensitivityDispositionSchema:
        # IDs are assigned upstream (1..N, explicit then latent) and must not be renumbered.
        ids = [entry.id for entry in self.sensitivity_disposition]
        expected = list(range(1, len(ids) + 1))
        if ids != expected:
            raise ValueError(f"Entity IDs must be sequential 1..{len(ids)} in input order; got {ids}")

        return self

    @property
    def protected_entities(self) -> list[EntityDispositionSchema]:
        return [e for e in self.sensitivity_disposition if e.needs_protection]

    @property
    def medium_and_high_sensitivity_entities(self) -> list[EntityDispositionSchema]:
        return [
            e for e in self.sensitivity_disposition if e.sensitivity in (SensitivityLevel.medium, SensitivityLevel.high)
        ]

    def get_entities_by_sensitivity(self, level: SensitivityLevel | str) -> list[EntityDispositionSchema]:
        if isinstance(level, str):
            level = SensitivityLevel(level)
        return [e for e in self.sensitivity_disposition if e.sensitivity == level]

    def get_entities_by_method(self, method: ProtectionMethod | str) -> list[EntityDispositionSchema]:
        if isinstance(method, str):
            method = ProtectionMethod(method)
        return [e for e in self.sensitivity_disposition if e.protection_method_suggestion == method]

    def format_for_rewrite_context(self) -> str:
        """Format disposition for injection into rewrite prompts — all entities needing protection."""
        entities = self.protected_entities
        if not entities:
            return "No entities needing protection."
        lines = []
        for e in entities:
            lines.append(
                f'- [{e.sensitivity.upper()}] {e.entity_label}: "{e.entity_value}" → {e.protection_method_suggestion} (Reason: {e.protection_reason})'
            )
        return "\n".join(lines)


class StrictProtectionMethod(str, Enum):
    replace = "replace"
    generalize = "generalize"
    remove = "remove"
    suppress_inference = "suppress_inference"


class StrictSensitivityLevel(str, Enum):
    medium = "medium"
    high = "high"


class StrictEntityDispositionSchema(EntityDispositionSchema):
    """Strict variant: leave_as_is and low sensitivity are excluded."""

    sensitivity: StrictSensitivityLevel
    protection_method_suggestion: StrictProtectionMethod


class StrictSensitivityDispositionSchema(SensitivityDispositionSchema):
    """Strict variant container: every entity must be protected."""

    sensitivity_disposition: list[StrictEntityDispositionSchema] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Meaning Units
# ---------------------------------------------------------------------------


class MeaningUnitImportance(str, Enum):
    critical = "critical"
    important = "important"


class MeaningUnitSchema(BaseModel):
    id: int = Field(ge=1)
    aspect: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    importance: MeaningUnitImportance


class MeaningUnitsSchema(BaseModel):
    """LLM output schema for meaning unit extraction step."""

    # Non-empty by design: meaning extraction only runs when entities were detected.
    units: list[MeaningUnitSchema] = Field(min_length=1)


# ---------------------------------------------------------------------------
# QA Generation
# ---------------------------------------------------------------------------


class QualityQAItemSchema(BaseModel):
    id: int
    aspect: str = Field(min_length=1)
    importance: MeaningUnitImportance
    question: str
    reference_answer: str


class QualityQAPairsSchema(BaseModel):
    """LLM output schema for quality QA generation step."""

    items: list[QualityQAItemSchema]


class PrivacyAnswer(str, Enum):
    yes = "yes"
    no = "no"


_NULL_PRIVACY_ANSWER_REASON = "Model returned null answer; defaulted to highest-confidence leak."
_NULL_PRIVACY_REASON = "Model returned no reason."


class PrivacyQuestionSchema(BaseModel):
    id: int
    question: str
    sensitivity: SensitivityLevel
    entity_label: str
    entity_value: str
    category: EntityCategory


class PrivacyQAPairsSchema(BaseModel):
    """Privacy QA pairs for a document — generated from disposition without an LLM.

    All questions expect the answer ``no``. A ``yes`` answer indicates a privacy leak.
    See ``generate_privacy_qa_from_disposition``.
    """

    items: list[PrivacyQuestionSchema]


# ---------------------------------------------------------------------------
# Rewrite
# ---------------------------------------------------------------------------


class RewriteOutputSchema(BaseModel):
    """LLM output schema for rewrite and repair steps."""

    rewritten_text: str


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _validate_id_coverage(expected_ids: list[int], returned_ids: list[int], label: str) -> None:
    """Enforce exact ID coverage: no missing, duplicate, or extra IDs."""
    expected_set = set(expected_ids)
    returned_set = set(returned_ids)

    missing = sorted(expected_set - returned_set)
    if missing:
        raise ValueError(f"Missing {label} IDs: {missing}")

    duplicates = sorted(id for id in returned_set if returned_ids.count(id) > 1)
    if duplicates:
        raise ValueError(f"Duplicate {label} IDs: {duplicates}")

    extra = sorted(returned_set - expected_set)
    if extra:
        raise ValueError(f"Extra {label} IDs not in expected set: {extra}")


class QualityAnswerSchema(BaseModel):
    id: int
    answer: str

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_null_answer(cls, value: object) -> object:
        """Treat an explicit JSON null like an omitted quality answer."""
        return "unknown" if value is None else value


class QualityAnswersSchema(BaseModel):
    """LLM output schema for quality QA re-answer step (on rewritten text).

    When validated with ``context={"expected_ids": [1, 2, ...]}``,
    enforces exact coverage: no missing, duplicate, or extra IDs.
    """

    answers: list[QualityAnswerSchema]

    @model_validator(mode="after")
    def _check_coverage(self, info: ValidationInfo) -> QualityAnswersSchema:
        expected_ids = (info.context or {}).get("expected_ids")
        if expected_ids is not None:
            _validate_id_coverage(expected_ids, [a.id for a in self.answers], "answer")
        return self


class PrivacyAnswerItemSchema(BaseModel):
    id: int
    answer: PrivacyAnswer
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str = Field(min_length=1, max_length=200)
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_null_privacy_fields(cls, value: object) -> object:
        """Normalize a null answer atomically so privacy evaluation fails closed."""
        if not isinstance(value, dict):
            return value

        normalized = value.copy()
        if "answer" in normalized and normalized["answer"] is None:
            normalized["answer"] = PrivacyAnswer.yes
            normalized["confidence"] = 1.0
            normalized["reason"] = _NULL_PRIVACY_ANSWER_REASON
        elif "reason" in normalized and normalized["reason"] is None:
            normalized["reason"] = _NULL_PRIVACY_REASON
        return normalized

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_null_confidence(cls, value: object) -> object:
        """Default an explicit JSON null to highest-confidence leakage."""
        return 1.0 if value is None else value

    @field_validator("evidence", mode="before")
    @classmethod
    def normalize_null_evidence(cls, value: object) -> object:
        """Treat an explicit JSON null like the existing empty-list default."""
        return [] if value is None else value


class PrivacyAnswersSchema(BaseModel):
    """LLM output schema for privacy QA re-answer step (on rewritten text).

    When validated with ``context={"expected_ids": [1, 2, ...]}``,
    enforces exact coverage: no missing, duplicate, or extra IDs.
    """

    answers: list[PrivacyAnswerItemSchema]

    @model_validator(mode="after")
    def _check_coverage(self, info: ValidationInfo) -> PrivacyAnswersSchema:
        expected_ids = (info.context or {}).get("expected_ids")
        if expected_ids is not None:
            _validate_id_coverage(expected_ids, [a.id for a in self.answers], "answer")
        return self


class QACompareItemSchema(BaseModel):
    id: int
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None

    @field_validator("score", mode="before")
    @classmethod
    def normalize_null_score(cls, value: object) -> object:
        """Default an explicit JSON null to the conservative zero score."""
        return 0.0 if value is None else value


class QACompareResultsSchema(BaseModel):
    """LLM output schema for quality QA comparison step.

    When validated with ``context={"expected_ids": [1, 2, ...]}``,
    enforces exact coverage: no missing, duplicate, or extra IDs.
    """

    per_item: list[QACompareItemSchema]

    @model_validator(mode="after")
    def _check_coverage(self, info: ValidationInfo) -> QACompareResultsSchema:
        expected_ids = (info.context or {}).get("expected_ids")
        if expected_ids is not None:
            _validate_id_coverage(expected_ids, [a.id for a in self.per_item], "compare")
        return self
