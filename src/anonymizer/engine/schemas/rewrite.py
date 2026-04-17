# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the rewrite pipeline.

Each schema group corresponds to one pipeline step:

    Step 1 — Domain classification
        DomainClassificationSchema

    Step 2 — Sensitivity disposition (per-entity protection plan)
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
                  ProtectionMethod, CombinedRiskLevel, MeaningUnitAspect, PrivacyAnswer
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------


class Domain(str, Enum):
    """Valid domain types for domain classification and meaning unit extraction."""

    BIOGRAPHY = "BIOGRAPHY"
    CHAT_EMAIL_CSAT = "CHAT_EMAIL_CSAT"
    PRODUCT_REVIEW = "PRODUCT_REVIEW"
    NEWS_JOURNALISM = "NEWS_JOURNALISM"
    MARKETING_ADVERTISING = "MARKETING_ADVERTISING"
    TECHNICAL_ENGINEERING_SOFTWARE = "TECHNICAL_ENGINEERING_SOFTWARE"
    SCIENTIFIC_ACADEMIC = "SCIENTIFIC_ACADEMIC"
    SECURITY_INFOSEC = "SECURITY_INFOSEC"
    FINANCIAL = "FINANCIAL"
    ECONOMIC = "ECONOMIC"
    POLICY_REGULATORY_COMPLIANCE = "POLICY_REGULATORY_COMPLIANCE"
    LEGAL = "LEGAL"
    HR_PEOPLE_OPS = "HR_PEOPLE_OPS"
    MANAGEMENT_OPERATIONS = "MANAGEMENT_OPERATIONS"
    CLINICAL_EHR_MEDICAL = "CLINICAL_EHR_MEDICAL"
    EDUCATIONAL_PEDAGOGICAL = "EDUCATIONAL_PEDAGOGICAL"
    FICTION_CREATIVE = "FICTION_CREATIVE"
    ENTERTAINMENT_MEDIA = "ENTERTAINMENT_MEDIA"
    SOCIAL_CULTURAL_OPED = "SOCIAL_CULTURAL_OPED"
    PROCEDURAL_INSTRUCTIONAL = "PROCEDURAL_INSTRUCTIONAL"
    META_TEXT = "META_TEXT"
    SOCIAL_MEDIA = "SOCIAL_MEDIA"
    TRANSCRIPTS_INTERVIEWS = "TRANSCRIPTS_INTERVIEWS"
    OTHER = "OTHER"


class DomainClassificationSchema(BaseModel):
    """LLM output schema for domain classification step."""

    domain: Domain
    domain_confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Sensitivity Disposition
# ---------------------------------------------------------------------------


class EntitySource(str, Enum):
    tagged = "tagged"  # from GLiNER + LLM validation (explicit in text)
    latent = "latent"  # from latent entity detection (inferred from context)


class EntityCategory(str, Enum):
    direct_identifier = "direct_identifier"
    quasi_identifier = "quasi_identifier"
    sensitive_attribute = "sensitive_attribute"
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


class CombinedRiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class EntityDispositionSchema(BaseModel):
    """Protection decision for one tagged or latent entity in rewrite planning.

    Each instance represents one entry in the sensitivity disposition, not each
    repeated text span where that entity may appear.
    """

    model_config = ConfigDict(use_enum_values=True)

    id: int = Field(ge=1)
    source: EntitySource
    category: EntityCategory
    sensitivity: SensitivityLevel
    entity_label: str = Field(min_length=1)
    entity_value: str = Field(min_length=1)
    needs_protection: bool
    protection_reason: str = Field(min_length=10, max_length=500)
    protection_method_suggestion: ProtectionMethod
    combined_risk_level: CombinedRiskLevel

    @model_validator(mode="after")
    def _validate_protection_consistency(self) -> EntityDispositionSchema:
        if not self.needs_protection and self.protection_method_suggestion != ProtectionMethod.leave_as_is:
            raise ValueError(
                f"Entity {self.id}: needs_protection=False requires protection_method_suggestion='leave_as_is', "
                f"got '{self.protection_method_suggestion}'"
            )
        if self.needs_protection and self.protection_method_suggestion == ProtectionMethod.leave_as_is:
            raise ValueError(
                f"Entity {self.id}: needs_protection=True cannot have protection_method_suggestion='leave_as_is'"
            )
        return self


class SensitivityDispositionSchema(BaseModel):
    """Complete sensitivity disposition for a document — LLM output schema.

    Validates that entity IDs are sequential from 1 and that each entity's
    ``needs_protection`` flag is consistent with its ``protection_method_suggestion``.

    ``sensitivity_disposition`` requires at least one entry (``min_length=1``).
    The orchestrator short-circuits before this step when detection finds no
    entities, so an empty list here indicates a pipeline bug.
    """

    # Non-empty by design: the rewrite workflow only runs when entities were detected.
    # The orchestrator is responsible for short-circuiting before this step if detection
    # found nothing, so an empty disposition indicates a pipeline bug, not a valid state.
    sensitivity_disposition: list[EntityDispositionSchema] = Field(min_length=1)

    @model_validator(mode="after")
    def _normalize_ids(self) -> SensitivityDispositionSchema:
        for i, entry in enumerate(self.sensitivity_disposition, start=1):
            entry.id = i
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


class StrictEntityDispositionSchema(BaseModel):
    """Strict variant: needs_protection is always True and leave_as_is is excluded."""

    model_config = ConfigDict(use_enum_values=True)

    id: int = Field(ge=1)
    source: EntitySource
    category: EntityCategory
    sensitivity: SensitivityLevel
    entity_label: str = Field(min_length=1)
    entity_value: str = Field(min_length=1)
    needs_protection: Literal[True]
    protection_reason: str = Field(min_length=10, max_length=500)
    protection_method_suggestion: StrictProtectionMethod
    combined_risk_level: CombinedRiskLevel


class StrictSensitivityDispositionSchema(BaseModel):
    """Strict variant container: every entity must have needs_protection=True."""

    sensitivity_disposition: list[StrictEntityDispositionSchema] = Field(min_length=1)

    @model_validator(mode="after")
    def _normalize_ids(self) -> StrictSensitivityDispositionSchema:
        for i, entry in enumerate(self.sensitivity_disposition, start=1):
            entry.id = i
        return self


# ---------------------------------------------------------------------------
# Meaning Units
# ---------------------------------------------------------------------------


class MeaningUnitAspect(str, Enum):
    ROLE = "role"
    PROCESS = "process"
    RELATIONSHIP = "relationship"
    ENVIRONMENT = "environment"
    ROUTINE = "routine"
    CREATIVE_OUTPUT = "creative_output"
    VALUE = "value"
    MOTIVATION = "motivation"
    INFLUENCE = "influence"
    AUDIENCE = "audience"
    LEGAL_BASIS = "legal_basis"
    INSTITUTION = "institution"
    JUSTIFICATION = "justification"
    PROCEDURAL_STATUS = "procedural_status"
    TEMPORAL_SEQUENCE = "temporal_sequence"
    RIGHTS_IMPACT = "rights_impact"


class MeaningUnitSchema(BaseModel):
    id: int = Field(ge=1)
    aspect: MeaningUnitAspect
    unit: str = Field(min_length=1)


class MeaningUnitsSchema(BaseModel):
    """LLM output schema for meaning unit extraction step."""

    # Non-empty by design: meaning extraction only runs when entities were detected.
    units: list[MeaningUnitSchema] = Field(min_length=1)


# ---------------------------------------------------------------------------
# QA Generation
# ---------------------------------------------------------------------------


class QualityQAItemSchema(BaseModel):
    id: int
    aspect: str
    question: str
    reference_answer: str


class QualityQAPairsSchema(BaseModel):
    """LLM output schema for quality QA generation step."""

    items: list[QualityQAItemSchema]


class PrivacyAnswer(str, Enum):
    yes = "yes"
    no = "no"


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
