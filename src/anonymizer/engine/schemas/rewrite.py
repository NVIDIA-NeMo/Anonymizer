# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pydantic schemas for the rewrite pipeline.

Each schema group corresponds to one pipeline step:

    Step 1 — Domain classification
        DomainClassificationSchema

    Step 2 — Sensitivity disposition (per-entity protection plan)
        EntityDispositionSchema, SensitivityDispositionSchema

    Step 3a — Meaning unit extraction
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
        JudgeEvaluationSchema         (holistic privacy + quality + naturalness scores)

Supporting enums: Domain, EntitySource, EntityCategory, SensitivityLevel,
                  ProtectionMethod, CombinedRiskLevel, MeaningUnitAspect, PrivacyAnswer

Utility: generate_privacy_qa_from_disposition — template-based, no LLM required.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

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
    paraphrase = "paraphrase"
    left_as_is = "left_as_is"


class CombinedRiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class EntityDispositionSchema(BaseModel):
    """Protection decision for a single entity."""

    model_config = ConfigDict(use_enum_values=True)

    id: int = Field(ge=1)
    source: EntitySource
    category: EntityCategory
    sensitivity: SensitivityLevel
    entity_label: str = Field(min_length=1)
    entity_value: str = Field(min_length=1)
    does_need_protection: bool
    protection_reason: str = Field(min_length=10, max_length=500)
    protection_method_suggestion: ProtectionMethod
    combined_risk_level: CombinedRiskLevel

    @model_validator(mode="after")
    def _validate_protection_consistency(self) -> EntityDispositionSchema:
        if not self.does_need_protection and self.protection_method_suggestion != ProtectionMethod.left_as_is:
            raise ValueError(
                f"Entity {self.id}: does_need_protection=False requires protection_method_suggestion='left_as_is', "
                f"got '{self.protection_method_suggestion}'"
            )
        if self.does_need_protection and self.protection_method_suggestion == ProtectionMethod.left_as_is:
            raise ValueError(
                f"Entity {self.id}: does_need_protection=True cannot have protection_method_suggestion='left_as_is'"
            )
        return self


class SensitivityDispositionSchema(BaseModel):
    """Complete sensitivity disposition for a document — LLM output schema.

    Validates that entity IDs are sequential from 1 and that each entity's
    ``does_need_protection`` flag is consistent with its ``protection_method_suggestion``.
    """

    # Non-empty by design: the rewrite workflow only runs when entities were detected.
    # The orchestrator is responsible for short-circuiting before this step if detection
    # found nothing, so an empty disposition indicates a pipeline bug, not a valid state.
    sensitivity_disposition: list[EntityDispositionSchema] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_ids_sequential(self) -> SensitivityDispositionSchema:
        ids = [e.id for e in self.sensitivity_disposition]
        expected = list(range(1, len(ids) + 1))
        if sorted(ids) != expected:
            raise ValueError(f"Entity IDs must be sequential starting from 1. Got: {sorted(ids)}, expected: {expected}")
        return self

    def entities_needing_protection(self) -> list[EntityDispositionSchema]:
        return [e for e in self.sensitivity_disposition if e.does_need_protection]

    def entities_by_sensitivity(self, level: SensitivityLevel | str) -> list[EntityDispositionSchema]:
        if isinstance(level, str):
            level = SensitivityLevel(level)
        return [e for e in self.sensitivity_disposition if e.sensitivity == level]


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


class QualityAnswerSchema(BaseModel):
    id: str
    answer: str


class QualityAnswersSchema(BaseModel):
    """LLM output schema for quality QA re-answer step (on rewritten text)."""

    answers: list[QualityAnswerSchema]


class PrivacyAnswerItemSchema(BaseModel):
    id: str
    answer: PrivacyAnswer


class PrivacyAnswersSchema(BaseModel):
    """LLM output schema for privacy QA re-answer step (on rewritten text)."""

    answers: list[PrivacyAnswerItemSchema]


class QACompareItemSchema(BaseModel):
    id: str
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


class QACompareResultsSchema(BaseModel):
    """LLM output schema for quality QA comparison step."""

    per_item: list[QACompareItemSchema]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


class JudgeScoreSchema(BaseModel):
    """Single rubric score from the final holistic judge."""

    score: int = Field(ge=1, le=10)
    reason: str = Field(default="")


class JudgeEvaluationSchema(BaseModel):
    """LLM output schema for the final judge step (privacy + quality + naturalness rubrics)."""

    privacy: JudgeScoreSchema
    quality: JudgeScoreSchema
    naturalness: JudgeScoreSchema


# ---------------------------------------------------------------------------
# Utility: template-based privacy QA generation (no LLM required)
# TODO(#32): move to engine/rewrite/steps/qa_generation.py once that step exists
# ---------------------------------------------------------------------------


def generate_privacy_qa_from_disposition(
    sensitivity_disposition: SensitivityDispositionSchema | dict[str, Any],
) -> PrivacyQAPairsSchema:
    """Generate privacy QA questions from sensitivity disposition without an LLM call.

    Produces one question per entity that needs protection:
        "Can the {entity_label} '{entity_value}' be deduced from the rewritten text?"
    All questions expect the answer "no" — a "yes" answer indicates a privacy leak.
    """
    if isinstance(sensitivity_disposition, dict):
        raw_entities = sensitivity_disposition.get("sensitivity_disposition", [])
        entities_to_protect = [e for e in raw_entities if e.get("does_need_protection", False)]
        questions = []
        for idx, e in enumerate(entities_to_protect, start=1):
            questions.append(
                PrivacyQuestionSchema(
                    id=idx,
                    question=f"Can the {e['entity_label']} '{e['entity_value']}' be deduced from the rewritten text?",
                    sensitivity=SensitivityLevel(e.get("sensitivity", "medium")),
                    entity_label=e["entity_label"],
                    entity_value=e["entity_value"],
                    category=EntityCategory(e.get("category", "quasi_identifier")),
                )
            )
    else:
        entities_to_protect = sensitivity_disposition.entities_needing_protection()
        questions = []
        for idx, e in enumerate(entities_to_protect, start=1):
            questions.append(
                PrivacyQuestionSchema(
                    id=idx,
                    question=f"Can the {e.entity_label} '{e.entity_value}' be deduced from the rewritten text?",
                    sensitivity=SensitivityLevel(e.sensitivity),
                    entity_label=e.entity_label,
                    entity_value=e.entity_value,
                    category=EntityCategory(e.category),
                )
            )
    return PrivacyQAPairsSchema(items=questions)
