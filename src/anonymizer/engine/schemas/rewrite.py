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
                  ProtectionMethod, MeaningUnitAspect, PrivacyAnswer
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

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


# Best-effort mapping from Anonymizer entity-labels to EntityCategory, used when
# the model outputs an entity_label in the `category` field (observed with small
# Gemma models on the disposition step). Values are from DEFAULT_ENTITY_LABELS in
# config/entity_labels.py; any label not in this table falls back to
# "quasi_identifier" which is the most conservative (protect-cautiously) choice.
_ENTITY_LABEL_TO_CATEGORY: dict[str, str] = {
    # Direct identifiers: strong re-id on their own.
    "first_name": "direct_identifier",
    "last_name": "direct_identifier",
    "full_name": "direct_identifier",
    "email": "direct_identifier",
    "phone_number": "direct_identifier",
    "fax_number": "direct_identifier",
    "ssn": "direct_identifier",
    "national_id": "direct_identifier",
    "passport_number": "direct_identifier",
    "drivers_license": "direct_identifier",
    "street_address": "direct_identifier",
    "address": "direct_identifier",
    "postcode": "direct_identifier",
    "zip_code": "direct_identifier",
    "credit_debit_card": "direct_identifier",
    "credit_card_number": "direct_identifier",
    "account_number": "direct_identifier",
    "bank_routing_number": "direct_identifier",
    "tax_id": "direct_identifier",
    "medical_record_number": "direct_identifier",
    "health_plan_beneficiary_number": "direct_identifier",
    "api_key": "direct_identifier",
    "password": "direct_identifier",
    "ipv4": "direct_identifier",
    "ipv6": "direct_identifier",
    "mac_address": "direct_identifier",
    "url": "direct_identifier",
    "user_name": "direct_identifier",
    "employee_id": "direct_identifier",
    "customer_id": "direct_identifier",
    "unique_id": "direct_identifier",
    "biometric_identifier": "direct_identifier",
    "device_identifier": "direct_identifier",
    "license_plate": "direct_identifier",
    "vehicle_identifier": "direct_identifier",
    "swift_bic": "direct_identifier",
    "pin": "direct_identifier",
    "cvv": "direct_identifier",
    "http_cookie": "direct_identifier",
    # Quasi-identifiers: weaker re-id, combinable with others.
    "age": "quasi_identifier",
    "date": "quasi_identifier",
    "date_of_birth": "quasi_identifier",
    "date_time": "quasi_identifier",
    "time": "quasi_identifier",
    "city": "quasi_identifier",
    "state": "quasi_identifier",
    "country": "quasi_identifier",
    "county": "quasi_identifier",
    "place_name": "quasi_identifier",
    "landmark": "quasi_identifier",
    "coordinate": "quasi_identifier",
    "occupation": "quasi_identifier",
    "organization_name": "quasi_identifier",
    "company_name": "quasi_identifier",
    "university": "quasi_identifier",
    "court_name": "quasi_identifier",
    "prison_detention_facility": "quasi_identifier",
    "degree": "quasi_identifier",
    "field_of_study": "quasi_identifier",
    "education_level": "quasi_identifier",
    "language": "quasi_identifier",
    "nationality": "quasi_identifier",
    "employment_status": "quasi_identifier",
    "monetary_amount": "quasi_identifier",
    "certificate_license_number": "quasi_identifier",
    # Sensitive attributes: harmful to disclose regardless of re-id.
    "gender": "sensitive_attribute",
    "sexuality": "sensitive_attribute",
    "race_ethnicity": "sensitive_attribute",
    "religious_belief": "sensitive_attribute",
    "political_view": "sensitive_attribute",
    "blood_type": "sensitive_attribute",
}


class EntityDispositionSchema(BaseModel):
    """Protection decision for one tagged or latent entity in rewrite planning.

    Each instance represents one entry in the sensitivity disposition, not each
    repeated text span where that entity may appear.
    """

    model_config = ConfigDict(use_enum_values=True)

    # Coerce small-model output before pydantic's strict checks run:
    #   - normalize `category` display-label drift (e.g. "DIRECT IDENTIFIERS"
    #     or "last_name") into the expected enum string where possible
    #     (common Gemma-family failure mode).
    #   - coerce unquoted ints on string fields (Gemma 4 26B emits postcodes
    #     like 98101 as bare integers).
    @model_validator(mode="before")
    @classmethod
    def _coerce_small_model_output(cls, data):
        if not isinstance(data, dict):
            return data
        # Coerce int → str on string fields; drop None so pydantic defaults apply
        # (e.g. protection_reason default kicks in when the model emits null).
        for k in ("entity_label", "entity_value", "protection_reason"):
            if k not in data:
                continue
            v = data[k]
            if isinstance(v, (int, float)):
                data[k] = str(v)
            elif v is None:
                data.pop(k)
        # Normalize `category` drift.
        # (1) display-label → enum, plural → singular;
        # (2) when the model confuses fields and puts the entity_label (e.g.
        #     "last_name", "date_of_birth") into the category slot, map it back
        #     to the most-likely EntityCategory using a known-labels table.
        cat = data.get("category")
        if isinstance(cat, str):
            normalized = cat.strip().lower().replace("-", "_").replace(" ", "_")
            allowed = {"direct_identifier", "quasi_identifier",
                       "sensitive_attribute", "latent_identifier"}
            if normalized in allowed:
                data["category"] = normalized
            elif normalized.endswith("s") and normalized[:-1] in allowed:
                data["category"] = normalized[:-1]
            elif normalized.endswith("_identifiers"):
                data["category"] = normalized[:-1]
            else:
                # Merged-enum hallucination: e.g. "latent_sensitive_attribute"
                # (observed from Nemotron on the oncology bench note). The model
                # splices two legitimate EntityCategory values. Resolve via
                # substring match; order = strongest protection wins so that
                # "latent_sensitive_attribute" maps to sensitive_attribute
                # (harm dimension) rather than latent_identifier (inference).
                # The `source` field (tagged|latent) preserves the latent
                # provenance separately.
                merged = None
                for sub, target in (
                    ("direct", "direct_identifier"),
                    ("sensitive", "sensitive_attribute"),
                    ("latent", "latent_identifier"),
                    ("quasi", "quasi_identifier"),
                ):
                    if sub in normalized:
                        merged = target
                        break
                if merged is not None:
                    data["category"] = merged
                else:
                    # Entity-label confusion: model wrote an entity_label
                    # value in the category slot. Map via a best-effort
                    # label → category table.
                    mapped = _ENTITY_LABEL_TO_CATEGORY.get(normalized)
                    if mapped is not None:
                        data["category"] = mapped
                    elif data.get("entity_label") == cat:
                        # Model definitely confused fields; conservative fallback.
                        data["category"] = "quasi_identifier"
                    # Otherwise leave as-is; pydantic raises a clear enum error.

        # Reconcile the needs_protection <=> protection_method_suggestion
        # consistency rule (enforced by @model_validator(mode="after")
        # below). Small models frequently emit inconsistent pairs in BOTH
        # directions: (needs_protection=False, method="suppress_inference")
        # or (needs_protection=True, method="leave_as_is"). Resolve by
        # biasing toward protection (privacy-safe):
        #   * if any non-trivial method is chosen, set needs_protection=True;
        #   * if needs_protection=True but method="leave_as_is", fall back
        #     to method="replace" (the most generic protective action).
        method = data.get("protection_method_suggestion")
        needs = data.get("needs_protection")
        if isinstance(method, str) and method and method != "leave_as_is":
            if needs is False:
                data["needs_protection"] = True
        elif method == "leave_as_is" and needs is True:
            data["protection_method_suggestion"] = "replace"
        return data

    id: int = Field(ge=1)
    source: EntitySource
    # NOTE: typed as `str` (not EntityCategory) so DD’s jsonschema pre-check
    # does not reject small-model enum drift (e.g. "last_name" in category,
    # "latent_sensitive_attribute" merges) before our @model_validator(mode=
    # "before") gets to coerce it. Enum membership is re-enforced after
    # coercion by _validate_category below.
    category: str = Field(
        min_length=1,
        description=(
            "One of: direct_identifier | quasi_identifier | sensitive_attribute "
            "| latent_identifier"
        ),
    )
    sensitivity: SensitivityLevel
    entity_label: str = Field(min_length=1)
    entity_value: str = Field(min_length=1)
    needs_protection: bool
    # Default makes this optional in the emitted JSON Schema so DD’s
    # jsonschema.validate() does not drop records when small models omit the
    # field (observed on gemma4-e2b, oncology bench note, 2026-04). A generic
    # placeholder is strictly better than dropping the row — the record still
    # flows downstream and the rewrite + judge passes can still evaluate it.
    protection_reason: str = Field(
        default="auto: model omitted protection_reason",
        min_length=10,
        max_length=500,
    )
    protection_method_suggestion: ProtectionMethod

    @field_validator("category", mode="after")
    @classmethod
    def _validate_category(cls, v: str) -> str:
        allowed = {c.value for c in EntityCategory}
        if v not in allowed:
            raise ValueError(
                f"category must be one of {sorted(allowed)}; got {v!r} "
                "(after @model_validator(mode='before') normalization)"
            )
        return v

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
