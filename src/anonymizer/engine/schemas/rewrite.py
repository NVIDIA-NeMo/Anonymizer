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

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from anonymizer.engine.schemas.shared import accept_bare_list, loose_list_wrapper_json_schema

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
    """LLM output schema for domain classification step.

    Wire contract is loose so DD's jsonschema pre-check cannot reject
    enum or type drift: ``domain`` is typed as ``str`` (not the Domain
    enum); ``domain_confidence`` accepts string-typed input. A pair of
    before-validators normalize drift before pydantic enforces ranges.
    Unknown domains fall back to ``Domain.OTHER``; unparseable
    confidences fall back to ``0.5``.
    """

    domain: str = Field(
        default=Domain.OTHER.value,
        # Enumerated inline (derived from the enum) because the field is
        # wire-typed as ``str`` and the enum is absent from the JSON schema the
        # model sees; the description is its only source of valid values.
        description=("One of: " + ", ".join(d.value for d in Domain) + ". Unknown values coerce to OTHER."),
    )
    domain_confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("domain", mode="before")
    @classmethod
    def _normalize_domain(cls, v: object) -> str:
        if v is None or not isinstance(v, str) or not v.strip():
            return Domain.OTHER.value
        cleaned = v.strip().upper().replace(" ", "_").replace("-", "_")
        allowed = {d.value for d in Domain}
        if cleaned in allowed:
            return cleaned
        # Substring match — pick first Domain that appears as substring.
        for d in Domain:
            if d.value in cleaned or cleaned in d.value:
                return d.value
        return Domain.OTHER.value

    @field_validator("domain_confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: object) -> float:
        """Accept "0.95", "85%", or numeric input; clamp to [0, 1]."""
        if isinstance(v, bool):
            return 0.5
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        if isinstance(v, str):
            try:
                raw = v.strip().rstrip("%")
                val = float(raw)
                if "%" in v:
                    val /= 100.0
                return max(0.0, min(1.0, val))
            except (ValueError, TypeError):
                return 0.5
        return 0.5


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


class CombinedRiskLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


# ---------------------------------------------------------------------------
# Entity-label -> EntityCategory mapping.
#
# Used by the disposition reconstructor (engine/rewrite/disposition_derivation.py)
# when the disposition LLM outputs an entity_label string in the ``category``
# slot — observed consistently with small Gemma models. Derived from two
# frozensets so the source of truth lives in one place per category.
# ``test_entity_label_to_category_covers_default_labels`` is a CI guard that
# fires when a label is added to ``DEFAULT_ENTITY_LABELS`` without a category
# assignment here. Any label not in this table falls back to
# ``"quasi_identifier"`` which is the most conservative (protect-cautiously)
# choice.
#
# Note: the original PR carried a third ``_SENSITIVE_ATTR_LABELS`` set
# mapping to a ``sensitive_attribute`` EntityCategory, but that enum value
# was removed when sensitivity disposition was recalibrated (#150) — the
# six former-sensitive labels (gender, sexuality, race_ethnicity,
# religious_belief, political_view, blood_type) are now folded into
# ``_QUASI_ID_LABELS`` (the conservative protect-cautiously choice given
# the now-3-value EntityCategory enum).
# ---------------------------------------------------------------------------

_DIRECT_ID_LABELS: frozenset[str] = frozenset(
    {
        "first_name",
        "last_name",
        "email",
        "phone_number",
        "fax_number",
        "ssn",
        "national_id",
        "street_address",
        "postcode",
        "credit_debit_card",
        "account_number",
        "bank_routing_number",
        "tax_id",
        "medical_record_number",
        "health_plan_beneficiary_number",
        "api_key",
        "password",
        "ipv4",
        "ipv6",
        "mac_address",
        "url",
        "user_name",
        "employee_id",
        "customer_id",
        "unique_id",
        "biometric_identifier",
        "device_identifier",
        "license_plate",
        "vehicle_identifier",
        "swift_bic",
        "pin",
        "cvv",
        "http_cookie",
    }
)
_QUASI_ID_LABELS: frozenset[str] = frozenset(
    {
        "age",
        "date",
        "date_of_birth",
        "date_time",
        "time",
        "city",
        "state",
        "country",
        "county",
        "place_name",
        "landmark",
        "coordinate",
        "occupation",
        "organization_name",
        "company_name",
        "university",
        "court_name",
        "prison_detention_facility",
        "degree",
        "field_of_study",
        "education_level",
        "language",
        "nationality",
        "employment_status",
        "monetary_amount",
        "certificate_license_number",
        # Former-sensitive labels (#150 collapsed sensitive_attribute into
        # quasi_identifier when EntityCategory was reduced to 3 values).
        "gender",
        "sexuality",
        "race_ethnicity",
        "religious_belief",
        "political_view",
        "blood_type",
    }
)

_ENTITY_LABEL_TO_CATEGORY: dict[str, str] = {lbl: "direct_identifier" for lbl in _DIRECT_ID_LABELS} | {
    lbl: "quasi_identifier" for lbl in _QUASI_ID_LABELS
}


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
    # No length constraints: this schema is reconstructed server-side from
    # trusted entity context, never directly from raw model output. The
    # reconstructor guarantees a non-empty templated protection_reason and
    # caps its length, so length bounds here would only be redundant tripwires
    # that risk dropping a record when a passthrough reason runs long.
    entity_label: str
    entity_value: str
    protection_reason: str
    protection_method_suggestion: ProtectionMethod
    combined_risk_level: CombinedRiskLevel

    @property
    def needs_protection(self) -> bool:
        return self.protection_method_suggestion != ProtectionMethod.leave_as_is

    @model_validator(mode="after")
    def _validate_protection_consistency(self) -> EntityDispositionSchema:
        if (
            self.combined_risk_level == CombinedRiskLevel.low
            and self.protection_method_suggestion != ProtectionMethod.leave_as_is
        ):
            raise ValueError(
                f"Entity {self.id}: combined_risk_level='low' requires protection_method_suggestion='leave_as_is', "
                f"got '{self.protection_method_suggestion}'"
            )
        if (
            self.combined_risk_level == CombinedRiskLevel.high
            and self.protection_method_suggestion == ProtectionMethod.leave_as_is
        ):
            raise ValueError(
                f"Entity {self.id}: combined_risk_level='high' cannot have protection_method_suggestion='leave_as_is'"
            )
        return self


class SensitivityDispositionSchema(BaseModel):
    """Complete sensitivity disposition for a document — LLM output schema.

    Validates that entity IDs are sequential from 1 and that each entity's
    ``protection_method_suggestion`` is consistent with its ``combined_risk_level``.

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


class StrictCombinedRiskLevel(str, Enum):
    medium = "medium"
    high = "high"


class StrictEntityDispositionSchema(EntityDispositionSchema):
    """Strict variant: leave_as_is and low combined_risk_level are excluded."""

    protection_method_suggestion: StrictProtectionMethod
    combined_risk_level: StrictCombinedRiskLevel


class StrictSensitivityDispositionSchema(SensitivityDispositionSchema):
    """Strict variant container: every entity must be protected."""

    sensitivity_disposition: list[StrictEntityDispositionSchema] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Loose wire-contract schemas for disposition (small-model tolerance)
#
# Used as the ``output_format`` for the disposition_analyzer LLM column. A
# server-side reconstruction column (see
# ``engine/rewrite/disposition_derivation.py``) pairs these with the entity
# context columns to produce the strict ``EntityDispositionSchema`` that
# downstream consumers read.
# ---------------------------------------------------------------------------


class SimpleDispositionItem(BaseModel):
    """Loose wire-contract shape for one disposition decision from the LLM.

    Why "loose": every field is typed as ``str`` (not the corresponding
    enum) and has a permissive default. This keeps the emitted JSON
    Schema free of ``enum``, ``required``, and ``minLength`` constraints
    that DataDesigner's ``jsonschema.validate()`` runs BEFORE pydantic's
    coercion. Small models drift at those constraints; the loose wire
    gate lets drifted output survive to the server-side reconstruction.
    """

    id: int = Field(ge=1)
    # Echoed from the entity context for belt-and-braces pairing. Optional
    # so the server can fall back to id-based lookup if the model omits.
    source: str = Field(default="")
    entity_label: str = Field(default="")
    entity_value: str = Field(default="")
    # LLM judgments; typed str so enum drift ("latent_sensitive_attribute",
    # "DIRECT IDENTIFIER", etc.) is accepted at the wire layer and
    # normalized during reconstruction. Valid values are enumerated inline
    # (derived from the enums) since the wire JSON schema carries no enum.
    category: str = Field(
        default="",
        description="One of: " + ", ".join(c.value for c in EntityCategory) + ".",
    )
    sensitivity: str = Field(
        default="",
        description="One of: " + ", ".join(s.value for s in SensitivityLevel) + ".",
    )
    protection_method_suggestion: str = Field(
        default="",
        description="One of: " + ", ".join(m.value for m in ProtectionMethod) + ".",
    )
    # Optional: when the model emits a document-specific rationale we keep
    # it verbatim; otherwise the reconstructor templates one from
    # (category, method, sensitivity).
    protection_reason: str = Field(default="")

    @field_validator(
        "source",
        "entity_label",
        "entity_value",
        "category",
        "sensitivity",
        "protection_method_suggestion",
        "protection_reason",
        mode="before",
    )
    @classmethod
    def _coerce_scalar_to_str(cls, v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, (int, float, bool)):
            return str(v)
        return v


class SimpleDispositionResult(BaseModel):
    """Wire-contract wrapper around a list of ``SimpleDispositionItem``.

    Tolerates two LLM output shapes at the wire layer:

    1. Canonical wrapper: ``{"sensitivity_disposition": [item, ...]}``
    2. Bare list at the top level: ``[item, ...]`` — observed
       consistently on ``nemotron-3-nano:4b`` (rewrite mode, 5/5 records)
       and intermittently on Gemma4-edge models for dense entity sets.

    The fix is two-layered (see ``schemas/shared.py``):

    - ``__get_pydantic_json_schema__`` widens the emitted JSON Schema to
      a ``oneOf`` of {wrapper-object, bare-array} so DD's
      ``jsonschema.validate()`` pre-check accepts both.
    - ``_accept_bare_list`` (mode="before") normalizes the bare-list
      shape to the wrapper dict so downstream consumers continue to read
      the canonical ``sensitivity_disposition`` field.
    """

    sensitivity_disposition: list[SimpleDispositionItem] = Field(default_factory=list)

    _accept_bare_list = model_validator(mode="before")(accept_bare_list(list_field="sensitivity_disposition"))

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        return loose_list_wrapper_json_schema(handler, schema, list_field="sensitivity_disposition")


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


class MeaningUnitImportance(str, Enum):
    critical = "critical"
    important = "important"


class MeaningUnitSchema(BaseModel):
    """Single meaning unit extracted by the meaning_extractor role.

    Loose wire: ``aspect`` and ``importance`` are typed as ``str`` (not
    the corresponding enum) so small-model enum drift (``"ROLE"`` vs
    ``"role"``, ``"the role"``, ``"crit"``, etc.) is accepted at the
    wire layer and normalized by before-validators. ``id`` defaults to 1
    so missing-id rows survive; ``MeaningUnitsSchema._ensure_list``
    re-numbers when ids collide.
    """

    id: int = Field(ge=1, default=1)
    aspect: str = Field(
        default="",
        # Enumerated inline (derived from the enum, not hand-maintained) because
        # the field is wire-typed as ``str``: the enum is absent from the JSON
        # schema the model sees, so the description is its only source of truth.
        description=("One of: " + ", ".join(a.value for a in MeaningUnitAspect) + ". Use the closest match."),
    )
    unit: str = Field(default="")
    importance: str = Field(
        default=MeaningUnitImportance.important.value,
        description=(
            "One of: " + ", ".join(i.value for i in MeaningUnitImportance) + ". Defaults to 'important' if unsure."
        ),
    )

    @field_validator("aspect", mode="before")
    @classmethod
    def _normalize_aspect(cls, v: object) -> str:
        if v is None or not isinstance(v, str) or not v.strip():
            return ""
        cleaned = v.strip().lower().replace(" ", "_").replace("-", "_")
        allowed = {a.value for a in MeaningUnitAspect}
        if cleaned in allowed:
            return cleaned
        for a in MeaningUnitAspect:
            if a.value in cleaned or cleaned in a.value:
                return a.value
        return ""

    @field_validator("importance", mode="before")
    @classmethod
    def _normalize_importance(cls, v: object) -> str:
        if v is None or not isinstance(v, str) or not v.strip():
            return MeaningUnitImportance.important.value
        cleaned = v.strip().lower()
        allowed = {a.value for a in MeaningUnitImportance}
        if cleaned in allowed:
            return cleaned
        for a in MeaningUnitImportance:
            if a.value in cleaned or cleaned in a.value:
                return a.value
        return MeaningUnitImportance.important.value

    @field_validator("unit", mode="before")
    @classmethod
    def _coerce_unit(cls, v: object) -> str:
        if v is None:
            return ""
        if isinstance(v, (int, float, bool)):
            return str(v)
        return v


class MeaningUnitsSchema(BaseModel):
    """LLM output schema for meaning unit extraction step.

    Outer list has ``default_factory=list`` (was ``min_length=1``); if
    the model emits an empty list the record still survives so the
    pipeline can decide what to do downstream.

    Tolerates two LLM output shapes at the wire layer (same pattern as
    ``SimpleDispositionResult``):

    1. Canonical wrapper: ``{"units": [item, ...]}``
    2. Bare list at the top level: ``[item, ...]`` — observed on
       qwen3.5:4b for legal-court documents.
    """

    units: list[MeaningUnitSchema] = Field(default_factory=list)

    _accept_bare_list = model_validator(mode="before")(accept_bare_list(list_field="units"))

    @classmethod
    def __get_pydantic_json_schema__(cls, schema, handler):
        return loose_list_wrapper_json_schema(handler, schema, list_field="units")

    @field_validator("units", mode="before")
    @classmethod
    def _ensure_list(cls, v: object) -> list:
        if not isinstance(v, list):
            v = [v] if isinstance(v, dict) else []
        # ``MeaningUnitSchema.id`` defaults to 1; if the LLM omits ids the
        # wire collapses every unit to id=1. Reassign sequentially when
        # any id is missing or duplicated. Explicit unique ids are kept.
        if isinstance(v, list) and v:
            raw_ids = [item.get("id") if isinstance(item, dict) else getattr(item, "id", None) for item in v]
            valid = [i for i in raw_ids if isinstance(i, int) and i >= 1]
            if len(valid) != len(raw_ids) or len(set(valid)) != len(valid):
                for idx, item in enumerate(v, start=1):
                    if isinstance(item, dict):
                        item["id"] = idx
                    elif hasattr(item, "id"):
                        item.id = idx  # type: ignore[misc]
        return v


# ---------------------------------------------------------------------------
# QA Generation
# ---------------------------------------------------------------------------


class QualityQAItemSchema(BaseModel):
    id: int
    aspect: str
    importance: MeaningUnitImportance
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


def _normalize_id_covered_list(
    raw: list,
    *,
    expected_ids: list[int] | None,
    default_item_for_id,
) -> list:
    """Normalize a list of id-bearing items to exact-coverage shape.

    Used as the wire-layer normalizer on all context-validated answer
    schemas (``QualityAnswersSchema`` / ``PrivacyAnswersSchema`` /
    ``QACompareResultsSchema``). Dedupes by id (first occurrence wins),
    pads missing ids via ``default_item_for_id(id) -> dict``, drops
    extras. When ``expected_ids`` is None, only dedupes.

    Each item can be a dict or a BaseModel instance. Items without a
    parseable ``id`` are skipped.
    """
    seen: set[int] = set()
    deduped: list = []
    for item in raw:
        if hasattr(item, "model_dump"):
            item = item.model_dump()
        if not isinstance(item, dict):
            continue
        try:
            iid = int(item.get("id"))
        except (TypeError, ValueError):
            continue
        if iid in seen:
            continue
        seen.add(iid)
        deduped.append(item)

    if expected_ids is None:
        return deduped

    expected_set = set(expected_ids)
    deduped = [it for it in deduped if int(it["id"]) in expected_set]
    present = {int(it["id"]) for it in deduped}
    for eid in expected_ids:
        if eid not in present:
            deduped.append(default_item_for_id(eid))
    order = {eid: i for i, eid in enumerate(expected_ids)}
    deduped.sort(key=lambda it: order.get(int(it["id"]), len(order)))
    return deduped


class QualityAnswerSchema(BaseModel):
    id: int
    answer: str


class QualityAnswersSchema(BaseModel):
    """LLM output schema for quality QA re-answer step (on rewritten text).

    When validated with ``context={"expected_ids": [1, 2, ...]}``,
    normalizes the returned list to exactly that ID set: dedupes by id
    (first wins), pads missing ids with a placeholder ``"missing"``
    answer, and drops extras. This prevents an LLM that emits a
    duplicate or misses an id from dropping the whole record (observed
    on gpt-oss-20b × 05_legal_court rewrite, qwen3.5:9b on 4-entity
    notes).
    """

    answers: list[QualityAnswerSchema] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_answers(cls, data: object, info: ValidationInfo) -> object:
        if not isinstance(data, dict):
            return data
        raw = data.get("answers") or []
        if not isinstance(raw, list):
            raw = []
        expected = (info.context or {}).get("expected_ids") if info.context else None
        data["answers"] = _normalize_id_covered_list(
            raw,
            expected_ids=expected,
            default_item_for_id=lambda i: {"id": i, "answer": "missing"},
        )
        return data

    @model_validator(mode="after")
    def _check_coverage(self, info: ValidationInfo) -> QualityAnswersSchema:
        # Soft check: the before-validator already normalized. If coverage
        # still fails here it indicates a schema-level bug, not LLM drift.
        expected_ids = (info.context or {}).get("expected_ids")
        if expected_ids is not None:
            try:
                _validate_id_coverage(expected_ids, [a.id for a in self.answers], "answer")
            except ValueError as e:
                import logging

                logging.getLogger(__name__).warning("QualityAnswersSchema post-normalization coverage warning: %s", e)
        return self


class PrivacyAnswerItemSchema(BaseModel):
    id: int
    answer: PrivacyAnswer
    confidence: float = Field(ge=0.0, le=1.0)
    # No length constraints: _truncate_reason (below) already forces every
    # value into a non-empty, <=200-char envelope before validation, so a
    # min_length/max_length here could never trip and only adds drift surface.
    reason: str
    evidence: list[str] = Field(default_factory=list)

    @field_validator("reason", mode="before")
    @classmethod
    def _truncate_reason(cls, v: object) -> object:
        """Coerce small-model reason drift into a non-empty, <=200-char
        envelope rather than dropping the record. (The field no longer
        carries min_length/max_length constraints; this validator is the
        sole guard, so it always normalizes into range.)

        Three observed drift modes:

        - 250+ char prose (nemotron-3-nano on vLLM) → truncated to 197
          chars + "..."
        - ``None`` (some models omit the field on ``answer="no"``) →
          placeholder "no reason provided"
        - Empty / whitespace-only string → placeholder (would otherwise
          fail ``min_length=1``)
        """
        if isinstance(v, str) and len(v) > 200:
            return v[:197].rstrip() + "..."
        if v is None or (isinstance(v, str) and not v.strip()):
            return "no reason provided"
        return v


class PrivacyAnswersSchema(BaseModel):
    """LLM output schema for privacy QA re-answer step (on rewritten text).

    When validated with ``context={"expected_ids": [1, 2, ...]}``,
    normalizes the returned list to exactly that ID set: dedupes by id
    (first wins), pads missing ids with a pessimistic answer
    (``"yes"`` = assume leak), and drops extras. Pessimistic default
    because a missing answer should bias toward triggering human review
    rather than silently passing.
    """

    answers: list[PrivacyAnswerItemSchema] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_answers(cls, data: object, info: ValidationInfo) -> object:
        if not isinstance(data, dict):
            return data
        raw = data.get("answers") or []
        if not isinstance(raw, list):
            raw = []
        expected = (info.context or {}).get("expected_ids") if info.context else None
        data["answers"] = _normalize_id_covered_list(
            raw,
            expected_ids=expected,
            default_item_for_id=lambda i: {
                "id": i,
                "answer": "yes",  # pessimistic default — flag for human review
                "confidence": 0.5,
                "reason": "missing answer - defaulted to pessimistic",
            },
        )
        return data

    @model_validator(mode="after")
    def _check_coverage(self, info: ValidationInfo) -> PrivacyAnswersSchema:
        expected_ids = (info.context or {}).get("expected_ids")
        if expected_ids is not None:
            try:
                _validate_id_coverage(expected_ids, [a.id for a in self.answers], "answer")
            except ValueError as e:
                import logging

                logging.getLogger(__name__).warning("PrivacyAnswersSchema post-normalization coverage warning: %s", e)
        return self


class QACompareItemSchema(BaseModel):
    id: int
    score: float = Field(ge=0.0, le=1.0)
    reason: str | None = None


class QACompareResultsSchema(BaseModel):
    """LLM output schema for quality QA comparison step.

    When validated with ``context={"expected_ids": [1, 2, ...]}``,
    normalizes the returned list to exactly that ID set: dedupes, pads
    missing ids with a neutral 0.5 score, and drops extras.
    """

    per_item: list[QACompareItemSchema] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_per_item(cls, data: object, info: ValidationInfo) -> object:
        if not isinstance(data, dict):
            return data
        raw = data.get("per_item") or []
        if not isinstance(raw, list):
            raw = []
        expected = (info.context or {}).get("expected_ids") if info.context else None
        data["per_item"] = _normalize_id_covered_list(
            raw,
            expected_ids=expected,
            default_item_for_id=lambda i: {"id": i, "score": 0.5, "reason": None},
        )
        return data

    @model_validator(mode="after")
    def _check_coverage(self, info: ValidationInfo) -> QACompareResultsSchema:
        expected_ids = (info.context or {}).get("expected_ids")
        if expected_ids is not None:
            try:
                _validate_id_coverage(expected_ids, [a.id for a in self.per_item], "compare")
            except ValueError as e:
                import logging

                logging.getLogger(__name__).warning("QACompareResultsSchema post-normalization coverage warning: %s", e)
        return self
