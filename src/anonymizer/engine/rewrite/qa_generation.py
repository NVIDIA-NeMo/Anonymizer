# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import RewriteModelSelection
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT,
    COL_MEANING_UNITS,
    COL_MEANING_UNITS_SERIALIZED,
    COL_PRIVACY_QA,
    COL_QUALITY_QA,
    COL_SENSITIVITY_DISPOSITION,
    COL_SENSITIVITY_DISPOSITION_BLOCK,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.schemas import (
    Domain,
    DomainClassificationSchema,
    EntityCategory,
    MeaningUnitsSchema,
    PrivacyQAPairsSchema,
    PrivacyQuestionSchema,
    QualityQAPairsSchema,
    SensitivityDispositionSchema,
    SensitivityLevel,
)

# Derived from the schema so the Jinja key stays in sync with the field name.
_DOMAIN_KEY = next(
    (name for name, info in DomainClassificationSchema.model_fields.items() if info.annotation is Domain),
    None,
)
if _DOMAIN_KEY is None:
    raise RuntimeError("DomainClassificationSchema must define a field annotated with Domain")

# ---------------------------------------------------------------------------
# Stage 1 pre-step: format disposition → disposition block
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_SENSITIVITY_DISPOSITION])
def _format_disposition_block(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize sensitivity disposition into a JSON block for the meaning unit extraction prompt."""
    disposition = SensitivityDispositionSchema.model_validate(row.get(COL_SENSITIVITY_DISPOSITION, {}))
    block = [
        {
            "entity_value": e.entity_value,
            "does_need_protection": e.needs_protection,
            "protection_method_suggestion": e.protection_method_suggestion,
            "category": e.category,
        }
        for e in disposition.sensitivity_disposition
    ]
    row[COL_SENSITIVITY_DISPOSITION_BLOCK] = json.dumps(block, ensure_ascii=False)
    return row


# ---------------------------------------------------------------------------
# Stage 1 prompt: meaning unit extraction
# ---------------------------------------------------------------------------


def _get_meaning_unit_extraction_prompt() -> str:
    prompt = """You are extracting IMPORTANT MEANING UNITS from a text to evaluate anonymization quality.
The original text may contain sensitive entities; a separate system will rewrite it.
Your job is to capture the core informational content that MUST be preserved for the text to remain useful,
while avoiding unsafe disclosure of identifying details.

You are given:
- The ORIGINAL TEXT.
- A SENSITIVITY DISPOSITION BLOCK (entities with protection decisions).
- A PRIMARY DOMAIN and DOMAIN-SPECIFIC GUIDANCE.

Your output will be a JSON array of meaning units. Each unit is a short statement
that captures ONE important fact, process, value, or relationship that should
remain true after anonymization.

<entity_protection_rules>
You are given a SENSITIVITY DISPOSITION BLOCK, which contains entries like:
- entity_value (surface form from the original text)
- does_need_protection (True/False)
- protection_method_suggestion (replace/remove/generalize/paraphrase/left_as_is)
- category (direct_identifier/quasi_identifier/sensitive_attribute/latent_identifier/etc.)

Use it as follows:

A) HARD BANS (must not appear in meaning units)
If an entry has:
  - does_need_protection = True
  AND protection_method_suggestion is "replace" OR "remove"
Then treat its entity_value as a BANNED SURFACE FORM:
  - Do NOT include that exact string anywhere in any unit.
  - Also do NOT include near-verbatim variants that obviously preserve the same identifier.
  - If the underlying fact is important, express it WITHOUT that value using abstraction
    (roles, relationships, high-level descriptions).
  - If it cannot be expressed safely without carrying identifying detail, DROP the unit.

B) TRANSFORM-ALLOWED (allowed only if generalized/paraphrased)
If an entry has:
  - does_need_protection = True
  AND protection_method_suggestion is "generalize" OR "paraphrase"
Then you MAY still capture the meaning, BUT you must NOT use the entity_value itself.
Instead: preserve the semantic role while moving to a broader, less identifying level of abstraction.
This may include:

  • Geographic hierarchy: city → state → region → country
  • Institutional hierarchy: named organization → organization type
  • Role hierarchy: specific specialty → broader profession
  • Temporal abstraction: exact date → approximate period
  • Quantitative abstraction: exact number → rough scale
  • Named program/product → generic descriptive category

The generalized phrasing must prevent recovery or lookup of the original entity_value while
still preserving the meaning needed for usefulness.

C) SAFE / LEFT-AS-IS (no special avoidance required)
If an entry has:
  - does_need_protection = False
Then you do NOT need to avoid it for privacy reasons in meaning units.
However:
  - still prefer concise phrasing and avoid unnecessary precision
  - avoid adding extra identifying detail beyond what is needed for meaning
</entity_protection_rules>

<importance_criteria>
KEEP a detail as a meaning unit if removing it would significantly weaken
understanding of:
- The main roles or functions of the key actors.
- Core processes, workflows, decisions, or events.
- Key outputs, products, or creative/technical results.
- Important relationships, obligations, or dependencies.
- Central values, beliefs, or motivations that clearly drive behavior.
- Critical contextual constraints (e.g., high-level timelines, key conditions).

DROP details that are:
- Purely decorative, anecdotal, or atmospheric.
- Redundant restatements of a previously captured unit.
- Hyper-local or overly specific identifiers that don't affect the underlying logic or meaning.

Each meaning unit must contain only the information required to support a single quality-check question.
</importance_criteria>

<segmentation_rules>
- Each meaning unit must express ONE coherent idea.
- If a sentence encodes multiple important ideas, split them into separate units.
- If two sentences form a single coherent idea that cannot be separated without losing meaning,
  merge them into one unit.
</segmentation_rules>

<domain_context>
Primary domain: <<DOMAIN>>

DOMAIN-SPECIFIC GUIDANCE (CONTEXT):

<<DOMAIN_SUPPLEMENT>>

Use this guidance to decide what is "important" versus "nice-to-have" in this text.
For example:
- In LEGAL/POLICY, obligations, conditions, exceptions, and remedies are critical.
- In CLINICAL/EHR, symptoms, diagnoses, interventions, and timelines of care are critical.
- In TECHNICAL, inputs/outputs, requirements, constraints, and failure modes are critical.
- In CHAT/EMAIL/CSAT, the problem, key actions, decisions, and resolutions are critical.
But always apply this in context of the actual text, not blindly.
</domain_context>

<input>
Original text:
<<<
<<TEXT>>
>>>

Sensitivity disposition:
<<<
<<ENTITY_PROTECTION_BLOCK>>
>>>
</input>"""
    return (
        prompt.replace("<<ENTITY_PROTECTION_BLOCK>>", _jinja(COL_SENSITIVITY_DISPOSITION_BLOCK))
        .replace("<<DOMAIN>>", _jinja(COL_DOMAIN, key=_DOMAIN_KEY))
        .replace("<<DOMAIN_SUPPLEMENT>>", _jinja(COL_DOMAIN_SUPPLEMENT))
        .replace("<<TEXT>>", _jinja(COL_TEXT))
    )


# ---------------------------------------------------------------------------
# Stage 2 pre-step: serialize meaning units → JSON string
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_MEANING_UNITS])
def _serialize_meaning_units(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize MeaningUnitsSchema to a JSON string for injection into the QA generation prompt."""
    meaning_units = MeaningUnitsSchema.model_validate(row.get(COL_MEANING_UNITS, {}))
    row[COL_MEANING_UNITS_SERIALIZED] = json.dumps([u.model_dump() for u in meaning_units.units], ensure_ascii=False)
    return row


# ---------------------------------------------------------------------------
# Stage 2 prompt: quality QA generation from meaning units
# ---------------------------------------------------------------------------


def _get_quality_qa_prompt() -> str:
    prompt = """You are creating QUALITY-CHECK QUESTIONS for anonymized text.

You will be given a JSON array of MEANING UNITS that are already abstracted and PII-safe.
Each meaning unit has:
- "id": an integer identifier
- "aspect": a label describing the type of information
- "unit": a short sentence or clause capturing one important idea

Your task is to generate, for EACH meaning unit:
- ONE open-ended question that checks whether that unit is still true in an anonymized version of the text.
- ONE concise reference answer that matches the meaning unit.

<constraints>
1. Base questions and answers ONLY on the information present in the meaning units.
   Do NOT invent or recall any extra details not explicitly in the "unit" field.

2. Do NOT introduce any specific identifiers not present in the unit itself, including:
   - Personal names, family member names, or character names.
   - Street names, cities, regions, or institutions.
   - Program names, event names, branded titles, or other unique labels.
   If a unit uses abstract phrasing (e.g., "the individual", "a family member"), keep the
   same level of abstraction in the question and answer.

3. Questions should be specific enough to verify the meaning unit is preserved, but open-ended.
   - Avoid yes/no questions.
   - Start with: "what", "how", "why", "when", "where", or "in what way".

4. The reference answer should:
   - Be short and factual.
   - Align directly with the "unit" text at the same abstraction level.
   - NOT add new information beyond what is in the unit.
</constraints>

<input>
Meaning units:
---
<<MEANING_UNITS_SERIALIZED>>
---
</input>"""
    return prompt.replace("<<MEANING_UNITS_SERIALIZED>>", _jinja(COL_MEANING_UNITS_SERIALIZED))


# ---------------------------------------------------------------------------
# Stage 3: privacy QA generation (pure Python, no LLM)
# ---------------------------------------------------------------------------


def generate_privacy_qa_from_disposition(
    disposition: SensitivityDispositionSchema,
) -> PrivacyQAPairsSchema:
    """Generate privacy QA questions from sensitivity disposition without an LLM call.

    Produces one question per entity that needs protection:
        "Can the {entity_label} '{entity_value}' be deduced from the rewritten text?"
    All questions expect the answer "no" — a "yes" answer indicates a privacy leak.
    """
    questions = []
    for idx, entity in enumerate(disposition.protected_entities, start=1):
        questions.append(
            PrivacyQuestionSchema(
                id=idx,
                question=f"Can the {entity.entity_label} '{entity.entity_value}' be deduced from the rewritten text?",
                sensitivity=SensitivityLevel(entity.sensitivity),
                entity_label=entity.entity_label,
                entity_value=entity.entity_value,
                category=EntityCategory(entity.category),
            )
        )
    return PrivacyQAPairsSchema(items=questions)


@custom_column_generator(required_columns=[COL_SENSITIVITY_DISPOSITION])
def _generate_privacy_qa_column(row: dict[str, Any]) -> dict[str, Any]:
    """Generate privacy QA questions from sensitivity disposition without an LLM call."""
    disposition = SensitivityDispositionSchema.model_validate(row.get(COL_SENSITIVITY_DISPOSITION, {}))
    row[COL_PRIVACY_QA] = generate_privacy_qa_from_disposition(disposition)
    return row


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class QAGenerationWorkflow:
    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
    ) -> list[ColumnConfigT]:
        meaning_extractor_alias = resolve_model_alias("meaning_extractor", selected_models)
        qa_generator_alias = resolve_model_alias("qa_generator", selected_models)
        return [
            CustomColumnConfig(
                name=COL_SENSITIVITY_DISPOSITION_BLOCK,
                generator_function=_format_disposition_block,
            ),
            LLMStructuredColumnConfig(
                name=COL_MEANING_UNITS,
                prompt=_get_meaning_unit_extraction_prompt(),
                model_alias=meaning_extractor_alias,
                output_format=MeaningUnitsSchema,
            ),
            CustomColumnConfig(
                name=COL_MEANING_UNITS_SERIALIZED,
                generator_function=_serialize_meaning_units,
            ),
            LLMStructuredColumnConfig(
                name=COL_QUALITY_QA,
                prompt=_get_quality_qa_prompt(),
                model_alias=qa_generator_alias,
                output_format=QualityQAPairsSchema,
            ),
            CustomColumnConfig(
                name=COL_PRIVACY_QA,
                generator_function=_generate_privacy_qa_column,
            ),
        ]
