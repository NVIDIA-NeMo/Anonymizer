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
_DOMAIN_KEY = next(name for name, info in DomainClassificationSchema.model_fields.items() if info.annotation is Domain)

# ---------------------------------------------------------------------------
# Stage 1 pre-step: format disposition → disposition block
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_SENSITIVITY_DISPOSITION])
def _format_disposition_block(row: dict[str, Any]) -> dict[str, Any]:
    """Serialize sensitivity disposition into a JSON block for the meaning unit extraction prompt."""
    disposition: SensitivityDispositionSchema = row[COL_SENSITIVITY_DISPOSITION]
    block = [
        {
            "entity_value": e.entity_value,
            "needs_protection": e.needs_protection,
            "protection_method": e.protection_method_suggestion,  # shortened key for prompt brevity
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
Your job is to capture the core informational content that MUST be preserved for the text to remain
useful, while avoiding unsafe disclosure of identifying details.

<entity_protection_rules>
You are given an ENTITY PROTECTION BLOCK — a JSON array where each entity has an "entity_value" string
(e.g., a name, address, occupation, city, diagnosis). Treat every entity_value where needs_protection=true
as a BANNED SURFACE FORM:

- None of these entity_value strings may appear anywhere in any "unit".
- If a unit would require using one of these exact strings, you MUST either:
  - rewrite it in more abstract terms that do NOT contain that value, OR
  - DROP that unit if it cannot be expressed safely.

Apply this by type:

1. DIRECT IDENTIFIERS (names, exact addresses, exact ages, record numbers)
   - NEVER include them.
   - Keep only abstract roles or relations (e.g., "the individual",
     "a close family member", "a colleague").
   - If the idea is only interesting because of the specific identity,
     drop it.

2. QUASI-IDENTIFIERS (occupation, city, school, degree, event names, etc.)
   - Keep the KIND of information, but generalize so the original value
     is not recoverable.
   - DO NOT repeat the exact value text from the sensitive block.
   - Use broader or paraphrased categories (e.g., "a medical specialist",
     "a higher-education program", "a local community event", "a city").

3. LATENT / HIGHLY SENSITIVE ATTRIBUTES (medical_condition, family_death,
   religion, etc.)
   - If not essential to the meaning, DROP them.
   - If essential (e.g., motivation, history), describe them in softened,
     high-level form (e.g., "a health difficulty", "a personal loss")
     and NEVER use the exact value string.

4. GENERALIZATION AND SAFETY
   - Prefer abstract phrases like "the individual", "a family member",
     "a local area", "a professional setting", "a long-term challenge".
   - Avoid precise times, counts, and locations when they combine with
     other details to make someone easy to identify.
   - When in doubt, DROP the detail rather than risk leaking a sensitive value.
</entity_protection_rules>

<importance_criteria>
KEEP a detail if removing it would significantly weaken understanding of:
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

Domain-specific guidance:
---
<<DOMAIN_SUPPLEMENT>>
---
</domain_context>

<input>
Entity protection block:
---
<<ENTITY_PROTECTION_BLOCK>>
---

Original text:
---
<<TEXT>>
---
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
    meaning_units: MeaningUnitsSchema = row[COL_MEANING_UNITS]
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
    disposition: SensitivityDispositionSchema = row[COL_SENSITIVITY_DISPOSITION]
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
