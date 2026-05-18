# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAGGED_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.schemas import SensitivityDispositionSchema, StrictSensitivityDispositionSchema


def _get_sensitivity_disposition_prompt(
    privacy_goal: PrivacyGoal, data_summary: str | None = None, strict_entity_protection: bool = False
) -> str:
    privacy_goal_str = privacy_goal.to_prompt_string()
    # TODO: align entity detection prompts (validation, augment, latent) to use "Dataset description:" label
    data_summary_line = (
        f"\nDataset description: {data_summary.strip()}" if data_summary and data_summary.strip() else ""
    )

    strict_protection_block = ""
    if strict_entity_protection:
        strict_protection_block = """
<strict_entity_protection>
STRICT PROTECTION MODE IS ENABLED.

All entities MUST be protected — you only decide HOW. Choose the most appropriate
protection_method_suggestion for each entity. leave_as_is is not available.
combined_risk_level must be 'high' or 'medium'. 'low' is not valid in strict mode
because it implies no protection is needed.

Ignore the MINIMUM NECESSARY CHANGE principle — it does not apply in strict mode.
Ignore the QUASI-IDENTIFIERS guidance that says "Not automatically protected" —
all quasi-identifiers must be protected in strict mode.

</strict_entity_protection>
"""

    prompt = """You are responsible for creating a unified sensitivity disposition for privacy-preserving text rewriting.

Your task is to analyze ALL entities in the text and produce a structured protection plan that specifies:
- Which entities need protection.
- Why they need (or don't need) protection.
- How they should be protected.

Focus on decisions grounded in the specific document and threat model. Do NOT rewrite the text itself.

<privacy_goal>
<<PRIVACY_GOAL>>
</privacy_goal>

<domain_context>
Domain: <<DOMAIN>><<DATA_SUMMARY>>

Domain-Specific Preservation Requirements:
---
<<DOMAIN_SUPPLEMENT>>
---
</domain_context>

<input_tagged_text>
The text below has explicit entities tagged inline using this format: ⟦entity_value|entity_label⟧

Example:
  "A 29-year-old male from Portland" becomes:
  "A ⟦29-year-old|age⟧ ⟦male|gender⟧ from ⟦Portland|city⟧"

Rules for interpreting tags:
- The substring BEFORE the "|" is the entity_value (record EXACTLY as shown).
- The substring AFTER the "|" is the entity_label.
- Do NOT include the brackets ⟦ ⟧ in entity_value.

Tagged Text:
---
<<TAGGED_TEXT>>
---
</input_tagged_text>

<input_detected_entities>
Entities detected and validated from the text:
<<FINAL_ENTITIES>>
</input_detected_entities>

<input_latent_entities>
Latent entities are NOT explicitly stated but can be inferred by an adversary from context.
Each has: label, value, confidence, evidence, rationale.

Latent Entities:
<<LATENT_ENTITIES>>
</input_latent_entities>

<threat_model>
Assume a realistic re-identification adversary who may:
- Access the rewritten document in full.
- Possess general domain knowledge and publicly available information.
- In some cases, have partial prior familiarity with the subject (professional/social context).

Plan changes to prevent both:
- **Public-linkage re-identification**: Linking to public records or online information.
- **Recipient-recognition**: Recognition by someone who plausibly knows the individual.

Do NOT assume the adversary knows the original text or internal annotations.

Re-identification is successful if the adversary can reasonably narrow identity to a small, plausible set.
</threat_model>

<<STRICT_PROTECTION_BLOCK>>
<entity_categories>
Assign sensitivity based on re-identification risk only — not on the nature of the content itself.
Attributes like religion, political views, or sexual orientation may require protection,
but their sensitivity level is set by re-identification risk alone.

DIRECT IDENTIFIERS — sensitivity: always high
  Uniquely identify an individual on their own.
  Examples: full name, email, phone number, SSN, exact address, full date of birth,
  account number, medical record number, national ID, tax ID.
  Note: full date of birth (month + day + year) qualifies. A year-only or decade
  reference is a quasi-identifier, not a direct identifier.

  Default protection: replace with a plausible synthetic alternative.
  Use generalization or removal only if replacement distorts meaning.

QUASI-IDENTIFIERS — sensitivity: high, medium, or low
  Not identifying alone, but narrowing in combination with other known facts.
  Examples: age, city, occupation, employer, gender, nationality, education,
  marital status, religion, political view, sexual orientation.

  Sensitivity assignment:
  Assign high when the entity value is itself so rare or distinctive that it
  substantially narrows identity even without other context (e.g. a very small
  ethnic group, an uncommon birthplace, a highly specific institutional role).
  Assign medium when this specific value, in this document, meaningfully narrows
  the candidate set — not merely because the entity type is generally quasi-identifying.
  Assign low when this specific value adds little narrowing in context, even if
  the entity type is generally quasi-identifying (e.g., a nationality obvious from
  the respondent state).

  Default protection:
  Not automatically protected: modify only if the combination increases re-identification risk.
  When modification is needed: prefer generalization over removal.

LATENT IDENTIFIERS — sensitivity: high, medium, or low
  Inferred from context rather than explicitly stated.
  Apply the same re-identification risk logic as quasi-identifiers.
  Mitigation requires paraphrasing, removing supporting details, or generalizing facts.
  Replace is rarely appropriate since the value is not explicitly in the text.
</entity_categories>

<combined_risk_assessment>
combined_risk_level reflects how dangerous this entity is given what else is being
retained in the output — not its intrinsic identifying power alone.

sensitivity and combined_risk_level serve different purposes:
- sensitivity measures how much damage this entity does if it leaks (feeds leakage scoring)
- combined_risk_level determines whether to protect it in the first place

The protection decision follows from combined_risk_level, not sensitivity:
- combined_risk_level: high → must be protected (protection_method_suggestion must not be "leave_as_is")
- combined_risk_level: medium → protect only if the entity meaningfully
  contributes to a dangerous combination that cannot be broken by protecting other entities
- combined_risk_level: low → leave as-is (protection_method_suggestion must be "leave_as_is")

combined_risk_level can exceed sensitivity when context amplifies an otherwise weak entity:
  e.g. male gender (sensitivity: low) becomes combined_risk_level: high if it is the
  final disambiguating fact in an otherwise nearly-identifying bundle.

combined_risk_level should not exceed medium for an entity that adds little narrowing
regardless of what surrounds it. If protecting stronger anchors in the same document
would break the combination without touching this entity, its combined_risk_level is low.

When multiple entities together form an identifying bundle, do not assign
combined_risk_level: medium to every element because they "form a combination."
Identify the load-bearing element — the one without which the bundle is no longer
identifying — and assign that combined_risk_level: high. Other elements that only
add narrowing because of that anchor are combined_risk_level: low once the anchor
is protected. Assign medium only if an entity contributes meaningful residual
narrowing after all high combined_risk_level anchors in this document are suppressed.
</combined_risk_assessment>

<protection_principles>
1. MINIMUM NECESSARY CHANGE: If a detail doesn't meaningfully increase re-identification risk
   and isn't a sensitive attribute, leave it unchanged.
   EXCEPTION: Always replace the names of people regardless of re-identification risk.

2. CONTEXTUAL EVALUATION: A tag does NOT automatically require action. Every change must be
   justified by contextual privacy risk under the threat model.

3. COMBINED RISK ASSESSMENT: Consider how retained entities combine. Three "safe" quasi-identifiers
   together may be identifying.

4. PRESERVE SEMANTIC FIDELITY: Maintain narrative flow, causal structure, and referential clarity.
   Avoid unnecessary abstraction.

5. DOMAIN-SPECIFIC PRESERVATION: Apply the domain guidance above—preserve what matters for utility.

</protection_principles>
<output_requirements>
CONSISTENCY RULES:
- If combined_risk_level='low' → protection_method_suggestion MUST be "leave_as_is".
- If combined_risk_level='high' → protection_method_suggestion MUST NOT be "leave_as_is".
- For latent entities, "replace" is rarely appropriate (value not in text).
- For source="tagged": entity_value MUST match tag exactly.
- For source="latent": entity_label/value MUST match the provided latent entity.

COVERAGE REQUIREMENTS:
- Include ONE entry for EVERY unique listed entity
- Include ONE entry for EVERY provided latent entity.
- IDs must be sequential starting from 1.

QUALITY REQUIREMENTS:
- protection_reason must be specific to this document and must cover:
  (1) what specific combination or narrowing this entity contributes in this document,
  (2) for combined_risk_level='low', why the combination is adequately broken without it.
- combined_risk_level must reflect the entity's risk given other retained entities,
  not sensitivity.
- combined_risk_level: low requires protection_method_suggestion='leave_as_is' unless
  it is part of a bundle where all elements must be generalized together to break the
  combination, in which case combined_risk_level must be high, not low.
- combined_risk_level: medium requires protection_reason to name which retained entities
  form the dangerous combination AND explain why
  protecting a stronger anchor in this document would not break the combination
  without touching this entity. "Stronger anchors" means all entities with
  combined_risk_level: high being protected in this document — not just names.
  The reason must show the combination persists even after all high combined_risk_level
  entities are suppressed.
</output_requirements>"""
    return substitute_placeholders(
        prompt,
        {
            "<<PRIVACY_GOAL>>": privacy_goal_str,
            "<<DOMAIN>>": _jinja(COL_DOMAIN, key="domain"),
            "<<DATA_SUMMARY>>": data_summary_line,
            "<<DOMAIN_SUPPLEMENT>>": _jinja(COL_DOMAIN_SUPPLEMENT_PRIVACY),
            "<<TAGGED_TEXT>>": _jinja(COL_TAGGED_TEXT),
            "<<FINAL_ENTITIES>>": _jinja(COL_ENTITIES_BY_VALUE),
            "<<LATENT_ENTITIES>>": _jinja(COL_LATENT_ENTITIES),
            "<<STRICT_PROTECTION_BLOCK>>": strict_protection_block,
        },
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class SensitivityDispositionWorkflow:
    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        data_summary: str | None = None,
        strict_entity_protection: bool = False,
    ) -> list[ColumnConfigT]:
        disposition_alias = resolve_model_alias("disposition_analyzer", selected_models)
        output_schema = StrictSensitivityDispositionSchema if strict_entity_protection else SensitivityDispositionSchema
        return [
            LLMStructuredColumnConfig(
                name=COL_SENSITIVITY_DISPOSITION,
                prompt=_get_sensitivity_disposition_prompt(
                    privacy_goal,
                    data_summary,
                    strict_entity_protection=strict_entity_protection,
                ),
                model_alias=disposition_alias,
                output_format=output_schema,
            ),
        ]
