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
from anonymizer.engine.schemas import SensitivityDispositionSchema


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

needs_protection MUST be True for every entity, no exceptions.
Do NOT output needs_protection=False under any circumstances.

Ignore the MINIMUM NECESSARY CHANGE principle — it does not apply in strict mode.
Ignore the QUASI-IDENTIFIERS guidance that says "NOT automatically sensitive" —
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
DIRECT IDENTIFIERS (high-risk, standalone)
- Uniquely identify on their own: full names, exact addresses, SSNs, email, phone.
- Default action: REPLACE with plausible synthetic alternative.
- Use abstraction/removal only if replacement distorts meaning.

QUASI-IDENTIFIERS (context-dependent)
- Not unique alone, but identifying in combination: age, city, occupation, dates, institutions.
- NOT automatically sensitive: modify ONLY if combination increases re-identification risk.
- When modification needed: prefer generalization over removal.

SENSITIVE ATTRIBUTES (harmful to disclose)
- Sensitive because of content, not identifiability: health conditions, sexual orientation,
  mental health, substance use, legal issues, political/religious views.
- Protect when disclosure itself is harmful, even if identifiability risk is low.
- Can ALSO function as quasi-identifiers if distinctive in context.

LATENT IDENTIFIERS (inferred)
- Can be deduced from context even if not explicitly stated.
- Mitigation may require: paraphrasing context, removing supporting details, generalizing facts.
- Usually cannot use "replace" since value isn't explicitly in text.
</entity_categories>

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
- If needs_protection=false → protection_method_suggestion MUST be "leave_as_is".
- If needs_protection=true → protection_method_suggestion MUST NOT be "leave_as_is".
- For latent entities, "replace" is rarely appropriate (value not in text).
- For source="tagged": entity_value MUST match tag exactly.
- For source="latent": entity_label/value MUST match the provided latent entity.

COVERAGE REQUIREMENTS:
- Include ONE entry for EVERY unique listed entity
- Include ONE entry for EVERY provided latent entity.
- IDs must be sequential starting from 1.

QUALITY REQUIREMENTS:
- protection_reason must be specific to this document (not generic boilerplate).
- combined_risk_level must consider other entities being retained.
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
        return [
            LLMStructuredColumnConfig(
                name=COL_SENSITIVITY_DISPOSITION,
                prompt=_get_sensitivity_disposition_prompt(
                    privacy_goal,
                    data_summary,
                    strict_entity_protection=strict_entity_protection,
                ),
                model_alias=disposition_alias,
                output_format=SensitivityDispositionSchema,
            ),
        ]
