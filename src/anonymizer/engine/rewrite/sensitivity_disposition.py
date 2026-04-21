# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_SENSITIVITY_DISPOSITION,
    COL_SIMPLE_DISPOSITION,
    COL_TAGGED_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.rewrite.disposition_derivation import reconstruct_full_disposition
from anonymizer.engine.schemas import SimpleDispositionResult


def _get_sensitivity_disposition_prompt(privacy_goal: PrivacyGoal, data_summary: str | None = None) -> str:
    privacy_goal_str = privacy_goal.to_prompt_string()
    # TODO: align entity detection prompts (validation, augment, latent) to use "Dataset description:" label
    data_summary_line = (
        f"\nDataset description: {data_summary.strip()}" if data_summary and data_summary.strip() else ""
    )

    # Prompt opening kept stable so tools/bench_harness.py::ROLE_PROMPT_PREFIXES
    # continues to classify this LLM call as disposition_analyzer in tap logs.
    prompt = """You are responsible for creating a unified sensitivity disposition for privacy-preserving text rewriting.

Your task is to analyze ALL entities in the text and classify each one on two axes:
  (1) category — what KIND of privacy risk the entity poses.
  (2) sensitivity — how much harm disclosing this value would cause ON ITS OWN.
And pick:
  (3) protection_method_suggestion — what should be done with it.
Optionally:
  (4) protection_reason — a short, document-specific rationale (we will template one if you omit this).

Other fields (needs_protection, combined risk, etc.) are derived server-side —
do not emit them. Focus your budget on the four fields above.

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

<entity_categories>
category = "direct_identifier" (high-risk, standalone)
- Uniquely identify on their own: full names, exact addresses, SSNs, email, phone.
- Default action: REPLACE with plausible synthetic alternative.
- Use abstraction/removal only if replacement distorts meaning.

category = "quasi_identifier" (context-dependent)
- Not unique alone, but identifying in combination: age, city, occupation, dates, institutions.
- NOT automatically sensitive: modify ONLY if combination increases re-identification risk.
- When modification needed: prefer generalization over removal.

category = "sensitive_attribute" (harmful to disclose)
- Sensitive because of content, not identifiability: health conditions, sexual orientation,
  mental health, substance use, legal issues, political/religious views.
- Protect when disclosure itself is harmful, even if identifiability risk is low.
- Can ALSO function as quasi-identifiers if distinctive in context.

category = "latent_identifier" (inferred)
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

<output_format>
Produce one list entry per unique entity from <input_detected_entities> plus one per <input_latent_entities>.
IDs sequential from 1. Each item has these fields:

  id:                             integer, sequential from 1
  category:                       one of "direct_identifier" | "quasi_identifier" | "sensitive_attribute" | "latent_identifier"
  sensitivity:                    one of "low" | "medium" | "high"
  protection_method_suggestion:   one of "replace" | "generalize" | "remove" | "suppress_inference" | "leave_as_is"
  protection_reason:              (optional) short, document-specific rationale — omit if not useful

Guidance:
- For latent entities, "replace" is rarely appropriate (the value is not literally in the text).
- When protection_method_suggestion is "leave_as_is", the server records needs_protection=false;
  for any other method, needs_protection=true. Do not emit needs_protection yourself.
- You MAY also echo entity_label / entity_value / source to help the server pair ids — but if
  you do, match the input exactly. These are not required.
</output_format>"""
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
        },
    )


# ---------------------------------------------------------------------------
# Reconstruction column
# ---------------------------------------------------------------------------


@custom_column_generator(
    required_columns=[COL_SIMPLE_DISPOSITION, COL_ENTITIES_BY_VALUE, COL_LATENT_ENTITIES]
)
def _reconstruct_full_disposition_column(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the strict EntityDispositionSchema list from the loose LLM
    output in COL_SIMPLE_DISPOSITION plus the entity context columns.
    Writes COL_SENSITIVITY_DISPOSITION so every downstream consumer reads
    the same column name / shape as before this refactor.
    """
    simple_raw = row.get(COL_SIMPLE_DISPOSITION, {}) or {}
    # DD may deliver either a dict or the already-parsed pydantic model as
    # a dict; wrap defensively.
    if isinstance(simple_raw, SimpleDispositionResult):
        simple = simple_raw
    else:
        simple = SimpleDispositionResult.model_validate(simple_raw)

    ebv = row.get(COL_ENTITIES_BY_VALUE) or {}
    if isinstance(ebv, dict):
        tagged_ctx = ebv.get("entities_by_value") or []
    else:
        tagged_ctx = ebv or []

    lat = row.get(COL_LATENT_ENTITIES) or {}
    if isinstance(lat, dict):
        latent_ctx = lat.get("latent_entities") or []
    else:
        latent_ctx = lat or []

    full = reconstruct_full_disposition(simple, tagged_ctx, latent_ctx)
    row[COL_SENSITIVITY_DISPOSITION] = full.model_dump()
    return row


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
    ) -> list[ColumnConfigT]:
        disposition_alias = resolve_model_alias("disposition_analyzer", selected_models)
        return [
            # Step 1 — the LLM call with the SHRUNKEN wire contract. JSON
            # Schema emitted to the model has no enum / minLength / strict
            # required, so DataDesigner's jsonschema.validate() cannot
            # reject small-model drift before our before-validator runs.
            LLMStructuredColumnConfig(
                name=COL_SIMPLE_DISPOSITION,
                prompt=_get_sensitivity_disposition_prompt(privacy_goal, data_summary),
                model_alias=disposition_alias,
                output_format=SimpleDispositionResult,
            ),
            # Step 2 — pure-python reconstruction into the strict schema
            # that every downstream consumer already reads. No LLM call;
            # deterministic; handles id pairing, needs_protection
            # derivation, and protection_reason templating.
            CustomColumnConfig(
                name=COL_SENSITIVITY_DISPOSITION,
                generator_function=_reconstruct_full_disposition_column,
            ),
        ]
