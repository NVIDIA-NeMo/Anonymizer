# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT
from pydantic import ValidationError

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_DOMAIN_SUPPLEMENT_PRIVACY,
    COL_ENTITIES_BY_VALUE,
    COL_LATENT_ENTITIES,
    COL_SENSITIVITY_DISPOSITION,
    COL_SIMPLE_DISPOSITION,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.rewrite.disposition_derivation import (
    _flatten_context,
    derive_combined_risk_level,
    reconstruct_full_disposition,
    template_protection_reason,
)
from anonymizer.engine.schemas import (
    EntityDispositionSchema,
    SensitivityDispositionSchema,
    SimpleDispositionResult,
)
from anonymizer.engine.schemas.rewrite import _ENTITY_LABEL_TO_CATEGORY

logger = logging.getLogger(__name__)


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
The text below contains inline entity tags marking identified entities.
{% if <<TAG_NOTATION>> == 'bracket' %}Tags use the format [[entity_value|entity_label]] (e.g. [[Portland|city]]). The substring before "|" is the entity_value — record it EXACTLY, without the [[ ]] brackets.
{% elif <<TAG_NOTATION>> == 'xml' %}Tags use the format <entity_label>entity_value</entity_label> (e.g. <city>Portland</city>). The inner text is the entity_value — record it EXACTLY, without the surrounding tags.
{% elif <<TAG_NOTATION>> == 'paren' %}Tags use the format ((SENSITIVE:entity_label|entity_value)) (e.g. ((SENSITIVE:city|Portland))). The substring after "|" is the entity_value — record it EXACTLY, without the ((SENSITIVE:...)) wrapper.
{% elif <<TAG_NOTATION>> == 'sentinel' %}Tags use the format <<SENSITIVE:entity_label>>entity_value<</SENSITIVE:entity_label>> (e.g. <<SENSITIVE:city>>Portland<</SENSITIVE:city>>). The text between the sentinels is the entity_value — record it EXACTLY, without any <<SENSITIVE:...>> markers.
{% endif %}
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
  If combined_risk_level is medium or high: mitigation requires paraphrasing, removing
  supporting details, or generalizing facts (suppress_inference). Replace is rarely
  appropriate since the value is not explicitly in the text.
  If combined_risk_level is low: leave_as_is — do NOT use suppress_inference or any
  other protection method.
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
            "<<TAG_NOTATION>>": COL_TAG_NOTATION,
            "<<TAGGED_TEXT>>": _jinja(COL_TAGGED_TEXT),
            "<<FINAL_ENTITIES>>": _jinja(COL_ENTITIES_BY_VALUE),
            "<<LATENT_ENTITIES>>": _jinja(COL_LATENT_ENTITIES),
            "<<STRICT_PROTECTION_BLOCK>>": strict_protection_block,
        },
    )


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pessimistic fallback when reconstruction yields nothing
# ---------------------------------------------------------------------------


def _pessimistic_fallback_disposition(
    entities_by_value: object,
    latent_entities: object,
) -> SensitivityDispositionSchema:
    """Build a worst-case disposition from the entity context alone.

    Used when ``reconstruct_full_disposition`` returns an empty list — e.g.
    every ``SimpleDispositionItem`` was an orphan, or the LLM emitted no
    items at all. Without this fallback, downstream
    ``parse_sensitivity_disposition`` raises ``ValidationError`` on
    ``min_length=1`` and the row drops, defeating the whole loose-wire +
    server-reconstruction architecture this PR exists to add.

    Policy (per Lipika/Andre's review on PR #130, addressing the
    record-drop concern):
      * ``direct_identifier`` -> ``replace`` (high risk, must be masked).
      * everything else -> ``generalize`` (medium risk, mask but keep
        rough semantics for utility).

    Categories come from the per-entity ``entity_label`` via the
    ``_ENTITY_LABEL_TO_CATEGORY`` map (the same source of truth the
    reconstructor uses for entity-label-stuffed-into-category drift);
    unmapped labels fall back to ``quasi_identifier``.
    """
    flat = _flatten_context(entities_by_value, latent_entities)
    items: list[EntityDispositionSchema] = []
    for idx, slot in enumerate(flat, start=1):
        label = (slot.get("entity_label") or "").strip()
        value = (slot.get("entity_value") or "").strip()
        source = slot.get("source") or "tagged"
        if not label or not value:
            continue
        if source == "latent":
            category = "latent_identifier"
        else:
            category = _ENTITY_LABEL_TO_CATEGORY.get(label, "quasi_identifier")
        method = "replace" if category == "direct_identifier" else "generalize"
        sensitivity = "high" if category == "direct_identifier" else "medium"
        combined_risk = derive_combined_risk_level(category, method, sensitivity)
        reason = template_protection_reason(category, method, sensitivity)
        items.append(
            EntityDispositionSchema(
                id=idx,
                source=source,
                category=category,
                sensitivity=sensitivity,
                entity_label=label,
                entity_value=value,
                protection_method_suggestion=method,
                combined_risk_level=combined_risk,
                protection_reason=reason,
            )
        )
    if not items:
        # Genuinely no entities at all in context. The orchestrator should have
        # short-circuited rows with no detected entities before this step, so
        # this is a pipeline-invariant violation — but this is the last-resort
        # path whose contract is "never drop the row." Emitting an empty list
        # would raise on SensitivityDispositionSchema's (and the downstream
        # parser's) min_length=1 invariant and drop the record, so we log loudly
        # and emit a single no-op (leave_as_is/low) disposition instead. It is
        # excluded from protected_entities, so it never reaches the rewrite.
        logger.error(
            "pessimistic fallback: empty entity context at the disposition step "
            "(orchestrator should have short-circuited entity-free rows); emitting "
            "a single no-op disposition so the row is not dropped"
        )
        items.append(
            EntityDispositionSchema(
                id=1,
                source="tagged",
                category="quasi_identifier",
                sensitivity="low",
                entity_label="",
                entity_value="",
                protection_method_suggestion="leave_as_is",
                combined_risk_level=derive_combined_risk_level("quasi_identifier", "leave_as_is", "low"),
                protection_reason=template_protection_reason("quasi_identifier", "leave_as_is", "low"),
            )
        )
    return SensitivityDispositionSchema(sensitivity_disposition=items)


# ---------------------------------------------------------------------------
# Reconstruction column
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_SIMPLE_DISPOSITION, COL_ENTITIES_BY_VALUE, COL_LATENT_ENTITIES])
def _reconstruct_full_disposition_column(row: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the strict EntityDispositionSchema list from the loose LLM
    output in ``COL_SIMPLE_DISPOSITION`` plus the entity context columns.

    Writes ``COL_SENSITIVITY_DISPOSITION`` so every downstream consumer
    reads the same column name / shape as before this refactor.

    Empty-result fallback: when the model returns nothing usable (every
    item is an orphan, or the LLM omitted the field entirely), build a
    pessimistic disposition from the entity context (direct identifiers
    -> replace, everything else -> generalize). This addresses the
    record-drop concern Lipika and Andre raised on PR #130 — emitting an
    empty disposition would have failed downstream
    ``parse_sensitivity_disposition``'s ``min_length=1`` check anyway.
    """
    simple_raw = row.get(COL_SIMPLE_DISPOSITION, {}) or {}
    if isinstance(simple_raw, SimpleDispositionResult):
        simple = simple_raw
    else:
        if isinstance(simple_raw, str):
            import json as _json

            try:
                simple_raw = _json.loads(simple_raw)
            except Exception:
                simple_raw = {}
        try:
            simple = SimpleDispositionResult.model_validate(simple_raw)
        except ValidationError as exc:
            logger.warning(
                "reconstruct: SimpleDispositionResult failed to validate (%s); "
                "falling back to pessimistic disposition from entity context",
                str(exc)[:200],
            )
            simple = SimpleDispositionResult()

    entities_by_value = row.get(COL_ENTITIES_BY_VALUE)
    latent_entities = row.get(COL_LATENT_ENTITIES)

    if not simple.sensitivity_disposition:
        logger.warning(
            "reconstruct: empty SimpleDispositionResult for row; "
            "falling back to pessimistic disposition from entity context"
        )
        full = _pessimistic_fallback_disposition(entities_by_value, latent_entities)
    else:
        try:
            full = reconstruct_full_disposition(simple, entities_by_value, latent_entities)
        except ValidationError as exc:
            logger.warning(
                "reconstruct: ValidationError after orphan-skipping (likely all items out of context range); "
                "falling back to pessimistic disposition. detail=%s",
                str(exc)[:200],
            )
            full = _pessimistic_fallback_disposition(entities_by_value, latent_entities)

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
        strict_entity_protection: bool = False,
    ) -> list[ColumnConfigT]:
        """Two-step pipeline for small-model robustness:

        1. LLM column emits the loose ``SimpleDispositionResult`` to a
           hidden ``COL_SIMPLE_DISPOSITION`` column. The wire schema has
           no enum/required/minLength constraints, so DataDesigner's
           jsonschema pre-validate gate accepts drifted small-model
           output that strict ``SensitivityDispositionSchema`` would
           reject. ``drop=True`` keeps this internal hand-off out of the
           user-facing preview DataFrame.
        2. Pure-python reconstruction column rebuilds the strict
           ``SensitivityDispositionSchema`` from the loose wire output
           plus the entity-context columns. No LLM call; deterministic;
           handles id pairing, category/method drift normalization,
           ``combined_risk_level`` derivation, and pessimistic fallback
           when the LLM produces nothing usable.

        ``strict_entity_protection`` continues to flow into the prompt's
        ``<strict_entity_protection>`` block — the contract is enforced
        at prompt time. The output_format selection between
        ``SensitivityDispositionSchema`` and
        ``StrictSensitivityDispositionSchema`` is no longer needed
        because we always emit ``SimpleDispositionResult`` on the wire
        and reconstruct into the canonical (non-strict) schema, which
        downstream consumers already accept.
        """
        disposition_alias = resolve_model_alias("disposition_analyzer", selected_models)
        return [
            LLMStructuredColumnConfig(
                name=COL_SIMPLE_DISPOSITION,
                prompt=_get_sensitivity_disposition_prompt(
                    privacy_goal,
                    data_summary,
                    strict_entity_protection=strict_entity_protection,
                ),
                model_alias=disposition_alias,
                output_format=SimpleDispositionResult,
                drop=True,
            ),
            CustomColumnConfig(
                name=COL_SENSITIVITY_DISPOSITION,
                generator_function=_reconstruct_full_disposition_column,
            ),
        ]
