# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.rewrite import PrivacyGoal

from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_LATENT_ENTITIES,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
)
from anonymizer.engine.detection.constants import DEFAULT_ENTITY_LABELS, _jinja
from anonymizer.engine.detection.custom_columns import (
    apply_validation_and_finalize,
    apply_validation_to_seed_entities,
    merge_and_build_candidates,
    parse_detected_entities,
    prepare_validation_inputs,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias


class ValidationChoice(str, Enum):
    keep = "keep"
    reclass = "reclass"
    drop = "drop"


class ValidationDecision(BaseModel):
    """Per-entity validation decision from the LLM validator."""

    id: str
    decision: ValidationChoice
    proposed_label: str = Field(
        default="",
        description="Correct label when decision is 'reclass', otherwise empty",
    )
    reason: str | None = None


class ValidationDecisions(BaseModel):
    decisions: list[ValidationDecision] = Field(default_factory=list)


class AugmentedEntity(BaseModel):
    value: str = Field(min_length=1)
    label: str = Field(min_length=1)
    reason: str | None = None


class AugmentedEntities(BaseModel):
    entities: list[AugmentedEntity] = Field(default_factory=list)


class LatentEntity(BaseModel):
    category: LatentCategory
    label: str = Field(
        min_length=1,
        description=(
            "General category/class of the inference in snake_case "
            "(e.g., employer, specific_institution, home_location, medication, health_condition)"
        ),
    )
    value: str = Field(
        min_length=1,
        description="Concise inferred value (generalize if not pinned down strongly by evidence)",
    )
    confidence: LatentConfidence
    evidence: list[str] = Field(
        min_length=1,
        max_length=2,
        description="One or two short quotes from the text that support this inference",
    )
    rationale: str = Field(
        min_length=20,
        max_length=150,
        description="One sentence explaining the inference without adding new facts",
    )


class LatentEntities(BaseModel):
    latent_entities: list[LatentEntity] = Field(default_factory=list)


class LatentCategory(str, Enum):
    latent_identifier = "latent_identifier"
    latent_sensitive_attribute = "latent_sensitive_attribute"


class LatentConfidence(str, Enum):
    high = "high"
    medium = "medium"


@dataclass(frozen=True)
class EntityDetectionResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class EntityDetectionWorkflow:
    """Detection workflow using NDD LLM + custom-column steps."""

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def detect_and_validate_entities(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        compute_grouped: bool = True,
        preview_num_records: int | None = None,
    ) -> EntityDetectionResult:
        labels = _merge_labels(entity_labels)
        workflow_model_configs = self._inject_detector_params(
            model_configs=model_configs,
            selected_models=selected_models,
            labels=labels,
            gliner_detection_threshold=gliner_detection_threshold,
        )

        detection_alias = resolve_model_alias("entity_detector", selected_models)
        validator_alias = resolve_model_alias("entity_validator", selected_models)
        augmenter_alias = resolve_model_alias("entity_augmenter", selected_models)

        detection_result = self._adapter.run_workflow(
            dataframe,
            model_configs=workflow_model_configs,
            columns=[
                LLMTextColumnConfig(
                    name=COL_RAW_DETECTED,
                    prompt=_jinja(COL_TEXT),
                    model_alias=detection_alias,
                ),
                CustomColumnConfig(
                    name=COL_SEED_ENTITIES,
                    generator_function=parse_detected_entities,
                ),
                CustomColumnConfig(
                    name=COL_VALIDATION_CANDIDATES,
                    generator_function=prepare_validation_inputs,
                ),
                LLMStructuredColumnConfig(
                    name=COL_VALIDATED_ENTITIES,
                    prompt=_get_validation_prompt(data_summary=data_summary, labels=labels),
                    model_alias=validator_alias,
                    output_format=ValidationDecisions,
                ),
                CustomColumnConfig(
                    name=COL_SEED_ENTITIES_JSON,
                    generator_function=apply_validation_to_seed_entities,
                ),
                LLMStructuredColumnConfig(
                    name=COL_AUGMENTED_ENTITIES,
                    prompt=_get_augment_prompt(data_summary=data_summary, labels=labels),
                    model_alias=augmenter_alias,
                    output_format=AugmentedEntities,
                ),
                CustomColumnConfig(
                    name=COL_MERGED_ENTITIES,
                    generator_function=merge_and_build_candidates,
                ),
                CustomColumnConfig(
                    name=COL_DETECTED_ENTITIES,
                    generator_function=apply_validation_and_finalize,
                ),
            ],
            workflow_name="entity-detection",
            preview_num_records=preview_num_records,
        )
        detected_df = detection_result.dataframe.copy()
        if not compute_grouped and COL_ENTITIES_BY_VALUE in detected_df.columns:
            detected_df = detected_df.drop(columns=[COL_ENTITIES_BY_VALUE])
        return EntityDetectionResult(dataframe=detected_df, failed_records=detection_result.failed_records)

    def identify_latent_entities(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        entity_labels: list[str] | None = None,
        privacy_goal: PrivacyGoal | None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> EntityDetectionResult:
        labels = _merge_labels(entity_labels)
        workflow_model_configs = self._inject_detector_params(
            model_configs=model_configs,
            selected_models=selected_models,
            labels=labels,
            gliner_detection_threshold=gliner_detection_threshold,
        )
        latent_alias = resolve_model_alias("latent_detector", selected_models)
        latent_result = self._adapter.run_workflow(
            dataframe,
            model_configs=workflow_model_configs,
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_LATENT_ENTITIES,
                    prompt=_get_latent_prompt(
                        data_summary=data_summary,
                        privacy_goal=privacy_goal,
                    ),
                    model_alias=latent_alias,
                    output_format=LatentEntities,
                )
            ],
            workflow_name="latent-entity-detection",
            preview_num_records=preview_num_records,
        )
        return EntityDetectionResult(dataframe=latent_result.dataframe, failed_records=latent_result.failed_records)

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        gliner_detection_threshold: float,
        entity_labels: list[str] | None = None,
        privacy_goal: PrivacyGoal | None = None,
        data_summary: str | None = None,
        tag_latent_entities: bool = True,
        compute_grouped_entities: bool | None = None,
        preview_num_records: int | None = None,
    ) -> EntityDetectionResult:
        if tag_latent_entities and privacy_goal is None:
            raise ValueError("privacy_goal is required when tag_latent_entities=True (rewrite mode)")

        compute_grouped = True if compute_grouped_entities is None else compute_grouped_entities
        detected_result = self.detect_and_validate_entities(
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            entity_labels=entity_labels,
            data_summary=data_summary,
            compute_grouped=compute_grouped,
            preview_num_records=preview_num_records,
        )

        if tag_latent_entities:
            latent_result = self.identify_latent_entities(
                detected_result.dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                gliner_detection_threshold=gliner_detection_threshold,
                entity_labels=entity_labels,
                privacy_goal=privacy_goal,
                data_summary=data_summary,
                preview_num_records=preview_num_records,
            )
            final_df = latent_result.dataframe.copy()
            final_failures = [*detected_result.failed_records, *latent_result.failed_records]
        else:
            final_df = detected_result.dataframe.copy()
            final_failures = detected_result.failed_records

        if COL_DETECTED_ENTITIES in final_df.columns:
            final_df[COL_FINAL_ENTITIES] = final_df[COL_DETECTED_ENTITIES]
        if "original_text_column" in dataframe.attrs:
            final_df.attrs["original_text_column"] = dataframe.attrs["original_text_column"]
        return EntityDetectionResult(
            dataframe=final_df,
            failed_records=final_failures,
        )

    def _inject_detector_params(
        self,
        *,
        model_configs: list[ModelConfig],
        selected_models: DetectionModelSelection,
        labels: list[str],
        gliner_detection_threshold: float,
    ) -> list[ModelConfig]:
        resolved = deepcopy(model_configs)
        for config in resolved:
            if config.alias != selected_models.entity_detector:
                continue
            if config.inference_parameters.extra_body is None:
                config.inference_parameters.extra_body = {}
            config.inference_parameters.extra_body["labels"] = labels
            config.inference_parameters.extra_body["threshold"] = gliner_detection_threshold
            config.inference_parameters.extra_body["chunk_length"] = 384
            config.inference_parameters.extra_body["overlap"] = 128
            config.inference_parameters.extra_body["flat_ner"] = False
            break
        return resolved


def _merge_labels(entity_labels: list[str] | None) -> list[str]:
    merged = list(DEFAULT_ENTITY_LABELS)
    for label in entity_labels or []:
        cleaned = label.strip().lower()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
    return merged


def _get_validation_prompt(*, data_summary: str | None, labels: list[str]) -> str:
    prompt = """Validate entity tags for privacy-sensitive information. For each entity, decide: keep, reclassify, or drop.
<<DATA_SUMMARY>>
Valid labels: <<VALID_CLASSES>>

What to KEEP:
- Direct identifiers: names, emails, IDs, phone numbers, addresses, SSNs
- Quasi-identifiers: age, locations (cities, states, regions), occupations, dates, company names, education
- Technical data: credentials, URLs, device IDs, account numbers
- Demographics: gender, race, religion, language, etc.

What to RECLASS (reclassify):
- Entity IS privacy-relevant but has the WRONG label
- Set decision="reclass" and proposed_label to the correct label from the valid labels list
- Example: "San Diego" labeled as "country" → reclass to "city"

What to DROP:
- Universally famous entities ONLY (e.g., "Barack Obama")
  Note: Most cities/companies ARE quasi-identifiers and should be KEPT (Portland, London, Brasilia, etc.)
- Placeholders or label names (e.g., "name", "email", "date")
- Syntax/metadata in structured formats (field names, column headers, keywords)
- Generic time references that are not specific dates (e.g., "today", "now", "soon")

Additional rules:
- Check context matches label (not just format)
- Use exact text from the tagged span
- Prefer ssn over national_id or account_number if ambiguous
- Prefer phone_number over fax_number unless "fax" is explicit
- VIN length 17 is vehicle_identifier, shorter is license_plate (check context for regional variations)
- Numbers preceded by '$' are monetary values, not entities like age or account number
- If classified as date_of_birth, verify context is appropriate; otherwise use date

Use the candidates list as the authoritative source for entity values and labels.

Example:
{%- if <<TAG_NOTATION>> == "xml" -%}
Input text: My <first_name>name</first_name> is <first_name>Amy</first_name>, <age>35</age> in <country>San Diego</country>, <state>California</state>.
{%- elif <<TAG_NOTATION>> == "bracket" -%}
Input text: My [[name|first_name]] is [[Amy|first_name]], [[35|age]] in [[San Diego|country]], [[California|state]].
{%- elif <<TAG_NOTATION>> == "paren" -%}
Input text: My ((PII:first_name|name)) is ((PII:first_name|Amy)), ((PII:age|35)) in ((PII:country|San Diego)), ((PII:state|California)).
{%- else -%}
Input text: My <<PII:first_name>>name<</PII:first_name>> is <<PII:first_name>>Amy<</PII:first_name>>, <<PII:age>>35<</PII:age>> in <<PII:country>>San Diego<</PII:country>>, <<PII:state>>California<</PII:state>>.
{%- endif -%}
Candidates: [{"id": "id_name", "value": "name", "label": "first_name", "context_before": "My ", "context_after": " is Amy"}, {"id": "id_amy", "value": "Amy", "label": "first_name", "context_before": " is ", "context_after": ", 35"}, {"id": "id_35", "value": "35", "label": "age", "context_before": "Amy, ", "context_after": ", software"}, {"id": "id_sd", "value": "San Diego", "label": "country", "context_before": " in ", "context_after": ", California"}, {"id": "id_ca", "value": "California", "label": "state", "context_before": "Diego, ", "context_after": "."}]
Output: [{"id": "id_name", "decision": "drop", "proposed_label": "", "reason": "placeholder not actual name"}, {"id": "id_amy", "decision": "keep", "proposed_label": "", "reason": "valid first name"}, {"id": "id_35", "decision": "keep", "proposed_label": "", "reason": "quasi-identifier"}, {"id": "id_sd", "decision": "reclass", "proposed_label": "city", "reason": "city not country"}, {"id": "id_ca", "decision": "keep", "proposed_label": "", "reason": "valid state"}]

---
Input text: <<TAGGED_TEXT>>
Candidates: <<CANDIDATES>>
"""
    prompt = (
        prompt.replace("<<TAG_NOTATION>>", COL_TAG_NOTATION)
        .replace("<<TAGGED_TEXT>>", _jinja(COL_MERGED_TAGGED_TEXT))
        .replace("<<CANDIDATES>>", _jinja(COL_VALIDATION_CANDIDATES))
        .replace("<<VALID_CLASSES>>", ", ".join(labels))
    )
    context_section = f"\nData context: {data_summary}\n" if data_summary else ""
    return prompt.replace("<<DATA_SUMMARY>>", context_section)


def _get_augment_prompt(*, data_summary: str | None, labels: list[str]) -> str:
    prompt = """Task: Find untagged sensitive entities in text (ignore already tagged entities). Focus on:
- Direct identifiers: Uniquely identify entities (names, emails, IDs), records (transaction IDs, case numbers), resources (file paths, URLs), or instances (server names, hostnames)
- Quasi-identifiers: Attributes that combine to narrow specificity (age, location, job title, timestamps, technical specs)
- Technical secrets: Credentials (passwords, API keys, tokens), access (internal URLs, endpoints), proprietary terms

We have the following type of data: <<DATA_SUMMARY>>
Valid labels: <<VALID_CLASSES>>
If no valid label fits, create descriptive snake_case label (e.g., clinic_name, server_name, transaction_id).

Rules:
- Tag actual values, not placeholders
- Do not repeat already-tagged entities
- In structured formats, distinguish between:
  - Syntax/metadata: field names, column headers, function names, keywords
  - Data: assigned values, cell contents, literals, user input
  Only tag data, never syntax

Example:
{%- if <<TAG_NOTATION>> == "xml" -%}
Input text: My name is Amy Steier in <city>San Diego</city>.
{%- elif <<TAG_NOTATION>> == "bracket" -%}
Input text: My name is Amy Steier in [[San Diego|city]].
{%- elif <<TAG_NOTATION>> == "paren" -%}
Input text: My name is Amy Steier in ((PII:city|San Diego)).
{%- else -%}
Input text: My name is Amy Steier in <<PII:city>>San Diego<</PII:city>>.
{%- endif -%}
Already-detected entities: [{"value": "San Diego", "label": "city"}]
Output: [{"value": "Amy", "label": "first_name", "reason": "first name"}, {"value": "Steier", "label": "last_name", "reason": "last name"}]

---
Input text: <<TAGGED_TEXT>>
Already-detected entities: <<SEED_ENTITIES>>
"""
    prompt = (
        prompt.replace("<<TAG_NOTATION>>", COL_TAG_NOTATION)
        .replace("<<TAGGED_TEXT>>", _jinja(COL_INITIAL_TAGGED_TEXT))
        .replace("<<SEED_ENTITIES>>", _jinja(COL_SEED_ENTITIES_JSON))
        .replace("<<VALID_CLASSES>>", ", ".join(labels))
    )
    context_section = data_summary if data_summary else "Not provided"
    return prompt.replace("<<DATA_SUMMARY>>", context_section)


def _get_latent_prompt(*, data_summary: str | None, privacy_goal: PrivacyGoal | None) -> str:
    summary_line = data_summary.strip() if data_summary else "Not provided"
    privacy_goal_text = _format_privacy_goal(privacy_goal)
    prompt = """You are performing: LATENT ENTITY & INFERENCE ANALYSIS for privacy protection.

The text will be rewritten according to this privacy goal: %s

Goal: Identify sensitive information that is NOT explicitly stated in the text, but is reasonably inferable from context and could materially increase re-identification risk or reveal sensitive attributes about a real person. Treat inference as a first-class privacy surface; do not assume removing explicit identifiers is sufficient.

Return only inferences that would help an adversary narrow down or recognize the subject in the real world, or would reveal sensitive attributes about them.

Examples of eligible latent content when strongly supported:
- Identity linkage: specific employer, school, institution, unique role, affiliation group, recognizable relationship
- Where/when linkage: home area, prior location, travel pattern, event timeframe, distinctive routine
- Sensitive attributes: health condition/procedure/medication, legal situation, immigration status, assault/violence, substance use, etc.

Non-examples (exclude):
- Generic domain nouns that do not narrow identity (e.g., "a company", "a school", "a clinic", "a court", "a neighborhood")
- Broad facts that are not person-linking or sensitive (e.g., "consistent blood levels")
- Anything explicitly stated verbatim in the text

Threat model:
Assume an adversary who can read the text in full, has general domain knowledge and access to public information, and may have partial prior familiarity with the subject.

Data type summary:
%s

Input text (identifiers already tagged inline):
---
%s
---

Rules (strict)
1) True latent only:
   - Do NOT repeat any already-tagged entities.
   - Do NOT output anything explicitly stated in the raw text.
2) Evidence-bounded inference:
   - Every latent entity MUST include 1-2 short quotes from the text as evidence.
   - Do NOT "jump" to a specific named entity unless the evidence pins it down strongly.
   - If multiple plausible specifics exist, DO NOT guess: either generalize or omit.
3) High signal only:
   - Prefer returning an EMPTY list over generic filler.
   - If the inference does not materially increase identifiability or reveal a sensitive attribute, exclude it.
4) Non-redundant:
   - Do not output multiple variants of the same inference.

Quality checks before finalizing:
- Remove any item whose value is too generic to help an adversary.
- Remove any item that restates explicit text.
- Remove any item without clear evidence quotes.

Now produce the JSON for the input.
""" % (
        privacy_goal_text,
        summary_line,
        _jinja(COL_TAGGED_TEXT),
    )
    return prompt


def _format_privacy_goal(privacy_goal: PrivacyGoal | None) -> str:
    if privacy_goal is None:
        return "Not provided"
    return privacy_goal.to_prompt_string()
