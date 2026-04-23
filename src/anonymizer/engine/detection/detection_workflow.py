# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass

import pandas as pd
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.anonymizer_config import Detect as AnonymizerDetectConfig
from anonymizer.config.models import DetectionModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_AUGMENTED_ENTITIES,
    COL_DETECTED_ENTITIES,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_LATENT_ENTITIES,
    COL_MERGED_ENTITIES,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_TAGGED_TEXT,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATION_DECISIONS,
    COL_VALIDATION_SKELETON,
    DEFAULT_ENTITY_LABELS,
    ENTITY_LABEL_EXAMPLES,
    _jinja,
)
from anonymizer.engine.detection.chunked_validation import (
    ChunkedValidationParams,
    make_chunked_validation_generator,
)
from anonymizer.engine.detection.custom_columns import (
    apply_validation_and_finalize,
    apply_validation_to_seed_entities,
    enrich_validation_decisions,
    merge_and_build_candidates,
    parse_detected_entities,
    prepare_validation_inputs,
)
from anonymizer.engine.detection.postprocess import EntitySpan, group_entities_by_value
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias, resolve_model_aliases
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.schemas import (
    AugmentedEntitiesSchema,
    EntitiesByValueSchema,
    EntitiesSchema,
    LatentEntitiesSchema,
)

logger = logging.getLogger("anonymizer.detection")

# Defaults for the two chunked-validation knobs. Sourced from the Detect config
# so there is a single source of truth; the workflow method defaults exist so
# internal tests and ad-hoc callers do not have to wire plumbing by hand.
_DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL: int = AnonymizerDetectConfig.model_fields[
    "validation_max_entities_per_call"
].default
_DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS: int = AnonymizerDetectConfig.model_fields[
    "validation_excerpt_window_chars"
].default


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
        validation_max_entities_per_call: int = _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = _DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        entity_labels: list[str] | None = None,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> EntityDetectionResult:
        """Run the core detection pipeline: GLiNER NER, LLM validation, LLM augmentation, and finalization.

        This is the primary detection workflow. It detects entities via GLiNER,
        validates/reclassifies them with an LLM (chunked across a pool of
        validator aliases), augments with additional entities the detector may
        have missed, and produces final standoff entity spans with overlap
        resolution.
        """
        labels = _resolve_detection_labels(entity_labels)
        workflow_model_configs = self._inject_detector_params(
            model_configs=model_configs,
            selected_models=selected_models,
            labels=labels,
            gliner_detection_threshold=gliner_detection_threshold,
        )

        detection_alias = resolve_model_alias("entity_detector", selected_models)
        validator_aliases = resolve_model_aliases("entity_validator", selected_models)
        augmenter_alias = resolve_model_alias("entity_augmenter", selected_models)
        logger.debug(
            "detection aliases: detector=%s, validator_pool=%s, augmenter=%s",
            detection_alias,
            validator_aliases,
            augmenter_alias,
        )
        # ModelConfig.max_parallel_requests caps concurrency *per alias*. When
        # the pool has multiple validators each gets its own cap, so total
        # in-flight validator calls can reach sum(per-alias caps). Operators
        # provisioning rate budgets for downstream providers should size each
        # alias's cap accordingly.
        if len(validator_aliases) > 1:
            logger.warning(
                "entity validation runs across a pool of %d aliases (%s). "
                "max_parallel_requests is enforced per alias, so the pool "
                "multiplies total in-flight validator calls; budget provider "
                "TPM/RPM accordingly.",
                len(validator_aliases),
                validator_aliases,
            )

        validator_generator = make_chunked_validation_generator(validator_aliases)
        validator_params = ChunkedValidationParams(
            pool=list(validator_aliases),
            max_entities_per_call=validation_max_entities_per_call,
            excerpt_window_chars=validation_excerpt_window_chars,
            prompt_template=_get_validation_prompt(data_summary=data_summary, labels=labels),
        )

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
                    name=COL_SEED_VALIDATION_CANDIDATES,
                    generator_function=prepare_validation_inputs,
                ),
                CustomColumnConfig(
                    name=COL_VALIDATION_DECISIONS,
                    generator_function=validator_generator,
                    generator_params=validator_params,
                    drop=True,
                ),
                CustomColumnConfig(
                    name=COL_VALIDATED_ENTITIES,
                    generator_function=enrich_validation_decisions,
                ),
                CustomColumnConfig(
                    name=COL_SEED_ENTITIES_JSON,
                    generator_function=apply_validation_to_seed_entities,
                ),
                LLMStructuredColumnConfig(
                    name=COL_AUGMENTED_ENTITIES,
                    prompt=_get_augment_prompt(
                        data_summary=data_summary, labels=labels, strict_labels=entity_labels is not None
                    ),
                    model_alias=augmenter_alias,
                    output_format=AugmentedEntitiesSchema,
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
        """Detect latent/inferred entities that could enable re-identification.

        Runs after ``detect_and_validate_entities`` when rewrite mode is
        enabled. Uses an LLM to identify entities inferable from context.
        """
        labels = _resolve_detection_labels(entity_labels)
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
                    output_format=LatentEntitiesSchema,
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
        validation_max_entities_per_call: int = _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
        validation_excerpt_window_chars: int = _DEFAULT_VALIDATION_EXCERPT_WINDOW_CHARS,
        entity_labels: list[str] | None = None,
        privacy_goal: PrivacyGoal | None = None,
        data_summary: str | None = None,
        tag_latent_entities: bool = True,
        compute_grouped_entities: bool | None = None,
        preview_num_records: int | None = None,
    ) -> EntityDetectionResult:
        """Run the full detection pipeline, optionally including latent entity detection.

        Calls ``detect_and_validate_entities`` first, then optionally
        ``identify_latent_entities`` if ``tag_latent_entities`` is True
        (rewrite mode). Merges failures from both stages.
        """
        if tag_latent_entities and privacy_goal is None:
            raise ValueError("privacy_goal is required when tag_latent_entities=True (rewrite mode)")

        compute_grouped = True if compute_grouped_entities is None else compute_grouped_entities
        detected_result = self.detect_and_validate_entities(
            dataframe,
            model_configs=model_configs,
            selected_models=selected_models,
            gliner_detection_threshold=gliner_detection_threshold,
            validation_max_entities_per_call=validation_max_entities_per_call,
            validation_excerpt_window_chars=validation_excerpt_window_chars,
            entity_labels=entity_labels,
            data_summary=data_summary,
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

        # When entity_labels is explicitly provided (even if it matches DEFAULT_ENTITY_LABELS),
        # the augmenter is strict and out-of-scope labels are filtered.
        # entity_labels=None is the only way to get permissive augmentation.
        # TODO(docs): document this None-vs-explicit contract in user-facing docs.
        if COL_DETECTED_ENTITIES in final_df.columns:
            allowed = set(entity_labels) if entity_labels is not None else None
            final_df[COL_FINAL_ENTITIES] = final_df[COL_DETECTED_ENTITIES].apply(
                lambda raw: _materialize_final_entities(raw, allowed_labels=allowed)
            )
            if compute_grouped:
                final_df[COL_ENTITIES_BY_VALUE] = final_df[COL_FINAL_ENTITIES].apply(_build_entities_by_value)
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


def _resolve_detection_labels(entity_labels: list[str] | None) -> list[str]:
    if entity_labels is None:
        return list(DEFAULT_ENTITY_LABELS)
    return list(entity_labels)


def _materialize_final_entities(raw: object, *, allowed_labels: set[str] | None) -> dict:
    """Build COL_FINAL_ENTITIES, optionally filtering to *allowed_labels*."""
    parsed = EntitiesSchema.from_raw(raw)
    if allowed_labels is None:
        return parsed.model_dump()
    kept = [e for e in parsed.entities if e.label in allowed_labels]
    return EntitiesSchema(entities=kept).model_dump()


def _build_entities_by_value(final_entities_raw: object) -> dict:
    """Derive COL_ENTITIES_BY_VALUE from COL_FINAL_ENTITIES."""
    parsed = EntitiesSchema.from_raw(final_entities_raw)
    spans = [
        EntitySpan(
            entity_id=e.id,
            value=e.value,
            label=e.label,
            start_position=e.start_position,
            end_position=e.end_position,
            score=e.score,
            source=e.source,
        )
        for e in parsed.entities
    ]
    return EntitiesByValueSchema(entities_by_value=group_entities_by_value(entities=spans)).model_dump(mode="json")


def _format_label_examples(labels: list[str]) -> str:
    """Build a formatted list of entity classes with examples.

    Labels present in ENTITY_LABEL_EXAMPLES get their examples; custom labels
    added by the user appear without examples so the LLM still knows they're valid.
    """
    lines: list[str] = []
    for label in labels:
        examples = ENTITY_LABEL_EXAMPLES.get(label)
        if examples:
            lines.append(f"- {label}: {', '.join(examples)}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def _get_validation_prompt(*, data_summary: str | None, labels: list[str]) -> str:
    prompt = """Validate entity tags for privacy-sensitive information. For each entity in the template below, fill in the "decision" and "reason" fields. Fill in "proposed_label" only when decision is "reclass".
<<DATA_SUMMARY>>

Here are all the valid entity classes with examples (only classes from this list can be proposed):
<<LABEL_EXAMPLES>>

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
- Kinship and family-role terms used as relationship descriptors are not quasi-identifiers.
- Filenames that are system executables or binaries (e.g., ending in .exe, .dll, .sys).
- If a single abbreviated letter (e.g. "A.") is tagged as a first_name but is not followed by a last_name, drop it.

PARTIAL-TOKEN RULE (HARD DROP):
- If the tagged value is only a substring of a larger contiguous token (letters, digits, underscore with no whitespace boundary), DROP it.
- A contiguous token means adjacent letters, digits, or underscore with no whitespace boundary.
- If characters immediately before or after the tagged span are alphanumeric, or underscore, the tag is a partial token and must be dropped.
- Example:
    {%- if <<TAG_NOTATION>> == "xml" -%}
    "internal_<unique_id>procID</unique_id>_id" → drop, because "procID" is inside "internal_procID_id"
    {%- elif <<TAG_NOTATION>> == "bracket" -%}
    "internal_[[procID|unique_id]]_id" → drop, because "procID" is inside "internal_procID_id"
    {%- elif <<TAG_NOTATION>> == "paren" -%}
    "internal_((SENSITIVE:unique_id|procID))_id" → drop, because "procID" is inside "internal_procID_id"
    {%- else -%}
    "internal_<<SENSITIVE:unique_id>>procID<</SENSITIVE:unique_id>>_id" → drop, because "procID" is inside "internal_procID_id"
    {%- endif -%}
- Example:
    {%- if <<TAG_NOTATION>> == "xml" -%}
    "<political_view>dem</political_view>eanor" → drop, because "dem" is inside "demeanor"
    {%- elif <<TAG_NOTATION>> == "bracket" -%}
    "[[dem|political_view]]eanor" → drop, because "dem" is inside "demeanor"
    {%- elif <<TAG_NOTATION>> == "paren" -%}
    "((SENSITIVE:political_view|dem))eanor" → drop, because "dem" is inside "demeanor"
    {%- else -%}
    "<<SENSITIVE:political_view>>dem<</SENSITIVE:political_view>>eanor" → drop, because "dem" is inside "demeanor"
    {%- endif -%}
- Apply this even if the substring itself looks privacy-relevant.

AGE RULE:
- Keep "age" only when the number clearly refers to a person's age (e.g., "35 years old", "35-year-old", "aged 35", "John, 35, ...")
- Patterns like "X years", "X months", or "X days" describing how long something happened indicate duration, not age.
- If the number refers to time, duration, counts, prices, measurements, codes, or dates — it is NOT age.
- If unclear, drop the tag.


Additional rules:
- Check context matches label (not just format)
- Prefer ssn over national_id or account_number if ambiguous
- Prefer phone_number over fax_number unless "fax" is explicit
- VIN length 17 is vehicle_identifier, shorter is license_plate (check context for regional variations)
- Numbers preceded by '$' are monetary values, not entities like age or account number
- If classified as date_of_birth, verify context is appropriate; otherwise use date
- The word "straight" rarely has the label "sexuality"; if the context doesn't absolutely imply "sexuality", drop this tag
- The entity label "occupation" refers only to a specific paid job title or profession (e.g., "registered nurse", "software engineer", "retail salesperson", "teacher", "bartender"). Do NOT label generic roles, activities, or vague work-related nouns as occupations (e.g., "volunteer", "mentor", "guide", "partner", "supplier", "helper")
- You MUST fill in a decision for EVERY entry in the template — do not skip any
- Return ONLY the entries from the template — do not add new entries for entities not already in the template
- Copy ids exactly as given; never modify entries

Example:
{%- if <<TAG_NOTATION>> == "xml" -%}
Input text: My <first_name>name</first_name> is <first_name>Amy</first_name>, <age>35</age> in <country>San Diego</country>, <state>California</state>.
{%- elif <<TAG_NOTATION>> == "bracket" -%}
Input text: My [[name|first_name]] is [[Amy|first_name]], [[35|age]] in [[San Diego|country]], [[California|state]].
{%- elif <<TAG_NOTATION>> == "paren" -%}
Input text: My ((SENSITIVE:first_name|name)) is ((SENSITIVE:first_name|Amy)), ((SENSITIVE:age|35)) in ((SENSITIVE:country|San Diego)), ((SENSITIVE:state|California)).
{%- else -%}
Input text: My <<SENSITIVE:first_name>>name<</SENSITIVE:first_name>> is <<SENSITIVE:first_name>>Amy<</SENSITIVE:first_name>>, <<SENSITIVE:age>>35<</SENSITIVE:age>> in <<SENSITIVE:country>>San Diego<</SENSITIVE:country>>, <<SENSITIVE:state>>California<</SENSITIVE:state>>.
{%- endif -%}
Template: {"decisions": [{"id": "first_name_3_7", "value": "name", "label": "first_name", "decision": null, "proposed_label": null, "reason": null}, {"id": "first_name_11_14", "value": "Amy", "label": "first_name", "decision": null, "proposed_label": null, "reason": null}, {"id": "age_16_18", "value": "35", "label": "age", "decision": null, "proposed_label": null, "reason": null}, {"id": "country_22_31", "value": "San Diego", "label": "country", "decision": null, "proposed_label": null, "reason": null}, {"id": "state_33_43", "value": "California", "label": "state", "decision": null, "proposed_label": null, "reason": null}]}
Output: {"decisions": [{"id": "first_name_3_7", "value": "name", "label": "first_name", "decision": "drop", "proposed_label": "", "reason": "placeholder not actual name"}, {"id": "first_name_11_14", "value": "Amy", "label": "first_name", "decision": "keep", "proposed_label": "", "reason": "real first name"}, {"id": "age_16_18", "value": "35", "label": "age", "decision": "keep", "proposed_label": "", "reason": "age quasi-identifier"}, {"id": "country_22_31", "value": "San Diego", "label": "country", "decision": "reclass", "proposed_label": "city", "reason": "city not country"}, {"id": "state_33_43", "value": "California", "label": "state", "decision": "keep", "proposed_label": "", "reason": "state quasi-identifier"}]}

---
Input text: <<TAGGED_TEXT>>
Template: <<VALIDATION_SKELETON>>
"""
    context_section = f"\nData context: {data_summary}\n" if data_summary else ""
    return substitute_placeholders(
        prompt,
        {
            "<<TAG_NOTATION>>": COL_TAG_NOTATION,
            "<<TAGGED_TEXT>>": _jinja(COL_SEED_TAGGED_TEXT),
            "<<VALIDATION_SKELETON>>": _jinja(COL_VALIDATION_SKELETON),
            "<<LABEL_EXAMPLES>>": _format_label_examples(labels),
            "<<DATA_SUMMARY>>": context_section,
        },
    )


def _get_augment_prompt(*, data_summary: str | None, labels: list[str], strict_labels: bool = False) -> str:
    if strict_labels:
        label_block = (
            "Here are the allowed entity classes. Use ONLY labels from this list:\n"
            "<<VALID_CLASSES>>\n\n"
            "Do not create new labels. If an entity does not fit any label in the list, skip it."
        )
        example_block = """\
Example (allowed labels: first_name, last_name, city — note that employment_status is NOT in the allowed list):
{%- if <<TAG_NOTATION>> == "xml" -%}
Input text: Jane Doe lives in <city>Santa Clara</city>. She works full-time.
{%- elif <<TAG_NOTATION>> == "bracket" -%}
Input text: Jane Doe lives in [[Santa Clara|city]]. She works full-time.
{%- elif <<TAG_NOTATION>> == "paren" -%}
Input text: Jane Doe lives in ((SENSITIVE:city|Santa Clara)). She works full-time.
{%- else -%}
Input text: Jane Doe lives in <<SENSITIVE:city>>Santa Clara<</SENSITIVE:city>>. She works full-time.
{%- endif -%}
Already-detected entities: [{"value": "Santa Clara", "label": "city"}]
Output: {"entities": [{"value": "Jane", "label": "first_name", "reason": "first name"}, {"value": "Doe", "label": "last_name", "reason": "last name"}]}"""

    else:
        label_block = (
            "Here are the known entity classes. Strongly prefer labels from this list when they fit:\n"
            "<<VALID_CLASSES>>\n\n"
            "If no known label fits, create a concise snake_case label (e.g., clinic_name, server_name, transaction_id)."
        )
        example_block = """\
Example:
{%- if <<TAG_NOTATION>> == "xml" -%}
Input text: Jane Doe lives in <city>Santa Clara</city>. She works full-time.
{%- elif <<TAG_NOTATION>> == "bracket" -%}
Input text: Jane Doe lives in [[Santa Clara|city]]. She works full-time.
{%- elif <<TAG_NOTATION>> == "paren" -%}
Input text: Jane Doe lives in ((SENSITIVE:city|Santa Clara)). She works full-time.
{%- else -%}
Input text: Jane Doe lives in <<SENSITIVE:city>>Santa Clara<</SENSITIVE:city>>. She works full-time.
{%- endif -%}
Already-detected entities: [{"value": "Santa Clara", "label": "city"}]
Output: {"entities": [{"value": "Jane", "label": "first_name", "reason": "first name"}, {"value": "Doe", "label": "last_name", "reason": "last name"}, {"value": "full-time", "label": "employment_status", "reason": "employment status"}]}"""

    prompt = """Task: Find untagged sensitive entities in text (ignore already tagged entities). Focus on:
- Direct identifiers: Uniquely identify entities (names, emails, IDs), records (transaction IDs, case numbers), resources (file paths, URLs), or instances (server names, hostnames)
- Quasi-identifiers: Attributes that combine to narrow specificity (age, location, job title, timestamps, technical specs)
- Technical secrets: Credentials (passwords, API keys, tokens), access (internal URLs, endpoints), proprietary terms

We have the following type of data: <<DATA_SUMMARY>>

<<LABEL_BLOCK>>

Rules:
- Tag actual values, not placeholders
- Do not repeat already-tagged entities
- In structured formats, distinguish between:
  - Syntax/metadata: field names, column headers, function names, keywords
  - Data: assigned values, cell contents, literals, user input
  Only tag data, never syntax

Other information:
- unique_id rules: unique_id's are never only one character long (ex: "6" is not a unique_id). They are never event names or real words all strung together (ex: "ProcessManagementEvent" is NOT a unique_id).
- ipv4 label: This includes specific IP addresses as well as internal IP subnet ranges (ex: subnet=[224.0.0.0/4, 10.0.0.0/8]).
- Filename Exclusions: The "filename" label should be reserved for user-created documents or data exports (e.g., .pdf, .xlsx, .csv, .txt).
- Executable Distinction: Do not tag executable binaries (ending in .exe, .dll, or .sys) as filename. Treat these extensions as non-sensitive system identifiers.

<<EXAMPLE_BLOCK>>

---
Input text: <<TAGGED_TEXT>>
Already-detected entities: <<SEED_ENTITIES>>
"""
    # Pre-substitute nested placeholders inside the block strings before
    # passing them into the single-pass substitution of the main prompt.
    label_block = label_block.replace("<<VALID_CLASSES>>", ", ".join(labels))
    example_block = example_block.replace("<<TAG_NOTATION>>", COL_TAG_NOTATION)
    context_section = data_summary if data_summary else "Not provided"
    return substitute_placeholders(
        prompt,
        {
            "<<LABEL_BLOCK>>": label_block,
            "<<EXAMPLE_BLOCK>>": example_block,
            "<<TAGGED_TEXT>>": _jinja(COL_INITIAL_TAGGED_TEXT),
            "<<SEED_ENTITIES>>": _jinja(COL_SEED_ENTITIES_JSON),
            "<<DATA_SUMMARY>>": context_section,
        },
    )


def _get_latent_prompt(*, data_summary: str | None, privacy_goal: PrivacyGoal | None) -> str:
    summary_line = data_summary.strip() if data_summary else "Not provided"
    privacy_goal_text = _format_privacy_goal(privacy_goal)
    prompt = """You are performing: LATENT ENTITY & INFERENCE ANALYSIS for privacy protection.

The text will be rewritten according to this privacy goal: <<PRIVACY_GOAL>>

Goal: Identify sensitive information that is NOT explicitly stated in the text, \
but is reasonably inferable from context and could materially increase re-identification \
risk or reveal sensitive attributes about a real person. Treat inference as a first-class \
privacy surface; do not assume removing explicit identifiers is sufficient.

Return only inferences that would help an adversary narrow down or recognize the subject in the real world, or would reveal sensitive attributes about them.

Examples of *eligible* latent content when strongly supported by contextual evidence:
- Identity linkage: specific employer, school, institution, unique role, affiliation group, recognizable relationship
- Where/when linkage: home area, prior location, travel pattern, event timeframe, distinctive routine
- Personal characteristics: gender, marital status (e.g. 'married', 'single', 'divorced', 'seperated'), date of birth,
  race/ethnicity, age, education level, employment status, occupation
- Sensitive attributes: health condition/procedure/medication, legal situation, immigration status, assault/violence, substance use, etc.
- Medical procedures, screenings, or anatomy-specific tests may imply biological sex or gender. These are valid latent inferences
  when strongly supported by clinical context (e.g., Pap smear → female).
- Derived attributes inferred from contextual signals.

Non-examples (exclude):
- Generic domain nouns that do not narrow identity (e.g., "a company", "a school", "a clinic", "a court", "a neighborhood")
- Broad facts that are not person-linking or sensitive (e.g., "consistent blood levels")
- Anything explicitly stated verbatim in the text

Threat model:
Assume an adversary who can read the text in full, has general domain knowledge and access to public information, and may have partial prior familiarity with the subject.

Data type summary:
<<DATA_SUMMARY>>

Input text (identifiers already tagged inline):
---
<<TAGGED_TEXT>>
---

Rules (strict)
1) True latent only:
{%- if <<TAG_NOTATION>> == "xml" -%}
   - Do NOT repeat any already-tagged entities, e.g., <first_name>Alex</first_name>.
{%- elif <<TAG_NOTATION>> == "bracket" -%}
   - Do NOT repeat any already-tagged entities, e.g., [[Alex|first_name]].
{%- elif <<TAG_NOTATION>> == "paren" -%}
   - Do NOT repeat any already-tagged entities, e.g., ((SENSITIVE:first_name|Alex)).
{%- else -%}
   - Do NOT repeat any already-tagged entities, e.g., <<SENSITIVE:first_name>>Alex<</SENSITIVE:first_name>>.
{%- endif -%}
   - Do NOT output verbatim spans from the text.
   - You MAY output structured attributes that are logically implied by the text,
     even if the exact phrase does not appear.
   Derived attributes are allowed when they summarize or normalize multiple
   contextual cues into a structured attribute useful for identification.
   Examples:
   - “my husband passed last year” → marital_status = widowed
   - references to "undergrad", "college", or "university" → bachelor's degree
   - finance-market cues (Bloomberg screens, earnings season, roadshow, closing bell)
     → works in finance / financial analyst role
   These inferred attributes must not simply restate the SAME value already
   tagged in the text. However, normalizing or summarizing tagged information
   into a higher-level structured attribute (e.g., "college/undergrad" →
   bachelor's degree) is allowed.
2) Evidence-bounded inference:
   - Every latent entity MUST include 1-2 short quotes from the text as evidence.
   - Do NOT "jump" to a specific named entity unless the evidence pins it down strongly.
   - If multiple plausible specifics exist, DO NOT guess: either generalize (e.g., "a major Boston cancer center") or omit.
3) High signal only:
   - Prefer returning an EMPTY list over generic filler.
   Attributes such as occupation, education level, immigration status,
   ethnicity, language ability, and institutional affiliation may be
   included when strongly supported, even if individually common,
   because they become identifying when intersected.
   Education inferences are eligible when strongly supported, including
   normalized education attributes such as undergraduate degree or bachelor's-level education.
4) Non-redundant:
   - Do not output multiple variants of the same inference.
Quality checks before finalizing
- Remove any item whose value is too generic to help an adversary (e.g., "a company", "a school", "a clinic", "a court", "a neighborhood") unless additional context would allow a real person to be narrowed down.
- Remove any item that restates explicit text.
- Remove any item without clear evidence quotes.

Now produce the JSON for the input.
"""
    return substitute_placeholders(
        prompt,
        {
            "<<TAG_NOTATION>>": COL_TAG_NOTATION,
            "<<PRIVACY_GOAL>>": privacy_goal_text,
            "<<DATA_SUMMARY>>": summary_line,
            "<<TAGGED_TEXT>>": _jinja(COL_TAGGED_TEXT),
        },
    )


def _format_privacy_goal(privacy_goal: PrivacyGoal | None) -> str:
    if privacy_goal is None:
        return "Not provided"
    return privacy_goal.to_prompt_string()
