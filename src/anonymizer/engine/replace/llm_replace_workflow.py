# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITIES_FOR_REPLACE,
    COL_ENTITIES_FOR_REPLACE_JSON,
    COL_ENTITY_EXAMPLES,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_SOURCE,
    ENTITY_LABEL_EXAMPLES,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import merge_and_reorder, split_rows
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityReplacementMapSchema

logger = logging.getLogger("anonymizer.replace.llm_workflow")
REPLACEMENT_MAP_SOURCE_LLM = "llm"

# Workflow-internal scratch columns used only to build the replacement-generator
# prompt. Created in `generate_map_only` and dropped before returning — nothing
# downstream consumes them, and they carry pyarrow-backed pandas extension
# dtypes that would break trace_dataframe.to_parquet round-trip.
_INTERNAL_COLUMNS = (COL_ENTITY_EXAMPLES, COL_ENTITIES_FOR_REPLACE, COL_ENTITIES_FOR_REPLACE_JSON)


@dataclass(frozen=True)
class LlmReplaceResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class LlmReplaceWorkflow:
    """Generate replacement maps via LLM workflow."""

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def generate_map_only(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        instructions: str | None = None,
        entities_column: str = COL_ENTITIES_BY_VALUE,
        preview_num_records: int | None = None,
    ) -> LlmReplaceResult:
        replace_alias = resolve_model_alias("replacement_generator", selected_models)

        working_df = dataframe.copy()

        # Parse the per-row entity payload once, then reuse it for prompt inputs.
        parsed_entities = working_df[entities_column].apply(EntitiesByValueSchema.from_raw)
        working_df[COL_ENTITY_EXAMPLES] = parsed_entities.apply(_create_entity_examples)
        working_df[COL_ENTITIES_FOR_REPLACE] = parsed_entities.apply(_enrich_entities_for_template)
        working_df[COL_ENTITIES_FOR_REPLACE_JSON] = working_df[COL_ENTITIES_FOR_REPLACE].apply(json.dumps)

        # Partition: rows with an empty entity list bypass replacement-map generation.
        entity_rows, passthrough_rows = split_rows(working_df, column=COL_ENTITIES_FOR_REPLACE, predicate=bool)
        passthrough_rows[COL_REPLACEMENT_MAP] = [{"replacements": []} for _ in range(len(passthrough_rows))]
        passthrough_rows[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM

        if entity_rows.empty:
            passthrough_only = merge_and_reorder(passthrough_rows)
            return LlmReplaceResult(
                dataframe=passthrough_only.drop(columns=list(_INTERNAL_COLUMNS), errors="ignore"),
                failed_records=[],
            )

        # Only rows with actual entities are sent to the LLM.
        effective_preview_num_records = (
            min(preview_num_records, len(entity_rows)) if preview_num_records is not None else None
        )
        run_result = self._adapter.run_workflow(
            entity_rows,
            model_configs=model_configs,
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_REPLACEMENT_MAP,
                    prompt=_get_replacement_mapping_prompt(
                        entities_column=COL_ENTITIES_FOR_REPLACE,
                        instructions=instructions,
                    ),
                    model_alias=replace_alias,
                    output_format=EntityReplacementMapSchema,
                )
            ],
            workflow_name="replace-map-generation",
            preview_num_records=effective_preview_num_records,
        )
        output_df = run_result.dataframe.copy()
        # Drop any LLM-returned replacements that were not requested for this row.
        output_df[COL_REPLACEMENT_MAP] = output_df.apply(
            lambda row: _filter_replacement_map_to_input_entities(
                raw_map=row.get(COL_REPLACEMENT_MAP, {"replacements": []}),
                parsed_entities=EntitiesByValueSchema.from_raw(row.get(entities_column, {})),
                record_id=str(row.get("_anonymizer_record_id", "")),
            ),
            axis=1,
        )
        output_df[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM

        combined = merge_and_reorder(output_df, passthrough_rows)
        return LlmReplaceResult(
            dataframe=combined.drop(columns=list(_INTERNAL_COLUMNS), errors="ignore"),
            failed_records=run_result.failed_records,
        )


def _enrich_entities_for_template(parsed: EntitiesByValueSchema) -> list[dict[str, str | list[str]]]:
    """Add ``labels_str`` for Jinja template rendering."""
    return [{"value": e.value, "labels": e.labels, "labels_str": ", ".join(e.labels)} for e in parsed.entities_by_value]


def _create_entity_examples(parsed: EntitiesByValueSchema) -> str:
    labels: set[str] = set()
    for entity in parsed.entities_by_value:
        labels.update(label for label in entity.labels if label)
    if not labels:
        return ""
    examples = {
        label: _EXAMPLE_LOOKUP.get(label, "(generate realistic synthetic replacement)") for label in sorted(labels)
    }
    return json.dumps(examples, ensure_ascii=True)


def _filter_replacement_map_to_input_entities(
    *,
    raw_map: object,
    parsed_entities: EntitiesByValueSchema,
    record_id: str = "",
) -> dict[str, list[dict[str, str]]]:
    """Keep only replacement entries that correspond to actual requested entities."""
    if hasattr(raw_map, "model_dump"):
        raw_map = raw_map.model_dump(mode="python")
    if not isinstance(raw_map, dict):
        logger.warning(
            "Replacement map has unexpected type for record %s: %s",
            record_id or "<unknown>",
            type(raw_map).__name__,
        )
        return {"replacements": []}

    parsed_map = EntityReplacementMapSchema.model_validate(raw_map)

    allowed_pairs = {
        (entity.value, label)
        for entity in parsed_entities.entities_by_value
        for label in entity.labels
        if entity.value and label
    }
    protected_original_values = {value for value, _ in allowed_pairs}
    filtered: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    synthetic_collision_labels: Counter[str] = Counter()
    for replacement in parsed_map.replacements:
        key = (replacement.original, replacement.label)
        if key not in allowed_pairs or key in seen:
            continue
        if replacement.synthetic in protected_original_values:
            synthetic_collision_labels[replacement.label] += 1
            seen.add(key)
            filtered.append(
                {
                    "original": replacement.original,
                    "label": replacement.label,
                    "synthetic": _collision_safe_synthetic(
                        replacement.label,
                        index=synthetic_collision_labels[replacement.label],
                        protected_original_values=protected_original_values,
                    ),
                }
            )
            continue
        seen.add(key)
        filtered.append(replacement.model_dump())

    if synthetic_collision_labels:
        logger.warning(
            "Replacement map repaired synthetic-original collision entries for record %s; repaired=%d "
            "(repaired_by_label=%s)",
            record_id or "<unknown>",
            sum(synthetic_collision_labels.values()),
            dict(synthetic_collision_labels),
        )
    if logger.isEnabledFor(logging.DEBUG):
        raw_pairs = {(r.original, r.label) for r in parsed_map.replacements}
        filtered_pairs = {(f["original"], f["label"]) for f in filtered}
        unrequested_labels = Counter(label for _, label in (raw_pairs - allowed_pairs))
        unfilled_labels = Counter(label for _, label in (allowed_pairs - filtered_pairs))
        logger.debug(
            "Replacement map record %s: requested=%d raw=%d filtered=%d%s%s%s",
            record_id or "<unknown>",
            len(allowed_pairs),
            len(parsed_map.replacements),
            len(filtered),
            f" unrequested_by_label={dict(unrequested_labels)}" if unrequested_labels else "",
            f" unfilled_by_label={dict(unfilled_labels)}" if unfilled_labels else "",
            f" synthetic_original_collision_by_label={dict(synthetic_collision_labels)}"
            if synthetic_collision_labels
            else "",
        )
    if not filtered and allowed_pairs:
        requested_labels = Counter(label for _, label in allowed_pairs)
        logger.warning(
            "Replacement map empty after filtering for record %s; requested=%d raw=%d (requested_by_label=%s)",
            record_id or "<unknown>",
            len(allowed_pairs),
            len(parsed_map.replacements),
            dict(requested_labels),
        )
    return {"replacements": filtered}


def _collision_safe_synthetic(label: str, *, index: int, protected_original_values: set[str]) -> str:
    label_token = "".join(char.upper() if char.isalnum() else "_" for char in label).strip("_") or "VALUE"
    while True:
        candidate = f"[SUBSTITUTE_{label_token}_{index}]"
        if candidate not in protected_original_values:
            return candidate
        index += 1


def _get_replacement_mapping_prompt(*, entities_column: str, instructions: str | None = None) -> str:
    instruction_block = f"\nAdditional instructions: {instructions}\n" if instructions else ""
    prompt = """Generate synthetic replacements for sensitive entities. ONE value per entity, used consistently.
<<INSTRUCTION_BLOCK>>

The replacements must:
   - prevent re-identification
   - remain plausible in context
   - match the grammatical role of the original value
    (e.g., verbs remain verbs, nouns remain nouns)
   - be of the same class label as the original

The replacements must NOT:
   - be a synonym or near-synonym of the original value
   - e a closely related specialization of the same role

Prefer replacements that shift the concept to a different but plausible value within a related domain, \
or a nearby role that is clearly distinct from the original.
Avoid overly broad or vague generalizations.

Examples of bad replacements that are too close in meaning:
"mentor" -> "advisor"
"schoolteacher" -> "educator"

Example of a good replacement that shifts subfields:
"costume designer" -> "set designer"

Context: {{ tagged_text }}

Entities to replace:
{%- for entity in <<ENTITIES_COLUMN>> %}
- "{{ entity.value }}" ({{ entity.labels_str }})
{%- endfor %}

Per-label type references (generate a NEW realistic value of that kind; do not reuse these example values, and never copy this reference text literally): {{ <<ENTITY_EXAMPLES_COLUMN>> }}

Rules:
1. Related entities must stay consistent:
   - Geographic: city/state/zip must match (Portland→Austin, Oregon→Texas, 97201→78701)
   - Personal: name/email must match (Sarah Chen→Michael Torres, sarah.chen@x.com→michael.torres@x.com)
   - Organizational: company/domain must match (Acme Corp→TechStart, acme.com→techstart.com)
   - Temporal: age/birthdate must match (DOB 1989-05-15→1985-03-20, age 35→39)
   - Contact: phone country code/country must match (+1→+44, USA→UK)

2. Preserve wildcards and patterns:
   - CHANGE concrete parts, KEEP wildcards (* % ?) in same positions
   - "10.0.*.*" → "192.168.*.*" (changed 10.0, kept *.*)
   - "file_*.log" → "output_*.log" (changed file, kept *)
   - "user_%@%.com" → "person_%@%.net" (changed user/com, kept %@%)
   - DON'T return original unchanged! Change the non-wildcard parts

3. Maintain format and type:
   - Same structure, length patterns, character types
   - Same demographic characteristics (Indian name → Indian name)
   - Do not add new words that were not present in the original span.
     Match the structural pattern of the span.

4. Fit the surrounding text naturally:
   - Check that the synthetic value reads correctly with the words immediately before and after it in the original text.

5. When replacing geographic locations, keep the replacements within
   a coherent region so that travel routes and residences remain plausible.

CRITICAL: Every entity MUST have a different synthetic value. Never return original=synthetic.

Before generating replacements, verify:
   - That each new_value is clearly different in meaning from the original and is not a synonym or simple rewording.
   - That all geographic replacements remain mutually consistent (cities belong to the replaced state/region, and travel routes remain plausible).
"""
    return substitute_placeholders(
        prompt,
        {
            "<<INSTRUCTION_BLOCK>>": instruction_block,
            "<<ENTITIES_COLUMN>>": entities_column,
            "<<ENTITY_EXAMPLES_COLUMN>>": COL_ENTITY_EXAMPLES,
        },
    )


_EXAMPLE_LOOKUP: dict[str, str] = {
    label: f"such as {', '.join(examples)}" for label, examples in ENTITY_LABEL_EXAMPLES.items()
}
