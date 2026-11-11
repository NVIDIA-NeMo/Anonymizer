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
from pydantic import BaseModel

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITIES_FOR_REPLACE,
    COL_ENTITIES_FOR_REPLACE_JSON,
    COL_ENTITY_EXAMPLES,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_SOURCE,
    COL_REPLACEMENT_RETRY_CONTEXT,
    ENTITY_LABEL_EXAMPLES,
    _jinja,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import ROW_ORDER_COL, merge_and_reorder, split_rows
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityReplacementMapSchema
from anonymizer.interface.errors import AnonymizerWorkflowError

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

    def generate_validated_maps(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        instructions: str | None = None,
        entities_column: str = COL_ENTITIES_BY_VALUE,
        preview_num_records: int | None = None,
    ) -> LlmReplaceResult:
        """Generate complete Substitute maps, retrying missing pairs once.

        Expected pairs are captured before generation. Schema-valid entries are
        filtered before checking coverage; accepted entries survive the retry.
        Rows still incomplete after that retry are reported as failed records.
        DataDesigner's schema-validation retries apply to each call. A failed
        completeness retry also fails only the affected records.
        """
        source = dataframe.copy()
        source[ROW_ORDER_COL] = range(len(source))
        if RECORD_ID_COLUMN not in source and not source.empty:
            source[RECORD_ID_COLUMN] = source.apply(NddAdapter._compute_record_id, axis=1)
        expected = [EntitiesByValueSchema.from_raw(raw) for raw in source[entities_column]]
        missing = [_entity_pairs(entities) for entities in expected]
        accepted: list[dict[str, list[dict[str, str]]]] = [{"replacements": []} for _ in expected]
        pending = [position for position, pairs in enumerate(missing) if pairs]
        if preview_num_records is not None:
            pending = pending[:preview_num_records]
        eligible = {position for position, pairs in enumerate(missing) if not pairs} | set(pending)
        failed: list[FailedRecord] = []
        dropped: set[int] = set()

        for attempt in range(2):
            if not pending:
                break
            batch = source.iloc[pending].copy()
            requested = [_entities_for_pairs(missing[position]) for position in pending]
            batch[COL_ENTITIES_FOR_REPLACE] = [_enrich_entities_for_template(entities) for entities in requested]
            batch[COL_ENTITIES_FOR_REPLACE_JSON] = batch[COL_ENTITIES_FOR_REPLACE].apply(json.dumps)
            batch[COL_ENTITY_EXAMPLES] = [_create_entity_examples(entities) for entities in requested]
            batch[COL_REPLACEMENT_RETRY_CONTEXT] = [json.dumps(accepted[position]) for position in pending]
            try:
                result = self._run_generation(
                    batch,
                    model_configs=model_configs,
                    selected_models=selected_models,
                    instructions=instructions,
                    preview_num_records=preview_num_records,
                    retry=bool(attempt),
                )
            except AnonymizerWorkflowError:
                if not attempt:
                    raise
                # A wholly failed retry may raise instead of returning an empty
                # frame. Complete records from the first attempt still survive.
                for position in pending:
                    failed.append(
                        FailedRecord(
                            record_id=str(source[RECORD_ID_COLUMN].iloc[position]),
                            step="replace-map-generation",
                            reason="Replacement map retry failed before a complete map was returned",
                        )
                    )
                dropped.update(pending)
                pending = []
                break
            failed.extend(result.failed_records)
            returned: set[int] = set()
            for _, row in result.dataframe.iterrows():
                position = int(row[ROW_ORDER_COL])
                returned.add(position)
                filtered = _filter_replacement_map_to_input_entities(
                    raw_map=row.get(COL_REPLACEMENT_MAP),
                    parsed_entities=expected[position],
                    record_id=str(source[RECORD_ID_COLUMN].iloc[position]),
                )
                additions = [
                    entry
                    for entry in filtered["replacements"]
                    if (entry["original"], entry["label"]) in missing[position]
                ]
                accepted[position]["replacements"].extend(additions)
                missing[position] -= {(entry["original"], entry["label"]) for entry in additions}

            # Positional association survives duplicate record IDs and reordered
            # workflow output. Do not retry rows already dropped by DataDesigner.
            reported = Counter(failure.record_id for failure in result.failed_records)
            for position in sorted(set(pending) - returned):
                dropped.add(position)
                record_id = str(source[RECORD_ID_COLUMN].iloc[position])
                if reported[record_id]:
                    reported[record_id] -= 1
                else:
                    failed.append(
                        FailedRecord(record_id, "replace-map-generation", "Record missing from workflow output")
                    )
            pending = [position for position in pending if position in returned and missing[position]]
            if pending and attempt == 0:
                logger.warning(
                    "Substitution maps incomplete in %d record(s); retrying %d missing pair(s) once "
                    "(missing_by_label=%s)",
                    len(pending),
                    sum(len(missing[position]) for position in pending),
                    dict(sorted(Counter(label for position in pending for _, label in missing[position]).items())),
                )

        for position in pending:
            labels = dict(sorted(Counter(label for _, label in missing[position]).items()))
            failed.append(
                FailedRecord(
                    record_id=str(source[RECORD_ID_COLUMN].iloc[position]),
                    step="replace-map-generation",
                    reason=f"Replacement map incomplete after one retry: {len(missing[position])} requested pair(s) "
                    f"have no mapping (missing_by_label={labels})",
                )
            )
        kept = sorted(eligible - dropped - set(pending))
        output = source.iloc[kept].drop(columns=[ROW_ORDER_COL]).copy().reset_index(drop=True)
        output[COL_REPLACEMENT_MAP] = pd.Series([accepted[position] for position in kept], dtype=object)
        output[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM
        return LlmReplaceResult(dataframe=output, failed_records=failed)

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
        run_result = self._run_generation(
            entity_rows,
            model_configs=model_configs,
            selected_models=selected_models,
            instructions=instructions,
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

    def _run_generation(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        instructions: str | None,
        preview_num_records: int | None,
        retry: bool = False,
    ) -> WorkflowRunResult:
        return self._adapter.run_workflow(
            dataframe,
            model_configs=model_configs,
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_REPLACEMENT_MAP,
                    prompt=_get_replacement_mapping_prompt(
                        entities_column=COL_ENTITIES_FOR_REPLACE,
                        instructions=instructions,
                        retry=retry,
                    ),
                    model_alias=resolve_model_alias("replacement_generator", selected_models),
                    output_format=EntityReplacementMapSchema,
                )
            ],
            workflow_name="replace-map-generation",
            preview_num_records=min(preview_num_records, len(dataframe)) if preview_num_records is not None else None,
        )


def _entity_pairs(entities: EntitiesByValueSchema) -> set[tuple[str, str]]:
    return {
        (entity.value, label)
        for entity in entities.entities_by_value
        for label in entity.labels
        if entity.value and label
    }


def _entities_for_pairs(pairs: set[tuple[str, str]]) -> EntitiesByValueSchema:
    grouped: dict[str, list[str]] = {}
    for value, label in sorted(pairs):
        grouped.setdefault(value, []).append(label)
    return EntitiesByValueSchema.model_validate(
        {
            "entities_by_value": [{"value": value, "labels": labels} for value, labels in grouped.items()],
        }
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
    if isinstance(raw_map, BaseModel):
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


def _get_replacement_mapping_prompt(
    *,
    entities_column: str,
    instructions: str | None = None,
    retry: bool = False,
) -> str:
    instruction_block = f"\nAdditional instructions: {instructions}\n" if instructions else ""
    prompt = """Generate synthetic replacements for sensitive entities. ONE value per entity, used consistently.
<<INSTRUCTION_BLOCK>><<RETRY_BLOCK>>

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

Examples: {{ <<ENTITY_EXAMPLES_COLUMN>> }}

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
            "<<RETRY_BLOCK>>": (
                "\nThis is the one retry for missing mappings. Generate mappings only for the entities listed below. "
                "Keep them consistent with these accepted mappings; do not change the accepted values:\n"
                + _jinja(COL_REPLACEMENT_RETRY_CONTEXT)
                + "\n"
                if retry
                else ""
            ),
            "<<ENTITIES_COLUMN>>": entities_column,
            "<<ENTITY_EXAMPLES_COLUMN>>": COL_ENTITY_EXAMPLES,
        },
    )


_EXAMPLE_LOOKUP: dict[str, str] = {
    label: f"(e.g. {', '.join(examples)})" for label, examples in ENTITY_LABEL_EXAMPLES.items()
}
