# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Container
from dataclasses import dataclass
from typing import Literal

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
    ENTITY_LABEL_EXAMPLES,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.backend import DataDesignerBackend
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import ROW_ORDER_COL, merge_and_reorder, split_rows
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityReplacementMapSchema
from anonymizer.measurement import record_stage, stage_timer

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
    source_positions: tuple[int, ...] | None = None


class LlmReplaceWorkflow:
    """Generate replacement maps via LLM workflow."""

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter
        self._backend = DataDesignerBackend(adapter)

    def generate_map_only(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        instructions: str | None = None,
        entities_column: str = COL_ENTITIES_BY_VALUE,
        preview_num_records: int | None = None,
        consistency: Literal["record", "run"] = "record",
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
            record_stage(
                "LlmReplaceWorkflow.fold",
                elapsed_sec=0.0,
                consistency=consistency,
                input_row_count=len(dataframe),
                entry_count=0,
                records_with_placeholder=0,
                fallback_counts={},
            )
            return LlmReplaceResult(
                dataframe=merge_and_reorder(passthrough_rows).drop(columns=list(_INTERNAL_COLUMNS), errors="ignore"),
                failed_records=[],
                source_positions=tuple(range(len(dataframe))),
            )

        # Only rows with actual entities are sent to the LLM.
        run_result = self._backend.run(
            entity_rows,
            model_configs=model_configs,
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_REPLACEMENT_MAP,
                    prompt=_get_replacement_mapping_prompt(
                        entities_column=COL_ENTITIES_FOR_REPLACE,
                        instructions=instructions,
                    ),
                    model_alias=resolve_model_alias("replacement_generator", selected_models),
                    output_format=EntityReplacementMapSchema,
                )
            ],
            workflow_name="replace-map-generation",
            preview_num_records=min(preview_num_records, len(entity_rows)) if preview_num_records is not None else None,
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

        # Input positions of the rows that came back, in output order: the rows the LLM dropped are absent.
        positions = tuple(sorted((*passthrough_rows[ROW_ORDER_COL], *output_df[ROW_ORDER_COL])))
        combined = merge_and_reorder(output_df, passthrough_rows)
        requested = [_requested_values(EntitiesByValueSchema.from_raw(v)) for v in combined[entities_column]]
        tally: Counter[tuple[str, ...]] = Counter()
        with stage_timer("LlmReplaceWorkflow.fold", consistency=consistency, input_row_count=len(combined)) as fields:
            maps = _fold_replacement_maps(combined[COL_REPLACEMENT_MAP].tolist(), consistency, requested, tally)
            fields["entry_count"] = sum(len(m.get("replacements", [])) for m in maps)
            fields["records_with_placeholder"] = tally.pop(("records_with_placeholder",), 0)
            fallbacks: dict[str, dict[str, int]] = {}
            for (label, step), n in sorted(tally.items()):
                fallbacks.setdefault(label, {})[step] = n
            fields["fallback_counts"] = fallbacks
        combined[COL_REPLACEMENT_MAP] = maps
        return LlmReplaceResult(
            dataframe=combined.drop(columns=list(_INTERNAL_COLUMNS), errors="ignore"),
            failed_records=run_result.failed_records,
            source_positions=positions,
        )


def _fold(value: str) -> str:
    return " ".join(value.casefold().split())


def _recase(synthetic: str, *, like: str, source: str) -> str:
    if like == source:
        return synthetic
    if like.isupper():
        return synthetic.upper()
    if like.islower():
        return synthetic.lower()
    return synthetic.title() if like[:1].isupper() and (synthetic.isupper() or synthetic.islower()) else synthetic


def _fold_replacement_maps(
    maps: list[dict],
    consistency: Literal["record", "run"],
    requested: list[set[str]] | None = None,
    tally: Counter[tuple[str, ...]] | None = None,
) -> list[dict]:
    """Give entries with the same (label, case- and whitespace-folded value) one substitute, re-cased to each original.

    ``"record"`` shares within each record; ``"run"`` also gives each key the first record's substitute. An emitted
    substitute collides when its folded form equals a value in its record (its map's originals and ``requested``) or a substitute another key already emitted
    there; a shared substitute also collides with another key's own proposal. A colliding entry falls back to its own
    substitute, then to a placeholder. The first entry of a key fixes the record's shared substitute; a later
    colliding variant falls back for itself only, so a third variant still tries the shared one. In ``"run"`` mode a
    substitute emitted for one key anywhere in the run collides for every other key, placeholders are numbered across
    the run, and a key holding a run-wide mapping ignores other keys' proposals, even when it uses its own fallback in
    that record, so the proposer falls back.

    When given, ``tally`` counts each fallback as ``(label, step)`` with step ``own``, ``placeholder_taken`` or
    ``placeholder_protected`` (its own proposal folds to a value in its record), plus ``("records_with_placeholder",)``.
    """
    run_wide: dict[tuple[str, str], tuple[str, str]] = {}
    taken: dict[str, set[tuple[str, str]]] = {}
    run_placeholders: Counter[str] = Counter()
    folded: list[dict] = []
    for index, raw in enumerate(maps):
        keyed = [((e["label"], _fold(e["original"])), e) for e in raw.get("replacements", [])]
        originals = {key[1] for key, _ in keyed} | {_fold(v) for v in (requested[index] if requested else ())}
        proposed: dict[str, set[tuple[str, str]]] = {}
        for key, entry in keyed:
            proposed.setdefault(_fold(entry["synthetic"]), set()).add(key)
        emitted: dict[str, set[tuple[str, str]]] = {}

        def collides(value: str, key: tuple[str, str], *, shared: bool) -> bool:
            rivals = proposed.get(value, set()) if shared and key not in run_wide else set()
            others = rivals | emitted.get(value, set()) | taken.get(value, set())
            return value in originals or bool(others - {key})

        record: dict[tuple[str, str], tuple[str, str]] = {}
        placeholders: Counter[str] = run_placeholders if consistency == "run" else Counter()
        used = 0
        out = []
        for key, entry in keyed:
            own = (entry["original"], entry["synthetic"])
            first = key not in record
            source = record.setdefault(key, run_wide.get(key, own))
            synthetic = _recase(source[1], like=entry["original"], source=source[0])
            if collides(_fold(synthetic), key, shared=True):
                step = "own" if entry["synthetic"] != synthetic else None
                source = own
                synthetic = entry["synthetic"]
                if collides(_fold(synthetic), key, shared=False):
                    in_record = _fold(synthetic) in originals
                    step = "placeholder_protected" if in_record else "placeholder_taken"
                    placeholders[entry["label"]] += 1
                    used += 1
                    synthetic = _collision_safe_synthetic(
                        entry["label"],
                        index=placeholders[entry["label"]],
                        protected_original_values=(originals, proposed, emitted, taken),
                    )
                    source = (entry["original"], synthetic)
                if first:
                    record[key] = source
                if tally is not None and step:
                    tally[(entry["label"], step)] += 1
            emitted.setdefault(_fold(synthetic), set()).add(key)
            out.append({**entry, "synthetic": synthetic})
        if used and tally is not None:
            tally[("records_with_placeholder",)] += 1
        if used:
            logger.warning("Replacement map used placeholder substitutes to avoid collisions; replaced=%d", used)
        if consistency == "run":
            for key, source in record.items():
                run_wide.setdefault(key, source)
            for value, keys in emitted.items():
                taken.setdefault(value, set()).update(keys)
        folded.append({**raw, "replacements": out})
    return folded


def _requested_values(parsed: EntitiesByValueSchema) -> set[str]:
    return {entity.value for entity in parsed.entities_by_value if entity.value}


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
            "Replacement map has unexpected type.",
        )
        return {"replacements": []}

    parsed_map = EntityReplacementMapSchema.model_validate(raw_map)

    allowed_pairs = {
        (entity.value, label)
        for entity in parsed_entities.entities_by_value
        for label in entity.labels
        if entity.value and label
    }
    filtered: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for replacement in parsed_map.replacements:
        key = (replacement.original, replacement.label)
        if key not in allowed_pairs or key in seen:
            continue
        seen.add(key)
        filtered.append(replacement.model_dump())

    if logger.isEnabledFor(logging.DEBUG):
        raw_pairs = {(r.original, r.label) for r in parsed_map.replacements}
        filtered_pairs = {(f["original"], f["label"]) for f in filtered}
        unrequested_labels = Counter(label for _, label in (raw_pairs - allowed_pairs))
        unfilled_labels = Counter(label for _, label in (allowed_pairs - filtered_pairs))
        logger.debug(
            "Replacement map: requested=%d raw=%d filtered=%d unrequested=%d unfilled=%d",
            len(allowed_pairs),
            len(parsed_map.replacements),
            len(filtered),
            sum(unrequested_labels.values()),
            sum(unfilled_labels.values()),
        )
    if not filtered and allowed_pairs:
        logger.warning(
            "Replacement map empty after filtering; requested=%d raw=%d",
            len(allowed_pairs),
            len(parsed_map.replacements),
        )
    return {"replacements": filtered}


def _collision_safe_synthetic(label: str, *, index: int, protected_original_values: tuple[Container[str], ...]) -> str:
    label_token = "".join(char.upper() if char.isalnum() else "_" for char in label).strip("_") or "VALUE"
    while True:
        candidate = f"[SUBSTITUTE_{label_token}_{index}]"
        if not any(_fold(candidate) in values for values in protected_original_values):
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
            "<<ENTITIES_COLUMN>>": entities_column,
            "<<ENTITY_EXAMPLES_COLUMN>>": COL_ENTITY_EXAMPLES,
        },
    )


_EXAMPLE_LOOKUP: dict[str, str] = {
    label: f"(e.g. {', '.join(examples)})" for label, examples in ENTITY_LABEL_EXAMPLES.items()
}
