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
    COL_REPLACEMENT_RETRY_CONTEXT,
    ENTITY_LABEL_EXAMPLES,
    RECORD_ID_COLUMN,
    _jinja,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.backend import BackendRun, DataDesignerBackend
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.row_partitioning import ROW_ORDER_COL, merge_and_reorder, split_rows
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityReplacementMapSchema
from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.measurement import record_stage, stage_timer

logger = logging.getLogger("anonymizer.replace.llm_workflow")
REPLACEMENT_MAP_SOURCE_LLM = "llm"

# Workflow-internal scratch columns used only to build the replacement-generator
# prompt. Created in `generate_map_only` and dropped before returning — nothing
# downstream consumes them, and they carry pyarrow-backed pandas extension
# dtypes that would break trace_dataframe.to_parquet round-trip.
_INTERNAL_COLUMNS = (COL_ENTITY_EXAMPLES, COL_ENTITIES_FOR_REPLACE, COL_ENTITIES_FOR_REPLACE_JSON)

# Compare known space variants without folding other characters, changing
# whitespace counts, or admitting tabs and line breaks.
_UNICODE_SPACE_TRANSLATION = str.maketrans({"\u00a0": " ", "\u2009": " ", "\u202f": " "})


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

    def generate_validated_maps(
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
        """Generate complete Substitute maps, retrying missing pairs once.

        Expected pairs are captured before generation. Schema-valid entries are
        filtered before checking coverage; accepted entries survive the retry.
        Rows still incomplete after that retry are reported as failed records.
        DataDesigner's schema-validation retries apply to each call. A failed
        completeness retry also fails only the affected records. The complete
        maps then go through ``_fold_replacement_maps`` under ``consistency``
        like any other proposal.
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
            batch = source.iloc[pending].drop(columns=[ROW_ORDER_COL])
            requested = [_entities_for_pairs(missing[position]) for position in pending]
            batch[COL_ENTITY_EXAMPLES] = [_create_entity_examples(entities) for entities in requested]
            batch[COL_ENTITIES_FOR_REPLACE] = [_enrich_entities_for_template(entities) for entities in requested]
            batch[COL_ENTITIES_FOR_REPLACE_JSON] = batch[COL_ENTITIES_FOR_REPLACE].apply(json.dumps)
            batch[ROW_ORDER_COL] = pending
            if attempt:
                # The retry sees the accepted entries as one record's fold would show them; the run's fold comes last.
                batch[COL_REPLACEMENT_RETRY_CONTEXT] = [
                    json.dumps(_fold_replacement_maps([accepted[p]], "record", [_requested_values(expected[p])])[0])
                    for p in pending
                ]
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
                    reconcile_unicode_whitespace=True,
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
        output[COL_REPLACEMENT_MAP] = _fold_frame(output, entities_column, consistency)
        return LlmReplaceResult(dataframe=output, failed_records=failed, source_positions=tuple(kept))

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
        run_result = self._run_generation(
            entity_rows,
            model_configs=model_configs,
            selected_models=selected_models,
            instructions=instructions,
            preview_num_records=preview_num_records,
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
        combined[COL_REPLACEMENT_MAP] = _fold_frame(combined, entities_column, consistency)
        return LlmReplaceResult(
            dataframe=combined.drop(columns=list(_INTERNAL_COLUMNS), errors="ignore"),
            failed_records=run_result.failed_records,
            source_positions=positions,
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
    ) -> BackendRun:
        return self._backend.run(
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


def _fold_frame(frame: pd.DataFrame, entities_column: str, consistency: Literal["record", "run"]) -> list[dict]:
    """``_fold_replacement_maps`` over ``frame``'s maps, recorded as the ``LlmReplaceWorkflow.fold`` stage."""
    requested = [_requested_values(EntitiesByValueSchema.from_raw(v)) for v in frame[entities_column]]
    tally: Counter[tuple[str, ...]] = Counter()
    with stage_timer("LlmReplaceWorkflow.fold", consistency=consistency, input_row_count=len(frame)) as fields:
        maps = _fold_replacement_maps(frame[COL_REPLACEMENT_MAP].tolist(), consistency, requested, tally)
        fields["entry_count"] = sum(len(m.get("replacements", [])) for m in maps)
        fields["records_with_placeholder"] = tally.pop(("records_with_placeholder",), 0)
        fallbacks: dict[str, dict[str, int]] = {}
        for (label, step), n in sorted(tally.items()):
            fallbacks.setdefault(label, {})[step] = n
        fields["fallback_counts"] = fallbacks
    return maps


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

    ``"record"`` shares within each record; ``"run"`` also gives each key the first record's substitute. A shared
    substitute collides when its folded form equals a value in its record (its map's originals and ``requested``), a
    substitute another key already emitted there, or another key's own proposal; a colliding entry falls back to its own
    substitute. In ``"record"`` mode that fallback is kept unless it equals a value in its record, so two keys of a
    record can share a substitute; otherwise it becomes a placeholder. The first entry of a key fixes the record's
    shared substitute; a later colliding variant falls back for itself only, so a third variant still tries the shared
    one.

    In ``"run"`` mode the fallback also collides with a substitute another key emitted in the record, and a substitute
    emitted for one key anywhere in the run collides for every other key, so two keys never share one; placeholders are
    numbered across the run. A key holding a run-wide mapping ignores other keys' proposals, so its run-wide substitute
    wins over them. When that substitute collides in a record and the key falls back, its fallback is checked like any
    other key's: a key earlier in the record that already emitted the same value keeps it, and the holder moves on to a
    placeholder.

    When given, ``tally`` counts each fallback as ``(label, step)`` with step ``own``, ``placeholder_taken``
    (``"run"`` only) or ``placeholder_protected`` (its own proposal folds to a value in its record), plus
    ``("records_with_placeholder",)``.
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
                value = _fold(synthetic)
                if value in originals if consistency == "record" else collides(value, key, shared=False):
                    in_record = value in originals
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
    reconcile_unicode_whitespace: bool = False,
) -> dict[str, list[dict[str, str]]]:
    """Keep requested entries, optionally reconciling unambiguous Unicode spaces.

    Exact entries take precedence. Reconciliation compares the original under
    the same exact label against the full requested inventory, then restores
    the trusted original. Ambiguous or otherwise changed originals are dropped.
    Collisions between a substitute and a value are left to
    ``_fold_replacement_maps``.
    """
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
    exact_pairs = {(entry.original, entry.label) for entry in parsed_map.replacements} & allowed_pairs
    equivalent_pairs: dict[tuple[str, str], set[tuple[str, str]]] = {}
    if reconcile_unicode_whitespace:
        for value, label in allowed_pairs:
            key = (value.translate(_UNICODE_SPACE_TRANSLATION), label)
            equivalent_pairs.setdefault(key, set()).add((value, label))
    filtered: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    reconciled_labels: Counter[str] = Counter()
    for replacement in parsed_map.replacements:
        key = (replacement.original, replacement.label)
        if key not in allowed_pairs:
            normalized_key = (replacement.original.translate(_UNICODE_SPACE_TRANSLATION), replacement.label)
            candidates = equivalent_pairs.get(normalized_key, set())
            if len(candidates) != 1:
                continue
            key = next(iter(candidates))
            if key in exact_pairs:
                continue
        if key in seen:
            continue
        if replacement.original != key[0]:
            replacement = replacement.model_copy(update={"original": key[0]})
            reconciled_labels[replacement.label] += 1
        seen.add(key)
        filtered.append(replacement.model_dump())

    if reconciled_labels:
        logger.debug(
            "Replacement map reconciled Unicode whitespace for record %s; reconciled=%d (reconciled_by_label=%s)",
            record_id or "<unknown>",
            sum(reconciled_labels.values()),
            dict(sorted(reconciled_labels.items())),
        )
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


def _placeholder_prefix(label: str) -> str:
    label_token = "".join(char.upper() if char.isalnum() else "_" for char in label).strip("_") or "VALUE"
    return f"[SUBSTITUTE_{label_token}_"


def is_placeholder(label: str, synthetic: str) -> bool:
    """Whether ``synthetic`` is a placeholder ``_collision_safe_synthetic`` builds for ``label``, in any case (the fold
    re-cases a substitute to each variant of its original)."""
    prefix = _placeholder_prefix(label).casefold()
    return synthetic.casefold().startswith(prefix) and synthetic.endswith("]") and synthetic[len(prefix) : -1].isdigit()


def _collision_safe_synthetic(label: str, *, index: int, protected_original_values: tuple[Container[str], ...]) -> str:
    prefix = _placeholder_prefix(label)
    while True:
        candidate = f"{prefix}{index}]"
        if not any(_fold(candidate) in values for values in protected_original_values):
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
   - be a closely related specialization of the same role

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

6. Keep replacements valid in their surrounding structure, including dictionary keys.
   Do not introduce duplicate keys or change value types.

CRITICAL: Every replacement must differ from its original, and distinct original values
must not collapse to the same replacement.

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
