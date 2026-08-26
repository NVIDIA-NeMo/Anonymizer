# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
from typing import Any

import pandas as pd
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT

from anonymizer.config.models import RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_FINAL_ENTITIES,
    COL_FULL_REWRITE,
    COL_REPLACEMENT_APPLICATION,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_BASELINE_TEXT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_REWRITE_REPLACEMENT_READY,
    COL_REWRITE_TAGGED_TEXT,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.detection.postprocess import EntitySpan, build_tagged_text
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.prompt_utils import substitute_placeholders
from anonymizer.engine.replace.strategies import (
    ReplacementEntry,
    _parse_replacements,
    apply_replacements_to_spans,
)
from anonymizer.engine.rewrite.parsers import parse_sensitivity_disposition
from anonymizer.engine.schemas import (
    EntitiesSchema,
    EntitySchema,
    RewriteOutputSchema,
)

logger = logging.getLogger("anonymizer.rewrite.generation")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def _get_rewrite_prompt(privacy_goal: PrivacyGoal, data_summary: str | None = None) -> str:
    """Build the full rewrite prompt with XML section headers."""
    data_context_section = ""
    if data_summary and data_summary.strip():
        data_context_section = "\n<data_context>\nDataset description: " + data_summary.strip() + "\n</data_context>\n"

    prompt = """You are an expert writer. You excel at rewriting, paraphrasing, rewording, and following instructions.

<instructions>
Your task is to rewrite the text below so that it protects the privacy of the entities described,
following the entity protection rules and replacement map provided. The rewrite must read naturally as
plain, fluent text — no tags, brackets, or annotation artifacts.

Apply each protection decision consistently across ALL occurrences of the same entity value.
Do not add justification text or commentary in the output. Only output the rewritten text.
</instructions>

<privacy_goal>
<<PRIVACY_GOAL>>
</privacy_goal>
<<DATA_CONTEXT>>
<input>
The text below contains inline entity tags marking identified entities.
{% if <<TAG_NOTATION>> == 'bracket' %}Tags use the format [[entity_value|entity_label]]. Remove all [[...]] tags.
{% elif <<TAG_NOTATION>> == 'xml' %}Tags use the format <entity_label>entity_value</entity_label>. Remove all XML entity tags.
{% elif <<TAG_NOTATION>> == 'paren' %}Tags use the format ((SENSITIVE:entity_label|entity_value)). Remove all ((SENSITIVE:...)) tags.
{% elif <<TAG_NOTATION>> == 'sentinel' %}Tags use the format <<SENSITIVE:entity_label>>entity_value<</SENSITIVE:entity_label>>. Remove all <<SENSITIVE:...>> tags.
{% endif %}
The rewritten text must read like normal prose with no tags remaining.

Tagged text:
<<TAGGED_TEXT>>
</input>

<sensitivity_disposition>
Protection decisions for each entity that needs protection:
{% for entity in <<REWRITE_DISPOSITION_BLOCK>> %}
- {{ entity.entity_label }}: "{{ entity.entity_value }}"
  Sensitivity: {{ entity.sensitivity }}
  Protection method: {{ entity.protection_method_suggestion }}
  Reason: {{ entity.protection_reason }}
{% endfor %}

Entities NOT listed above may be kept as-is.
</sensitivity_disposition>

{% if <<REPLACEMENT_MAP_COL>>.replacements %}
<replacement_map>
Synthetic replacement values for entities with protection_method "replace":
<<REPLACEMENT_MAP>>
</replacement_map>
{% endif %}
<output_requirements>
Apply each protection method as follows:
- "replace": Substitute the entity value with the corresponding synthetic value from the replacement map.
  Use the synthetic value consistently for every occurrence.
- "generalize": Replace with a broader category or range
  (e.g., a specific city → "a city in the Pacific Northwest", exact age → "in their late 30s").
- "remove": Omit the detail entirely. Rewrite the surrounding sentence so it reads naturally without it.
- "suppress_inference": Modify the text so the attribute cannot be reliably inferred by a motivated reader.

Rules:
1. ALL entity tags (as described above) must be removed. Output must be plain text.
2. Apply changes consistently — the same entity value must be treated the same way everywhere it appears.
3. Entities with protection_method_suggestion="leave_as_is" should be retained verbatim (tags removed only).
4. The rewritten text must flow naturally and preserve the meaning and narrative structure of the original.
5. Do not introduce new identifying details not present in the original.
</output_requirements>"""
    return substitute_placeholders(
        prompt,
        {
            "<<PRIVACY_GOAL>>": privacy_goal.to_prompt_string(),
            "<<DATA_CONTEXT>>": data_context_section,
            "<<TAG_NOTATION>>": COL_TAG_NOTATION,
            "<<TAGGED_TEXT>>": _jinja(COL_REWRITE_TAGGED_TEXT),
            "<<REWRITE_DISPOSITION_BLOCK>>": COL_REWRITE_DISPOSITION_BLOCK,
            "<<REPLACEMENT_MAP_COL>>": COL_REPLACEMENT_MAP_FOR_PROMPT,
            "<<REPLACEMENT_MAP>>": _jinja(COL_REPLACEMENT_MAP_FOR_PROMPT),
        },
    )


# ---------------------------------------------------------------------------
# Custom column generators (pure Python, no LLM)
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_SENSITIVITY_DISPOSITION])
def _format_rewrite_disposition_block(row: dict[str, Any]) -> dict[str, Any]:
    """Pre-filter and serialize protected entities (protection_method_suggestion != "leave_as_is") for the rewrite prompt."""
    disposition = parse_sensitivity_disposition(row[COL_SENSITIVITY_DISPOSITION])
    block = []
    for e in disposition.sensitivity_disposition:
        if not e.needs_protection:
            continue
        d = e.model_dump(mode="json")
        block.append(
            {
                "entity_label": d["entity_label"],
                "entity_value": d["entity_value"],
                "sensitivity": d["sensitivity"],
                "protection_method_suggestion": d["protection_method_suggestion"],
                "protection_reason": d["protection_reason"],
            }
        )
    row[COL_REWRITE_DISPOSITION_BLOCK] = block
    return row


@custom_column_generator(required_columns=[COL_REPLACEMENT_MAP, COL_REWRITE_DISPOSITION_BLOCK])
def _filter_replacement_map_for_prompt(row: dict[str, Any]) -> dict[str, Any]:
    """Keep only replacement entries for entities with protection_method_suggestion='replace'."""
    disposition_block: list[dict] = row.get(COL_REWRITE_DISPOSITION_BLOCK, [])
    replace_pairs = {
        (str(e.get("entity_value", "")), str(e.get("entity_label", "")))
        for e in disposition_block
        if e.get("protection_method_suggestion") == "replace"
    }
    raw_map = row.get(COL_REPLACEMENT_MAP)
    if raw_map is None:
        if replace_pairs:
            logger.warning(
                "COL_REPLACEMENT_MAP is None but entities require replacement; prompt will have no replacements."
            )
        row[COL_REPLACEMENT_MAP_FOR_PROMPT] = {"replacements": []}
        return row
    filtered = [
        {"original": replacement.original, "label": replacement.label, "synthetic": replacement.synthetic}
        for replacement in _parse_replacements(raw_map)
        if (replacement.original, replacement.label) in replace_pairs
    ]
    row[COL_REPLACEMENT_MAP_FOR_PROMPT] = {"replacements": filtered}
    return row


@custom_column_generator(
    required_columns=[
        COL_TEXT,
        COL_FINAL_ENTITIES,
        COL_REPLACEMENT_MAP,
        COL_REWRITE_DISPOSITION_BLOCK,
        COL_TAG_NOTATION,
        COL_TAGGED_TEXT,
    ],
    side_effect_columns=[
        COL_REPLACEMENT_APPLICATION,
        COL_REWRITE_REPLACEMENT_READY,
        COL_REWRITE_BASELINE_TEXT,
    ],
)
def _prepare_rewrite_tagged_text(row: dict[str, Any]) -> dict[str, Any]:
    """Apply strict, label-aware replacements before the LLM sees rewrite input."""
    entities = EntitiesSchema.from_raw(row.get(COL_FINAL_ENTITIES, {}))
    replace_pairs = _replace_pairs(row.get(COL_REWRITE_DISPOSITION_BLOCK, []))
    target_entities = EntitiesSchema(entities=[e for e in entities.entities if (e.value, e.label) in replace_pairs])
    replacements = _parse_replacements(row.get(COL_REPLACEMENT_MAP))
    baseline, application = apply_replacements_to_spans(
        str(row.get(COL_TEXT, "")), target_entities, replacements, allow_value_fallback=False
    )
    metrics: dict[str, Any] = application.to_metrics()
    # DataDesigner checkpoints side-effect columns to Parquet as part of this same
    # adapter call, potentially across multiple row-group/batch files whose schemas
    # are inferred independently. A nested dict column that is sometimes `{}` and
    # sometimes non-empty gets inferred as incompatible Arrow types across those
    # files (an all-empty batch infers as `null`, a populated one as a `struct`),
    # and reunifying them fails. Serialize to a JSON string instead -- always the
    # same Arrow type regardless of content -- and let
    # ``restore_empty_skipped_span_label_counts`` decode it back to a dict once the
    # dataframe is back in our hands (see the analogous drop-before-run_workflow
    # pattern in replace_runner.py / entity_coverage_judge.py, which isn't available
    # here since this column is produced *during* the DataDesigner run).
    metrics["skipped_span_label_counts"] = json.dumps(metrics["skipped_span_label_counts"], sort_keys=True)
    row[COL_REPLACEMENT_APPLICATION] = metrics
    admitted_pairs = {(entity.value, entity.label) for entity in target_entities.entities}
    row[COL_REWRITE_REPLACEMENT_READY] = (
        replace_pairs <= admitted_pairs and application.applied_span_count == application.targeted_span_count
    )
    if not row[COL_REWRITE_REPLACEMENT_READY]:
        # Keep the original tags for a diagnostic-safe unavailable result; extraction
        # below prevents this row from being accepted as a successful rewrite.
        row[COL_REWRITE_TAGGED_TEXT] = row.get(COL_TAGGED_TEXT, "")
        row[COL_REWRITE_BASELINE_TEXT] = None
        return row
    row[COL_REWRITE_BASELINE_TEXT] = baseline
    row[COL_REWRITE_TAGGED_TEXT] = build_tagged_text(
        baseline,
        _shift_entities(entities, replace_pairs=replace_pairs, replacements=replacements),
        notation=str(row.get(COL_TAG_NOTATION, "bracket")),
    )
    return row


def restore_empty_skipped_span_label_counts(dataframe: pd.DataFrame) -> None:
    """Undo the Parquet-safe JSON-string encoding written by ``_prepare_rewrite_tagged_text``.

    Mutates ``dataframe`` in place so ``skipped_span_label_counts`` is always a dict
    (empty or not) in the trace returned to callers, matching ``ReplacementApplication``'s
    documented contract.
    """
    if COL_REPLACEMENT_APPLICATION not in dataframe.columns:
        return

    def _restore(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        counts = value.get("skipped_span_label_counts")
        if not isinstance(counts, str):
            return value
        return {**value, "skipped_span_label_counts": json.loads(counts)}

    dataframe[COL_REPLACEMENT_APPLICATION] = dataframe[COL_REPLACEMENT_APPLICATION].map(_restore)


def _replace_pairs(disposition_block: object) -> set[tuple[str, str]]:
    if not isinstance(disposition_block, list):
        return set()
    return {
        (str(item.get("entity_value", "")), str(item.get("entity_label", "")))
        for item in disposition_block
        if isinstance(item, dict) and item.get("protection_method_suggestion") == "replace"
    }


def _shift_entities(
    entities: EntitiesSchema,
    *,
    replace_pairs: set[tuple[str, str]],
    replacements: list[ReplacementEntry],
) -> list[EntitySpan]:
    replacement_by_pair = _unique_replacements(replacements)
    shifted_entities = []
    delta = 0
    for entity in sorted(entities.entities, key=lambda item: item.start_position):
        synthetic = replacement_by_pair.get((entity.value, entity.label))
        value = synthetic if (entity.value, entity.label) in replace_pairs and synthetic is not None else entity.value
        start = entity.start_position + delta
        shifted_entities.append(_shifted_entity(entity, value=value, start=start))
        delta += len(value) - (entity.end_position - entity.start_position)
    return shifted_entities


def _shifted_entity(entity: EntitySchema, *, value: str, start: int) -> EntitySpan:
    return EntitySpan(
        entity_id=entity.id,
        value=value,
        label=entity.label,
        start_position=start,
        end_position=start + len(value),
        score=entity.score,
        source=entity.source,
    )


def _unique_replacements(replacements: list[ReplacementEntry]) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], set[str]] = {}
    for replacement in replacements:
        grouped.setdefault((replacement.original, replacement.label), set()).add(replacement.synthetic)
    return {key: next(iter(values)) for key, values in grouped.items() if len(values) == 1}


@custom_column_generator(required_columns=[COL_FULL_REWRITE, COL_REWRITE_REPLACEMENT_READY])
def _extract_rewritten_text(row: dict[str, Any]) -> dict[str, Any]:
    """Extract rewritten_text from the LLM structured output.

    Sets ``COL_REWRITTEN_TEXT`` to ``None`` on failure or blank output so
    downstream steps (repair, judge, human-review flagging) can distinguish
    a failed rewrite from a valid one.
    """
    if not row.get(COL_REWRITE_REPLACEMENT_READY, True):
        logger.warning("Required rewrite replacement was unavailable; marking rewritten text unavailable.")
        row[COL_REWRITTEN_TEXT] = None
        return row
    try:
        full_rewrite = row[COL_FULL_REWRITE]
        if hasattr(full_rewrite, "model_dump"):
            full_rewrite = full_rewrite.model_dump(mode="python")
        text = str(full_rewrite["rewritten_text"])
        if not text.strip():
            logger.warning("LLM returned blank rewritten_text; marking as unavailable.")
            row[COL_REWRITTEN_TEXT] = None
        else:
            row[COL_REWRITTEN_TEXT] = text
    except Exception:
        logger.warning("Failed to extract rewritten_text from COL_FULL_REWRITE; marking as unavailable.")
        row[COL_REWRITTEN_TEXT] = None
    return row


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RewriteGenerationWorkflow:
    """Column factory for the rewrite generation step.

    Returns column configs for disposition-block formatting,
    replacement-map filtering, LLM rewrite, and text extraction.
    The orchestrator (``RewriteWorkflow``) collects these alongside
    domain/disposition/QA columns for a single adapter call.
    """

    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        data_summary: str | None = None,
    ) -> list[ColumnConfigT]:
        rewriter_alias = resolve_model_alias("rewriter", selected_models)
        return [
            CustomColumnConfig(
                name=COL_REWRITE_DISPOSITION_BLOCK,
                generator_function=_format_rewrite_disposition_block,
            ),
            CustomColumnConfig(
                name=COL_REPLACEMENT_MAP_FOR_PROMPT,
                generator_function=_filter_replacement_map_for_prompt,
            ),
            CustomColumnConfig(
                name=COL_REWRITE_TAGGED_TEXT,
                generator_function=_prepare_rewrite_tagged_text,
            ),
            LLMStructuredColumnConfig(
                name=COL_FULL_REWRITE,
                prompt=_get_rewrite_prompt(privacy_goal, data_summary),
                model_alias=rewriter_alias,
                output_format=RewriteOutputSchema,
            ),
            CustomColumnConfig(
                name=COL_REWRITTEN_TEXT,
                generator_function=_extract_rewritten_text,
            ),
        ]
