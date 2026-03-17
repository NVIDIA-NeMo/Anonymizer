# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
from data_designer.config import custom_column_generator
from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.rewrite import PrivacyGoal
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_FULL_REWRITE,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_FOR_PROMPT,
    COL_REWRITE_DISPOSITION_BLOCK,
    COL_REWRITE_GENERATION_ROW_ORDER,
    COL_REWRITTEN_TEXT,
    COL_SENSITIVITY_DISPOSITION,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    _jinja,
)
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.schemas import (
    EntitiesByValueSchema,
    EntityReplacementMapSchema,
    RewriteOutputSchema,
    SensitivityDispositionSchema,
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
- "paraphrase": Rewrite the surrounding context to obscure the entity without explicitly referencing it.

Rules:
1. ALL entity tags (as described above) must be removed. Output must be plain text.
2. Apply changes consistently — the same entity value must be treated the same way everywhere it appears.
3. Entities with needs_protection=false should be retained verbatim (tags removed only).
4. The rewritten text must flow naturally and preserve the meaning and narrative structure of the original.
5. Do not introduce new identifying details not present in the original.
</output_requirements>"""
    return (
        prompt.replace("<<PRIVACY_GOAL>>", privacy_goal.to_prompt_string())
        .replace("<<DATA_CONTEXT>>", data_context_section)
        .replace("<<TAG_NOTATION>>", COL_TAG_NOTATION)
        .replace("<<TAGGED_TEXT>>", _jinja(COL_TAGGED_TEXT))
        .replace("<<REWRITE_DISPOSITION_BLOCK>>", COL_REWRITE_DISPOSITION_BLOCK)
        .replace("<<REPLACEMENT_MAP_COL>>", COL_REPLACEMENT_MAP_FOR_PROMPT)
        .replace("<<REPLACEMENT_MAP>>", _jinja(COL_REPLACEMENT_MAP_FOR_PROMPT))
    )


# ---------------------------------------------------------------------------
# Custom column generators (pure Python, no LLM)
# ---------------------------------------------------------------------------


@custom_column_generator(required_columns=[COL_SENSITIVITY_DISPOSITION])
def _format_rewrite_disposition_block(row: dict[str, Any]) -> dict[str, Any]:
    """Pre-filter and serialize needs_protection=True entities for the rewrite prompt."""
    disposition = SensitivityDispositionSchema.model_validate(row[COL_SENSITIVITY_DISPOSITION])
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
    replace_values = {
        e["entity_value"] for e in disposition_block if e.get("protection_method_suggestion") == "replace"
    }
    raw_map = row.get(COL_REPLACEMENT_MAP, {})
    if hasattr(raw_map, "model_dump"):
        raw_map = raw_map.model_dump(mode="python")
    parsed_map = EntityReplacementMapSchema.model_validate(raw_map if isinstance(raw_map, dict) else {})
    filtered = [
        replacement.model_dump() for replacement in parsed_map.replacements if replacement.original in replace_values
    ]
    row[COL_REPLACEMENT_MAP_FOR_PROMPT] = {"replacements": filtered}
    return row


@custom_column_generator(required_columns=[COL_FULL_REWRITE])
def _extract_rewritten_text(row: dict[str, Any]) -> dict[str, Any]:
    """Extract rewritten_text from the LLM structured output, falling back to original on failure."""
    try:
        full_rewrite = row[COL_FULL_REWRITE]
        if hasattr(full_rewrite, "model_dump"):
            full_rewrite = full_rewrite.model_dump(mode="python")
        if isinstance(full_rewrite, dict):
            row[COL_REWRITTEN_TEXT] = str(full_rewrite["rewritten_text"])
        else:
            row[COL_REWRITTEN_TEXT] = str(full_rewrite.rewritten_text)
    except Exception:
        logger.warning("Failed to extract rewritten_text from COL_FULL_REWRITE; falling back to original text.")
        row[COL_REWRITTEN_TEXT] = row.get(COL_TEXT, "")
    return row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_entities(entities_by_value: object) -> bool:
    """Return True if this record has at least one detected entity."""
    if not entities_by_value:
        return False
    try:
        parsed = EntitiesByValueSchema.from_raw(entities_by_value)
        return len(parsed.entities_by_value) > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewriteGenerationResult:
    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


class RewriteGenerationWorkflow:
    """Generate rewritten text for records that have detected entities.

    Follows the column-factory pattern: ``columns()`` returns a list of
    ``ColumnConfigT`` objects intended to be passed to a single
    ``NddAdapter.run_workflow()`` call by the top-level ``RewriteWorkflow``.

    Fast path: rows with no entities in ``COL_ENTITIES_BY_VALUE`` receive
    ``COL_REWRITTEN_TEXT = COL_TEXT`` without any LLM calls.
    """

    def __init__(self, adapter: NddAdapter) -> None:
        self._adapter = adapter

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig],
        selected_models: RewriteModelSelection,
        replace_model_selection: ReplaceModelSelection,
        privacy_goal: PrivacyGoal,
        data_summary: str | None = None,
        preview_num_records: int | None = None,
    ) -> RewriteGenerationResult:
        """Run the full rewrite generation workflow.

        Records with no entities are passed through immediately; records
        with entities go through LLM replacement-map generation followed by
        the disposition-block + rewrite + text-extraction column pipeline.
        """
        working_df = dataframe.copy()
        working_df[COL_REWRITE_GENERATION_ROW_ORDER] = range(len(working_df))

        has_entities_mask = working_df[COL_ENTITIES_BY_VALUE].apply(_has_entities)
        entity_rows = working_df[has_entities_mask].copy()
        passthrough_rows = working_df[~has_entities_mask].copy()

        passthrough_rows[COL_REWRITTEN_TEXT] = passthrough_rows[COL_TEXT]

        all_failed: list[FailedRecord] = []

        if entity_rows.empty:
            combined = (
                passthrough_rows.sort_values(COL_REWRITE_GENERATION_ROW_ORDER)
                .drop(columns=[COL_REWRITE_GENERATION_ROW_ORDER])
                .reset_index(drop=True)
            )
            combined.attrs = {**dataframe.attrs}
            return RewriteGenerationResult(dataframe=combined, failed_records=all_failed)

        # Step 1 — Replacement map (LLM): reuse LlmReplaceWorkflow.
        # TODO: replace with single-workflow column architecture (see REFACTOR_PLAN.md).
        replace_workflow = LlmReplaceWorkflow(adapter=self._adapter)
        replace_result = replace_workflow.generate_map_only(
            entity_rows,
            model_configs=model_configs,
            selected_models=replace_model_selection,
        )
        entity_rows = replace_result.dataframe
        all_failed.extend(replace_result.failed_records)

        # Steps 2–4: disposition block, prompt replacement map, LLM rewrite, text extraction.
        columns = self.columns(
            selected_models=selected_models,
            privacy_goal=privacy_goal,
            data_summary=data_summary,
        )

        run_result = self._adapter.run_workflow(
            entity_rows,
            model_configs=model_configs,
            columns=columns,
            workflow_name="rewrite-generation",
            preview_num_records=preview_num_records,
        )
        rewrite_df = run_result.dataframe
        all_failed.extend(run_result.failed_records)

        combined = (
            pd.concat([rewrite_df, passthrough_rows], ignore_index=True)
            .sort_values(COL_REWRITE_GENERATION_ROW_ORDER)
            .drop(columns=[COL_REWRITE_GENERATION_ROW_ORDER])
            .reset_index(drop=True)
        )
        combined.attrs = {**run_result.dataframe.attrs, **dataframe.attrs}
        return RewriteGenerationResult(dataframe=combined, failed_records=all_failed)

    def columns(
        self,
        *,
        selected_models: RewriteModelSelection,
        privacy_goal: PrivacyGoal,
        data_summary: str | None = None,
    ) -> list[ColumnConfigT]:
        """Return column configs for Steps 2–4 of the rewrite generation workflow.

        Intended to be collected alongside other rewrite-pipeline columns and
        passed to a single ``NddAdapter.run_workflow()`` call.

        Steps 2 and 4 are pure-Python ``CustomColumnConfig``; Step 3
        (rewrite LLM call) is an ``LLMStructuredColumnConfig`` using the
        ``rewriter`` alias.
        """
        rewriter_alias = resolve_model_alias("rewriter", selected_models)
        return [
            # Step 2 — Disposition block (pure Python): filter and serialize protected entities
            CustomColumnConfig(
                name=COL_REWRITE_DISPOSITION_BLOCK,
                generator_function=_format_rewrite_disposition_block,
            ),
            # Step 3 — Filter replacement map to "replace"-method entities only
            CustomColumnConfig(
                name=COL_REPLACEMENT_MAP_FOR_PROMPT,
                generator_function=_filter_replacement_map_for_prompt,
            ),
            # Step 4 — Rewrite (LLM), output alias: "rewriter"
            LLMStructuredColumnConfig(
                name=COL_FULL_REWRITE,
                prompt=_get_rewrite_prompt(privacy_goal, data_summary),
                model_alias=rewriter_alias,
                output_format=RewriteOutputSchema,
            ),
            # Step 5 — Extract text (pure Python)
            CustomColumnConfig(
                name=COL_REWRITTEN_TEXT,
                generator_function=_extract_rewritten_text,
            ),
        ]
