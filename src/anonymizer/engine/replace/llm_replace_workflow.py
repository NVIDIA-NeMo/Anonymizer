# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE, COL_REPLACEMENT_MAP, ENTITY_LABEL_EXAMPLES
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import resolve_model_alias


class EntityReplacement(BaseModel):
    original: str = Field(min_length=1, description="The original entity value")
    label: str = Field(min_length=1, description="The entity label/type")
    synthetic: str = Field(min_length=1, description="The synthetic replacement value")


class ReplacementMap(BaseModel):
    replacements: list[EntityReplacement] = Field(default_factory=list, description="List of entity replacements")


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
        working_df["_entity_examples"] = working_df.apply(
            lambda row: _create_entity_examples(row.get(entities_column, [])),
            axis=1,
        )
        working_df["_entities_for_replace"] = working_df.apply(
            lambda row: _normalize_entities_by_value(row.get(entities_column, [])),
            axis=1,
        )
        working_df["_entities_for_replace_json"] = working_df["_entities_for_replace"].apply(json.dumps)

        run_result = self._adapter.run_workflow(
            working_df,
            model_configs=model_configs,
            columns=[
                LLMStructuredColumnConfig(
                    name=COL_REPLACEMENT_MAP,
                    prompt=_get_replacement_mapping_prompt(
                        entities_column="_entities_for_replace",
                        instructions=instructions,
                    ),
                    model_alias=replace_alias,
                    output_format=ReplacementMap,
                )
            ],
            workflow_name="replace-map-generation",
            preview_num_records=preview_num_records,
        )
        output_df = run_result.dataframe.copy()
        # Carry pipeline metadata (e.g. original_text_column) through to downstream steps.
        # pandas .attrs is experimental and not preserved through merge/concat/groupby,
        # but is preserved through copy which is all we do here.
        # TODO: consider wrapping df + metadata in a typed container.
        output_df.attrs = {**run_result.dataframe.attrs, **dataframe.attrs}
        return LlmReplaceResult(dataframe=output_df, failed_records=run_result.failed_records)


def _normalize_entities_by_value(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    normalized: list[dict[str, Any]] = []
    for entity in raw:
        if not isinstance(entity, dict):
            continue
        enriched = {**entity}
        labels = enriched.get("labels", [])
        enriched["labels_str"] = ", ".join(str(label) for label in labels) if isinstance(labels, list) else ""
        normalized.append(enriched)
    return normalized


def _create_entity_examples(entities_by_value: Any) -> str:
    normalized = _normalize_entities_by_value(entities_by_value)
    labels: set[str] = set()
    for entity in normalized:
        labels.update(str(label) for label in entity.get("labels", []) if str(label))
    if not labels:
        return ""
    examples = {
        label: _EXAMPLE_LOOKUP.get(label, "(generate realistic synthetic replacement)") for label in sorted(labels)
    }
    return json.dumps(examples, ensure_ascii=True)


def _get_replacement_mapping_prompt(*, entities_column: str, instructions: str | None = None) -> str:
    instruction_block = f"\nAdditional instructions: {instructions}\n" if instructions else ""
    prompt = """Generate synthetic replacements for sensitive entities. ONE value per entity, used consistently.
<<INSTRUCTION_BLOCK>>
Context: {{ tagged_text }}

Entities to replace:
{%- for entity in <<ENTITIES_COLUMN>> %}
- "{{ entity.value }}" ({{ entity.labels_str }})
{%- endfor %}

Examples: {{ _entity_examples }}

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

4. Fit the surrounding text naturally:
   - Check that the synthetic value reads correctly with the words immediately before and after it in the original text.

CRITICAL: Every entity MUST have a different synthetic value. Never return original=synthetic.
"""
    return prompt.replace("<<INSTRUCTION_BLOCK>>", instruction_block).replace("<<ENTITIES_COLUMN>>", entities_column)


_EXAMPLE_LOOKUP: dict[str, str] = {
    label: f"(e.g. {', '.join(examples)})" for label, examples in ENTITY_LABEL_EXAMPLES.items()
}
