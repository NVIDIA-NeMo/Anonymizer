# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig
from data_designer.config.models import ModelConfig
from pydantic import BaseModel, Field

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.ndd.adapter import FailedRecord, NddAdapter
from anonymizer.engine.ndd.model_loader import get_model_alias


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

    def __init__(self, adapter: NddAdapter, config_dir: Path | None = None) -> None:
        self._adapter = adapter
        self._config_dir = config_dir

    def generate_map_only(
        self,
        dataframe: pd.DataFrame,
        *,
        model_configs: list[ModelConfig] | str | Path,
        model_providers: list[Any] | str | Path | None,
        selected_models: ReplaceModelSelection,
        instructions: str | None = None,
        entities_column: str = "_entities_by_value",
        preview_num_records: int | None = None,
    ) -> LlmReplaceResult:
        replace_alias = self._resolve_alias(
            workflow_name="replace_workflow",
            role="replacer",
            fallback=selected_models.replacement_generator,
        )

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
            model_providers=model_providers,
            columns=[
                LLMStructuredColumnConfig(
                    name="_replacement_map",
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
        return LlmReplaceResult(dataframe=run_result.dataframe, failed_records=run_result.failed_records)

    def _resolve_alias(self, workflow_name: str, role: str, fallback: str) -> str:
        if self._config_dir is None:
            return fallback
        try:
            return get_model_alias(workflow_name=workflow_name, role=role, config_dir=self._config_dir)
        except (FileNotFoundError, ValueError):
            return fallback


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
    instruction_block = "\nAdditional instructions: %s\n" % instructions if instructions else ""
    return """Generate synthetic replacements for sensitive entities. ONE value per entity, used consistently.
%s
Context: {{ tagged_text }}

Entities to replace:
{%% for entity in %s %%}
- "{{ entity.value }}" ({{ entity.labels_str }})
{%% endfor %%}

Examples: {{ _entity_examples }}

Rules:
1. Related entities must stay consistent:
   - Geographic: city/state/zip must match
   - Personal: name/email must match
   - Organizational: company/domain must match
   - Temporal: age/birthdate must match
2. Maintain format and type:
   - Same structure, length patterns, character types
3. Every entity MUST have a different synthetic value.
""" % (
        instruction_block,
        entities_column,
    )


_EXAMPLE_LOOKUP: dict[str, str] = {
    "first_name": "(e.g. Michael, Ethan, Isabella)",
    "last_name": "(e.g. Smith, Williams, McKenzie)",
    "email": "(e.g. alex.torres@example.com)",
    "phone_number": "(e.g. +1-212-555-6789)",
    "city": "(e.g. Austin, Houston, Brooklyn)",
    "state": "(e.g. Texas, New York, California)",
    "country": "(e.g. USA, United Kingdom)",
    "date_of_birth": "(e.g. 1988-03-02)",
    "ssn": "(e.g. 252-96-0016)",
    "address": "(e.g. 739 Main St)",
    "organization": "(e.g. TechStart Labs)",
    "occupation": "(e.g. software engineer)",
}
