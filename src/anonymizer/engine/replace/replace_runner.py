# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import (
    Annotate,
    Hash,
    Redact,
    ReplaceMethod,
    Substitute,
)
from anonymizer.engine.constants import COL_FINAL_ENTITIES
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.strategies import apply_local_replace_strategy, apply_replacement_map
from anonymizer.engine.schemas import EntitiesSchema

logger = logging.getLogger("anonymizer.replace")


@dataclass(frozen=True)
class ReplacementResult:
    """Result of a replacement workflow execution."""

    dataframe: pd.DataFrame
    failed_records: list[FailedRecord]


class ReplacementWorkflow:
    """Dispatch replace execution between local and LLM-backed strategies."""

    def __init__(self, llm_workflow: LlmReplaceWorkflow | None = None) -> None:
        self._llm_workflow = llm_workflow

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        replace_method: ReplaceMethod,
        model_configs: list[ModelConfig],
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None = None,
    ) -> ReplacementResult:
        working_df = _filter_entities_by_label(dataframe, replace_method.filter_labels)
        logger.debug("replacement strategy: %s on %d records", type(replace_method).__name__, len(working_df))

        if isinstance(replace_method, (Annotate, Redact, Hash)):
            local_df = apply_local_replace_strategy(working_df, strategy=replace_method)
            return ReplacementResult(dataframe=local_df, failed_records=[])

        if isinstance(replace_method, Substitute):
            if self._llm_workflow is None:
                raise ValueError("Substitute requires an llm_workflow, but none was provided.")
            map_result = self._llm_workflow.generate_map_only(
                working_df,
                model_configs=model_configs,
                selected_models=selected_models,
                instructions=replace_method.instructions,
                preview_num_records=preview_num_records,
            )
            replaced_df = apply_replacement_map(map_result.dataframe)
            return ReplacementResult(dataframe=replaced_df, failed_records=map_result.failed_records)

        raise ValueError(f"Unsupported replace method: {type(replace_method).__name__}")


def _filter_entities_by_label(
    dataframe: pd.DataFrame,
    filter_labels: list[str] | None,
    entities_column: str = COL_FINAL_ENTITIES,
) -> pd.DataFrame:
    """Keep only entities whose label is in filter_labels. No-op when filter_labels is None."""
    if filter_labels is None:
        return dataframe
    allowed = {label.lower() for label in filter_labels}
    filtered = dataframe.copy()
    filtered[entities_column] = filtered[entities_column].apply(
        lambda raw: _filter_entities(EntitiesSchema.from_raw(raw), allowed).model_dump()
    )
    return filtered


def _filter_entities(entities: EntitiesSchema, allowed: set[str]) -> EntitiesSchema:
    return EntitiesSchema(entities=[e for e in entities.entities if e.label.lower() in allowed])
