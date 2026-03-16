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
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.strategies import apply_local_replace_strategy, apply_replacement_map

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
        logger.debug("replacement strategy: %s on %d records", type(replace_method).__name__, len(dataframe))

        if isinstance(replace_method, (Annotate, Redact, Hash)):
            local_df = apply_local_replace_strategy(dataframe, strategy=replace_method)
            return ReplacementResult(dataframe=local_df, failed_records=[])

        if isinstance(replace_method, Substitute):
            if self._llm_workflow is None:
                raise ValueError("Substitute requires an llm_workflow, but none was provided.")
            map_result = self._llm_workflow.generate_map_only(
                dataframe,
                model_configs=model_configs,
                selected_models=selected_models,
                instructions=replace_method.instructions,
                preview_num_records=preview_num_records,
            )
            replaced_df = apply_replacement_map(map_result.dataframe)
            return ReplacementResult(dataframe=replaced_df, failed_records=map_result.failed_records)

        raise ValueError(f"Unsupported replace method: {type(replace_method).__name__}")
