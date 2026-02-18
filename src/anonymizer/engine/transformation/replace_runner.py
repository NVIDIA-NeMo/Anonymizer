# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import (
    HashReplace,
    LabelReplace,
    LLMReplace,
    RedactReplace,
    ReplaceStrategy,
)
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.transformation.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.transformation.strategies import apply_local_replace_strategy, apply_replacement_map


class ReplaceRunner:
    """Dispatch replace execution between local and LLM-backed strategies."""

    def __init__(self, llm_workflow: LlmReplaceWorkflow | None = None) -> None:
        self._llm_workflow = llm_workflow

    def run(
        self,
        dataframe: pd.DataFrame,
        *,
        replace_strategy: ReplaceStrategy,
        model_configs: list[ModelConfig] | str | Path,
        model_providers: list[Any] | str | Path | None,
        selected_models: ReplaceModelSelection,
        preview_num_records: int | None = None,
    ) -> tuple[pd.DataFrame, list[FailedRecord]]:
        if isinstance(replace_strategy, (LabelReplace, RedactReplace, HashReplace)):
            local_df = apply_local_replace_strategy(dataframe, strategy=replace_strategy)
            return local_df, []

        if isinstance(replace_strategy, LLMReplace):
            if self._llm_workflow is None:
                raise ValueError("LLMReplace strategy requires an llm_workflow, but none was provided.")
            map_result = self._llm_workflow.generate_map_only(
                dataframe,
                model_configs=model_configs,
                model_providers=model_providers,
                selected_models=selected_models,
                instructions=replace_strategy.instructions,
                preview_num_records=preview_num_records,
            )
            replaced_df = apply_replacement_map(map_result.dataframe)
            return replaced_df, map_result.failed_records

        raise ValueError(f"Unsupported replace strategy type: {type(replace_strategy).__name__}")
