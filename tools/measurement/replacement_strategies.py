#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental replacement strategies for benchmark-only performance probes."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum

import pandas as pd
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE
from anonymizer.engine.replace import llm_replace_workflow as lrw
from anonymizer.engine.replace.structured_substitute import apply_structured_substitution_maps


class ExperimentalReplacementStrategy(StrEnum):
    default = "default"
    local_structured_substitute = "local_structured_substitute"


@contextmanager
def experimental_replacement_strategy_context(strategy: ExperimentalReplacementStrategy) -> Iterator[None]:
    """Temporarily apply a benchmark-only replacement strategy."""
    if strategy == ExperimentalReplacementStrategy.default:
        yield
        return

    original_method = lrw.LlmReplaceWorkflow.generate_map_only
    if strategy == ExperimentalReplacementStrategy.local_structured_substitute:
        lrw.LlmReplaceWorkflow.generate_map_only = _local_structured_generate_map_only  # type: ignore[method-assign]
    else:
        raise ValueError(f"unsupported experimental replacement strategy: {strategy}")
    try:
        yield
    finally:
        lrw.LlmReplaceWorkflow.generate_map_only = original_method  # type: ignore[method-assign]


def _local_structured_generate_map_only(
    self: lrw.LlmReplaceWorkflow,
    dataframe: pd.DataFrame,
    *,
    model_configs: list[ModelConfig],
    selected_models: ReplaceModelSelection,
    instructions: str | None = None,
    entities_column: str = COL_ENTITIES_BY_VALUE,
    preview_num_records: int | None = None,
) -> lrw.LlmReplaceResult:
    _ = self, model_configs, selected_models, instructions, preview_num_records
    return lrw.LlmReplaceResult(
        dataframe=apply_structured_substitution_maps(dataframe, entities_column=entities_column),
        failed_records=[],
    )
