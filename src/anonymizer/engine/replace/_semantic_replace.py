# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility entries into graph-owned Substitute map and application stages."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.replace._semantic_map import _BINDING
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.engine.replace.strategies import apply_replacement_map


def run_substitute(receiver: ReplacementWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]) -> ReplacementResult:
    helper = cast(Any, receiver._llm_workflow)
    if helper is None:
        reject(EffectCode.MISSING)
    mapping: dict[str, Any] = dict(
        model_configs=arguments["model_configs"],
        selected_models=arguments["selected_models"],
        instructions=arguments["replace_method"].instructions,
        entities_column=COL_ENTITIES_BY_VALUE,
        preview_num_records=arguments["preview_num_records"],
    )
    binding = _BINDING.get()
    if binding is not None:
        if (
            binding.replacement_receiver is not receiver
            or binding.receiver is not helper
            or frame is not binding.entry
            or binding.replacement_configuration != arguments["replace_method"].model_dump_json()
            or mapping["model_configs"] is not binding.arguments["model_configs"]
            or mapping["selected_models"] is not binding.arguments["selected_models"]
        ):
            reject(EffectCode.FOREIGN_OWNER)
        binding.authenticate()
        result = helper.generate_map_only(frame, **mapping)
        return ReplacementResult(
            result.dataframe if binding.claim.locked() else apply_replacement_map(result.dataframe),
            list(result.failed_records),
        )
    if isinstance(helper, LlmReplaceWorkflow):
        from anonymizer.engine.replace._semantic_map_profile import run_direct_map

        result = run_direct_map(helper, frame, mapping, apply=True)
        return ReplacementResult(result.dataframe, result.failed_records, getattr(result, "_execution", None))
    from anonymizer.engine.detection._opaque_redact import run_evaluation

    def callback(requests: Any) -> ReplacementResult:
        result = helper.generate_map_only(frame, **mapping)
        return ReplacementResult(apply_replacement_map(result.dataframe), list(result.failed_records))

    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW

    borrow = _WORKER_BORROW.get()
    if borrow is not None:
        return callback(borrow)
    return run_evaluation(frame, callback, retained_owner=None, name="opaque-substitute-map-callback")
