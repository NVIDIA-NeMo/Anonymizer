# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The builtin Substitute runner's standalone entry into graph-owned map and application stages."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.ndd._route import builtin_map
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
    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW

    if builtin_map(helper):
        from anonymizer.engine.replace._semantic_map_profile import run_direct_map

        # One standalone invocation, also when called from inside a caller's runner.
        token = _WORKER_BORROW.set(None)
        try:
            result = run_direct_map(helper, frame, mapping, apply=True)
        finally:
            _WORKER_BORROW.reset(token)
        return ReplacementResult(result.dataframe, result.failed_records, getattr(result, "_execution", None))

    # A caller's map helper runs opaque, as in the release.
    def callback(requests: Any) -> ReplacementResult:
        result = helper.generate_map_only(frame, **mapping)
        return ReplacementResult(apply_replacement_map(result.dataframe), list(result.failed_records))

    borrow = _WORKER_BORROW.get()
    if borrow is not None:
        return callback(borrow)
    from anonymizer.engine.detection._opaque_redact import run_evaluation

    return run_evaluation(frame, callback, retained_owner=None, name="opaque-substitute-map-callback")
