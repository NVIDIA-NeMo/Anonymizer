# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DD detection/Redact consumers of the single graph executor.

The public facade selects execution-only publication with unknown physical
request visibility: DataDesigner keeps its own transport, retries and scheduling.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import pandas as pd
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethod
from anonymizer.engine.detection._graph_invocation import DetectionRedactInvocation
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.graph_sdk._execution_delegated import SubmitterPool
from anonymizer.engine.graph_sdk.executor import LocalCompleted, LocalFailure
from anonymizer.engine.harness import kernel
from anonymizer.engine.replace._graph_redact import local_configuration
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow


def run_detection_redact(
    source: pd.DataFrame,
    *,
    detection: EntityDetectionWorkflow,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    detection_options: Mapping[str, Any],
    replace_method: ReplaceMethod | None = None,
    replacement: ReplacementWorkflow | None = None,
    replacement_models: ReplaceModelSelection | None = None,
) -> ReplacementResult:
    """Run detection and replacement as one opaque graph invocation, synchronously on the caller's thread.

    Without ``replacement`` the graph applies Redact, Annotate or Hash itself; with it, the caller's
    (or builtin) replacement runner runs inside the same invocation.
    """
    selected_method = replace_method.model_copy(deep=True) if replace_method is not None else Redact()
    local_config = None
    if replacement is None:
        if not isinstance(selected_method, (Redact, Hash, Annotate)):
            raise ValueError("Substitute requires a replacement workflow")
        local_config = local_configuration(selected_method)

    async def operation(
        pool: SubmitterPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> ReplacementResult:
        if replacement is not None:
            from anonymizer.engine.detection._opaque_redact import _DetectionInvocation, execute_injected_redact

            assert replacement_models is not None and replace_method is not None
            result = await execute_injected_redact(
                _DetectionInvocation(
                    source,
                    detection,
                    partial(
                        replacement.run,
                        replace_method=replace_method,
                        model_configs=model_configs,
                        selected_models=replacement_models,
                        preview_num_records=detection_options.get("preview_num_records"),
                    ),
                    model_configs,
                    selected_models,
                    detection_options,
                    handoff=handoff,
                    replacement_receiver=replacement,
                    replacement_arguments=dict(
                        replace_method=replace_method,
                        model_configs=model_configs,
                        selected_models=replacement_models,
                        preview_num_records=detection_options.get("preview_num_records"),
                    ),
                ),
                pool,
            )
            return kernel.project_detection_redact(result)
        from anonymizer.engine.detection._opaque_redact import execute_injected_redact

        assert local_config is not None
        result = await execute_injected_redact(
            DetectionRedactInvocation(
                source,
                (),
                detection,
                model_configs,
                selected_models,
                local_config,
                detection_options,
                "redacted",
                "ok",
                handoff=handoff,
            ),
            pool,
        )
        return kernel.project_detection_redact(result)

    return kernel._run_sync_profile(operation)
