# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native vs opaque route resolution: the only place that compares runner methods by identity.

A builtin runner on a builtin adapter is lowered natively into the graph. A caller's runner, or a
builtin runner on an adapter whose ``run_workflow`` the caller replaced, runs opaque, as in the
release: with no graph claim on its DataDesigner work. Either route runs on the caller's thread.
"""

from __future__ import annotations

from typing import Any


def _bound_to(method: object, function: object) -> bool:
    return getattr(method, "__func__", None) is function


def native_adapter(adapter: Any) -> bool:
    """Whether DataDesigner work goes through the builtin ``NddAdapter.run_workflow``.

    Reads ``run_workflow`` eagerly: an adapter without one fails here with its own AttributeError, as in the
    release.
    """
    from anonymizer.engine.ndd.adapter import NddAdapter

    return _bound_to(adapter.run_workflow, NddAdapter.run_workflow)


def builtin_detection(detection: object) -> bool:
    """Whether a detection runner is the builtin one, lowered by the graph."""
    from anonymizer.engine.detection.detection_workflow import _DETECTION_RUN

    return _bound_to(getattr(detection, "run", None), _DETECTION_RUN)


def builtin_map(helper: object) -> bool:
    """Whether a replacement-map helper is the builtin one, lowered by the graph."""
    from anonymizer.engine.replace.llm_replace_workflow import _GENERATE_MAP_ONLY, LlmReplaceWorkflow

    return isinstance(helper, LlmReplaceWorkflow) and _bound_to(helper.generate_map_only, _GENERATE_MAP_ONLY)


def native_map(receiver: Any) -> bool:
    """Whether a direct map call runs the class's current ``generate_map_only`` on a builtin adapter."""
    from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow

    return native_adapter(receiver._adapter) and _bound_to(
        receiver.generate_map_only, LlmReplaceWorkflow.generate_map_only
    )


def builtin_replacement(receiver: object) -> bool:
    """Whether a replacement runner is the builtin one."""
    from anonymizer.engine.replace.replace_runner import _REPLACEMENT_RUN

    return _bound_to(getattr(receiver, "run", None), _REPLACEMENT_RUN)


def builtin_rewrite_runner(runner: object) -> bool:
    """Whether a Rewrite runner is the builtin one, so the facade lowers it natively."""
    from anonymizer.engine.rewrite.rewrite_workflow import _REWRITE_RUN, RewriteWorkflow

    return (
        isinstance(runner, RewriteWorkflow)
        and _bound_to(runner.run, _REWRITE_RUN)
        and callable(getattr(runner._adapter, "run_workflow", None))
    )
