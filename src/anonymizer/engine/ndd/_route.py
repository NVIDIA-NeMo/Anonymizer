# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native vs opaque route resolution for detection: the only place that compares runner methods by identity.

A builtin detection runner on a builtin adapter is lowered natively into the graph. A caller's runner, or a
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
