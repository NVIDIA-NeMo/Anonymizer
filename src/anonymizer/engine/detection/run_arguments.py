# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Keyword arguments a caller-supplied detection workflow's ``run`` can accept."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

# Keyword arguments EntityDetectionWorkflow.run() gained after v0.4.0. A caller-supplied
# detection_workflow written against v0.4.0's exact run() signature (none of these, no
# **kwargs) raises TypeError if they are passed unconditionally.
POST_040_RUN_KEYWORDS = ("entity_label_examples", "source_frame")


def accepts_keyword(method: Callable[..., object], name: str) -> bool:
    """True if ``method``'s signature accepts keyword ``name`` (directly or via ``**kwargs``).

    Inspection failures (e.g. a non-introspectable callable) fail open and report the
    keyword as accepted, so such a callable keeps receiving every argument.
    """
    try:
        parameters = inspect.signature(method).parameters
    except (TypeError, ValueError):
        return True
    return name in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def detection_run_arguments(run: Callable[..., object], arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Withhold only the post-v0.4.0 keywords that ``run`` cannot accept; pass everything else.

    The built-in ``EntityDetectionWorkflow.run`` accepts all of them, so its arguments are
    unchanged. Any other argument is passed through as-is, so unrelated signature mismatches
    still fail exactly as they would on a direct call.
    """
    return {
        name: value
        for name, value in arguments.items()
        if name not in POST_040_RUN_KEYWORDS or accepts_keyword(run, name)
    }
