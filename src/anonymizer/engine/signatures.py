# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which keywords a caller's override accepts, for the checks that keep older overrides working."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Callable

if sys.version_info >= (3, 14):
    import annotationlib


def accepts_keyword(method: Callable[..., object], name: str) -> bool:
    """Check keyword support; preserve forwarding when a callable cannot be inspected."""
    try:
        if sys.version_info >= (3, 14):
            # Leave unresolvable lazy annotations (e.g. TYPE_CHECKING-only types) as forward references.
            signature = inspect.signature(method, annotation_format=annotationlib.Format.FORWARDREF)
        else:
            signature = inspect.signature(method)
        parameters = signature.parameters
    except (TypeError, ValueError):
        return True
    parameter = parameters.get(name)
    return any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()) or (
        parameter is not None
        and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )
