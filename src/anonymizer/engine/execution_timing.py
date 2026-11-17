# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small, context-local wall-clock timing snapshots for saved reports."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from typing import ParamSpec, Protocol, TypeVar


class _TimedResult(Protocol):
    execution_timings: dict[str, dict[str, float | None]]


_P = ParamSpec("_P")
_R = TypeVar("_R", bound=_TimedResult)
_active: ContextVar[dict[str, float | None] | None] = ContextVar("execution_timings", default=None)


def record_elapsed(task: str, seconds: float | None) -> None:
    """Save an elapsed interval, without retaining text, entities, or raw traces."""
    active = _active.get()
    if active is not None:
        active[task] = seconds


def capture_execution(phase: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Snapshot one invocation; always reset context, including on failure."""

    def decorate(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            timings: dict[str, float | None] = {}
            token = _active.set(timings)
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
                timings["Total elapsed"] = time.perf_counter() - started
                result.execution_timings = {**getattr(result, "execution_timings", {}), phase: timings}
                return result
            finally:
                _active.reset(token)

        return wrapped

    return decorate
