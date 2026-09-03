# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Content-free, non-authoritative observations for private context framing."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from anonymizer.measurement.session import current_collector

_SCHEMA_VERSION = "context-observation-v1"
_SEMANTIC_PROFILE = "target-context-v1"
_IMPLEMENTATION_PROFILE = "pandas-runtime-v1"
_ROUTE = "private_dataframe"
_PRIVATE_CONTEXT_COLLECTOR: ContextVar[object | None] = ContextVar(
    "anonymizer_private_context_observation_collector",
    default=None,
)
_RECORDING_CONTEXT_OBSERVATION: ContextVar[bool] = ContextVar(
    "anonymizer_recording_context_observation",
    default=False,
)


@dataclass(slots=True)
class _ContextObservationTerminal:
    outcome: str = "completed"
    reason: str = "none"
    reconciliation: str = "not_entered"
    cleanup: str = "not_entered"


@contextmanager
def _observe_context_boundary(
    boundary: str,
    *,
    target_count: int,
    context_count: int,
    byte_count: int = 0,
) -> Iterator[_ContextObservationTerminal]:
    """Record one best-effort start/terminal pair without semantic authority."""
    terminal = _ContextObservationTerminal()
    collector = current_collector() or _PRIVATE_CONTEXT_COLLECTOR.get()
    started = time.perf_counter()
    common = {
        "observation_schema": _SCHEMA_VERSION,
        "semantic_profile": _SEMANTIC_PROFILE,
        "implementation_profile": _IMPLEMENTATION_PROFILE,
        "route": _ROUTE,
        "boundary": boundary,
        "target_count_bucket": _count_bucket(target_count),
        "context_count_bucket": _count_bucket(context_count),
        "byte_count_bucket": _byte_bucket(byte_count),
    }
    _record_safely(collector, event="start", duration_sec=0.0, outcome="started", reason="none", **common)
    try:
        yield terminal
    except BaseException:
        terminal.outcome = "error"
        if terminal.reason == "none":
            terminal.reason = "boundary_error"
        raise
    finally:
        _record_safely(
            collector,
            event="terminal",
            duration_sec=max(0.0, time.perf_counter() - started),
            outcome=terminal.outcome,
            reason=terminal.reason,
            reconciliation=terminal.reconciliation,
            cleanup=terminal.cleanup,
            **common,
        )


@contextmanager
def _private_context_observation_session() -> Iterator[None]:
    """Preserve only the safe Phase 5 observer across private trace suppression."""
    token = _PRIVATE_CONTEXT_COLLECTOR.set(current_collector())
    try:
        yield
    finally:
        _PRIVATE_CONTEXT_COLLECTOR.reset(token)


def _record_safely(collector: object, **fields: object) -> None:
    if _RECORDING_CONTEXT_OBSERVATION.get():
        return
    record = getattr(collector, "record", None)
    if not callable(record):
        return
    token = _RECORDING_CONTEXT_OBSERVATION.set(True)
    try:
        record("context_workframe", **fields)
    except BaseException:
        return
    finally:
        _RECORDING_CONTEXT_OBSERVATION.reset(token)


def _count_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value <= 4:
        return "2-4"
    if value <= 16:
        return "5-16"
    if value <= 64:
        return "17-64"
    return "65+"


def _byte_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 256:
        return "1-256"
    if value <= 4096:
        return "257-4096"
    if value <= 65_536:
        return "4097-65536"
    return "65537+"
