# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model's response to one structured question: parsed, parsed with defaults, or unavailable."""

from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

import pandas as pd
from data_designer.config.run_config import RunConfig
from data_designer.engine.models.errors import ModelGenerationValidationFailureError

T = TypeVar("T")

ResponseState = Literal["parsed", "defaulted", "unavailable"]

UNPARSEABLE_RESPONSE = "unparseable_response"
STOPPED_EARLY = "stopped_early"

logger = logging.getLogger("anonymizer.ndd.response")


class StoppedEarly(RuntimeError):
    """Raised in place of a request to a step its run has stopped; the adapter lists the row as ``stopped_early``."""


class ResponseCheckError(ValueError):
    """A response check's rejection that can say why in a log: ``cause`` holds a fixed code and counts only."""

    def __init__(self, message: str, *, cause: str) -> None:
        super().__init__(message)
        self.cause = cause


@dataclass
class RunStop:
    """One workflow run's early stop, per step.

    A step stops when, over its last ``window`` responses, the share that gave no usable response reaches ``rate``
    (the run config's ``shutdown_error_window`` and ``shutdown_error_rate``); from then on its requests raise
    ``StoppedEarly``. Other steps keep running. Nothing stops when ``enabled`` is off (``disable_early_shutdown``).
    """

    window: int = 10
    rate: float = 0.5
    enabled: bool = True
    stopped: list[str] = field(default_factory=list)
    _recent: dict[str, deque[bool]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_run_config(cls, run_config: Any) -> RunStop:
        if not isinstance(run_config, RunConfig):  # e.g. a stand-in designer without a run config
            return cls()
        return cls(
            window=run_config.shutdown_error_window,
            rate=run_config.shutdown_error_rate,
            enabled=not run_config.disable_early_shutdown,
        )

    def check(self, step: str) -> None:
        """Raise ``StoppedEarly`` when ``step`` has stopped."""
        if step in self.stopped:
            raise StoppedEarly(f"{STOPPED_EARLY}: {step}")

    def record(self, step: str, unusable: bool) -> None:
        """Count one response of ``step``; stop the step, with one WARNING, when its window reaches the rate."""
        with self._lock:
            if not self.enabled or step in self.stopped:
                return
            recent = self._recent.setdefault(step, deque(maxlen=self.window))
            recent.append(unusable)
            misses = sum(recent)
            if len(recent) < self.window or misses / self.window < self.rate:
                return
            self.stopped.append(step)
        logger.warning(
            "Stopped model requests for step %s: %d of its last %d rows had no usable response, a rate of %.2f over "
            "shutdown_error_window=%d, at or above shutdown_error_rate=%.2f. Rows that finished are returned; rows "
            "not run are listed in failed_records as %s (%s). Check that step's model, or set "
            "disable_early_shutdown=True to run every row.",
            step,
            misses,
            self.window,
            misses / self.window,
            self.window,
            self.rate,
            STOPPED_EARLY,
            step,
        )


_RUN_STOP: ContextVar[RunStop | None] = ContextVar("anonymizer_run_stop", default=None)
_STEP_NAMES: ContextVar[Mapping[str, str]] = ContextVar("anonymizer_step_names", default={})


@contextmanager
def step_names(names: Mapping[str, str]) -> Generator[None]:
    """Count the requests made in this context under ``names[step]``: a graph that runs a step once per round names
    each round's column, so each round has its own window, as when each round is its own run."""
    token = _STEP_NAMES.set(names)
    try:
        yield
    finally:
        _STEP_NAMES.reset(token)


@contextmanager
def run_stop(stop: RunStop) -> Generator[RunStop]:
    """Make ``stop`` the stop of the requests made in this context (DataDesigner's tasks inherit it)."""
    token = _RUN_STOP.set(stop)
    try:
        yield stop
    finally:
        _RUN_STOP.reset(token)


@dataclass(frozen=True)
class RetryBudget:
    """Correction turns per conversation and conversation restarts; at most ``requests`` model requests."""

    corrections: int
    restarts: int

    @property
    def requests(self) -> int:
        return (1 + self.corrections) * (1 + self.restarts)


@dataclass(frozen=True)
class ModelResponse(Generic[T]):
    """One question's answer: ``parsed``; ``defaulted`` when the parser filled items the model omitted with their
    fail-closed default (only omitted items: a field the schema fills for an explicit null does not count); or
    ``unavailable`` when no response parsed within the budget.

    ``value`` is ``None`` only when unavailable. ``cause`` is a fixed code, never the model's text. ``requests`` is
    the number of responses the model returned.
    """

    state: ResponseState
    value: T | None
    cause: str = ""
    requests: int = 0


async def ask(
    model: Any,
    prompt: str,
    parser: Callable[[str], tuple[T, bool]],
    budget: RetryBudget,
    *,
    step: str,
) -> ModelResponse[T]:
    """Ask ``model`` through its facade's ``agenerate``; ``parser`` returns the value and whether it filled defaults.

    A response that still does not parse once the budget is spent is ``unavailable``. Any other error, such as a
    provider failure, propagates. ``step`` names the step for the run's early stop.
    """
    response, _ = await ask_with_trace(
        model,
        parser,
        step=step,
        prompt=prompt,
        max_correction_steps=budget.corrections,
        max_conversation_restarts=budget.restarts,
    )
    return response


async def ask_with_trace(
    model: Any,
    parser: Callable[[str], tuple[T, bool]],
    *,
    step: str,
    **request: Any,
) -> tuple[ModelResponse[T], list[Any]]:
    """``ask`` with the facade's request arguments passed as given, returning the conversation trace too.

    The trace is empty when the response is unavailable. When the run has stopped ``step``, raises ``StoppedEarly``
    without a request; otherwise the response counts toward the step's stop.
    """
    stop, step = _RUN_STOP.get(), _STEP_NAMES.get().get(step, step)
    if stop is not None:
        stop.check(step)
    requests = 0

    def counted(response: str) -> tuple[T, bool]:
        nonlocal requests
        requests += 1
        return parser(response)

    try:
        (value, defaulted), trace = await model.agenerate(parser=counted, **request)
    except ModelGenerationValidationFailureError:
        if stop is not None:
            stop.record(step, True)
        return ModelResponse("unavailable", None, UNPARSEABLE_RESPONSE, requests), []
    if stop is not None:
        stop.record(step, False)
    return ModelResponse("defaulted" if defaulted else "parsed", value, "", requests), trace


def no_response(cell: object) -> bool:
    """Whether a stored response cell holds no response: ``None``, or the missing value a frame gives it."""
    return cell is None or cell is pd.NA or (isinstance(cell, float) and cell != cell)
