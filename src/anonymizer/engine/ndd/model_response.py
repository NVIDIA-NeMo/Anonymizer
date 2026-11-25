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


class ResponseCheckError(ValueError):
    """A response check's rejection that can say why in a log: ``cause`` holds a fixed code and counts only."""

    def __init__(self, message: str, *, cause: str) -> None:
        super().__init__(message)
        self.cause = cause


class StopWiringError(RuntimeError):
    """An internal fault: a stopped step was asked to refuse a request that carries no row key."""


class Refusals:
    """The rows a run's stop refused: each row key with the stopped step that refused it first. ``RunStop`` writes it
    under its lock while the run executes; the adapter reads it after the run."""

    def __init__(self) -> None:
        self._by_key: dict[str, str] = {}

    def add(self, key: str, step: str) -> None:
        """Record that ``step`` refused the row keyed ``key``, unless a step already did."""
        self._by_key.setdefault(key, step)

    def __contains__(self, key: object) -> bool:
        return key in self._by_key

    def __getitem__(self, key: str) -> str:
        return self._by_key[key]

    def as_dict(self) -> dict[str, str]:
        return dict(self._by_key)


@dataclass
class RunStop:
    """One workflow run's early stop, per step.

    A step stops when, over its last ``window`` responses, the share that gave no usable response reaches ``rate``
    (the run config's ``shutdown_error_window`` and ``shutdown_error_rate``). From then on its requests are refused
    without a request: the refused row is kept with the step's unavailable cell, every later request for that row is
    refused too, and the row's key (its ``row_key`` column) is recorded with that first step in ``refusals``, which
    ``run_stop`` yields, so the adapter lists the row once as ``stopped_early (<step>)``. Rows already past the step
    keep running and are returned. A refusal is not a task failure, so DataDesigner's own early shutdown never sees
    it. Nothing stops when ``enabled`` is off (``disable_early_shutdown``).
    """

    window: int = 10
    rate: float = 0.5
    enabled: bool = True
    row_key: str | None = None
    stopped: list[str] = field(default_factory=list)
    refusals: Refusals = field(default_factory=Refusals, repr=False)
    _recent: dict[str, deque[bool]] = field(default_factory=dict, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @classmethod
    def from_run_config(cls, run_config: Any, *, row_key: str | None = None) -> RunStop:
        if not isinstance(run_config, RunConfig):  # e.g. a stand-in designer without a run config
            return cls(row_key=row_key)
        return cls(
            window=run_config.shutdown_error_window,
            rate=run_config.shutdown_error_rate,
            enabled=not run_config.disable_early_shutdown,
            row_key=row_key,
        )

    def refuses(self, step: str, key: str | None) -> bool:
        """Whether to refuse ``step``'s request for the row keyed ``key``: ``step`` has stopped, or a stopped step
        already refused the row. Records the first refusing step per row; a refusal with no key is a wiring fault
        and raises ``StopWiringError``."""
        with self._lock:
            if not self.stopped:
                return False
            if key is None:
                if step in self.stopped:
                    raise StopWiringError(f"{STOPPED_EARLY}: {step} refused a request with no row key")
                return False
            if step not in self.stopped and key not in self.refusals:
                return False
            self.refusals.add(key, step)
            return True

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
def run_stop(stop: RunStop) -> Generator[Refusals]:
    """Make ``stop`` the stop of the requests made in this context (DataDesigner's tasks inherit it), and yield the
    ``Refusals`` it fills: read it once the run is over."""
    token = _RUN_STOP.set(stop)
    try:
        yield stop.refusals
    finally:
        _RUN_STOP.reset(token)


def row_key(row: Mapping[str, Any]) -> str | None:
    """``row``'s key for the current run's stop, which a stopped step needs to refuse the row; ``None`` outside a
    run or when the row carries no key."""
    stop = _RUN_STOP.get()
    return None if stop is None or stop.row_key is None else row.get(stop.row_key)


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
    fail-closed default (only omitted items: a field the schema fills for an explicit null does not count, except a
    privacy item's null ``answer``, which its parser counts as omitted); or
    ``unavailable`` when no response parsed within the budget.

    ``value`` is ``None`` only when unavailable. ``cause`` is a fixed code, never the model's text. ``requests`` is
    the number of responses the model returned.
    """

    state: ResponseState
    value: T | None
    cause: str = ""
    requests: int = 0

    @property
    def refused(self) -> bool:
        """Whether the run's early stop refused the request: no request was sent, and the adapter lists the row as
        stopped early."""
        return self.cause == STOPPED_EARLY

    @classmethod
    def refusal(cls) -> ModelResponse[T]:
        return cls("unavailable", None, STOPPED_EARLY, 0)


def log_unavailable(
    log: logging.Logger, column: str, response: ModelResponse[Any], *, cause: str | None = None, kept: str = "the row"
) -> None:
    """Log that ``column`` has no response for a row: a WARNING naming the cause (``cause`` overrides the response's),
    or, for a request the early stop refused, which the adapter lists instead, only a DEBUG line."""
    if response.refused:
        log.debug("%s not asked: the run stopped early.", column)
        return
    log.warning(
        "%s unavailable after %d responses (%s); keeping %s.", column, response.requests, cause or response.cause, kept
    )


async def ask(
    model: Any,
    prompt: str,
    parser: Callable[[str], tuple[T, bool]],
    budget: RetryBudget,
    *,
    step: str,
    key: str | None = None,
) -> ModelResponse[T]:
    """Ask ``model`` through its facade's ``agenerate``; ``parser`` returns the value and whether it filled defaults.

    A response that still does not parse once the budget is spent is ``unavailable``. Any other error, such as a
    provider failure, propagates. ``step`` names the step for the run's early stop, and ``key`` (``row_key(row)``)
    identifies the row it refuses.
    """
    response, _ = await ask_with_trace(
        model,
        parser,
        step=step,
        key=key,
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
    key: str | None = None,
    **request: Any,
) -> tuple[ModelResponse[T], list[Any]]:
    """``ask`` with the facade's request arguments passed as given, returning the conversation trace too.

    The trace is empty when the response is unavailable. When the run's stop refuses the row keyed ``key`` (``step`` has
    stopped, or a stopped step refused the row before), returns ``ModelResponse.refusal()`` without a request, and it
    does not count; otherwise the response counts toward the step's stop.
    """
    stop, step = _RUN_STOP.get(), _STEP_NAMES.get().get(step, step)
    if stop is not None and stop.refuses(step, key):
        return ModelResponse.refusal(), []
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
