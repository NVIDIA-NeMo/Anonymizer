# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest
from data_designer.engine.models.errors import ModelGenerationValidationFailureError, ModelTimeoutError

from anonymizer.engine.ndd.model_response import (
    STOPPED_EARLY,
    ModelResponse,
    RetryBudget,
    RunStop,
    ask,
    no_response,
    run_stop,
)


class _Facade:
    """Feeds ``responses`` to the parser in order; raises the facade's parse failure when they run out."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses)
        self.kwargs: dict[str, Any] = {}

    async def agenerate(self, *, prompt: str, parser: Any, **kwargs: Any) -> tuple[Any, list]:
        self.kwargs = kwargs
        for response in self.responses:
            try:
                return parser(response), []
            except ValueError:
                continue
        raise ModelGenerationValidationFailureError("could not be parsed")


def _parser(response: str) -> tuple[str, bool]:
    if response == "bad":
        raise ValueError(response)
    return response, response == "partial"


@pytest.mark.parametrize(
    ("responses", "expected"),
    [
        (("ok",), ModelResponse("parsed", "ok", "", 1)),
        (("bad", "partial"), ModelResponse("defaulted", "partial", "", 2)),
        (("bad", "bad"), ModelResponse("unavailable", None, "unparseable_response", 2)),
    ],
    ids=["parsed", "defaulted", "unavailable"],
)
def test_ask_states(responses: tuple[str, ...], expected: ModelResponse[str]) -> None:
    facade = _Facade(*responses)

    response = asyncio.run(ask(facade, "prompt", _parser, RetryBudget(corrections=3, restarts=1), step="answer"))

    assert response == expected
    assert facade.kwargs == {"max_correction_steps": 3, "max_conversation_restarts": 1}


def test_ask_lets_other_model_errors_through() -> None:
    class _Down:
        async def agenerate(self, **_: Any) -> Any:
            raise ModelTimeoutError("Timeout")

    with pytest.raises(ModelTimeoutError):
        asyncio.run(ask(_Down(), "prompt", _parser, RetryBudget(corrections=0, restarts=0), step="answer"))


def test_budget_bounds_requests() -> None:
    assert RetryBudget(corrections=3, restarts=1).requests == 8


@pytest.mark.parametrize(("cell", "expected"), [(None, True), (pd.NA, True), (float("nan"), True), ({}, False)])
def test_no_response(cell: object, expected: bool) -> None:
    assert no_response(cell) is expected


def _stopped(*steps: str) -> RunStop:
    stop = RunStop(window=1, rate=1.0, row_key="key")
    for step in steps:
        stop.record(step, True)
    return stop


def test_a_stopped_step_refuses_a_row_and_every_later_request_for_it() -> None:
    stop = _stopped("first")

    assert stop.refuses("first", {"key": "a"})
    assert stop.refuses("second", {"key": "a"})  # a later step skips a row a stopped step refused
    assert not stop.refuses("second", {"key": "b"})
    stop.record("second", True)
    assert stop.refuses("second", {"key": "b"})
    assert stop.refused == {"a": "first", "b": "second"}  # each row names the first step that refused it


def test_a_refusal_needs_the_row_key() -> None:
    with pytest.raises(RuntimeError, match="no row key"):
        _stopped("first").refuses("first", {"other": "a"})


def test_a_refused_request_is_not_sent_and_not_counted() -> None:
    stop = _stopped("answer")
    facade = _Facade("ok")

    async def asked() -> ModelResponse[str]:
        with run_stop(stop):
            return await ask(facade, "prompt", _parser, RetryBudget(0, 0), step="answer", row={"key": "a"})

    assert asyncio.run(asked()) == ModelResponse("unavailable", None, STOPPED_EARLY, 0)
    assert facade.kwargs == {}  # no request
    assert stop.refused == {"a": "answer"}
