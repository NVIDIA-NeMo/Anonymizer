# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pandas as pd
import pytest
from data_designer.engine.models.errors import ModelGenerationValidationFailureError, ModelTimeoutError

from anonymizer.engine.ndd.model_reply import ModelReply, RetryBudget, ask, no_reply


class _Facade:
    """Feeds ``replies`` to the parser in order; raises the facade's parse failure when they run out."""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.kwargs: dict[str, Any] = {}

    async def agenerate(self, *, prompt: str, parser: Any, **kwargs: Any) -> tuple[Any, list]:
        self.kwargs = kwargs
        for reply in self.replies:
            try:
                return parser(reply), []
            except ValueError:
                continue
        raise ModelGenerationValidationFailureError("could not be parsed")


def _parser(reply: str) -> tuple[str, bool]:
    if reply == "bad":
        raise ValueError(reply)
    return reply, reply == "partial"


@pytest.mark.parametrize(
    ("replies", "expected"),
    [
        (("ok",), ModelReply("parsed", "ok", "", 1)),
        (("bad", "partial"), ModelReply("defaulted", "partial", "", 2)),
        (("bad", "bad"), ModelReply("unavailable", None, "unparseable_reply", 2)),
    ],
    ids=["parsed", "defaulted", "unavailable"],
)
def test_ask_states(replies: tuple[str, ...], expected: ModelReply[str]) -> None:
    facade = _Facade(*replies)

    reply = asyncio.run(ask(facade, "prompt", _parser, RetryBudget(corrections=3, restarts=1)))

    assert reply == expected
    assert facade.kwargs == {"max_correction_steps": 3, "max_conversation_restarts": 1}


def test_ask_lets_other_model_errors_through() -> None:
    class _Down:
        async def agenerate(self, **_: Any) -> Any:
            raise ModelTimeoutError("Timeout")

    with pytest.raises(ModelTimeoutError):
        asyncio.run(ask(_Down(), "prompt", _parser, RetryBudget(corrections=0, restarts=0)))


def test_budget_bounds_requests() -> None:
    assert RetryBudget(corrections=3, restarts=1).requests == 8


@pytest.mark.parametrize(("cell", "expected"), [(None, True), (pd.NA, True), (float("nan"), True), ({}, False)])
def test_no_reply(cell: object, expected: bool) -> None:
    assert no_reply(cell) is expected
