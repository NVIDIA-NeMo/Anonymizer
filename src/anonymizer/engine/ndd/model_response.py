# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model's response to one structured question: parsed, parsed with defaults, or unavailable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import pandas as pd
from data_designer.engine.models.errors import ModelGenerationValidationFailureError

T = TypeVar("T")

ResponseState = Literal["parsed", "defaulted", "unavailable"]

UNPARSEABLE_RESPONSE = "unparseable_response"


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
) -> ModelResponse[T]:
    """Ask ``model`` through its facade's ``agenerate``; ``parser`` returns the value and whether it filled defaults.

    A response that still does not parse once the budget is spent is ``unavailable``. Any other error, such as a
    provider failure, propagates.
    """
    response, _ = await ask_with_trace(
        model,
        parser,
        prompt=prompt,
        max_correction_steps=budget.corrections,
        max_conversation_restarts=budget.restarts,
    )
    return response


async def ask_with_trace(
    model: Any,
    parser: Callable[[str], tuple[T, bool]],
    **request: Any,
) -> tuple[ModelResponse[T], list[Any]]:
    """``ask`` with the facade's request arguments passed as given, returning the conversation trace too.

    The trace is empty when the response is unavailable.
    """
    requests = 0

    def counted(response: str) -> tuple[T, bool]:
        nonlocal requests
        requests += 1
        return parser(response)

    try:
        (value, defaulted), trace = await model.agenerate(parser=counted, **request)
    except ModelGenerationValidationFailureError:
        return ModelResponse("unavailable", None, UNPARSEABLE_RESPONSE, requests), []
    return ModelResponse("defaulted" if defaulted else "parsed", value, "", requests), trace


def no_response(cell: object) -> bool:
    """Whether a stored response cell holds no response: ``None``, or the missing value a frame gives it."""
    return cell is None or cell is pd.NA or (isinstance(cell, float) and cell != cell)
