# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A model's reply to one structured question: parsed, parsed with defaults, or unavailable."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import pandas as pd
from data_designer.engine.models.errors import ModelGenerationValidationFailureError

T = TypeVar("T")

ReplyState = Literal["parsed", "defaulted", "unavailable"]

UNPARSEABLE_REPLY = "unparseable_reply"


@dataclass(frozen=True)
class RetryBudget:
    """Correction turns per conversation and conversation restarts; at most ``requests`` model requests."""

    corrections: int
    restarts: int

    @property
    def requests(self) -> int:
        return (1 + self.corrections) * (1 + self.restarts)


@dataclass(frozen=True)
class ModelReply(Generic[T]):
    """One question's answer: ``parsed``; ``defaulted`` when the parser filled items the model omitted with their
    fail-closed default (only omitted items: a field the schema fills for an explicit null does not count); or
    ``unavailable`` when no reply parsed within the budget.

    ``value`` is ``None`` only when unavailable. ``cause`` is a fixed code, never the model's text. ``requests`` is
    the number of replies the model returned.
    """

    state: ReplyState
    value: T | None
    cause: str = ""
    requests: int = 0


async def ask(
    model: Any,
    prompt: str,
    parser: Callable[[str], tuple[T, bool]],
    budget: RetryBudget,
) -> ModelReply[T]:
    """Ask ``model`` through its facade's ``agenerate``; ``parser`` returns the value and whether it filled defaults.

    A reply that still does not parse once the budget is spent is ``unavailable``. Any other error, such as a
    provider failure, propagates.
    """
    requests = 0

    def counted(response: str) -> tuple[T, bool]:
        nonlocal requests
        requests += 1
        return parser(response)

    try:
        (value, defaulted), _ = await model.agenerate(
            prompt=prompt,
            parser=counted,
            max_correction_steps=budget.corrections,
            max_conversation_restarts=budget.restarts,
        )
    except ModelGenerationValidationFailureError:
        return ModelReply("unavailable", None, UNPARSEABLE_REPLY, requests)
    return ModelReply("defaulted" if defaulted else "parsed", value, "", requests)


def no_reply(cell: object) -> bool:
    """Whether a stored reply cell holds no reply: ``None``, or the missing value a frame gives it."""
    return cell is None or cell is pd.NA or (isinstance(cell, float) and cell != cell)
