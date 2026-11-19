# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, ValidationError


class FrozenConfig(BaseModel):
    """Base for the frozen config models: unknown keys rejected, and no error repeats a value it was given."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    if not TYPE_CHECKING:  # hidden from type checkers, so they still report assignment to a frozen field

        def __setattr__(self, name: str, value: Any) -> None:
            # pydantic's own frozen error ignores hide_input_in_errors and would echo the value, which may be PII.
            raise ValidationError.from_exception_data(
                type(self).__name__, [{"type": "frozen_instance", "loc": (name,), "input": value}], hide_input=True
            )
