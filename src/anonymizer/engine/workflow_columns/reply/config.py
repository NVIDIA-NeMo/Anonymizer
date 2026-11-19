# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from data_designer.config.column_configs import LLMStructuredColumnConfig
from pydantic import Field


class ReplyColumnConfig(LLMStructuredColumnConfig):
    """A structured model column whose row survives a reply that never parses.

    The column renders, asks and parses exactly as ``LLMStructuredColumnConfig`` does, with the run config's
    corrections and restarts. A reply that still does not parse writes ``unavailable`` instead of failing the record.
    When a column in ``ask_only_if`` holds no value, the model is not asked and the column writes ``unavailable``.
    """

    column_type: Literal["anonymizer-structured-reply"] = "anonymizer-structured-reply"  # ty: ignore[invalid-attribute-override]
    unavailable: dict[str, Any] | None = None
    ask_only_if: list[str] = Field(default_factory=list)

    @property
    def required_columns(self) -> list[str]:
        return list(dict.fromkeys([*super().required_columns, *self.ask_only_if]))
