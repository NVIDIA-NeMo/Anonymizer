# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Literal

from data_designer.config.column_configs import LLMStructuredColumnConfig
from pydantic import Field, field_serializer


class ResponseColumnConfig(LLMStructuredColumnConfig):
    """A structured model column whose row survives a response that never parses.

    The column renders, asks and parses exactly as ``LLMStructuredColumnConfig`` does, with the run config's
    corrections and restarts. A response that still does not parse writes ``unavailable`` instead of failing the record.
    When a column in ``ask_only_if`` holds no value, the model is not asked and the column writes ``unavailable``.
    ``unavailable_marker`` names a column that says whether ``unavailable`` was written: a response can equal the
    ``unavailable`` value, so the value alone cannot tell.
    ``response_check`` is an optional callable ``(response, row)`` that raises ``ValueError`` for a response the schema accepts
    but the row does not, such as one with the wrong number of entries; that response is corrected or re-asked like one
    that does not parse.
    """

    column_type: Literal["anonymizer-structured-response"] = "anonymizer-structured-response"  # ty: ignore[invalid-attribute-override]
    unavailable: dict[str, Any] | None = None
    ask_only_if: list[str] = Field(default_factory=list)
    unavailable_marker: str | None = None
    response_check: Any = None

    @field_serializer("response_check")
    def _serialize_response_check(self, check: Any) -> str | None:
        return None if check is None else getattr(getattr(check, "func", check), "__name__", repr(check))

    @property
    def required_columns(self) -> list[str]:
        return list(dict.fromkeys([*super().required_columns, *self.ask_only_if]))

    @property
    def side_effect_columns(self) -> list[str]:
        return [*super().side_effect_columns, *([self.unavailable_marker] if self.unavailable_marker else [])]
