# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GeneralizationCandidate(BaseModel):
    """Broader wording proposed by the generalization generator."""

    entity_id: int = Field(ge=1)
    suggested_value: str | None = Field(min_length=1)


class GeneralizationCandidates(BaseModel):
    """Minimal output from the generalization generator."""

    generalization_suggestions: list[GeneralizationCandidate]


class GeneralizationSuggestion(BaseModel):
    """Validated wording for one supplied generalization target."""

    entity_id: int = Field(ge=1)
    suggested_value: str | None = Field(min_length=1)
    status: Literal["ready", "needs_context_change", "no_effective_generalization"]
    privacy_reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status(self) -> GeneralizationSuggestion:
        if self.status == "no_effective_generalization":
            if self.suggested_value is not None:
                raise ValueError("no_effective_generalization requires suggested_value=None")
        elif self.suggested_value is None or not self.suggested_value.strip():
            raise ValueError(f"{self.status!r} requires non-empty suggested_value")
        return self


class GeneralizationSuggestions(BaseModel):
    """Structured generalization output, validated against target IDs downstream."""

    generalization_suggestions: list[GeneralizationSuggestion]
