# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class GeneralizationSuggestion(BaseModel):
    """Wording and contextual instructions for one supplied generalization target."""

    entity_id: int = Field(ge=1)
    suggested_value: str | None = Field(min_length=1)
    status: Literal["ready", "needs_context_change", "no_effective_generalization"]
    privacy_reason: str = Field(min_length=1)
    rewrite_instruction: str
    related_entity_ids: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status(self) -> GeneralizationSuggestion:
        if self.status == "no_effective_generalization":
            if self.suggested_value is not None:
                raise ValueError("no_effective_generalization requires suggested_value=None")
        elif self.suggested_value is None or not self.suggested_value.strip():
            raise ValueError(f"{self.status!r} requires non-empty suggested_value")
        if self.status != "ready" and not self.rewrite_instruction.strip():
            raise ValueError(f"{self.status!r} requires a rewrite_instruction explaining the unresolved protection")
        return self


class GeneralizationSuggestions(BaseModel):
    """Structured generalization output, validated against target IDs downstream."""

    generalization_suggestions: list[GeneralizationSuggestion]


class GeneralizationDefect(BaseModel):
    """Concrete evidence of a defect in one candidate."""

    entity_id: int = Field(ge=1)
    evidence: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    conflicting_entity_ids: list[int] = Field(default_factory=list)


class GeneralizationReview(BaseModel):
    """Defects are reported before the complete corrected suggestion set."""

    defects: list[GeneralizationDefect]
    generalization_suggestions: list[GeneralizationSuggestion]
