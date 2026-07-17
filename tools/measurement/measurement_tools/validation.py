# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private validation building blocks shared by measurement-tool models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, StrictStr


class StrictFrozenModel(BaseModel):
    """Strict, immutable model base for non-redacted validation errors."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RedactedStrictFrozenModel(BaseModel):
    """Strict, immutable model base that never echoes rejected input values."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, hide_input_in_errors=True)


VisibleIdentifier = Annotated[StrictStr, Field(min_length=1, max_length=256)]
VisibleSlurmIdentifier = Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,255}$")]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
NonNegativeFloat = Annotated[FiniteFloat, Field(ge=0)]
Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]
Percentage = Annotated[FiniteFloat, Field(ge=0, le=100)]
MeasurementStrategy = Literal["Annotate", "Hash", "Redact", "Rewrite", "Substitute"]
RatBenchAttackerEndpointKind = Literal["bigiron", "nvidia_inference_fallback"]
