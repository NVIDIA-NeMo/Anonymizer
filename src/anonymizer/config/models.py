# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelSelection(BaseModel):
    """Model alias selection for each workflow phase."""

    detection: str = Field(default="text")
    replace: str = Field(default="text")
    rewrite: str = Field(default="reasoning")
    evaluation: str = Field(default="reasoning")
