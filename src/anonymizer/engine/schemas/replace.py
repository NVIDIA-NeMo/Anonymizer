# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field


class EntityReplacementSchema(BaseModel):
    original: str = Field(min_length=1, description="The original entity value")
    label: str = Field(min_length=1, description="The entity label/type")
    synthetic: str = Field(min_length=1, description="The synthetic replacement value")


class EntityReplacementMapSchema(BaseModel):
    replacements: list[EntityReplacementSchema] = Field(default_factory=list, description="List of entity replacements")
