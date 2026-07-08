# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pydantic import BaseModel, Field


class EntityReplacementSchema(BaseModel):
    # No min_length: this is the LLM output schema for Substitute mode, so a
    # single drifted entry (e.g. a small model emitting an empty synthetic)
    # must not fail-validate the whole replacement map and drop the record.
    # Empty original/label are filtered downstream (they cannot match a
    # requested entity in _filter_replacement_map_to_input_entities); an empty
    # synthetic results in the entity being removed at apply time, which is
    # privacy-safe (no PII leak) even if utility-poor.
    original: str = Field(default="", description="The original entity value")
    label: str = Field(default="", description="The entity label/type")
    synthetic: str = Field(default="", description="The synthetic replacement value")


class EntityReplacementMapSchema(BaseModel):
    replacements: list[EntityReplacementSchema] = Field(default_factory=list, description="List of entity replacements")
