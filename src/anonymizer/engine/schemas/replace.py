# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field


class EntityReplacementSchema(BaseModel):
    original: str = Field(min_length=1, description="The original entity value")
    label: str = Field(min_length=1, description="The entity label/type")
    synthetic: str = Field(min_length=1, description="The synthetic replacement value")


class EntityReplacementMapSchema(BaseModel):
    replacements: list[EntityReplacementSchema] = Field(default_factory=list, description="List of entity replacements")

    @classmethod
    def from_raw(cls, raw: object, *, on_invalid: Literal["skip", "drop"]) -> list[EntityReplacementSchema]:
        """Read the entries of a replacement-map cell (model, dict or JSON string).

        ``skip`` keeps every usable entry: fields are coerced with ``str()`` and an entry
        that is not a dict or has an empty field is skipped. Schema validation does not
        decide here, so an integer synthetic (a postcode) is kept. ``drop`` validates the
        whole map and returns no entries when anything fails; it never raises.
        """
        parsed = raw
        if on_invalid == "skip":
            model_dump = getattr(raw, "model_dump", None)
            if callable(model_dump):
                parsed = model_dump(mode="python")
        elif isinstance(raw, BaseModel):
            parsed = raw.model_dump(mode="python")
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return []
            except ValueError:
                if on_invalid == "drop":
                    return []
                raise
        if not isinstance(parsed, dict):
            return []
        if on_invalid == "drop":
            try:
                return cls.model_validate(parsed).replacements
            except Exception:
                return []
        replacements = parsed.get("replacements", [])
        if not isinstance(replacements, list):
            return []
        entries: list[EntityReplacementSchema] = []
        for replacement in replacements:
            if not isinstance(replacement, dict):
                continue
            original = str(replacement.get("original", ""))
            label = str(replacement.get("label", ""))
            synthetic = str(replacement.get("synthetic", ""))
            if original and label and synthetic:
                entries.append(
                    EntityReplacementSchema.model_construct(original=original, label=label, synthetic=synthetic)
                )
        return entries
