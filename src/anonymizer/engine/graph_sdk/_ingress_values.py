# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed resource envelope for inspecting foreign callback values."""

from __future__ import annotations

from dataclasses import dataclass, fields

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class IngressLimits(PrivateValue):
    """Bound response counts; byte ceilings apply only when explicitly selected.

    None selects no ceiling for variable nested payload items or bytes. Closed
    shape, UTF-8 validation and observed item/byte accounting remain mandatory;
    association, output and depth limits remain finite.
    """

    max_associations: int = 1024
    max_outputs_per_association: int = 256
    max_collection_items: int | None = 65536
    max_assessments: int = 4096
    max_metadata_items: int | None = 65536
    max_text_bytes: int | None = None
    max_metadata_bytes: int | None = None
    max_depth: int = 16

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None and field.name in {
                "max_text_bytes",
                "max_metadata_bytes",
                "max_collection_items",
                "max_metadata_items",
            }:
                continue
            if type(value) is not int or value <= 0:
                reject(EffectCode.INVALID_VALUE)
