# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed resource envelope for inspecting foreign callback values."""

from __future__ import annotations

from dataclasses import dataclass, fields

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class IngressLimits(PrivateValue):
    """Bound one response independently of retained artifact capacity."""

    max_associations: int = 1024
    max_outputs_per_association: int = 256
    max_collection_items: int = 65536
    max_assessments: int = 4096
    max_metadata_items: int = 65536
    max_text_bytes: int = 16777216
    max_metadata_bytes: int = 1048576
    max_depth: int = 16

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value <= 0:
                reject(EffectCode.INVALID_VALUE)
