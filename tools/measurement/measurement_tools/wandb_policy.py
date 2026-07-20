# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light exposure vocabulary for W&B fields."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict

__all__ = ["DataClass", "Exposure", "FieldPolicy"]


class DataClass(StrEnum):
    operational = "operational"
    pseudonymous = "pseudonymous"
    sensitive = "sensitive"


class Exposure(StrEnum):
    local_only = "local_only"
    aggregate = "aggregate"
    table_opt_in = "table_opt_in"
    never = "never"


class FieldPolicy(BaseModel):
    data_class: DataClass
    exposure: Exposure

    model_config = ConfigDict(extra="forbid", frozen=True)
