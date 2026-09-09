# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

from data_designer.config.column_configs import LLMStructuredColumnConfig


class TolerantStructuredColumnConfig(LLMStructuredColumnConfig):
    """Structured LLM column accepting fenced or bare JSON responses."""

    column_type: Literal["anonymizer-tolerant-structured"] = "anonymizer-tolerant-structured"
