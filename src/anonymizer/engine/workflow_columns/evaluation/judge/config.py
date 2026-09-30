# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Literal

from data_designer.config.column_configs import LLMTextColumnConfig


class JudgeKind(str, Enum):
    DETECTION = "detection"
    TYPE_FIDELITY = "type_fidelity"
    ATTRIBUTE_FIDELITY = "attribute_fidelity"
    RELATIONAL_CONSISTENCY = "relational_consistency"


class JudgeColumnConfig(LLMTextColumnConfig):
    column_type: Literal["anonymizer-evaluation-judge"] = "anonymizer-evaluation-judge"
    judge_kind: JudgeKind

    @staticmethod
    def get_column_emoji() -> str:
        return "A"
