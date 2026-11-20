# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parse helpers and schema field validation for rewrite workflows."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from anonymizer.engine.schemas.rewrite import (
    PrivacyAnswerItemSchema,
    PrivacyAnswersSchema,
    PrivacyQAPairsSchema,
    QACompareResultsSchema,
    QualityAnswerSchema,
    QualityAnswersSchema,
    QualityQAPairsSchema,
    SensitivityDispositionSchema,
)
from anonymizer.engine.schemas.shared import normalize_payload as normalize_payload
from anonymizer.engine.schemas.shared import read_cell

logger = logging.getLogger("anonymizer.rewrite.parsers")


def field(model: type[BaseModel], name: str) -> str:
    """Return *name* after verifying it exists on *model* as a Pydantic field.

    Called at module import time so a renamed schema field raises
    ``KeyError`` immediately instead of producing a broken prompt at runtime.
    """
    if name not in model.model_fields:
        raise KeyError(f"{model.__name__} has no field '{name}'")
    return name


def parse_privacy_answers(raw: Any) -> list[PrivacyAnswerItemSchema]:
    return read_cell(PrivacyAnswersSchema, raw).answers


def parse_quality_qa(raw: Any) -> QualityQAPairsSchema:
    return read_cell(QualityQAPairsSchema, raw)


def parse_quality_answers(raw: Any) -> list[QualityAnswerSchema]:
    return read_cell(QualityAnswersSchema, raw).answers


def parse_quality_compare(raw: Any) -> tuple[list[int], list[float]]:
    """Return (ids, scores) from a QACompareResultsSchema or dict."""
    items = read_cell(QACompareResultsSchema, raw).per_item
    return [item.id for item in items], [item.score for item in items]


def parse_privacy_qa(raw: Any) -> PrivacyQAPairsSchema:
    return read_cell(PrivacyQAPairsSchema, raw)


def parse_sensitivity_disposition(raw: Any) -> SensitivityDispositionSchema:
    return read_cell(SensitivityDispositionSchema, raw, error=ValueError)
