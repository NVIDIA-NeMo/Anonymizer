# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parse helpers and schema field validation for rewrite workflows."""

from __future__ import annotations

from typing import Any

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


def field(model: type, name: str) -> str:
    """Return *name* after verifying it exists on *model* as a Pydantic field.

    Called at module import time so a renamed schema field raises
    ``KeyError`` immediately instead of producing a broken prompt at runtime.
    """
    if name not in model.model_fields:
        raise KeyError(f"{model.__name__} has no field '{name}'")
    return name


def parse_privacy_answers(raw: Any) -> list[PrivacyAnswerItemSchema]:
    if isinstance(raw, PrivacyAnswersSchema):
        return raw.answers
    if isinstance(raw, dict):
        return PrivacyAnswersSchema.model_validate(raw).answers
    raise TypeError(f"Expected PrivacyAnswersSchema or dict, got {type(raw).__name__}")


def parse_quality_qa(raw: Any) -> QualityQAPairsSchema:
    if isinstance(raw, QualityQAPairsSchema):
        return raw
    if isinstance(raw, dict):
        return QualityQAPairsSchema.model_validate(raw)
    raise TypeError(f"Expected QualityQAPairsSchema or dict, got {type(raw).__name__}")


def parse_quality_answers(raw: Any) -> list[QualityAnswerSchema]:
    if isinstance(raw, QualityAnswersSchema):
        return raw.answers
    if isinstance(raw, dict):
        return QualityAnswersSchema.model_validate(raw).answers
    raise TypeError(f"Expected QualityAnswersSchema or dict, got {type(raw).__name__}")


def parse_quality_compare(raw: Any) -> tuple[list[int], list[float]]:
    """Return (ids, scores) from a QACompareResultsSchema or dict."""
    if isinstance(raw, QACompareResultsSchema):
        return [item.id for item in raw.per_item], [item.score for item in raw.per_item]
    if isinstance(raw, dict):
        parsed = QACompareResultsSchema.model_validate(raw)
        return [item.id for item in parsed.per_item], [item.score for item in parsed.per_item]
    raise TypeError(f"Expected QACompareResultsSchema or dict, got {type(raw).__name__}")


def parse_privacy_qa(raw: Any) -> PrivacyQAPairsSchema:
    if isinstance(raw, PrivacyQAPairsSchema):
        return raw
    if isinstance(raw, dict):
        return PrivacyQAPairsSchema.model_validate(raw)
    raise TypeError(f"Expected PrivacyQAPairsSchema or dict, got {type(raw).__name__}")


def parse_sensitivity_disposition(raw: Any) -> SensitivityDispositionSchema:
    if isinstance(raw, SensitivityDispositionSchema):
        return raw
    if isinstance(raw, dict):
        return SensitivityDispositionSchema.model_validate(raw)
    raise ValueError(f"Cannot parse sensitivity disposition from {type(raw)}")
