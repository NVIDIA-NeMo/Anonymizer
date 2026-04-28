# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared parse helpers and schema field validation for rewrite workflows."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from anonymizer.engine.schemas.rewrite import (
    PrivacyAnswerItemSchema,
    PrivacyAnswersSchema,
    PrivacyQAPairsSchema,
    QACompareResultsSchema,
    QualityAnswerSchema,
    QualityAnswersSchema,
    QualityQAPairsSchema,
    SensitivityDispositionSchema,
    StrictSensitivityDispositionSchema,
)

logger = logging.getLogger("anonymizer.rewrite.parsers")


def field(model: type, name: str) -> str:
    """Return *name* after verifying it exists on *model* as a Pydantic field.

    Called at module import time so a renamed schema field raises
    ``KeyError`` immediately instead of producing a broken prompt at runtime.
    """
    if name not in model.model_fields:
        raise KeyError(f"{model.__name__} has no field '{name}'")
    return name


def normalize_payload(raw: Any) -> Any:
    """Normalize values that may have been JSON-stringified by parquet round-trip."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return raw
    return _to_python(raw)


def _to_python(raw: Any) -> Any:
    """Recursively normalize nested model/array/scalar payloads to plain Python."""
    if isinstance(raw, BaseModel):
        return _to_python(raw.model_dump(mode="python"))
    if isinstance(raw, dict):
        return {key: _to_python(value) for key, value in raw.items()}
    if isinstance(raw, (list, tuple)):
        return [_to_python(value) for value in raw]
    if hasattr(raw, "tolist") and callable(raw.tolist):
        return _to_python(raw.tolist())
    if hasattr(raw, "item") and callable(raw.item):
        try:
            return raw.item()
        except (TypeError, ValueError):
            pass
    return raw


def parse_privacy_answers(raw: Any) -> list[PrivacyAnswerItemSchema]:
    raw = normalize_payload(raw)
    if isinstance(raw, PrivacyAnswersSchema):
        return raw.answers
    if isinstance(raw, dict):
        return PrivacyAnswersSchema.model_validate(raw).answers
    raise TypeError(f"Expected PrivacyAnswersSchema or dict, got {type(raw).__name__}")


def parse_quality_qa(raw: Any) -> QualityQAPairsSchema:
    raw = normalize_payload(raw)
    if isinstance(raw, QualityQAPairsSchema):
        return raw
    if isinstance(raw, dict):
        return QualityQAPairsSchema.model_validate(raw)
    raise TypeError(f"Expected QualityQAPairsSchema or dict, got {type(raw).__name__}")


def parse_quality_answers(raw: Any) -> list[QualityAnswerSchema]:
    raw = normalize_payload(raw)
    if isinstance(raw, QualityAnswersSchema):
        return raw.answers
    if isinstance(raw, dict):
        return QualityAnswersSchema.model_validate(raw).answers
    raise TypeError(f"Expected QualityAnswersSchema or dict, got {type(raw).__name__}")


def parse_quality_compare(raw: Any) -> tuple[list[int], list[float]]:
    """Return (ids, scores) from a QACompareResultsSchema or dict."""
    raw = normalize_payload(raw)
    if isinstance(raw, QACompareResultsSchema):
        return [item.id for item in raw.per_item], [item.score for item in raw.per_item]
    if isinstance(raw, dict):
        parsed = QACompareResultsSchema.model_validate(raw)
        return [item.id for item in parsed.per_item], [item.score for item in parsed.per_item]
    raise TypeError(f"Expected QACompareResultsSchema or dict, got {type(raw).__name__}")


def parse_privacy_qa(raw: Any) -> PrivacyQAPairsSchema:
    raw = normalize_payload(raw)
    if isinstance(raw, PrivacyQAPairsSchema):
        return raw
    if isinstance(raw, dict):
        return PrivacyQAPairsSchema.model_validate(raw)
    raise TypeError(f"Expected PrivacyQAPairsSchema or dict, got {type(raw).__name__}")


def parse_sensitivity_disposition(raw: Any) -> SensitivityDispositionSchema:
    raw = normalize_payload(raw)
    if isinstance(raw, StrictSensitivityDispositionSchema):
        return raw.to_sensitivity_disposition()
    if isinstance(raw, SensitivityDispositionSchema):
        return raw
    if isinstance(raw, dict):
        try:
            return SensitivityDispositionSchema.model_validate(raw)
        except ValidationError:
            return StrictSensitivityDispositionSchema.model_validate(raw).to_sensitivity_disposition()
    raise ValueError(f"Cannot parse sensitivity disposition from {type(raw)}")
