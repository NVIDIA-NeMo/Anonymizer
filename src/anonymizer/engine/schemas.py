# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import TypeVar

from pydantic import BaseModel, Field, ValidationError

T = TypeVar("T", bound=BaseModel)


def _parse_raw_wrapper(
    model_cls: type[T],
    raw: object,
    key: str,
    fallback_keys: tuple[str, ...] = (),
) -> T:
    """Parse raw DataFrame cell value (dict, list, or numpy array) into a wrapper schema."""

    def _safe_validate(candidate_list: list[object]) -> T:
        try:
            return model_cls.model_validate({key: candidate_list})
        except ValidationError:
            return model_cls()

    if isinstance(raw, dict):
        candidate = raw.get(key)
        if candidate is None:
            for fk in fallback_keys:
                candidate = raw.get(fk)
                if candidate is not None:
                    break
        if isinstance(candidate, list):
            return _safe_validate(candidate)
        if hasattr(candidate, "tolist"):
            as_list = candidate.tolist()
            if isinstance(as_list, list):
                return _safe_validate(as_list)
        return model_cls()
    if isinstance(raw, list):
        return _safe_validate(raw)
    if hasattr(raw, "tolist"):
        as_list = raw.tolist()
        if isinstance(as_list, list):
            return _safe_validate(as_list)
    return model_cls()


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------


class ValidationChoice(str, Enum):
    keep = "keep"
    reclass = "reclass"
    drop = "drop"


# ---------------------------------------------------------------------------
# Entity schemas
# ---------------------------------------------------------------------------


class EntitySchema(BaseModel):
    """Canonical entity span used across workflow columns."""

    id: str = Field(default="")
    value: str = Field(default="")
    label: str = Field(default="")
    start_position: int = Field(default=0)
    end_position: int = Field(default=0)
    score: float = Field(default=0.0)
    source: str = Field(default="detector")


class EntitiesSchema(BaseModel):
    """Canonical ``{"entities": [...]}`` wrapper for entity spans."""

    entities: list[EntitySchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> EntitiesSchema:
        return _parse_raw_wrapper(cls, raw, "entities")


# ---------------------------------------------------------------------------
# Validation candidate schemas
# ---------------------------------------------------------------------------


class ValidationCandidateSchema(BaseModel):
    id: str = Field(default="")
    value: str = Field(default="")
    label: str = Field(default="")
    context_before: str = Field(default="")
    context_after: str = Field(default="")


class ValidationCandidatesSchema(BaseModel):
    candidates: list[ValidationCandidateSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> ValidationCandidatesSchema:
        return _parse_raw_wrapper(cls, raw, "candidates", fallback_keys=("entities",))


# ---------------------------------------------------------------------------
# Raw validation decision schemas (LLM output before enrichment)
# ---------------------------------------------------------------------------


class RawValidationDecisionSchema(BaseModel):
    id: str = Field(default="")
    decision: ValidationChoice | None = None
    proposed_label: str = Field(default="")
    reason: str | None = None


class RawValidationDecisionsSchema(BaseModel):
    decisions: list[RawValidationDecisionSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> RawValidationDecisionsSchema:
        return _parse_raw_wrapper(cls, raw, "decisions")


# ---------------------------------------------------------------------------
# Validated (enriched) decision schemas
# ---------------------------------------------------------------------------


class ValidatedDecisionSchema(BaseModel):
    id: str = Field(default="")
    decision: ValidationChoice | None = None
    proposed_label: str = Field(default="")
    reason: str | None = None
    value: str = Field(default="")
    label: str = Field(default="")


class ValidatedDecisionsSchema(BaseModel):
    decisions: list[ValidatedDecisionSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> ValidatedDecisionsSchema:
        return _parse_raw_wrapper(cls, raw, "decisions")


# ---------------------------------------------------------------------------
# Validation skeleton schemas
# ---------------------------------------------------------------------------


class ValidationSkeletonDecisionSchema(BaseModel):
    id: str = Field(default="")
    value: str = Field(default="")
    label: str = Field(default="")
    decision: ValidationChoice | None = None
    proposed_label: str | None = None
    reason: str | None = None


class ValidationSkeletonSchema(BaseModel):
    decisions: list[ValidationSkeletonDecisionSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> ValidationSkeletonSchema:
        return _parse_raw_wrapper(cls, raw, "decisions")


# ---------------------------------------------------------------------------
# Entities-by-value schemas (grouped for replacement)
# ---------------------------------------------------------------------------


class EntityByValueSchema(BaseModel):
    value: str = Field(default="")
    labels: list[str] = Field(default_factory=list)


class EntitiesByValueSchema(BaseModel):
    entities_by_value: list[EntityByValueSchema] = Field(default_factory=list)

    @classmethod
    def from_raw(cls, raw: object) -> EntitiesByValueSchema:
        return _parse_raw_wrapper(cls, raw, "entities_by_value", fallback_keys=("entities",))
