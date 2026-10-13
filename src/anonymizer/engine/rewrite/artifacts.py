# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable runtime values for the existing Rewrite evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias, cast

from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape, content_bytes
from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject
from anonymizer.engine.schemas.rewrite import EntityCategory, MeaningUnitImportance, SensitivityLevel
from anonymizer.graph.workflow import ArtifactType


class QualityQuestion(NamedTuple):
    id: int
    aspect: str
    importance: str
    question: str
    reference_answer: str


class PrivacyQuestion(NamedTuple):
    id: int
    question: str
    sensitivity: str
    entity_label: str
    entity_value: str
    category: str


class QualityAnswer(NamedTuple):
    id: int
    answer: str


class PrivacyAnswer(NamedTuple):
    id: int
    answer: str
    confidence: float
    reason: str
    evidence: tuple[str, ...]


class QualityComparison(NamedTuple):
    id: int
    score: float
    reason: str | None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class _RewriteValue(PrivateValue):
    shape: ArtifactShape = field(init=False, compare=False)

    def __post_init__(self) -> None:
        content = rewrite_content(self)
        if content is None:
            reject(EffectCode.INVALID_TYPE)
        atoms, items = content
        object.__setattr__(self, "shape", ArtifactShape(sum(content_bytes(v) for v in atoms if type(v) is str), items))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RewriteQAValue(_RewriteValue):
    quality: tuple[QualityQuestion, ...]
    privacy: tuple[PrivacyQuestion, ...]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class QualityAnswersValue(_RewriteValue):
    answers: tuple[QualityAnswer, ...]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PrivacyAnswersValue(_RewriteValue):
    answers: tuple[PrivacyAnswer, ...]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class QualityComparisonValue(_RewriteValue):
    comparisons: tuple[QualityComparison, ...]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class EvaluationMetricsValue(_RewriteValue):
    utility: float
    leakage: float
    weighted_leakage: float
    high_leak: bool
    needs_repair: bool


RewriteArtifactValue: TypeAlias = (
    RewriteQAValue | QualityAnswersValue | PrivacyAnswersValue | QualityComparisonValue | EvaluationMetricsValue
)
REWRITE_TYPES = {
    RewriteQAValue: ArtifactType(name="anonymizer.rewrite-qa", revision=1),
    QualityAnswersValue: ArtifactType(name="anonymizer.quality-answers", revision=1),
    PrivacyAnswersValue: ArtifactType(name="anonymizer.privacy-answers", revision=1),
    QualityComparisonValue: ArtifactType(name="anonymizer.quality-comparison", revision=1),
    EvaluationMetricsValue: ArtifactType(name="anonymizer.evaluation-metrics", revision=1),
}


def rewrite_content(
    value: object, max_items: int | None = None
) -> tuple[tuple[str | int | float | bool | None, ...], int] | None:
    """Validate only the closed evaluator rows and independently expose their content."""
    groups: tuple[tuple[object, type, tuple[type | tuple[type, ...], ...]], ...]
    if type(value) is RewriteQAValue:
        groups = (
            (getattr(value, "quality", None), QualityQuestion, (int, str, str, str, str)),
            (getattr(value, "privacy", None), PrivacyQuestion, (int, str, str, str, str, str)),
        )
    elif type(value) is QualityAnswersValue:
        groups = ((getattr(value, "answers", None), QualityAnswer, (int, str)),)
    elif type(value) is PrivacyAnswersValue:
        groups = ((getattr(value, "answers", None), PrivacyAnswer, (int, str, float, str, tuple)),)
    elif type(value) is QualityComparisonValue:
        groups = ((getattr(value, "comparisons", None), QualityComparison, (int, float, (str, type(None)))),)
    elif type(value) is EvaluationMetricsValue:
        atoms = tuple(
            getattr(value, name, None)
            for name in ("utility", "leakage", "weighted_leakage", "high_leak", "needs_repair")
        )
        if any(type(v) is not float or not math.isfinite(v) for v in atoms[:3]) or any(
            type(v) is not bool for v in atoms[3:]
        ):
            return None
        if max_items is not None and max_items < 1:
            return None
        return cast(tuple[float | bool, ...], atoms), 1
    else:
        return None
    atoms: list[str | int | float | bool | None] = []
    count = 0
    for rows, row_type, fields in groups:
        if type(rows) is not tuple:
            return None
        count += len(rows)
        if max_items is not None and count > max_items:
            return None
        for row in rows:
            if (
                type(row) is not row_type
                or len(row) != len(fields)
                or any(type(v) not in (t if isinstance(t, tuple) else (t,)) for v, t in zip(row, fields, strict=True))
            ):
                return None
            if type(row) is QualityQuestion and (
                not row.aspect or row.importance not in MeaningUnitImportance._value2member_map_
            ):
                return None
            if type(row) is PrivacyQuestion and (
                row.sensitivity not in SensitivityLevel._value2member_map_
                or row.category not in EntityCategory._value2member_map_
            ):
                return None
            if type(row) is PrivacyAnswer and (
                row.answer not in {"yes", "no"} or not 0 <= row.confidence <= 1 or not 1 <= len(row.reason) <= 200
            ):
                return None
            if type(row) is QualityComparison and not 0 <= row.score <= 1:
                return None
            for atom in row:
                if type(atom) is tuple:
                    count += len(atom)
                    if max_items is not None and count > max_items:
                        return None
                    if any(type(v) is not str for v in atom):
                        return None
                    atoms.extend(atom)
                elif type(atom) is float and not math.isfinite(atom):
                    return None
                else:
                    atoms.append(cast(str | int | float | bool | None, atom))
    return tuple(atoms), count
