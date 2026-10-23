# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable runtime values for the existing Rewrite evaluator."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import NamedTuple, TypeAlias, cast

from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape, content_bytes
from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject
from anonymizer.engine.graph_sdk._workflow_values import ArtifactType
from anonymizer.engine.schemas.rewrite import EntityCategory, MeaningUnitImportance, SensitivityLevel


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
    iteration: int = 0


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RewriteControlValue(_RewriteValue):
    """Observed legacy continuation decision; never QA or protection evidence."""

    needs_repair: bool
    iteration: int = 0


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, eq=False)
class RewriteCandidateValue(_RewriteValue):
    """A legacy scalar observation, without text or protection qualification.

    NaN has an explicit tag so retained equality does not depend on NaN's
    non-reflexive comparison. The payload itself is retained in the artifact.
    """

    value: object
    missing: bool = False
    opaque: bool = False

    def __eq__(self, other: object) -> bool:
        if type(other) is not RewriteCandidateValue:
            return False
        return (
            self.missing == other.missing
            and self.opaque == other.opaque
            and (
                self.value is other.value
                if self.opaque
                else (
                    type(self.value) is type(other.value)
                    or type(self.value) in (int, float)
                    and type(other.value) in (int, float)
                )
                and self.value == other.value
            )
        )

    @classmethod
    def capture(cls, value: object) -> RewriteCandidateValue:
        import numpy as np
        import pandas as pd

        if isinstance(value, (np.integer, np.floating, np.bool_)):
            value = value.item()
        if value is pd.NA or type(value) is float and math.isnan(value):
            return cls(value=None, missing=True)
        return cls(value=value, opaque=type(value) not in (str, int, float, bool, type(None)))

    def project(self) -> object:
        # ``shape`` was computed in __post_init__ from this frozen value's immutable content (opaque payloads
        # count as one item with no byte claim), so it is not recomputed here.
        if rewrite_content(self) is None:
            reject(EffectCode.INVALID_TYPE)
        return float("nan") if self.missing else self.value


RewriteArtifactValue: TypeAlias = (
    RewriteQAValue
    | QualityAnswersValue
    | PrivacyAnswersValue
    | QualityComparisonValue
    | EvaluationMetricsValue
    | RewriteCandidateValue
    | RewriteControlValue
)
REWRITE_TYPES = {
    RewriteControlValue: ArtifactType(name="anonymizer.rewrite-control-observation", revision=1),
    RewriteCandidateValue: ArtifactType(name="anonymizer.rewrite-candidate-observation", revision=1),
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
    if type(value) is RewriteControlValue:
        needs, iteration = getattr(value, "needs_repair", None), getattr(value, "iteration", None)
        if (
            type(needs) is not bool
            or type(iteration) is not int
            or iteration < 0
            or max_items is not None
            and max_items < 1
        ):
            return None
        return (needs, iteration), 1
    if type(value) is RewriteCandidateValue:
        payload, missing = getattr(value, "value", object()), getattr(value, "missing", None)
        opaque = getattr(value, "opaque", None)
        if type(opaque) is not bool:
            return None
        if opaque:
            if missing is not False or max_items is not None and max_items < 1:
                return None
            # Arbitrary caller cells are retained observations, with no claim
            # about text content, semantic validity or hidden object byte size.
            return (), 1
        if (
            type(payload) not in (str, int, float, bool, type(None))
            or type(missing) is not bool
            or missing
            and payload is not None
            or type(payload) is float
            and math.isnan(payload)
            or max_items is not None
            and max_items < 1
        ):
            return None
        return (cast(str | int | float | bool | None, payload), missing), 1
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
        iteration = getattr(value, "iteration", None)
        if type(iteration) is not int or iteration < 0:
            return None
        if max_items is not None and max_items < 1:
            return None
        return (*cast(tuple[float | bool, ...], atoms), iteration), 1
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
