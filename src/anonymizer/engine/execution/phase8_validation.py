# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure keyed Phase 8 result validation and group metrics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class _Phase8Metric:
    utility_score: float
    leakage_mass: float
    weighted_leakage_rate: float
    any_high_leaked: bool
    needs_repair: bool


def _has_exact_keys(expected: tuple[object, ...], observed: tuple[object, ...]) -> bool:
    return len(observed) == len(expected) and len(set(observed)) == len(observed) and set(observed) == set(expected)


def _validate_complete_revisions(expected: tuple[object, ...], revisions: object) -> bool:
    return type(revisions) is dict and _has_exact_keys(expected, tuple(revisions)) and all(type(value) is str for value in revisions.values())


def _evaluate_metrics(
    privacy: tuple[tuple[str, float, bool], ...], utility: tuple[tuple[int, float], ...], *, repair_any_high: bool, repair_threshold: float, utility_floor: float
) -> _Phase8Metric | None:
    if not all(isfinite(score) and 0 <= score <= 1 for _, score in utility) or not all(isfinite(confidence) and 0 <= confidence <= 1 for _, confidence, _ in privacy):
        return None
    weights = {"high": 1.0, "medium": 0.6, "low": 0.3}
    if any(sensitivity not in weights for sensitivity, _, _ in privacy):
        return None
    leakage = sum(weights[sensitivity] * confidence for sensitivity, confidence, yes in privacy if yes)
    denominator = sum(weights[sensitivity] for sensitivity, _, _ in privacy)
    utility_score = sum(weight * score for weight, score in utility) / sum(weight for weight, _ in utility) if utility else 0.0
    high = any(sensitivity == "high" and yes for sensitivity, _, yes in privacy)
    return _Phase8Metric(utility_score, leakage, leakage / denominator if denominator else 0.0, high, (high if repair_any_high else leakage > repair_threshold) or utility_score < utility_floor)
