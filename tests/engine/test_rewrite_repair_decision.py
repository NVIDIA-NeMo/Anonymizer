# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which failing Rewrite evaluations are sent to repair, as released.

Repair follows replacement readiness, never the candidate text: a blank or missing rewrite whose
replacements were applied is repaired on both schedules, whether the missing cell reached the
evaluator as `None`, NaN or `pd.NA` (that depends on the schedule and the pandas version). The released
staged loop and the combined graph both repair a rewrite with unavailable replacements like any other.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest

from anonymizer.config.rewrite import EvaluationCriteria
from anonymizer.engine.rewrite._semantic_evaluation import RewriteEvaluationBinding, _repairable
from anonymizer.engine.rewrite.artifacts import (
    PrivacyAnswer,
    PrivacyAnswersValue,
    PrivacyQuestion,
    QualityAnswersValue,
    QualityComparison,
    QualityComparisonValue,
    QualityQuestion,
    RewriteCandidateValue,
    RewriteQAValue,
)

_BASELINE = "Syn travels."
_CANDIDATES = {
    "none": None,
    "nan": float("nan"),
    "na": pd.NA,
    "empty": "",
    "space": "  \n ",
    "text": "Syn journeys.",
}


def _bound(*, candidate: object, baseline: object, leaked: bool) -> Any:
    qa = RewriteQAValue(
        quality=(QualityQuestion(1, "event", "important", "Who travels?", "Syn"),),
        privacy=(PrivacyQuestion(1, "Is the name Alice?", "high", "first_name", "Alice", "direct_identifier"),),
    )
    values = {
        "qa": qa,
        "candidate": RewriteCandidateValue.capture(candidate),
        "baseline": RewriteCandidateValue.capture(baseline),
        "answers": QualityAnswersValue(answers=()),
        "privacy": PrivacyAnswersValue(
            answers=(PrivacyAnswer(1, "yes" if leaked else "no", 1.0, "scripted", ("Alice",) if leaked else ()),)
        ),
        "comparison": QualityComparisonValue(comparisons=(QualityComparison(1, 1.0, None),)),
    }
    return SimpleNamespace(inputs=[SimpleNamespace(port=f"in_{name}", value=value) for name, value in values.items()])


def _needs_repair(*, combined: bool, candidate: object, baseline: object, leaked: bool) -> bool:
    # `_metric` reads only the schedule and the evaluation criteria from its binding.
    binding = SimpleNamespace(combined=combined, criteria=EvaluationCriteria())
    bound = _bound(candidate=candidate, baseline=baseline, leaked=leaked)
    return RewriteEvaluationBinding._metric(cast(Any, binding), bound, 0).needs_repair


@pytest.mark.parametrize(
    ("combined", "ready", "expected"),
    [(False, True, True), (False, False, True), (True, True, True), (True, False, True)],
    ids=["staged-ready", "staged-withheld", "combined-ready", "combined-withheld"],
)
def test_repair_decision_follows_replacement_readiness(combined: bool, ready: bool, expected: bool) -> None:
    assert _repairable(combined=combined, ready=ready) is expected


@pytest.mark.parametrize("candidate", _CANDIDATES.values(), ids=list(_CANDIDATES))
@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_failing_ready_rewrite_is_repaired_whatever_its_text(combined: bool, candidate: object) -> None:
    assert _needs_repair(combined=combined, candidate=candidate, baseline=_BASELINE, leaked=True)


@pytest.mark.parametrize("candidate", _CANDIDATES.values(), ids=list(_CANDIDATES))
@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_passing_rewrite_is_not_repaired_whatever_its_text(combined: bool, candidate: object) -> None:
    assert not _needs_repair(combined=combined, candidate=candidate, baseline=_BASELINE, leaked=False)


@pytest.mark.parametrize("baseline", [None, float("nan"), pd.NA], ids=["none", "nan", "na"])
@pytest.mark.parametrize("candidate", [None, float("nan"), "Syn journeys."], ids=["none", "nan", "text"])
@pytest.mark.parametrize("combined", [False, True], ids=["staged", "combined"])
def test_failing_withheld_rewrite_keeps_its_released_schedule(
    combined: bool, candidate: object, baseline: object
) -> None:
    # An unavailable replacement is repaired on both schedules.
    assert _needs_repair(combined=combined, candidate=candidate, baseline=baseline, leaked=True) is True
