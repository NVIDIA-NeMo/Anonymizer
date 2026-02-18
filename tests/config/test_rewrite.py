# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.config.rewrite import (
    EvaluationCriteria,
    PrivacyGoal,
    RiskTolerance,
)

# ---------------------------------------------------------------------------
# PrivacyGoal validation
# ---------------------------------------------------------------------------


def test_privacy_goal_valid() -> None:
    goal = PrivacyGoal(
        protect="Protect direct patient identifiers from disclosure.",
        preserve="Preserve medical reasoning and treatment sequence.",
    )
    assert "Protect direct patient" in goal.protect


def test_privacy_goal_rejects_too_few_words() -> None:
    with pytest.raises(ValueError, match="at least 3 words"):
        PrivacyGoal(protect="Tooshort oneword", preserve="Preserve medical reasoning and treatment sequence.")


def test_privacy_goal_rejects_too_short_string() -> None:
    with pytest.raises(ValueError):
        PrivacyGoal(protect="ab", preserve="Preserve medical reasoning and treatment sequence.")


def test_privacy_goal_strips_whitespace() -> None:
    goal = PrivacyGoal(
        protect="  Protect patient direct identifiers from disclosure.  ",
        preserve="  Preserve reasoning and treatment sequence.  ",
    )
    assert not goal.protect.startswith(" ")
    assert not goal.preserve.endswith(" ")


def test_privacy_goal_to_prompt_string() -> None:
    goal = PrivacyGoal(
        protect="Protect direct patient identifiers from disclosure.",
        preserve="Preserve medical reasoning and treatment sequence.",
    )
    prompt = goal.to_prompt_string()
    assert "PROTECT:" in prompt
    assert "PRESERVE:" in prompt


# ---------------------------------------------------------------------------
# EvaluationCriteria — effective threshold
# ---------------------------------------------------------------------------


def test_effective_threshold_uses_manual_when_set() -> None:
    criteria = EvaluationCriteria(max_leakage_mass=1.3)
    assert criteria.get_effective_threshold(domain="medical") == 1.3


def test_effective_threshold_uses_risk_tolerance_default() -> None:
    criteria = EvaluationCriteria(risk_tolerance=RiskTolerance.strict)
    assert criteria.get_effective_threshold() == 0.6


@pytest.mark.parametrize(
    "tolerance,expected",
    [
        (RiskTolerance.strict, 0.6),
        (RiskTolerance.conservative, 1.0),
        (RiskTolerance.moderate, 1.5),
        (RiskTolerance.permissive, 2.0),
    ],
)
def test_effective_threshold_per_tolerance(tolerance: RiskTolerance, expected: float) -> None:
    criteria = EvaluationCriteria(risk_tolerance=tolerance)
    assert criteria.get_effective_threshold() == expected


def test_effective_threshold_auto_adjust_by_domain() -> None:
    criteria = EvaluationCriteria(auto_adjust_by_domain=True)
    assert criteria.get_effective_threshold(domain="medical") == 0.6  # strict
    assert criteria.get_effective_threshold(domain="social_media") == 2.0  # permissive


def test_effective_threshold_auto_adjust_unknown_domain_defaults_to_moderate() -> None:
    criteria = EvaluationCriteria(auto_adjust_by_domain=True)
    assert criteria.get_effective_threshold(domain="unknown_domain") == 1.5


def test_effective_threshold_auto_adjust_normalizes_domain_string() -> None:
    """Hyphens, spaces, and case should be normalized."""
    criteria = EvaluationCriteria(auto_adjust_by_domain=True)
    assert criteria.get_effective_threshold(domain="Human Resources") == 1.0  # conservative
    assert criteria.get_effective_threshold(domain="MEDICAL") == 0.6


def test_effective_threshold_auto_adjust_ignored_without_domain() -> None:
    criteria = EvaluationCriteria(auto_adjust_by_domain=True, risk_tolerance=RiskTolerance.permissive)
    assert criteria.get_effective_threshold(domain=None) == 2.0


def test_effective_threshold_manual_overrides_auto_adjust() -> None:
    criteria = EvaluationCriteria(max_leakage_mass=0.8, auto_adjust_by_domain=True)
    assert criteria.get_effective_threshold(domain="medical") == 0.8


# ---------------------------------------------------------------------------
# EvaluationCriteria — sensitivity_weights validation
# ---------------------------------------------------------------------------


def test_sensitivity_weights_defaults() -> None:
    criteria = EvaluationCriteria()
    assert set(criteria.sensitivity_weights.keys()) == {"high", "medium", "low"}


def test_sensitivity_weights_rejects_missing_level() -> None:
    with pytest.raises(ValueError, match="missing required levels"):
        EvaluationCriteria(sensitivity_weights={"high": 1.0, "medium": 0.6})


def test_sensitivity_weights_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        EvaluationCriteria(sensitivity_weights={"high": 1.0, "medium": -0.5, "low": 0.3})
