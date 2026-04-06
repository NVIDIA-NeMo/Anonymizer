# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.config.rewrite import (
    EvaluationCriteria,
    PrivacyGoal,
    RiskTolerance,
)


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
# EvaluationCriteria: risk_tolerance presets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tolerance,expected_threshold",
    [
        (RiskTolerance.minimal, 0.6),
        (RiskTolerance.low, 1.0),
        (RiskTolerance.moderate, 1.5),
        (RiskTolerance.high, 2.0),
    ],
)
def test_repair_threshold_per_tolerance(tolerance: RiskTolerance, expected_threshold: float) -> None:
    criteria = EvaluationCriteria(risk_tolerance=tolerance)
    assert criteria.repair_threshold == expected_threshold


def test_default_is_low() -> None:
    criteria = EvaluationCriteria()
    assert criteria.risk_tolerance == RiskTolerance.low
    assert criteria.repair_threshold == 1.0
    assert criteria.max_repair_iterations == 3


def test_minimal_bundles_aggressive_review_flags() -> None:
    criteria = EvaluationCriteria(risk_tolerance=RiskTolerance.minimal)
    assert criteria.flag_utility_below == 0.6
    assert criteria.flag_leakage_above == 1.0
    assert criteria.repair_any_high_leak is True


def test_high_does_not_repair_individual_high_leaks() -> None:
    criteria = EvaluationCriteria(risk_tolerance=RiskTolerance.high)
    assert criteria.repair_any_high_leak is False


def test_sensitivity_weights_have_required_keys() -> None:
    criteria = EvaluationCriteria()
    assert set(criteria.sensitivity_weights.keys()) == {"high", "medium", "low"}


def test_max_repair_iterations_rejects_negative() -> None:
    with pytest.raises(ValueError):
        EvaluationCriteria(max_repair_iterations=-1)
