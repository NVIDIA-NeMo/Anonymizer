# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.config.anonymizer_config import (
    AnonymizerConfig,
    HashReplace,
    LabelReplace,
    LLMReplace,
    PrivacyGoal,
    RedactReplace,
)
from anonymizer.config.params import (
    EvaluationCriteria,
    RewriteParams,
    RiskTolerance,
)


def test_hash_replace_is_deterministic() -> None:
    strategy = HashReplace()
    value = strategy.replace(text="alice@example.com", label="email")
    assert value == strategy.replace(text="alice@example.com", label="email")


def test_rewrite_defaults_privacy_goal() -> None:
    config = AnonymizerConfig(
        replace=RedactReplace(),
        rewrite=RewriteParams(),
    )
    assert config.privacy_goal is not None


def test_data_summary_is_top_level() -> None:
    config = AnonymizerConfig(
        replace=LLMReplace(),
        data_summary="Medical clinic visit notes from outpatient encounters.",
    )
    assert config.data_summary is not None


def test_privacy_goal_without_rewrite_raises() -> None:
    with pytest.raises(ValueError, match="privacy_goal is only valid"):
        AnonymizerConfig(
            replace=LabelReplace(),
            privacy_goal=PrivacyGoal(
                protect="Protect direct patient identifiers from disclosure.",
                preserve="Preserve medical reasoning and treatment sequence.",
            ),
        )


def test_evaluation_criteria_manual_threshold_precedence() -> None:
    criteria = EvaluationCriteria(
        risk_tolerance=RiskTolerance.strict,
        max_leakage_mass=1.3,
        auto_adjust_by_domain=True,
    )
    assert criteria.get_effective_threshold(domain="medical") == 1.3


def test_label_replace_accepts_custom_template() -> None:
    strategy = LabelReplace(format_template="[{label}]::{text}")
    assert strategy.replace(text="Alice", label="name") == "[name]::Alice"


def test_redact_replace_defaults_to_label_aware_output() -> None:
    strategy = RedactReplace()
    assert strategy.replace(text="Alice", label="first_name") == "[REDACTED_FIRST_NAME]"


def test_redact_replace_allows_constant_template() -> None:
    strategy = RedactReplace(redact_template="****")
    assert strategy.replace(text="Alice", label="first_name") == "****"


def test_rewrite_params_do_not_include_repair_iterations() -> None:
    assert "max_repair_iterations" not in RewriteParams.model_fields
