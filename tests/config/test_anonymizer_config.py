# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import (
    HashReplace,
    LabelReplace,
    RedactReplace,
)
from anonymizer.config.rewrite import (
    PrivacyGoal,
    RewriteParams,
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


def test_data_summary_on_input() -> None:
    inp = AnonymizerInput(
        source="data.csv",
        data_summary="Medical clinic visit notes from outpatient encounters.",
    )
    assert inp.data_summary is not None


def test_privacy_goal_without_rewrite_raises() -> None:
    with pytest.raises(ValueError, match="privacy_goal is only valid"):
        AnonymizerConfig(
            replace=LabelReplace(),
            privacy_goal=PrivacyGoal(
                protect="Protect direct patient identifiers from disclosure.",
                preserve="Preserve medical reasoning and treatment sequence.",
            ),
        )


def test_label_replace_accepts_custom_template() -> None:
    strategy = LabelReplace(format_template="[{label}]::{text}")
    assert strategy.replace(text="Alice", label="name") == "[name]::Alice"


def test_redact_replace_defaults_to_label_aware_output() -> None:
    strategy = RedactReplace()
    assert strategy.replace(text="Alice", label="first_name") == "[REDACTED_FIRST_NAME]"


def test_redact_replace_allows_constant_template() -> None:
    strategy = RedactReplace(format_template="****")
    assert strategy.replace(text="Alice", label="first_name") == "****"
