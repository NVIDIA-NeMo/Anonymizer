# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anonymizer import Detect, RegexLimits
from anonymizer.interface.anonymizer import _detection_regex_kwargs
from anonymizer.interface.errors import InvalidConfigError


def test_regex_limits_defaults_and_config_round_trip() -> None:
    assert Detect().regex_limits == RegexLimits()
    config = Detect(
        regex_limits={
            "max_matches_per_rule": 1234,
            "min_timeout_seconds": 0.2,
            "timeout_seconds_per_million_characters": 2,
        }
    )
    assert Detect.model_validate_json(config.model_dump_json()).regex_limits == config.regex_limits


def test_injected_workflow_cannot_silently_ignore_nondefault_limits() -> None:
    def legacy(*, builtin_regexes: bool, regex_rules: list[object]) -> None:
        pass

    assert "regex_limits" not in _detection_regex_kwargs(legacy, Detect())
    with pytest.raises(InvalidConfigError, match="does not accept regex_limits"):
        _detection_regex_kwargs(legacy, Detect(regex_limits={"max_matches_per_rule": 123}))


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_matches_per_rule", 0),
        ("max_matches_per_rule", 1.5),
        ("min_timeout_seconds", 0),
        ("min_timeout_seconds", float("inf")),
        ("timeout_seconds_per_million_characters", 0),
        ("timeout_seconds_per_million_characters", float("nan")),
    ],
)
def test_regex_limits_reject_invalid_values(field: str, value: float) -> None:
    with pytest.raises(ValidationError):
        RegexLimits.model_validate({field: value})


def test_max_matches_per_rule_accepts_a_digit_string() -> None:
    """The CLI passes text; a string of digits is the number it spells."""
    assert RegexLimits.model_validate({"max_matches_per_rule": "5"}).max_matches_per_rule == 5


@pytest.mark.parametrize("value", ["5.5", "5.0", "-5", "five", True, 5.0])
def test_max_matches_per_rule_still_refuses_non_integers(value: object) -> None:
    with pytest.raises(ValidationError):
        RegexLimits.model_validate({"max_matches_per_rule": value})
