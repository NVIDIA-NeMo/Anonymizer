# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anonymizer import Detect, RegexLimits


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
