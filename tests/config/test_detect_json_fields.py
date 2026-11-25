# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Detect fields the CLI fills with JSON text."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anonymizer import BuiltinRegex, Detect, RegexRule


def test_entity_label_examples_accepts_a_json_object() -> None:
    detect = Detect(entity_label_examples='{"first_name": ["Alice", "Bob"]}')
    assert detect.entity_label_examples == {"first_name": ["Alice", "Bob"]}


def test_regex_rules_accepts_one_json_rule_per_item() -> None:
    detect = Detect(
        regex_rules=['{"label": "ticket_id", "pattern": "TKT-[0-9]+"}', '{"label": "email", "enabled": false}']
    )
    custom, builtin = detect.regex_rules
    assert isinstance(custom, RegexRule) and custom.pattern == "TKT-[0-9]+"
    assert isinstance(builtin, BuiltinRegex) and builtin.enabled is False


def test_regex_rules_accepts_a_json_list() -> None:
    detect = Detect(regex_rules='[{"label": "ticket_id", "pattern": "TKT-[0-9]+"}]')
    assert [rule.label for rule in detect.regex_rules] == ["ticket_id"]


@pytest.mark.parametrize("field", ["entity_label_examples", "regex_rules"])
def test_text_that_is_not_json_is_refused(field: str) -> None:
    with pytest.raises(ValidationError, match="must be JSON"):
        Detect.model_validate({field: "not json"})


def test_regex_rules_accepts_a_json_list_as_one_item() -> None:
    """The CLI passes ``--regex-rules '[...]'`` as one item holding a list."""
    detect = Detect(
        regex_rules=['[{"label": "ticket_id", "pattern": "TKT-[0-9]+"}, {"label": "email", "enabled": false}]']
    )
    assert [rule.label for rule in detect.regex_rules] == ["ticket_id", "email"]
