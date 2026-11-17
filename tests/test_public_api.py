# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import anonymizer
from anonymizer import BUILTIN_REGEX_LABELS, BuiltinRegex
from anonymizer.config.regex import BUILTIN_REGEX_LABELS as CONFIG_BUILTIN_REGEX_LABELS
from anonymizer.engine.detection.regex_detection import resolve_regex_rules


def test_builtin_regex_labels_are_an_immutable_public_reexport() -> None:
    assert "BUILTIN_REGEX_LABELS" in anonymizer.__all__
    assert BUILTIN_REGEX_LABELS is CONFIG_BUILTIN_REGEX_LABELS
    assert isinstance(BUILTIN_REGEX_LABELS, tuple)
    assert len(BUILTIN_REGEX_LABELS) == len(set(BUILTIN_REGEX_LABELS))


@pytest.mark.parametrize("label", BUILTIN_REGEX_LABELS)
def test_public_builtin_labels_can_configure_builtin_rules(label: str) -> None:
    assert BuiltinRegex(label=label).label == label


def test_public_builtin_labels_have_registered_recognizers() -> None:
    rules = resolve_regex_rules(labels=list(BUILTIN_REGEX_LABELS), builtin_regexes=True, rules=[])
    assert {rule.label for rule in rules} == set(BUILTIN_REGEX_LABELS)
    assert all(rule.validator_id is not None for rule in rules)
