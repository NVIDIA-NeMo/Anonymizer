# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Label resolution and the example merge, through `Detect.resolve()` and `config.entity_labels`."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from anonymizer.config.anonymizer_config import Detect
from anonymizer.config.entity_labels import merge_entity_label_examples
from anonymizer.config.regex import RegexRule
from anonymizer.engine.constants import DEFAULT_ENTITY_LABELS, ENTITY_LABEL_EXAMPLES


def test_unset_labels_resolve_to_the_defaults() -> None:
    options = Detect().resolve()
    assert options.ordered_labels == tuple(DEFAULT_ENTITY_LABELS)
    assert options.entity_labels is None
    assert options.strict_labels is False


def test_explicit_labels_do_not_add_the_defaults_and_are_strict() -> None:
    options = Detect(
        entity_labels=["vendor_api_key"], entity_label_examples={"vendor_api_key": ["acme_live_abc123"]}
    ).resolve()
    assert options.ordered_labels == ("vendor_api_key",)
    assert options.strict_labels is True


def test_explicit_labels_are_normalized() -> None:
    assert Detect(entity_labels=["FIRST_NAME", " email "]).resolve().ordered_labels == ("email", "first_name")


def test_enabled_custom_rule_labels_follow_the_defaults() -> None:
    options = Detect(regex_rules=[RegexRule(label="case_id", pattern=r"CASE-\d+")]).resolve()
    assert options.ordered_labels == (*DEFAULT_ENTITY_LABELS, "case_id")


def test_merge_puts_configured_examples_after_built_ins_without_global_mutation() -> None:
    original = list(ENTITY_LABEL_EXAMPLES["api_key"])

    validator, augmenter = merge_entity_label_examples(["api_key"], {"api_key": ["sk-ant-api03-abc123", original[0]]})

    assert validator["api_key"] == (*original, "sk-ant-api03-abc123")
    assert augmenter == {"api_key": ("sk-ant-api03-abc123", original[0])}
    assert ENTITY_LABEL_EXAMPLES["api_key"] == original


def test_resolve_gives_every_label_validator_examples_and_ignores_excluded_examples() -> None:
    options = Detect(
        entity_labels=["first_name", "vendor_key", "email"],
        entity_label_examples={"vendor_key": ["acme_live_1"], "email": ["a@example.com"]},
        excluded_entity_labels=["email"],
    ).resolve()
    assert list(options.validator_examples) == ["first_name", "vendor_key"]
    assert options.validator_examples["vendor_key"] == ("acme_live_1",)
    assert dict(options.augmenter_examples) == {"vendor_key": ("acme_live_1",)}


def test_sequential_resolutions_are_isolated() -> None:
    first = Detect(
        entity_labels=[*DEFAULT_ENTITY_LABELS, "vendor_api_key"],
        entity_label_examples={"vendor_api_key": ["acme_live_abc123"]},
    ).resolve()
    second = Detect().resolve()

    assert "vendor_api_key" in first.ordered_labels
    assert "vendor_api_key" not in second.ordered_labels
    assert "vendor_api_key" not in second.validator_examples
    assert dict(second.augmenter_examples) == {}


def test_example_label_outside_the_active_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="outside the active label set"):
        Detect(entity_label_examples={"vendor_api_key": ["acme_live_abc123"]})


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"entity_labels": ["first_name", "email", "city"], "excluded_entity_labels": ["email"]},
            ("city", "first_name"),
        ),
        ({"entity_labels": ["first_name", " Email "], "excluded_entity_labels": [" EMAIL "]}, ("first_name",)),
        ({"entity_labels": ["email", "city"]}, ("city", "email")),
    ],
)
def test_exclusions_remove_normalized_labels(kwargs: dict[str, Any], expected: tuple[str, ...]) -> None:
    assert Detect(**kwargs).resolve().ordered_labels == expected


def test_exclusions_apply_to_the_defaults() -> None:
    labels = Detect(excluded_entity_labels=["ssn", "first_name"]).resolve().ordered_labels
    assert "ssn" not in labels
    assert "first_name" not in labels
    assert "email" in labels


def test_empty_effective_set_is_rejected() -> None:
    with pytest.raises(ValidationError, match="empty effective detection set"):
        Detect(entity_labels=["email"], excluded_entity_labels=["email"])


@pytest.mark.parametrize("label", ["NAGROUP_g1", " nagroup_G42 "])
def test_reserved_prompt_labels_are_rejected_at_construction(label: str) -> None:
    with pytest.raises(ValidationError, match="reserved for internal prompt markup"):
        Detect(entity_labels=["email", label])
    with pytest.raises(ValidationError, match="reserved for internal prompt markup"):
        RegexRule(label=label, pattern=r"G\d+")
    with pytest.raises(ValidationError, match="reserved for internal prompt markup"):
        Detect(entity_label_examples={label: ["x"]})


def _excluding_every_label() -> Detect:
    detect = Detect(entity_labels=["email", "phone_number"], excluded_entity_labels=["phone_number"])
    assert detect.excluded_entity_labels is not None
    detect.excluded_entity_labels.append("email")
    return detect


def _appending_a_reserved_label() -> Detect:
    detect = Detect(entity_labels=["email"])
    assert detect.entity_labels is not None
    detect.entity_labels.append("nagroup_g1")
    return detect


def _adding_an_example_outside_the_labels() -> Detect:
    detect = Detect(entity_labels=["email"])
    detect.entity_label_examples["vendor_key"] = ["k"]
    return detect


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_excluding_every_label, "empty effective detection set"),
        (_appending_a_reserved_label, "reserved for internal prompt markup"),
        (_adding_an_example_outside_the_labels, "outside the active label set"),
    ],
)
def test_resolve_rejects_list_edits_made_after_construction(edit: Any, message: str) -> None:
    detect = edit()
    with pytest.raises(ValidationError, match=message):
        detect.resolve()
