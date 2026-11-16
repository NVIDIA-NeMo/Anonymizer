# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from anonymizer.engine.detection.detection_workflow import _get_augment_prompt, _get_validation_prompt


def _example_labels(prompt: str) -> set[str]:
    output = prompt.split("Output: ", 1)[1].split("\n", 1)[0]
    return {entity["label"] for entity in json.loads(output)["entities"]}


@pytest.mark.parametrize(
    ("labels", "strict", "excluded", "expected"),
    [
        (["first_name", "city"], True, [], {"first_name", "city"}),
        (["email"], True, [], set()),
        (["first_name", "last_name"], False, ["employment_status"], {"first_name", "last_name", "city"}),
    ],
)
def test_worked_example_uses_only_permitted_labels(
    labels: list[str], strict: bool, excluded: list[str], expected: set[str]
) -> None:
    prompt = _get_augment_prompt(
        data_summary=None, labels=labels, strict_labels=strict, excluded_entity_labels=excluded
    )

    assert _example_labels(prompt) == expected


def test_excluded_labels_reach_both_prompts_only_when_configured() -> None:
    labels = ["first_name", "city"]
    augment = _get_augment_prompt(data_summary=None, labels=labels, excluded_entity_labels=["age"])
    validation = _get_validation_prompt(data_summary=None, labels=labels, excluded_entity_labels=["age"])

    assert "excluded types or close synonyms of them: age" in augment
    assert "Excluded entity types:" in validation and "reclass for: age" in validation
    assert "excluded types" not in _get_augment_prompt(data_summary=None, labels=labels)
    assert "Excluded entity types" not in _get_validation_prompt(data_summary=None, labels=labels)


def test_label_rules_render_only_for_active_labels() -> None:
    with_rule = _get_augment_prompt(data_summary=None, labels=["unique_id", "city"])
    without_rule = _get_augment_prompt(data_summary=None, labels=["city"])

    assert "unique_id rules" in with_rule
    assert "Other information:" not in without_rule


def test_missing_data_summary_is_omitted() -> None:
    assert "Data summary" not in _get_augment_prompt(data_summary=None, labels=["city"])
    assert "Data summary: legal filings" in _get_augment_prompt(data_summary="legal filings", labels=["city"])


def _validation_example_decisions(labels: list[str], excluded: list[str]) -> dict[str, dict[str, str]]:
    prompt = _get_validation_prompt(data_summary=None, labels=labels, excluded_entity_labels=excluded)
    output = next(line for line in prompt.splitlines() if line.startswith('Output: {"decisions"'))
    return {item["id"]: item for item in json.loads(output.removeprefix("Output: "))["decisions"]}


@pytest.mark.parametrize(
    ("labels", "excluded"),
    [
        (["first_name", "age", "city", "state"], ["age", "city"]),
        (["first_name", "state"], []),
        (["email"], []),
    ],
)
def test_validator_example_never_keeps_or_reclasses_to_unpermitted_labels(
    labels: list[str], excluded: list[str]
) -> None:
    permitted = set(labels) - set(excluded)
    decisions = _validation_example_decisions(labels, excluded)

    assert all(d["proposed_label"] in permitted for d in decisions.values() if d["decision"] == "reclass")
    assert decisions["c3"]["decision"] == ("keep" if "age" in permitted else "drop")
    assert decisions["c4"]["decision"] == ("reclass" if "city" in permitted else "drop")


@pytest.mark.parametrize("strict", [False, True])
def test_augment_prompt_has_no_unfilled_placeholders(strict: bool) -> None:
    prompt = _get_augment_prompt(data_summary=None, labels=["first_name"], strict_labels=strict)

    assert "<<" not in prompt
