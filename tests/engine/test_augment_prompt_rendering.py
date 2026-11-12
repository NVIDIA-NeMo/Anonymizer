# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline rendering checks for the augmenter prompt across label-policy configurations.

No model calls. Each case resolves labels the way the workflow does, renders the
augmenter prompt, and checks it for internal contradictions. Item numbers refer to
``plans/augmenter-prompt-edge-cases.md``.

Masked-view, chunked, and reasonless variants are not implemented on this branch;
add them as dimensions when they land.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest
from jinja2 import Environment, StrictUndefined

from anonymizer.engine.constants import COL_TEXT, DEFAULT_ENTITY_LABELS
from anonymizer.engine.detection.detection_workflow import _get_augment_prompt, _get_validation_prompt
from anonymizer.engine.detection.entity_label_examples import (
    ResolvedEntityLabelConfig,
    resolve_entity_label_config,
)

# Labels the worked example can demonstrate before filtering.
_EXAMPLE_LABELS = frozenset({"first_name", "last_name", "city", "employment_status"})

# Labels that carry label-specific "Other information" rules.
_RULE_LABELS = ("unique_id", "ipv4", "filename")

_STRICT_WORDING = ("Use ONLY labels from this list", "Do not create new labels")
_PERMISSIVE_WORDING = ("Strongly prefer labels from this list", "create a concise snake_case label")

_IDENTITY_LABELS = ["first_name", "last_name", "email", "user_name", "company_name"]
_RETAIL_LABELS = ["order_id", "payment_id", "email"]


@dataclass(frozen=True)
class PromptCase:
    name: str
    entity_labels: list[str] | None
    excluded: list[str] = field(default_factory=list)
    data_summary: str | None = None
    configured_examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def strict(self) -> bool:
        return self.entity_labels is not None

    def resolve(self) -> ResolvedEntityLabelConfig:
        return resolve_entity_label_config(
            entity_labels=self.entity_labels,
            excluded_entity_labels=self.excluded,
            entity_label_examples=self.configured_examples or None,
        )

    def active_labels(self) -> set[str]:
        return set(self.resolve().labels)


CASES = [
    PromptCase("default", None),
    PromptCase("default_with_summary", None, data_summary="Retail customer-support chats"),
    PromptCase("default_excluding_occupation", None, excluded=["occupation"]),
    PromptCase("strict_defaults", list(DEFAULT_ENTITY_LABELS), data_summary="General text"),
    PromptCase("strict_identity", _IDENTITY_LABELS, data_summary="Meeting transcripts"),
    PromptCase("strict_identity_excluding_email", _IDENTITY_LABELS, excluded=["email"]),
    PromptCase("strict_retail", _RETAIL_LABELS, data_summary="Retail customer-support chats"),
    PromptCase(
        "strict_retail_with_examples",
        _RETAIL_LABELS,
        data_summary="Retail customer-support chats",
        configured_examples={"order_id": ["#W0000000"]},
    ),
    PromptCase(
        "strict_retail_excluding_configured_label",
        _RETAIL_LABELS,
        excluded=["order_id"],
        configured_examples={"order_id": ["#W0000000"]},
    ),
]


def _params() -> list[Any]:
    return [pytest.param(case, id=case.name) for case in CASES]


def _render_augment(case: PromptCase) -> str:
    config = case.resolve()
    template = _get_augment_prompt(
        data_summary=case.data_summary,
        labels=config.labels,
        strict_labels=config.strict_labels,
        configured_examples=config.augmenter_examples,
        excluded_entity_labels=case.excluded,
    )
    return Environment(undefined=StrictUndefined).from_string(template).render({COL_TEXT: "SOURCE_TEXT_SENTINEL"})


def _example_output_labels(rendered: str) -> set[str]:
    output_line = next(line for line in rendered.splitlines() if line.startswith("Output: "))
    payload = json.loads(output_line.removeprefix("Output: "))
    return {item["label"] for item in payload["entities"]}


def _label_line(rendered: str) -> str:
    lines = rendered.splitlines()
    header = next(i for i, line in enumerate(lines) if "entity classes" in line)
    return lines[header + 1]


@pytest.mark.parametrize("case", _params())
def test_renders_without_leftover_placeholders(case: PromptCase) -> None:
    rendered = _render_augment(case)

    assert "SOURCE_TEXT_SENTINEL" in rendered
    for marker in ("<<", "{{", "{%"):
        assert marker not in rendered


@pytest.mark.parametrize("case", _params())
def test_strict_and_permissive_wording_are_exclusive(case: PromptCase) -> None:
    rendered = _render_augment(case)
    present, absent = (_STRICT_WORDING, _PERMISSIVE_WORDING) if case.strict else (_PERMISSIVE_WORDING, _STRICT_WORDING)

    for phrase in present:
        assert phrase in rendered
    for phrase in absent:
        assert phrase not in rendered


@pytest.mark.parametrize("case", _params())
def test_label_line_lists_exactly_the_active_labels(case: PromptCase) -> None:
    rendered = _render_augment(case)
    listed = {label.strip() for label in _label_line(rendered).split(",")}

    assert listed == case.active_labels()
    assert not listed & set(case.excluded)


@pytest.mark.parametrize("case", _params())
def test_configured_examples_render_only_for_active_labels(case: PromptCase) -> None:
    rendered = _render_augment(case)
    active = case.active_labels()

    for label, examples in case.configured_examples.items():
        for example in examples:
            assert (example in rendered) == (label in active)


def test_empty_strict_label_list_is_rejected_before_any_prompt() -> None:
    with pytest.raises(ValueError, match="effective detection label set is empty"):
        resolve_entity_label_config(entity_labels=[])


def test_excluding_every_strict_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="effective detection label set is empty"):
        resolve_entity_label_config(entity_labels=["email"], excluded_entity_labels=["email"])


@pytest.mark.parametrize("case", _params())
def test_example_uses_only_allowed_labels(case: PromptCase) -> None:
    """Item 1: a strict example never demonstrates a label outside the list."""
    rendered = _render_augment(case)
    example_labels = _example_output_labels(rendered)

    assert not example_labels & set(case.excluded)
    if case.strict:
        assert example_labels <= case.active_labels()
        if not example_labels:
            assert "nothing is returned" in rendered
    else:
        assert example_labels <= _EXAMPLE_LABELS | case.active_labels()


@pytest.mark.parametrize("case", _params())
def test_rules_only_mention_active_labels(case: PromptCase) -> None:
    """Item 2: label-specific rules render only for active labels."""
    rendered = _render_augment(case)
    active = case.active_labels()

    if not active & set(_RULE_LABELS):
        assert "Other information:" not in rendered
        return
    rules = rendered.split("Other information:")[1].split("Additional extraction requirements:")[0]
    for label in _RULE_LABELS:
        assert (label in rules) == (label in active)


@pytest.mark.parametrize("case", _params())
def test_augmenter_prompt_states_excluded_labels(case: PromptCase) -> None:
    """Item 3: excluded labels are named in the augmenter prompt."""
    rendered = _render_augment(case)
    do_not_return = [line for line in rendered.splitlines() if "do not return" in line.lower()]

    for label in case.excluded:
        assert any(label in line for line in do_not_return)


@pytest.mark.parametrize("case", _params())
def test_validator_prompt_states_excluded_labels(case: PromptCase) -> None:
    """Item 3: excluded labels are named in the validator prompt."""
    config = case.resolve()
    prompt = _get_validation_prompt(
        data_summary=case.data_summary,
        labels=config.labels,
        examples_by_label=config.validator_examples,
        configured_examples=config.augmenter_examples,
        excluded_entity_labels=case.excluded,
    )
    do_not_return = [line for line in prompt.splitlines() if "do not return" in line.lower()]

    for label in case.excluded:
        assert any(label in line for line in do_not_return)


@pytest.mark.parametrize("case", _params())
def test_missing_data_summary_is_omitted(case: PromptCase) -> None:
    """Item 5: no literal 'Not provided' when the summary is absent."""
    rendered = _render_augment(case)

    assert "Not provided" not in rendered
    if case.data_summary is not None:
        assert case.data_summary in rendered
