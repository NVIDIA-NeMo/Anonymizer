# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import Any

import pytest

from anonymizer.engine.constants import (
    COL_DOMAIN,
    COL_MEANING_UNITS_SERIALIZED,
    COL_QUALITY_QA,
    COL_SENSITIVITY_DISPOSITION,
    COL_TEXT,
)
from anonymizer.engine.rewrite.chunked_steps import (
    WindowedStepParams,
    _compile_template,
    run_windowed_step,
)
from anonymizer.engine.rewrite.domain_classification import _first_output, _get_domain_classification_prompt
from anonymizer.engine.rewrite.qa_generation import (
    _batch_units_by_size,
    _concat_meaning_units,
    _get_quality_qa_prompt,
    generate_quality_qa_row,
)
from anonymizer.engine.rewrite.sensitivity_disposition import _make_disposition_merge
from anonymizer.engine.schemas import (
    DomainClassificationSchema,
    MeaningUnitsSchema,
    QualityQAPairsSchema,
    SensitivityDispositionSchema,
)


class _FakeFacade:
    """Facade stub that parses a fixed JSON response on every ``generate`` call."""

    def __init__(self, response_obj: dict[str, Any]) -> None:
        self._payload = "```json\n" + json.dumps(response_obj) + "\n```"
        self.calls = 0

    def generate(self, *, prompt: Any, parser: Any, system_prompt: Any = None, purpose: Any = None, **_: Any) -> Any:
        self.calls += 1
        return parser(self._payload), []


def _disposition_entry(value: str, risk: str, method: str) -> dict[str, Any]:
    return {
        "id": 1,
        "source": "tagged",
        "category": "direct_identifier",
        "sensitivity": "high",
        "entity_label": "first_name",
        "entity_value": value,
        "protection_reason": "name is identifying in this document",
        "protection_method_suggestion": method,
        "combined_risk_level": risk,
    }


# ---------------------------------------------------------------------------
# run_windowed_step
# ---------------------------------------------------------------------------


def test_fast_path_single_call_when_under_cap() -> None:
    facade = _FakeFacade({"domain": "OTHER", "domain_confidence": 0.9})
    row = run_windowed_step(
        {COL_TEXT: "short"},
        WindowedStepParams(
            alias="d",
            prompt_template=_get_domain_classification_prompt(None),
            output_column=COL_DOMAIN,
            text_column=COL_TEXT,
            max_render_chars=1_000_000,
            first_only=True,
        ),
        {"d": facade},
        schema=DomainClassificationSchema,
        merge_fn=_first_output,
        purpose_prefix="domain",
    )
    assert facade.calls == 1
    assert DomainClassificationSchema.model_validate(row[COL_DOMAIN]).domain.value == "OTHER"


def test_first_only_classifies_just_one_window() -> None:
    facade = _FakeFacade({"domain": "OTHER", "domain_confidence": 0.9})
    long_text = ("x" * 4000 + "\n") * 4
    run_windowed_step(
        {COL_TEXT: long_text},
        WindowedStepParams(
            alias="d",
            prompt_template=_get_domain_classification_prompt(None),
            output_column=COL_DOMAIN,
            text_column=COL_TEXT,
            max_render_chars=4000,
            safety_margin_chars=0,
            first_only=True,
        ),
        {"d": facade},
        schema=DomainClassificationSchema,
        merge_fn=_first_output,
        purpose_prefix="domain",
    )
    assert facade.calls == 1


def test_missing_alias_raises() -> None:
    with pytest.raises(KeyError):
        run_windowed_step(
            {COL_TEXT: "x"},
            WindowedStepParams(
                alias="missing",
                prompt_template="{{ _text }}",
                output_column=COL_DOMAIN,
                text_column=COL_TEXT,
                max_render_chars=1000,
            ),
            {},
            schema=DomainClassificationSchema,
            merge_fn=_first_output,
            purpose_prefix="domain",
        )


class _FlakyFacade:
    """Facade stub that raises on selected (1-indexed) calls, else returns fixed JSON."""

    def __init__(
        self, response_obj: dict[str, Any], *, fail_calls: tuple[int, ...] = (), fail_all: bool = False
    ) -> None:
        self._payload = "```json\n" + json.dumps(response_obj) + "\n```"
        self._fail_calls = set(fail_calls)
        self._fail_all = fail_all
        self.calls = 0

    def generate(self, *, prompt: Any, parser: Any, system_prompt: Any = None, purpose: Any = None, **_: Any) -> Any:
        self.calls += 1
        if self._fail_all or self.calls in self._fail_calls:
            raise RuntimeError("transient model error")
        return parser(self._payload), []


def _multi_window_params() -> WindowedStepParams:
    return WindowedStepParams(
        alias="d",
        prompt_template=_get_domain_classification_prompt(None),
        output_column=COL_DOMAIN,
        text_column=COL_TEXT,
        max_render_chars=4000,
        safety_margin_chars=0,
    )


def test_windowed_step_skips_failed_window_and_merges_survivors() -> None:
    # First window's model call fails; remaining windows still run and merge.
    facade = _FlakyFacade({"domain": "OTHER", "domain_confidence": 0.9}, fail_calls=(1,))
    long_text = ("x" * 4000 + "\n") * 3
    row = run_windowed_step(
        {COL_TEXT: long_text},
        _multi_window_params(),
        {"d": facade},
        schema=DomainClassificationSchema,
        merge_fn=_first_output,
        purpose_prefix="domain",
    )
    assert facade.calls >= 2  # the window after the failure was still attempted
    assert DomainClassificationSchema.model_validate(row[COL_DOMAIN]).domain.value == "OTHER"


def test_windowed_step_raises_when_all_windows_fail() -> None:
    facade = _FlakyFacade({"domain": "OTHER", "domain_confidence": 0.9}, fail_all=True)
    long_text = ("x" * 4000 + "\n") * 3
    with pytest.raises(RuntimeError, match="all .* window"):
        run_windowed_step(
            {COL_TEXT: long_text},
            _multi_window_params(),
            {"d": facade},
            schema=DomainClassificationSchema,
            merge_fn=_first_output,
            purpose_prefix="domain",
        )


# ---------------------------------------------------------------------------
# Merges
# ---------------------------------------------------------------------------


def test_disposition_merge_keeps_highest_risk_and_reids() -> None:
    low = SensitivityDispositionSchema.model_validate(
        {"sensitivity_disposition": [_disposition_entry("Alice", "low", "leave_as_is")]}
    )
    high = SensitivityDispositionSchema.model_validate(
        {"sensitivity_disposition": [_disposition_entry("Alice", "high", "replace")]}
    )
    merged = _make_disposition_merge(SensitivityDispositionSchema)([low, high])
    entries = SensitivityDispositionSchema.model_validate(merged).sensitivity_disposition
    assert len(entries) == 1  # deduped by (source, label, value)
    assert entries[0].combined_risk_level == "high"
    assert entries[0].id == 1


def test_meaning_units_concat_reids_sequentially() -> None:
    out_a = MeaningUnitsSchema.model_validate(
        {"units": [{"id": 1, "aspect": "role", "unit": "works in tech", "importance": "critical"}]}
    )
    out_b = MeaningUnitsSchema.model_validate(
        {"units": [{"id": 1, "aspect": "process", "unit": "follows a workflow", "importance": "important"}]}
    )
    merged = _concat_meaning_units([out_a, out_b])
    units = MeaningUnitsSchema.model_validate(merged).units
    assert [u.id for u in units] == [1, 2]


# ---------------------------------------------------------------------------
# Quality-QA batching
# ---------------------------------------------------------------------------


def test_batch_units_by_size_splits_when_over_allowance() -> None:
    units = [{"id": i, "aspect": "role", "unit": f"unit number {i}", "importance": "important"} for i in range(10)]
    unit_len = len(json.dumps(units[0], ensure_ascii=False)) + 1
    base = 1000
    budget = base + 3 * unit_len  # ~3 units per batch
    batches = _batch_units_by_size(units, base, budget)
    assert len(batches) > 1
    assert sum(len(b) for b in batches) == len(units)
    assert all(len(b) <= 3 for b in batches)


def test_quality_qa_batches_and_reids() -> None:
    prompt = _get_quality_qa_prompt()
    units = [
        {"id": i + 1, "aspect": "role", "unit": f"unit text number {i} about work", "importance": "important"}
        for i in range(12)
    ]
    row = {COL_MEANING_UNITS_SERIALIZED: json.dumps(units, ensure_ascii=False)}
    base = len(_compile_template(prompt).render(**{**row, COL_MEANING_UNITS_SERIALIZED: "[]"}))
    unit_len = len(json.dumps(units[0], ensure_ascii=False)) + 1
    cap = base + 3 * unit_len
    facade = _FakeFacade(
        {
            "items": [
                {"id": 1, "aspect": "role", "importance": "critical", "question": "q1?", "reference_answer": "a1"},
                {"id": 2, "aspect": "role", "importance": "important", "question": "q2?", "reference_answer": "a2"},
            ]
        }
    )
    out = generate_quality_qa_row(
        dict(row),
        {"q": facade},
        alias="q",
        prompt_template=prompt,
        max_render_chars=cap,
        safety_margin_chars=0,
    )
    items = QualityQAPairsSchema.model_validate(out[COL_QUALITY_QA]).items
    assert facade.calls > 1  # batched
    assert [it.id for it in items] == list(range(1, len(items) + 1))
    assert len(items) == 2 * facade.calls
    assert COL_SENSITIVITY_DISPOSITION not in out  # sanity: only quality QA written
