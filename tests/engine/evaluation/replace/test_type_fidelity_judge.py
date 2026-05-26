# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import (
    COL_REPLACEMENT_MAP,
    COL_TYPE_FIDELITY_INVALID_REPLACEMENTS,
    COL_TYPE_FIDELITY_JUDGE,
    COL_TYPE_FIDELITY_VALID,
)
from anonymizer.engine.evaluation.replace.type_fidelity_judge import (
    TypeFidelityJudgeWorkflow,
    TypeFidelityJudgmentSchema,
    _flatten_judgment,
    _judge_prompt,
    _label_examples_for_judge,
    _replacements_for_judge,
)

# ---------------------------------------------------------------------------
# Tests: _judge_prompt
# ---------------------------------------------------------------------------


def test_judge_prompt_uses_xml_sections() -> None:
    prompt = _judge_prompt()
    for tag in (
        "scope",
        "replacements",
        "reference_label_examples",
        "task",
        "class_membership_rules",
        "format_type_rules",
        "edge_cases",
        "output_rules",
    ):
        assert f"<{tag}>" in prompt
        assert f"</{tag}>" in prompt


def test_judge_prompt_iterates_replacement_triples() -> None:
    prompt = _judge_prompt()
    assert "for entry in _replacements_for_type_fidelity_judge" in prompt
    assert "entry.original" in prompt
    assert "entry.label" in prompt
    assert "entry.synthetic" in prompt


def test_judge_prompt_disambiguates_from_neighbouring_metrics() -> None:
    """Prompt must call out that semantic attributes and cross-entity consistency are
    OUT of scope, otherwise the judge will silently penalize valid replacements."""
    prompt = _judge_prompt()
    assert "DIFFERENT metric" in prompt
    assert "gender of a name" in prompt
    assert "city/state" in prompt


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


def test_replacements_for_judge_flattens_dict_form() -> None:
    raw = {
        "replacements": [
            {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
            {"original": "Acme", "label": "company_name", "synthetic": "NovaCorp"},
        ]
    }
    assert _replacements_for_judge(raw) == [
        {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
        {"original": "Acme", "label": "company_name", "synthetic": "NovaCorp"},
    ]


def test_replacements_for_judge_accepts_json_string() -> None:
    payload = '{"replacements":[{"original":"Alice","label":"first_name","synthetic":"Maya"}]}'
    assert _replacements_for_judge(payload) == [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]


def test_replacements_for_judge_returns_empty_for_malformed() -> None:
    assert _replacements_for_judge(None) == []
    assert _replacements_for_judge("not json") == []
    assert _replacements_for_judge(42) == []
    assert _replacements_for_judge({"replacements": "oops"}) == []


def test_label_examples_for_judge_only_includes_labels_in_replacements() -> None:
    examples_json = _label_examples_for_judge([{"original": "Alice", "label": "first_name", "synthetic": "Maya"}])
    assert "first_name" in examples_json
    assert "ssn" not in examples_json


def test_label_examples_for_judge_empty_when_no_replacements() -> None:
    assert _label_examples_for_judge([]) == "{}"


# ---------------------------------------------------------------------------
# Tests: _flatten_judgment
# ---------------------------------------------------------------------------


def test_flatten_judgment_all_valid_path() -> None:
    valid, invalid = _flatten_judgment({"all_valid": True, "invalid_replacements": []})
    assert valid is True
    assert invalid == []


def test_flatten_judgment_returns_invalid_entries() -> None:
    raw = {
        "all_valid": False,
        "invalid_replacements": [
            {
                "original": "Alice",
                "label": "first_name",
                "synthetic": "[REDACTED]",
                "reasoning": "class membership: placeholder, not a person name",
            },
        ],
    }
    valid, invalid = _flatten_judgment(raw)
    assert valid is False
    assert invalid == [
        {
            "original": "Alice",
            "label": "first_name",
            "synthetic": "[REDACTED]",
            "reasoning": "class membership: placeholder, not a person name",
        }
    ]


def test_flatten_judgment_accepts_pydantic_model() -> None:
    payload = TypeFidelityJudgmentSchema(all_valid=True, invalid_replacements=[])
    valid, invalid = _flatten_judgment(payload)
    assert valid is True
    assert invalid == []


def test_flatten_judgment_none_returns_unavailable_sentinel() -> None:
    assert _flatten_judgment(None) == (None, [])


def test_flatten_judgment_malformed_returns_unavailable_sentinel() -> None:
    assert _flatten_judgment("not json") == (None, [])
    assert _flatten_judgment(42) == (None, [])
    assert _flatten_judgment({"missing": "all_valid"}) == (None, [])


# ---------------------------------------------------------------------------
# Tests: TypeFidelityJudgeWorkflow.evaluate
# ---------------------------------------------------------------------------


def _map_payload(items: list[dict]) -> dict:
    return {"replacements": items}


def test_evaluate_short_circuits_when_no_replacements(
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    df = pd.DataFrame({COL_REPLACEMENT_MAP: [_map_payload([])]})

    class _UnusedAdapter:
        def run_workflow(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("run_workflow should not be called when there are no replacements")

    wf = TypeFidelityJudgeWorkflow(adapter=_UnusedAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_replace_model_selection)
    assert result.failed_records == []
    assert bool(result.dataframe[COL_TYPE_FIDELITY_VALID].iloc[0]) is True
    assert result.dataframe[COL_TYPE_FIDELITY_INVALID_REPLACEMENTS].iloc[0] == []


def test_evaluate_invokes_adapter_with_correct_alias_and_schema(
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    df = pd.DataFrame(
        {
            COL_REPLACEMENT_MAP: [
                _map_payload(
                    [
                        {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                        {"original": "alice@x.com", "label": "email", "synthetic": "not-an-email"},
                    ]
                )
            ]
        }
    )

    captured: dict = {}

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            captured["columns"] = columns
            captured["workflow_name"] = workflow_name
            out = frame.copy()
            out[COL_TYPE_FIDELITY_JUDGE] = [
                {
                    "all_valid": False,
                    "invalid_replacements": [
                        {
                            "original": "alice@x.com",
                            "label": "email",
                            "synthetic": "not-an-email",
                            "reasoning": "format: missing '@' and domain",
                        }
                    ],
                }
            ]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = TypeFidelityJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_replace_model_selection)

    assert captured["workflow_name"] == "replace-type-fidelity-judge"
    col = captured["columns"][0]
    assert isinstance(col, LLMStructuredColumnConfig)
    assert col.name == COL_TYPE_FIDELITY_JUDGE
    assert col.model_alias == stub_replace_model_selection.type_fidelity_judge
    assert col.output_format == TypeFidelityJudgmentSchema.model_json_schema()

    assert bool(result.dataframe[COL_TYPE_FIDELITY_VALID].iloc[0]) is False
    invalid = result.dataframe[COL_TYPE_FIDELITY_INVALID_REPLACEMENTS].iloc[0]
    assert invalid == [
        {
            "original": "alice@x.com",
            "label": "email",
            "synthetic": "not-an-email",
            "reasoning": "format: missing '@' and domain",
        }
    ]


def test_evaluate_preserves_row_order_when_mixing_empty_and_populated_maps(
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    df = pd.DataFrame(
        {
            COL_REPLACEMENT_MAP: [
                _map_payload([]),
                _map_payload([{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]),
            ]
        }
    )

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            out = frame.copy()
            out[COL_TYPE_FIDELITY_JUDGE] = [{"all_valid": True, "invalid_replacements": []}]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = TypeFidelityJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_replace_model_selection)

    assert [bool(v) for v in result.dataframe[COL_TYPE_FIDELITY_VALID]] == [True, True]


def test_evaluate_marks_unavailable_for_malformed_payload(
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    df = pd.DataFrame(
        {COL_REPLACEMENT_MAP: [_map_payload([{"original": "Alice", "label": "first_name", "synthetic": "Maya"}])]}
    )

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            out = frame.copy()
            out[COL_TYPE_FIDELITY_JUDGE] = ["not json"]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = TypeFidelityJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_replace_model_selection)

    assert result.dataframe[COL_TYPE_FIDELITY_VALID].iloc[0] is None
    assert result.dataframe[COL_TYPE_FIDELITY_INVALID_REPLACEMENTS].iloc[0] == []
