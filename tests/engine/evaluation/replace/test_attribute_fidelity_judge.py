# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES,
    COL_ATTRIBUTE_FIDELITY_JUDGE,
    COL_ATTRIBUTE_FIDELITY_VALID,
    COL_REPLACEMENT_MAP,
)
from anonymizer.engine.evaluation.replace.attribute_fidelity_judge import (
    AttributeFidelityJudgeWorkflow,
    AttributeFidelityJudgmentSchema,
    _judge_prompt,
    _replacements_for_judge,
)

_flatten_judgment = AttributeFidelityJudgeWorkflow._flatten_judgment

# ---------------------------------------------------------------------------
# Tests: _judge_prompt
# ---------------------------------------------------------------------------


def test_judge_prompt_uses_xml_sections() -> None:
    prompt = _judge_prompt()
    for tag in ("scope", "replacements", "task", "salient_attributes_by_label", "rules"):
        assert f"<{tag}>" in prompt
        assert f"</{tag}>" in prompt


def test_judge_prompt_iterates_replacement_triples() -> None:
    prompt = _judge_prompt()
    assert "for entry in _replacements_for_attribute_fidelity_judge" in prompt
    assert "entry.original" in prompt
    assert "entry.label" in prompt
    assert "entry.synthetic" in prompt


def test_judge_prompt_carves_out_neighbouring_metrics() -> None:
    """Prompt must explicitly declare type fidelity and cross-entity coherence as out of scope."""
    prompt = _judge_prompt()
    assert "DIFFERENT metric" in prompt
    assert "city <-> state" in prompt  # mentions cross-entity case as out of scope


def test_judge_prompt_scopes_to_gender_and_age_bucket_only() -> None:
    """Prompt must restrict checks to the two designated attributes and skip everything else."""
    prompt = _judge_prompt()
    assert "GENDER OF NAME" in prompt
    assert "AGE BUCKET" in prompt
    assert "ALL OTHER LABELS — SKIP" in prompt


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


def test_replacements_for_judge_flattens_dict_form() -> None:
    raw = {
        "replacements": [
            {"original": "Sarah", "label": "first_name", "synthetic": "Michael"},
            {"original": "Tokyo", "label": "city", "synthetic": "Paris"},
        ]
    }
    assert _replacements_for_judge(raw) == [
        {"original": "Sarah", "label": "first_name", "synthetic": "Michael"},
        {"original": "Tokyo", "label": "city", "synthetic": "Paris"},
    ]


def test_replacements_for_judge_returns_empty_for_malformed() -> None:
    assert _replacements_for_judge(None) == []
    assert _replacements_for_judge("not json") == []
    assert _replacements_for_judge(42) == []


# ---------------------------------------------------------------------------
# Tests: _flatten_judgment
# ---------------------------------------------------------------------------


def test_flatten_judgment_all_valid_keeps_invalid_empty() -> None:
    raw = {
        "all_valid": True,
        "entities": [
            {
                "original": "Sarah",
                "label": "first_name",
                "synthetic": "Maria",
                "attributes_checked": ["gender"],
                "passes": True,
                "reasoning": "Both names imply feminine gender.",
            }
        ],
    }
    valid, invalid = _flatten_judgment(raw)
    assert valid is True
    assert invalid == []


def test_flatten_judgment_extracts_failing_entries_only() -> None:
    raw = {
        "all_valid": False,
        "entities": [
            {
                "original": "Sarah",
                "label": "first_name",
                "synthetic": "Maria",
                "attributes_checked": ["gender"],
                "passes": True,
                "reasoning": "Both feminine.",
            },
            {
                "original": "40",
                "label": "age",
                "synthetic": "12",
                "attributes_checked": ["age_bucket"],
                "passes": False,
                "reasoning": "Adult bucket changed to child.",
            },
        ],
    }
    valid, invalid = _flatten_judgment(raw)
    assert valid is False
    assert len(invalid) == 1
    assert invalid[0]["original"] == "40"
    assert invalid[0]["passes"] is False


def test_flatten_judgment_accepts_pydantic_model() -> None:
    payload = AttributeFidelityJudgmentSchema(all_valid=True, entities=[])
    valid, invalid = _flatten_judgment(payload)
    assert valid is True
    assert invalid == []


def test_flatten_judgment_none_returns_unavailable_sentinel() -> None:
    assert _flatten_judgment(None) == (None, [])


def test_flatten_judgment_malformed_returns_unavailable_sentinel() -> None:
    assert _flatten_judgment("not json") == (None, [])
    assert _flatten_judgment(42) == (None, [])
    assert _flatten_judgment({"missing": True}) == (None, [])


# ---------------------------------------------------------------------------
# Tests: AttributeFidelityJudgeWorkflow.evaluate
# ---------------------------------------------------------------------------


def _map_payload(items: list[dict]) -> dict:
    return {"replacements": items}


def test_evaluate_short_circuits_when_no_replacements(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    df = pd.DataFrame({COL_REPLACEMENT_MAP: [_map_payload([])]})

    class _UnusedAdapter:
        def run_workflow(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("run_workflow should not be called when there are no replacements")

    wf = AttributeFidelityJudgeWorkflow(adapter=_UnusedAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)
    assert result.failed_records == []
    assert bool(result.dataframe[COL_ATTRIBUTE_FIDELITY_VALID].iloc[0]) is True
    assert result.dataframe[COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES].iloc[0] == []


def test_evaluate_invokes_adapter_with_correct_alias_and_schema(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    df = pd.DataFrame(
        {
            COL_REPLACEMENT_MAP: [
                _map_payload(
                    [
                        {"original": "Sarah", "label": "first_name", "synthetic": "Michael"},
                        {"original": "40", "label": "age", "synthetic": "12"},
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
            out[COL_ATTRIBUTE_FIDELITY_JUDGE] = [
                {
                    "all_valid": False,
                    "entities": [
                        {
                            "original": "Sarah",
                            "label": "first_name",
                            "synthetic": "Michael",
                            "attributes_checked": ["gender"],
                            "passes": False,
                            "reasoning": "Feminine -> masculine.",
                        },
                        {
                            "original": "40",
                            "label": "age",
                            "synthetic": "12",
                            "attributes_checked": ["age_bucket"],
                            "passes": False,
                            "reasoning": "Adult -> child.",
                        },
                    ],
                }
            ]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = AttributeFidelityJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)

    assert captured["workflow_name"] == "replace-attribute-fidelity-judge"
    col = captured["columns"][0]
    assert isinstance(col, LLMStructuredColumnConfig)
    assert col.name == COL_ATTRIBUTE_FIDELITY_JUDGE
    assert col.model_alias == stub_evaluate_model_selection.replace_attribute_fidelity_judge
    assert col.output_format == AttributeFidelityJudgmentSchema.model_json_schema()

    assert bool(result.dataframe[COL_ATTRIBUTE_FIDELITY_VALID].iloc[0]) is False
    invalid = result.dataframe[COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES].iloc[0]
    assert len(invalid) == 2
    assert {item["original"] for item in invalid} == {"Sarah", "40"}


def test_evaluate_marks_unavailable_for_malformed_payload(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    df = pd.DataFrame(
        {COL_REPLACEMENT_MAP: [_map_payload([{"original": "Sarah", "label": "first_name", "synthetic": "Maria"}])]}
    )

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            out = frame.copy()
            out[COL_ATTRIBUTE_FIDELITY_JUDGE] = ["not json"]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = AttributeFidelityJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)

    assert result.dataframe[COL_ATTRIBUTE_FIDELITY_VALID].iloc[0] is None
    assert result.dataframe[COL_ATTRIBUTE_FIDELITY_INVALID_ENTITIES].iloc[0] == []
