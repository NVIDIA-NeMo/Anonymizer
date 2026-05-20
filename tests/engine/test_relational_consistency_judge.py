# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.engine.constants import (
    COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS,
    COL_RELATIONAL_CONSISTENCY_JUDGE,
    COL_RELATIONAL_CONSISTENCY_VALID,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
)
from anonymizer.engine.replace.relational_consistency_judge import (
    RelationalConsistencyJudgeWorkflow,
    RelationalConsistencyJudgmentSchema,
    _flatten_judgment,
    _judge_prompt,
    _replacements_for_judge,
)

# ---------------------------------------------------------------------------
# Tests: _judge_prompt
# ---------------------------------------------------------------------------


def test_judge_prompt_uses_xml_sections() -> None:
    prompt = _judge_prompt()
    for tag in ("scope", "replaced_text", "replacements", "task", "relations_to_inspect", "rules", "edge_cases"):
        assert f"<{tag}>" in prompt
        assert f"</{tag}>" in prompt


def test_judge_prompt_references_replaced_text_column() -> None:
    prompt = _judge_prompt()
    assert COL_REPLACED_TEXT in prompt


def test_judge_prompt_iterates_replacement_triples() -> None:
    prompt = _judge_prompt()
    assert "for entry in _replacements_for_relational_consistency_judge" in prompt
    assert "entry.original" in prompt
    assert "entry.label" in prompt
    assert "entry.synthetic" in prompt


def test_judge_prompt_disambiguates_from_neighbouring_metrics() -> None:
    """Prompt must call out that type/format and semantic-attribute checks are out of scope."""
    prompt = _judge_prompt()
    assert "DIFFERENT metric" in prompt


def test_judge_prompt_requires_passing_relations_in_output() -> None:
    """Denominator depends on the judge listing passes AND fails."""
    prompt = _judge_prompt()
    assert "denominator" in prompt.lower()


def test_judge_prompt_blocks_generic_date_as_date_of_birth() -> None:
    """The judge must not treat a generic `date` (career year, etc.) as a `date_of_birth`.

    Regression guard for a real failure observed on the biographies dataset, where a
    sentence like "returning home in 2012" caused the judge to pair the `date` entity
    with `age` and compute `current_year - 2012 != age`, producing a false negative.
    """
    prompt = _judge_prompt()
    assert "literally `date_of_birth`" in prompt
    assert "generic `date`" in prompt
    assert "SKIP the temporal relation" in prompt


def test_judge_prompt_requires_literal_label_matching() -> None:
    """Relations are matched by the literal label field, not by inferring from the value's surface form."""
    prompt = _judge_prompt()
    assert "LITERAL `label` field" in prompt
    assert "Do NOT infer a label from" in prompt


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


def test_replacements_for_judge_flattens_dict_form() -> None:
    raw = {
        "replacements": [
            {"original": "Austin", "label": "city", "synthetic": "Portland"},
            {"original": "TX", "label": "state", "synthetic": "OR"},
        ]
    }
    assert _replacements_for_judge(raw) == [
        {"original": "Austin", "label": "city", "synthetic": "Portland"},
        {"original": "TX", "label": "state", "synthetic": "OR"},
    ]


def test_replacements_for_judge_returns_empty_for_malformed() -> None:
    assert _replacements_for_judge(None) == []
    assert _replacements_for_judge("not json") == []
    assert _replacements_for_judge(42) == []


# ---------------------------------------------------------------------------
# Tests: _flatten_judgment
# ---------------------------------------------------------------------------


def test_flatten_judgment_all_consistent_keeps_invalid_empty() -> None:
    raw = {
        "all_consistent": True,
        "relations": [
            {
                "description": "city <-> state",
                "entities": [
                    "Austin (city) -> Portland",
                    "TX (state) -> OR",
                ],
                "passes": True,
                "reasoning": "Portland is in Oregon.",
            }
        ],
    }
    valid, invalid = _flatten_judgment(raw)
    assert valid is True
    assert invalid == []


def test_flatten_judgment_extracts_failing_relations_only() -> None:
    raw = {
        "all_consistent": False,
        "relations": [
            {
                "description": "city <-> state",
                "entities": [],
                "passes": True,
                "reasoning": "ok",
            },
            {
                "description": "date_of_birth <-> age",
                "entities": [],
                "passes": False,
                "reasoning": "DOB 1990 vs age 12 is impossible.",
            },
        ],
    }
    valid, invalid = _flatten_judgment(raw)
    assert valid is False
    assert len(invalid) == 1
    assert invalid[0]["description"] == "date_of_birth <-> age"
    assert invalid[0]["passes"] is False


def test_flatten_judgment_accepts_pydantic_model() -> None:
    payload = RelationalConsistencyJudgmentSchema(all_consistent=True, relations=[])
    valid, invalid = _flatten_judgment(payload)
    assert valid is True
    assert invalid == []


def test_flatten_judgment_none_returns_unavailable_sentinel() -> None:
    assert _flatten_judgment(None) == (None, [])


def test_flatten_judgment_malformed_returns_unavailable_sentinel() -> None:
    assert _flatten_judgment("not json") == (None, [])
    assert _flatten_judgment(42) == (None, [])
    assert _flatten_judgment({"missing_top_level": True}) == (None, [])


# ---------------------------------------------------------------------------
# Tests: RelationalConsistencyJudgeWorkflow.evaluate
# ---------------------------------------------------------------------------


def _map_payload(items: list[dict]) -> dict:
    return {"replacements": items}


def test_evaluate_short_circuits_when_fewer_than_two_replacements(
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    df = pd.DataFrame(
        {
            COL_REPLACED_TEXT: ["Alice"],
            COL_REPLACEMENT_MAP: [_map_payload([{"original": "Alice", "label": "first_name", "synthetic": "Maya"}])],
        }
    )

    class _UnusedAdapter:
        def run_workflow(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("run_workflow should not be called when there are <2 replacements")

    wf = RelationalConsistencyJudgeWorkflow(adapter=_UnusedAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_replace_model_selection)
    assert result.failed_records == []
    assert bool(result.dataframe[COL_RELATIONAL_CONSISTENCY_VALID].iloc[0]) is True
    assert result.dataframe[COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS].iloc[0] == []


def test_evaluate_invokes_adapter_with_correct_alias_and_schema(
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    df = pd.DataFrame(
        {
            COL_REPLACED_TEXT: ["Maya works in Portland, OR"],
            COL_REPLACEMENT_MAP: [
                _map_payload(
                    [
                        {"original": "Alice", "label": "first_name", "synthetic": "Maya"},
                        {"original": "Austin", "label": "city", "synthetic": "Portland"},
                        {"original": "TX", "label": "state", "synthetic": "OR"},
                    ]
                )
            ],
        }
    )

    captured: dict = {}

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            captured["columns"] = columns
            captured["workflow_name"] = workflow_name
            out = frame.copy()
            out[COL_RELATIONAL_CONSISTENCY_JUDGE] = [
                {
                    "all_consistent": True,
                    "relations": [
                        {
                            "description": "city <-> state",
                            "entities": [
                                "Austin (city) -> Portland",
                                "TX (state) -> OR",
                            ],
                            "passes": True,
                            "reasoning": "Portland is in Oregon.",
                        }
                    ],
                }
            ]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = RelationalConsistencyJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_replace_model_selection)

    assert captured["workflow_name"] == "replace-relational-consistency-judge"
    col = captured["columns"][0]
    assert isinstance(col, LLMStructuredColumnConfig)
    assert col.name == COL_RELATIONAL_CONSISTENCY_JUDGE
    assert col.model_alias == stub_replace_model_selection.relational_consistency_judge
    assert col.output_format == RelationalConsistencyJudgmentSchema.model_json_schema()

    assert bool(result.dataframe[COL_RELATIONAL_CONSISTENCY_VALID].iloc[0]) is True
    assert result.dataframe[COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS].iloc[0] == []


def test_evaluate_marks_unavailable_for_malformed_payload(
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    df = pd.DataFrame(
        {
            COL_REPLACED_TEXT: ["Maya works in Portland, OR"],
            COL_REPLACEMENT_MAP: [
                _map_payload(
                    [
                        {"original": "Austin", "label": "city", "synthetic": "Portland"},
                        {"original": "TX", "label": "state", "synthetic": "OR"},
                    ]
                )
            ],
        }
    )

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            out = frame.copy()
            out[COL_RELATIONAL_CONSISTENCY_JUDGE] = ["not json"]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = RelationalConsistencyJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_replace_model_selection)

    assert result.dataframe[COL_RELATIONAL_CONSISTENCY_VALID].iloc[0] is None
    assert result.dataframe[COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS].iloc[0] == []


def test_evaluate_propagates_failing_relations(
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    df = pd.DataFrame(
        {
            COL_REPLACED_TEXT: ["..."],
            COL_REPLACEMENT_MAP: [
                _map_payload(
                    [
                        {"original": "1990", "label": "date_of_birth", "synthetic": "2015"},
                        {"original": "35", "label": "age", "synthetic": "35"},
                    ]
                )
            ],
        }
    )

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            out = frame.copy()
            out[COL_RELATIONAL_CONSISTENCY_JUDGE] = [
                {
                    "all_consistent": False,
                    "relations": [
                        {
                            "description": "date_of_birth <-> age",
                            "entities": [
                                "1990 (date_of_birth) -> 2015",
                                "35 (age) -> 35",
                            ],
                            "passes": False,
                            "reasoning": "A 2015 birthdate does not yield age 35.",
                        }
                    ],
                }
            ]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = RelationalConsistencyJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_replace_model_selection)
    assert bool(result.dataframe[COL_RELATIONAL_CONSISTENCY_VALID].iloc[0]) is False
    invalid = result.dataframe[COL_RELATIONAL_CONSISTENCY_INVALID_RELATIONS].iloc[0]
    assert len(invalid) == 1
    assert invalid[0]["description"] == "date_of_birth <-> age"
