# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pandas as pd
from data_designer.config.column_configs import LLMStructuredColumnConfig

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_TEXT,
)
from anonymizer.engine.evaluation.detection_judge import (
    DetectionJudgeWorkflow,
    DetectionJudgmentSchema,
    _entities_for_judge,
    _flatten_judgment,
    _judge_prompt,
    _label_examples_for_judge,
)
from anonymizer.engine.schemas import EntitiesByValueSchema

# ---------------------------------------------------------------------------
# Tests: _judge_prompt
# ---------------------------------------------------------------------------


def test_judge_prompt_uses_xml_sections() -> None:
    prompt = _judge_prompt()
    for tag in ("original_text", "detected_entities", "task", "invalid_criteria", "valid_criteria"):
        assert f"<{tag}>" in prompt
        assert f"</{tag}>" in prompt


def test_judge_prompt_references_original_text_column() -> None:
    prompt = _judge_prompt()
    assert COL_TEXT in prompt


def test_judge_prompt_iterates_detected_entities() -> None:
    prompt = _judge_prompt()
    assert "for entity in _entities_for_detection_judge" in prompt
    assert "entity.value" in prompt
    assert "entity.label" in prompt


# ---------------------------------------------------------------------------
# Tests: helpers
# ---------------------------------------------------------------------------


def test_entities_for_judge_flattens_labels() -> None:
    parsed = EntitiesByValueSchema.from_raw(
        {
            "entities_by_value": [
                {"value": "Alice", "labels": ["first_name"]},
                {"value": "Acme", "labels": ["company_name", "organization_name"]},
            ]
        }
    )
    rows = _entities_for_judge(parsed)
    assert rows == [
        {"value": "Alice", "label": "first_name"},
        {"value": "Acme", "label": "company_name"},
        {"value": "Acme", "label": "organization_name"},
    ]


def test_label_examples_for_judge_returns_json_keyed_by_label() -> None:
    parsed = EntitiesByValueSchema.from_raw({"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]})
    examples_json = _label_examples_for_judge(parsed)
    assert "first_name" in examples_json
    assert examples_json.startswith("{")


def test_label_examples_for_judge_empty_when_no_entities() -> None:
    parsed = EntitiesByValueSchema()
    assert _label_examples_for_judge(parsed) == "{}"


# ---------------------------------------------------------------------------
# Tests: _flatten_judgment
# ---------------------------------------------------------------------------


def test_flatten_judgment_all_valid_path() -> None:
    valid, invalid = _flatten_judgment({"all_valid": True, "invalid_entities": []})
    assert valid is True
    assert invalid == []


def test_flatten_judgment_returns_invalid_entries() -> None:
    raw = {
        "all_valid": False,
        "invalid_entities": [
            {"value": "morning", "label": "date_time", "reasoning": "common word"},
        ],
    }
    valid, invalid = _flatten_judgment(raw)
    assert valid is False
    assert invalid == [{"value": "morning", "label": "date_time", "reasoning": "common word"}]


def test_flatten_judgment_accepts_pydantic_model() -> None:
    payload = DetectionJudgmentSchema(all_valid=True, invalid_entities=[])
    valid, invalid = _flatten_judgment(payload)
    assert valid is True
    assert invalid == []


def test_flatten_judgment_accepts_json_string() -> None:
    valid, invalid = _flatten_judgment('{"all_valid": true, "invalid_entities": []}')
    assert valid is True
    assert invalid == []


def test_flatten_judgment_none_returns_unavailable_sentinel() -> None:
    assert _flatten_judgment(None) == (None, [])


def test_flatten_judgment_malformed_returns_unavailable_sentinel() -> None:
    assert _flatten_judgment("not json") == (None, [])
    assert _flatten_judgment(42) == (None, [])
    assert _flatten_judgment({"missing": "all_valid"}) == (None, [])


# ---------------------------------------------------------------------------
# Tests: DetectionJudgeWorkflow.evaluate
# ---------------------------------------------------------------------------


def _entities_payload(entities: list[dict]) -> dict:
    return {"entities_by_value": entities}


def test_evaluate_short_circuits_when_no_entities(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    """Rows with no detected entities skip the LLM call and pass trivially."""
    df = pd.DataFrame(
        {
            COL_TEXT: ["plain text"],
            COL_ENTITIES_BY_VALUE: [_entities_payload([])],
        }
    )

    class _UnusedAdapter:
        def run_workflow(self, *args, **kwargs):  # pragma: no cover - should not be called
            raise AssertionError("run_workflow should not be called for empty-entity rows")

    wf = DetectionJudgeWorkflow(adapter=_UnusedAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)
    assert result.failed_records == []
    assert bool(result.dataframe[COL_DETECTION_VALID].iloc[0]) is True
    assert result.dataframe[COL_DETECTION_INVALID_ENTITIES].iloc[0] == []


def test_evaluate_invokes_adapter_for_rows_with_entities(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    """Rows with entities get a structured-column workflow keyed on detection_judge."""
    df = pd.DataFrame(
        {
            COL_TEXT: ["Alice works at Acme"],
            COL_ENTITIES_BY_VALUE: [
                _entities_payload(
                    [
                        {"value": "Alice", "labels": ["first_name"]},
                        {"value": "Acme", "labels": ["company_name"]},
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
            out[COL_DETECTION_JUDGE] = [
                {
                    "all_valid": False,
                    "invalid_entities": [{"value": "Acme", "label": "company_name", "reasoning": "spurious"}],
                }
            ]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = DetectionJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)

    assert captured["workflow_name"] == "replace-detection-judge"
    assert len(captured["columns"]) == 1
    col = captured["columns"][0]
    assert isinstance(col, LLMStructuredColumnConfig)
    assert col.name == COL_DETECTION_JUDGE
    assert col.model_alias == stub_evaluate_model_selection.detection_validity_judge
    assert col.output_format == DetectionJudgmentSchema.model_json_schema()

    assert bool(result.dataframe[COL_DETECTION_VALID].iloc[0]) is False
    invalid = result.dataframe[COL_DETECTION_INVALID_ENTITIES].iloc[0]
    assert invalid == [{"value": "Acme", "label": "company_name", "reasoning": "spurious"}]


def test_evaluate_merges_entity_and_empty_rows_in_order(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    """Rows are returned in their original order, even when one bypasses the LLM."""
    df = pd.DataFrame(
        {
            COL_TEXT: ["no entities here", "Alice was here"],
            COL_ENTITIES_BY_VALUE: [
                _entities_payload([]),
                _entities_payload([{"value": "Alice", "labels": ["first_name"]}]),
            ],
        }
    )

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            out = frame.copy()
            out[COL_DETECTION_JUDGE] = [{"all_valid": True, "invalid_entities": []}]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = DetectionJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)

    assert list(result.dataframe[COL_TEXT]) == ["no entities here", "Alice was here"]
    assert [bool(v) for v in result.dataframe[COL_DETECTION_VALID]] == [True, True]


def test_evaluate_marks_judge_unavailable_for_malformed_payload(
    stub_evaluate_model_selection: EvaluateModelSelection,
) -> None:
    """Malformed judge output leaves detection_valid=None rather than fabricating a verdict."""
    df = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_ENTITIES_BY_VALUE: [_entities_payload([{"value": "Alice", "labels": ["first_name"]}])],
        }
    )

    class _StubAdapter:
        def run_workflow(self, frame, *, model_configs, columns, workflow_name, preview_num_records=None):
            out = frame.copy()
            out[COL_DETECTION_JUDGE] = ["not json"]

            class _Result:
                dataframe = out
                failed_records: list = []

            return _Result()

    wf = DetectionJudgeWorkflow(adapter=_StubAdapter())
    result = wf.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)

    assert result.dataframe[COL_DETECTION_VALID].iloc[0] is None
    assert result.dataframe[COL_DETECTION_INVALID_ENTITIES].iloc[0] == []
