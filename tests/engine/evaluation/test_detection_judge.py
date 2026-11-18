# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from io import BytesIO

import pandas as pd
import pytest
from pydantic import ValidationError

from anonymizer.config.models import EvaluateModelSelection
from anonymizer.engine.constants import (
    COL_DETECTION_INVALID_ENTITIES,
    COL_DETECTION_JUDGE,
    COL_DETECTION_VALID,
    COL_ENTITIES_BY_VALUE,
    COL_REPLACEMENT_APPLICATION,
    COL_TEXT,
    RECORD_ID_COLUMN,
)
from anonymizer.engine.evaluation.detection_judge import (
    DetectionJudgeWorkflow,
    DetectionJudgmentSchema,
    _entities_for_judge,
    _judge_prompt,
    _label_examples_for_judge,
)
from anonymizer.engine.ndd.adapter import FailedRecord, WorkflowRunResult
from anonymizer.engine.schemas import EntitiesByValueSchema
from anonymizer.engine.workflow_columns.evaluation.judge.config import JudgeColumnConfig, JudgeKind

_flatten_judgment = DetectionJudgeWorkflow._flatten_judgment

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


@pytest.mark.parametrize(
    ("all_valid", "invalid_entities", "accepted"),
    [
        (True, None, True),
        (True, [], True),
        (
            True,
            [{"value": "morning", "label": "date_time", "reasoning": "common word"}],
            False,
        ),
        (False, None, False),
        (False, [], False),
        (
            False,
            [{"value": "morning", "label": "date_time", "reasoning": "common word"}],
            True,
        ),
    ],
)
def test_judgment_schema_normalizes_positive_null_and_enforces_verdict_consistency(
    all_valid: bool,
    invalid_entities: list[dict[str, str]] | None,
    accepted: bool,
) -> None:
    payload = {"all_valid": all_valid, "invalid_entities": invalid_entities}
    if not accepted:
        with pytest.raises(ValidationError):
            DetectionJudgmentSchema.model_validate(payload)
        return

    judgment = DetectionJudgmentSchema.model_validate(payload)
    assert judgment.model_dump()["invalid_entities"] == (invalid_entities or [])


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


class _ParquetCheckingJudgeAdapter:
    def __init__(self, returned_positions: tuple[int, ...]) -> None:
        self.returned_positions = returned_positions

    def run_workflow(self, frame: pd.DataFrame, **kwargs: object) -> WorkflowRunResult:
        buffer = BytesIO()
        frame.to_parquet(buffer)
        assert COL_REPLACEMENT_APPLICATION not in frame.columns
        buffer.seek(0)
        roundtripped = pd.read_parquet(buffer)
        out = roundtripped.iloc[list(self.returned_positions)].copy().reset_index(drop=True)
        out[COL_DETECTION_JUDGE] = [{"all_valid": True, "invalid_entities": []} for _ in range(len(out))]
        dropped = frame[~frame[RECORD_ID_COLUMN].isin(out[RECORD_ID_COLUMN])]
        failed_records = [
            FailedRecord(record_id=record_id, step="replace-detection-judge", reason="Synthetic judge failure")
            for record_id in dropped[RECORD_ID_COLUMN]
        ]
        return WorkflowRunResult(dataframe=out, failed_records=failed_records)


@pytest.mark.parametrize(
    "returned_positions",
    [(0, 1, 2), (2, 1, 0), (2, 0)],
    ids=["all-rows", "reordered", "dropped-and-reordered"],
)
@pytest.mark.parametrize("telemetry_case", ["all-empty", "mixed", "absent"])
def test_evaluate_serializes_seed_and_preserves_telemetry(
    stub_evaluate_model_selection: EvaluateModelSelection,
    returned_positions: tuple[int, ...],
    telemetry_case: str,
) -> None:
    names = [["Alice"], [], ["Bob", "Carol"], ["Dan", "Eve", "Frank"]]
    df = pd.DataFrame(
        {
            RECORD_ID_COLUMN: [f"rec-{position}" for position in range(len(names))],
            COL_TEXT: ["Alice", "no sensitive content", "Bob met Carol", "Dan met Eve and Frank"],
            COL_ENTITIES_BY_VALUE: [
                _entities_payload([{"value": name, "labels": ["first_name"]} for name in row_names])
                for row_names in names
            ],
        },
        index=pd.Index([10, 20, 30, 40]),
    )
    if telemetry_case != "absent":
        skipped_counts = [0, 0, 0, 1] if telemetry_case == "mixed" else [0, 0, 0, 0]
        df[COL_REPLACEMENT_APPLICATION] = [
            {
                "targeted_span_count": len(row_names),
                "applied_span_count": len(row_names) - skipped_count,
                "skipped_span_count": skipped_count,
                "skipped_span_label_counts": {"first_name": skipped_count} if skipped_count else {},
            }
            for row_names, skipped_count in zip(names, skipped_counts, strict=True)
        ]
    original = df.copy(deep=True)
    workflow = DetectionJudgeWorkflow(adapter=_ParquetCheckingJudgeAdapter(returned_positions))

    result = workflow.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)

    entity_positions = (0, 2, 3)
    expected_positions = sorted([1, *(entity_positions[position] for position in returned_positions)])
    assert result.dataframe[RECORD_ID_COLUMN].tolist() == df.iloc[expected_positions][RECORD_ID_COLUMN].tolist()
    assert result.dataframe[COL_DETECTION_VALID].tolist() == [True] * len(expected_positions)
    assert result.dataframe[COL_DETECTION_INVALID_ENTITIES].tolist() == [[] for _ in expected_positions]
    if telemetry_case != "absent":
        telemetry_rows = result.dataframe[COL_REPLACEMENT_APPLICATION].tolist()
        assert telemetry_rows == df.iloc[expected_positions][COL_REPLACEMENT_APPLICATION].tolist()
        for telemetry in telemetry_rows:
            assert type(telemetry) is dict
            for count_name in ("targeted_span_count", "applied_span_count", "skipped_span_count"):
                assert type(telemetry[count_name]) is int
            label_counts = telemetry["skipped_span_label_counts"]
            assert type(label_counts) is dict
            assert all(type(count) is int for count in label_counts.values())
    else:
        assert COL_REPLACEMENT_APPLICATION not in result.dataframe.columns
    expected_dropped_ids = [
        df.iloc[original_position][RECORD_ID_COLUMN]
        for position, original_position in enumerate(entity_positions)
        if position not in returned_positions
    ]
    assert [failure.record_id for failure in result.failed_records] == expected_dropped_ids
    pd.testing.assert_frame_equal(df, original)


def test_evaluate_short_circuits_when_no_entities(
    stub_evaluate_model_selection: EvaluateModelSelection,
    caplog,
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
    with caplog.at_level(logging.INFO, logger="anonymizer.evaluation.judge_base"):
        result = wf.evaluate(df, model_configs=[], selected_models=stub_evaluate_model_selection)
    assert result.failed_records == []
    assert bool(result.dataframe[COL_DETECTION_VALID].iloc[0]) is True
    assert result.dataframe[COL_DETECTION_INVALID_ENTITIES].iloc[0] == []
    assert "detection_valid set to True (trivially valid)" in caplog.text


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
    assert isinstance(col, JudgeColumnConfig)
    assert col.name == COL_DETECTION_JUDGE
    assert col.model_alias == stub_evaluate_model_selection.detection_validity_judge
    assert col.judge_kind == JudgeKind.DETECTION.value

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
