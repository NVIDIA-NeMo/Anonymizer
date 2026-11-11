# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from unittest.mock import Mock

import pandas as pd
import pytest
from data_designer.config.models import ModelConfig

from anonymizer.config.models import ReplaceModelSelection
from anonymizer.config.replace_strategies import Substitute
from anonymizer.engine.constants import (
    COL_ENTITIES_BY_VALUE,
    COL_ENTITIES_FOR_REPLACE,
    COL_FINAL_ENTITIES,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_TAGGED_TEXT,
    COL_TEXT,
)
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.row_partitioning import ROW_ORDER_COL

_ALICE = {"original": "Alice", "label": "first_name", "synthetic": "Maya"}
_BOB = {"original": "Bob", "label": "first_name", "synthetic": "Noah"}


def _input_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            RECORD_ID_COLUMN: ["partial", "complete", "empty"],
            COL_TEXT: ["Alice sees Bob.", "Bob sees Bob.", "No entities."],
            COL_TAGGED_TEXT: ["Alice sees Bob.", "Bob sees Bob.", "No entities."],
            COL_FINAL_ENTITIES: [
                {
                    "entities": [
                        {"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5},
                        {"value": "Bob", "label": "first_name", "start_position": 11, "end_position": 14},
                    ]
                },
                {
                    "entities": [
                        {"value": "Bob", "label": "first_name", "start_position": 0, "end_position": 3},
                        {"value": "Bob", "label": "first_name", "start_position": 9, "end_position": 12},
                    ]
                },
                {"entities": []},
            ],
            COL_ENTITIES_BY_VALUE: [
                {
                    "entities_by_value": [
                        {"value": "Alice", "labels": ["first_name"]},
                        {"value": "Bob", "labels": ["first_name"]},
                    ]
                },
                {"entities_by_value": [{"value": "Bob", "labels": ["first_name"]}]},
                {"entities_by_value": []},
            ],
        }
    )


@pytest.mark.parametrize(
    "extra_entry",
    [
        pytest.param(None, id="omitted"),
        pytest.param(dict(_ALICE), id="duplicate-entry"),
        pytest.param({**_BOB, "original": "Unrequested"}, id="unrequested-original"),
        pytest.param({**_BOB, "original": "bob"}, id="original-case"),
        pytest.param({**_BOB, "label": "name"}, id="wrong-label"),
    ],
)
def test_incomplete_maps_retry_missing_pairs_and_preserve_accepted_mappings(
    extra_entry: dict[str, str] | None,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[pd.DataFrame] = []

    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        calls.append(frame.copy(deep=True))
        output = frame.copy()
        if len(calls) == 1:
            output[COL_REPLACEMENT_MAP] = [
                {"replacements": [_ALICE, *([extra_entry] if extra_entry is not None else [])]}
                if record_id == "partial"
                else {"replacements": [_BOB]}
                for record_id in frame[RECORD_ID_COLUMN]
            ]
        else:
            output[COL_REPLACEMENT_MAP] = [
                {"replacements": [_BOB, {**_ALICE, "synthetic": "Changed"}]} for _ in range(len(output))
            ]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    source = _input_frame()
    original = source.copy(deep=True)
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=adapter)).run(
            source,
            replace_method=Substitute(),
            model_configs=stub_model_configs,
            selected_models=stub_replace_model_selection,
        )

    pd.testing.assert_frame_equal(source, original)
    assert result.failed_records == []
    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["partial", "complete", "empty"]
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya sees Noah.", "Noah sees Noah.", "No entities."]
    assert len(calls) == 2
    assert calls[0][RECORD_ID_COLUMN].tolist() == ["partial", "complete"]
    assert calls[1][RECORD_ID_COLUMN].tolist() == ["partial"]
    assert calls[1][COL_ENTITIES_FOR_REPLACE].tolist() == [
        [{"value": "Bob", "labels": ["first_name"], "labels_str": "first_name"}]
    ]
    for value in ["Alice", "Bob", "Maya", "Noah", "Changed", "Unrequested"]:
        assert value not in caplog.text


@pytest.mark.parametrize("drop_retry", [False, True], ids=["incomplete-retry", "dropped-retry"])
def test_second_unsuccessful_attempt_fails_only_the_incomplete_record(
    drop_retry: bool,
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    calls: list[pd.DataFrame] = []

    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        calls.append(frame.copy())
        if len(calls) == 2 and drop_retry:
            return WorkflowRunResult(
                dataframe=frame.iloc[:0],
                failed_records=[
                    FailedRecord(record_id="partial", step="replace-map-generation", reason="Model failed")
                ],
            )
        output = frame.copy()
        output[COL_REPLACEMENT_MAP] = [
            {"replacements": []} if record_id == "partial" else {"replacements": [_BOB]}
            for record_id in frame[RECORD_ID_COLUMN]
        ]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    with caplog.at_level(logging.WARNING, logger="anonymizer"):
        result = ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=adapter)).run(
            _input_frame(),
            replace_method=Substitute(),
            model_configs=stub_model_configs,
            selected_models=stub_replace_model_selection,
        )

    assert len(calls) == 2
    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["complete", "empty"]
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Noah sees Noah.", "No entities."]
    assert [failure.record_id for failure in result.failed_records] == ["partial"]
    for value in ["Alice", "Bob", "Maya", "Noah"]:
        assert value not in result.failed_records[0].reason
        assert value not in caplog.text


def test_unicode_space_mutation_does_not_count_as_the_requested_original(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    source = _input_frame().iloc[:1].copy()
    source[COL_ENTITIES_BY_VALUE] = [{"entities_by_value": [{"value": "Bob\u202fJones", "labels": ["full_name"]}]}]
    calls: list[pd.DataFrame] = []

    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        calls.append(frame.copy())
        output = frame.copy()
        output[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {
                        "original": "Bob\u2009Jones" if len(calls) == 1 else "Bob\u202fJones",
                        "label": "full_name",
                        "synthetic": "Noah Smith",
                    }
                ]
            }
        ]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    result = LlmReplaceWorkflow(adapter=adapter).generate_validated_maps(
        source,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    assert len(calls) == 2
    assert result.failed_records == []
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0]["replacements"][0]["original"] == "Bob\u202fJones"


def test_completeness_compares_against_the_original_pre_call_inventory(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    calls: list[pd.DataFrame] = []

    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        calls.append(frame.copy())
        output = frame.copy()
        output[COL_ENTITIES_BY_VALUE] = [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}]
        output[COL_REPLACEMENT_MAP] = [{"replacements": [_ALICE] if len(calls) == 1 else [_BOB]}]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    source = _input_frame().iloc[:1]
    result = LlmReplaceWorkflow(adapter=adapter).generate_validated_maps(
        source,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert len(calls) == 2
    assert calls[1][COL_ENTITIES_FOR_REPLACE].iloc[0] == [
        {"value": "Bob", "labels": ["first_name"], "labels_str": "first_name"}
    ]
    assert result.failed_records == []
    assert result.dataframe[COL_ENTITIES_BY_VALUE].iloc[0] == source[COL_ENTITIES_BY_VALUE].iloc[0]
    assert result.dataframe[COL_REPLACEMENT_MAP].iloc[0]["replacements"] == [_ALICE, _BOB]


def test_missing_label_for_a_shared_value_still_requires_its_own_mapping(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    source = _input_frame().iloc[:1].copy()
    source[COL_ENTITIES_BY_VALUE] = [
        {"entities_by_value": [{"value": "Jordan", "labels": ["first_name", "last_name"]}]}
    ]
    calls: list[pd.DataFrame] = []

    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        calls.append(frame.copy())
        output = frame.copy()
        output[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {
                        "original": "Jordan",
                        "label": "first_name" if len(calls) == 1 else "last_name",
                        "synthetic": "Taylor",
                    }
                ]
            }
        ]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    result = LlmReplaceWorkflow(adapter=adapter).generate_validated_maps(
        source,
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )

    assert len(calls) == 2
    assert calls[1][COL_ENTITIES_FOR_REPLACE].iloc[0] == [
        {"value": "Jordan", "labels": ["last_name"], "labels_str": "last_name"}
    ]
    assert result.failed_records == []
    assert {entry["label"] for entry in result.dataframe[COL_REPLACEMENT_MAP].iloc[0]["replacements"]} == {
        "first_name",
        "last_name",
    }


def test_complete_maps_and_no_entity_records_do_not_retry(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        output = frame.copy()
        output[COL_REPLACEMENT_MAP] = [{"replacements": [_BOB]} for _ in range(len(output))]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    workflow = LlmReplaceWorkflow(adapter=adapter)
    result = workflow.generate_validated_maps(
        _input_frame().iloc[1:],
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    assert result.failed_records == []
    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["complete", "empty"]
    adapter.run_workflow.assert_called_once()

    adapter.reset_mock()
    for source in [_input_frame().iloc[2:], _input_frame().iloc[:0]]:
        result = workflow.generate_validated_maps(
            source,
            model_configs=stub_model_configs,
            selected_models=stub_replace_model_selection,
        )
        assert len(result.dataframe) == len(source)
        assert result.failed_records == []
    adapter.run_workflow.assert_not_called()


def test_retry_keeps_record_positions_when_ids_repeat_and_output_is_reordered(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    source = _input_frame().iloc[[0, 0]].copy()
    source[RECORD_ID_COLUMN] = ["duplicate", "duplicate"]
    calls: list[pd.DataFrame] = []

    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        calls.append(frame.copy())
        output = frame.iloc[::-1].copy()
        output[COL_REPLACEMENT_MAP] = [
            {
                "replacements": [
                    {
                        "original": "Alice" if len(calls) == 1 else "Bob",
                        "label": "first_name",
                        "synthetic": ["Maya", "Lena"][position] if len(calls) == 1 else ["Noah", "Liam"][position],
                    }
                ]
            }
            for position in output[ROW_ORDER_COL]
        ]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    result = ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter=adapter)).run(
        source,
        replace_method=Substitute(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
    )
    assert len(calls) == 2
    assert result.failed_records == []
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya sees Noah.", "Lena sees Liam."]


def test_preview_retries_only_records_in_the_requested_entity_sample(
    stub_model_configs: list[ModelConfig],
    stub_replace_model_selection: ReplaceModelSelection,
) -> None:
    calls: list[pd.DataFrame] = []

    def generate(frame: pd.DataFrame, **_: object) -> WorkflowRunResult:
        calls.append(frame.copy())
        output = frame.iloc[:1].copy()
        output[COL_REPLACEMENT_MAP] = [{"replacements": [_ALICE] if len(calls) == 1 else [_BOB]}]
        return WorkflowRunResult(dataframe=output, failed_records=[])

    adapter = Mock()
    adapter.run_workflow.side_effect = generate
    result = LlmReplaceWorkflow(adapter=adapter).generate_validated_maps(
        _input_frame(),
        model_configs=stub_model_configs,
        selected_models=stub_replace_model_selection,
        preview_num_records=1,
    )
    assert len(calls) == 2
    assert all(frame[RECORD_ID_COLUMN].tolist() == ["partial"] for frame in calls)
    assert result.dataframe[RECORD_ID_COLUMN].tolist() == ["partial", "empty"]
    assert result.failed_records == []
