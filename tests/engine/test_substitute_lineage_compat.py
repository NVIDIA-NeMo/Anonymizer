# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Durable observable contracts from the Substitute compatibility lineage."""

from __future__ import annotations

import json
from threading import current_thread, main_thread
from typing import Any, cast
from unittest.mock import Mock

import httpx
import numpy as np
import pandas as pd
import pytest

from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection, RewriteModelSelection
from anonymizer.config.replace_strategies import Substitute
from anonymizer.config.rewrite import EvaluationCriteria, PrivacyGoal
from anonymizer.engine.constants import (
    COL_ANY_HIGH_LEAKED,
    COL_ENTITIES_BY_VALUE,
    COL_FINAL_ENTITIES,
    COL_LATENT_ENTITIES,
    COL_LEAKAGE_MASS,
    COL_NEEDS_HUMAN_REVIEW,
    COL_NEEDS_REPAIR,
    COL_PRIVACY_QA,
    COL_QUALITY_QA,
    COL_REPAIR_ITERATIONS,
    COL_REPLACED_TEXT,
    COL_REPLACEMENT_MAP,
    COL_REWRITTEN_TEXT,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_UTILITY_SCORE,
)
from anonymizer.engine.detection._graph_profile import run_detection_redact
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult, EntityDetectionWorkflow
from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow
from anonymizer.engine.ndd.adapter import RECORD_ID_COLUMN, FailedRecord, NddAdapter, WorkflowRunResult
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult, LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.engine.rewrite import rewrite_workflow
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
from anonymizer.engine.row_partitioning import ROW_ORDER_COL
from anonymizer.graph import InvocationId
from anonymizer.interface.errors import AnonymizerWorkflowError
from tests.engine import test_detection_source_frame as detection_tests
from tests.engine.test_detection_source_frame import _modify_dd_responses

pytestmark = pytest.mark.compat
scripted_detection = detection_tests.scripted_detection


def _entities() -> dict[str, Any]:
    return {"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}


def _map() -> dict[str, Any]:
    return {"replacements": [{"original": "Alice", "label": "first_name", "synthetic": "Maya"}]}


def _rewrite_arguments() -> dict[str, Any]:
    return {
        "model_configs": [],
        "selected_models": RewriteModelSelection(**{key: "test" for key in RewriteModelSelection.model_fields}),
        "replace_model_selection": ReplaceModelSelection(replacement_generator="test"),
        "privacy_goal": PrivacyGoal(protect="Protect all names", preserve="Preserve original meaning"),
        "evaluation": EvaluationCriteria(max_repair_iterations=0),
    }


class _RewriteAdapter:
    def __init__(self, candidate: Any = None, human_review: bool = False) -> None:
        self.candidate = candidate
        self.human_review = human_review
        self.calls: list[tuple[str, list[Any], Any]] = []

    def run_workflow(self, frame: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        initial = kwargs["workflow_name"] in {"rewrite-pipeline", "rewrite-combined"}
        texts = frame[COL_TEXT] if initial else frame[COL_REWRITTEN_TEXT]
        self.calls.append((kwargs["workflow_name"], texts.tolist(), current_thread()))
        output = frame.copy()
        output[COL_REWRITTEN_TEXT] = (
            ["safe " + text if initial else text for text in texts] if self.candidate is None else self.candidate
        )
        for column, value in (
            (COL_UTILITY_SCORE, 1.0),
            (COL_LEAKAGE_MASS, 0.0),
            (COL_ANY_HIGH_LEAKED, False),
            (COL_NEEDS_REPAIR, False),
            (COL_REPAIR_ITERATIONS, 0),
            (COL_NEEDS_HUMAN_REVIEW, self.human_review),
        ):
            output[column] = value
        output[COL_QUALITY_QA] = [{"questions": []} for _ in range(len(frame))]
        output[COL_PRIVACY_QA] = [{"questions": []} for _ in range(len(frame))]
        return WorkflowRunResult(output, [], tuple(range(len(output))))


@pytest.mark.parametrize("mode", ["empty", "mixed", "entities"])
def test_map_partition_preserves_nullable_string_and_category_dtypes(mode: str) -> None:
    source = pd.DataFrame(
        {
            COL_TEXT: ["Alice", "Alice"],
            COL_ENTITIES_BY_VALUE: [_entities(), _entities()],
            "nullable_int": pd.array([1, pd.NA], dtype="Int64"),
            "nullable_text": pd.array(["hello", pd.NA], dtype="string"),
            "category": pd.Categorical(["a", "b"]),
        },
        index=pd.Index([7, 7]),
    )
    if mode in {"empty", "mixed"}:
        source[COL_ENTITIES_BY_VALUE] = [
            {"entities_by_value": []},
            {"entities_by_value": []} if mode == "empty" else _entities(),
        ]
    adapter = Mock(spec=NddAdapter)

    def respond(frame: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        assert len(frame) == (1 if mode == "mixed" else 2)
        assert all(value["entities_by_value"] for value in frame[COL_ENTITIES_BY_VALUE])
        output = frame.copy()
        output[COL_REPLACEMENT_MAP] = [_map() for _ in range(len(frame))]
        return WorkflowRunResult(output, [], tuple(range(len(frame))))

    adapter.run_workflow.side_effect = respond
    result = LlmReplaceWorkflow(adapter).generate_map_only(
        source, model_configs=[], selected_models=ReplaceModelSelection(replacement_generator="test")
    )
    pd.testing.assert_frame_equal(result.dataframe[list(source.columns)], source.reset_index(drop=True))
    assert result.source_positions == tuple(range(len(source)))
    assert result.failed_records == []
    assert result.dataframe[COL_REPLACEMENT_MAP].tolist() == (
        [{"replacements": []}] * 2
        if mode == "empty"
        else [{"replacements": []}, _map()]
        if mode == "mixed"
        else [_map(), _map()]
    )
    assert adapter.run_workflow.call_count == (1 if mode in {"mixed", "entities"} else 0)


def test_no_entity_rewrite_never_constructs_or_calls_supplied_map_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    source = pd.DataFrame(
        {COL_TEXT: ["plain one", "plain two"], COL_ENTITIES_BY_VALUE: [{"entities_by_value": []}] * 2},
        index=pd.Index([7, 7]),
    )
    factory = Mock(side_effect=AssertionError("no-entity Rewrite must bypass map factory"))
    adapter = Mock(spec=NddAdapter)
    adapter.run_workflow.side_effect = AssertionError("no-entity Rewrite must bypass adapter")
    monkeypatch.setattr(rewrite_workflow, "LlmReplaceWorkflow", factory)
    result = RewriteWorkflow(adapter).run(source, **_rewrite_arguments())
    factory.assert_not_called()
    factory.return_value.generate_map_only.assert_not_called()
    adapter.run_workflow.assert_not_called()
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == ["plain one", "plain two"]
    assert result.dataframe[COL_UTILITY_SCORE].tolist() == [1.0, 1.0]
    assert result.dataframe[COL_LEAKAGE_MASS].tolist() == [0.0, 0.0]
    assert result.source_positions == (0, 1) and not result.failed_records


@pytest.mark.parametrize("positions", ["explicit", "legacy_unique", "legacy_duplicate"])
def test_rewrite_opaque_map_receives_original_entity_rows_and_preserves_drop_reorder(
    monkeypatch: pytest.MonkeyPatch, positions: str
) -> None:
    source = pd.DataFrame(
        {
            COL_TEXT: ["Alice first", "plain empty", "Alice second", "Alice third"],
            COL_ENTITIES_BY_VALUE: [_entities(), {"entities_by_value": []}, _entities(), _entities()],
            RECORD_ID_COLUMN: ["same"] * 4 if positions == "legacy_duplicate" else ["r0", "r1", "r2", "r3"],
            "marker": ["first", "empty", "second", "third"],
        },
        index=pd.Index([7, 7, 8, 8]),
    )
    arguments = _rewrite_arguments()
    calls: list[pd.DataFrame] = []
    failure = FailedRecord("r2", "map-callback", "synthetic drop")

    class SuppliedMap(LlmReplaceWorkflow):
        def generate_map_only(self, dataframe: pd.DataFrame, **kwargs: Any) -> LlmReplaceResult:
            assert current_thread() is main_thread()
            assert set(kwargs) == {"model_configs", "selected_models"}
            assert kwargs["model_configs"] == arguments["model_configs"]
            assert kwargs["selected_models"] == arguments["replace_model_selection"]
            assert dataframe[COL_TEXT].tolist() == ["Alice first", "Alice second", "Alice third"]
            assert dataframe.index.tolist() == [7, 8, 8]
            assert dataframe[ROW_ORDER_COL].tolist() == [0, 2, 3]
            calls.append(dataframe)
            output = dataframe.iloc[[2, 0]].copy()
            output[COL_REPLACEMENT_MAP] = [_map(), _map()]
            return LlmReplaceResult(output, [failure], (2, 0) if positions == "explicit" else None)

    monkeypatch.setattr(rewrite_workflow, "LlmReplaceWorkflow", SuppliedMap)
    adapter = _RewriteAdapter()
    runner = RewriteWorkflow(cast(NddAdapter, adapter))
    if positions == "legacy_duplicate":
        with pytest.raises(AnonymizerWorkflowError, match="ambiguous source correspondence"):
            runner.run(source, **arguments)
        assert len(calls) == 1 and not adapter.calls
        return
    result = runner.run(source, **arguments)
    assert len(calls) == 1
    assert result.dataframe[COL_REWRITTEN_TEXT].tolist() == ["safe Alice first", "plain empty", "safe Alice third"]
    assert result.dataframe.marker.tolist() == ["first", "empty", "third"]
    assert result.source_positions == (0, 1, 3) and result.failed_records == [failure]
    assert [(name, texts) for name, texts, _ in adapter.calls] == [
        ("rewrite-pipeline", ["Alice third", "Alice first"]),
        ("rewrite-evaluate-0", ["safe Alice third", "safe Alice first"]),
    ]
    assert all(thread is main_thread() for _, _, thread in adapter.calls)


def test_custom_detector_runs_builtin_substitute_as_its_own_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    source = pd.DataFrame({COL_TEXT: ["Alice"]})
    owners: list[Any] = []
    allocated: list[InvocationId] = []
    original = InvocationId.new
    models: list[Any] = []
    selected = DetectionModelSelection(**{key: "test" for key in DetectionModelSelection.model_fields})

    def allocate(*args: Any, **kwargs: Any) -> InvocationId:
        invocation = original(*args, **kwargs)
        allocated.append(invocation)
        return invocation

    class CustomDetect:
        def run(self, dataframe: pd.DataFrame, **kwargs: Any) -> EntityDetectionResult:
            assert dataframe is source and current_thread() is main_thread()
            assert set(kwargs) == {"model_configs", "selected_models"}
            assert kwargs["model_configs"] is models and kwargs["selected_models"] is selected
            owners.append(source_worker_borrow().owner)
            output = dataframe.copy()
            output[COL_ENTITIES_BY_VALUE] = [_entities()]
            output[COL_FINAL_ENTITIES] = [
                {"entities": [{"value": "Alice", "label": "first_name", "start_position": 0, "end_position": 5}]}
            ]
            return EntityDetectionResult(output, [], (0,))

    adapter = Mock(spec=NddAdapter)

    def respond(frame: pd.DataFrame, **kwargs: Any) -> WorkflowRunResult:
        assert current_thread() is main_thread()
        owners.append(source_worker_borrow().owner)
        output = frame.copy()
        output[COL_REPLACEMENT_MAP] = [_map()]
        return WorkflowRunResult(output, [], (0,))

    monkeypatch.setattr(InvocationId, "new", allocate)
    adapter.run_workflow.side_effect = respond
    result = run_detection_redact(
        source,
        detection=cast(EntityDetectionWorkflow, CustomDetect()),
        model_configs=models,
        selected_models=selected,
        detection_options={},
        replace_method=Substitute(),
        replacement=ReplacementWorkflow(llm_workflow=LlmReplaceWorkflow(adapter)),
        replacement_models=ReplaceModelSelection(replacement_generator="test"),
    )
    assert result.dataframe[COL_REPLACED_TEXT].tolist() == ["Maya"]
    # The caller's detector runs opaque; the builtin Substitute it hands off to runs standalone, as in the release.
    assert len(allocated) == 2 and len(owners) == 2 and owners[0] is not owners[1]
    assert result.dataframe[COL_TEXT].tolist() == ["Alice"] and not result.failed_records


def test_map_accepts_normalized_nested_numpy_source(scripted_detection: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    designer, models, _, calls = scripted_detection
    map_prompts: list[str] = []

    def respond(request: httpx.Request, answer: httpx.Response) -> httpx.Response:
        messages = json.dumps(json.loads(request.content)["messages"])
        if "Generate synthetic replacements" not in messages:
            return answer
        map_prompts.append(messages)
        payload = answer.json()
        payload["choices"][0]["message"]["content"] = json.dumps(_map())
        return httpx.Response(200, json=payload)

    _modify_dd_responses(scripted_detection, monkeypatch, respond)
    nested = {
        "entities_by_value": np.array(
            [{"value": np.str_("Alice"), "labels": np.array(["first_name"], dtype=object)}], dtype=object
        )
    }
    source = pd.DataFrame(
        {
            COL_TEXT: ["Alice"],
            COL_TAGGED_TEXT: ["<first_name>Alice</first_name>"],
            COL_ENTITIES_BY_VALUE: [nested],
            COL_LATENT_ENTITIES: [{"latent_entities": np.array([], dtype=object)}],
            "number": pd.array([pd.NA], dtype="Int64"),
        },
        index=pd.Index([7]),
    )
    workflow = LlmReplaceWorkflow(NddAdapter(designer))
    selected = ReplaceModelSelection(replacement_generator="singleton")
    result = workflow.generate_map_only(source, model_configs=models, selected_models=selected)
    assert result.dataframe[COL_REPLACEMENT_MAP].tolist() == [_map()]
    assert result.source_positions == (0,) and not result.failed_records
    assert result.dataframe.number.isna().tolist() == [True]
    assert result.dataframe.number.dtype == object
    assert result.dataframe.number.iloc[0] is pd.NA
    assert len(map_prompts) == 1 and "Alice" in map_prompts[0]
    assert nested["entities_by_value"][0]["value"] == "Alice"
    assert nested["entities_by_value"][0]["labels"].tolist() == ["first_name"]
    assert result._execution is not None and len(result._execution.frames) == 1
    # DataDesigner's own transport makes the map request: the graph accounts it as one opaque invocation.
    assert not result._execution.requests.dispatches
    assert not result._execution.requests.remote_outstanding and not result._execution.requests.local_in_flight


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("case", ["integer", "nan_review_true", "nan_review_false"])
def test_rewrite_scalar_schedule_preserves_numeric_value_and_review_flag(
    monkeypatch: pytest.MonkeyPatch, combined: bool, case: str
) -> None:
    candidate = 42 if case == "integer" else float("nan")
    review = case == "nan_review_true"
    adapter = _RewriteAdapter(candidate=candidate, human_review=review)
    helper = Mock()
    helper.return_value.generate_map_only.side_effect = lambda frame, **kwargs: LlmReplaceResult(
        frame.copy(), [], tuple(range(len(frame)))
    )
    monkeypatch.setattr(rewrite_workflow, "LlmReplaceWorkflow", helper)
    opaque = object()
    source = pd.DataFrame({COL_TEXT: ["Alice"], COL_ENTITIES_BY_VALUE: [_entities()], "extra_opaque": [opaque]})
    runner = (CombinedRewriteWorkflow if combined else RewriteWorkflow)(cast(NddAdapter, adapter))
    result = runner.run(source, **_rewrite_arguments())
    actual = result.dataframe[COL_REWRITTEN_TEXT].iloc[0]
    if case == "integer":
        assert actual == 42
        assert isinstance(actual, (int, np.integer))
    else:
        assert pd.isna(actual)
    expected_review = case != "integer" and (review or not combined)
    assert bool(result.dataframe[COL_NEEDS_HUMAN_REVIEW].iloc[0]) is expected_review
    assert result.dataframe.extra_opaque.iloc[0] is opaque
    assert len(adapter.calls) == (1 if combined else 2)
