# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both existing Rewrite schedules share one nonqualifying invocation owner."""

from __future__ import annotations

import asyncio
import pickle
import sqlite3
from contextvars import ContextVar
from threading import get_ident

import pandas as pd
import pytest

from anonymizer import AnonymizerConfig, AnonymizerInput, Rewrite
from anonymizer.engine.constants import COL_REWRITTEN_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.graph_sdk.executor import ExecutionResult
from anonymizer.engine.harness.frames import InvocationFrame
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult
from tests.engine import test_detection_source_frame as detection_fixtures
from tests.engine.test_detection_source_frame import public_anonymizer
from tests.interface.test_anonymizer_interface import _make_anonymizer
from tests.interface.test_injected_redact_graph import _input
from tests.retained import retained

scripted_detection = detection_fixtures.scripted_detection


@pytest.mark.parametrize("combined,preview", [(False, False), (False, True), (True, False), (True, True)])
def test_rewrite_invocation_preserves_callbacks_options_and_public_owner(tmp_path, combined, preview):
    anonymizer, detector, replacement, runner = _make_anonymizer()
    anonymizer._combined_rewrite_runner = runner
    thread = get_ident()
    context = ContextVar("rewrite_context", default="missing")
    context.set("caller")
    connection = sqlite3.connect(":memory:")
    cell = {"opaque": [1]}
    detected = pd.DataFrame({COL_TEXT: ["same", "same"]}, index=pd.Index([7, 7]))
    output = pd.DataFrame({COL_TEXT: ["same"], COL_REWRITTEN_TEXT: ["second"], "extra": [cell]}, index=pd.Index([7]))
    first = FailedRecord("dropped-detection", "detection", "literal first")
    second = FailedRecord("dropped-rewrite", "rewrite", "literal second")
    strategy = Rewrite(use_combined_graph=combined, max_repair_iterations=2)
    config = AnonymizerConfig(rewrite=strategy)

    def check_thread():
        assert get_ident() == thread and context.get() == "caller"
        assert connection.execute("select 23").fetchone() == (23,)

    def detect(frame, **kwargs):
        check_thread()
        assert len(frame) == 2
        assert kwargs["tag_latent_entities"] is True and kwargs["compute_grouped_entities"] is True
        assert kwargs["privacy_goal"] is strategy.privacy_goal
        return EntityDetectionResult(detected, [first])

    def rewrite(frame, **kwargs):
        check_thread()
        assert frame is detected
        assert kwargs == dict(
            model_configs=anonymizer._model_configs,
            selected_models=anonymizer._selected_models.rewrite,
            replace_model_selection=anonymizer._selected_models.replace,
            privacy_goal=strategy.privacy_goal,
            evaluation=strategy.evaluation,
            data_summary=None,
            preview_num_records=2 if preview else None,
            strict_entity_protection=strategy.strict_entity_protection,
        )
        return RewriteResult(output, [second])

    detector.run.side_effect, runner.run.side_effect = detect, rewrite

    async def active_loop():
        data = _input(tmp_path)
        return (
            anonymizer.preview(config=config, data=data, num_records=2)
            if preview
            else anonymizer.run(config=config, data=data)
        )

    try:
        result = asyncio.run(active_loop())
    finally:
        connection.close()
    assert detector.run.call_count == runner.run.call_count == 1
    assert replacement.run.call_count == 0
    assert result.failed_records == [first, second]
    assert result.trace_dataframe.index.tolist() == [7]
    assert result.trace_dataframe.extra.iloc[0] is cell
    assert result.dataframe.text_rewritten.tolist() == ["second"]
    execution = result._execution
    assert isinstance(execution, ExecutionResult)
    assert execution._execution.context.prepared.declared_request_upper_bound is None
    assert not execution.requests.dispatches and not execution.final_outputs
    assert isinstance(execution.frames[0], InvocationFrame)
    assert execution.frames[0].row_counts == (2, 1)
    restored = pickle.loads(pickle.dumps(result))
    assert restored._execution is None
    pd.testing.assert_frame_equal(restored.trace_dataframe, result.trace_dataframe)


@pytest.mark.parametrize("stage", ["detection", "rewrite"])
def test_rewrite_callback_exception_identity_retains_execution(tmp_path, stage):
    anonymizer, detector, _, runner = _make_anonymizer()
    error = ValueError("literal injected rewrite failure")
    (detector if stage == "detection" else runner).run.side_effect = error
    with pytest.raises(ValueError) as caught:
        anonymizer.run(config=AnonymizerConfig(rewrite=Rewrite()), data=_input(tmp_path))
    assert caught.value is error
    execution = retained(error)
    assert isinstance(execution, ExecutionResult)
    assert not execution.frames and not execution.requests.dispatches
    assert detector.run.call_count == 1
    assert runner.run.call_count == (stage == "rewrite")


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    "texts,preview,error",
    [
        ([], False, "Failed to load generated dataset"),
        (["Alice travels."], True, "Dataset is empty"),
        (["Alice fails."], False, "scripted row failure"),
    ],
)
def test_builtin_rewrite_zero_source_and_all_failed_execution(
    scripted_detection, tmp_path, combined, texts, preview, error
):
    from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection
    from anonymizer.interface.errors import AnonymizerWorkflowError

    facade = public_anonymizer(scripted_detection)
    facade._selected_models.replace = ReplaceModelSelection(replacement_generator="singleton")
    facade._selected_models.rewrite = RewriteModelSelection(
        **{key: "singleton" for key in RewriteModelSelection.model_fields}
    )
    path = tmp_path / "rewrite.csv"
    pd.DataFrame({"text": pd.Series(texts, dtype=str)}).to_csv(path, index=False)
    config = AnonymizerConfig(rewrite=Rewrite(use_combined_graph=combined), emit_telemetry=False)
    with pytest.raises(AnonymizerWorkflowError, match=error) as caught:
        options = dict(config=config, data=AnonymizerInput(source=str(path)))
        facade.preview(**options, num_records=0) if preview else facade.run(**options)
    execution = retained(caught.value)
    assert not execution.frames and not execution.final_outputs
    assert not execution.requests.dispatches
    assert len(execution.record.targets) == (0 if preview or not texts else 1)


@pytest.mark.parametrize("combined", [False, True])
def test_exact_supplied_callback_adapter_preserves_caller_thread_and_single_invocation(
    tmp_path, monkeypatch, scripted_detection, combined
):
    from unittest.mock import Mock

    from anonymizer.engine.constants import (
        COL_ANY_HIGH_LEAKED,
        COL_LEAKAGE_MASS,
        COL_NEEDS_HUMAN_REVIEW,
        COL_NEEDS_REPAIR,
        COL_REPAIR_ITERATIONS,
        COL_UTILITY_SCORE,
    )
    from anonymizer.engine.ndd.adapter import WorkflowRunResult
    from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
    from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow
    from anonymizer.graph import InvocationId

    caller = get_ident()
    invocation_ids = []
    new = InvocationId.new

    def allocate(*args, **kwargs):
        value = new(*args, **kwargs)
        invocation_ids.append(value)
        return value

    adapter = Mock()

    def response(frame, **kwargs):
        assert get_ident() == caller
        result = frame.copy()
        result[COL_REWRITTEN_TEXT] = "observed safe"
        result[COL_UTILITY_SCORE] = 0.9
        result[COL_LEAKAGE_MASS] = 0.0
        result[COL_ANY_HIGH_LEAKED] = False
        result[COL_NEEDS_REPAIR] = False
        result[COL_REPAIR_ITERATIONS] = 0
        result[COL_NEEDS_HUMAN_REVIEW] = False
        return WorkflowRunResult(result, [], tuple(range(len(result))))

    adapter.run_workflow.side_effect = response
    runner = (CombinedRewriteWorkflow if combined else RewriteWorkflow)(adapter)
    facade = public_anonymizer(
        scripted_detection, **{"combined_rewrite_runner" if combined else "rewrite_runner": runner}
    )
    from anonymizer.config.models import ReplaceModelSelection, RewriteModelSelection

    facade._selected_models.replace = ReplaceModelSelection(replacement_generator="singleton")
    facade._selected_models.rewrite = RewriteModelSelection(
        **{key: "singleton" for key in RewriteModelSelection.model_fields}
    )
    from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE
    from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow

    def detect(frame, **kwargs):
        assert get_ident() == caller
        detected = frame.copy()
        detected[COL_ENTITIES_BY_VALUE] = [{"entities_by_value": [{"value": "Alice", "labels": ["first_name"]}]}] * len(
            frame
        )
        return EntityDetectionResult(detected, [], tuple(range(len(frame))))

    monkeypatch.setattr(facade._detection_workflow, "run", detect)
    monkeypatch.setattr(
        LlmReplaceWorkflow,
        "generate_map_only",
        lambda self, frame, **kwargs: WorkflowRunResult(frame.copy(), [], tuple(range(len(frame)))),
    )
    monkeypatch.setattr(InvocationId, "new", allocate)
    path = tmp_path / "supplied.csv"
    pd.DataFrame({"text": ["Alice visits."]}).to_csv(path, index=False)
    result = facade.run(
        config=AnonymizerConfig(
            rewrite=Rewrite(use_combined_graph=combined, max_repair_iterations=0), emit_telemetry=False
        ),
        data=AnonymizerInput(source=str(path)),
    )
    assert result.dataframe.text_rewritten.tolist() == ["observed safe"]
    assert len(invocation_ids) == 1
    assert not result._execution.assessments


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize("order", ["before_delegate", "after_delegate"])
def test_caller_detector_mutating_privacy_goal_runs_on_the_admitted_goal(
    tmp_path, monkeypatch, scripted_detection, combined, order
):
    """C1: a caller detector that mutates rewrite.privacy_goal and delegates to builtin detection succeeds;
    Rewrite, result.rewrite_config and evaluate() all use the goal admitted with the run."""
    import json
    from unittest.mock import Mock

    import httpx

    from anonymizer.config.models import EvaluateModelSelection, ReplaceModelSelection, RewriteModelSelection
    from anonymizer.engine.constants import (
        COL_ANY_HIGH_LEAKED,
        COL_LEAKAGE_MASS,
        COL_NEEDS_HUMAN_REVIEW,
        COL_NEEDS_REPAIR,
        COL_REPAIR_ITERATIONS,
        COL_UTILITY_SCORE,
    )
    from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
    from anonymizer.engine.ndd.adapter import WorkflowRunResult
    from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
    from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow
    from anonymizer.engine.rewrite.rewrite_workflow import RewriteWorkflow

    marker = "MUTATED-BY-CALLER-DETECTOR"
    prompts = []
    adapter = Mock()

    def response(frame, **kwargs):
        prompts.append(json.dumps([column.model_dump(mode="json") for column in kwargs["columns"]]))
        result = frame.copy()
        result[COL_REWRITTEN_TEXT] = "observed safe"
        result[COL_UTILITY_SCORE] = 0.9
        result[COL_LEAKAGE_MASS] = 0.0
        result[COL_ANY_HIGH_LEAKED] = False
        result[COL_NEEDS_REPAIR] = False
        result[COL_REPAIR_ITERATIONS] = 0
        result[COL_NEEDS_HUMAN_REVIEW] = False
        return WorkflowRunResult(result, [], tuple(range(len(result))))

    adapter.run_workflow.side_effect = response
    runner = (CombinedRewriteWorkflow if combined else RewriteWorkflow)(adapter)

    class MutatingDetector(EntityDetectionWorkflow):
        def run(self, dataframe, **kwargs):  # type: ignore[override]
            if order == "before_delegate":
                kwargs["privacy_goal"].preserve = marker
            detected = super().run(dataframe, **kwargs)
            if order == "after_delegate":
                kwargs["privacy_goal"].preserve = marker
            return detected

    base = public_anonymizer(scripted_detection)
    facade = public_anonymizer(
        scripted_detection,
        detection_workflow=MutatingDetector(base._adapter),
        **{"combined_rewrite_runner" if combined else "rewrite_runner": runner},
    )
    facade._selected_models.replace = ReplaceModelSelection(replacement_generator="singleton")
    facade._selected_models.rewrite = RewriteModelSelection(
        **{key: "singleton" for key in RewriteModelSelection.model_fields}
    )
    monkeypatch.setattr(
        LlmReplaceWorkflow,
        "generate_map_only",
        lambda self, frame, **kwargs: WorkflowRunResult(frame.copy(), [], tuple(range(len(frame)))),
    )
    from tests.engine.test_detection_graph_publication import _modify_dd_responses

    def latent(request, answer):
        if "LatentEntitiesSchema" not in request.content.decode():
            return answer
        payload = answer.json()
        payload["choices"][0]["message"]["content"] = '```json\n{"latent_entities": []}\n```'
        return httpx.Response(200, json=payload)

    _modify_dd_responses(scripted_detection, monkeypatch, latent)
    strategy = Rewrite(use_combined_graph=combined, max_repair_iterations=0)
    assert strategy.privacy_goal is not None
    admitted = strategy.privacy_goal.model_copy(deep=True)
    path = tmp_path / "goal.csv"
    pd.DataFrame({"text": ["Alice travels."]}).to_csv(path, index=False)
    result = facade.run(
        config=AnonymizerConfig(rewrite=strategy, emit_telemetry=False), data=AnonymizerInput(source=str(path))
    )
    assert strategy.privacy_goal.preserve == marker  # the caller's own object changed
    assert result.rewrite_config == admitted and result.rewrite_config.preserve != marker
    assert all(marker not in prompt for prompt in prompts)

    judged = []

    class Stop(Exception):
        pass

    def evaluate(*args, **kwargs):
        judged.append(kwargs["privacy_goal"])
        raise Stop

    monkeypatch.setattr(facade._rewrite_runner, "evaluate", evaluate)
    facade._selected_models.evaluate = EvaluateModelSelection(
        **{key: "singleton" for key in EvaluateModelSelection.model_fields}
    )
    with pytest.raises(Stop):
        facade.evaluate(result)
    assert judged == [admitted]
