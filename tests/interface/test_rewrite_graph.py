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
from anonymizer.engine.detection._frame_publication import InvocationFrame
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.graph_sdk.executor import ExecutionResult
from anonymizer.engine.ndd.adapter import FailedRecord
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult
from tests.engine import test_detection_source_frame as detection_fixtures
from tests.engine.test_detection_source_frame import public_anonymizer
from tests.interface.test_anonymizer_interface import _make_anonymizer
from tests.interface.test_injected_redact_graph import _input

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
    execution = getattr(error, "_anonymizer_execution")
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
    execution = getattr(caught.value, "_anonymizer_execution")
    assert not execution.frames and not execution.final_outputs
    assert not execution.requests.dispatches
    assert len(execution.record.targets) == (0 if preview or not texts else 1)
