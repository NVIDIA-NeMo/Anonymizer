# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opaque callback execution preserves compatibility without invented provenance."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, Substitute
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.replace.replace_runner import ReplacementResult
from tests.interface.test_anonymizer_interface import _make_anonymizer
from tests.retained import retained


def _input(tmp_path, rows=2):
    path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["same"] * rows}).to_csv(path, index=False)
    return AnonymizerInput(source=str(path))


@pytest.mark.parametrize(
    "strategy,preview",
    [(Redact(), False), (Redact(), True), (Hash(), False), (Annotate(), True), (Substitute(), False)],
)
def test_injected_redact_keeps_frame_and_calls(tmp_path, strategy, preview):
    detected = pd.DataFrame(
        {COL_TEXT: ["same", "same"], COL_FINAL_ENTITIES: [{"entities": []}] * 2}, index=pd.Index([7, 7])
    )
    cell = object()
    replaced = pd.DataFrame(
        {COL_TEXT: ["same", "same"], COL_REPLACED_TEXT: ["second", "first"], "extra": [cell, cell]},
        index=pd.Index([7, 7]),
    )
    anonymizer, detector, replacement, _ = _make_anonymizer(
        detection_return=EntityDetectionResult(detected, []),
        replace_return=ReplacementResult(replaced, []),
    )
    config = AnonymizerConfig(replace=strategy)
    result = (
        anonymizer.preview(config=config, data=_input(tmp_path), num_records=2)
        if preview
        else anonymizer.run(config=config, data=_input(tmp_path))
    )
    assert detector.run.call_count == replacement.run.call_count == 1
    assert not hasattr(detector, "_adapter")
    assert set(detector.run.call_args.kwargs) == {
        "model_configs",
        "selected_models",
        "gliner_detection_threshold",
        "validation_max_entities_per_call",
        "validation_excerpt_window_chars",
        "entity_labels",
        "excluded_entity_labels",
        "entity_label_examples",
        "privacy_goal",
        "data_summary",
        "tag_latent_entities",
        "compute_grouped_entities",
        "preview_num_records",
    }
    assert replacement.run.call_args.args[0] is detected
    assert replacement.run.call_args.kwargs["replace_method"] is config.replace
    assert result.trace_dataframe["text_replaced"].tolist() == ["second", "first"]
    assert result.trace_dataframe.index.tolist() == [7, 7]
    assert result.trace_dataframe["extra"].iloc[0] is cell


@pytest.mark.parametrize("stage", ["detection", "replacement"])
def test_injected_exception_identity_survives(tmp_path, stage):
    anonymizer, detector, replacement, _ = _make_anonymizer()
    error = ValueError("callback-owned failure")
    (detector if stage == "detection" else replacement).run.side_effect = error
    with pytest.raises(ValueError) as caught:
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_input(tmp_path, 1))
    assert caught.value is error
    assert detector.run.call_count == 1
    assert replacement.run.call_count == (stage == "replacement")


def test_cancellation_drains_detection_without_starting_replacement():
    import asyncio
    from threading import Event

    from anonymizer.engine.detection._opaque_redact import _DetectionInvocation, execute_injected_redact
    from anonymizer.engine.graph_sdk.blocking import BlockingPool

    entered, release = Event(), Event()
    anonymizer, detector, replacement, _ = _make_anonymizer()

    def detection(frame, **kwargs):
        entered.set()
        assert release.wait(5)
        return EntityDetectionResult(frame, [])

    detector.run.side_effect = detection

    async def run():
        pool = BlockingPool(max_workers=1)
        callback = _DetectionInvocation(
            pd.DataFrame({COL_TEXT: ["same"]}),
            detector,
            partial(replacement.run, replace_method=Redact()),
            anonymizer._model_configs,
            anonymizer._selected_models.detection,
            {},
        )
        task = asyncio.create_task(execute_injected_redact(callback, pool))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done()
            assert replacement.run.call_count == 0
            release.set()
            with pytest.raises(asyncio.CancelledError) as caught:
                await task
            result = retained(caught.value)
            assert not result.frames and not result.requests.dispatches
            assert [terminal.category for terminal in result.record.terminals] == ["cancelled"]
            assert replacement.run.call_count == 0
        finally:
            release.set()
            await pool.close()

    asyncio.run(run())


@pytest.mark.parametrize("parent,child,accepted", [(2, None, False), (None, 2, True)])
def test_unknown_request_bound_preserves_finite_parent_promises(parent, child, accepted):
    from anonymizer.graph import ContractViolation
    from tests.graph_sdk.test_workflow_static import _one_node_factory, _zero_operation
    from tests.graph_sdk.workflow_construction import admit_static_workflow, substitute

    def operation(bound):
        original = _zero_operation()
        outcome = original.outcomes[0]
        return replace(
            original, outcomes=(replace(outcome, ceiling=replace(outcome.ceiling, max_model_requests=bound)),)
        )

    target = _one_node_factory(lambda: operation(parent))
    target = admit_static_workflow(
        workflow=target.workflow,
        interface=target.interface,
        nodes=tuple(target.nodes),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(target.outcome_bindings),
        sequence=(),
        choices=(),
        protection=(),
        limits=replace(target.limits, max_nodes=2, max_subgraph_depth=2, max_bindings=2),
    )
    replacement = _one_node_factory(lambda: operation(child))
    if accepted:
        substitute(workflow=target, target=next(iter(target.operations)).id, replacement=replacement)
    else:
        with pytest.raises(ContractViolation):
            substitute(workflow=target, target=next(iter(target.operations)).id, replacement=replacement)


def test_empty_injected_frame_still_calls_both_callbacks(tmp_path):
    detected = pd.DataFrame({COL_TEXT: []})
    replaced = pd.DataFrame({COL_TEXT: [], COL_REPLACED_TEXT: []})
    anonymizer, detector, replacement, _ = _make_anonymizer(
        detection_return=EntityDetectionResult(detected, []), replace_return=ReplacementResult(replaced, [])
    )
    result = anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_input(tmp_path, 0))
    assert detector.run.call_count == replacement.run.call_count == 1
    assert result.dataframe.empty


def _execute_callback():
    import asyncio

    from anonymizer.engine.detection._opaque_redact import _DetectionInvocation, execute_injected_redact
    from anonymizer.engine.graph_sdk.blocking import BlockingPool

    anonymizer, detector, replacement, _ = _make_anonymizer()
    callback = _DetectionInvocation(
        pd.DataFrame({COL_TEXT: ["same"]}),
        detector,
        partial(replacement.run, replace_method=Redact()),
        anonymizer._model_configs,
        anonymizer._selected_models.detection,
        {},
    )

    async def run():
        pool = BlockingPool(max_workers=1)
        try:
            return await execute_injected_redact(callback, pool)
        finally:
            await pool.close()

    return lambda: asyncio.run(run()), detector, replacement


@pytest.mark.parametrize("field", ["max_remote_outstanding", "max_runtime_artifact_bytes"])
@pytest.mark.parametrize("ceiling", [0, 1])
def test_opaque_core_rejects_unsupported_selected_finite_ceiling(monkeypatch, field, ceiling):
    from anonymizer.engine.harness import kernel
    from anonymizer.graph import EffectRejected

    original = kernel.start_execution

    async def supplied(**kwargs):
        services = kwargs["services"]
        kwargs["services"] = replace(services, limits=replace(services.limits, **{field: ceiling}))
        return await original(**kwargs)

    monkeypatch.setattr(kernel, "start_execution", supplied)
    run, detector, replacement = _execute_callback()
    with pytest.raises(EffectRejected):
        run()
    assert detector.run.call_count == replacement.run.call_count == 0


@pytest.mark.parametrize("fault", [None, "missing_dataframe", "wrong_dataframe", "wrong_failures"])
def test_opaque_frame_is_validated_before_publication(monkeypatch, fault):
    from anonymizer.engine.detection._opaque_redact import _DetectionInvocation
    from anonymizer.engine.harness.kernel import project_detection_redact

    original = _DetectionInvocation.run

    def callback(self, inputs, requests):
        result = original(self, inputs, requests)
        from anonymizer.engine.graph_sdk.executor import LocalCompleted
        from anonymizer.engine.harness.frames import InvocationFrame

        assert isinstance(result, LocalCompleted) and isinstance(result.frame, InvocationFrame)
        frame = replace(result.frame)
        if fault == "missing_dataframe":
            object.__delattr__(frame, "dataframe")
        elif fault == "wrong_dataframe":
            object.__setattr__(frame, "dataframe", "not a frame")
        elif fault == "wrong_failures":
            object.__setattr__(frame, "detection_failures", ["not a FailedRecord"])
        return replace(result, frame=frame)

    monkeypatch.setattr(_DetectionInvocation, "run", callback)
    run, _, _ = _execute_callback()
    result = run()
    assert [terminal.category for terminal in result.record.terminals] == ["success" if fault is None else "failure"]
    assert len(result.frames) == (fault is None)
    if fault is None:
        assert isinstance(project_detection_redact(result).dataframe, pd.DataFrame)


def test_injected_callbacks_keep_caller_thread_context_and_running_loop(tmp_path):
    import asyncio
    import sqlite3
    from contextvars import ContextVar
    from threading import get_ident

    owner = get_ident()
    context: ContextVar[str | None] = ContextVar("caller_marker", default=None)
    connection = sqlite3.connect(":memory:")
    anonymizer, detector, replacement, _ = _make_anonymizer()
    calls = []
    for runner in (detector, replacement):
        returned = runner.run.return_value

        def callback(*args, _returned=returned, **kwargs):
            assert get_ident() == owner
            assert context.get() == "caller"
            assert asyncio.get_running_loop() is loop
            assert connection.execute("select 37").fetchone() == (37,)
            calls.append(_returned)
            context.set("caller")
            return _returned

        runner.run.side_effect = callback

    async def run():
        nonlocal loop
        loop = asyncio.get_running_loop()
        context.set("caller")
        return anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_input(tmp_path, 1))

    loop = None
    try:
        result = asyncio.run(run())
        assert len(calls) == 2 and result.dataframe is not None
    finally:
        connection.close()
