# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opaque callback execution preserves compatibility without invented provenance."""

from __future__ import annotations

import pickle
from dataclasses import replace

import pandas as pd
import pytest

from anonymizer.config.anonymizer_config import AnonymizerConfig, AnonymizerInput
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_FINAL_ENTITIES, COL_REPLACED_TEXT, COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionResult
from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection, PreparationRejected
from anonymizer.engine.graph_sdk.executor import ExecutionResult
from anonymizer.engine.graph_sdk.inspection import InspectionLimits, bind_inspection_source, explain, inspect
from anonymizer.engine.graph_sdk.preparation import prepare
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.replace.replace_runner import ReplacementResult
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.interface.test_anonymizer_interface import _make_anonymizer


def _input(tmp_path, rows=2):
    path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["same"] * rows}).to_csv(path, index=False)
    return AnonymizerInput(source=str(path))


@pytest.mark.parametrize("preview", [False, True])
def test_injected_redact_keeps_frame_calls_and_reports_unknown_work(tmp_path, preview):
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
    config = AnonymizerConfig(replace=Redact())
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
    execution = result._execution
    assert isinstance(execution, ExecutionResult)
    admitted = execution._execution
    prepared = admitted.context.prepared
    assert prepared.declared_request_upper_bound is None
    assert not execution.requests.dispatches and not execution.final_outputs
    assert execution.invocation_state is not None
    assert [entry.status for entry in execution.invocation_state.entries] == ["success"]
    from anonymizer.engine.detection._frame_publication import InvocationFrame

    frame = execution.frames[0]
    assert isinstance(frame, InvocationFrame)
    assert frame.row_counts == (2, 2)
    assert frame.text_bytes == 27
    assert len(execution.record.terminals) == 1
    assert execution.cleanup[0].disposition == "left_open"
    source = bind_inspection_source(
        prepared=prepared,
        context=admitted.context,
        execution_plan=admitted,
        execution=execution,
        limits=InspectionLimits(
            max_nodes=100,
            max_depth=20,
            max_pairs=100,
            max_events=100,
            max_reason_counts=20,
            max_encoded_bytes=65536,
            max_count=100,
        ),
    )
    forecast = explain(source=source)
    assert forecast.requests.upper_bound.knowledge == "unknown"
    assert forecast.hard_budget_support == "unsupported"
    observed = inspect(source=source)
    assert observed.requests.total.knowledge == "unknown"
    assert observed.usage.input_units.knowledge == "unknown"
    qualification, current, submissions = _inputs(admitted, execution)
    assert not qualify(admitted=qualification, result=execution, current=current, submissions=submissions).qualified
    from anonymizer.engine.graph_sdk._effect_values import EffectRejected
    from tests.graph_sdk.child_qualification_fixtures import result_with

    forged = result_with(execution, requests=replace(execution.requests, dispatched_count=1))
    with pytest.raises(EffectRejected):
        forged_admission, forged_current, forged_submissions = _inputs(admitted, forged)
        qualify(admitted=forged_admission, result=forged, current=forged_current, submissions=forged_submissions)
    restored = pickle.loads(pickle.dumps(result))
    assert restored._execution is None
    assert detector.run.call_count == replacement.run.call_count == 1
    with pytest.raises(PreparationRejected, match="hard_budget_incompatible"):
        prepare(
            data=prepared.data,
            workflow=prepared.workflow,
            activation_limits=prepared.activation_limits,
            bound_inputs=(),
            configuration=replace(prepared.configuration, hard_request_limit=1),
            state=prepared.state,
            selections=tuple(
                # The real consumer's selected implementation is reused unchanged.
                ImplementationSelection(
                    node=item.node,
                    implementation=item.capability.implementation,
                    configuration=item.capability.configuration,
                )
                for item in prepared.implementations
            ),
            capabilities=admitted.capabilities,
            limits=prepared.limits,
        )
    assert detector.run.call_count == replacement.run.call_count == 1


@pytest.mark.parametrize("stage", ["detection", "replacement"])
def test_injected_exception_identity_and_owner_survive(tmp_path, stage):
    anonymizer, detector, replacement, _ = _make_anonymizer()
    error = ValueError("callback-owned failure")
    (detector if stage == "detection" else replacement).run.side_effect = error
    with pytest.raises(ValueError) as caught:
        anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_input(tmp_path, 1))
    assert caught.value is error
    execution = getattr(error, "_anonymizer_execution")
    assert execution._execution.context.prepared.declared_request_upper_bound is None
    assert not execution.requests.dispatches and not execution.frames
    assert detector.run.call_count == 1
    assert replacement.run.call_count == (stage == "replacement")


def test_cancellation_drains_detection_without_starting_replacement():
    import asyncio
    from threading import Event

    from anonymizer.engine.detection._opaque_redact import _InjectedRedact, execute_injected_redact
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
        callback = _InjectedRedact(
            pd.DataFrame({COL_TEXT: ["same"]}),
            detector,
            replacement,
            anonymizer._model_configs,
            anonymizer._selected_models.detection,
            anonymizer._selected_models.replace,
            {},
            Redact(),
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
            result = getattr(caught.value, "_anonymizer_execution")
            assert not result.frames and not result.requests.dispatches
            assert all(terminal.category != "success" for terminal in result.record.terminals)
            assert replacement.run.call_count == 0
        finally:
            release.set()
            await pool.close()

    asyncio.run(run())


@pytest.mark.parametrize("parent,child,accepted", [(2, None, False), (None, 2, True)])
def test_unknown_request_bound_preserves_finite_parent_promises(parent, child, accepted):
    from anonymizer.graph._values import ContractViolation
    from anonymizer.graph.workflow import admit_static_workflow, substitute
    from tests.graph_sdk.test_workflow_static import _one_node_factory, _zero_operation

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
        substitute(workflow=target, target=next(iter(target.nodes)).id, replacement=replacement)
    else:
        with pytest.raises(ContractViolation):
            substitute(workflow=target, target=next(iter(target.nodes)).id, replacement=replacement)


def test_empty_injected_frame_still_calls_both_callbacks(tmp_path):
    detected = pd.DataFrame({COL_TEXT: []})
    replaced = pd.DataFrame({COL_TEXT: [], COL_REPLACED_TEXT: []})
    anonymizer, detector, replacement, _ = _make_anonymizer(
        detection_return=EntityDetectionResult(detected, []), replace_return=ReplacementResult(replaced, [])
    )
    result = anonymizer.run(config=AnonymizerConfig(replace=Redact()), data=_input(tmp_path, 0))
    assert detector.run.call_count == replacement.run.call_count == 1
    assert result.dataframe.empty
    assert result._execution is not None
