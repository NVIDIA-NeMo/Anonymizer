# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal registered local outputs and canonical zero-request ownership."""

from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest

from anonymizer.engine.constants import (
    COL_DETECTED_ENTITIES,
    COL_INITIAL_TAGGED_TEXT,
    COL_MERGED_ENTITIES,
    COL_MERGED_TAGGED_TEXT,
    COL_RAW_DETECTED,
    COL_SEED_ENTITIES,
    COL_SEED_ENTITIES_JSON,
    COL_SEED_TAGGED_TEXT,
    COL_SEED_VALIDATION_CANDIDATES,
    COL_TAG_NOTATION,
    COL_TAGGED_TEXT,
    COL_TEXT,
    COL_VALIDATED_ENTITIES,
    COL_VALIDATED_SEED_ENTITIES,
    COL_VALIDATION_CANDIDATES,
)
from anonymizer.engine.workflow_columns.detection.config import DetectionTransformConfig, DetectionTransformOperation
from anonymizer.engine.workflow_columns.detection.impl import DetectionTransformGenerator


def _generator(operation):
    return DetectionTransformGenerator(
        DetectionTransformConfig(name="result", operation=operation),
        resource_provider=SimpleNamespace(),  # ty: ignore[invalid-argument-type]
    )


@pytest.fixture
def executions(monkeypatch):
    from anonymizer.engine.detection import _graph_profile

    observed = []
    original = _graph_profile._await_profile_execution

    async def capture(running):
        result = await original(running)
        observed.append(result)
        return result

    monkeypatch.setattr(_graph_profile, "_await_profile_execution", capture)
    return observed


# Expected native payloads are literal; no production transform computes the oracle.
@pytest.mark.parametrize(
    "operation,expected",
    [
        ("parse_detected_entities", {COL_SEED_ENTITIES: {"entities": []}, COL_TAG_NOTATION: "xml"}),
        ("prepare_validation_inputs", {COL_SEED_TAGGED_TEXT: "", COL_SEED_VALIDATION_CANDIDATES: {"candidates": []}}),
        ("enrich_validation_decisions", {COL_VALIDATED_ENTITIES: {"decisions": []}}),
        (
            "apply_validation_to_seed_entities",
            {COL_VALIDATED_SEED_ENTITIES: {"entities": []}, COL_SEED_ENTITIES_JSON: "[]", COL_INITIAL_TAGGED_TEXT: ""},
        ),
        (
            "merge_and_build_candidates",
            {
                COL_MERGED_ENTITIES: {"entities": []},
                COL_MERGED_TAGGED_TEXT: "",
                COL_VALIDATION_CANDIDATES: {"candidates": []},
            },
        ),
        ("apply_validation_and_finalize", {COL_DETECTED_ENTITIES: {"entities": []}, COL_TAGGED_TEXT: ""}),
    ],
)
@pytest.mark.parametrize("asynchronous", [False, True])
def test_literal_empty_payload_and_canonical_local_closure(executions, operation, expected, asynchronous):
    row = {"opaque": object()}
    generator = _generator(operation)
    output = asyncio.run(generator.agenerate(row)) if asynchronous else generator.generate(row)
    assert output == {"opaque": row["opaque"], **expected}
    assert (output is row) is (not asynchronous)  # Native base async contract copies the input row.
    (result,) = executions
    assert result.frames[0].project(result).at[0, "result"] is output
    assert not result.requests.dispatches and not result.final_outputs and not result.artifacts
    assert not result.record.targets and len(result.record.terminals) == 1
    assert {entry.status for entry in result.invocation_state.entries} == {"success"}
    assert result._execution.context.prepared.declared_request_upper_bound == 0


def test_native_parse_entity_literal(executions):
    row = {
        COL_TEXT: "Alice",
        COL_RAW_DETECTED: '{"entities":[{"text":"Alice","label":"person","start":0,"end":5,"score":0.9}]}',
    }
    result = _generator("parse_detected_entities").generate(row)
    assert result[COL_SEED_ENTITIES] == {
        "entities": [
            {
                "id": "person_0_5",
                "value": "Alice",
                "label": "person",
                "start_position": 0,
                "end_position": 5,
                "score": 0.9,
                "source": "detector",
            }
        ]
    }
    assert len(executions) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
def test_original_exception_identity_and_canonical_failure(monkeypatch, executions, asynchronous):
    from anonymizer.engine.workflow_columns.detection import impl

    failure = ValueError("literal transform failure")

    def fail(row):
        raise failure

    monkeypatch.setitem(impl._TRANSFORMS, DetectionTransformOperation.PARSE_DETECTED_ENTITIES, fail)
    with pytest.raises(ValueError) as caught:
        generator = _generator("parse_detected_entities")
        asyncio.run(generator.agenerate({})) if asynchronous else generator.generate({})
    assert caught.value is failure
    (result,) = executions
    assert getattr(failure, "_anonymizer_execution") is result
    assert result.submission_failures[0].cause is failure
    assert not result.frames and not result.requests.dispatches
    assert {entry.status for entry in result.invocation_state.entries} == {"failure"}


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "wrong",
        "results",
        "assessments",
        "failed_sources",
        "empty_results",
        "empty_assessments",
        "empty_failed_sources",
    ],
)
def test_malformed_local_frame_is_not_published(monkeypatch, executions, fault):
    from anonymizer.engine.graph_sdk.executor import LocalCompleted
    from anonymizer.engine.workflow_columns.detection import _local_transform
    from anonymizer.interface.errors import AnonymizerWorkflowError

    original = _local_transform._Transform.run

    async def malformed(self, request):
        result = await original(self, request)
        assert isinstance(result, LocalCompleted)
        if fault == "missing":
            return LocalCompleted(results=())
        if fault == "wrong":
            object.__setattr__(result.frame, "dataframe", [])
        elif fault.startswith("empty_"):
            object.__setattr__(result, fault.removeprefix("empty_"), [])
        else:
            object.__setattr__(result, fault, [object()])
        return result

    monkeypatch.setattr(_local_transform._Transform, "run", malformed)
    with pytest.raises(AnonymizerWorkflowError):
        _generator("parse_detected_entities").generate({})
    (result,) = executions
    assert not result.frames and not result.requests.dispatches
    assert {entry.status for entry in result.invocation_state.entries} == {"failure"}


def test_repeated_async_observer_cancel_retains_local_worker(monkeypatch, executions):
    from anonymizer.engine.workflow_columns.detection import impl

    entered, release = Event(), Event()

    def held(row):
        entered.set()
        assert release.wait(5)
        return row

    monkeypatch.setitem(impl._TRANSFORMS, DetectionTransformOperation.PARSE_DETECTED_ENTITIES, held)

    async def run():
        task = asyncio.create_task(_generator("parse_detected_entities").agenerate({}))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0.01)
                assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    (result,) = executions
    assert len(result.record.terminals) == 1 and not result.requests.dispatches


def test_managed_missing_source_context_cannot_start_fresh_invocation(monkeypatch):
    from anonymizer.graph._values import InvocationId
    from anonymizer.interface.errors import AnonymizerWorkflowError

    generator = _generator("parse_detected_entities")
    generator._anonymizer_source_frame = None
    monkeypatch.setattr(InvocationId, "new", lambda **kwargs: pytest.fail("unexpected new invocation"))
    with pytest.raises(AnonymizerWorkflowError):
        generator.generate({})


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_self_cancellation_closes_and_preserves_error(monkeypatch, executions, asynchronous):
    from anonymizer.engine.workflow_columns.detection import impl

    cancellation = asyncio.CancelledError("literal local cancellation")

    def cancel(row):
        raise cancellation

    monkeypatch.setitem(impl._TRANSFORMS, DetectionTransformOperation.PARSE_DETECTED_ENTITIES, cancel)
    with pytest.raises(asyncio.CancelledError) as caught:
        generator = _generator("parse_detected_entities")
        asyncio.run(generator.agenerate({})) if asynchronous else generator.generate({})
    assert caught.value is cancellation
    result = getattr(caught.value, "_anonymizer_execution")
    assert not result.frames and not result.requests.dispatches
    assert {entry.status for entry in result.invocation_state.entries} == {"cancelled"}


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fault", [None, "foreign", "expired", "missing_owner"])
def test_managed_transform_reuses_authenticated_owner(monkeypatch, asynchronous, fault):
    from anonymizer.engine.graph_sdk._effect_values import EffectRejected
    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW
    from anonymizer.engine.graph_sdk.executor import LocalFailure
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation
    from anonymizer.engine.ndd._source_frame import _SOURCE_WORK, SourceWork
    from anonymizer.graph._values import InvocationId, TaskAttemptId
    from tests.graph_sdk.test_delegated_execution import _execute, _fixture
    from tests.graph_sdk.test_delegated_worker import _loop_and_pool, _worker_services

    created = []
    original = InvocationId.new

    def count(**kwargs):
        result = original(**kwargs)
        created.append(result)
        return result

    monkeypatch.setattr(InvocationId, "new", count)

    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, *_ = _fixture()
            generator = _generator("parse_detected_entities")

            def callback(inputs, borrowed):
                association = inputs[0].association
                if fault == "foreign":
                    association = SemanticAssociation(task=TaskAttemptId.new(activation=association.task.activation))
                token = _SOURCE_WORK.set(SourceWork(association, "result"))
                owner_token = _WORKER_BORROW.set(None) if fault == "missing_owner" else None
                borrowed.expired = fault == "expired"
                try:

                    def generate():
                        return (
                            asyncio.run_coroutine_threadsafe(generator.agenerate({}), loop).result()
                            if asynchronous
                            else generator.generate({})
                        )

                    if fault:
                        with pytest.raises(EffectRejected):
                            generate()
                    else:
                        assert generate() == {COL_SEED_ENTITIES: {"entities": []}, COL_TAG_NOTATION: "xml"}
                finally:
                    borrowed.expired = False
                    if owner_token is not None:
                        _WORKER_BORROW.reset(owner_token)
                    _SOURCE_WORK.reset(token)
                return LocalFailure(failure="permanent")

            execution, services, _ = _worker_services(execution, services, loop, pool, callback)
            before = len(created)
            await _execute(execution, services)
            assert len(created) == before + 1 and pool.occupied == 0

    asyncio.run(run())


@pytest.mark.parametrize("operation", list(DetectionTransformOperation))
def test_native_registered_config_hydration_keeps_wire_shape(operation):
    from data_designer.config.column_types import get_column_config_cls_from_type

    original = DetectionTransformConfig(name="result", operation=operation, excluded_entity_labels=["person"])
    payload = original.model_dump(mode="json")
    native = get_column_config_cls_from_type(payload["column_type"]).model_validate(payload)
    assert type(native) is DetectionTransformConfig
    assert native.model_dump(mode="json") == payload
    assert payload["column_type"] == "anonymizer-detection-transform"
    assert payload["operation"] == operation.value
    assert "generator_function" not in payload and "generator_params" not in payload
