# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registered detection transforms: literal outputs, release-identical standalone behavior, owned reuse."""

from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace
from typing import Any

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
def test_literal_empty_payload_runs_directly_like_release(monkeypatch, operation, expected, asynchronous):
    from anonymizer.graph import InvocationId

    monkeypatch.setattr(InvocationId, "new", lambda **kwargs: pytest.fail("standalone transform started an invocation"))
    row = {"opaque": object()}
    generator = _generator(operation)
    output = asyncio.run(generator.agenerate(row)) if asynchronous else generator.generate(row)
    assert output == {"opaque": row["opaque"], **expected}
    assert (output is row) is (not asynchronous)  # Native base async contract copies the input row.


def test_native_parse_entity_literal():
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


@pytest.mark.parametrize("asynchronous", [False, True])
def test_original_exception_identity_like_release(monkeypatch, asynchronous):
    from anonymizer.engine.workflow_columns.detection import impl

    failure = ValueError("literal transform failure")

    def fail(row):
        raise failure

    monkeypatch.setitem(impl._TRANSFORMS, DetectionTransformOperation.PARSE_DETECTED_ENTITIES, fail)
    with pytest.raises(ValueError) as caught:
        generator = _generator("parse_detected_entities")
        asyncio.run(generator.agenerate({})) if asynchronous else generator.generate({})
    assert caught.value is failure
    assert not hasattr(failure, "_anonymizer_execution")


def test_repeated_async_observer_cancel_retains_local_worker(monkeypatch):
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


def test_managed_missing_source_context_cannot_start_fresh_invocation(monkeypatch):
    from anonymizer.graph import InvocationId
    from anonymizer.interface.errors import AnonymizerWorkflowError

    generator = _generator("parse_detected_entities")
    generator._anonymizer_source_frame = None
    monkeypatch.setattr(InvocationId, "new", lambda **kwargs: pytest.fail("unexpected new invocation"))
    with pytest.raises(AnonymizerWorkflowError):
        generator.generate({})


@pytest.mark.parametrize("asynchronous", [False, True])
def test_local_self_cancellation_preserves_error(monkeypatch, asynchronous):
    from anonymizer.engine.workflow_columns.detection import impl

    cancellation = asyncio.CancelledError("literal local cancellation")

    def cancel(row):
        raise cancellation

    monkeypatch.setitem(impl._TRANSFORMS, DetectionTransformOperation.PARSE_DETECTED_ENTITIES, cancel)
    with pytest.raises(asyncio.CancelledError) as caught:
        generator = _generator("parse_detected_entities")
        asyncio.run(generator.agenerate({})) if asynchronous else generator.generate({})
    assert caught.value is cancellation
    assert not hasattr(caught.value, "_anonymizer_execution")


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("fault", [None, "foreign", "expired", "missing_owner"])
def test_managed_transform_reuses_authenticated_owner(monkeypatch, asynchronous, fault):
    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW
    from anonymizer.engine.graph_sdk.executor import LocalFailure
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation
    from anonymizer.engine.ndd._source_frame import _SOURCE_WORK, SourceWork
    from anonymizer.graph import EffectRejected, InvocationId, TaskAttemptId
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


@pytest.mark.parametrize("asynchronous", [False, True])
def test_registered_finalize_preserves_explicit_label_scope(asynchronous: bool) -> None:
    generator = DetectionTransformGenerator(
        DetectionTransformConfig(
            name="result",
            operation=DetectionTransformOperation.APPLY_VALIDATION_AND_FINALIZE,
            allowed_entity_labels=["country"],
        ),
        resource_provider=SimpleNamespace(),  # ty: ignore[invalid-argument-type]
    )
    row: dict[str, Any] = {
        COL_TEXT: "San Diego",
        COL_MERGED_ENTITIES: {
            "entities": [
                {"id": "city_0_9", "value": "San Diego", "label": "city", "start_position": 0, "end_position": 9}
            ]
        },
    }
    result = asyncio.run(generator.agenerate(row)) if asynchronous else generator.generate(row)
    assert result[COL_DETECTED_ENTITIES] == {"entities": []}
    assert result[COL_TAGGED_TEXT] == "San Diego"
