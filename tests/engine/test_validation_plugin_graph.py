# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The validator borrows an owned source or opens one standalone invocation."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from anonymizer.engine.constants import COL_VALIDATION_DECISIONS
from anonymizer.engine.workflow_columns.detection.config import ChunkedValidationConfig
from anonymizer.engine.workflow_columns.detection.impl import ChunkedValidationGenerator
from tests.engine.test_chunked_validation import (
    _MINIMAL_TEMPLATE,
    FakeFacade,
    FakeModelRegistry,
    _build_row,
    _candidates_schema,
    _entity_span,
)
from tests.retained import retained

# Hang-guard bound for waits on normally-fast in-memory async events/futures in
# this module. Not a measurement of expected runtime -- only a generous ceiling
# so a hung test fails instead of blocking CI, chosen to stay robust under a
# loaded CI host.
_HANG_GUARD_TIMEOUT = 30.0


def _generator(model):
    return ChunkedValidationGenerator(
        ChunkedValidationConfig(
            name=COL_VALIDATION_DECISIONS,
            pool=["v0"],
            max_entities_per_call=1,
            excerpt_window_chars=50,
            max_parallel_chunks=2,
            prompt_template=_MINIMAL_TEMPLATE,
        ),
        resource_provider=SimpleNamespace(model_registry=FakeModelRegistry({"v0": model})),  # ty: ignore[invalid-argument-type]
    )


def _row():
    return _build_row(
        text="Alice Bob",
        seed_entities=[_entity_span("a", "Alice", "first_name", 0, 5), _entity_span("b", "Bob", "first_name", 6, 9)],
        candidates=_candidates_schema(("a", "Alice", "first_name"), ("b", "Bob", "first_name")),
    )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_plugin_projects_canonical_row_and_preserves_identity(monkeypatch, asynchronous):
    from anonymizer.engine.harness import kernel
    from anonymizer.graph import InvocationId

    invocations = []
    new = InvocationId.new

    def created(*args, **kwargs):
        value = new(*args, **kwargs)
        invocations.append(value)
        return value

    monkeypatch.setattr(InvocationId, "new", created)
    observed = []
    original = kernel._await_profile_execution

    async def start(running):
        execution = await original(running)
        observed.append(execution)
        return execution

    monkeypatch.setattr(kernel, "_await_profile_execution", start)
    model = FakeFacade(
        "v0", response=lambda prompt: {"decisions": [{"id": "a" if "'id': 'a'" in prompt else "b", "decision": "keep"}]}
    )
    generator = _generator(model)
    row = _row()
    opaque = object()
    row["opaque"] = opaque
    result = asyncio.run(generator.agenerate(row)) if asynchronous else generator.generate(row)
    assert result is row and result["opaque"] is opaque
    assert [value["id"] for value in result[COL_VALIDATION_DECISIONS]["decisions"]] == ["a", "b"]
    assert len(model.calls) == 2 and len(observed) == len(invocations) == 1
    execution = observed[0]
    assert not execution.requests.dispatches and not execution.final_outputs
    assert execution.frames[0].project(execution).at[0, COL_VALIDATION_DECISIONS] is row
    assert execution._execution.context.prepared.declared_request_upper_bound is None


@pytest.mark.parametrize("cancel", [False, True])
def test_async_plugin_drains_borrowed_calls_before_failure_or_cancellation(cancel):
    async def run():
        entered, release, failed = asyncio.Event(), asyncio.Event(), asyncio.Event()
        loop = asyncio.get_running_loop()
        failure = ValueError("literal failed chunk")

        class Model(FakeFacade):
            async def agenerate(self, *, prompt, **kwargs):
                assert asyncio.get_running_loop() is loop
                if "'id': 'a'" in prompt and not cancel:
                    await entered.wait()
                    failed.set()
                    raise failure
                entered.set()
                await release.wait()
                return self._call(prompt=prompt, parser=kwargs["parser"], system_prompt=None, purpose=None, kwargs={})

        model = Model("v0", response={"decisions": []})
        task = asyncio.create_task(_generator(model).agenerate(_row()))
        await asyncio.wait_for(entered.wait(), _HANG_GUARD_TIMEOUT)
        if cancel:
            task.cancel()
        else:
            await asyncio.wait_for(failed.wait(), _HANG_GUARD_TIMEOUT)
        for _ in range(20):
            await asyncio.sleep(0.005)
        assert not task.done(), "plugin returned while its borrowed chunk was still live"
        release.set()
        with pytest.raises(asyncio.CancelledError if cancel else ValueError):
            await asyncio.wait_for(task, _HANG_GUARD_TIMEOUT)
        assert not loop.is_closed()

    asyncio.run(run())


def test_sync_bridge_timeout_retains_actual_borrowed_completion(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    from data_designer.engine.models.clients.errors import SyncClientUnavailableError
    from data_designer.engine.models.errors import ModelTimeoutError

    from anonymizer.engine.workflow_columns.detection import impl

    entered, release, finished = Event(), Event(), Event()
    monkeypatch.setattr(impl, "_compute_bridge_timeout", lambda **kwargs: 0.01)

    class Model(FakeFacade):
        request_timeout = 1.0

        def generate(self, **kwargs):
            raise SyncClientUnavailableError("async only")

        async def agenerate(self, **kwargs):
            entered.set()
            try:
                await asyncio.to_thread(release.wait)
                return self._call(
                    prompt=kwargs["prompt"], parser=kwargs["parser"], system_prompt=None, purpose=None, kwargs={}
                )
            finally:
                finished.set()

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(_generator(Model("v0", response={"decisions": []})).generate, _row())
        try:
            assert entered.wait(_HANG_GUARD_TIMEOUT)
            assert not finished.wait(0.1)
            assert not future.done()
        finally:
            release.set()
        with pytest.raises(ModelTimeoutError):
            future.result(timeout=_HANG_GUARD_TIMEOUT)
        assert finished.is_set()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_original_chunk_error_is_preserved_with_canonical_failure(asynchronous):
    failure = ValueError("original plugin error")

    class Model(FakeFacade):
        def _call(self, **kwargs):
            raise failure

    generator = _generator(Model("v0"))
    with pytest.raises(ValueError) as caught:
        if asynchronous:
            asyncio.run(generator.agenerate(_row()))
        else:
            generator.generate(_row())
    assert caught.value is failure
    execution = retained(failure)
    assert not execution.frames and not execution.requests.dispatches
    assert execution.submission_failures[0].cause is failure


def test_concurrent_plugin_rows_keep_borrowed_loop_and_row_independence():
    async def run():
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []

        class Model(FakeFacade):
            async def agenerate(self, **kwargs):
                assert asyncio.get_running_loop() is loop
                calls.append(kwargs["prompt"])
                if len(calls) == 4:
                    entered.set()
                await release.wait()
                return self._call(
                    prompt=kwargs["prompt"], parser=kwargs["parser"], system_prompt=None, purpose=None, kwargs={}
                )

        generator = _generator(Model("v0", response={"decisions": []}))
        left, right = _row(), _row()
        left["opaque"], right["opaque"] = object(), object()
        tasks = [asyncio.create_task(generator.agenerate(row)) for row in (left, right)]
        try:
            await asyncio.wait_for(entered.wait(), _HANG_GUARD_TIMEOUT)
            assert all(not task.done() for task in tasks)
        finally:
            release.set()
        results = await asyncio.gather(*tasks)
        assert results[0] is left and results[1] is right
        assert results[0]["opaque"] is not results[1]["opaque"]
        assert len(calls) == 4
        # The caller's borrowed loop remains available after both graph owners drain.
        assert await asyncio.create_task(asyncio.sleep(0, result=17)) == 17

    asyncio.run(run())


@pytest.mark.parametrize("self_cancel", [False, True])
def test_async_plugin_self_cancellation_remains_cancellation(self_cancel):
    async def run():
        class Model(FakeFacade):
            async def agenerate(self, **kwargs):
                if self_cancel:
                    task = asyncio.current_task()
                    assert task is not None
                    task.cancel()
                    await asyncio.sleep(0)
                raise asyncio.CancelledError()

        with pytest.raises(asyncio.CancelledError) as caught:
            await _generator(Model("v0")).agenerate(_row())
        execution = retained(caught.value)
        assert not execution.frames and not execution.requests.dispatches
        assert {entry.status for entry in execution.invocation_state.entries} == {"cancelled"}

    asyncio.run(run())


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "foreign",
        "expired",
        "closed",
        "missing_owner",
        "malformed",
        "missing_managed",
        "malformed_frame",
        "foreign_frame",
        "unmanaged",
    ],
)
def test_owned_plugin_borrows_source_without_another_invocation(monkeypatch, asynchronous, fault):
    from contextvars import copy_context

    import pandas as pd

    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW
    from anonymizer.engine.graph_sdk.executor import LocalFailure
    from anonymizer.engine.graph_sdk.requests import SemanticAssociation
    from anonymizer.engine.ndd._source_frame import _SOURCE_WORK, SourceWork, WorkflowSourceFrame
    from anonymizer.graph import EffectRejected, InvocationId, TaskAttemptId
    from anonymizer.interface.errors import AnonymizerWorkflowError
    from tests.graph_sdk.test_delegated_execution import _execute, _fixture
    from tests.graph_sdk.test_delegated_worker import _loop_and_pool, _worker_services

    created = []
    new = InvocationId.new

    def invocation(*args, **kwargs):
        value = new(*args, **kwargs)
        created.append(value)
        return value

    monkeypatch.setattr(InvocationId, "new", invocation)

    async def run():
        async with _loop_and_pool() as (loop, pool):
            execution, services, *_ = _fixture()
            model = FakeFacade("v0", response={"decisions": []})
            generator = _generator(model)
            captured = {}

            def callback(inputs, borrowed):
                association = inputs[0].association
                if fault == "foreign":
                    association = SemanticAssociation(task=TaskAttemptId.new(activation=association.task.activation))
                work = SourceWork(association, COL_VALIDATION_DECISIONS) if fault != "malformed" else object()
                if fault == "foreign_frame":
                    peer = SemanticAssociation(task=TaskAttemptId.new(activation=association.task.activation))
                    generator._anonymizer_source_frame = WorkflowSourceFrame.capture(
                        pd.DataFrame({"text": ["Alice"]}), associations=(peer,)
                    )
                if fault in {"missing_managed", "malformed_frame"}:
                    generator._anonymizer_source_frame = (
                        WorkflowSourceFrame.capture(pd.DataFrame({"text": ["Alice"]}), associations=(association,))
                        if fault == "missing_managed"
                        else None
                    )
                if fault in {"missing_managed", "malformed_frame", "unmanaged"}:
                    work = None
                token = _SOURCE_WORK.set(work)  # ty: ignore[invalid-argument-type]
                owner_token = _WORKER_BORROW.set(None) if fault == "missing_owner" else None
                borrowed.expired = fault == "expired"
                borrowed.owner.closed = fault == "closed"
                captured["context"] = copy_context()
                try:

                    def generate():
                        if asynchronous:
                            return asyncio.run_coroutine_threadsafe(generator.agenerate(_row()), loop).result()
                        return generator.generate(_row())

                    if fault and fault != "unmanaged":
                        with pytest.raises((EffectRejected, AnonymizerWorkflowError)):
                            generate()
                        assert not model.calls
                    else:
                        assert COL_VALIDATION_DECISIONS in generate()
                        assert len(model.calls) == 2
                finally:
                    borrowed.expired = False
                    borrowed.owner.closed = False
                    if owner_token is not None:
                        _WORKER_BORROW.reset(owner_token)
                    _SOURCE_WORK.reset(token)
                return LocalFailure(failure="permanent")

            execution, services, _ = _worker_services(execution, services, loop, pool, callback)
            before_execution = len(created)
            await _execute(execution, services)
            assert len(created) == before_execution + (2 if fault == "unmanaged" else 1)
            assert pool.occupied == 0 and not loop.is_closed()
            if fault is None:
                with pytest.raises(EffectRejected):
                    await asyncio.wrap_future(
                        captured["context"].run(asyncio.run_coroutine_threadsafe, generator.agenerate(_row()), loop)
                    )
                assert len(model.calls) == 2 and len(created) == before_execution + 1

    asyncio.run(run())


@pytest.mark.parametrize("phase", ["authorization", "chunks"])
def test_owned_async_plugin_repeated_cancel_keeps_submission_until_chunks_end(monkeypatch, phase):
    from threading import Event

    from anonymizer.engine.graph_sdk.executor import LocalFailure
    from anonymizer.engine.ndd._source_frame import _SOURCE_WORK, SourceWork
    from tests.graph_sdk.test_delegated_execution import _execute, _fixture
    from tests.graph_sdk.test_delegated_worker import _loop_and_pool, _worker_services

    async def run():
        async with _loop_and_pool() as (loop, pool):
            entered, release = Event(), Event()
            captured = {}
            finished = []

            class Model(FakeFacade):
                async def agenerate(self, **kwargs):
                    entered.set()
                    while not release.is_set():
                        try:
                            await asyncio.sleep(0.01)
                        except asyncio.CancelledError:
                            continue
                    finished.append(kwargs["prompt"])
                    return self._call(
                        prompt=kwargs["prompt"], parser=kwargs["parser"], system_prompt=None, purpose=None, kwargs={}
                    )

            if phase == "authorization":
                from anonymizer.engine.graph_sdk._execution_delegated import _DelegatedOwner

                authorize = _DelegatedOwner._authorize_source

                async def held(owner, association):
                    entered.set()
                    while not release.is_set():
                        await asyncio.sleep(0.01)
                    return await authorize(owner, association)

                monkeypatch.setattr(_DelegatedOwner, "_authorize_source", held)
            generator = _generator(Model("v0", response={"decisions": []}))
            execution, services, *_ = _fixture()

            async def generate():
                captured["task"] = asyncio.current_task()
                try:
                    return await generator.agenerate(_row())
                except asyncio.CancelledError as error:
                    captured["cancellation"] = error.args
                    raise

            def callback(inputs, borrowed):
                captured["borrow"] = borrowed
                token = _SOURCE_WORK.set(SourceWork(inputs[0].association, COL_VALIDATION_DECISIONS))
                try:
                    asyncio.run_coroutine_threadsafe(generate(), loop).result()
                    return LocalFailure(failure="permanent")
                finally:
                    _SOURCE_WORK.reset(token)

            execution, services, _ = _worker_services(execution, services, loop, pool, callback)
            running = asyncio.create_task(_execute(execution, services))
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                for reason in ("first", "second"):
                    loop.call_soon_threadsafe(captured["task"].cancel, reason)
                    await asyncio.sleep(0.03)
                assert not running.done()
                assert pool.occupied == 1
                assert not captured["borrow"].expired
                assert not finished
            finally:
                release.set()
            result = await asyncio.wait_for(running, _HANG_GUARD_TIMEOUT)
            assert len(finished) == (2 if phase == "chunks" else 0)
            if phase == "authorization":
                assert captured["cancellation"] == ("first",)
            assert captured["borrow"].expired and pool.occupied == 0
            assert not result.requests.dispatches and not loop.is_closed()

    asyncio.run(run())
