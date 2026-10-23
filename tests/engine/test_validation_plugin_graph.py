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


@pytest.mark.parametrize("cancel", [False])
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
