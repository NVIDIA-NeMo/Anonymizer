# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit bounded worker adapters for ordinary synchronous callbacks."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import BoundedSemaphore, Lock, local
from typing import Protocol

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_count, require_instance
from anonymizer.engine.graph_sdk.context import ContextSelector, RetrievalBounds, SourceResult
from anonymizer.engine.graph_sdk.requests import (
    DispatchEnvelope,
    PhysicalRequestId,
    RequestAssociation,
    StopResult,
    StopUnknown,
    TransportResult,
)

_WORKER = local()
_NOT_STARTED = object()
_MALFORMED = object()


def _reject_worker_entry() -> None:
    if getattr(_WORKER, "active", False):
        reject(EffectCode.UNSUPPORTED)


class _Cancellation(Protocol):
    @property
    def cancelled(self) -> bool: ...


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BlockingAffinity(PrivateValue):
    """Caller assertions about the callback and every captured client/handle.

    Cross-thread permission excludes creator-thread and event-loop affinity.
    A nonconcurrent handle is serialized by its adapter. Handles shared through
    multiple adapters or pools must declare concurrency_safe=True.
    """

    cross_thread: bool
    concurrency_safe: bool
    close_on_event_loop: bool

    def __post_init__(self) -> None:
        if any(type(v) is not bool for v in (self.cross_thread, self.concurrency_safe, self.close_on_event_loop)):
            reject(EffectCode.INVALID_TYPE)
        if not self.cross_thread:
            reject(EffectCode.UNSUPPORTED)


class BlockingPool(PrivateValue):
    """An explicit pool with no submissions beyond its occupied worker slots."""

    def __init__(self, *, max_workers: int = 4) -> None:
        require_count(max_workers, positive=True)
        self.max_workers = max_workers
        self._slots = BoundedSemaphore(max_workers)
        self._lock = Lock()
        self._closed = False
        self._active = 0
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="anonymizer-graph")

    @property
    def occupied(self) -> int:
        with self._lock:
            return self._active

    async def close(self) -> None:
        """Stop admission and wait asynchronously for actual worker completion."""
        _reject_worker_entry()
        with self._lock:
            self._closed = True
        while self.occupied:
            await asyncio.sleep(0)
        self._executor.shutdown(wait=False)

    async def _run(self, callback: Callable[[], object], before: Callable[[], bool], control: _Cancellation) -> object:
        while True:
            if control.cancelled:
                return _NOT_STARTED
            with self._lock:
                if self._closed:
                    return _NOT_STARTED
                if self._slots.acquire(blocking=False):
                    self._active += 1
                    break
            await asyncio.sleep(0)

        def call() -> object:
            try:
                if not before():
                    return _NOT_STARTED
                _WORKER.active = True
                value = callback()
                if inspect.isawaitable(value):
                    if inspect.iscoroutine(value):
                        value.close()
                    return _MALFORMED
                return value
            finally:
                _WORKER.active = False
                with self._lock:
                    self._active -= 1
                    self._slots.release()

        try:
            future = self._executor.submit(call)
        except BaseException:
            with self._lock:
                self._active -= 1
                self._slots.release()
            raise
        # Cancelling a coordinator observer must never cancel a queued worker
        # or release its slot; the request driver retains this completion.
        completion = asyncio.wrap_future(future)
        try:
            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            # Keep adapter serialization and local-user ownership until the
            # actual callable ends, even if the internal observer is cancelled.
            while not completion.done():
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            # The request owner still needs late settlement metadata. Returning
            # completion here cannot reopen its already recorded terminal.
            return completion.result()


@dataclass(frozen=True, slots=True)
class _BlockingInvocation:
    adapter: _BlockingAdapter
    callback: Callable[[], object]

    async def run(self, before: Callable[[], bool], control: _Cancellation) -> object:
        with self.adapter._lock:
            if self.adapter._closed:
                return _NOT_STARTED
            self.adapter._users += 1
        try:
            return await self._run(before, control)
        finally:
            with self.adapter._lock:
                self.adapter._users -= 1

    async def _run(self, before: Callable[[], bool], control: _Cancellation) -> object:
        gate = self.adapter._serial
        if gate is not None:
            while not gate.acquire(blocking=False):
                if control.cancelled:
                    return _NOT_STARTED
                await asyncio.sleep(0)
        try:
            return await self.adapter.pool._run(self.callback, before, control)
        finally:
            if gate is not None:
                gate.release()


class _BlockingAdapter(PrivateValue):
    def __init__(
        self,
        *,
        callback: Callable[..., object],
        pool: BlockingPool,
        affinity: BlockingAffinity,
        cancel: Callable[[PhysicalRequestId], Awaitable[StopResult]] | None = None,
        close: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        require_instance(pool, BlockingPool)
        require_instance(affinity, BlockingAffinity)
        if not callable(callback) or inspect.iscoroutinefunction(callback):
            reject(EffectCode.INVALID_TYPE)
        if close is not None and not affinity.close_on_event_loop:
            reject(EffectCode.UNSUPPORTED)
        self.pool = pool
        self.affinity = affinity
        self._callback = callback
        self._cancel = cancel
        self._close = close
        self._serial = None if affinity.concurrency_safe else Lock()
        self._lock = Lock()
        self._users = 0
        self._closed = False

    @property
    def local_users(self) -> int:
        with self._lock:
            return self._users

    async def cancel(self, request: PhysicalRequestId) -> StopResult:
        return StopUnknown() if self._cancel is None else await self._cancel(request)

    async def close(self) -> None:
        with self._lock:
            if self._users:
                reject(EffectCode.CONTRADICTORY)
            if self._closed:
                return
            self._closed = True
        if self._close is not None:
            await self._close()


class BlockingContextProvider(_BlockingAdapter):
    """Synchronous retrieval, invoked only by the SDK's request driver."""

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResult:
        reject(EffectCode.UNSUPPORTED)

    def _invocation(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> _BlockingInvocation:
        return _BlockingInvocation(
            self, lambda: self._callback(request=request, association=association, selector=selector, bounds=bounds)
        )


class BlockingTransport(_BlockingAdapter):
    """Synchronous transport, invoked only by the SDK's request driver."""

    async def dispatch(self, request: DispatchEnvelope) -> TransportResult:
        reject(EffectCode.UNSUPPORTED)

    def _invocation(self, request: DispatchEnvelope) -> _BlockingInvocation:
        return _BlockingInvocation(self, lambda: self._callback(request))


def _validate_provider_execution(provider: object, execution: str) -> None:
    if isinstance(provider, BlockingContextProvider) != (execution == "blocking"):
        reject(EffectCode.CONTRADICTORY)
