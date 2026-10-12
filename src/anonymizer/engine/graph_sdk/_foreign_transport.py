# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Closed-profile foreign transport entry; resource admission remains deferred."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, TimeoutError
from inspect import iscoroutinefunction
from threading import Lock

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_count, require_instance
from anonymizer.engine.graph_sdk._execution_values import RequestTransport
from anonymizer.engine.graph_sdk.blocking import _NOT_STARTED
from anonymizer.engine.graph_sdk.requests import DispatchEnvelope, PhysicalRequestId, StopResult, TransportResult


class _EntryGate:
    """One acknowledgement, ordered against timeout/cancellation; no request facts."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._ack: Future[bool] = Future()

    def invalidate(self) -> None:
        with self._lock:
            if not self._ack.done():
                self._ack.set_result(False)

    def admit(self, before: Callable[[], bool]) -> None:
        with self._lock:
            if self._ack.done():
                return
            try:
                accepted = before()
            except BaseException as error:
                self._ack.set_exception(error)
            else:
                self._ack.set_result(accepted)

    def wait(self, timeout: float) -> bool:
        # Called only on the foreign thread, holding no owner/client/gate lock.
        try:
            return self._ack.result(timeout=timeout)
        except TimeoutError:
            # Acceptance wins if the non-awaiting host gate already completed.
            self.invalidate()
            return self._ack.result()


class ForeignLoopTransport(PrivateValue):
    """Bind one original transport to its exact live loop; never retry or own it.

    This adapter is not admitted by graph services until resource conformance X3.
    Both loops must remain live through retained work. No lock may be held by the
    foreign caller across its host gate; the host never waits synchronously for it.
    """

    def __init__(self, *, inner: RequestTransport, loop: asyncio.AbstractEventLoop, gate_timeout_ns: int) -> None:
        require_count(gate_timeout_ns, positive=True)
        if not isinstance(loop, asyncio.AbstractEventLoop) or not isinstance(inner, RequestTransport):
            reject(EffectCode.INVALID_TYPE)
        if not iscoroutinefunction(inner.dispatch) or not iscoroutinefunction(inner.cancel):
            reject(EffectCode.UNSUPPORTED)
        self._inner = inner
        self._loop = loop
        self._gate_timeout_ns = gate_timeout_ns

    async def dispatch(self, request: DispatchEnvelope) -> TransportResult:
        reject(EffectCode.UNSUPPORTED)

    async def cancel(self, request: PhysicalRequestId) -> StopResult:
        reject(EffectCode.UNSUPPORTED)

    def _invocation(self, envelope: DispatchEnvelope) -> _ForeignInvocation:
        require_instance(envelope, DispatchEnvelope)
        return _ForeignInvocation(self, envelope)


class _ForeignInvocation(PrivateValue):
    def __init__(self, transport: ForeignLoopTransport, envelope: DispatchEnvelope) -> None:
        self.transport = transport
        self.envelope = envelope
        self._loop = transport._loop
        self._inner = transport._inner
        self._timeout_ns = transport._gate_timeout_ns
        self._gate = _EntryGate()
        self._used = False

    def invalidate(self) -> None:
        self._gate.invalidate()

    async def cancel(self) -> StopResult:
        future = asyncio.run_coroutine_threadsafe(self._inner.cancel(self.envelope.request), self._loop)
        completion = asyncio.wrap_future(future)
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                continue
        return completion.result()

    async def run(self, before: Callable[[], bool]) -> object:
        host = asyncio.get_running_loop()
        if self._used or host is self._loop:
            reject(EffectCode.UNSUPPORTED)
        self._used = True
        if self._loop.is_closed() or not self._loop.is_running():
            reject(EffectCode.UNSUPPORTED)

        async def enter() -> object:
            host.call_soon_threadsafe(self._gate.admit, before)
            if not self._gate.wait(self._timeout_ns / 1_000_000_000):
                return _NOT_STARTED
            # Direct await enters the original coroutine on this same stack.
            # There is no async suspension/cancellation check between acceptance
            # and entry. The first suspension belongs to the original transport.
            return await self._inner.dispatch(self.envelope)

        # Retain the scheduled task/future before any observer can suspend here.
        future = asyncio.run_coroutine_threadsafe(enter(), self._loop)
        completion = asyncio.wrap_future(future)
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                self.invalidate()
        return completion.result()
