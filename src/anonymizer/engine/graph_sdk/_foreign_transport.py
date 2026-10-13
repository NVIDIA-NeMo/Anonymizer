# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Foreign physical entry and its exclusive, loop-created client lease."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Future, TimeoutError
from contextlib import ExitStack
from inspect import iscoroutinefunction
from threading import Lock
from typing import Self

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    PrivateValue,
    reject,
    require_count,
    require_instance,
    require_literal,
)
from anonymizer.engine.graph_sdk._execution_values import RequestTransport
from anonymizer.engine.graph_sdk.blocking import _NOT_STARTED
from anonymizer.engine.graph_sdk.requests import DispatchEnvelope, PhysicalRequestId, StopResult, TransportResult
from anonymizer.engine.graph_sdk.resources import CloseableResource, ResourceLease, ResourceOwner


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
    """Bind one transport to its loop; registered clients have one invocation lease.

    Both loops must remain live through retained work. The factory must create a
    fresh client used exclusively by this adapter until its invocation drains.
    No lock may be held across the host gate; the host never blocks on that loop.
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
        self._lock = Lock()
        self._users: set[object] = set()
        self._lease: ResourceLease | None = None
        self._creation_loop: asyncio.AbstractEventLoop | None = None
        self._created_inner: RequestTransport | None = None
        self._claimed = False
        self._close_future: Future[None] | None = None

    @classmethod
    def create(
        cls, *, factory: Callable[[], RequestTransport], owner: ResourceOwner, gate_timeout_ns: int, exclusive: bool
    ) -> Self:
        """Create an exclusive client on its actual loop; caller keeps that loop live."""
        if exclusive is not True:
            reject(EffectCode.UNSUPPORTED)
        require_literal(owner, frozenset({"caller", "sdk"}))
        require_count(gate_timeout_ns, positive=True)
        loop = asyncio.get_running_loop()
        inner = factory()
        if not isinstance(inner, CloseableResource) or not iscoroutinefunction(inner.close):
            reject(EffectCode.UNSUPPORTED)
        result = cls(inner=inner, loop=loop, gate_timeout_ns=gate_timeout_ns)
        result._creation_loop = loop
        result._created_inner = inner
        result._lease = ResourceLease.create(owner=owner, safe_detachment="forbidden", handle=result)
        return result

    @property
    def lease(self) -> ResourceLease:
        if self._lease is None:
            reject(EffectCode.UNSUPPORTED)
        return self._lease

    @property
    def local_users(self) -> int:
        with self._lock:
            return len(self._users)

    def validate(self, lease: ResourceLease | None, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            if (
                self._lease is None
                or lease is not self._lease
                or self._inner is not self._created_inner
                or self._loop is not self._creation_loop
                or self._loop is not loop
                or loop is asyncio.get_running_loop()
                or not loop.is_running()
                or loop.is_closed()
                or self._claimed
                or self._close_future is not None
                or self._users
            ):
                reject(EffectCode.UNSUPPORTED)

    @staticmethod
    def claim_many(transports: tuple[ForeignLoopTransport, ...]) -> None:
        """Atomically reserve the exact adapters; a rejected set consumes none."""
        unique = {id(item): item for item in transports}
        ordered = [unique[key] for key in sorted(unique)]
        with ExitStack() as locks:
            for item in ordered:
                locks.enter_context(item._lock)
            if len({id(item._created_inner) for item in ordered}) != len(ordered):
                reject(EffectCode.UNSUPPORTED)
            if any(
                item._lease is None
                or item._inner is not item._created_inner
                or item._loop is not item._creation_loop
                or item._claimed
                or item._close_future is not None
                or item._users
                or not item._loop.is_running()
                or item._loop.is_closed()
                for item in ordered
            ):
                reject(EffectCode.UNSUPPORTED)
            for item in ordered:
                item._claimed = True

    async def close(self) -> None:
        with self._lock:
            if self._users:
                reject(EffectCode.REQUEST_CAUSALITY)
            if self._close_future is None:
                inner, loop = self._created_inner, self._creation_loop
                if not isinstance(inner, CloseableResource) or loop is None:
                    reject(EffectCode.UNSUPPORTED)
                self._close_future = asyncio.run_coroutine_threadsafe(inner.close(), loop)
            future = self._close_future
        completion = asyncio.wrap_future(future)
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                continue
        completion.result()

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
        self._loop = transport._creation_loop or transport._loop
        self._inner = transport._created_inner or transport._inner
        self._timeout_ns = transport._gate_timeout_ns
        self._gate = _EntryGate()
        self._used = False

    def invalidate(self) -> None:
        self._gate.invalidate()

    async def cancel(self) -> StopResult:
        future = asyncio.run_coroutine_threadsafe(self._inner.cancel(self.envelope.request), self._loop)
        with self.transport._lock:
            self.transport._users.add(future)
        try:
            completion = asyncio.wrap_future(future)
            while not completion.done():
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError:
                    continue
            return completion.result()
        finally:
            with self.transport._lock:
                self.transport._users.discard(future)

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
        with self.transport._lock:
            self.transport._users.add(future)
        try:
            completion = asyncio.wrap_future(future)
            while not completion.done():
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError:
                    self.invalidate()
            return completion.result()
        finally:
            with self.transport._lock:
                self.transport._users.discard(future)
