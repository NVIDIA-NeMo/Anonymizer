# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from threading import Event, get_ident
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._request_driver import _RequestAuthority, drive_request
from anonymizer.engine.graph_sdk.blocking import BlockingAffinity, BlockingPool, BlockingTransport
from anonymizer.engine.graph_sdk.requests import (
    ExactUsage,
    ExternalSettlement,
    InvocationRequestScope,
    PhysicalRequestId,
    RequestPolicyBinding,
    Reserve,
    StopUnknown,
    TransportFailure,
    initialize_requests,
    request_receipt,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease, close_resource
from anonymizer.graph import EffectCode, EffectRejected
from anonymizer.graph._values import InvocationId, PlanId
from tests.graph_sdk.test_requests import _association, _policy

AFFINITY = BlockingAffinity(cross_thread=True, concurrency_safe=True, close_on_event_loop=True)


@dataclass
class Control:
    cancelled: bool = False


def reserved():
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    policy = _policy(attempts=1)
    association = _association(invocation, 0)
    authority = _RequestAuthority(initialize_requests(scope=scope, hard_limit=1, policies=frozenset({policy})))
    authority.bind(RequestPolicyBinding.create(association=association, policies=frozenset({policy})))
    request = PhysicalRequestId.new(scope=scope)
    authority.apply(Reserve(request=request, purpose="initial", associations=frozenset({association}), policy=policy))
    # Only the driver and adapter are under test; callback does not consume an envelope's operation.
    return authority, request


async def until(predicate) -> None:
    async with asyncio.timeout(5):
        while not predicate():
            await asyncio.sleep(0)


def launch(adapter, authority, request, control):
    from anonymizer.engine.graph_sdk.blocking import _BlockingInvocation

    return asyncio.create_task(
        drive_request(
            authority,
            request,
            control,
            lambda: adapter.dispatch(None),
            adapter.cancel,
            blocking=_BlockingInvocation(adapter, lambda: adapter._callback(request)),
        )
    )


@pytest.mark.parametrize("raises", [False, True])
def test_worker_slot_cancelled_waiter_late_completion_and_async_progress(raises: bool) -> None:
    async def run() -> None:
        pool = BlockingPool(max_workers=1)
        entered, release = Event(), Event()
        calls = []
        coordinator = get_ident()

        def callback(request):
            calls.append(request)
            assert get_ident() != coordinator
            entered.set()
            assert release.wait(5)
            if raises:
                raise RuntimeError("private callback failure")
            return TransportFailure(
                failure="permanent",
                settlement=ExternalSettlement(
                    request=request,
                    disposition="completed",
                    usage=ExactUsage(input_units=1, output_units=2),
                    remote_stopped=True,
                ),
            )

        adapter = BlockingTransport(callback=callback, pool=pool, affinity=AFFINITY)
        first, one = reserved()
        second, two = reserved()
        third, three = reserved()
        c1, c2, c3 = Control(), Control(), Control()
        t1 = launch(adapter, first, one, c1)
        tasks = [t1]
        try:
            await until(entered.is_set)
            assert first.state.dispatched == frozenset({one}) and pool.occupied == 1
            t2 = launch(adapter, second, two, c2)
            t3 = launch(adapter, third, three, c3)
            tasks += [t2, t3]
            await until(lambda: adapter.local_users == 3)
            c2.cancelled = True
            completion2 = await asyncio.wait_for(t2, 5)
            assert completion2.cancelled and completion2.stop_confirmed
            assert not second.state.dispatched and not second.state.reserved
            c1.cancelled = True
            await until(lambda: bool(first.state.terminals))
            assert first.state.terminals[0].category == "lost"
            assert not t1.done() and pool.occupied == 1 and not third.state.dispatched
            # Independent async dispatch remains available while the pool is full.
            other, four = reserved()

            async def immediate():
                return TransportFailure(failure="permanent", settlement=None)

            await drive_request(other, four, Control(), immediate, adapter.cancel)
            assert four in other.state.dispatched
            lease = ResourceLease.create(owner="sdk", safe_detachment="independent_after_dispatch", handle=adapter)
            assert (await close_resource(lease)).disposition == "left_open"
            release.set()
            completion1, completion3 = await asyncio.wait_for(asyncio.gather(t1, t3), 5)
            assert completion1.cancelled and completion1.raised == raises
            assert not completion3.cancelled
            assert first.state.terminals[0].category == "lost"
            receipt = request_receipt(first.state)
            assert receipt.dispatched_count == 1
            assert bool(receipt.remote_outstanding) == raises
            assert pool.occupied == 0 and calls == [one, three]
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await pool.close()

    asyncio.run(run())


def test_blocking_affinity_bounds_awaitables_reentry_and_cleanup() -> None:
    for count in (True, 0, -1):
        with pytest.raises(EffectRejected):
            BlockingPool(max_workers=count)
    with pytest.raises(EffectRejected) as failure:
        BlockingAffinity(cross_thread=False, concurrency_safe=False, close_on_event_loop=True)
    assert failure.value.code is EffectCode.UNSUPPORTED

    async def run() -> None:
        pool = BlockingPool()
        assert pool.max_workers == 4
        closed = []

        async def close():
            closed.append(get_ident())

        async def unwanted():
            raise AssertionError("must not await synchronous callback result")

        try:
            adapter = BlockingTransport(callback=lambda request: unwanted(), pool=pool, affinity=AFFINITY, close=close)
            authority, request = reserved()
            result = await launch(adapter, authority, request, Control())
            assert not result.raised and not result.cancelled
            assert not isinstance(result.value, TransportFailure)
            caller = ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=adapter)
            assert (await close_resource(caller)).disposition == "left_open" and not closed
            sdk = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=adapter)
            assert (await close_resource(sdk)).disposition == "closed" and closed == [get_ident()]
            from anonymizer.engine.graph_sdk.executor import execute_sync

            def recursive(request):
                with pytest.raises(EffectRejected) as rejected:
                    cast(Any, execute_sync)(admitted=None, capabilities=(), services=None)
                assert rejected.value.code is EffectCode.UNSUPPORTED
                return TransportFailure(failure="permanent", settlement=None)

            adapter2 = BlockingTransport(callback=recursive, pool=pool, affinity=AFFINITY)
            authority2, request2 = reserved()
            result2 = await launch(adapter2, authority2, request2, Control())
            assert not result2.raised
        finally:
            await pool.close()

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["success", "cancel", "exception", "awaitable"])
def test_actual_initial_blocking_provider(mode: str) -> None:
    from anonymizer.engine.graph_sdk.blocking import BlockingContextProvider
    from anonymizer.engine.graph_sdk.context import SourceItem, SourceResponse
    from tests.graph_sdk.test_binding import SOURCE, _initial_running

    async def run() -> None:
        pool = BlockingPool(max_workers=1)
        entered, release = Event(), Event()
        calls, closes = [], []
        cancellation_seen = asyncio.Event()

        async def cancel(request):
            cancellation_seen.set()
            return StopUnknown()

        async def close():
            closes.append(True)

        async def unwanted():
            raise AssertionError("must not await")

        def callback(*, request, association, selector, bounds):
            calls.append(request)
            entered.set()
            if mode == "cancel":
                assert release.wait(5)
            if mode == "exception":
                raise RuntimeError("private failure")
            if mode == "awaitable":
                return unwanted()
            return SourceResponse(
                source=SOURCE,
                items=(SourceItem(association=association, key=0, version=1, text="retrieved"),),
                settlement=ExternalSettlement(
                    request=request,
                    disposition="completed",
                    usage=ExactUsage(input_units=1, output_units=1),
                    remote_stopped=True,
                ),
            )

        provider = BlockingContextProvider(callback=callback, pool=pool, affinity=AFFINITY, close=close, cancel=cancel)
        running = None
        try:
            running, _, _, _ = await _initial_running(provider)
            await until(entered.is_set)
            if mode == "cancel":
                running.request_cancel()
                await asyncio.wait_for(cancellation_seen.wait(), 5)
                assert pool.occupied == 1
                release.set()
            result = await asyncio.wait_for(running.wait(), 5)
            receipt = result.receipt.requests
            assert receipt.dispatched_count == 1 and len(calls) == 1
            assert not closes  # The actual entry path retains caller ownership.
            if mode == "success":
                assert receipt.terminals[0].category == "success"
                assert result.context
            else:
                assert not result.context
                assert receipt.terminals[0].category == ("failure" if mode == "awaitable" else "lost")
            if mode in {"success", "cancel"}:
                assert len(receipt.settlements) == 1
                assert receipt.settlements[0].usage == ExactUsage(input_units=1, output_units=1)
                assert not receipt.remote_outstanding
            else:
                assert receipt.remote_outstanding
        finally:
            release.set()
            if running is not None:
                await asyncio.wait_for(running.wait(), 5)
            await pool.close()

    asyncio.run(run())


def test_actual_retrieval_queue_cancellation_preserves_worker_capacity() -> None:
    from anonymizer.engine.graph_sdk.blocking import BlockingContextProvider
    from anonymizer.engine.graph_sdk.context import SourceItem, SourceResponse
    from tests.graph_sdk.test_binding import SOURCE, _initial_running

    async def run() -> None:
        pool = BlockingPool(max_workers=1)
        entered, release = Event(), Event()
        stopped = asyncio.Event()
        calls = []

        async def cancel(request):
            stopped.set()
            return StopUnknown()

        def callback(*, request, association, selector, bounds):
            calls.append(request)
            entered.set()
            assert release.wait(5)
            return SourceResponse(
                source=SOURCE,
                items=(SourceItem(association=association, key=0, version=1, text="retrieved"),),
                settlement=ExternalSettlement(
                    request=request,
                    disposition="completed",
                    usage=ExactUsage(input_units=1, output_units=1),
                    remote_stopped=True,
                ),
            )

        provider = BlockingContextProvider(callback=callback, pool=pool, affinity=AFFINITY, cancel=cancel)

        async def start():
            values = await _initial_running(provider)
            return values[0]

        running = []
        try:
            one = await start()
            running.append(one)
            await until(entered.is_set)
            two, three = await start(), await start()
            running.extend((two, three))
            await until(lambda: provider.local_users == 3)
            two.request_cancel()
            result_two = await asyncio.wait_for(two.wait(), 5)
            receipt_two = result_two.receipt.requests
            assert receipt_two.dispatched_count == 0 and not receipt_two.remote_outstanding
            one.request_cancel()
            await asyncio.wait_for(stopped.wait(), 5)
            assert len(calls) == 1 and pool.occupied == 1
            release.set()
            result_one, result_three = await asyncio.wait_for(asyncio.gather(one.wait(), three.wait()), 5)
            receipt_one = result_one.receipt.requests
            receipt_three = result_three.receipt.requests
            assert receipt_one.terminals[0].category == "lost"
            assert receipt_three.terminals[0].category == "success"
            assert len(receipt_one.settlements) == len(receipt_three.settlements) == 1
            assert not result_one.context
            assert len(calls) == 2 and pool.occupied == 0
        finally:
            release.set()
            await asyncio.wait_for(asyncio.gather(*(item.wait() for item in running)), 5)
            await pool.close()

    asyncio.run(run())


@pytest.mark.parametrize("trusted_stop", [False, True])
def test_actual_blocking_transport_late_failure_and_sdk_cleanup(trusted_stop: bool) -> None:
    from anonymizer.engine.graph_sdk.requests import DispatchEnvelope, StopConfirmed
    from tests.graph_sdk.test_adaptive_executor import _external_running

    async def run() -> None:
        pool = BlockingPool(max_workers=1)
        entered, release = Event(), Event()
        cancelled = asyncio.Event()
        calls, closes = [], []

        async def cancel(request):
            cancelled.set()
            return StopConfirmed(usage=ExactUsage(input_units=1, output_units=1)) if trusted_stop else StopUnknown()

        async def close():
            assert release.is_set() and pool.occupied == 0
            closes.append(get_ident())

        def callback(envelope):
            assert isinstance(envelope, DispatchEnvelope)
            calls.append(envelope.request)
            entered.set()
            assert release.wait(5)
            return TransportFailure(
                failure="permanent",
                settlement=ExternalSettlement(
                    request=envelope.request,
                    disposition="completed",
                    usage=ExactUsage(input_units=1, output_units=1),
                    remote_stopped=True,
                ),
            )

        adapter = BlockingTransport(callback=callback, pool=pool, affinity=AFFINITY, cancel=cancel, close=close)
        running = await _external_running(adapter)
        try:
            await until(entered.is_set)
            running.request_cancel()
            await asyncio.wait_for(cancelled.wait(), 5)
            assert pool.occupied == 1 and not closes
            release.set()
            result = await asyncio.wait_for(running.wait(), 5)
            assert result.requests.terminals[0].category == ("cancelled" if trusted_stop else "lost")
            assert result.requests.dispatched_count == 1 and len(calls) == 1
            assert len(result.requests.settlements) == 1 and not result.requests.remote_outstanding
            assert not result.final_outputs and closes == [get_ident()]
        finally:
            release.set()
            await asyncio.wait_for(running.wait(), 5)
            await pool.close()

    asyncio.run(run())


def test_closed_pool_never_dispatches_or_charges() -> None:
    async def run() -> None:
        pool = BlockingPool(max_workers=1)
        await pool.close()

        def callback(request):
            raise AssertionError("closed pool must not invoke callbacks")

        adapter = BlockingTransport(callback=callback, pool=pool, affinity=AFFINITY)
        authority, request = reserved()
        result = await launch(adapter, authority, request, Control())
        assert result.cancelled and result.stop_confirmed
        assert not authority.state.dispatched and not authority.state.reserved
        assert pool.occupied == 0

    asyncio.run(run())


@pytest.mark.parametrize("raises", [False, True])
def test_internal_observer_cancellation_keeps_handle_until_callable_ends(raises: bool) -> None:
    async def run() -> None:
        pool = BlockingPool(max_workers=2)
        entered, release = Event(), Event()
        cancelled = asyncio.Event()
        calls, closes = [], []

        async def cancel(request):
            cancelled.set()
            return StopUnknown()

        async def close():
            closes.append(True)

        def callback(request):
            calls.append(request)
            entered.set()
            assert release.wait(5)
            if raises:
                raise RuntimeError("private worker exception")
            return TransportFailure(
                failure="permanent",
                settlement=ExternalSettlement(
                    request=request,
                    disposition="completed",
                    usage=ExactUsage(input_units=1, output_units=1),
                    remote_stopped=True,
                ),
            )

        adapter = BlockingTransport(
            callback=callback,
            pool=pool,
            affinity=BlockingAffinity(cross_thread=True, concurrency_safe=False, close_on_event_loop=True),
            cancel=cancel,
            close=close,
        )
        authority, request = reserved()
        control = Control()
        first = launch(adapter, authority, request, control)
        tasks = [first]
        try:
            await until(entered.is_set)
            control.cancelled = True
            await asyncio.wait_for(cancelled.wait(), 5)
            await until(lambda: bool(authority.state.terminals))
            # This is the internal await phase after the semantic terminal.
            first.cancel()
            second_authority, second_request = reserved()
            second = launch(adapter, second_authority, second_request, Control())
            tasks.append(second)
            await until(lambda: adapter.local_users == 2)
            for _ in range(3):
                first.cancel()
                await asyncio.sleep(0)  # Deliver each cancellation at an await checkpoint.
                assert not first.done() and adapter.local_users == 2
            assert pool.occupied == 1 and not second_authority.state.dispatched
            lease = ResourceLease.create(owner="sdk", safe_detachment="independent_after_dispatch", handle=adapter)
            assert (await close_resource(lease)).disposition == "left_open"
            assert not closes and len(calls) == 1 and not first.done()
            release.set()
            await asyncio.wait_for(asyncio.gather(first, second), 5)
            assert pool.occupied == adapter.local_users == 0 and len(calls) == 2
            assert authority.state.terminals[0].category == "lost"
            assert len(authority.state.settlements) == (0 if raises else 1)
            assert bool(authority.state.remote_outstanding) == raises
            assert (await close_resource(lease)).disposition == "closed" and closes == [True]
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            await pool.close()

    asyncio.run(run())
