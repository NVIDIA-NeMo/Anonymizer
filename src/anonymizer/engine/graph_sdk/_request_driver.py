# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One physical dispatch, cancellation and settlement driver for every request owner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol

from anonymizer.engine.graph_sdk._ingress import (
    inspect_failure,
    inspect_settlement,
    inspect_source_identity,
    inspect_usage,
)
from anonymizer.engine.graph_sdk.blocking import _MALFORMED, _NOT_STARTED, _BlockingInvocation
from anonymizer.engine.graph_sdk.context import ContextSourceRef, SourceFailure, SourceLost, SourceResponse
from anonymizer.engine.graph_sdk.observations import _CURRENT, _exception_terminal, _Span
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    AssociationResult,
    Dispatch,
    MarkLost,
    PhysicalRequestId,
    RejectSettlement,
    RequestAssociation,
    RequestCancel,
    RequestEvent,
    RequestPolicyBinding,
    RequestState,
    StopAcknowledged,
    StopConfirmed,
    TransportFailure,
    TransportLost,
    TransportSuccess,
    advance_requests,
    bind_request_policies,
)


class _Cancellation(Protocol):
    @property
    def cancelled(self) -> bool: ...


@dataclass(slots=True)
class _RequestAuthority:
    state: RequestState
    _lock: RLock = field(default_factory=RLock, repr=False)

    def dispatch_if_active(self, request: PhysicalRequestId, control: _Cancellation) -> bool:
        with self._lock:
            if control.cancelled:
                return False
            self.apply(Dispatch(request=request))
            return True

    def apply(self, event: RequestEvent) -> None:
        with self._lock:
            self.state = advance_requests(state=self.state, event=event)

    def bind(self, binding: RequestPolicyBinding) -> None:
        with self._lock:
            self.state = bind_request_policies(state=self.state, binding=binding)


@dataclass(frozen=True, slots=True)
class _PhysicalCompletion:
    value: object
    cancelled: bool
    stop_confirmed: bool
    settlement_valid: bool
    raised: bool = False
    malformed: bool = False


def late_source_event(
    value: object,
    request: PhysicalRequestId,
    association: RequestAssociation,
    outcome: str,
    source: ContextSourceRef,
    validate: Callable[[SourceResponse], bool],
) -> AcceptResult | AcceptFailure | None:
    """Reduce a late retrieval to metadata without publishing its values."""
    if type(value) is SourceResponse:
        if validate(value):
            return AcceptResult(
                request=request,
                results=(
                    AssociationResult(
                        association=association,
                        outcome=outcome,
                        outputs=(),
                        consumed_context_ports=frozenset(),
                    ),
                ),
            )
        return AcceptFailure(request=request, failure="malformed_response")
    if type(value) is SourceFailure:
        valid = inspect_source_identity(value.source, source) and not isinstance(
            inspect_settlement(value.settlement, request), RejectSettlement
        )
        return AcceptFailure(request=request, failure=inspect_failure(value.failure) if valid else "malformed_response")
    return None


async def drive_request(
    authority: _RequestAuthority,
    request: PhysicalRequestId,
    control: _Cancellation,
    invoke: Callable[[], Awaitable[object]],
    cancel: Callable[[PhysicalRequestId], Awaitable[object]],
    accept_late: Callable[[object], AcceptResult | AcceptFailure | None] | None = None,
    *,
    blocking: _BlockingInvocation | None = None,
) -> _PhysicalCompletion:
    """Observe actual dispatch, including delayed ordinary-worker starts."""
    current = _CURRENT.get()
    physical: _Span | None = None

    def dispatched() -> None:
        nonlocal physical
        if current is not None:
            physical = _Span("physical_request", session=current[0], parent=current[1])

    try:
        result = await _drive_request_body(
            authority, request, control, invoke, cancel, accept_late, blocking=blocking, dispatched=dispatched
        )
        if physical is not None:
            if result.cancelled:
                physical.outcome = "cancelled" if result.stop_confirmed else "lost"
                physical.reasons = frozenset({"cancelled" if result.stop_confirmed else "lost"})
            elif result.raised:
                physical.outcome = "failure"
                physical.reasons = frozenset({"provider_failure"})
            elif result.malformed or not result.settlement_valid:
                physical.outcome = "failure"
                physical.reasons = frozenset({"malformed_response"})
            elif type(result.value) in (TransportFailure, SourceFailure):
                physical.outcome = "failure"
                physical.reasons = frozenset({"provider_failure"})
            elif type(result.value) in (TransportLost, SourceLost):
                physical.outcome = "lost"
                physical.reasons = frozenset({"lost"})
        return result
    except BaseException as error:
        if physical is not None:
            physical.outcome, physical.reasons = _exception_terminal(error)
        raise
    finally:
        if physical is not None:
            physical.finish()


async def _drive_request_body(
    authority: _RequestAuthority,
    request: PhysicalRequestId,
    control: _Cancellation,
    invoke: Callable[[], Awaitable[object]],
    cancel: Callable[[PhysicalRequestId], Awaitable[object]],
    accept_late: Callable[[object], AcceptResult | AcceptFailure | None] | None = None,
    *,
    blocking: _BlockingInvocation | None = None,
    dispatched: Callable[[], None],
) -> _PhysicalCompletion:
    """Drive one reserved request without publishing callback payloads."""
    if control.cancelled:
        authority.apply(RequestCancel(request=request))
        return _PhysicalCompletion(None, True, True, True)
    if blocking is None:
        authority.apply(Dispatch(request=request))
        dispatched()

    async def call() -> object:
        if blocking is not None:

            def before() -> bool:
                if not authority.dispatch_if_active(request, control):
                    return False
                dispatched()
                return True

            return await blocking.run(before, control)
        return await invoke()

    task = asyncio.create_task(call())
    while not task.done() and not control.cancelled:
        await asyncio.sleep(0)
    cancelled = control.cancelled and not task.done()
    stopped: object = None
    if cancelled:
        authority.apply(RequestCancel(request=request))
        if request in authority.state.dispatched:
            try:
                stopped = await cancel(request)
            except Exception:
                pass
        if blocking is None:
            task.cancel()
        else:
            # Publish the first semantic terminal while the actual worker still
            # owns its slot. Its eventual value can only settle usage.
            if request in authority.state.dispatched:
                if type(stopped) is StopConfirmed and inspect_usage(stopped.usage):
                    authority.apply(StopAcknowledged(request=request, usage=stopped.usage))
                else:
                    authority.apply(MarkLost(request=request))
    raised = False
    try:
        value = await task
    except (Exception, asyncio.CancelledError):
        value = None
        raised = True
    if value is _NOT_STARTED:
        if not cancelled:
            authority.apply(RequestCancel(request=request))
        return _PhysicalCompletion(None, True, True, True)
    confirmed = type(stopped) is StopConfirmed and inspect_usage(stopped.usage)
    if cancelled:
        if blocking is not None:
            pass  # Already terminal; only late metadata may be added below.
        elif confirmed:
            assert isinstance(stopped, StopConfirmed)
            authority.apply(StopAcknowledged(request=request, usage=stopped.usage))
        else:
            authority.apply(MarkLost(request=request))
        if accept_late is not None:
            event = accept_late(value)
            if event is not None:
                authority.apply(event)
    with _Span("request_reconciliation") as reconciliation:
        valid = value is not _MALFORMED
        if not valid:
            authority.apply(RejectSettlement(request=request))
        if type(value) in {
            TransportSuccess,
            TransportFailure,
            TransportLost,
            SourceResponse,
            SourceFailure,
            SourceLost,
        }:
            assert isinstance(
                value, (TransportSuccess, TransportFailure, TransportLost, SourceResponse, SourceFailure, SourceLost)
            )
            event = inspect_settlement(value.settlement, request)
            if event is None and type(value) is SourceResponse:
                event = RejectSettlement(request=request)
            if event is not None:
                authority.apply(event)
            valid = not isinstance(event, RejectSettlement)
        if not valid:
            reconciliation.outcome = "failure"
            reconciliation.reasons = frozenset({"malformed_response"})
        elif raised:
            reconciliation.outcome = "unknown"
            reconciliation.reasons = frozenset({"provider_failure"})
        return _PhysicalCompletion(value, cancelled, confirmed, valid, raised, value is _MALFORMED)
