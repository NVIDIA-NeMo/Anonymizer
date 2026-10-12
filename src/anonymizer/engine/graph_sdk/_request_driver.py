# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One physical dispatch, cancellation and settlement driver for every request owner."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from threading import RLock
from typing import Protocol

from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._foreign_transport import _ForeignInvocation
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
    RequestChain,
    RequestEvent,
    RequestPolicyBinding,
    RequestState,
    StopAcknowledged,
    StopConfirmed,
    TransportFailure,
    TransportLost,
    TransportSuccess,
    advance_requests,
    bind_request_chain,
    bind_request_policies,
)


class _Cancellation(Protocol):
    @property
    def cancelled(self) -> bool: ...


@dataclass(slots=True)
class _RetainedCancellation:
    source: _Cancellation
    observer_cancelled: bool = False

    @property
    def cancelled(self) -> bool:
        return self.observer_cancelled or self.source.cancelled


@dataclass(slots=True)
class _RequestAuthority:
    state: RequestState
    _lock: RLock = field(default_factory=RLock, repr=False)
    _observations: dict[PhysicalRequestId, tuple[_Span, ...]] = field(default_factory=dict, repr=False)

    def dispatch_if_active(self, request: PhysicalRequestId, control: _Cancellation) -> bool:
        with self._lock:
            if control.cancelled:
                return False
            self.apply(Dispatch(request=request))
            return True

    def apply(self, event: RequestEvent) -> RequestState:
        with self._lock:
            self.state = advance_requests(state=self.state, event=event)
            self._finish_observations()
            return self.state

    def observe_completion(self, request: PhysicalRequestId, spans: tuple[_Span, ...]) -> None:
        """Retain completed physical work until its existing owner classifies ingress."""
        with self._lock:
            self._observations[request] = spans
            self._finish_observations()

    def _finish_observations(self) -> None:
        if not self._observations:
            return
        for terminal in self.state.terminals:
            spans = self._observations.pop(terminal.request, ())
            for span in spans:
                span.outcome = terminal.category
                if terminal.category in {"cancelled", "lost"}:
                    span.reasons = frozenset({terminal.category})
                elif terminal.category == "failure":
                    span.reasons = frozenset(
                        {"malformed_response" if terminal.failure == "malformed_response" else "provider_failure"}
                    )
                elif terminal.category == "inconsistent":
                    span.reasons = frozenset({"malformed_response"})
                span.finish()

    def bind(self, binding: RequestPolicyBinding) -> None:
        with self._lock:
            self.state = bind_request_policies(state=self.state, binding=binding)

    def bind_chain(self, chain: RequestChain) -> None:
        with self._lock:
            self.state = bind_request_chain(state=self.state, chain=chain)


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
    foreign: _ForeignInvocation | None = None,
    prepare_dispatch: Callable[[], Callable[[], None]] | None = None,
    drain_on_cancel: bool = False,
) -> _PhysicalCompletion:
    """Observe actual dispatch, including delayed ordinary-worker starts."""
    current = _CURRENT.get()
    spans: list[_Span] = []

    def dispatched() -> None:
        if current is not None:
            spans.append(_Span("physical_request", session=current[0], parent=current[1]))

    def reconciling() -> None:
        if current is not None:
            spans.append(_Span("request_reconciliation", session=current[0], parent=current[1]))

    retained = _RetainedCancellation(control) if foreign is not None else None
    try:
        body = _drive_request_body(
            authority,
            request,
            retained if retained is not None else control,
            invoke,
            cancel,
            accept_late,
            blocking=blocking,
            foreign=foreign,
            prepare_dispatch=prepare_dispatch,
            drain_on_cancel=drain_on_cancel or foreign is not None,
            dispatched=dispatched,
            reconciling=reconciling,
        )
        if retained is None:
            result = await body
        else:
            task = asyncio.create_task(body)
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not task.done():
                        retained.observer_cancelled = True
            result = task.result()
        if spans:
            authority.observe_completion(request, tuple(spans))
        return result
    except BaseException as error:
        for span in spans:
            span.outcome, span.reasons = _exception_terminal(error)
            span.finish()
        raise


async def _drive_request_body(
    authority: _RequestAuthority,
    request: PhysicalRequestId,
    control: _Cancellation,
    invoke: Callable[[], Awaitable[object]],
    cancel: Callable[[PhysicalRequestId], Awaitable[object]],
    accept_late: Callable[[object], AcceptResult | AcceptFailure | None] | None = None,
    *,
    blocking: _BlockingInvocation | None = None,
    foreign: _ForeignInvocation | None = None,
    prepare_dispatch: Callable[[], Callable[[], None]] | None = None,
    drain_on_cancel: bool = False,
    dispatched: Callable[[], None],
    reconciling: Callable[[], None],
) -> _PhysicalCompletion:
    """Drive one reserved request without publishing callback payloads."""
    if (blocking is not None and (prepare_dispatch is not None or foreign is not None)) or (
        foreign is not None and foreign.envelope.request != request
    ):
        reject(EffectCode.UNSUPPORTED)
    if control.cancelled:
        authority.apply(RequestCancel(request=request))
        return _PhysicalCompletion(None, True, True, True)

    async def call() -> object:
        if foreign is not None:

            def before() -> bool:
                if control.cancelled:
                    return False
                commit = prepare_dispatch() if prepare_dispatch is not None else None
                if not authority.dispatch_if_active(request, control):
                    return False
                if commit is not None:
                    commit()
                dispatched()
                return True

            return await foreign.run(before)
        if blocking is None:
            if control.cancelled:
                return _NOT_STARTED
            commit = prepare_dispatch() if prepare_dispatch is not None else None
            authority.apply(Dispatch(request=request))
            if commit is not None:
                commit()
            dispatched()
            return await invoke()
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
        if foreign is not None:
            foreign.invalidate()
        authority.apply(RequestCancel(request=request))
        if request in authority.state.dispatched:
            try:
                stopped = await foreign.cancel() if foreign is not None else await cancel(request)
            except Exception:
                pass
        if blocking is None and not drain_on_cancel:
            task.cancel()
        else:
            # Retain the initiated call until local drain, independently of the
            # first terminal. Its eventual value can only settle usage.
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
    if request not in authority.state.dispatched:
        if any(item.request == request for item in authority.state.reserved):
            authority.apply(RequestCancel(request=request))
        return _PhysicalCompletion(None, True, True, True, raised)
    confirmed = type(stopped) is StopConfirmed and inspect_usage(stopped.usage)
    if cancelled:
        if blocking is not None or drain_on_cancel:
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
    reconciling()
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
    return _PhysicalCompletion(value, cancelled, confirmed, valid, raised, value is _MALFORMED)
