# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure request accounting state transitions."""

from __future__ import annotations

from dataclasses import replace
from typing import Literal

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
    require_count,
    require_instance,
    require_literal,
)
from anonymizer.engine.graph_sdk._request_values import (
    _FAILURES,
    _PURPOSES,
    AcceptFailure,
    AcceptResult,
    AssociationResult,
    BindingAssociation,
    BindingRequestScope,
    Dispatch,
    ExternalSettlement,
    InvocationRequestScope,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    RequestAssociation,
    RequestCancel,
    RequestDefect,
    RequestDenialFact,
    RequestEvent,
    RequestPolicyBinding,
    RequestReceipt,
    RequestReservation,
    RequestScope,
    RequestState,
    RequestTerminal,
    RequestTerminalFact,
    Reserve,
    ScopeCancel,
    SemanticAssociation,
    StopAcknowledged,
    T,
    _association_in_scope,
)


def initialize_requests(
    *, scope: RequestScope, hard_limit: int | None, policies: frozenset[PhysicalRequestPolicy]
) -> RequestState:
    """Create empty request authority for one invocation or binding."""
    require_instance(scope, (InvocationRequestScope, BindingRequestScope))
    if hard_limit is not None:
        require_count(hard_limit)
    if not isinstance(policies, frozenset) or any(not isinstance(item, PhysicalRequestPolicy) for item in policies):
        reject(EffectCode.INVALID_TYPE)
    return RequestState(
        scope=scope,
        hard_limit=hard_limit,
        policies=policies,
        bindings=frozenset(),
        reserved=frozenset(),
        dispatched=frozenset(),
        dispatches=(),
        denials=(),
        terminals=(),
        settlements=(),
        defects=(),
        local_in_flight=frozenset(),
        remote_outstanding=frozenset(),
        cancel_requested=frozenset(),
        cancelled=False,
    )


def bind_request_policies(*, state: RequestState, binding: RequestPolicyBinding) -> RequestState:
    """Bind one association to its owner-admitted policy set before reservation."""
    require_instance(state, RequestState)
    require_instance(binding, RequestPolicyBinding)
    if not _association_in_scope(binding.association, state.scope):
        reject(EffectCode.FOREIGN_OWNER)
    if not binding.policies <= state.policies:
        reject(EffectCode.CROSS_POLICY)
    if any(item.association == binding.association for item in state.bindings):
        reject(EffectCode.DUPLICATE)
    if any(binding.association in reservation.associations for reservation in state.reserved):
        reject(EffectCode.CONTRADICTORY)
    return replace(state, bindings=state.bindings | {binding})


def request_receipt(state: RequestState) -> RequestReceipt:
    """Project immutable inspection facts without changing request authority."""
    require_instance(state, RequestState)
    return RequestReceipt(
        scope=state.scope,
        budget_limit=state.hard_limit,
        policies=state.policies,
        bindings=state.bindings,
        dispatched_count=len(state.dispatches),
        dispatches=state.dispatches,
        denials=state.denials,
        terminals=state.terminals,
        settlements=state.settlements,
        defects=state.defects,
        local_in_flight=state.local_in_flight,
        remote_outstanding=state.remote_outstanding,
        cancel_requested=state.cancel_requested,
    )


def advance_requests(*, state: RequestState, event: RequestEvent) -> RequestState:
    """Apply one bounded request-accounting event."""
    require_instance(state, RequestState)
    require_instance(
        event,
        (
            Reserve,
            Dispatch,
            AcceptResult,
            AcceptFailure,
            RequestCancel,
            ScopeCancel,
            StopAcknowledged,
            MarkLost,
            ObserveSettlement,
        ),
    )
    if isinstance(event, ScopeCancel):
        terminals = state.terminals
        defects = state.defects
        for reservation in state.reserved:
            terminals, defects = _terminal(
                replace(state, terminals=terminals, defects=defects),
                _terminal_fact(reservation.request, "cancelled"),
            )
        return replace(
            state,
            reserved=frozenset(),
            terminals=terminals,
            defects=defects,
            cancelled=True,
        )
    request = event.settlement.request if isinstance(event, ObserveSettlement) else event.request
    if request.scope != state.scope:
        reject(EffectCode.FOREIGN_OWNER)
    if isinstance(event, Reserve):
        return _reserve(state, event)
    if isinstance(event, Dispatch):
        reservation = _reservation(state, request)
        return replace(
            state,
            reserved=state.reserved - {reservation},
            dispatched=state.dispatched | {request},
            dispatches=(*state.dispatches, reservation),
            local_in_flight=state.local_in_flight | {request},
            remote_outstanding=state.remote_outstanding | {request},
        )
    if isinstance(event, RequestCancel):
        reservations = frozenset(item for item in state.reserved if item.request != request)
        if reservations != state.reserved:
            terminal, defects = _terminal(state, _terminal_fact(request, "cancelled"))
            return replace(state, reserved=reservations, terminals=terminal, defects=defects)
        if request not in state.dispatched:
            reject(EffectCode.MISSING)
        if any(item.request == request for item in state.terminals):
            return state
        return replace(state, cancel_requested=state.cancel_requested | {request})
    if isinstance(event, StopAcknowledged):
        if request not in state.dispatched:
            reject(EffectCode.MISSING)
        terminals, defects = _terminal(state, _terminal_fact(request, "cancelled"))
        return replace(
            state,
            terminals=terminals,
            defects=defects,
            local_in_flight=state.local_in_flight - {request},
            remote_outstanding=state.remote_outstanding - {request},
        )
    if isinstance(event, MarkLost):
        if request not in state.dispatched:
            reject(EffectCode.MISSING)
        terminals, defects = _terminal(state, _terminal_fact(request, "lost"))
        return replace(
            state,
            terminals=terminals,
            defects=defects,
            local_in_flight=state.local_in_flight - {request},
        )
    if isinstance(event, ObserveSettlement):
        return _observe_settlement(state, event.settlement)
    if request not in state.dispatched:
        reject(EffectCode.MISSING)
    if isinstance(event, AcceptFailure):
        require_literal(event.failure, _FAILURES)
        terminal = RequestTerminalFact(request=request, category="failure", failure=event.failure, results=())
        terminals, defects = _terminal(state, terminal)
        clear_remote = not any(item.request == request for item in state.terminals)
        return replace(
            state,
            terminals=terminals,
            defects=defects,
            local_in_flight=state.local_in_flight - {request},
            remote_outstanding=state.remote_outstanding - ({request} if clear_remote else set()),
        )
    return _accept_results(state, event)


def can_reserve_followup(
    *,
    state: RequestState,
    purpose: Literal["retry", "correction", "failover"],
    associations: frozenset[RequestAssociation],
    policy: PhysicalRequestPolicy,
) -> bool:
    """Return whether the request authority would admit this follow-up relationship."""
    require_instance(state, RequestState)
    require_literal(purpose, frozenset({"retry", "correction", "failover"}))
    if not isinstance(associations, frozenset) or any(
        not isinstance(item, (SemanticAssociation, BindingAssociation)) for item in associations
    ):
        reject(EffectCode.INVALID_TYPE)
    require_instance(policy, PhysicalRequestPolicy)
    if any(
        sum(association in reservation.associations for reservation in state.dispatches) >= policy.max_attempts
        for association in associations
    ):
        return False
    return _followup_rejection(state, purpose, associations, policy) is None


def _reserve(state: RequestState, event: Reserve) -> RequestState:
    require_instance(event.request, PhysicalRequestId)
    require_literal(event.purpose, _PURPOSES)
    if not isinstance(event.associations, frozenset) or any(
        not isinstance(item, (SemanticAssociation, BindingAssociation)) for item in event.associations
    ):
        reject(EffectCode.INVALID_TYPE)
    require_instance(event.policy, PhysicalRequestPolicy)
    if not event.associations:
        reject(EffectCode.INVALID_VALUE)
    if event.request in state.dispatched or any(item.request == event.request for item in state.reserved):
        reject(EffectCode.DUPLICATE)
    if event.policy not in state.policies:
        reject(EffectCode.CROSS_POLICY)
    bindings = {item.association: item.policies for item in state.bindings}
    if any(event.policy not in bindings.get(item, frozenset()) for item in event.associations):
        reject(EffectCode.CROSS_POLICY)
    if any(not _association_in_scope(item, state.scope) for item in event.associations):
        reject(EffectCode.FOREIGN_OWNER)
    eligible = set(event.associations)
    _validate_followup(state, event)
    exhausted = {
        item
        for item in eligible
        if sum(item in reservation.associations for reservation in state.dispatches) >= event.policy.max_attempts
    }
    denials = state.denials
    if exhausted:
        denials += (
            RequestDenialFact(
                request=event.request,
                associations=frozenset(exhausted),
                category="request_limit_stopped",
            ),
        )
        eligible -= exhausted
    if eligible and state.hard_limit is not None and len(state.dispatches) + len(state.reserved) >= state.hard_limit:
        denials += (
            RequestDenialFact(
                request=event.request,
                associations=frozenset(eligible),
                category="budget_stopped",
            ),
        )
        eligible.clear()
    if not eligible:
        return replace(state, denials=denials)
    reservation = RequestReservation(
        request=event.request,
        purpose=event.purpose,
        associations=frozenset(eligible),
        policy=event.policy,
    )
    return replace(state, reserved=state.reserved | {reservation}, denials=denials)


def _reservation(state: RequestState, request: PhysicalRequestId) -> RequestReservation:
    matches = [item for item in state.reserved if item.request == request]
    if len(matches) != 1:
        reject(EffectCode.MISSING)
    return matches[0]


def _terminal_fact(request: PhysicalRequestId, category: RequestTerminal) -> RequestTerminalFact:
    return RequestTerminalFact(request=request, category=category, failure=None, results=())


def _terminal(
    state: RequestState, candidate: RequestTerminalFact
) -> tuple[tuple[RequestTerminalFact, ...], tuple[RequestDefect, ...]]:
    existing = next((item for item in state.terminals if item.request == candidate.request), None)
    if existing is None:
        return (*state.terminals, candidate), state.defects
    if existing == candidate:
        return state.terminals, state.defects
    defect = RequestDefect(
        request=candidate.request,
        code="conflicting_terminal",
        association=None,
        terminal=candidate,
        settlement=None,
    )
    return state.terminals, _append_unique(state.defects, defect)


def _accept_results(state: RequestState, event: AcceptResult) -> RequestState:
    if not isinstance(event.results, tuple) or any(not isinstance(item, AssociationResult) for item in event.results):
        reject(EffectCode.INVALID_TYPE)
    reservation = _dispatched_reservation(state, event.request)
    expected = reservation.associations
    observed = [item.association for item in event.results]
    defects = state.defects
    for association in set(observed):
        if observed.count(association) > 1:
            defects = _append_unique(
                defects,
                RequestDefect(
                    request=event.request,
                    code="duplicate_keyed_result",
                    association=association,
                    terminal=None,
                    settlement=None,
                ),
            )
        if not _association_in_scope(association, state.scope):
            defects = _append_unique(
                defects,
                RequestDefect(
                    request=event.request,
                    code="foreign_keyed_result",
                    association=association,
                    terminal=None,
                    settlement=None,
                ),
            )
        elif association not in expected:
            defects = _append_unique(
                defects,
                RequestDefect(
                    request=event.request,
                    code="extra_keyed_result",
                    association=association,
                    terminal=None,
                    settlement=None,
                ),
            )
    for association in expected - set(observed):
        defects = _append_unique(
            defects,
            RequestDefect(
                request=event.request,
                code="missing_keyed_result",
                association=association,
                terminal=None,
                settlement=None,
            ),
        )
    malformed = defects != state.defects
    terminal = RequestTerminalFact(
        request=event.request,
        category="inconsistent" if malformed else "success",
        failure=None,
        results=() if malformed else event.results,
    )
    interim = replace(state, defects=defects)
    terminals, defects = _terminal(interim, terminal)
    clear_remote = not any(item.request == event.request for item in state.terminals)
    return replace(
        state,
        terminals=terminals,
        defects=defects,
        local_in_flight=state.local_in_flight - {event.request},
        remote_outstanding=state.remote_outstanding - ({event.request} if clear_remote else set()),
    )


def _validate_followup(state: RequestState, event: Reserve) -> None:
    if event.purpose not in {"retry", "correction", "failover"}:
        return
    rejection = _followup_rejection(state, event.purpose, event.associations, event.policy)
    if rejection is not None:
        reject(rejection)


def _followup_rejection(
    state: RequestState,
    purpose: Literal["retry", "correction", "failover"],
    associations: frozenset[RequestAssociation],
    policy: PhysicalRequestPolicy,
) -> EffectCode | None:
    for association in associations:
        latest = next(
            (reservation for reservation in reversed(state.dispatches) if association in reservation.associations),
            None,
        )
        if latest is None:
            return EffectCode.MISSING_PREDECESSOR
        terminal = next((item for item in state.terminals if item.request == latest.request), None)
        if terminal is None:
            return EffectCode.MISSING_PREDECESSOR
        if terminal.failure is None:
            code = {
                "retry": EffectCode.INVALID_RETRY,
                "correction": EffectCode.INVALID_CORRECTION,
                "failover": EffectCode.INVALID_FAILOVER,
            }[purpose]
            return code
        failure = terminal.failure
        if purpose == "retry":
            if failure not in {"rejected_before_acceptance", "retryable", "transport_unknown"}:
                return EffectCode.INVALID_RETRY
            permitted = failure == "rejected_before_acceptance" and policy.replay in {
                "before_acceptance",
                "idempotent",
            }
            permitted = permitted or failure in {"retryable", "transport_unknown"} and policy.replay == "idempotent"
        elif purpose == "correction":
            if failure != "malformed_response":
                return EffectCode.INVALID_CORRECTION
            permitted = policy.replay == "idempotent"
        else:
            # Exact failover eligibility is admitted by the execution policy;
            # the reducer enforces predecessor and replay authority.
            if failure in {"malformed_response", "retryable", "transport_unknown"}:
                return EffectCode.INVALID_FAILOVER
            permitted = policy.replay == "idempotent" or (
                failure == "rejected_before_acceptance" and policy.replay == "before_acceptance"
            )
        if latest.policy != policy:
            return EffectCode.CROSS_POLICY
        if not permitted:
            return EffectCode.REPLAY_FORBIDDEN
    return None


def _dispatched_reservation(state: RequestState, request: PhysicalRequestId) -> RequestReservation:
    matches = [item for item in state.dispatches if item.request == request]
    if len(matches) != 1:
        reject(EffectCode.MISSING)
    return matches[0]


def _observe_settlement(state: RequestState, settlement: ExternalSettlement) -> RequestState:
    existing = next((item for item in state.settlements if item.request == settlement.request), None)
    if existing is None:
        settlements = (*state.settlements, settlement)
        defects = state.defects
        remote = state.remote_outstanding - ({settlement.request} if settlement.remote_stopped is True else set())
    elif existing == settlement:
        return state
    else:
        settlements = state.settlements
        defects = _append_unique(
            state.defects,
            RequestDefect(
                request=settlement.request,
                code="conflicting_settlement",
                association=None,
                terminal=None,
                settlement=settlement,
            ),
        )
        remote = state.remote_outstanding
    return replace(state, settlements=settlements, defects=defects, remote_outstanding=frozenset(remote))


def _append_unique(values: tuple[T, ...], value: T) -> tuple[T, ...]:
    return values if value in values else (*values, value)
