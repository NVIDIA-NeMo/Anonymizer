# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure request accounting state transitions."""

from __future__ import annotations

from dataclasses import fields, replace
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
    AcceptChildEvidence,
    AcceptFailure,
    AcceptResult,
    AssociationResult,
    BindingAssociation,
    BindingRequestScope,
    ChildEvidence,
    Dispatch,
    ExternalSettlement,
    InvocationRequestScope,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    RejectSettlement,
    RequestAssociation,
    RequestAssociationSummary,
    RequestCancel,
    RequestChain,
    RequestChild,
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
    UnknownUsage,
    _association_in_scope,
    _RequestLedger,
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
        budget_limit=hard_limit,
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
    return RequestReceipt(**{item.name: getattr(state, item.name) for item in fields(_RequestLedger)})


def advance_requests(*, state: RequestState, event: RequestEvent) -> RequestState:
    """Apply one bounded request-accounting event."""
    require_instance(state, RequestState)
    require_instance(
        event,
        (
            AcceptChildEvidence,
            Reserve,
            Dispatch,
            AcceptResult,
            AcceptFailure,
            RequestCancel,
            ScopeCancel,
            StopAcknowledged,
            MarkLost,
            ObserveSettlement,
            RejectSettlement,
        ),
    )
    if isinstance(event, AcceptChildEvidence):
        return _accept_child_evidence(state, event.evidence)
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
        if reservation.child is not None:
            _validate_child(state, reservation)
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
    if request not in state.dispatched:
        reject(EffectCode.MISSING)
    if isinstance(event, ObserveSettlement):
        return _observe_settlement(state, event.settlement)
    if isinstance(event, RejectSettlement):
        interim = _observe_settlement(
            state,
            ExternalSettlement(
                request=request,
                disposition="unknown",
                usage=UnknownUsage(),
                remote_stopped=None,
            ),
        )
        defect = RequestDefect(
            request=request, code="invalid_settlement", association=None, terminal=None, settlement=None
        )
        return replace(interim, defects=_append_unique(interim.defects, defect))
    if isinstance(event, AcceptFailure):
        require_literal(event.failure, _FAILURES)
        terminal = RequestTerminalFact(request=request, category="failure", failure=event.failure, results=())
        terminals, defects = _terminal(state, terminal)
        clear_remote = (
            not any(item.request == request for item in state.terminals)
            and not any(item.request == request and item.code == "invalid_settlement" for item in state.defects)
            and not any(item.request == request and item.remote_stopped is not True for item in state.settlements)
        )
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
    if event.child is None:
        if any(chain.associations & event.associations for chain in state.chains):
            reject(EffectCode.CROSS_POLICY)
        _validate_followup(state, event)
    else:
        _validate_child(state, event)
    attempts = (*state.dispatches, *state.reserved) if event.child is not None else state.dispatches
    exhausted = {
        item
        for item in eligible
        if sum(item in reservation.associations for reservation in attempts) >= event.policy.max_attempts
    }
    denials = state.denials
    if exhausted:
        if event.child is not None:
            exhausted = eligible.copy()
        denials += (
            RequestDenialFact(
                request=event.request,
                associations=frozenset(exhausted),
                category="request_limit_stopped",
            ),
        )
        eligible -= exhausted
    if (
        eligible
        and state.budget_limit is not None
        and len(state.dispatches) + len(state.reserved) >= state.budget_limit
    ):
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
        child=event.child,
    )
    return replace(state, reserved=state.reserved | {reservation}, denials=denials)


def bind_request_chain(*, state: RequestState, chain: RequestChain) -> RequestState:
    """Bind a child contract; runnable delegation/activation admission is separate."""
    require_instance(state, RequestState)
    require_instance(chain, RequestChain)
    if state.cancelled:
        reject(EffectCode.REQUEST_CAUSALITY)
    if chain.chain.scope != state.scope:
        reject(EffectCode.FOREIGN_OWNER)
    if any(item.chain == chain.chain for item in state.chains):
        reject(EffectCode.DUPLICATE)
    if any(
        not any(binding.association == association and chain.policy in binding.policies for binding in state.bindings)
        for association in chain.associations
    ):
        reject(EffectCode.CROSS_POLICY)
    if any(item.associations & chain.associations for item in (*state.dispatches, *state.reserved)):
        reject(EffectCode.REQUEST_CAUSALITY)
    if any(
        sum(association in item.associations for item in state.chains) >= chain.policy.max_attempts
        for association in chain.associations
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    return replace(state, chains=(*state.chains, chain))


def _child_history(state: RequestState, child: RequestChild) -> tuple[RequestReservation, ...]:
    return tuple(item for item in state.dispatches if item.child is not None and item.child.chain == child.chain)


def _validate_child(state: RequestState, event: Reserve | RequestReservation) -> None:
    child = event.child
    if state.cancelled:
        reject(EffectCode.REQUEST_CAUSALITY)
    require_instance(child, RequestChild)
    assert child is not None
    contract = next((item for item in state.chains if item.chain == child.chain), None)
    if contract is None or child.chain.scope != state.scope:
        reject(EffectCode.FOREIGN_OWNER)
    if contract.policy != event.policy or contract.associations != event.associations or contract.role != child.role:
        reject(EffectCode.CROSS_POLICY)
    purpose = "correction" if child.source_kind == "conversation_restart" else child.source_kind
    if purpose != event.purpose:
        reject(EffectCode.CONTRADICTORY)
    if any(
        item.request != event.request and item.child is not None and item.child.chain == child.chain
        for item in state.reserved
    ):
        reject(EffectCode.PENDING_LIMIT)
    if event.request in {item.request for item in state.terminals}:
        reject(EffectCode.DUPLICATE)
    if child.source_kind != "initial" and child.source_kind not in contract.followups:
        reject(EffectCode.UNSUPPORTED)
    history = _child_history(state, child)
    if not history:
        if (
            purpose != "initial"
            or child.predecessor is not None
            or child.evidence is not None
            or child.route != contract.routes[0].name
        ):
            reject(EffectCode.MISSING_PREDECESSOR)
        return
    prior = history[-1]
    assert prior.child is not None
    if purpose == "initial" or child.predecessor != prior.request:
        reject(EffectCode.REQUEST_CAUSALITY)
    terminal = next((item for item in state.terminals if item.request == prior.request), None)
    if terminal is None:
        reject(EffectCode.MISSING_PREDECESSOR)
    if any(item.request == prior.request for item in state.defects):
        reject(EffectCode.CONTRADICTORY)
    if purpose == "failover":
        names = tuple(route.name for route in contract.routes)
        index = names.index(prior.child.route) + 1
        if index >= len(names) or child.route != names[index]:
            reject(EffectCode.INVALID_FAILOVER)
        _require_child_evidence(state, child, "generation_exhausted")
        if event.policy.replay != "idempotent":
            reject(EffectCode.REPLAY_FORBIDDEN)
    else:
        if child.route != prior.child.route:
            reject(EffectCode.CROSS_POLICY)
        if purpose == "correction":
            _require_child_evidence(state, child, "parser_rejected")
            route = next(route for route in contract.routes if route.name == child.route)
            parsed = sum(
                item.kind == "parser_rejected"
                and item.chain == child.chain
                and any(
                    row.request == item.request and row.child is not None and row.child.route == child.route
                    for row in history
                )
                for item in state.child_evidence
            )
            if parsed >= route.parse_attempts:
                reject(EffectCode.LIMIT_EXCEEDED)
        elif child.evidence is not None:
            reject(EffectCode.CONTRADICTORY)
        _validate_followup(replace(state, dispatches=history), event)


def _require_child_evidence(state: RequestState, child: RequestChild, kind: str) -> None:
    if child.evidence is None or child.evidence.kind != kind or child.evidence not in state.child_evidence:
        reject(EffectCode.MISSING_PREDECESSOR)
    _validate_child_evidence(state, child.evidence)


def _accept_child_evidence(state: RequestState, evidence: ChildEvidence) -> RequestState:
    _validate_child_evidence(state, evidence)
    if evidence in state.child_evidence:
        reject(EffectCode.DUPLICATE)
    history = tuple(item for item in state.dispatches if item.child is not None and item.child.chain == evidence.chain)
    if history[-1].request != evidence.request:
        reject(EffectCode.REQUEST_CAUSALITY)
    return replace(state, child_evidence=(*state.child_evidence, evidence))


def _validate_child_evidence(state: RequestState, evidence: ChildEvidence) -> None:
    """Recheck historical proof support without changing its accepted facts."""
    require_instance(evidence, ChildEvidence)
    if evidence.chain.scope != state.scope:
        reject(EffectCode.FOREIGN_OWNER)
    if any(item.request == evidence.request for item in state.defects):
        reject(EffectCode.CONTRADICTORY)
    history = tuple(item for item in state.dispatches if item.child is not None and item.child.chain == evidence.chain)
    prior = next((item for item in history if item.request == evidence.request), None)
    if prior is None:
        reject(EffectCode.REQUEST_CAUSALITY)
    assert prior.child is not None
    terminal = next((item for item in state.terminals if item.request == evidence.request), None)
    if terminal is None or terminal.category != "failure" or terminal.failure != "malformed_response":
        reject(EffectCode.INVALID_CORRECTION)
    if not any(item.request == evidence.request and item.disposition == "completed" for item in state.settlements):
        reject(EffectCode.INVALID_SETTLEMENT)
    if evidence.kind == "generation_exhausted":
        contract = next(item for item in state.chains if item.chain == evidence.chain)
        route = next(item for item in contract.routes if item.name == prior.child.route)
        failures = {item.request for item in history if item.child is not None and item.child.route == route.name}
        supports = tuple(
            item
            for item in state.child_evidence
            if item.chain == evidence.chain and item.kind == "parser_rejected" and item.request in failures
        )
        if evidence.request not in {item.request for item in supports} or len(supports) != route.parse_attempts:
            reject(EffectCode.INVALID_FAILOVER)
        for support in supports:
            _validate_child_evidence(state, support)


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
    if type(event.results) is not tuple or any(
        type(item) not in {AssociationResult, RequestAssociationSummary} for item in event.results
    ):
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
        results=() if malformed else tuple(_acceptance_summary(item) for item in event.results),
    )
    interim = replace(state, defects=defects)
    terminals, defects = _terminal(interim, terminal)
    clear_remote = (
        not any(item.request == event.request for item in state.terminals)
        and not any(item.request == event.request and item.code == "invalid_settlement" for item in state.defects)
        and not any(item.request == event.request and item.remote_stopped is not True for item in state.settlements)
    )
    return replace(
        state,
        terminals=terminals,
        defects=defects,
        local_in_flight=state.local_in_flight - {event.request},
        remote_outstanding=state.remote_outstanding - ({event.request} if clear_remote else set()),
    )


def _acceptance_summary(result: AssociationResult | RequestAssociationSummary) -> RequestAssociationSummary:
    if isinstance(result, RequestAssociationSummary):
        # The physical reducer has no publication authority. Exact occurrence joins
        # resolve artifacts after publication, without rewriting this first terminal.
        if result.artifacts:
            reject(EffectCode.FOREIGN_OWNER)
        return result
    return RequestAssociationSummary(
        association=result.association,
        outcome=result.outcome,
        category="success",
        failure=None,
        condition="result",
        consumed_context_ports=result.consumed_context_ports,
    )


def _validate_followup(state: RequestState, event: Reserve | RequestReservation) -> None:
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
        summary = next((item for item in terminal.results if item.association == association), None)
        failure = terminal.failure if summary is None else summary.failure
        if failure is None:
            code = {
                "retry": EffectCode.INVALID_RETRY,
                "correction": EffectCode.INVALID_CORRECTION,
                "failover": EffectCode.INVALID_FAILOVER,
            }[purpose]
            return code
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
