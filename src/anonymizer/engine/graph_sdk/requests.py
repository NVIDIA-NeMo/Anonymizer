# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure physical-request accounting for graph execution and context binding."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal, TypeAlias, TypeVar

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    OpaqueIdentity,
    PrivateValue,
    reject,
    require_count,
    require_instance,
    require_literal,
    require_text,
)
from anonymizer.graph._values import ArtifactRef, InvocationId, TaskAttemptId
from anonymizer.graph.workflow import ArtifactType, OperationSpec

RequestPurpose: TypeAlias = Literal[
    "initial", "retry", "correction", "repair", "failover", "initial_binding", "adaptive_retrieval"
]
ReplaySafety: TypeAlias = Literal["never", "before_acceptance", "idempotent"]
FailureClass: TypeAlias = Literal[
    "rejected_before_acceptance",
    "retryable",
    "malformed_response",
    "permanent",
    "transport_unknown",
    "implementation_exception",
]
RequestTerminal: TypeAlias = Literal["success", "failure", "cancelled", "lost", "inconsistent"]
RequestDenialCategory: TypeAlias = Literal["budget_stopped", "request_limit_stopped"]
SettlementDisposition: TypeAlias = Literal["completed", "rejected", "stopped", "unknown"]
RequestDefectCode: TypeAlias = Literal[
    "conflicting_terminal",
    "conflicting_settlement",
    "duplicate_keyed_result",
    "missing_keyed_result",
    "extra_keyed_result",
    "foreign_keyed_result",
]

_PURPOSES = frozenset({"initial", "retry", "correction", "repair", "failover", "initial_binding", "adaptive_retrieval"})
_FAILURES = frozenset(
    {
        "rejected_before_acceptance",
        "retryable",
        "malformed_response",
        "permanent",
        "transport_unknown",
        "implementation_exception",
    }
)
T = TypeVar("T")


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PhysicalRequestPolicy(PrivateValue):
    visibility: str
    pre_dispatch_control: str
    retry_owner: str
    replay: ReplaySafety
    max_attempts: int

    def __post_init__(self) -> None:
        require_literal(self.visibility, frozenset({"none", "before_dispatch", "dispatch_and_settlement"}))
        require_literal(self.pre_dispatch_control, frozenset({"none", "executor"}))
        require_literal(self.retry_owner, frozenset({"none", "executor", "implementation"}))
        require_literal(self.replay, frozenset({"never", "before_acceptance", "idempotent"}))
        require_count(self.max_attempts)
        if self.retry_owner == "implementation":
            reject(EffectCode.CONTRADICTORY)


class BindingId(OpaqueIdentity):
    __slots__ = ()

    @classmethod
    def new(cls) -> BindingId:
        return cls._new()  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class InvocationRequestScope(PrivateValue):
    invocation: InvocationId

    def __post_init__(self) -> None:
        require_instance(self.invocation, InvocationId)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BindingRequestScope(PrivateValue):
    binding: BindingId

    def __post_init__(self) -> None:
        require_instance(self.binding, BindingId)


RequestScope: TypeAlias = InvocationRequestScope | BindingRequestScope


class PhysicalRequestId(OpaqueIdentity):
    __slots__ = ("scope",)
    scope: RequestScope

    @classmethod
    def new(cls, *, scope: RequestScope) -> PhysicalRequestId:
        require_instance(scope, (InvocationRequestScope, BindingRequestScope))
        return cls._new(scope=scope)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SemanticAssociation(PrivateValue):
    task: TaskAttemptId

    def __post_init__(self) -> None:
        require_instance(self.task, TaskAttemptId)


_DECLARATION_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class BindingDeclarationId(PrivateValue):
    binding: BindingId
    ordinal: int

    def __init__(self, *, _key: object, binding: BindingId, ordinal: int) -> None:
        if _key is not _DECLARATION_KEY:
            raise TypeError("binding declaration identities are created by their binding owner")
        require_instance(binding, BindingId)
        require_count(ordinal)
        object.__setattr__(self, "binding", binding)
        object.__setattr__(self, "ordinal", ordinal)

    @classmethod
    def new(cls, *, binding: BindingId, ordinal: int) -> BindingDeclarationId:
        return cls(_key=_DECLARATION_KEY, binding=binding, ordinal=ordinal)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BindingAssociation(PrivateValue):
    declaration: BindingDeclarationId

    def __post_init__(self) -> None:
        require_instance(self.declaration, BindingDeclarationId)


RequestAssociation: TypeAlias = SemanticAssociation | BindingAssociation


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExactUsage(PrivateValue):
    input_units: int
    output_units: int

    def __post_init__(self) -> None:
        require_count(self.input_units)
        require_count(self.output_units)


@dataclass(frozen=True, slots=True, repr=False)
class UnknownUsage(PrivateValue):
    pass


Usage: TypeAlias = ExactUsage | UnknownUsage


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TextArtifactValue(PrivateValue):
    text: str

    def __post_init__(self) -> None:
        require_text(self.text, empty=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TextCollectionItem(PrivateValue):
    key: int
    version: int
    value: TextArtifactValue

    def __post_init__(self) -> None:
        require_count(self.key)
        require_count(self.version, positive=True)
        require_instance(self.value, TextArtifactValue)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TextCollectionValue(PrivateValue):
    items: tuple[TextCollectionItem, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or any(not isinstance(item, TextCollectionItem) for item in self.items):
            reject(EffectCode.INVALID_TYPE)
        keys = [(item.key, item.version) for item in self.items]
        if len(keys) != len(set(keys)):
            reject(EffectCode.DUPLICATE)
        if keys != sorted(keys):
            reject(EffectCode.INVALID_VALUE)


ArtifactValue: TypeAlias = TextArtifactValue | TextCollectionValue


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PortArtifact(PrivateValue):
    port: str
    artifact_type: ArtifactType
    artifact: ArtifactRef | None
    value: ArtifactValue

    def __post_init__(self) -> None:
        require_text(self.port)
        require_instance(self.artifact_type, ArtifactType)
        if self.artifact is not None:
            require_instance(self.artifact, ArtifactRef)
        require_instance(self.value, (TextArtifactValue, TextCollectionValue))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExternalSettlement(PrivateValue):
    request: PhysicalRequestId
    disposition: SettlementDisposition
    usage: Usage
    remote_stopped: bool | None

    def __post_init__(self) -> None:
        require_instance(self.request, PhysicalRequestId)
        require_literal(self.disposition, frozenset({"completed", "rejected", "stopped", "unknown"}))
        require_instance(self.usage, (ExactUsage, UnknownUsage))
        if self.remote_stopped is not None and not isinstance(self.remote_stopped, bool):
            reject(EffectCode.INVALID_SETTLEMENT)
        if (self.disposition == "unknown") != (self.remote_stopped is not True):
            reject(EffectCode.INVALID_SETTLEMENT)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AssociationResult(PrivateValue):
    association: RequestAssociation
    outcome: str
    outputs: tuple[PortArtifact, ...]
    consumed_context_ports: frozenset[str]

    def __post_init__(self) -> None:
        require_instance(self.association, (SemanticAssociation, BindingAssociation))
        require_text(self.outcome)
        if not isinstance(self.outputs, tuple) or any(not isinstance(item, PortArtifact) for item in self.outputs):
            reject(EffectCode.INVALID_TYPE)
        if not isinstance(self.consumed_context_ports, frozenset) or any(
            not isinstance(item, str) for item in self.consumed_context_ports
        ):
            reject(EffectCode.INVALID_TYPE)
        if any(not item for item in self.consumed_context_ports):
            reject(EffectCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AssociationInput(PrivateValue):
    association: RequestAssociation
    inputs: tuple[PortArtifact, ...]

    def __post_init__(self) -> None:
        require_instance(self.association, (SemanticAssociation, BindingAssociation))
        if not isinstance(self.inputs, tuple) or any(not isinstance(item, PortArtifact) for item in self.inputs):
            reject(EffectCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DispatchEnvelope(PrivateValue):
    request: PhysicalRequestId
    purpose: RequestPurpose
    operation: OperationSpec
    associations: tuple[AssociationInput, ...]

    def __post_init__(self) -> None:
        require_instance(self.request, PhysicalRequestId)
        require_literal(self.purpose, _PURPOSES)
        require_instance(self.operation, OperationSpec)
        if not isinstance(self.associations, tuple) or any(
            not isinstance(item, AssociationInput) for item in self.associations
        ):
            reject(EffectCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TransportSuccess(PrivateValue):
    results: tuple[AssociationResult, ...]
    settlement: ExternalSettlement | None

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple) or any(not isinstance(item, AssociationResult) for item in self.results):
            reject(EffectCode.INVALID_TYPE)
        if self.settlement is not None:
            require_instance(self.settlement, ExternalSettlement)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TransportFailure(PrivateValue):
    failure: FailureClass
    settlement: ExternalSettlement | None

    def __post_init__(self) -> None:
        require_literal(self.failure, _FAILURES)
        if self.settlement is not None:
            require_instance(self.settlement, ExternalSettlement)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TransportLost(PrivateValue):
    settlement: ExternalSettlement | None

    def __post_init__(self) -> None:
        if self.settlement is not None:
            require_instance(self.settlement, ExternalSettlement)


TransportResult: TypeAlias = TransportSuccess | TransportFailure | TransportLost


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StopConfirmed(PrivateValue):
    usage: Usage

    def __post_init__(self) -> None:
        require_instance(self.usage, (ExactUsage, UnknownUsage))


@dataclass(frozen=True, slots=True, repr=False)
class StopUnknown(PrivateValue):
    pass


StopResult: TypeAlias = StopConfirmed | StopUnknown


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestTerminalFact(PrivateValue):
    request: PhysicalRequestId
    category: RequestTerminal
    failure: FailureClass | None
    results: tuple[AssociationResult, ...]

    def __post_init__(self) -> None:
        require_instance(self.request, PhysicalRequestId)
        require_literal(self.category, frozenset({"success", "failure", "cancelled", "lost", "inconsistent"}))
        if self.failure is not None:
            require_literal(self.failure, _FAILURES)
        if not isinstance(self.results, tuple) or any(not isinstance(item, AssociationResult) for item in self.results):
            reject(EffectCode.INVALID_TYPE)
        if self.category == "success" and (self.failure is not None or not self.results):
            reject(EffectCode.CONTRADICTORY)
        if self.category == "failure" and self.failure is None:
            reject(EffectCode.CONTRADICTORY)
        if self.category not in {"success", "failure"} and (self.failure is not None or self.results):
            reject(EffectCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestDefect(PrivateValue):
    request: PhysicalRequestId
    code: RequestDefectCode
    association: RequestAssociation | None
    terminal: RequestTerminalFact | None
    settlement: ExternalSettlement | None

    def __post_init__(self) -> None:
        require_instance(self.request, PhysicalRequestId)
        require_literal(
            self.code,
            frozenset(
                {
                    "conflicting_terminal",
                    "conflicting_settlement",
                    "duplicate_keyed_result",
                    "missing_keyed_result",
                    "extra_keyed_result",
                    "foreign_keyed_result",
                }
            ),
        )
        if self.association is not None:
            require_instance(self.association, (SemanticAssociation, BindingAssociation))
        if self.terminal is not None:
            require_instance(self.terminal, RequestTerminalFact)
        if self.settlement is not None:
            require_instance(self.settlement, ExternalSettlement)
        keyed = self.code.endswith("keyed_result")
        valid = (
            (keyed and self.association is not None and self.terminal is None and self.settlement is None)
            or (
                self.code == "conflicting_terminal"
                and self.association is None
                and self.terminal is not None
                and self.settlement is None
            )
            or (
                self.code == "conflicting_settlement"
                and self.association is None
                and self.terminal is None
                and self.settlement is not None
            )
        )
        if not valid:
            reject(EffectCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestDenialFact(PrivateValue):
    request: PhysicalRequestId
    associations: frozenset[RequestAssociation]
    category: RequestDenialCategory

    def __post_init__(self) -> None:
        require_instance(self.request, PhysicalRequestId)
        if not isinstance(self.associations, frozenset) or any(
            not isinstance(item, (SemanticAssociation, BindingAssociation)) for item in self.associations
        ):
            reject(EffectCode.INVALID_TYPE)
        if not self.associations:
            reject(EffectCode.INVALID_VALUE)
        require_literal(self.category, frozenset({"budget_stopped", "request_limit_stopped"}))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestReservation(PrivateValue):
    request: PhysicalRequestId
    purpose: RequestPurpose
    associations: frozenset[RequestAssociation]
    policy: PhysicalRequestPolicy

    def __post_init__(self) -> None:
        require_instance(self.request, PhysicalRequestId)
        require_literal(self.purpose, _PURPOSES)
        if not isinstance(self.associations, frozenset) or any(
            not isinstance(item, (SemanticAssociation, BindingAssociation)) for item in self.associations
        ):
            reject(EffectCode.INVALID_TYPE)
        if not self.associations:
            reject(EffectCode.INVALID_VALUE)
        require_instance(self.policy, PhysicalRequestPolicy)
        if any(not _association_in_scope(item, self.request.scope) for item in self.associations):
            reject(EffectCode.FOREIGN_OWNER)


_BINDING_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class RequestPolicyBinding(PrivateValue):
    association: RequestAssociation
    policies: frozenset[PhysicalRequestPolicy]

    def __init__(
        self,
        *,
        _key: object,
        association: RequestAssociation,
        policies: frozenset[PhysicalRequestPolicy],
    ) -> None:
        if _key is not _BINDING_KEY:
            raise TypeError("request policy bindings must be created by their owner")
        require_instance(association, (SemanticAssociation, BindingAssociation))
        if not isinstance(policies, frozenset) or any(not isinstance(item, PhysicalRequestPolicy) for item in policies):
            reject(EffectCode.INVALID_TYPE)
        if not policies:
            reject(EffectCode.INVALID_VALUE)
        object.__setattr__(self, "association", association)
        object.__setattr__(self, "policies", policies)

    @classmethod
    def create(
        cls,
        *,
        association: RequestAssociation,
        policies: frozenset[PhysicalRequestPolicy],
    ) -> RequestPolicyBinding:
        return cls(_key=_BINDING_KEY, association=association, policies=policies)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestState(PrivateValue):
    scope: RequestScope
    hard_limit: int | None
    policies: frozenset[PhysicalRequestPolicy]
    bindings: frozenset[RequestPolicyBinding]
    reserved: frozenset[RequestReservation]
    dispatched: frozenset[PhysicalRequestId]
    dispatches: tuple[RequestReservation, ...]
    denials: tuple[RequestDenialFact, ...]
    terminals: tuple[RequestTerminalFact, ...]
    settlements: tuple[ExternalSettlement, ...]
    defects: tuple[RequestDefect, ...]
    local_in_flight: frozenset[PhysicalRequestId]
    remote_outstanding: frozenset[PhysicalRequestId]
    cancelled: bool


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestReceipt(PrivateValue):
    scope: RequestScope
    budget_limit: int | None
    policies: frozenset[PhysicalRequestPolicy]
    bindings: frozenset[RequestPolicyBinding]
    dispatched_count: int
    dispatches: tuple[RequestReservation, ...]
    denials: tuple[RequestDenialFact, ...]
    terminals: tuple[RequestTerminalFact, ...]
    settlements: tuple[ExternalSettlement, ...]
    defects: tuple[RequestDefect, ...]
    local_in_flight: frozenset[PhysicalRequestId]
    remote_outstanding: frozenset[PhysicalRequestId]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Reserve(PrivateValue):
    request: PhysicalRequestId
    purpose: RequestPurpose
    associations: frozenset[RequestAssociation]
    policy: PhysicalRequestPolicy


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Dispatch(PrivateValue):
    request: PhysicalRequestId


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AcceptResult(PrivateValue):
    request: PhysicalRequestId
    results: tuple[AssociationResult, ...]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AcceptFailure(PrivateValue):
    request: PhysicalRequestId
    failure: FailureClass


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestCancel(PrivateValue):
    request: PhysicalRequestId


@dataclass(frozen=True, slots=True, repr=False)
class ScopeCancel(PrivateValue):
    pass


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StopAcknowledged(PrivateValue):
    request: PhysicalRequestId
    usage: Usage

    def __post_init__(self) -> None:
        require_instance(self.request, PhysicalRequestId)
        if not isinstance(self.usage, (ExactUsage, UnknownUsage)):
            reject(EffectCode.INVALID_USAGE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MarkLost(PrivateValue):
    request: PhysicalRequestId


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObserveSettlement(PrivateValue):
    settlement: ExternalSettlement


RequestEvent: TypeAlias = (
    Reserve
    | Dispatch
    | AcceptResult
    | AcceptFailure
    | RequestCancel
    | ScopeCancel
    | StopAcknowledged
    | MarkLost
    | ObserveSettlement
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
        return state
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


def _association_in_scope(association: RequestAssociation, scope: RequestScope) -> bool:
    if isinstance(association, SemanticAssociation) and isinstance(scope, InvocationRequestScope):
        return association.task.activation.invocation == scope.invocation
    if isinstance(association, BindingAssociation) and isinstance(scope, BindingRequestScope):
        return association.declaration.binding == scope.binding
    return False
