# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request policies, values, receipts and events."""

from __future__ import annotations

from dataclasses import dataclass
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
        if isinstance(self.association, BindingAssociation) and (
            self.outcome != "retrieved" or self.outputs or self.consumed_context_ports
        ):
            reject(EffectCode.CONTRADICTORY)


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
    cancel_requested: frozenset[PhysicalRequestId]
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
    cancel_requested: frozenset[PhysicalRequestId]


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


def _association_in_scope(association: RequestAssociation, scope: RequestScope) -> bool:
    if isinstance(association, SemanticAssociation) and isinstance(scope, InvocationRequestScope):
        return association.task.activation.invocation == scope.invocation
    if isinstance(association, BindingAssociation) and isinstance(scope, BindingRequestScope):
        return association.declaration.binding == scope.binding
    return False
