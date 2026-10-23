# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request policies, values, receipts and events."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Literal, TypeAlias, TypeVar

from anonymizer.engine.detection.artifacts import EntitySpansValue
from anonymizer.engine.graph_sdk._artifact_shape import ArtifactShape, content_bytes
from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    OpaqueIdentity,
    PrivateValue,
    reject,
    require_count,
    require_instance,
    require_literal,
    require_owner,
    require_text,
)
from anonymizer.engine.graph_sdk._graph_values import ArtifactRef, InvocationId, TaskAttemptId
from anonymizer.engine.graph_sdk._workflow_values import ArtifactType, OperationSpec
from anonymizer.engine.graph_sdk.rewrite_artifacts import REWRITE_TYPES, RewriteArtifactValue

RequestPurpose: TypeAlias = Literal["initial", "retry", "correction", "repair", "failover", "initial_binding"]


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

RuntimeCondition: TypeAlias = Literal[
    "result",
    "failure",
    "cancel_before_start",
    "cancel_after_start",
    "cancel_after_dispatch",
    "lost",
    "request_inconsistent",
    "budget_exhausted",
    "request_limit_exhausted",
    "artifact_limit_exhausted",
    "deadline_exhausted",
]

AssociationTerminal: TypeAlias = Literal[
    "success", "failure", "cancelled", "lost", "inconsistent", "publication_failed"
]


RequestDenialCategory: TypeAlias = Literal["budget_stopped", "request_limit_stopped"]


SettlementDisposition: TypeAlias = Literal["completed", "rejected", "stopped", "unknown"]


RequestDefectCode: TypeAlias = Literal[
    "conflicting_terminal",
    "conflicting_settlement",
    "duplicate_keyed_result",
    "missing_keyed_result",
    "extra_keyed_result",
    "foreign_keyed_result",
    "invalid_settlement",
]


_PURPOSES = frozenset({"initial", "retry", "correction", "repair", "failover", "initial_binding"})


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
        # Exact type: hashing and equality then run only on the engine-issued attempt token.
        if type(self.task) is not TaskAttemptId:
            reject(EffectCode.INVALID_TYPE)


_DECLARATION_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BindingDeclarationId(PrivateValue):
    _key: InitVar[object]
    binding: BindingId
    ordinal: int

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _DECLARATION_KEY)
        require_instance(self.binding, BindingId)
        require_count(self.ordinal)

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
    shape: ArtifactShape = field(init=False, compare=False)

    def __post_init__(self) -> None:
        require_text(self.text, empty=True)
        object.__setattr__(self, "shape", ArtifactShape(content_bytes(self.text)))


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
    shape: ArtifactShape = field(init=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or any(not isinstance(item, TextCollectionItem) for item in self.items):
            reject(EffectCode.INVALID_TYPE)
        keys = [(item.key, item.version) for item in self.items]
        if len(keys) != len(set(keys)):
            reject(EffectCode.DUPLICATE)
        if keys != sorted(keys):
            reject(EffectCode.INVALID_VALUE)
        object.__setattr__(
            self, "shape", ArtifactShape(sum(item.value.shape.content_bytes for item in self.items), len(self.items))
        )


ArtifactValue: TypeAlias = TextArtifactValue | TextCollectionValue | EntitySpansValue | RewriteArtifactValue


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
        require_instance(self.value, (TextArtifactValue, TextCollectionValue, EntitySpansValue, *REWRITE_TYPES))


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
class RequestAssociationSummary(PrivateValue):
    """Payload-free first acceptance; publication is resolved through occurrence facts."""

    association: RequestAssociation
    outcome: str | None
    category: AssociationTerminal
    failure: FailureClass | None
    condition: RuntimeCondition | None
    consumed_context_ports: frozenset[str]
    artifacts: tuple[ArtifactRef, ...] = ()

    def __post_init__(self) -> None:
        require_instance(self.association, (SemanticAssociation, BindingAssociation))
        require_literal(
            self.category,
            frozenset(
                {
                    "success",
                    "failure",
                    "cancelled",
                    "lost",
                    "inconsistent",
                    "publication_failed",
                }
            ),
        )
        if self.outcome is not None:
            require_text(self.outcome)
        if self.failure is not None:
            require_literal(self.failure, _FAILURES)
        if self.condition is not None:
            require_literal(
                self.condition,
                frozenset(
                    {
                        "result",
                        "failure",
                        "cancel_before_start",
                        "cancel_after_start",
                        "cancel_after_dispatch",
                        "lost",
                        "request_inconsistent",
                        "budget_exhausted",
                        "request_limit_exhausted",
                        "artifact_limit_exhausted",
                        "deadline_exhausted",
                    }
                ),
            )
        if type(self.consumed_context_ports) is not frozenset or any(
            type(port) is not str or not port for port in self.consumed_context_ports
        ):
            reject(EffectCode.INVALID_TYPE)
        if type(self.artifacts) is not tuple or any(type(item) is not ArtifactRef for item in self.artifacts):
            reject(EffectCode.INVALID_TYPE)
        if self.category == "success":
            if self.outcome is None or self.failure is not None or self.condition != "result":
                reject(EffectCode.CONTRADICTORY)
        elif self.outcome is not None or self.consumed_context_ports or self.artifacts:
            reject(EffectCode.CONTRADICTORY)
        if self.category == "failure" and (self.failure is None or self.condition != "failure"):
            reject(EffectCode.CONTRADICTORY)
        if self.category in {"cancelled", "lost", "inconsistent"} and (
            self.failure is not None or self.condition is not None
        ):
            reject(EffectCode.CONTRADICTORY)
        if self.artifacts and (
            not isinstance(self.association, SemanticAssociation)
            or any(item.invocation != self.association.task.activation.invocation for item in self.artifacts)
        ):
            reject(EffectCode.FOREIGN_OWNER)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestTerminalFact(PrivateValue):
    request: PhysicalRequestId
    category: RequestTerminal
    failure: FailureClass | None
    results: tuple[RequestAssociationSummary, ...]
    schema_version: Literal[2] = 2

    def __post_init__(self) -> None:
        require_instance(self.request, PhysicalRequestId)
        require_literal(self.category, frozenset({"success", "failure", "cancelled", "lost", "inconsistent"}))
        if self.failure is not None:
            require_literal(self.failure, _FAILURES)
        if type(self.schema_version) is not int or self.schema_version != 2:
            reject(EffectCode.INVALID_VALUE)
        if type(self.results) is not tuple or any(type(item) is not RequestAssociationSummary for item in self.results):
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
                    "invalid_settlement",
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
                self.code == "invalid_settlement"
                and self.association is None
                and self.terminal is None
                and self.settlement is None
            )
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


ChildSourceKind: TypeAlias = Literal["initial", "retry", "correction", "conversation_restart", "failover"]


class RequestChainId(OpaqueIdentity):
    """Graph-owner-issued identity; this does not register an activation."""

    __slots__ = ("scope",)
    scope: RequestScope

    @classmethod
    def new(cls, *, scope: RequestScope) -> RequestChainId:
        require_instance(scope, (InvocationRequestScope, BindingRequestScope))
        return cls._new(scope=scope)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ChildRoute(PrivateValue):
    name: str
    parse_attempts: int

    def __post_init__(self) -> None:
        require_text(self.name)
        require_count(self.parse_attempts, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestChain(PrivateValue):
    """Bounded, ordered fallback routes admitted by the request owner."""

    chain: RequestChainId
    associations: frozenset[RequestAssociation]
    policy: PhysicalRequestPolicy
    role: Literal["generation", "readiness"]
    routes: tuple[ChildRoute, ...]
    followups: frozenset[ChildSourceKind] = frozenset({"correction", "failover"})

    def __post_init__(self) -> None:
        require_instance(self.chain, RequestChainId)
        require_instance(self.policy, PhysicalRequestPolicy)
        require_literal(self.role, frozenset({"generation", "readiness"}))
        if type(self.associations) is not frozenset or not self.associations:
            reject(EffectCode.INVALID_TYPE)
        for association in self.associations:
            require_instance(association, (SemanticAssociation, BindingAssociation))
            if not _association_in_scope(association, self.chain.scope):
                reject(EffectCode.FOREIGN_OWNER)
        if (
            type(self.routes) is not tuple
            or not self.routes
            or any(type(route) is not ChildRoute for route in self.routes)
        ):
            reject(EffectCode.INVALID_TYPE)
        if len({route.name for route in self.routes}) != len(self.routes):
            reject(EffectCode.DUPLICATE)
        if type(self.followups) is not frozenset or not self.followups <= {
            "retry",
            "correction",
            "conversation_restart",
            "failover",
        }:
            reject(EffectCode.INVALID_VALUE)
        if sum(route.parse_attempts for route in self.routes) > self.policy.max_attempts:
            reject(EffectCode.LIMIT_EXCEEDED)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ChildEvidence(PrivateValue):
    chain: RequestChainId
    request: PhysicalRequestId
    kind: Literal["parser_rejected", "generation_exhausted"]

    def __post_init__(self) -> None:
        require_instance(self.chain, RequestChainId)
        require_instance(self.request, PhysicalRequestId)
        require_literal(self.kind, frozenset({"parser_rejected", "generation_exhausted"}))
        if self.request.scope != self.chain.scope:
            reject(EffectCode.FOREIGN_OWNER)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestChild(PrivateValue):
    chain: RequestChainId
    role: Literal["generation", "readiness"]
    route: str
    source_kind: ChildSourceKind
    predecessor: PhysicalRequestId | None = None
    evidence: ChildEvidence | None = None
    leaf: TaskAttemptId | None = None

    def __post_init__(self) -> None:
        require_instance(self.chain, RequestChainId)
        require_literal(self.role, frozenset({"generation", "readiness"}))
        require_text(self.route)
        require_literal(
            self.source_kind, frozenset({"initial", "retry", "correction", "conversation_restart", "failover"})
        )
        if self.leaf is not None:
            require_instance(self.leaf, TaskAttemptId)
            if not isinstance(self.chain.scope, InvocationRequestScope) or (
                self.leaf.activation.invocation != self.chain.scope.invocation
            ):
                reject(EffectCode.FOREIGN_OWNER)
        if self.predecessor is not None:
            require_instance(self.predecessor, PhysicalRequestId)
            if self.predecessor.scope != self.chain.scope:
                reject(EffectCode.FOREIGN_OWNER)
        if self.evidence is not None:
            require_instance(self.evidence, ChildEvidence)
            if self.evidence.chain != self.chain or self.evidence.request != self.predecessor:
                reject(EffectCode.FOREIGN_OWNER)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AcceptChildEvidence(PrivateValue):
    evidence: ChildEvidence


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestReservation(PrivateValue):
    request: PhysicalRequestId
    purpose: RequestPurpose
    associations: frozenset[RequestAssociation]
    policy: PhysicalRequestPolicy
    child: RequestChild | None = None

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
        if self.child is not None:
            require_instance(self.child, RequestChild)
            if self.child.chain.scope != self.request.scope:
                reject(EffectCode.FOREIGN_OWNER)
        if any(not _association_in_scope(item, self.request.scope) for item in self.associations):
            reject(EffectCode.FOREIGN_OWNER)


_BINDING_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestPolicyBinding(PrivateValue):
    _key: InitVar[object]
    association: RequestAssociation
    policies: frozenset[PhysicalRequestPolicy]

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _BINDING_KEY)
        # create() is the public verb: both fields are caller-supplied ingress.
        require_instance(self.association, (SemanticAssociation, BindingAssociation))
        if not isinstance(self.policies, frozenset) or any(
            not isinstance(item, PhysicalRequestPolicy) for item in self.policies
        ):
            reject(EffectCode.INVALID_TYPE)
        if not self.policies:
            reject(EffectCode.INVALID_VALUE)

    @classmethod
    def create(
        cls,
        *,
        association: RequestAssociation,
        policies: frozenset[PhysicalRequestPolicy],
    ) -> RequestPolicyBinding:
        return cls(_key=_BINDING_KEY, association=association, policies=policies)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class _RequestLedger(PrivateValue):
    """Request-accounting facts shared by the live state and its receipt."""

    scope: RequestScope
    budget_limit: int | None
    policies: frozenset[PhysicalRequestPolicy]
    bindings: frozenset[RequestPolicyBinding]
    dispatches: tuple[RequestReservation, ...]
    denials: tuple[RequestDenialFact, ...]
    terminals: tuple[RequestTerminalFact, ...]
    settlements: tuple[ExternalSettlement, ...]
    defects: tuple[RequestDefect, ...]
    local_in_flight: frozenset[PhysicalRequestId]
    remote_outstanding: frozenset[PhysicalRequestId]
    cancel_requested: frozenset[PhysicalRequestId]
    chains: tuple[RequestChain, ...] = ()
    child_evidence: tuple[ChildEvidence, ...] = ()

    @property
    def dispatched_count(self) -> int:
        return len(self.dispatches)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestState(_RequestLedger):
    reserved: frozenset[RequestReservation]
    dispatched: frozenset[PhysicalRequestId]
    cancelled: bool


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestReceipt(_RequestLedger):
    pass


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Reserve(PrivateValue):
    request: PhysicalRequestId
    purpose: RequestPurpose
    associations: frozenset[RequestAssociation]
    policy: PhysicalRequestPolicy
    child: RequestChild | None = None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class Dispatch(PrivateValue):
    request: PhysicalRequestId


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AcceptResult(PrivateValue):
    request: PhysicalRequestId
    results: tuple[AssociationResult | RequestAssociationSummary, ...]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AcceptFailure(PrivateValue):
    request: PhysicalRequestId
    failure: FailureClass


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RejectSettlement(PrivateValue):
    request: PhysicalRequestId


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
    AcceptChildEvidence
    | Reserve
    | Dispatch
    | AcceptResult
    | AcceptFailure
    | RejectSettlement
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
