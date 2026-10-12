# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution declarations and sealed result values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    EffectRejected,
    OpaqueIdentity,
    PrivateValue,
    reject,
    require_count,
    require_instance,
    require_literal,
    require_text,
)
from anonymizer.engine.graph_sdk._ingress_values import IngressLimits
from anonymizer.engine.graph_sdk._workflow import reachable_workflows
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationCapability,
    ImplementationRef,
)
from anonymizer.engine.graph_sdk.context import (
    AdmittedContextPlan,
    BindingArtifactRef,
    ContextResource,
)
from anonymizer.engine.graph_sdk.observations import NodeMeasurement
from anonymizer.engine.graph_sdk.preparation import StateRevisionView
from anonymizer.engine.graph_sdk.records import (
    AbsenceRef,
    CandidateRef,
    CanonicalRecord,
)
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    BindingDeclarationId,
    ChildEvidence,
    ChildRoute,
    ChildSourceKind,
    DispatchEnvelope,
    FailureClass,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    RequestReceipt,
    RuntimeCondition,
    SemanticAssociation,
    StopResult,
    TransportResult,
)
from anonymizer.engine.graph_sdk.resources import (
    CleanupAssociation,
    CleanupFact,
    ResourceLease,
    SafeDetachment,
)
from anonymizer.graph._values import (
    ActivationKey,
    ArtifactRef,
    DatumId,
    InvocationId,
)
from anonymizer.graph.activation import (
    ActivationState,
)
from anonymizer.graph.workflow import (
    ArtifactType,
    ContextInputRef,
    Node,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationSpec,
    OutcomeClass,
    OutcomeSpec,
    WorkflowId,
    WorkflowInputRef,
)

ExecutionKind: TypeAlias = Literal["local", "external", "decision", "delegated"]


ArtifactRole: TypeAlias = Literal["artifact", "candidate", "decision", "evidence"]


AssessmentStatus: TypeAlias = Literal["satisfied", "unsatisfied", "unknown"]


class ExecutionRejected(EffectRejected):
    """Pure execution admission or pre-effect rejection."""


class NestedEventLoopError(RuntimeError):
    """Raised when a synchronous execution is requested from an event loop."""


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RuntimeOutcome(PrivateValue):
    condition: RuntimeCondition
    reported_outcome: str | None
    failure: FailureClass | None
    outcome: str | None
    category: OutcomeClass

    def __post_init__(self) -> None:
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
        if self.reported_outcome is not None:
            require_text(self.reported_outcome)
        if self.failure is not None:
            require_literal(
                self.failure,
                frozenset(
                    {
                        "rejected_before_acceptance",
                        "retryable",
                        "malformed_response",
                        "permanent",
                        "transport_unknown",
                        "implementation_exception",
                    }
                ),
            )
        if self.outcome is not None:
            require_text(self.outcome)
        require_literal(
            self.category,
            frozenset({"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}),
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DecisionOutcome(PrivateValue):
    decision: str
    outcome: str

    def __post_init__(self) -> None:
        require_text(self.decision)
        require_text(self.outcome)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DecisionDeclaration(PrivateValue):
    node: NodeId
    artifact_port: str
    outcomes: tuple[DecisionOutcome, ...]
    max_lifetime_ns: int

    def __post_init__(self) -> None:
        require_instance(self.node, NodeId)
        require_text(self.artifact_port)
        if not isinstance(self.outcomes, tuple) or any(not isinstance(item, DecisionOutcome) for item in self.outcomes):
            reject(EffectCode.INVALID_TYPE)
        decisions = [item.decision for item in self.outcomes]
        if not decisions or len(decisions) != len(set(decisions)):
            reject(EffectCode.DUPLICATE)
        require_count(self.max_lifetime_ns, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MapExpansionDecl(PrivateValue):
    expander: NodeId
    outcome: str
    membership_port: str
    item_type: ArtifactType

    def __post_init__(self) -> None:
        require_instance(self.expander, NodeId)
        require_text(self.outcome)
        require_text(self.membership_port)
        require_instance(self.item_type, ArtifactType)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExecutionImplementation(PrivateValue):
    implementation: ImplementationRef
    configuration: FrozenConfig
    capability: ImplementationCapability
    request: PhysicalRequestPolicy | None

    def __post_init__(self) -> None:
        require_instance(self.implementation, ImplementationRef)
        require_instance(self.configuration, FrozenConfig)
        require_instance(self.capability, ImplementationCapability)
        if self.request is not None:
            require_instance(self.request, PhysicalRequestPolicy)
        if self.implementation != self.capability.implementation or self.configuration != self.capability.configuration:
            reject(EffectCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DelegatedRequestContract(PrivateValue):
    """Fixed recipe declaration for bounded delegated requests."""

    mode: Literal["qualified", "opaque"]
    child_slots: tuple[int, ...]
    routes: tuple[ChildRoute, ...]
    followups: frozenset[ChildSourceKind] = frozenset({"correction", "failover"})
    usage_requirement: Literal["exact", "unknown_allowed"] = "exact"

    def __post_init__(self) -> None:
        require_literal(self.mode, frozenset({"qualified", "opaque"}))
        require_literal(self.usage_requirement, frozenset({"exact", "unknown_allowed"}))
        require_instance(self.child_slots, tuple)
        for slot in self.child_slots:
            require_count(slot)
        if len(set(self.child_slots)) != len(self.child_slots):
            reject(EffectCode.DUPLICATE)
        require_instance(self.routes, tuple)
        for route in self.routes:
            require_instance(route, ChildRoute)
            route.__post_init__()
        if len({route.name for route in self.routes}) != len(self.routes):
            reject(EffectCode.DUPLICATE)
        if type(self.followups) is not frozenset or not self.followups <= {
            "retry",
            "correction",
            "conversation_restart",
            "failover",
        }:
            reject(EffectCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OperationExecutionPolicy(PrivateValue):
    node: NodeId
    kind: ExecutionKind
    request: PhysicalRequestPolicy | None
    safe_detachment: SafeDetachment
    implementations: tuple[ExecutionImplementation, ...]
    result_outcomes: frozenset[str]
    runtime_outcomes: tuple[RuntimeOutcome, ...]
    initial_request_purpose: Literal["initial", "repair"] = "initial"
    delegated: DelegatedRequestContract | None = None

    def __post_init__(self) -> None:
        require_instance(self.node, NodeId)
        require_literal(self.kind, frozenset({"local", "external", "decision", "delegated"}))
        if self.delegated is not None:
            require_instance(self.delegated, DelegatedRequestContract)
            self.delegated.__post_init__()
        if (self.kind == "delegated") != (self.delegated is not None):
            reject(EffectCode.CONTRADICTORY)
        require_literal(self.initial_request_purpose, frozenset({"initial", "repair"}))
        if self.initial_request_purpose == "repair" and (self.kind != "external" or self.request is None):
            reject(EffectCode.CONTRADICTORY)
        if self.request is not None:
            require_instance(self.request, PhysicalRequestPolicy)
        require_literal(self.safe_detachment, frozenset({"forbidden", "independent_after_dispatch"}))
        if not isinstance(self.implementations, tuple) or any(
            not isinstance(item, ExecutionImplementation) for item in self.implementations
        ):
            reject(EffectCode.INVALID_TYPE)
        if not isinstance(self.result_outcomes, frozenset) or any(
            not isinstance(item, str) for item in self.result_outcomes
        ):
            reject(EffectCode.INVALID_TYPE)
        if not isinstance(self.runtime_outcomes, tuple) or any(
            not isinstance(item, RuntimeOutcome) for item in self.runtime_outcomes
        ):
            reject(EffectCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AssessmentFinding(PrivateValue):
    status: AssessmentStatus
    code: str

    def __post_init__(self) -> None:
        require_literal(self.status, frozenset({"satisfied", "unsatisfied", "unknown"}))
        require_text(self.code)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class EvidenceProductionDecl(PrivateValue):
    node: NodeId
    outcome: str
    promise: str
    evidence_port: str
    absence_queries: frozenset[int]
    supported_findings: frozenset[AssessmentFinding]

    def __post_init__(self) -> None:
        require_instance(self.node, NodeId)
        require_text(self.outcome)
        require_text(self.promise)
        require_text(self.evidence_port)
        if not isinstance(self.absence_queries, frozenset):
            reject(EffectCode.INVALID_TYPE)
        for query in self.absence_queries:
            require_count(query)
        if not isinstance(self.supported_findings, frozenset) or any(
            not isinstance(item, AssessmentFinding) for item in self.supported_findings
        ):
            reject(EffectCode.INVALID_TYPE)
        if not self.supported_findings:
            reject(EffectCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AssessmentLimits(PrivateValue):
    max_productions: int
    max_findings_per_production: int
    max_finding_code_bytes: int
    max_absence_queries: int
    max_assessment_facts: int
    max_port_facts: int
    max_provenance_edges: int

    def __post_init__(self) -> None:
        for value in (
            self.max_productions,
            self.max_findings_per_production,
            self.max_finding_code_bytes,
            self.max_absence_queries,
            self.max_assessment_facts,
            self.max_port_facts,
            self.max_provenance_edges,
        ):
            require_count(value)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LocalAssessmentResult(PrivateValue):
    association: SemanticAssociation
    promise: str
    evidence_port: str
    finding: AssessmentFinding

    def __post_init__(self) -> None:
        require_instance(self.association, SemanticAssociation)
        require_text(self.promise)
        require_text(self.evidence_port)
        require_instance(self.finding, AssessmentFinding)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AssessmentEnvironment(PrivateValue):
    configuration: FrozenConfig
    state: StateRevisionView
    absences: frozenset[AbsenceRef]

    def __post_init__(self) -> None:
        require_instance(self.configuration, FrozenConfig)
        require_instance(self.state, StateRevisionView)
        if not isinstance(self.absences, frozenset) or any(not isinstance(item, AbsenceRef) for item in self.absences):
            reject(EffectCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OperationOutputKey(PrivateValue):
    activation: ActivationKey
    target: DatumId
    port: str

    def __post_init__(self) -> None:
        require_instance(self.activation, ActivationKey)
        require_instance(self.target, DatumId)
        require_text(self.port)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RootInputKey(PrivateValue):
    target: DatumId
    port: str

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_text(self.port)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BoundInputKey(PrivateValue):
    target: DatumId
    node: NodeId
    port: str
    binding_artifact: BindingArtifactRef

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_instance(self.node, NodeId)
        require_text(self.port)
        require_instance(self.binding_artifact, BindingArtifactRef)
        self.binding_artifact.__post_init__()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class InitialCollectionKey(PrivateValue):
    target: DatumId
    node: NodeId
    port: str
    declaration: BindingDeclarationId

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_instance(self.node, NodeId)
        require_text(self.port)
        require_instance(self.declaration, BindingDeclarationId)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class MapItemKey(PrivateValue):
    expander: ActivationKey
    member: ActivationKey
    target: DatumId
    port: str
    item_key: int
    item_version: int

    def __post_init__(self) -> None:
        self._validate_fields()
        if self.member.invocation != self.expander.invocation or self.member.parent != self.expander:
            reject(EffectCode.FOREIGN_OWNER)

    def _validate_fields(self) -> None:
        """Validate shape and scalars separately from occurrence ownership."""
        require_instance(self.expander, ActivationKey)
        require_instance(self.member, ActivationKey)
        require_instance(self.target, DatumId)
        require_text(self.port)
        require_count(self.item_key)
        require_count(self.item_version, positive=True)


ProvenanceKey: TypeAlias = OperationOutputKey | RootInputKey | BoundInputKey | InitialCollectionKey | MapItemKey


_FACT_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ExecutionAssessmentFact(PrivateValue):
    activation: ActivationKey
    node: NodeId
    outcome: str
    promise: str
    evidence_artifact: ArtifactRef
    finding: AssessmentFinding
    environment: AssessmentEnvironment

    def __init__(self, *, _key: object, **values: object) -> None:
        _init_fact(self, _key, values)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ExecutionPortFact(PrivateValue):
    activation: ActivationKey
    node: NodeId
    target: DatumId
    port: str
    artifact: ArtifactRef
    artifact_type: ArtifactType
    role: ArtifactRole

    def __init__(self, *, _key: object, **values: object) -> None:
        _init_fact(self, _key, values)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class FinalOutputFact(PrivateValue):
    target: DatumId
    outcome: str
    port: str
    candidate: CandidateRef
    producer: ProvenanceKey

    def __init__(self, *, _key: object, **values: object) -> None:
        _init_fact(self, _key, values)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ArtifactProvenanceFact(PrivateValue):
    key: ProvenanceKey
    artifact: ArtifactRef
    parents: frozenset[ProvenanceKey]
    decision: bool

    def __init__(self, *, _key: object, **values: object) -> None:
        _init_fact(self, _key, values)


def _init_fact(value: object, key: object, fields: dict[str, object]) -> None:
    if key is not _FACT_KEY:
        raise TypeError("execution facts are created by the executor")
    for name, fact_value in fields.items():
        object.__setattr__(value, name, fact_value)


_PLAN_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class AdmittedExecutionPlan(PrivateValue):
    context: AdmittedContextPlan
    capabilities: tuple[ImplementationCapability, ...]
    policies: frozenset[OperationExecutionPolicy]
    decisions: frozenset[DecisionDeclaration]
    map_expansions: tuple[MapExpansionDecl, ...]
    assessment_productions: tuple[EvidenceProductionDecl, ...]
    assessment_limits: AssessmentLimits
    ingress_limits: IngressLimits
    nodes: Mapping[NodeId, Node] = field(compare=False)
    input_sources: Mapping[NodeInputRef, WorkflowInputRef | ContextInputRef | NodeOutputRef] = field(compare=False)
    materialization_schemas: Mapping[ArtifactType, Literal["scalar", "collection"]] = field(compare=False)

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _PLAN_KEY:
            raise TypeError("execution plans are created by admission")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        bodies = reachable_workflows(self.context.prepared.workflow)
        object.__setattr__(self, "nodes", MappingProxyType({node.id: node for body in bodies for node in body.nodes}))
        object.__setattr__(
            self,
            "input_sources",
            MappingProxyType(
                {binding.destination: binding.source for body in bodies for binding in body.input_bindings}
            ),
        )
        object.__setattr__(self, "materialization_schemas", MappingProxyType(self._compile_materialization_schemas()))

    def _compile_materialization_schemas(self) -> dict[ArtifactType, Literal["scalar", "collection"]]:
        schemas: dict[ArtifactType, Literal["scalar", "collection"]] = {}
        context = self.context.bound_context
        if context is not None:
            for fact in context.receipt.sources:
                schemas[fact.declaration.materialization.item_type] = "scalar"
                schemas[fact.declaration.artifact_type] = (
                    "scalar" if fact.declaration.materialization.kind == "single" else "collection"
                )
        for declaration in self.context.adaptive_retrievals:
            operation = self.nodes[declaration.node].operation
            artifact_type = next(
                item.artifact_type for item in operation.outputs if item.name == declaration.output_port
            )
            schemas[declaration.materialization.item_type] = "scalar"
            schemas[artifact_type] = "scalar" if declaration.materialization.kind == "single" else "collection"
        for expansion in self.map_expansions:
            operation = self.nodes[expansion.expander].operation
            artifact_type = next(
                item.artifact_type for item in operation.outputs if item.name == expansion.membership_port
            )
            schemas[expansion.item_type] = "scalar"
            schemas[artifact_type] = "collection"
        return schemas

    def _output_role(self, node: NodeId, outcome: OutcomeSpec, port: str) -> ArtifactRole:
        """Apply admitted occurrence-role precedence to an operation output."""
        if any(item.node == node for item in self.decisions):
            return "decision"
        productions = [
            item for item in self.assessment_productions if item.node == node and item.outcome == outcome.name
        ]
        promises = {item.name: item for item in outcome.evidence}
        if any(promises[item.promise].subject_port == port for item in productions) or any(
            isinstance(item.source, NodeOutputRef) and item.source.node == node and item.source.port == port
            for item in self.context.prepared.workflow.workflow.output_bindings
        ):
            return "candidate"
        return "evidence" if any(item.evidence_port == port for item in productions) else "artifact"


@runtime_checkable
class RequestTransport(Protocol):
    async def dispatch(self, request: DispatchEnvelope) -> TransportResult: ...

    async def cancel(self, request: object) -> StopResult: ...


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LocalCompleted(PrivateValue):
    results: tuple[AssociationResult, ...]
    assessments: tuple[LocalAssessmentResult, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LocalDecisionWait(PrivateValue):
    association: SemanticAssociation
    artifact: ArtifactRef


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LocalFailure(PrivateValue):
    failure: FailureClass


LocalResult: TypeAlias = LocalCompleted | LocalDecisionWait | LocalFailure


@runtime_checkable
class LocalCallable(Protocol):
    async def run(self, request: tuple[AssociationInput, ...]) -> LocalResult: ...


class DelegatedRequests(Protocol):
    """Invocation-owned access to declared physical sibling slots."""

    async def request(
        self,
        slot: int,
        *,
        source_kind: ChildSourceKind = "initial",
        predecessor: PhysicalRequestId | None = None,
        evidence: ChildEvidence | None = None,
    ) -> PhysicalRequestId: ...

    def parser_rejected(self, request: PhysicalRequestId) -> ChildEvidence: ...


@runtime_checkable
class DelegatedCallable(Protocol):
    @property
    def execution(self) -> Literal["same_loop"]: ...

    async def submit(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalResult: ...


class MonotonicClock(Protocol):
    def now_ns(self) -> int: ...


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExecutionLimits(PrivateValue):
    max_local_in_flight: int
    max_remote_outstanding: int
    max_runtime_artifacts: int
    max_runtime_artifact_bytes: int
    max_collection_items: int

    def __post_init__(self) -> None:
        require_count(self.max_local_in_flight)
        require_count(self.max_remote_outstanding)
        require_count(self.max_runtime_artifacts)
        require_count(self.max_runtime_artifact_bytes)
        require_count(self.max_collection_items)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DecisionLimits(PrivateValue):
    max_pending: int
    max_lifetime_ns: int

    def __post_init__(self) -> None:
        require_count(self.max_pending)
        require_count(self.max_lifetime_ns)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ImplementationHandle(PrivateValue):
    implementation: ImplementationRef
    operation: OperationSpec
    configuration: FrozenConfig
    local: LocalCallable | None
    transport: RequestTransport | None
    resource: ResourceLease | None
    measurement: NodeMeasurement | None = None
    delegated: DelegatedCallable | None = None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExecutionServices(PrivateValue):
    handles: tuple[ImplementationHandle, ...]
    context_resources: tuple[ContextResource, ...]
    limits: ExecutionLimits
    decision_limits: DecisionLimits
    clock: MonotonicClock
    absence_revisions: tuple[tuple[int, int], ...] = ()


class DecisionWaitId(OpaqueIdentity):
    __slots__ = ("invocation",)
    invocation: InvocationId

    @classmethod
    def new(cls, *, invocation: InvocationId) -> DecisionWaitId:
        require_instance(invocation, InvocationId)
        return cls._new(invocation=invocation)  # type: ignore[return-value]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DecisionWait(PrivateValue):
    wait: DecisionWaitId
    activation: ActivationKey
    workflow: WorkflowId
    artifact: ArtifactRef
    allowed_decisions: frozenset[str]
    deadline_ns: int

    def __post_init__(self) -> None:
        require_instance(self.wait, DecisionWaitId)
        require_instance(self.activation, ActivationKey)
        require_instance(self.workflow, WorkflowId)
        require_instance(self.artifact, ArtifactRef)
        if self.wait.invocation != self.activation.invocation:
            reject(EffectCode.FOREIGN_OWNER)
        if not isinstance(self.allowed_decisions, frozenset) or any(
            not isinstance(item, str) or not item for item in self.allowed_decisions
        ):
            reject(EffectCode.INVALID_TYPE)
        if not self.allowed_decisions:
            reject(EffectCode.INVALID_VALUE)
        require_count(self.deadline_ns)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DecisionResponse(PrivateValue):
    wait: DecisionWaitId
    workflow: WorkflowId
    artifact: ArtifactRef
    decision: str

    def __post_init__(self) -> None:
        require_instance(self.wait, DecisionWaitId)
        require_instance(self.workflow, WorkflowId)
        require_instance(self.artifact, ArtifactRef)
        require_text(self.decision)


_RESULT_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ExecutionResult(PrivateValue):
    _execution: AdmittedExecutionPlan
    _input_parents: tuple[tuple[DatumId, ActivationKey, str, ProvenanceKey], ...]
    _passthrough_parents: tuple[tuple[DatumId, ActivationKey, str, ProvenanceKey], ...]
    record: CanonicalRecord
    states: tuple[ActivationState, ...]
    requests: RequestReceipt
    cleanup: tuple[CleanupFact, ...]
    pending_decisions: tuple[DecisionWait, ...]
    artifacts: tuple[tuple[ArtifactRef, ArtifactValue], ...]
    assessments: tuple[ExecutionAssessmentFact, ...]
    ports: tuple[ExecutionPortFact, ...]
    final_outputs: tuple[FinalOutputFact, ...]
    provenance: tuple[ArtifactProvenanceFact, ...]
    cleanup_associations: tuple[CleanupAssociation, ...]
    publication_failures: tuple[PublicationFailureFact, ...]

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _RESULT_KEY:
            raise TypeError("execution results are created by the executor")
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PublicationFailureFact(PrivateValue):
    """Occurrence-local publication rejection, independent of physical acceptance."""

    association: SemanticAssociation
    target: DatumId
    condition: RuntimeCondition
    failure: FailureClass | None
