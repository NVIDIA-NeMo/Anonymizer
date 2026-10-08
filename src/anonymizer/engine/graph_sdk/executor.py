# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single asynchronous executor for admitted graph plans."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from itertools import product
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
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationCapability,
    ImplementationRef,
    validate_capability,
)
from anonymizer.engine.graph_sdk.context import (
    AdaptiveRetrievalDecl,
    AdmittedContextPlan,
    BindingArtifactRef,
    BoundTextArtifact,
    ContextProvider,
    ContextResource,
    ContextSelector,
    SelectorField,
    SourceFailure,
    SourceLost,
    SourceResponse,
)
from anonymizer.engine.graph_sdk.preparation import PreparedPlan, StateRevisionView, recheck_capabilities
from anonymizer.engine.graph_sdk.records import (
    AbsenceRef,
    CandidateRef,
    CanonicalRecord,
    ExpectedMembership,
    TargetStatus,
    TerminalFact,
)
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    BindingDeclarationId,
    Dispatch,
    DispatchEnvelope,
    ExternalSettlement,
    FailureClass,
    InvocationRequestScope,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    RequestCancel,
    RequestEvent,
    RequestPolicyBinding,
    RequestReceipt,
    RequestState,
    Reserve,
    ScopeCancel,
    SemanticAssociation,
    StopAcknowledged,
    StopConfirmed,
    StopResult,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
    TransportFailure,
    TransportLost,
    TransportResult,
    TransportSuccess,
    advance_requests,
    bind_request_policies,
    can_reserve_followup,
    initialize_requests,
    request_receipt,
)
from anonymizer.engine.graph_sdk.resources import (
    CleanupFact,
    ResourceId,
    ResourceLease,
    SafeDetachment,
    close_resource,
)
from anonymizer.graph._values import (
    ActivationKey,
    ArtifactRef,
    ContractViolation,
    DatumId,
    InvocationId,
    TaskAttemptId,
    ValidationCode,
)
from anonymizer.graph.activation import (
    ActivationEntry,
    ActivationSeed,
    ActivationState,
    CloseUnstarted,
    ObserveMembership,
    ObserveOverflow,
    ObserveTerminal,
    Select,
    Start,
    advance_activation,
    initialize_activation,
)
from anonymizer.graph.activation import (
    _depth as _activation_depth,
)
from anonymizer.graph.workflow import (
    AdmittedWorkflow,
    ArtifactType,
    ContextInputRef,
    InputBinding,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeClass,
    OutcomeSpec,
    OutputDependency,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    validate_dynamic_input_summaries,
)

ExecutionKind: TypeAlias = Literal["local", "external", "decision"]
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
ArtifactRole: TypeAlias = Literal["artifact", "candidate", "decision", "evidence"]
CleanupPurpose: TypeAlias = Literal["verification", "accounting", "transport_only"]
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
class OperationExecutionPolicy(PrivateValue):
    node: NodeId
    kind: ExecutionKind
    request: PhysicalRequestPolicy | None
    safe_detachment: SafeDetachment
    implementations: tuple[ExecutionImplementation, ...]
    result_outcomes: frozenset[str]
    runtime_outcomes: tuple[RuntimeOutcome, ...]

    def __post_init__(self) -> None:
        require_instance(self.node, NodeId)
        require_literal(self.kind, frozenset({"local", "external", "decision"}))
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
        require_instance(self.expander, ActivationKey)
        require_instance(self.member, ActivationKey)
        require_instance(self.target, DatumId)
        require_text(self.port)
        require_count(self.item_key)
        require_count(self.item_version, positive=True)
        if self.member.invocation != self.expander.invocation or self.member.parent != self.expander:
            reject(EffectCode.FOREIGN_OWNER)


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


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class CleanupAssociation(PrivateValue):
    resource: ResourceId
    targets: frozenset[DatumId]
    purpose: CleanupPurpose

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

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _PLAN_KEY:
            raise TypeError("execution plans are created by admission")
        for name, value in values.items():
            object.__setattr__(self, name, value)

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

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _RESULT_KEY:
            raise TypeError("execution results are created by the executor")
        for name, value in values.items():
            object.__setattr__(self, name, value)


def admit_execution_plan(
    *,
    context: AdmittedContextPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
    decisions: tuple[DecisionDeclaration, ...],
    assessment_productions: tuple[EvidenceProductionDecl, ...],
    assessment_limits: AssessmentLimits,
    map_expansions: tuple[MapExpansionDecl, ...] = (),
) -> AdmittedExecutionPlan:
    """Admit exact execution policies over one immutable prepared plan."""
    require_instance(context, AdmittedContextPlan)
    prepared = context.prepared
    if not isinstance(capabilities, tuple):
        reject(EffectCode.INVALID_TYPE)
    if len(capabilities) > prepared.limits.max_capabilities:
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(not isinstance(item, ImplementationCapability) for item in capabilities):
        reject(EffectCode.INVALID_TYPE)
    if len(capabilities) != len(set(capabilities)):
        reject(EffectCode.DUPLICATE)
    for capability in capabilities:
        validate_capability(capability)
    if not isinstance(policies, tuple) or any(not isinstance(item, OperationExecutionPolicy) for item in policies):
        reject(EffectCode.INVALID_TYPE)
    if not isinstance(decisions, tuple) or any(not isinstance(item, DecisionDeclaration) for item in decisions):
        reject(EffectCode.INVALID_TYPE)
    _validate_map_expansions(context, map_expansions)
    workflow_owners = _workflow_owners(prepared.workflow.workflow)
    if any(item.node.workflow not in workflow_owners for item in policies):
        reject(EffectCode.FOREIGN_OWNER)
    selected = {item.node: item.capability for item in prepared.implementations}
    if {item.node for item in policies} != set(selected):
        reject(EffectCode.MISSING)
    if len(policies) != len({item.node for item in policies}):
        reject(EffectCode.DUPLICATE)
    decision_nodes = {item.node for item in decisions}
    if len(decisions) != len(decision_nodes):
        reject(EffectCode.DUPLICATE)
    policy_by_node = {item.node: item for item in policies}
    for declaration in decisions:
        policy = policy_by_node.get(declaration.node)
        if policy is None or policy.kind != "decision":
            reject(EffectCode.UNSUPPORTED)
        operation = policy.implementations[0].capability.operation
        if declaration.artifact_port not in {item.name for item in operation.inputs}:
            reject(EffectCode.MISSING)
        declared_outcomes = {item.name for item in operation.outcomes}
        if any(item.outcome not in declared_outcomes for item in declaration.outcomes):
            reject(EffectCode.UNSUPPORTED)
        dependencies = {item.output: item for item in operation.output_dependencies}
        for decision_outcome in declaration.outcomes:
            outcome = next(item for item in operation.outcomes if item.name == decision_outcome.outcome)
            if any(
                dependencies.get(port) is None or dependencies[port].identity_input != declaration.artifact_port
                for port in outcome.produced_ports
            ):
                reject(EffectCode.UNSUPPORTED)
    for policy in policies:
        if not policy.implementations:
            reject(EffectCode.IMPLEMENTATION_COUNT)
        primary = policy.implementations[0]
        if primary.capability != selected[policy.node]:
            reject(EffectCode.UNSUPPORTED)
        if any(item.capability not in capabilities for item in policy.implementations):
            reject(EffectCode.UNSUPPORTED)
        _validate_policy(policy, policy.node in decision_nodes)
        adaptive = next((item for item in context.adaptive_retrievals if item.node == policy.node), None)
        if adaptive is not None:
            source = next(
                item
                for item in context.context_capabilities
                if item.source == adaptive.source and "adaptive_retrieval" in item.uses
            )
            if policy.kind != "external" or policy.request != source.request:
                reject(EffectCode.CONTRADICTORY)
            if any(item.capability.attribution != "per_task" for item in policy.implementations):
                reject(EffectCode.UNSUPPORTED)
    _validate_assessments(prepared, policies, assessment_productions, assessment_limits)
    _validate_execution_fact_capacity(prepared, assessment_productions, assessment_limits)
    return AdmittedExecutionPlan(
        _key=_PLAN_KEY,
        context=context,
        capabilities=capabilities,
        policies=frozenset(policies),
        decisions=frozenset(decisions),
        map_expansions=map_expansions,
        assessment_productions=assessment_productions,
        assessment_limits=assessment_limits,
    )


def _validate_map_expansions(
    context: AdmittedContextPlan,
    declarations: tuple[MapExpansionDecl, ...],
) -> None:
    if not isinstance(declarations, tuple) or any(not isinstance(item, MapExpansionDecl) for item in declarations):
        reject(EffectCode.INVALID_TYPE)
    workflow = context.prepared.workflow
    maps = [item for scope in workflow.scopes for item in scope.maps]
    expected = {(item.expander, outcome) for item in maps for outcome in item.expansion_outcomes}
    keys = [(item.expander, item.outcome) for item in declarations]
    if len(keys) != len(set(keys)):
        reject(EffectCode.DUPLICATE)
    if set(keys) != expected:
        reject(EffectCode.MISSING if set(keys) < expected else EffectCode.EXTRA)
    schemas: dict[ArtifactType, tuple[Literal["scalar", "collection"], ArtifactType, int]] = {}
    bound = context.bound_context
    if bound is not None:
        for fact in bound.receipt.sources:
            materialization = fact.declaration.materialization
            _add_materialization_schema(
                schemas,
                fact.declaration.artifact_type,
                materialization.item_type,
                materialization.kind,
                1,
            )
    for adaptive in context.adaptive_retrievals:
        operation = _node_operation(workflow.workflow, adaptive.node)
        output_type = next(item.artifact_type for item in operation.outputs if item.name == adaptive.output_port)
        _add_materialization_schema(
            schemas,
            output_type,
            adaptive.materialization.item_type,
            adaptive.materialization.kind,
            1,
        )
    replacement_choices: list[tuple[InputBinding, ...]] = []
    by_key = {(item.expander, item.outcome): item for item in declarations}
    for dynamic_map in maps:
        member_operation = _node_operation(workflow.workflow, dynamic_map.member)
        member_inputs = {item.name: item.artifact_type for item in member_operation.inputs}
        if dynamic_map.item_input is not None and any(
            fact.declaration.node == dynamic_map.member and fact.declaration.port == dynamic_map.item_input
            for fact in (() if bound is None else bound.receipt.sources)
        ):
            reject(EffectCode.CONTRADICTORY)
        map_replacements: list[InputBinding] = []
        for outcome_name in dynamic_map.expansion_outcomes:
            declaration = by_key[(dynamic_map.expander, outcome_name)]
            operation = _node_operation(workflow.workflow, dynamic_map.expander)
            outcome = next(item for item in operation.outcomes if item.name == outcome_name)
            output_types = {item.name: item.artifact_type for item in operation.outputs}
            if declaration.membership_port not in outcome.produced_ports:
                reject(EffectCode.UNSUPPORTED)
            collection_type = output_types.get(declaration.membership_port)
            if collection_type is None or collection_type == declaration.item_type:
                reject(EffectCode.CONTRADICTORY)
            if dynamic_map.item_input is not None:
                if member_inputs.get(dynamic_map.item_input) != declaration.item_type:
                    reject(EffectCode.CONTRADICTORY)
                map_replacements.append(
                    InputBinding(
                        source=NodeOutputRef(node=dynamic_map.expander, port=declaration.membership_port),
                        destination=NodeInputRef(node=dynamic_map.member, port=dynamic_map.item_input),
                    )
                )
            _add_materialization_schema(schemas, collection_type, declaration.item_type, "collection", 0)
        if map_replacements:
            replacement_choices.append(tuple(map_replacements))
    for replacements in product(*replacement_choices):
        try:
            validate_dynamic_input_summaries(workflow.workflow, replacements)
        except ContractViolation as error:
            if error.code is ValidationCode.MISSING:
                reject(EffectCode.MISSING)
            if error.code is ValidationCode.CONTRADICTORY:
                reject(EffectCode.CONTRADICTORY)
            raise


def _validate_execution_fact_capacity(
    prepared: PreparedPlan,
    productions: tuple[EvidenceProductionDecl, ...],
    limits: AssessmentLimits,
) -> None:
    """Prove structural and operation fact storage from the finite reservation recipe."""
    root = prepared.workflow.workflow
    operation_nodes: dict[NodeId, OperationSpec] = {}
    subgraph_nodes: dict[NodeId, OperationSpec] = {}
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for node in current.nodes:
            if isinstance(node, OperationNode):
                operation_nodes[node.id] = node.operation
            else:
                subgraph_nodes[node.id] = node.operation
                pending.append(node.body)
    port_facts = 0
    provenance_edges = 0
    assessment_facts = 0
    mapped_expanders = {
        declaration.expander: declaration.max_children
        for scope in prepared.workflow.scopes
        for declaration in scope.maps
        if declaration.item_input is not None
    }
    for slot in prepared.reservation_recipe:
        provenance_edges += mapped_expanders.get(slot.template, 0)
        operation = operation_nodes.get(slot.template)
        if operation is not None:
            port_facts += len(operation.inputs)
            port_facts += max(len(outcome.produced_ports) for outcome in operation.outcomes)
            dependencies = {item.output: item for item in operation.output_dependencies}
            provenance_edges += max(
                sum(len(dependencies[port].inputs) for port in outcome.produced_ports) for outcome in operation.outcomes
            )
            assessment_facts += max(
                sum(item.node == slot.template and item.outcome == outcome.name for item in productions)
                for outcome in operation.outcomes
            )
            continue
        interface = subgraph_nodes[slot.template]
        projected_count = max(len(outcome.produced_ports) for outcome in interface.outcomes)
        port_facts += projected_count
        provenance_edges += projected_count
    target_count = len(prepared.target_occurrences)
    if port_facts * target_count > limits.max_port_facts:
        reject(EffectCode.LIMIT_EXCEEDED)
    if provenance_edges * target_count > limits.max_provenance_edges:
        reject(EffectCode.LIMIT_EXCEEDED)
    if assessment_facts * target_count > limits.max_assessment_facts:
        reject(EffectCode.LIMIT_EXCEEDED)


def _add_materialization_schema(
    schemas: dict[ArtifactType, tuple[Literal["scalar", "collection"], ArtifactType, int]],
    output_type: ArtifactType,
    item_type: ArtifactType,
    kind: Literal["single", "collection"],
    minimum: int,
) -> None:
    scalar = ("scalar", item_type, 1)
    existing_item = schemas.get(item_type)
    if existing_item is not None and existing_item[0] == "collection":
        reject(EffectCode.CONTRADICTORY)
    schemas.setdefault(item_type, scalar)
    candidate = ("scalar", item_type, 1) if kind == "single" else ("collection", item_type, minimum)
    existing = schemas.get(output_type)
    if existing is not None and existing != candidate:
        reject(EffectCode.CONTRADICTORY)
    schemas[output_type] = candidate


def _validate_policy(policy: OperationExecutionPolicy, has_decision: bool) -> None:
    if policy.request is not None and policy.request.retry_owner == "implementation":
        reject(EffectCode.UNSUPPORTED)
    capabilities = [item.capability for item in policy.implementations]
    operation = capabilities[0].operation
    if any(item.operation != operation for item in capabilities):
        reject(EffectCode.CONTRADICTORY)
    if policy.kind in {"local", "decision"}:
        if len(policy.implementations) != 1 or policy.request is not None:
            reject(EffectCode.IMPLEMENTATION_COUNT)
        if capabilities[0].effect != "local" or policy.implementations[0].request is not None:
            reject(EffectCode.CONTRADICTORY)
    else:
        if policy.request is None or any(
            item.capability.effect != "external" or item.request != policy.request for item in policy.implementations
        ):
            reject(EffectCode.CONTRADICTORY)
        if any(item.max_physical_requests_per_activation != policy.request.max_attempts for item in capabilities):
            reject(EffectCode.CONTRADICTORY)
    if policy.safe_detachment == "independent_after_dispatch" and (
        policy.kind != "external" or any(item.resource_lifetime != "executor_owned" for item in capabilities)
    ):
        reject(EffectCode.CONTRADICTORY)
    if policy.kind != "external" and policy.safe_detachment != "forbidden":
        reject(EffectCode.CONTRADICTORY)
    if (policy.kind == "decision") != has_decision:
        reject(EffectCode.CONTRADICTORY)
    outcomes = {item.name: item.category for item in operation.outcomes}
    if policy.kind != "decision" and (not policy.result_outcomes or not policy.result_outcomes <= outcomes.keys()):
        reject(EffectCode.UNSUPPORTED)
    if any(item.outcome is not None and item.outcome not in outcomes for item in policy.runtime_outcomes):
        reject(EffectCode.UNSUPPORTED)
    keys = [(item.condition, item.reported_outcome, item.failure) for item in policy.runtime_outcomes]
    if len(keys) != len(set(keys)):
        reject(EffectCode.DUPLICATE)
    expected = _expected_runtime_keys(policy.kind, policy.result_outcomes)
    if set(keys) != expected:
        reject(EffectCode.MISSING if set(keys) < expected else EffectCode.EXTRA)
    for item in policy.runtime_outcomes:
        if item.condition == "result" and item.outcome != item.reported_outcome:
            reject(EffectCode.CONTRADICTORY)
        if item.outcome is not None and outcomes.get(item.outcome) != item.category:
            reject(EffectCode.CONTRADICTORY)
        if item.condition != "result" and item.outcome is not None and item.category == "success":
            reject(EffectCode.CONTRADICTORY)
        if item.outcome is None and item.category == "success":
            reject(EffectCode.CONTRADICTORY)
        if item.condition == "cancel_before_start" and (item.outcome is not None or item.category != "blocked"):
            reject(EffectCode.CONTRADICTORY)
        if item.condition == "request_inconsistent" and item.category != "inconsistent":
            reject(EffectCode.CONTRADICTORY)


def _expected_runtime_keys(kind: ExecutionKind, results: frozenset[str]) -> set[tuple[str, str | None, str | None]]:
    keys: set[tuple[str, str | None, str | None]] = {("result", item, None) for item in results}
    failures = (
        {"permanent", "implementation_exception"}
        if kind == "decision"
        else {
            "rejected_before_acceptance",
            "retryable",
            "malformed_response",
            "permanent",
            "transport_unknown",
            "implementation_exception",
        }
    )
    keys.update(("failure", None, item) for item in failures)
    conditions = {"cancel_before_start", "cancel_after_start", "artifact_limit_exhausted", "deadline_exhausted"}
    if kind == "external":
        conditions |= {
            "cancel_after_dispatch",
            "lost",
            "request_inconsistent",
            "budget_exhausted",
            "request_limit_exhausted",
        }
    keys.update((item, None, None) for item in conditions)
    return keys


def _validate_assessments(
    prepared: object,
    policies: tuple[OperationExecutionPolicy, ...],
    productions: tuple[EvidenceProductionDecl, ...],
    limits: AssessmentLimits,
) -> None:
    del prepared
    if not isinstance(productions, tuple) or any(not isinstance(item, EvidenceProductionDecl) for item in productions):
        reject(EffectCode.INVALID_TYPE)
    require_instance(limits, AssessmentLimits)
    if len(productions) > limits.max_productions:
        reject(EffectCode.LIMIT_EXCEEDED)
    production_keys = [(item.node, item.outcome, item.promise) for item in productions]
    if len(production_keys) != len(set(production_keys)):
        reject(EffectCode.DUPLICATE)
    policy_by_node = {item.node: item for item in policies}
    for item in productions:
        policy = policy_by_node.get(item.node)
        if policy is None or policy.kind != "local":
            reject(EffectCode.UNSUPPORTED)
        operation = policy.implementations[0].capability.operation
        outcome = next((value for value in operation.outcomes if value.name == item.outcome), None)
        if outcome is None or item.evidence_port not in outcome.produced_ports:
            reject(EffectCode.UNSUPPORTED)
        promise = next((value for value in outcome.evidence if value.name == item.promise), None)
        if promise is None:
            reject(EffectCode.UNSUPPORTED)
        dependency = next(
            (value for value in operation.output_dependencies if value.output == item.evidence_port),
            None,
        )
        if dependency is None or dependency.inputs != promise.consumed_ports:
            reject(EffectCode.CONTRADICTORY)
        if (
            len(item.supported_findings) > limits.max_findings_per_production
            or len(item.absence_queries) > limits.max_absence_queries
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        if any(len(value.code.encode()) > limits.max_finding_code_bytes for value in item.supported_findings):
            reject(EffectCode.LIMIT_EXCEEDED)


@dataclass(slots=True)
class _ExecutionControl:
    cancelled: bool = False
    scheduler_changed: asyncio.Event = field(default_factory=asyncio.Event)
    pending: dict[DecisionWaitId, DecisionWait] = field(default_factory=dict)
    responses: dict[DecisionWaitId, DecisionResponse] = field(default_factory=dict)
    closed: set[DecisionWaitId] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _FactCheckpoint:
    values: dict[ArtifactRef, ArtifactValue]
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef]
    provenance_count: int
    passthrough_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey]
    ports: tuple[ExecutionPortFact, ...]
    assessment_count: int
    next_artifact: int


@dataclass(slots=True)
class _ExecutionFacts:
    values: dict[ArtifactRef, ArtifactValue] = field(default_factory=dict)
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef] = field(default_factory=dict)
    provenance: list[ArtifactProvenanceFact] = field(default_factory=list)
    ports: list[ExecutionPortFact] = field(default_factory=list)
    assessments: list[ExecutionAssessmentFact] = field(default_factory=list)
    input_parents: list[tuple[DatumId, ActivationKey, str, ProvenanceKey]] = field(default_factory=list)
    passthrough_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = field(default_factory=dict)
    next_artifact: int = 0

    def checkpoint(self) -> _FactCheckpoint:
        return _FactCheckpoint(
            values=self.values.copy(),
            produced=self.produced.copy(),
            provenance_count=len(self.provenance),
            passthrough_parents=self.passthrough_parents.copy(),
            ports=tuple(self.ports),
            assessment_count=len(self.assessments),
            next_artifact=self.next_artifact,
        )

    def restore(self, checkpoint: _FactCheckpoint) -> None:
        self.values.clear()
        self.values.update(checkpoint.values)
        self.produced.clear()
        self.produced.update(checkpoint.produced)
        del self.provenance[checkpoint.provenance_count :]
        self.passthrough_parents.clear()
        self.passthrough_parents.update(checkpoint.passthrough_parents)
        self.ports.clear()
        self.ports.extend(checkpoint.ports)
        del self.assessments[checkpoint.assessment_count :]
        self.next_artifact = checkpoint.next_artifact


@dataclass(slots=True)
class _RequestAuthority:
    state: RequestState

    def apply(self, event: RequestEvent) -> None:
        self.state = advance_requests(state=self.state, event=event)

    def bind(self, binding: RequestPolicyBinding) -> None:
        self.state = bind_request_policies(state=self.state, binding=binding)


@dataclass(frozen=True, slots=True)
class _ExecutionJob:
    state_index: int
    target: DatumId
    activation: ActivationKey
    node: NodeId
    policy: OperationExecutionPolicy
    implementation: ExecutionImplementation
    inputs: tuple[AssociationInput, ...]
    input_parents: dict[str, ProvenanceKey]
    association: SemanticAssociation


def _group_external_jobs(jobs: list[_ExecutionJob]) -> list[list[_ExecutionJob]]:
    groups: list[list[_ExecutionJob]] = []
    for job in jobs:
        shared = all(item.capability.attribution == "keyed_shared_request" for item in job.policy.implementations)
        group = next(
            (group for group in groups if shared and group[0].node == job.node and group[0].policy == job.policy),
            None,
        )
        if group is None:
            groups.append([job])
        else:
            group.append(job)
    return groups


@dataclass(frozen=True, slots=True)
class _DeferredAcceptance:
    request: PhysicalRequestId
    results: tuple[AssociationResult, ...]
    settlement: ExternalSettlement | None


class RunningExecution:
    """Live owner of one graph invocation."""

    __slots__ = ("_control", "_task", "invocation")

    def __init__(
        self, invocation: InvocationId, control: _ExecutionControl, task: asyncio.Task[ExecutionResult]
    ) -> None:
        self.invocation = invocation
        self._control = control
        self._task = task

    def request_cancel(self) -> None:
        self._control.cancelled = True

    def pending_decisions(self) -> tuple[DecisionWait, ...]:
        return tuple(self._control.pending.values())

    def submit_decision(self, decision: DecisionResponse) -> None:
        require_instance(decision, DecisionResponse)
        if decision.wait.invocation != self.invocation:
            reject(EffectCode.FOREIGN_OWNER)
        wait = self._control.pending.get(decision.wait)
        if wait is None:
            reject(EffectCode.DUPLICATE if decision.wait in self._control.closed else EffectCode.MISSING)
        if decision.workflow != wait.workflow or decision.artifact != wait.artifact:
            reject(EffectCode.FOREIGN_OWNER)
        if decision.decision not in wait.allowed_decisions:
            reject(EffectCode.UNSUPPORTED)
        if decision.wait in self._control.responses:
            reject(EffectCode.DUPLICATE)
        self._control.responses[decision.wait] = decision

    async def wait(self) -> ExecutionResult:
        return await asyncio.shield(self._task)


async def start_execution(
    *,
    admitted: AdmittedExecutionPlan,
    capabilities: tuple[ImplementationCapability, ...],
    services: ExecutionServices,
) -> RunningExecution:
    """Recheck a plan before allocating one graph invocation and starting work."""
    require_instance(admitted, AdmittedExecutionPlan)
    require_instance(services, ExecutionServices)
    prepared = admitted.context.prepared
    recheck_capabilities(prepared=prepared, capabilities=capabilities)
    if any(
        item not in capabilities
        for policy in admitted.policies
        for item in (x.capability for x in policy.implementations)
    ):
        reject(EffectCode.CHANGED_FAILOVER_POLICY)
    _validate_services(admitted, services)
    _validate_absences(admitted, services)
    invocation = InvocationId.new(plan=prepared.plan)
    control = _ExecutionControl()
    task = asyncio.create_task(_run_execution(admitted, services, invocation, control))
    return RunningExecution(invocation, control, task)


def execute_sync(
    *,
    admitted: AdmittedExecutionPlan,
    capabilities: tuple[ImplementationCapability, ...],
    services: ExecutionServices,
) -> ExecutionResult:
    """Execute outside an active event loop using the authoritative async core."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise NestedEventLoopError("execute_sync cannot run inside an active event loop")

    async def run() -> ExecutionResult:
        running = await start_execution(admitted=admitted, capabilities=capabilities, services=services)
        return await running.wait()

    return asyncio.run(run())


def _validate_services(admitted: AdmittedExecutionPlan, services: ExecutionServices) -> None:
    require_instance(services.limits, ExecutionLimits)
    require_instance(services.decision_limits, DecisionLimits)
    if any(item.max_lifetime_ns > services.decision_limits.max_lifetime_ns for item in admitted.decisions):
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(
        item.materialization.kind == "collection" and item.bounds.max_items > services.limits.max_collection_items
        for item in admitted.context.adaptive_retrievals
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    bound_context = admitted.context.bound_context
    if bound_context is not None and any(
        item.declaration.materialization.kind == "collection"
        and item.declaration.bounds.max_items > services.limits.max_collection_items
        for item in bound_context.receipt.sources
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    _validate_initial_materialization_capacity(admitted, services.limits)
    if not isinstance(services.handles, tuple) or any(
        not isinstance(item, ImplementationHandle) for item in services.handles
    ):
        reject(EffectCode.INVALID_TYPE)
    retained = [item for policy in admitted.policies for item in policy.implementations]
    if len(services.handles) > admitted.context.prepared.limits.max_capabilities:
        reject(EffectCode.LIMIT_EXCEEDED)
    handle_keys = [(item.implementation, item.operation, item.configuration) for item in services.handles]
    expected = [(item.implementation, item.capability.operation, item.configuration) for item in retained]
    if len(handle_keys) != len(set(handle_keys)):
        reject(EffectCode.DUPLICATE)
    if set(handle_keys) != set(expected):
        reject(EffectCode.MISSING)
    capability_by_key = {
        (item.implementation, item.capability.operation, item.configuration): item.capability for item in retained
    }
    policy_by_key = {
        (item.implementation, item.capability.operation, item.configuration): policy
        for policy in admitted.policies
        for item in policy.implementations
    }
    for handle in services.handles:
        key = (handle.implementation, handle.operation, handle.configuration)
        capability = capability_by_key[key]
        policy = policy_by_key[key]
        if (handle.local is None) == (handle.transport is None):
            reject(EffectCode.CONTRADICTORY)
        if policy.kind == "external":
            if handle.transport is None or not isinstance(handle.transport, RequestTransport):
                reject(EffectCode.INVALID_TYPE)
        elif handle.local is None or not isinstance(handle.local, LocalCallable):
            reject(EffectCode.INVALID_TYPE)
        if capability.resource_lifetime == "stateless" and handle.resource is not None:
            reject(EffectCode.CONTRADICTORY)
        if capability.resource_lifetime != "stateless":
            if handle.resource is None:
                reject(EffectCode.MISSING)
            expected_owner = "caller" if capability.resource_lifetime == "caller_owned" else "sdk"
            if handle.resource.owner != expected_owner or handle.resource.safe_detachment != policy.safe_detachment:
                reject(EffectCode.CONTRADICTORY)
    if not isinstance(services.context_resources, tuple) or any(
        not isinstance(item, ContextResource) for item in services.context_resources
    ):
        reject(EffectCode.INVALID_TYPE)
    expected_context = {
        (item.source, item) for item in admitted.context.context_capabilities if "adaptive_retrieval" in item.uses
    }
    actual_context = {(item.source, item.capability) for item in services.context_resources}
    if actual_context != expected_context or len(actual_context) != len(services.context_resources):
        reject(EffectCode.MISSING if actual_context < expected_context else EffectCode.EXTRA)


def _validate_initial_materialization_capacity(admitted: AdmittedExecutionPlan, limits: ExecutionLimits) -> None:
    prepared = admitted.context.prepared
    datum_text = {item.id: item.text for item in prepared.data.datums}
    artifact_count = len(prepared.bound_inputs)
    artifact_bytes = sum(len(datum_text[item.source].encode()) for item in prepared.bound_inputs)
    provenance_edges = 0
    context = admitted.context.bound_context
    if context is not None:
        declarations = {item.identity: item.declaration for item in context.receipt.sources}
        groups: dict[BindingDeclarationId, list[BoundTextArtifact]] = {}
        for artifact in context.artifacts:
            groups.setdefault(artifact.reference.declaration, []).append(artifact)
        for identity, items in groups.items():
            declaration = declarations[identity]
            if declaration.materialization.kind == "collection" and len(items) > limits.max_collection_items:
                reject(EffectCode.LIMIT_EXCEEDED)
            item_bytes = sum(len(item.text.encode()) for item in items)
            artifact_count += len(items)
            artifact_bytes += item_bytes
            if declaration.materialization.kind == "collection":
                artifact_count += 1
                artifact_bytes += item_bytes
                provenance_edges += len(items)
    if artifact_count > limits.max_runtime_artifacts or artifact_bytes > limits.max_runtime_artifact_bytes:
        reject(EffectCode.LIMIT_EXCEEDED)
    if provenance_edges > admitted.assessment_limits.max_provenance_edges:
        reject(EffectCode.LIMIT_EXCEEDED)


def _validate_absences(admitted: AdmittedExecutionPlan, services: ExecutionServices) -> None:
    if not isinstance(services.absence_revisions, tuple) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in services.absence_revisions
    ):
        reject(EffectCode.INVALID_TYPE)
    for query, revision in services.absence_revisions:
        require_count(query)
        require_count(revision, positive=True)
    queries = [item[0] for item in services.absence_revisions]
    if len(queries) != len(set(queries)):
        reject(EffectCode.DUPLICATE)
    expected = {query for item in admitted.assessment_productions for query in item.absence_queries}
    if set(queries) != expected:
        reject(EffectCode.MISSING)


_ExecutionTask: TypeAlias = asyncio.Task[
    tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
]


class _InvocationRuntime:
    """Own mutable scheduling, publication, and cleanup for one invocation."""

    def __init__(
        self,
        admitted: AdmittedExecutionPlan,
        services: ExecutionServices,
        invocation: InvocationId,
        control: _ExecutionControl,
    ) -> None:
        self.admitted = admitted
        self.services = services
        self.invocation = invocation
        self.control = control
        self.prepared = self.admitted.context.prepared
        self.policies = {item.node: item for item in self.admitted.policies}
        self.handles = {
            (item.implementation, item.operation, item.configuration): item for item in self.services.handles
        }
        self.context_leases, self.failed_context_sources = _acquire_context_resources(self.services.context_resources)
        self.states: list[ActivationState] = []
        self.target_keys: dict[DatumId, dict[int, ActivationKey]] = {}
        self._initialize_activations()
        scope = InvocationRequestScope(invocation=self.invocation)
        self.request_authority = _RequestAuthority(
            state=initialize_requests(
                scope=scope,
                hard_limit=self.prepared.configuration.hard_request_limit,
                policies=frozenset((item.request for item in self.admitted.policies if item.request is not None)),
            )
        )
        self.deferred_acceptances: dict[SemanticAssociation, _DeferredAcceptance] = {}
        self.request_resources: dict[PhysicalRequestId, ResourceId] = {}
        self.facts = _ExecutionFacts()
        self.root_inputs: dict[tuple[DatumId, str], ArtifactRef] = {}
        self.subgraph_inputs: dict[tuple[DatumId, ActivationKey, str], ArtifactRef] = {}
        self.subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey] = {}
        self._initialize_root_inputs()
        self.facts.next_artifact = _materialize_bound_context(
            self.admitted,
            self.invocation,
            self.facts.values,
            self.facts.produced,
            self.facts.provenance,
            self.facts.next_artifact,
            self.services.limits,
        )
        self.attempts: dict[ActivationKey, TaskAttemptId] = {}
        self.cancelled_unstarted: set[ActivationKey] = set()
        self.jobs: dict[_ExecutionTask, _ExecutionJob] = {}
        self.physical_jobs: dict[asyncio.Task[object], asyncio.Task[object]] = {}

    def _initialize_activations(self) -> None:
        for target_map in self.prepared.target_occurrences:
            keys: dict[int, ActivationKey] = {}
            for slot in self.prepared.reservation_recipe:
                keys[slot.index] = ActivationKey(
                    invocation=self.invocation,
                    occurrence=target_map.occurrence_offset + slot.index,
                    parent=keys.get(slot.parent_index),
                    iteration=slot.iteration,
                )
            self.target_keys[target_map.target] = keys
            reservations = frozenset(
                (
                    ActivationSeed(template=slot.template, activation=keys[slot.index])
                    for slot in self.prepared.reservation_recipe
                )
            )
            initialized = initialize_activation(
                workflow=self.prepared.workflow,
                invocation=self.invocation,
                reservations=reservations,
                limits=self.prepared.activation_limits,
            )
            self.states.append(
                advance_activation(
                    state=initialized, event=Select(seeds=_initial_seeds(self.prepared.workflow.workflow, reservations))
                )
            )

    def _initialize_root_inputs(self) -> None:
        datum_text = {item.id: item.text for item in self.prepared.data.datums}
        for bound in self.prepared.bound_inputs:
            reference = ArtifactRef(invocation=self.invocation, key=self.facts.next_artifact, version=1)
            self.facts.next_artifact += 1
            self.facts.values[reference] = TextArtifactValue(text=datum_text[bound.source])
            self.root_inputs[bound.target, bound.port] = reference
            self.facts.provenance.append(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY,
                    key=RootInputKey(target=bound.target, port=bound.port),
                    artifact=reference,
                    parents=frozenset(),
                    decision=False,
                )
            )

    async def run(self) -> ExecutionResult:
        while True:
            scheduled = self._schedule_ready()
            if all((state.complete for state in self.states)) and (not self.jobs):
                break
            if not self.jobs:
                if not scheduled:
                    break
                continue
            await self._process_completed()
        return await self._finalize()

    def _schedule_ready(self) -> bool:
        scheduled = False
        external_jobs: list[_ExecutionJob] = []
        for state_index, (target_map, state) in enumerate(
            zip(self.prepared.target_occurrences, self.states, strict=True)
        ):
            ready = sorted(
                (item for item in state.entries if item.status == "ready"), key=lambda item: item.activation.occurrence
            )
            for entry in ready:
                scheduled = self._schedule_entry(state_index, target_map.target, entry, external_jobs) or scheduled
        self._dispatch_external(external_jobs)
        return scheduled

    def _schedule_entry(
        self, state_index: int, target: DatumId, entry: ActivationEntry, external_jobs: list[_ExecutionJob]
    ) -> bool:
        policy = self.policies.get(entry.template)
        if policy is None:
            return self._start_subgraph(state_index, target, entry)
        decision_jobs = sum((item.policy.kind == "decision" for item in self.jobs.values()))
        if policy.kind == "decision" and decision_jobs >= self.services.decision_limits.max_pending:
            if self.services.decision_limits.max_pending == 0:
                reject(EffectCode.PENDING_LIMIT)
            return False
        if _operation_has_omitted_context(self.admitted, target, entry.template):
            self.states[state_index] = advance_activation(
                state=self.states[state_index],
                event=CloseUnstarted(activation=entry.activation, category="blocked"),
            )
            return True
        waiting_activations = {wait.activation for wait in self.control.pending.values()}
        local_jobs = sum(
            (
                item.policy.kind != "external" and item.activation not in waiting_activations
                for item in self.jobs.values()
            )
        )
        remote_jobs = len(
            {self.physical_jobs.get(task, task) for task, item in self.jobs.items() if item.policy.kind == "external"}
        ) + len(_group_external_jobs(external_jobs))
        joins_pending_batch = any(
            (item.node == entry.template and item.policy == policy for item in external_jobs)
        ) and all((item.capability.attribution == "keyed_shared_request" for item in policy.implementations))
        if policy.kind != "external":
            if self.services.limits.max_local_in_flight == 0:
                reject(EffectCode.LIMIT_EXCEEDED)
            if local_jobs >= self.services.limits.max_local_in_flight:
                return False
        remote_capacity_stalled = (
            policy.kind == "external"
            and (not joins_pending_batch)
            and (
                max(len(self.request_authority.state.remote_outstanding), remote_jobs)
                >= self.services.limits.max_remote_outstanding
            )
        )
        if remote_capacity_stalled and remote_jobs:
            return False
        self._start_operation(state_index, target, entry, policy, external_jobs, remote_capacity_stalled)
        return True

    def _start_subgraph(self, state_index: int, target: DatumId, entry: ActivationEntry) -> bool:
        if _is_subgraph_node(self.prepared.workflow.workflow, entry.template):
            try:
                _materialize_subgraph_inputs(
                    self.admitted,
                    target,
                    self.states[state_index],
                    entry.activation,
                    entry.template,
                    self.root_inputs,
                    self.subgraph_inputs,
                    self.subgraph_input_parents,
                    self.facts.produced,
                    self.facts.provenance,
                )
            except EffectRejected as exc:
                if exc.code != EffectCode.MISSING:
                    raise
                self.states[state_index] = advance_activation(
                    state=self.states[state_index],
                    event=CloseUnstarted(activation=entry.activation, category="blocked"),
                )
                return True
            self.states[state_index] = advance_activation(
                state=self.states[state_index], event=Start(activation=entry.activation)
            )
            return True
        return False

    def _start_operation(
        self,
        state_index: int,
        target: DatumId,
        entry: ActivationEntry,
        policy: OperationExecutionPolicy,
        external_jobs: list[_ExecutionJob],
        remote_capacity_stalled: bool,
    ) -> None:
        if self.control.cancelled:
            self.cancelled_unstarted.add(entry.activation)
            self.states[state_index] = advance_activation(
                state=self.states[state_index], event=CloseUnstarted(activation=entry.activation, category="blocked")
            )
            return
        attempt = TaskAttemptId.new(activation=entry.activation)
        association = SemanticAssociation(task=attempt)
        try:
            inputs, input_parents = _operation_inputs(
                self.admitted,
                target,
                entry.template,
                entry.activation,
                self.states[state_index],
                association,
                self.root_inputs,
                self.subgraph_inputs,
                self.subgraph_input_parents,
                self.facts.produced,
                self.facts.values,
                self.facts.provenance,
            )
        except EffectRejected as exc:
            if exc.code != EffectCode.MISSING:
                raise
            self.states[state_index] = advance_activation(
                state=self.states[state_index], event=CloseUnstarted(activation=entry.activation, category="blocked")
            )
            return
        self.states[state_index] = advance_activation(
            state=self.states[state_index], event=Start(activation=entry.activation)
        )
        self.attempts[entry.activation] = attempt
        implementation = policy.implementations[0]
        job = _ExecutionJob(
            state_index=state_index,
            target=target,
            activation=entry.activation,
            node=entry.template,
            policy=policy,
            implementation=implementation,
            inputs=inputs,
            input_parents=input_parents,
            association=association,
        )
        self._record_input_ports(job)
        self._launch_job(job, external_jobs, remote_capacity_stalled)

    def _record_input_ports(self, job: _ExecutionJob) -> None:
        self.facts.input_parents.extend(
            (job.target, job.activation, port, parent) for port, parent in job.input_parents.items()
        )
        for input_artifact in job.inputs[0].inputs:
            if input_artifact.artifact is None:
                reject(EffectCode.CONTRADICTORY)
            self.facts.ports.append(
                ExecutionPortFact(
                    _key=_FACT_KEY,
                    activation=job.activation,
                    node=job.node,
                    target=job.target,
                    port=input_artifact.port,
                    artifact=input_artifact.artifact,
                    artifact_type=input_artifact.artifact_type,
                    role="decision"
                    if job.policy.kind == "decision"
                    and input_artifact.port
                    == next((item.artifact_port for item in self.admitted.decisions if item.node == job.node))
                    else _inherited_artifact_role(
                        job.input_parents.get(input_artifact.port), self.facts.provenance, self.facts.ports
                    ),
                )
            )

    def _launch_job(
        self, job: _ExecutionJob, external_jobs: list[_ExecutionJob], remote_capacity_stalled: bool
    ) -> None:
        policy = job.policy
        implementation = job.implementation
        handle = self.handles[
            implementation.implementation, implementation.capability.operation, implementation.configuration
        ]
        if policy.kind == "external":
            if remote_capacity_stalled:
                coroutine = _immediate_execution_result(_mapping(policy, "request_limit_exhausted", None, None))
            else:
                adaptive = next(
                    (item for item in self.admitted.context.adaptive_retrievals if item.node == job.node), None
                )
                if adaptive is None:
                    external_jobs.append(job)
                    return
                if adaptive.source in self.failed_context_sources:
                    coroutine = _immediate_execution_result(
                        _mapping(policy, "failure", None, "implementation_exception")
                    )
                else:
                    coroutine = _run_adaptive(
                        self.admitted,
                        policy,
                        adaptive,
                        job.association,
                        job.inputs,
                        self.request_authority,
                        self.context_leases,
                        self.control,
                        self.services.limits,
                        self.deferred_acceptances,
                        self.request_resources,
                    )
        elif policy.kind == "decision":
            coroutine = _run_decision(
                self.admitted, policy, handle, job.association, job.inputs, job.activation, self.services, self.control
            )
        else:
            coroutine = _run_local(policy, handle, job.association, job.inputs, self.control)
        self.jobs[asyncio.create_task(coroutine)] = job

    def _dispatch_external(self, external_jobs: list[_ExecutionJob]) -> None:
        groups = _group_external_jobs(external_jobs)
        for group in groups:
            physical = asyncio.create_task(
                _run_external_batch(
                    self.admitted,
                    group[0].policy,
                    self.handles,
                    tuple((value for job in group for value in job.inputs)),
                    self.request_authority,
                    self.control,
                    self.services.limits,
                    self.deferred_acceptances,
                    self.request_resources,
                )
            )
            for job in group:
                task = asyncio.create_task(_external_association_result(physical, job.association))
                self.jobs[task] = job
                self.physical_jobs[task] = physical

    async def _process_completed(self) -> None:
        scheduler_wakeup = asyncio.create_task(self.control.scheduler_changed.wait())
        try:
            completed, _ = await asyncio.wait((*self.jobs, scheduler_wakeup), return_when=asyncio.FIRST_COMPLETED)
        finally:
            scheduler_wakeup.cancel()
            with suppress(asyncio.CancelledError):
                await scheduler_wakeup
        if scheduler_wakeup in completed:
            self.control.scheduler_changed.clear()
        pending_completed = {task for task in self.jobs if task in completed}
        while pending_completed:
            peers = await self._complete_group(next(iter(pending_completed)))
            pending_completed.difference_update(peers)

    async def _complete_group(self, first: _ExecutionTask) -> list[_ExecutionTask]:
        physical = self.physical_jobs.get(first)
        peers = (
            [task for task in self.jobs if self.physical_jobs.get(task) is physical]
            if physical is not None
            else [first]
        )
        await asyncio.gather(*peers)
        deferred = self.deferred_acceptances.get(self.jobs[first].association)
        transaction = self.facts.checkpoint()
        prior_states = list(self.states)
        outcomes: list[tuple[_ExecutionJob, RuntimeOutcome]] = []
        for task in peers:
            job = self.jobs.pop(task)
            self.physical_jobs.pop(task, None)
            mapping, results, assessments = task.result()
            self.deferred_acceptances.pop(job.association, None)
            mapping, results, checkpoint, staged_result = self._stage_outputs(job, mapping, results, assessments)
            mapping = self._advance_completed(job, mapping, results, checkpoint, staged_result)
            outcomes.append((job, mapping))
        if deferred is not None:
            failed = any((mapping.condition != "result" for _, mapping in outcomes))
            if failed and len(peers) > 1:
                self.facts.restore(transaction)
                self.states[:] = prior_states
                for job, mapping in outcomes:
                    if mapping.condition == "result":
                        mapping = _mapping(job.policy, "failure", None, "malformed_response")
                    self.states[job.state_index] = advance_activation(
                        state=self.states[job.state_index],
                        event=ObserveTerminal(
                            activation=job.activation, outcome=mapping.outcome, category=mapping.category
                        ),
                    )
            if failed:
                self.request_authority.apply(AcceptFailure(request=deferred.request, failure="malformed_response"))
            else:
                self.request_authority.apply(AcceptResult(request=deferred.request, results=deferred.results))
            if deferred.settlement is not None:
                self.request_authority.apply(ObserveSettlement(settlement=deferred.settlement))
        return peers

    def _stage_outputs(
        self,
        job: _ExecutionJob,
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        assessments: tuple[LocalAssessmentResult, ...],
    ) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], _FactCheckpoint | None, bool]:
        staged_result = False
        checkpoint: _FactCheckpoint | None = None
        if mapping.condition == "result":
            if job.policy.kind == "decision" and (not results):
                results = (
                    AssociationResult(
                        association=job.association,
                        outcome=mapping.outcome or "",
                        outputs=(),
                        consumed_context_ports=frozenset(),
                    ),
                )
            if not _validate_assessment_returns(self.admitted, job.association, job.node, mapping, assessments):
                mapping = _mapping(job.policy, "failure", None, "malformed_response")
            elif not _valid_dynamic_membership_result(self.admitted, job.node, mapping, results):
                mapping = _mapping(job.policy, "failure", None, "malformed_response")
            else:
                checkpoint = self.facts.checkpoint()
                output_status, created = _accept_outputs(
                    self.admitted,
                    self.invocation,
                    job.target,
                    job.activation,
                    job.node,
                    job.implementation.capability.operation,
                    mapping,
                    job.association,
                    job.inputs,
                    job.input_parents,
                    results,
                    self.facts,
                    self.services.limits,
                )
                if output_status != "valid":
                    mapping = _mapping(
                        job.policy,
                        "artifact_limit_exhausted" if output_status == "limit" else "failure",
                        None,
                        None if output_status == "limit" else "malformed_response",
                    )
                elif not _capture_assessments(
                    self.admitted,
                    job.implementation,
                    job.association,
                    job.activation,
                    job.node,
                    mapping,
                    assessments,
                    job.target,
                    self.services,
                    self.facts,
                ):
                    self.facts.restore(checkpoint)
                    mapping = _mapping(job.policy, "failure", None, "malformed_response")
                else:
                    _mark_assessment_subjects(self.admitted, job.node, mapping, job.target, job.activation, self.facts)
                    staged_result = True
                del created
        return (mapping, results, checkpoint, staged_result)

    def _advance_completed(
        self,
        job: _ExecutionJob,
        mapping: RuntimeOutcome,
        results: tuple[AssociationResult, ...],
        checkpoint: _FactCheckpoint | None,
        staged_result: bool,
    ) -> RuntimeOutcome:
        prior_state = self.states[job.state_index]
        candidate_state = advance_activation(
            state=prior_state,
            event=ObserveTerminal(activation=job.activation, outcome=mapping.outcome, category=mapping.category),
        )
        if staged_result:
            try:
                membership = _dynamic_membership_value(self.admitted, job.node, mapping.outcome, results)
                if membership is not None:
                    candidate_state = _observe_dynamic_membership(
                        self.admitted,
                        candidate_state,
                        job.target,
                        job.activation,
                        job.node,
                        mapping.outcome,
                        membership,
                        self.facts,
                        self.services.limits,
                    )
            except (ContractViolation, EffectRejected) as exc:
                assert checkpoint is not None
                self.facts.restore(checkpoint)
                condition = (
                    "artifact_limit_exhausted"
                    if isinstance(exc, EffectRejected) and exc.code == EffectCode.LIMIT_EXCEEDED
                    else "failure"
                )
                mapping = _mapping(
                    job.policy,
                    condition,
                    None,
                    None if condition == "artifact_limit_exhausted" else "malformed_response",
                )
                candidate_state = advance_activation(
                    state=prior_state,
                    event=ObserveTerminal(
                        activation=job.activation, outcome=mapping.outcome, category=mapping.category
                    ),
                )
        try:
            while True:
                _materialize_subgraph_outputs(
                    self.prepared.workflow.workflow,
                    job.target,
                    candidate_state,
                    self.facts,
                    self.subgraph_input_parents,
                )
                candidate_state, bridged = _bridge_structural_map_membership(
                    self.admitted, candidate_state, job.target, self.facts, self.services.limits
                )
                if not bridged:
                    break
        except (ContractViolation, EffectRejected) as exc:
            if not staged_result or checkpoint is None:
                raise
            self.facts.restore(checkpoint)
            condition = (
                "artifact_limit_exhausted"
                if isinstance(exc, EffectRejected) and exc.code == EffectCode.LIMIT_EXCEEDED
                else "failure"
            )
            mapping = _mapping(
                job.policy, condition, None, None if condition == "artifact_limit_exhausted" else "malformed_response"
            )
            candidate_state = advance_activation(
                state=prior_state,
                event=ObserveTerminal(activation=job.activation, outcome=mapping.outcome, category=mapping.category),
            )
        self.states[job.state_index] = candidate_state
        return mapping

    async def _finalize(self) -> ExecutionResult:
        if self.control.cancelled:
            self.request_authority.apply(ScopeCancel())
        cleanup, cleanup_associations = await _cleanup_execution(
            self.admitted,
            self.services.handles,
            self.context_leases,
            self.request_authority.state,
            self.request_resources,
        )
        record = _canonical_record(
            self.prepared,
            self.invocation,
            tuple(self.states),
            self.target_keys,
            self.attempts,
            frozenset(self.facts.values),
            frozenset(self.cancelled_unstarted),
        )
        final_outputs = _final_outputs(self.prepared, tuple(self.states), self.facts.produced, self.facts.provenance)
        return ExecutionResult(
            _key=_RESULT_KEY,
            _execution=self.admitted,
            _input_parents=tuple(self.facts.input_parents),
            _passthrough_parents=tuple(
                (target, activation, port, parent)
                for (target, activation, port), parent in self.facts.passthrough_parents.items()
            ),
            record=record,
            states=tuple(self.states),
            requests=request_receipt(self.request_authority.state),
            cleanup=cleanup,
            pending_decisions=(),
            artifacts=tuple(self.facts.values.items()),
            assessments=tuple(self.facts.assessments),
            ports=tuple(self.facts.ports),
            final_outputs=final_outputs,
            provenance=tuple(self.facts.provenance),
            cleanup_associations=cleanup_associations,
        )


async def _run_execution(
    admitted: AdmittedExecutionPlan, services: ExecutionServices, invocation: InvocationId, control: _ExecutionControl
) -> ExecutionResult:
    return await _InvocationRuntime(admitted, services, invocation, control).run()


def _initial_seeds(workflow: AdmittedWorkflow, reservations: frozenset[ActivationSeed]) -> frozenset[ActivationSeed]:
    blocked = {member for choice in workflow.choices for branch in choice.branches for member in branch.members}
    return frozenset(seed for seed in reservations if seed.activation.parent is None and seed.template not in blocked)


def _operation_inputs(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    node: NodeId,
    activation: ActivationKey,
    state: ActivationState,
    association: SemanticAssociation,
    root_inputs: dict[tuple[DatumId, str], ArtifactRef],
    subgraph_inputs: dict[tuple[DatumId, ActivationKey, str], ArtifactRef],
    subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef],
    values: dict[ArtifactRef, ArtifactValue],
    provenance: list[ArtifactProvenanceFact],
) -> tuple[tuple[AssociationInput, ...], dict[str, ProvenanceKey]]:
    workflow, operation_node = _operation_owner(admitted.context.prepared.workflow.workflow, node)
    operation = operation_node.operation
    ports: list[PortArtifact] = []
    parents: dict[str, ProvenanceKey] = {}
    for port in operation.inputs:
        source = next(
            (
                binding.source
                for binding in workflow.input_bindings
                if binding.destination == NodeInputRef(node=node, port=port.name)
            ),
            None,
        )
        source = _loop_input_source(admitted, state, node, activation, port.name, source)
        reference: ArtifactRef | None = produced.get((target, activation, port.name))
        if reference is not None:
            item_parent = next(
                (
                    fact.key
                    for fact in provenance
                    if fact.artifact == reference
                    and isinstance(fact.key, MapItemKey)
                    and fact.key.member == activation
                    and fact.key.port == port.name
                ),
                None,
            )
            if item_parent is not None:
                parents[port.name] = item_parent
        mapped_item = _is_mapped_item_input(admitted, state, node, activation, port.name)
        if reference is None and mapped_item:
            reject(EffectCode.MISSING)
        if reference is None:
            reference = produced.get((target, node, port.name))
        if reference is None and isinstance(source, (WorkflowInputRef, ContextInputRef)):
            if activation.parent is not None:
                reference = subgraph_inputs.get((target, activation.parent, source.port))
                parent = subgraph_input_parents.get((target, activation.parent, source.port))
                if parent is not None:
                    parents[port.name] = parent
            if reference is None and isinstance(source, WorkflowInputRef):
                reference = root_inputs.get((target, source.port))
            if reference is not None and isinstance(source, WorkflowInputRef):
                parents.setdefault(port.name, RootInputKey(target=target, port=source.port))
        elif reference is None and isinstance(source, NodeOutputRef):
            source_activation = _source_activation(state, activation, source.node)
            if source_activation is not None:
                reference = produced.get((target, source_activation, source.port))
                if reference is not None:
                    parents[port.name] = OperationOutputKey(
                        activation=source_activation, target=target, port=source.port
                    )
        if reference is None:
            reject(EffectCode.MISSING)
        if port.name not in parents:
            candidates = [
                fact.key
                for fact in provenance
                if fact.artifact == reference
                and isinstance(fact.key, (BoundInputKey, InitialCollectionKey))
                and fact.key.target == target
                and fact.key.node == node
                and fact.key.port == port.name
            ]
            if len(candidates) != 1:
                reject(EffectCode.MISSING)
            parents[port.name] = candidates[0]
        ports.append(
            PortArtifact(
                port=port.name,
                artifact_type=port.artifact_type,
                artifact=reference,
                value=values[reference],
            )
        )
    return (AssociationInput(association=association, inputs=tuple(ports)),), parents


def _loop_input_source(
    admitted: AdmittedExecutionPlan,
    state: ActivationState,
    node: NodeId,
    activation: ActivationKey,
    port: str,
    default: WorkflowInputRef | ContextInputRef | NodeOutputRef | None,
) -> WorkflowInputRef | ContextInputRef | NodeOutputRef | None:
    if activation.parent is None or activation.iteration is None:
        return default
    parent = next(
        (item.template for item in state.entries if item.activation == activation.parent),
        None,
    )
    declaration = next(
        (
            item
            for scope in admitted.context.prepared.workflow.scopes
            for item in scope.loops
            if item.member == node and item.starter == parent
        ),
        None,
    )
    if declaration is None:
        return default
    bindings = declaration.initial if activation.iteration == 0 else declaration.carried
    selected = [item.source for item in bindings if item.destination == NodeInputRef(node=node, port=port)]
    if len(selected) > 1:
        reject(EffectCode.DUPLICATE)
    return selected[0] if selected else default


def _is_mapped_item_input(
    admitted: AdmittedExecutionPlan,
    state: ActivationState,
    node: NodeId,
    activation: ActivationKey,
    port: str,
) -> bool:
    if activation.parent is None:
        return False
    parent = next(
        (item.template for item in state.entries if item.activation == activation.parent),
        None,
    )
    if parent is None:
        return False
    return any(
        declaration.member == node and declaration.expander == parent and declaration.item_input == port
        for scope in admitted.context.prepared.workflow.scopes
        for declaration in scope.maps
    )


def _operation_has_omitted_context(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    node: NodeId,
) -> bool:
    context = admitted.context.bound_context
    if context is None:
        return False
    return any(
        fact.declaration.target == target
        and fact.declaration.node == node
        and fact.declaration.requirement == "optional"
        and fact.terminal == "omitted_optional"
        for fact in context.receipt.sources
    )


def _inherited_artifact_role(
    parent: ProvenanceKey | None,
    provenance: list[ArtifactProvenanceFact],
    ports: list[ExecutionPortFact],
) -> ArtifactRole:
    if parent is None:
        return "artifact"
    source = next((item for item in provenance if item.key == parent), None)
    if source is None:
        return "artifact"
    if source.decision:
        return "decision"
    if isinstance(parent, OperationOutputKey):
        occurrence = next(
            (
                item
                for item in ports
                if item.activation == parent.activation and item.target == parent.target and item.port == parent.port
            ),
            None,
        )
        if occurrence is not None and occurrence.role == "candidate":
            return "candidate"
    return "artifact"


def _source_activation(
    state: ActivationState,
    destination: ActivationKey,
    source: NodeId,
) -> ActivationKey | None:
    candidates = [item.activation for item in state.entries if item.template == source]
    if destination.iteration is not None:
        previous = [
            item
            for item in candidates
            if item.parent == destination.parent
            and item.iteration is not None
            and item.iteration == destination.iteration - 1
        ]
        if len(previous) == 1:
            return previous[0]
    dynamic_map = next(
        (declaration for scope in state.workflow.scopes for declaration in scope.maps if declaration.member == source),
        None,
    )
    if dynamic_map is not None:
        expanders = [
            item.activation
            for item in state.entries
            if item.template == dynamic_map.expander and item.activation.parent == destination.parent
        ]
        if len(expanders) != 1:
            return None
        expansion = next((item for item in state.expansions if item.parent == expanders[0]), None)
        if expansion is None or expansion.status != "closed" or len(expansion.members) != 1:
            return None
        selected = next(iter(expansion.members))
        return selected if selected in candidates else None
    dynamic_loop = next(
        (declaration for scope in state.workflow.scopes for declaration in scope.loops if declaration.member == source),
        None,
    )
    if dynamic_loop is not None:
        starters = [
            item.activation
            for item in state.entries
            if item.template == dynamic_loop.starter and item.activation.parent == destination.parent
        ]
        if len(starters) != 1:
            return None
        exited = [
            item.activation
            for item in state.entries
            if item.template == source
            and item.activation.parent == starters[0]
            and item.outcome in dynamic_loop.exit_outcomes
        ]
        return exited[0] if len(exited) == 1 else None
    same_context = [item for item in candidates if item.parent == destination.parent]
    if len(same_context) == 1:
        return same_context[0]
    if destination.parent in candidates:
        return destination.parent
    parent_context = [item for item in candidates if item.parent == destination]
    return parent_context[0] if len(parent_context) == 1 else None


def _observe_dynamic_membership(
    admitted: AdmittedExecutionPlan,
    state: ActivationState,
    target: DatumId,
    activation: ActivationKey,
    node: NodeId,
    outcome: str | None,
    collection: ArtifactValue,
    facts: _ExecutionFacts,
    limits: ExecutionLimits,
) -> ActivationState:
    scope = next(item for item in state.workflow.scopes if node.workflow == item.workflow.workflow)
    declaration = next((item for item in scope.maps if item.expander == node), None)
    if declaration is None or outcome not in declaration.expansion_outcomes:
        return state
    expansion = next(item for item in admitted.map_expansions if item.expander == node and item.outcome == outcome)
    if not isinstance(collection, TextCollectionValue):
        reject(EffectCode.INVALID_TYPE)
    observed_count = len(collection.items)
    if observed_count > declaration.max_children:
        return advance_activation(
            state=state,
            event=ObserveOverflow(parent=activation, observed_count=observed_count),
        )
    members = sorted(
        (
            seed.activation
            for seed in state.reservations
            if seed.template == declaration.member and seed.activation.parent == activation
        ),
        key=lambda item: item.occurrence,
    )
    selected = members[:observed_count]
    if declaration.item_input is not None:
        if len(facts.values) + len(selected) > limits.max_runtime_artifacts:
            reject(EffectCode.LIMIT_EXCEEDED)
        if (
            sum(_artifact_bytes(item) for item in facts.values.values())
            + sum(len(item.value.text.encode()) for item in collection.items)
            > limits.max_runtime_artifact_bytes
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        if sum(len(item.parents) for item in facts.provenance) + len(selected) > (
            admitted.assessment_limits.max_provenance_edges
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        parent_key = OperationOutputKey(
            activation=activation,
            target=target,
            port=expansion.membership_port,
        )
        if not any(item.key == parent_key for item in facts.provenance):
            reject(EffectCode.MISSING)
        for member, item in zip(selected, collection.items, strict=True):
            reference = ArtifactRef(
                invocation=activation.invocation,
                key=facts.next_artifact,
                version=item.version,
            )
            facts.next_artifact += 1
            facts.values[reference] = item.value
            facts.produced[(target, member, declaration.item_input)] = reference
            key = MapItemKey(
                expander=activation,
                member=member,
                target=target,
                port=declaration.item_input,
                item_key=item.key,
                item_version=item.version,
            )
            facts.provenance.append(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY,
                    key=key,
                    artifact=reference,
                    parents=frozenset({parent_key}),
                    decision=False,
                )
            )
    return advance_activation(
        state=state,
        event=ObserveMembership(parent=activation, members=frozenset(selected), closed=True),
    )


def _valid_dynamic_membership_result(
    admitted: AdmittedExecutionPlan,
    node: NodeId,
    mapping: RuntimeOutcome,
    results: tuple[AssociationResult, ...],
) -> bool:
    dynamic_map = next(
        (
            declaration
            for dynamic_scope in admitted.context.prepared.workflow.scopes
            for declaration in dynamic_scope.maps
            if declaration.expander == node
        ),
        None,
    )
    if dynamic_map is None or mapping.outcome not in dynamic_map.expansion_outcomes:
        return True
    declaration = next(
        item for item in admitted.map_expansions if item.expander == node and item.outcome == mapping.outcome
    )
    selected = [output for result in results for output in result.outputs if output.port == declaration.membership_port]
    return len(selected) == 1 and isinstance(selected[0].value, TextCollectionValue)


def _dynamic_membership_value(
    admitted: AdmittedExecutionPlan,
    node: NodeId,
    outcome: str | None,
    results: tuple[AssociationResult, ...],
) -> ArtifactValue | None:
    declaration = next(
        (item for item in admitted.map_expansions if item.expander == node and item.outcome == outcome),
        None,
    )
    if declaration is None:
        return None
    selected = [
        output.value for result in results for output in result.outputs if output.port == declaration.membership_port
    ]
    if len(selected) != 1:
        reject(EffectCode.MISSING)
    return selected[0]


def _operation_owner(root: AdmittedWorkflow, node: NodeId) -> tuple[AdmittedWorkflow, OperationNode]:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in current.nodes:
            if isinstance(candidate, OperationNode) and candidate.id == node:
                return current, candidate
            if isinstance(candidate, SubgraphNode):
                pending.append(candidate.body)
    reject(EffectCode.MISSING)


def _workflow_owners(root: AdmittedWorkflow) -> frozenset[WorkflowId]:
    owners: set[WorkflowId] = set()
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        owners.add(current.workflow)
        pending.extend(candidate.body for candidate in current.nodes if isinstance(candidate, SubgraphNode))
    return frozenset(owners)


def _node_operation(root: AdmittedWorkflow, node: NodeId) -> OperationSpec:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in current.nodes:
            if candidate.id == node:
                return candidate.operation
            if isinstance(candidate, SubgraphNode):
                pending.append(candidate.body)
    reject(EffectCode.MISSING)


def _is_subgraph_node(root: AdmittedWorkflow, node: NodeId) -> bool:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in current.nodes:
            if isinstance(candidate, SubgraphNode):
                if candidate.id == node:
                    return True
                pending.append(candidate.body)
    return False


def _materialize_subgraph_inputs(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    state: ActivationState,
    activation: ActivationKey,
    node: NodeId,
    root_inputs: dict[tuple[DatumId, str], ArtifactRef],
    subgraph_inputs: dict[tuple[DatumId, ActivationKey, str], ArtifactRef],
    subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef],
    provenance: list[ArtifactProvenanceFact],
) -> None:
    owner, declaration = _subgraph_owner(admitted.context.prepared.workflow.workflow, node)
    for port in declaration.operation.inputs:
        source = next(
            (
                binding.source
                for binding in owner.input_bindings
                if binding.destination == NodeInputRef(node=node, port=port.name)
            ),
            None,
        )
        reference: ArtifactRef | None = produced.get((target, activation, port.name))
        parent: ProvenanceKey | None = None
        if reference is None and isinstance(source, ContextInputRef):
            reference = produced.get((target, node, port.name))
            if reference is not None:
                parent = next(
                    (fact.key for fact in provenance if fact.artifact == reference),
                    None,
                )
        if reference is not None:
            parent = parent or next(
                (
                    fact.key
                    for fact in provenance
                    if fact.artifact == reference
                    and isinstance(fact.key, MapItemKey)
                    and fact.key.member == activation
                    and fact.key.port == port.name
                ),
                None,
            )
        mapped_item = _is_mapped_item_input(admitted, state, node, activation, port.name)
        if reference is None and mapped_item:
            reject(EffectCode.MISSING)
        if reference is None and isinstance(source, (WorkflowInputRef, ContextInputRef)):
            if activation.parent is not None:
                reference = subgraph_inputs.get((target, activation.parent, source.port))
                parent = subgraph_input_parents.get((target, activation.parent, source.port))
            if reference is None and isinstance(source, WorkflowInputRef):
                reference = root_inputs.get((target, source.port))
                if reference is not None:
                    parent = RootInputKey(target=target, port=source.port)
        elif reference is None and isinstance(source, NodeOutputRef):
            source_activation = _source_activation(state, activation, source.node)
            if source_activation is not None:
                reference = produced.get((target, source_activation, source.port))
                parent = OperationOutputKey(activation=source_activation, target=target, port=source.port)
        if reference is None or parent is None:
            reject(EffectCode.MISSING)
        key = (target, activation, port.name)
        subgraph_inputs[key] = reference
        subgraph_input_parents[key] = parent


def _subgraph_owner(root: AdmittedWorkflow, node: NodeId) -> tuple[AdmittedWorkflow, SubgraphNode]:
    pending = [root]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for candidate in current.nodes:
            if isinstance(candidate, SubgraphNode):
                if candidate.id == node:
                    return current, candidate
                pending.append(candidate.body)
    reject(EffectCode.MISSING)


def _materialize_subgraph_outputs(
    root: AdmittedWorkflow,
    target: DatumId,
    state: ActivationState,
    facts: _ExecutionFacts,
    input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
) -> None:
    for entry in sorted(
        state.entries, key=lambda item: (-_activation_depth(item.activation), item.activation.occurrence)
    ):
        if entry.status not in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}:
            continue
        try:
            _, declaration = _subgraph_owner(root, entry.template)
        except EffectRejected:
            continue
        if entry.outcome is None:
            continue
        for binding in declaration.body.output_bindings:
            destination = (target, entry.activation, binding.destination.port)
            if destination in facts.produced:
                continue
            source_key: ProvenanceKey
            if isinstance(binding.source, WorkflowInputRef):
                parent = input_parents.get((target, entry.activation, binding.source.port))
                if parent is None:
                    continue
                source_key = parent
            else:
                source_activation = _source_activation(state, entry.activation, binding.source.node)
                source_entry = next(
                    (
                        child
                        for child in state.entries
                        if child.activation == source_activation and child.status == "success"
                    ),
                    None,
                )
                if source_entry is None:
                    continue
                source_key = OperationOutputKey(
                    activation=source_entry.activation,
                    target=target,
                    port=binding.source.port,
                )
            source_fact = next((item for item in facts.provenance if item.key == source_key), None)
            if source_fact is None:
                continue
            facts.produced[destination] = source_fact.artifact
            if isinstance(binding.source, WorkflowInputRef):
                facts.passthrough_parents[destination] = source_key
            key = OperationOutputKey(
                activation=entry.activation,
                target=target,
                port=binding.destination.port,
            )
            facts.provenance.append(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY,
                    key=key,
                    artifact=source_fact.artifact,
                    parents=frozenset({source_key}),
                    decision=source_fact.decision,
                )
            )
            output_type = next(
                item.artifact_type for item in declaration.operation.outputs if item.name == binding.destination.port
            )
            facts.ports.append(
                ExecutionPortFact(
                    _key=_FACT_KEY,
                    activation=entry.activation,
                    node=entry.template,
                    target=target,
                    port=binding.destination.port,
                    artifact=source_fact.artifact,
                    artifact_type=output_type,
                    role=(
                        "decision"
                        if source_fact.decision
                        else "candidate"
                        if any(
                            isinstance(output.source, NodeOutputRef)
                            and output.source.node == entry.template
                            and output.source.port == binding.destination.port
                            for output in root.output_bindings
                        )
                        else "artifact"
                    ),
                )
            )


def _bridge_structural_map_membership(
    admitted: AdmittedExecutionPlan,
    state: ActivationState,
    target: DatumId,
    facts: _ExecutionFacts,
    limits: ExecutionLimits,
) -> tuple[ActivationState, bool]:
    root = admitted.context.prepared.workflow.workflow
    for entry in sorted(state.entries, key=lambda item: item.activation.occurrence):
        if entry.status != "success" or entry.outcome is None or not _is_subgraph_node(root, entry.template):
            continue
        dynamic_map = next(
            (
                declaration
                for scope in state.workflow.scopes
                for declaration in scope.maps
                if declaration.expander == entry.template and entry.outcome in declaration.expansion_outcomes
            ),
            None,
        )
        if dynamic_map is None:
            continue
        expansion_state = next(
            (item for item in state.expansions if item.parent == entry.activation),
            None,
        )
        if expansion_state is not None and expansion_state.status != "pending":
            continue
        expansion = next(
            item
            for item in admitted.map_expansions
            if item.expander == entry.template and item.outcome == entry.outcome
        )
        reference = facts.produced.get((target, entry.activation, expansion.membership_port))
        if reference is None or reference not in facts.values:
            reject(EffectCode.MISSING)
        updated = _observe_dynamic_membership(
            admitted,
            state,
            target,
            entry.activation,
            entry.template,
            entry.outcome,
            facts.values[reference],
            facts,
            limits,
        )
        return updated, True
    return state, False


async def _run_local(
    policy: OperationExecutionPolicy,
    handle: ImplementationHandle,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    control: _ExecutionControl,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    if handle.local is None:
        reject(EffectCode.MISSING)
    operation = asyncio.create_task(handle.local.run(inputs))
    try:
        while not operation.done():
            if control.cancelled:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                return _mapping(policy, "cancel_after_start", None, None), (), ()
            await asyncio.sleep(0)
        result = operation.result()
    except Exception:
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if isinstance(result, LocalFailure):
        return _mapping(policy, "failure", None, result.failure), (), ()
    if isinstance(result, LocalDecisionWait):
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if not isinstance(result, LocalCompleted):
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if len(result.results) != 1 or result.results[0].association != association:
        return _mapping(policy, "failure", None, "malformed_response"), (), ()
    reported = result.results[0].outcome
    if reported not in policy.result_outcomes:
        return _mapping(policy, "failure", None, "malformed_response"), (), ()
    return _mapping(policy, "result", reported, None), result.results, result.assessments


async def _immediate_execution_result(
    mapping: RuntimeOutcome,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    return mapping, (), ()


async def _run_decision(
    admitted: AdmittedExecutionPlan,
    policy: OperationExecutionPolicy,
    handle: ImplementationHandle,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    activation: ActivationKey,
    services: ExecutionServices,
    control: _ExecutionControl,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    if handle.local is None:
        reject(EffectCode.MISSING)
    declaration = next(item for item in admitted.decisions if item.node == policy.node)
    artifact = next((item.artifact for item in inputs[0].inputs if item.port == declaration.artifact_port), None)
    if artifact is None:
        reject(EffectCode.MISSING)
    operation = asyncio.create_task(handle.local.run(inputs))
    try:
        while not operation.done():
            if control.cancelled:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                return _mapping(policy, "cancel_after_start", None, None), (), ()
            await asyncio.sleep(0)
        result = operation.result()
    except Exception:
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if isinstance(result, LocalFailure):
        failure: FailureClass = (
            result.failure
            if result.failure in {"permanent", "implementation_exception"}
            else ("implementation_exception")
        )
        return _mapping(policy, "failure", None, failure), (), ()
    if not isinstance(result, LocalDecisionWait):
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    if result.association != association or result.artifact != artifact:
        return _mapping(policy, "failure", None, "implementation_exception"), (), ()
    wait = DecisionWait(
        wait=DecisionWaitId.new(invocation=activation.invocation),
        activation=activation,
        workflow=admitted.context.prepared.workflow.workflow.workflow,
        artifact=artifact,
        allowed_decisions=frozenset(item.decision for item in declaration.outcomes),
        deadline_ns=services.clock.now_ns() + declaration.max_lifetime_ns,
    )
    pending = control.pending
    responses = control.responses
    closed = control.closed
    pending[wait.wait] = wait
    control.scheduler_changed.set()
    while True:
        if control.cancelled:
            mapping = _mapping(policy, "cancel_after_start", None, None)
            break
        response = responses.pop(wait.wait, None)
        if response is not None:
            selected = next(item for item in declaration.outcomes if item.decision == response.decision)
            outcome = next(item for item in handle.operation.outcomes if item.name == selected.outcome)
            category = outcome.category
            mapping = RuntimeOutcome(
                condition="result",
                reported_outcome=selected.outcome,
                failure=None,
                outcome=selected.outcome,
                category=category,
            )
            available = {item.port: item for item in inputs[0].inputs}
            dependencies = {item.output: item for item in handle.operation.output_dependencies}
            output_types = {item.name: item.artifact_type for item in handle.operation.outputs}
            outputs = tuple(
                PortArtifact(
                    port=port,
                    artifact_type=output_types[port],
                    artifact=None,
                    value=available[_identity_input(dependencies[port])].value,
                )
                for port in sorted(outcome.produced_ports)
            )
            break
        if services.clock.now_ns() >= wait.deadline_ns:
            mapping = _mapping(policy, "deadline_exhausted", None, None)
            break
        await asyncio.sleep(0)
    del pending[wait.wait]
    closed.add(wait.wait)
    if mapping.condition != "result":
        return mapping, (), ()
    return (
        mapping,
        (
            AssociationResult(
                association=association,
                outcome=mapping.outcome or "",
                outputs=outputs,
                consumed_context_ports=frozenset(use.port for use in outcome.context if use.port in available),
            ),
        ),
        (),
    )


async def _external_association_result(
    physical: asyncio.Task[
        dict[
            SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
        ]
    ],
    association: SemanticAssociation,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    return (await physical)[association]


async def _run_external_batch(
    admitted: AdmittedExecutionPlan,
    policy: OperationExecutionPolicy,
    handles: dict[tuple[ImplementationRef, OperationSpec, FrozenConfig], ImplementationHandle],
    inputs: tuple[AssociationInput, ...],
    authority: _RequestAuthority,
    control: _ExecutionControl,
    limits: ExecutionLimits,
    deferred_acceptances: dict[SemanticAssociation, _DeferredAcceptance],
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> dict[SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]]:
    if policy.request is None:
        reject(EffectCode.MISSING)
    semantic: set[SemanticAssociation] = set()
    for value in inputs:
        if not isinstance(value.association, SemanticAssociation):
            reject(EffectCode.CONTRADICTORY)
        semantic.add(value.association)
    associations = frozenset(semantic)
    for association in associations:
        authority.bind(RequestPolicyBinding.create(association=association, policies=frozenset({policy.request})))

    def failed(
        mapping: RuntimeOutcome,
    ) -> dict[
        SemanticAssociation, tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]
    ]:
        return {association: (mapping, (), ()) for association in associations}

    implementation_index = 0
    purpose: Literal["initial", "retry", "correction", "failover"] = "initial"
    while True:
        if len(authority.state.remote_outstanding) >= limits.max_remote_outstanding:
            return failed(_mapping(policy, "request_limit_exhausted", None, None))
        implementation = policy.implementations[implementation_index]
        handle = handles[
            (implementation.implementation, implementation.capability.operation, implementation.configuration)
        ]
        if handle.transport is None:
            reject(EffectCode.MISSING)
        request = PhysicalRequestId.new(scope=authority.state.scope)
        authority.apply(
            Reserve(
                request=request,
                purpose=purpose,
                associations=associations,
                policy=policy.request,
            ),
        )
        if not any(item.request == request for item in authority.state.reserved):
            denial = next(item for item in reversed(authority.state.denials) if item.request == request)
            condition = "budget_exhausted" if denial.category == "budget_stopped" else "request_limit_exhausted"
            return failed(_mapping(policy, condition, None, None))
        authority.apply(Dispatch(request=request))
        if handle.resource is not None:
            request_resources[request] = handle.resource.resource
        envelope = DispatchEnvelope(
            request=request,
            purpose=purpose,
            operation=handle.operation,
            associations=inputs,
        )
        dispatch = asyncio.create_task(handle.transport.dispatch(envelope))
        while not dispatch.done() and not control.cancelled:
            await asyncio.sleep(0)
        if control.cancelled and not dispatch.done():
            authority.apply(RequestCancel(request=request))
            try:
                stopped = await handle.transport.cancel(request)
            except Exception:
                stopped = None
            dispatch.cancel()
            with suppress(asyncio.CancelledError):
                await dispatch
            if isinstance(stopped, StopConfirmed):
                authority.apply(StopAcknowledged(request=request, usage=stopped.usage))
                return failed(_mapping(policy, "cancel_after_dispatch", None, None))
            authority.apply(MarkLost(request=request))
            return failed(_mapping(policy, "lost", None, None))
        try:
            result = dispatch.result()
        except Exception:
            result = TransportLost(settlement=None)
        if isinstance(result, TransportSuccess):
            if result.settlement is not None and result.settlement.request != request:
                failure = "malformed_response"
                authority.apply(AcceptFailure(request=request, failure=failure))
                if can_reserve_followup(
                    state=authority.state,
                    purpose="correction",
                    associations=associations,
                    policy=policy.request,
                ):
                    purpose = "correction"
                    continue
                return failed(_mapping(policy, "failure", None, failure))
            returned = [value.association for value in result.results]
            keyed = len(returned) == len(associations) and set(returned) == associations
            mappings: dict[SemanticAssociation, RuntimeOutcome] = {}
            if keyed:
                for row in result.results:
                    if (
                        not isinstance(row.association, SemanticAssociation)
                        or row.outcome not in policy.result_outcomes
                    ):
                        break
                    mapping = _mapping(policy, "result", row.outcome, None)
                    row_inputs = tuple(value for value in inputs if value.association == row.association)
                    if not _validate_output_shape(
                        admitted, handle.operation, mapping, row.association, row_inputs, (row,)
                    ):
                        break
                    mappings[row.association] = mapping
            if keyed and len(mappings) == len(associations):
                deferred = _DeferredAcceptance(
                    request=request,
                    results=result.results,
                    settlement=result.settlement,
                )
                for association in associations:
                    deferred_acceptances[association] = deferred
                return {
                    association: (
                        mappings[association],
                        tuple(row for row in result.results if row.association == association),
                        (),
                    )
                    for association in associations
                }
            if not keyed:
                authority.apply(AcceptResult(request=request, results=result.results))
                if result.settlement is not None:
                    authority.apply(ObserveSettlement(settlement=result.settlement))
                return failed(_mapping(policy, "request_inconsistent", None, None))
            failure: FailureClass = "malformed_response"
            authority.apply(AcceptFailure(request=request, failure=failure))
            if result.settlement is not None and result.settlement.request == request:
                authority.apply(ObserveSettlement(settlement=result.settlement))
        elif isinstance(result, TransportFailure):
            if result.settlement is not None and result.settlement.request != request:
                failure = "malformed_response"
                authority.apply(AcceptFailure(request=request, failure=failure))
                if can_reserve_followup(
                    state=authority.state,
                    purpose="correction",
                    associations=associations,
                    policy=policy.request,
                ):
                    purpose = "correction"
                    continue
                return failed(_mapping(policy, "failure", None, failure))
            failure = result.failure
            authority.apply(AcceptFailure(request=request, failure=failure))
            if result.settlement is not None:
                authority.apply(ObserveSettlement(settlement=result.settlement))
        else:
            authority.apply(MarkLost(request=request))
            if (
                isinstance(result, TransportLost)
                and result.settlement is not None
                and result.settlement.request == request
            ):
                authority.apply(ObserveSettlement(settlement=result.settlement))
            return failed(_mapping(policy, "lost", None, None))

        if failure == "malformed_response" and can_reserve_followup(
            state=authority.state,
            purpose="correction",
            associations=associations,
            policy=policy.request,
        ):
            purpose = "correction"
        elif (
            failure in {"permanent", "implementation_exception"}
            and implementation_index + 1 < len(policy.implementations)
            and can_reserve_followup(
                state=authority.state,
                purpose="failover",
                associations=associations,
                policy=policy.request,
            )
        ):
            implementation_index += 1
            purpose = "failover"
        elif policy.request.retry_owner == "executor" and can_reserve_followup(
            state=authority.state,
            purpose="retry",
            associations=associations,
            policy=policy.request,
        ):
            purpose = "retry"
        else:
            return failed(_mapping(policy, "failure", None, failure))


async def _run_adaptive(
    admitted: AdmittedExecutionPlan,
    policy: OperationExecutionPolicy,
    declaration: AdaptiveRetrievalDecl,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    authority: _RequestAuthority,
    leases: dict[object, ResourceLease],
    control: _ExecutionControl,
    limits: ExecutionLimits,
    deferred_acceptances: dict[SemanticAssociation, _DeferredAcceptance],
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    capability = next(
        item
        for item in admitted.context.context_capabilities
        if item.source == declaration.source and "adaptive_retrieval" in item.uses
    )
    lease = leases.get(declaration.source)
    if lease is None or not isinstance(lease.handle, ContextProvider) or policy.request is None:
        reject(EffectCode.MISSING)
    provider = lease.handle
    authority.bind(RequestPolicyBinding.create(association=association, policies=frozenset({capability.request})))
    source_inputs = {item.port: item.value for item in inputs[0].inputs}
    fields: list[SelectorField] = []
    for port in declaration.selector_ports:
        value = source_inputs.get(port)
        if not isinstance(value, TextArtifactValue):
            return _mapping(policy, "failure", None, "malformed_response"), (), ()
        fields.append(SelectorField(name=port, value=value.text))
    selector = ContextSelector(fields=tuple(fields))
    purpose: Literal["adaptive_retrieval", "retry", "correction"] = "adaptive_retrieval"
    while True:
        if len(authority.state.remote_outstanding) >= limits.max_remote_outstanding:
            return _mapping(policy, "request_limit_exhausted", None, None), (), ()
        request = PhysicalRequestId.new(scope=authority.state.scope)
        authority.apply(
            Reserve(
                request=request,
                purpose=purpose,
                associations=frozenset({association}),
                policy=capability.request,
            ),
        )
        if not any(item.request == request for item in authority.state.reserved):
            denial = next(item for item in reversed(authority.state.denials) if item.request == request)
            condition = "budget_exhausted" if denial.category == "budget_stopped" else "request_limit_exhausted"
            return _mapping(policy, condition, None, None), (), ()
        authority.apply(Dispatch(request=request))
        request_resources[request] = lease.resource
        retrieval = asyncio.create_task(
            provider.retrieve(
                request=request,
                association=association,
                selector=selector,
                bounds=declaration.bounds,
            )
        )
        while not retrieval.done() and not control.cancelled:
            await asyncio.sleep(0)
        if control.cancelled and not retrieval.done():
            authority.apply(RequestCancel(request=request))
            try:
                stopped = await provider.cancel(request)
            except Exception:
                stopped = None
            retrieval.cancel()
            late_result: SourceResponse | SourceFailure | SourceLost | None = None
            try:
                late_result = await retrieval
            except asyncio.CancelledError:
                pass
            if isinstance(stopped, StopConfirmed):
                authority.apply(StopAcknowledged(request=request, usage=stopped.usage))
                mapping = _mapping(policy, "cancel_after_dispatch", None, None)
            else:
                authority.apply(MarkLost(request=request))
                mapping = _mapping(policy, "lost", None, None)
            if isinstance(late_result, SourceResponse):
                valid_late_result = (
                    late_result.source == declaration.source
                    and bool(late_result.items)
                    and all(item.association == association for item in late_result.items)
                    and len({(item.key, item.version) for item in late_result.items}) == len(late_result.items)
                )
                if valid_late_result:
                    authority.apply(
                        AcceptResult(
                            request=request,
                            results=(
                                AssociationResult(
                                    association=association,
                                    outcome=_adaptive_success_outcome(policy),
                                    outputs=(),
                                    consumed_context_ports=frozenset(),
                                ),
                            ),
                        )
                    )
                else:
                    authority.apply(AcceptFailure(request=request, failure="malformed_response"))
            late_settlement = late_result.settlement if late_result is not None else None
            if late_settlement is not None and late_settlement.request == request:
                authority.apply(ObserveSettlement(settlement=late_settlement))
            return mapping, (), ()
        try:
            result = retrieval.result()
        except Exception:
            result = SourceLost(source=declaration.source, settlement=None)
        if isinstance(result, SourceResponse):
            valid = (
                result.source == declaration.source
                and result.settlement.request == request
                and bool(result.items)
                and all(item.association == association for item in result.items)
                and len({(item.key, item.version) for item in result.items}) == len(result.items)
            )
            byte_count = sum(len(item.text.encode()) for item in result.items)
            oversize = (
                len(result.items) > declaration.bounds.max_items
                or byte_count > declaration.bounds.max_bytes
                or (declaration.materialization.kind == "single" and len(result.items) > 1)
            )
            if valid and oversize:
                authority.apply(
                    AcceptResult(
                        request=request,
                        results=(
                            AssociationResult(
                                association=association,
                                outcome=_adaptive_success_outcome(policy),
                                outputs=(),
                                consumed_context_ports=frozenset(),
                            ),
                        ),
                    )
                )
                authority.apply(ObserveSettlement(settlement=result.settlement))
                return _mapping(policy, "artifact_limit_exhausted", None, None), (), ()
            if valid:
                value: ArtifactValue
                if declaration.materialization.kind == "single":
                    value = TextArtifactValue(text=result.items[0].text)
                else:
                    value = TextCollectionValue(
                        items=tuple(
                            TextCollectionItem(
                                key=item.key,
                                version=item.version,
                                value=TextArtifactValue(text=item.text),
                            )
                            for item in sorted(result.items, key=lambda item: (item.key, item.version))
                        )
                    )
                outcome = _adaptive_success_outcome(policy)
                operation = policy.implementations[0].capability.operation
                artifact_type = next(
                    item.artifact_type for item in operation.outputs if item.name == declaration.output_port
                )
                accepted = AssociationResult(
                    association=association,
                    outcome=outcome,
                    outputs=(
                        PortArtifact(
                            port=declaration.output_port,
                            artifact_type=artifact_type,
                            artifact=None,
                            value=value,
                        ),
                    ),
                    consumed_context_ports=frozenset(),
                )
                deferred_acceptances[association] = _DeferredAcceptance(
                    request=request,
                    results=(accepted,),
                    settlement=result.settlement,
                )
                return _mapping(policy, "result", outcome, None), (accepted,), ()
            failure: FailureClass = "malformed_response"
            authority.apply(AcceptFailure(request=request, failure=failure))
            if result.settlement.request == request:
                authority.apply(ObserveSettlement(settlement=result.settlement))
        elif isinstance(result, SourceFailure):
            failure = (
                result.failure
                if result.source == declaration.source
                and result.disposition == "failed"
                and (result.settlement is None or result.settlement.request == request)
                else "malformed_response"
            )
            authority.apply(AcceptFailure(request=request, failure=failure))
            if result.settlement is not None and result.settlement.request == request:
                authority.apply(ObserveSettlement(settlement=result.settlement))
        else:
            if isinstance(result, SourceLost) and result.source != declaration.source:
                authority.apply(AcceptFailure(request=request, failure="malformed_response"))
                if can_reserve_followup(
                    state=authority.state,
                    purpose="correction",
                    associations=frozenset({association}),
                    policy=capability.request,
                ):
                    purpose = "correction"
                    continue
                return _mapping(policy, "failure", None, "malformed_response"), (), ()
            authority.apply(MarkLost(request=request))
            if (
                isinstance(result, SourceLost)
                and result.settlement is not None
                and result.settlement.request == request
            ):
                authority.apply(ObserveSettlement(settlement=result.settlement))
            return _mapping(policy, "lost", None, None), (), ()
        if failure == "malformed_response" and can_reserve_followup(
            state=authority.state,
            purpose="correction",
            associations=frozenset({association}),
            policy=capability.request,
        ):
            purpose = "correction"
        elif capability.request.retry_owner == "executor" and can_reserve_followup(
            state=authority.state,
            purpose="retry",
            associations=frozenset({association}),
            policy=capability.request,
        ):
            purpose = "retry"
        else:
            return _mapping(policy, "failure", None, failure), (), ()


def _adaptive_success_outcome(policy: OperationExecutionPolicy) -> str:
    matches = [
        item
        for item in policy.implementations[0].capability.operation.outcomes
        if item.name in policy.result_outcomes and item.category == "success"
    ]
    if len(matches) != 1:
        reject(EffectCode.CONTRADICTORY)
    return matches[0].name


def _mapping(
    policy: OperationExecutionPolicy,
    condition: str,
    reported_outcome: str | None,
    failure: FailureClass | None,
) -> RuntimeOutcome:
    matches = [
        item
        for item in policy.runtime_outcomes
        if (item.condition, item.reported_outcome, item.failure) == (condition, reported_outcome, failure)
    ]
    if len(matches) != 1:
        reject(EffectCode.MISSING)
    return matches[0]


def _accept_outputs(
    admitted: AdmittedExecutionPlan,
    invocation: InvocationId,
    target: DatumId,
    activation: ActivationKey,
    node: NodeId,
    operation: OperationSpec,
    mapping: RuntimeOutcome,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    input_parents: dict[str, ProvenanceKey],
    results: tuple[AssociationResult, ...],
    facts: _ExecutionFacts,
    limits: ExecutionLimits,
) -> tuple[Literal["valid", "malformed", "limit"], tuple[ArtifactRef, ...]]:
    if not _validate_output_shape(admitted, operation, mapping, association, inputs, results):
        return "malformed", ()
    assert mapping.outcome is not None
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    expected = {item.name: item.artifact_type for item in operation.outputs if item.name in outcome.produced_ports}
    outputs = results[0].outputs
    if len(outputs) != len(expected) or {item.port for item in outputs} != set(expected):
        return "malformed", ()
    if any(item.artifact is not None or item.artifact_type != expected[item.port] for item in outputs):
        return "malformed", ()
    dependencies = {item.output: item for item in operation.output_dependencies if item.output in expected}
    new_outputs = [item for item in outputs if dependencies[item.port].identity_input is None]
    if len(facts.values) + len(new_outputs) > limits.max_runtime_artifacts:
        return "limit", ()
    if any(
        isinstance(item.value, TextCollectionValue) and len(item.value.items) > limits.max_collection_items
        for item in outputs
    ):
        return "limit", ()
    if (
        sum(_artifact_bytes(item) for item in facts.values.values())
        + sum(_artifact_bytes(item.value) for item in new_outputs)
        > limits.max_runtime_artifact_bytes
    ):
        return "limit", ()
    input_artifacts = {item.port: item.artifact for item in inputs[0].inputs if item.artifact is not None}
    planned_parents: dict[str, frozenset[ProvenanceKey]] = {}
    for output in outputs:
        dependency = dependencies[output.port]
        if not dependency.inputs <= input_artifacts.keys():
            return "malformed", ()
        parent_keys = {input_parents[name] for name in dependency.inputs if name in input_parents}
        if len(parent_keys) != len(dependency.inputs):
            return "malformed", ()
        planned_parents[output.port] = frozenset(parent_keys)
        if dependency.identity_input is not None:
            aliased = input_artifacts.get(dependency.identity_input)
            if aliased is None or facts.values[aliased] != output.value:
                return "malformed", ()
    limits_assessment = admitted.assessment_limits
    if len(facts.ports) + len(outputs) > limits_assessment.max_port_facts:
        return "limit", ()
    current_edges = sum(len(item.parents) for item in facts.provenance)
    if current_edges + sum(len(item) for item in planned_parents.values()) > limits_assessment.max_provenance_edges:
        return "limit", ()
    created: list[ArtifactRef] = []
    decision_output = any(item.node == node for item in admitted.decisions)
    for output in outputs:
        identity_input = dependencies[output.port].identity_input
        if identity_input is None:
            reference = ArtifactRef(invocation=invocation, key=facts.next_artifact, version=1)
            facts.next_artifact += 1
            facts.values[reference] = output.value
            created.append(reference)
        else:
            reference = input_artifacts[identity_input]
        facts.produced[(target, activation, output.port)] = reference
        key = OperationOutputKey(activation=activation, target=target, port=output.port)
        facts.provenance.append(
            ArtifactProvenanceFact(
                _key=_FACT_KEY,
                key=key,
                artifact=reference,
                parents=planned_parents[output.port],
                decision=decision_output,
            )
        )
        facts.ports.append(
            ExecutionPortFact(
                _key=_FACT_KEY,
                activation=activation,
                node=node,
                target=target,
                port=output.port,
                artifact=reference,
                artifact_type=output.artifact_type,
                role=admitted._output_role(node, outcome, output.port),
            )
        )
    return "valid", tuple(created)


def _validate_output_shape(
    admitted: AdmittedExecutionPlan,
    operation: OperationSpec,
    mapping: RuntimeOutcome,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    results: tuple[AssociationResult, ...],
) -> bool:
    if len(results) != 1 or results[0].association != association or mapping.outcome is None:
        return False
    outcome = next((item for item in operation.outcomes if item.name == mapping.outcome), None)
    if outcome is None:
        return False
    expected = {item.name: item.artifact_type for item in operation.outputs if item.name in outcome.produced_ports}
    outputs = results[0].outputs
    present_inputs = {item.port for item in inputs[0].inputs}
    schemas = _materialization_schemas(admitted)
    return (
        len(outputs) == len(expected)
        and {item.port for item in outputs} == set(expected)
        and all(
            item.artifact is None
            and item.artifact_type == expected[item.port]
            and (
                isinstance(item.value, TextCollectionValue)
                if schemas.get(item.artifact_type) == "collection"
                else isinstance(item.value, TextArtifactValue)
            )
            for item in outputs
        )
        and results[0].consumed_context_ports
        == frozenset(use.port for use in outcome.context if use.port in present_inputs)
    )


def _identity_input(dependency: OutputDependency) -> str:
    value = dependency.identity_input
    if value is None:
        reject(EffectCode.CONTRADICTORY)
    return value


def _materialization_schemas(admitted: AdmittedExecutionPlan) -> dict[ArtifactType, Literal["scalar", "collection"]]:
    schemas: dict[ArtifactType, Literal["scalar", "collection"]] = {}
    context = admitted.context.bound_context
    if context is not None:
        for fact in context.receipt.sources:
            schemas[fact.declaration.materialization.item_type] = "scalar"
            schemas[fact.declaration.artifact_type] = (
                "scalar" if fact.declaration.materialization.kind == "single" else "collection"
            )
    for declaration in admitted.context.adaptive_retrievals:
        operation = _operation_owner(
            admitted.context.prepared.workflow.workflow,
            declaration.node,
        )[1].operation
        artifact_type = next(item.artifact_type for item in operation.outputs if item.name == declaration.output_port)
        schemas[declaration.materialization.item_type] = "scalar"
        schemas[artifact_type] = "scalar" if declaration.materialization.kind == "single" else "collection"
    for declaration in admitted.map_expansions:
        operation = _node_operation(admitted.context.prepared.workflow.workflow, declaration.expander)
        artifact_type = next(
            item.artifact_type for item in operation.outputs if item.name == declaration.membership_port
        )
        schemas[declaration.item_type] = "scalar"
        schemas[artifact_type] = "collection"
    return schemas


def _materialize_bound_context(
    admitted: AdmittedExecutionPlan,
    invocation: InvocationId,
    values: dict[ArtifactRef, ArtifactValue],
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef],
    provenance: list[ArtifactProvenanceFact],
    next_artifact: int,
    limits: ExecutionLimits,
) -> int:
    context = admitted.context.bound_context
    if context is None:
        return next_artifact
    declarations = {item.identity: item.declaration for item in context.receipt.sources}
    groups: dict[BindingDeclarationId, list[BoundTextArtifact]] = {}
    for artifact in context.artifacts:
        groups.setdefault(artifact.reference.declaration, []).append(artifact)
    for identity, raw_items in groups.items():
        declaration = declarations[identity]
        items = sorted(raw_items, key=lambda item: (item.reference.key, item.reference.version))
        if declaration.materialization.kind == "collection" and len(items) > limits.max_collection_items:
            reject(EffectCode.LIMIT_EXCEEDED)
        item_refs: list[ArtifactRef] = []
        parents: set[ProvenanceKey] = set()
        for item in items:
            reference = ArtifactRef(invocation=invocation, key=next_artifact, version=item.reference.version)
            next_artifact += 1
            values[reference] = TextArtifactValue(text=item.text)
            key = BoundInputKey(
                target=item.target,
                node=item.node,
                port=item.port,
                binding_artifact=item.reference,
            )
            provenance.append(
                ArtifactProvenanceFact(_key=_FACT_KEY, key=key, artifact=reference, parents=frozenset(), decision=False)
            )
            item_refs.append(reference)
            parents.add(key)
        destination = (declaration.target, declaration.node, declaration.port)
        if declaration.materialization.kind == "single":
            if len(item_refs) != 1:
                reject(EffectCode.CONTRADICTORY)
            produced[destination] = item_refs[0]
        else:
            reference = ArtifactRef(invocation=invocation, key=next_artifact, version=1)
            next_artifact += 1
            values[reference] = TextCollectionValue(
                items=tuple(
                    TextCollectionItem(
                        key=item.reference.key,
                        version=item.reference.version,
                        value=TextArtifactValue(text=item.text),
                    )
                    for item in items
                )
            )
            key = InitialCollectionKey(
                target=declaration.target,
                node=declaration.node,
                port=declaration.port,
                declaration=identity,
            )
            provenance.append(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY, key=key, artifact=reference, parents=frozenset(parents), decision=False
                )
            )
            produced[destination] = reference
    if len(values) > limits.max_runtime_artifacts or sum(_artifact_bytes(item) for item in values.values()) > (
        limits.max_runtime_artifact_bytes
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    return next_artifact


def _artifact_bytes(value: ArtifactValue) -> int:
    if isinstance(value, TextArtifactValue):
        return len(value.text.encode())
    return sum(len(item.value.text.encode()) for item in value.items)


def _capture_assessments(
    admitted: AdmittedExecutionPlan,
    implementation: ExecutionImplementation,
    association: SemanticAssociation,
    activation: ActivationKey,
    node: NodeId,
    mapping: RuntimeOutcome,
    returned: tuple[LocalAssessmentResult, ...],
    target: DatumId,
    services: ExecutionServices,
    facts: _ExecutionFacts,
) -> bool:
    if mapping.outcome is None:
        return not returned
    declarations = [
        item for item in admitted.assessment_productions if item.node == node and item.outcome == mapping.outcome
    ]
    if len(returned) != len(declarations):
        return False
    if len(facts.assessments) + len(declarations) > admitted.assessment_limits.max_assessment_facts:
        return False
    absence_map = dict(services.absence_revisions)
    operation = implementation.capability.operation
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    reads = {item for item in outcome.state_effects if item.kind == "read"}
    state_view = StateRevisionView(
        revisions=frozenset(item for item in admitted.context.prepared.state.revisions if item.effect in reads)
    )
    for declaration in declarations:
        matches = [
            item
            for item in returned
            if item.association == association
            and item.promise == declaration.promise
            and item.evidence_port == declaration.evidence_port
            and item.finding in declaration.supported_findings
        ]
        artifact = facts.produced.get((target, activation, declaration.evidence_port))
        if len(matches) != 1 or artifact is None:
            return False
        facts.assessments.append(
            ExecutionAssessmentFact(
                _key=_FACT_KEY,
                activation=activation,
                node=node,
                outcome=mapping.outcome,
                promise=declaration.promise,
                evidence_artifact=artifact,
                finding=matches[0].finding,
                environment=AssessmentEnvironment(
                    configuration=implementation.configuration,
                    state=state_view,
                    absences=frozenset(
                        AbsenceRef(
                            invocation=activation.invocation,
                            query=query,
                            scope_revision=absence_map[query],
                        )
                        for query in declaration.absence_queries
                    ),
                ),
            )
        )
    return True


def _mark_assessment_subjects(
    admitted: AdmittedExecutionPlan,
    node: NodeId,
    mapping: RuntimeOutcome,
    target: DatumId,
    activation: ActivationKey,
    facts: _ExecutionFacts,
) -> None:
    if mapping.outcome is None:
        return
    declarations = [
        item for item in admitted.assessment_productions if item.node == node and item.outcome == mapping.outcome
    ]
    if not declarations:
        return
    operation = _node_operation(admitted.context.prepared.workflow.workflow, node)
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    promises = {item.name: item for item in outcome.evidence}
    for port in {promises[item.promise].subject_port for item in declarations}:
        indexes = [
            index
            for index, fact in enumerate(facts.ports)
            if fact.activation == activation and fact.node == node and fact.target == target and fact.port == port
        ]
        if len(indexes) != 1 or facts.ports[indexes[0]].role == "decision":
            reject(EffectCode.CONTRADICTORY)
        fact = facts.ports[indexes[0]]
        facts.ports[indexes[0]] = ExecutionPortFact(
            _key=_FACT_KEY,
            activation=fact.activation,
            node=fact.node,
            target=fact.target,
            port=fact.port,
            artifact=fact.artifact,
            artifact_type=fact.artifact_type,
            role="candidate",
        )


def _validate_assessment_returns(
    admitted: AdmittedExecutionPlan,
    association: SemanticAssociation,
    node: NodeId,
    mapping: RuntimeOutcome,
    returned: tuple[LocalAssessmentResult, ...],
) -> bool:
    if mapping.outcome is None:
        return not returned
    declarations = [
        item for item in admitted.assessment_productions if item.node == node and item.outcome == mapping.outcome
    ]
    if len(returned) != len(declarations):
        return False
    return all(
        len(
            [
                item
                for item in returned
                if item.association == association
                and item.promise == declaration.promise
                and item.evidence_port == declaration.evidence_port
                and item.finding in declaration.supported_findings
            ]
        )
        == 1
        for declaration in declarations
    )


def _final_outputs(
    prepared: PreparedPlan,
    states: tuple[ActivationState, ...],
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef],
    provenance: list[ArtifactProvenanceFact],
) -> tuple[FinalOutputFact, ...]:
    results: list[FinalOutputFact] = []
    for target_map, state in zip(prepared.target_occurrences, states, strict=True):
        for output_binding in prepared.workflow.workflow.output_bindings:
            root_anchor = next(
                (entry.activation for entry in state.entries if entry.activation.parent is None),
                None,
            )
            if root_anchor is None:
                continue
            producer: ProvenanceKey
            if isinstance(output_binding.source, WorkflowInputRef):
                producer = RootInputKey(target=target_map.target, port=output_binding.source.port)
                source_fact = next((fact for fact in provenance if fact.key == producer), None)
                if source_fact is None:
                    continue
                artifact = source_fact.artifact
            else:
                source_activation = _source_activation(state, root_anchor, output_binding.source.node)
                terminal = next(
                    (
                        entry
                        for entry in state.entries
                        if entry.activation == source_activation and entry.status == "success"
                    ),
                    None,
                )
                if terminal is None or terminal.outcome is None:
                    continue
                artifact = produced.get((target_map.target, terminal.activation, output_binding.source.port))
                if artifact is None:
                    continue
                producer = OperationOutputKey(
                    activation=terminal.activation,
                    target=target_map.target,
                    port=output_binding.source.port,
                )
                if not any(fact.key == producer and fact.artifact == artifact for fact in provenance):
                    reject(EffectCode.MISSING)
            workflow_outcome = None
            for binding in prepared.workflow.workflow.outcome_bindings:
                outcome_activation = _source_activation(state, root_anchor, binding.source.node)
                outcome_entry = next(
                    (
                        entry
                        for entry in state.entries
                        if entry.activation == outcome_activation
                        and entry.status == "success"
                        and entry.outcome == binding.source.outcome
                    ),
                    None,
                )
                if outcome_entry is not None:
                    workflow_outcome = binding.destination.outcome
                    break
            if workflow_outcome is None:
                continue
            results.append(
                FinalOutputFact(
                    _key=_FACT_KEY,
                    target=target_map.target,
                    outcome=workflow_outcome,
                    port=output_binding.destination.port,
                    candidate=CandidateRef(artifact=artifact, target=target_map.target),
                    producer=producer,
                )
            )
    return tuple(results)


def _acquire_context_resources(
    resources: tuple[ContextResource, ...],
) -> tuple[dict[object, ResourceLease], frozenset[object]]:
    leases: dict[object, ResourceLease] = {}
    failed: set[object] = set()
    for resource in resources:
        if resource.lease is not None:
            leases[resource.source] = resource.lease
            continue
        if resource.factory is None:
            reject(EffectCode.MISSING)
        try:
            provider = resource.factory()
        except Exception:
            failed.add(resource.source)
            continue
        if not isinstance(provider, ContextProvider):
            failed.add(resource.source)
            continue
        leases[resource.source] = ResourceLease.create(
            owner="sdk",
            safe_detachment=resource.capability.safe_detachment,
            handle=provider,
        )
    return leases, frozenset(failed)


async def _cleanup_execution(
    admitted: AdmittedExecutionPlan,
    handles: tuple[ImplementationHandle, ...],
    context_leases: dict[object, ResourceLease],
    requests: RequestState,
    request_resources: dict[PhysicalRequestId, ResourceId],
) -> tuple[tuple[CleanupFact, ...], tuple[CleanupAssociation, ...]]:
    leases = {item.resource.resource: item.resource for item in handles if item.resource is not None}
    leases.update({item.resource: item for item in context_leases.values()})
    local_resources = {
        request_resources[request] for request in requests.local_in_flight if request in request_resources
    }
    remote_resources = {
        request_resources[request] for request in requests.remote_outstanding if request in request_resources
    }
    cleanup_values: list[CleanupFact] = []
    for lease in leases.values():
        if lease.owner == "sdk" and (
            lease.resource in local_resources
            or (lease.resource in remote_resources and lease.safe_detachment == "forbidden")
        ):
            cleanup_values.append(CleanupFact(resource=lease.resource, owner=lease.owner, disposition="left_open"))
        else:
            cleanup_values.append(await close_resource(lease))
    cleanup = tuple(cleanup_values)
    all_targets = admitted.context.prepared.data.targets
    external_resources = {
        handle.resource.resource
        for policy in admitted.policies
        if policy.kind == "external"
        for implementation in policy.implementations
        for handle in handles
        if handle.resource is not None
        and handle.implementation == implementation.implementation
        and handle.operation == implementation.capability.operation
        and handle.configuration == implementation.configuration
    }
    context_resource_ids = {item.resource for item in context_leases.values()}
    context_targets = {
        lease.resource: frozenset(
            target.target
            for target in admitted.context.prepared.target_occurrences
            if any(item.source == source for item in admitted.context.adaptive_retrievals)
        )
        for source, lease in context_leases.items()
    }
    associations = tuple(
        CleanupAssociation(
            _key=_FACT_KEY,
            resource=resource,
            targets=context_targets.get(resource, all_targets),
            purpose="accounting" if resource in external_resources | context_resource_ids else "verification",
        )
        for resource in leases
    )
    return cleanup, associations


def _canonical_record(
    prepared: PreparedPlan,
    invocation: InvocationId,
    states: tuple[ActivationState, ...],
    target_keys: dict[DatumId, dict[int, ActivationKey]],
    attempts: dict[ActivationKey, TaskAttemptId],
    artifacts: frozenset[ArtifactRef],
    cancelled_unstarted: frozenset[ActivationKey],
) -> CanonicalRecord:
    del target_keys
    all_entries = [entry for state in states for entry in state.entries]
    root_members = frozenset(entry.activation for entry in all_entries if entry.activation.parent is None)
    memberships: list[ExpectedMembership] = [
        ExpectedMembership(
            invocation=invocation,
            parent=None,
            members=root_members,
            closed=all(state.complete for state in states),
        )
    ]
    for state in states:
        child_parents = {entry.activation.parent for entry in state.entries if entry.activation.parent is not None}
        child_parents.update(expansion.parent for expansion in state.expansions)
        for parent in sorted(child_parents, key=lambda item: item.occurrence):
            assert parent is not None
            expansion = next((item for item in state.expansions if item.parent == parent), None)
            parent_entry = next((item for item in state.entries if item.activation == parent), None)
            memberships.append(
                ExpectedMembership(
                    invocation=invocation,
                    parent=parent,
                    members=frozenset(entry.activation for entry in state.entries if entry.activation.parent == parent),
                    closed=(
                        expansion.status != "pending"
                        if expansion is not None
                        else parent_entry is not None
                        and parent_entry.status
                        in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}
                    ),
                )
            )
    terminals: list[TerminalFact] = []
    for state in states:
        for entry in state.entries:
            if entry.status not in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}:
                continue
            structural = _is_subgraph_node(prepared.workflow.workflow, entry.template)
            reasons = {
                "failure": frozenset({"execution_failed"}),
                "cancelled": frozenset({"cancel_requested"}),
                "lost": frozenset({"transport_lost"}),
                "blocked": frozenset(
                    {"cancel_requested" if entry.activation in cancelled_unstarted else "prerequisite"}
                ),
                "inconsistent": frozenset({"contradictory"}),
            }.get(entry.status, frozenset())
            terminals.append(
                TerminalFact(
                    activation=entry.activation,
                    attempt=attempts.get(entry.activation),
                    category=entry.status,
                    reasons=reasons,
                    structural=structural,
                )
            )
    statuses = tuple(
        TargetStatus(
            target=target,
            completion="closed" if state.complete else "pending",
            qualification="not_assessed",
            artifact_available=bool(artifacts),
            protection_available=False,
        )
        for target, state in zip((item.target for item in prepared.target_occurrences), states, strict=True)
    )
    return CanonicalRecord(
        plan=prepared.plan,
        invocation=invocation,
        graph=prepared.data.graph,
        targets=prepared.data.targets,
        memberships=tuple(memberships),
        terminals=tuple(terminals),
        artifacts=artifacts,
        evidence=(),
        statuses=statuses,
    )
