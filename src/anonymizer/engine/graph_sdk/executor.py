# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single asynchronous executor for admitted graph plans."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
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
    AdmittedContextPlan,
    BoundTextArtifact,
    ContextResource,
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
    FailureClass,
    InvocationRequestScope,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    RequestPolicyBinding,
    RequestReceipt,
    RequestState,
    Reserve,
    ScopeCancel,
    SemanticAssociation,
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
from anonymizer.graph._values import ActivationKey, ArtifactRef, DatumId, InvocationId, TaskAttemptId
from anonymizer.graph.activation import (
    ActivationSeed,
    ActivationState,
    CloseUnstarted,
    ObserveTerminal,
    Select,
    Start,
    advance_activation,
    initialize_activation,
)
from anonymizer.graph.workflow import (
    AdmittedWorkflow,
    ArtifactType,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeClass,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
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
    binding_artifact: object

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_instance(self.node, NodeId)
        require_text(self.port)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class InitialCollectionKey(PrivateValue):
    target: DatumId
    node: NodeId
    port: str
    declaration: object

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_instance(self.node, NodeId)
        require_text(self.port)


ProvenanceKey: TypeAlias = OperationOutputKey | RootInputKey | BoundInputKey | InitialCollectionKey

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
    for name, field in fields.items():
        object.__setattr__(value, name, field)


_PLAN_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class AdmittedExecutionPlan(PrivateValue):
    context: AdmittedContextPlan
    capabilities: tuple[ImplementationCapability, ...]
    policies: frozenset[OperationExecutionPolicy]
    decisions: frozenset[DecisionDeclaration]
    assessment_productions: tuple[EvidenceProductionDecl, ...]
    assessment_limits: AssessmentLimits

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _PLAN_KEY:
            raise TypeError("execution plans are created by admission")
        for name, value in values.items():
            object.__setattr__(self, name, value)


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


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DecisionResponse(PrivateValue):
    wait: DecisionWaitId
    workflow: WorkflowId
    artifact: ArtifactRef
    decision: str


_RESULT_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ExecutionResult(PrivateValue):
    record: CanonicalRecord
    states: tuple[ActivationState, ...]
    requests: RequestReceipt
    cleanup: tuple[CleanupFact, ...]
    pending_decisions: tuple[DecisionWait, ...]
    artifacts: tuple[tuple[ArtifactRef, TextArtifactValue], ...]
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
    assessment_productions: tuple[EvidenceProductionDecl, ...] = (),
    assessment_limits: AssessmentLimits | None = None,
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
    selected = {item.node: item.capability for item in prepared.implementations}
    if {item.node for item in policies} != set(selected):
        reject(EffectCode.MISSING)
    if len(policies) != len({item.node for item in policies}):
        reject(EffectCode.DUPLICATE)
    decision_nodes = {item.node for item in decisions}
    if len(decisions) != len(decision_nodes):
        reject(EffectCode.DUPLICATE)
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
    actual_limits = assessment_limits or AssessmentLimits(
        max_productions=0,
        max_findings_per_production=0,
        max_finding_code_bytes=0,
        max_absence_queries=0,
        max_assessment_facts=0,
        max_port_facts=0,
        max_provenance_edges=0,
    )
    return AdmittedExecutionPlan(
        _key=_PLAN_KEY,
        context=context,
        capabilities=capabilities,
        policies=frozenset(policies),
        decisions=frozenset(decisions),
        assessment_productions=assessment_productions,
        assessment_limits=actual_limits,
    )


def _validate_policy(policy: OperationExecutionPolicy, has_decision: bool) -> None:
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
    if (policy.kind == "decision") != has_decision:
        reject(EffectCode.CONTRADICTORY)
    outcomes = {item.name: item.category for item in operation.outcomes}
    if policy.kind != "decision" and (not policy.result_outcomes or not policy.result_outcomes <= outcomes.keys()):
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
        if item.outcome is None and item.category == "success":
            reject(EffectCode.CONTRADICTORY)
        if item.condition == "cancel_before_start" and (item.outcome is not None or item.category != "blocked"):
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
    limits: AssessmentLimits | None,
) -> None:
    del prepared
    if not isinstance(productions, tuple) or any(not isinstance(item, EvidenceProductionDecl) for item in productions):
        reject(EffectCode.INVALID_TYPE)
    if productions and limits is None:
        reject(EffectCode.MISSING)
    if limits is None:
        return
    require_instance(limits, AssessmentLimits)
    if len(productions) > limits.max_productions:
        reject(EffectCode.LIMIT_EXCEEDED)
    policy_by_node = {item.node: item for item in policies}
    for item in productions:
        policy = policy_by_node.get(item.node)
        if policy is None or policy.kind != "local":
            reject(EffectCode.UNSUPPORTED)
        operation = policy.implementations[0].capability.operation
        outcome = next((value for value in operation.outcomes if value.name == item.outcome), None)
        if outcome is None or item.evidence_port not in outcome.produced_ports:
            reject(EffectCode.UNSUPPORTED)
        promises = {value.name for value in outcome.evidence}
        if item.promise not in promises:
            reject(EffectCode.UNSUPPORTED)
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
    decisions: list[DecisionResponse] | None = None


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
        return ()

    def submit_decision(self, decision: DecisionResponse) -> None:
        require_instance(decision, DecisionResponse)
        if self._control.decisions is None:
            self._control.decisions = []
        self._control.decisions.append(decision)

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
    control = _ExecutionControl(decisions=[])
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


async def _run_execution(
    admitted: AdmittedExecutionPlan,
    services: ExecutionServices,
    invocation: InvocationId,
    control: _ExecutionControl,
) -> ExecutionResult:
    prepared = admitted.context.prepared
    policies = {item.node: item for item in admitted.policies}
    handles = {(item.implementation, item.operation, item.configuration): item for item in services.handles}
    states: list[ActivationState] = []
    target_keys: dict[DatumId, dict[int, ActivationKey]] = {}
    for target_map in prepared.target_occurrences:
        keys: dict[int, ActivationKey] = {}
        for slot in prepared.reservation_recipe:
            keys[slot.index] = ActivationKey(
                invocation=invocation,
                occurrence=target_map.occurrence_offset + slot.index,
                parent=keys.get(slot.parent_index),
                iteration=slot.iteration,
            )
        target_keys[target_map.target] = keys
        reservations = frozenset(
            ActivationSeed(template=slot.template, activation=keys[slot.index]) for slot in prepared.reservation_recipe
        )
        initialized = initialize_activation(
            workflow=prepared.workflow,
            invocation=invocation,
            reservations=reservations,
            limits=prepared.activation_limits,
        )
        states.append(
            advance_activation(
                state=initialized,
                event=Select(seeds=_initial_seeds(prepared.workflow.workflow, reservations)),
            )
        )
    scope = InvocationRequestScope(invocation=invocation)
    request_state = initialize_requests(
        scope=scope,
        hard_limit=prepared.configuration.hard_request_limit,
        policies=frozenset(item.request for item in admitted.policies if item.request is not None),
    )
    artifact_values: dict[ArtifactRef, ArtifactValue] = {}
    root_inputs: dict[tuple[DatumId, str], ArtifactRef] = {}
    provenance: list[ArtifactProvenanceFact] = []
    next_artifact = 0
    datum_text = {item.id: item.text for item in prepared.data.datums}
    for bound in prepared.bound_inputs:
        reference = ArtifactRef(invocation=invocation, key=next_artifact, version=1)
        next_artifact += 1
        artifact_values[reference] = TextArtifactValue(text=datum_text[bound.source])
        root_inputs[(bound.target, bound.port)] = reference
        provenance.append(
            ArtifactProvenanceFact(
                _key=_FACT_KEY,
                key=RootInputKey(target=bound.target, port=bound.port),
                artifact=reference,
                parents=frozenset(),
                decision=False,
            )
        )
    produced: dict[tuple[DatumId, NodeId, str], ArtifactRef] = {}
    next_artifact = _materialize_bound_context(
        admitted,
        invocation,
        artifact_values,
        produced,
        provenance,
        next_artifact,
        services.limits,
    )
    attempts: dict[ActivationKey, TaskAttemptId] = {}
    port_facts: list[ExecutionPortFact] = []
    assessment_facts: list[ExecutionAssessmentFact] = []
    made_progress = True
    while made_progress:
        made_progress = False
        for state_index, (target_map, state) in enumerate(zip(prepared.target_occurrences, states, strict=True)):
            ready = sorted(
                (item for item in state.entries if item.status == "ready"),
                key=lambda item: item.activation.occurrence,
            )
            for entry in ready:
                policy = policies.get(entry.template)
                if policy is None:
                    continue
                made_progress = True
                if control.cancelled:
                    states[state_index] = advance_activation(
                        state=states[state_index],
                        event=CloseUnstarted(activation=entry.activation, category="blocked"),
                    )
                    continue
                states[state_index] = advance_activation(
                    state=states[state_index], event=Start(activation=entry.activation)
                )
                attempt = TaskAttemptId.new(activation=entry.activation)
                attempts[entry.activation] = attempt
                association = SemanticAssociation(task=attempt)
                inputs = _operation_inputs(
                    admitted,
                    target_map.target,
                    entry.template,
                    association,
                    root_inputs,
                    produced,
                    artifact_values,
                )
                for input_artifact in inputs[0].inputs:
                    if input_artifact.artifact is None:
                        reject(EffectCode.CONTRADICTORY)
                    port_facts.append(
                        ExecutionPortFact(
                            _key=_FACT_KEY,
                            activation=entry.activation,
                            node=entry.template,
                            target=target_map.target,
                            port=input_artifact.port,
                            artifact=input_artifact.artifact,
                            artifact_type=input_artifact.artifact_type,
                            role="artifact",
                        )
                    )
                implementation = policy.implementations[0]
                handle = handles[
                    (implementation.implementation, implementation.capability.operation, implementation.configuration)
                ]
                if policy.kind == "external":
                    mapping, results, request_state = await _run_external(
                        policy, handle, association, inputs, request_state
                    )
                    assessments: tuple[LocalAssessmentResult, ...] = ()
                else:
                    mapping, results, assessments = await _run_local(policy, handle, association, inputs)
                if mapping.condition == "result":
                    valid, created, next_artifact = _accept_outputs(
                        invocation,
                        target_map.target,
                        entry.activation,
                        entry.template,
                        implementation.capability.operation,
                        mapping,
                        association,
                        inputs,
                        results,
                        artifact_values,
                        produced,
                        provenance,
                        port_facts,
                        next_artifact,
                        services.limits,
                    )
                    if not valid:
                        mapping = _mapping(policy, "failure", None, "malformed_response")
                    elif not _capture_assessments(
                        admitted,
                        implementation,
                        association,
                        entry.activation,
                        entry.template,
                        mapping,
                        assessments,
                        produced,
                        target_map.target,
                        services,
                        assessment_facts,
                    ):
                        mapping = _mapping(policy, "failure", None, "malformed_response")
                    del created
                states[state_index] = advance_activation(
                    state=states[state_index],
                    event=ObserveTerminal(
                        activation=entry.activation,
                        outcome=mapping.outcome,
                        category=mapping.category,
                    ),
                )
        if all(state.complete for state in states):
            break
    if control.cancelled:
        request_state = advance_requests(state=request_state, event=ScopeCancel())
    cleanup = await _cleanup_execution(services.handles)
    record = _canonical_record(
        prepared,
        invocation,
        tuple(states),
        target_keys,
        attempts,
        frozenset(artifact_values),
    )
    final_outputs = _final_outputs(prepared, tuple(states), produced, provenance)
    return ExecutionResult(
        _key=_RESULT_KEY,
        record=record,
        states=tuple(states),
        requests=request_receipt(request_state),
        cleanup=cleanup,
        pending_decisions=(),
        artifacts=tuple(artifact_values.items()),
        assessments=tuple(assessment_facts),
        ports=tuple(port_facts),
        final_outputs=final_outputs,
        provenance=tuple(provenance),
        cleanup_associations=(),
    )


def _initial_seeds(workflow: AdmittedWorkflow, reservations: frozenset[ActivationSeed]) -> frozenset[ActivationSeed]:
    blocked = {edge.after for edge in workflow.sequence}
    blocked.update(member for choice in workflow.choices for branch in choice.branches for member in branch.members)
    return frozenset(seed for seed in reservations if seed.activation.parent is None and seed.template not in blocked)


def _operation_inputs(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    node: NodeId,
    association: SemanticAssociation,
    root_inputs: dict[tuple[DatumId, str], ArtifactRef],
    produced: dict[tuple[DatumId, NodeId, str], ArtifactRef],
    values: dict[ArtifactRef, ArtifactValue],
) -> tuple[AssociationInput, ...]:
    workflow, operation_node = _operation_owner(admitted.context.prepared.workflow.workflow, node)
    operation = operation_node.operation
    ports: list[PortArtifact] = []
    for port in operation.inputs:
        source = next(
            (
                binding.source
                for binding in workflow.input_bindings
                if binding.destination == NodeInputRef(node=node, port=port.name)
            ),
            None,
        )
        reference: ArtifactRef | None = None
        if isinstance(source, WorkflowInputRef):
            reference = root_inputs.get((target, source.port))
        elif isinstance(source, NodeOutputRef):
            reference = produced.get((target, source.node, source.port))
        if reference is None:
            reference = produced.get((target, node, port.name))
        if reference is None:
            reject(EffectCode.MISSING)
        ports.append(
            PortArtifact(
                port=port.name,
                artifact_type=port.artifact_type,
                artifact=reference,
                value=values[reference],
            )
        )
    return (AssociationInput(association=association, inputs=tuple(ports)),)


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


async def _run_local(
    policy: OperationExecutionPolicy,
    handle: ImplementationHandle,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], tuple[LocalAssessmentResult, ...]]:
    if handle.local is None:
        reject(EffectCode.MISSING)
    try:
        result = await handle.local.run(inputs)
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


async def _run_external(
    policy: OperationExecutionPolicy,
    handle: ImplementationHandle,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    state: RequestState,
) -> tuple[RuntimeOutcome, tuple[AssociationResult, ...], RequestState]:
    if policy.request is None or handle.transport is None:
        reject(EffectCode.MISSING)
    state = bind_request_policies(
        state=state,
        binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy.request})),
    )
    request = PhysicalRequestId.new(scope=state.scope)
    state = advance_requests(
        state=state,
        event=Reserve(
            request=request,
            purpose="initial",
            associations=frozenset({association}),
            policy=policy.request,
        ),
    )
    if not any(item.request == request for item in state.reserved):
        denial = next(item for item in reversed(state.denials) if item.request == request)
        condition = "budget_exhausted" if denial.category == "budget_stopped" else "request_limit_exhausted"
        return _mapping(policy, condition, None, None), (), state
    state = advance_requests(state=state, event=Dispatch(request=request))
    envelope = DispatchEnvelope(
        request=request,
        purpose="initial",
        operation=handle.operation,
        associations=inputs,
    )
    try:
        result = await handle.transport.dispatch(envelope)
    except Exception:
        result = TransportLost(settlement=None)
    if isinstance(result, TransportSuccess):
        state = advance_requests(state=state, event=AcceptResult(request=request, results=result.results))
        if result.settlement is not None:
            state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
        terminal = next(item for item in state.terminals if item.request == request)
        if terminal.category == "inconsistent":
            return _mapping(policy, "request_inconsistent", None, None), (), state
        if len(result.results) != 1 or result.results[0].association != association:
            return _mapping(policy, "request_inconsistent", None, None), (), state
        reported = result.results[0].outcome
        if reported not in policy.result_outcomes:
            return _mapping(policy, "failure", None, "malformed_response"), (), state
        return _mapping(policy, "result", reported, None), result.results, state
    if isinstance(result, TransportFailure):
        state = advance_requests(state=state, event=AcceptFailure(request=request, failure=result.failure))
        if result.settlement is not None:
            state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
        return _mapping(policy, "failure", None, result.failure), (), state
    state = advance_requests(state=state, event=MarkLost(request=request))
    if isinstance(result, TransportLost) and result.settlement is not None:
        state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
    return _mapping(policy, "lost", None, None), (), state


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
    invocation: InvocationId,
    target: DatumId,
    activation: ActivationKey,
    node: NodeId,
    operation: OperationSpec,
    mapping: RuntimeOutcome,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    results: tuple[AssociationResult, ...],
    values: dict[ArtifactRef, ArtifactValue],
    produced: dict[tuple[DatumId, NodeId, str], ArtifactRef],
    provenance: list[ArtifactProvenanceFact],
    port_facts: list[ExecutionPortFact],
    next_artifact: int,
    limits: ExecutionLimits,
) -> tuple[bool, tuple[ArtifactRef, ...], int]:
    if len(results) != 1 or results[0].association != association or mapping.outcome is None:
        return False, (), next_artifact
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    expected = {item.name: item.artifact_type for item in operation.outputs if item.name in outcome.produced_ports}
    outputs = results[0].outputs
    if len(outputs) != len(expected) or {item.port for item in outputs} != set(expected):
        return False, (), next_artifact
    if any(item.artifact is not None or item.artifact_type != expected[item.port] for item in outputs):
        return False, (), next_artifact
    if len(values) + len(outputs) > limits.max_runtime_artifacts:
        return False, (), next_artifact
    if any(
        isinstance(item.value, TextCollectionValue) and len(item.value.items) > limits.max_collection_items
        for item in outputs
    ):
        return False, (), next_artifact
    if (
        sum(_artifact_bytes(item) for item in values.values()) + sum(_artifact_bytes(item.value) for item in outputs)
        > limits.max_runtime_artifact_bytes
    ):
        return False, (), next_artifact
    created: list[ArtifactRef] = []
    for output in outputs:
        reference = ArtifactRef(invocation=invocation, key=next_artifact, version=1)
        next_artifact += 1
        values[reference] = output.value
        produced[(target, node, output.port)] = reference
        key = OperationOutputKey(activation=activation, target=target, port=output.port)
        dependency = next(item for item in operation.output_dependencies if item.output == output.port)
        input_artifacts = {item.port: item.artifact for item in inputs[0].inputs if item.artifact is not None}
        parent_artifacts = {input_artifacts[name] for name in dependency.inputs}
        parent_keys = {fact.key for fact in provenance if fact.artifact in parent_artifacts}
        if len(parent_keys) != len(parent_artifacts):
            return False, (), next_artifact
        provenance.append(
            ArtifactProvenanceFact(
                _key=_FACT_KEY, key=key, artifact=reference, parents=frozenset(parent_keys), decision=False
            )
        )
        port_facts.append(
            ExecutionPortFact(
                _key=_FACT_KEY,
                activation=activation,
                node=node,
                target=target,
                port=output.port,
                artifact=reference,
                artifact_type=output.artifact_type,
                role="artifact",
            )
        )
        created.append(reference)
    return True, tuple(created), next_artifact


def _materialize_bound_context(
    admitted: AdmittedExecutionPlan,
    invocation: InvocationId,
    values: dict[ArtifactRef, ArtifactValue],
    produced: dict[tuple[DatumId, NodeId, str], ArtifactRef],
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
        if len(items) > limits.max_collection_items:
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
    produced: dict[tuple[DatumId, NodeId, str], ArtifactRef],
    target: DatumId,
    services: ExecutionServices,
    facts: list[ExecutionAssessmentFact],
) -> bool:
    if mapping.outcome is None:
        return not returned
    declarations = [
        item for item in admitted.assessment_productions if item.node == node and item.outcome == mapping.outcome
    ]
    if len(returned) != len(declarations):
        return False
    absence_map = dict(services.absence_revisions)
    for declaration in declarations:
        matches = [
            item
            for item in returned
            if item.association == association
            and item.promise == declaration.promise
            and item.evidence_port == declaration.evidence_port
            and item.finding in declaration.supported_findings
        ]
        artifact = produced.get((target, node, declaration.evidence_port))
        if len(matches) != 1 or artifact is None:
            return False
        facts.append(
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
                    state=admitted.context.prepared.state,
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


def _final_outputs(
    prepared: PreparedPlan,
    states: tuple[ActivationState, ...],
    produced: dict[tuple[DatumId, NodeId, str], ArtifactRef],
    provenance: list[ArtifactProvenanceFact],
) -> tuple[FinalOutputFact, ...]:
    results: list[FinalOutputFact] = []
    for target_map, state in zip(prepared.target_occurrences, states, strict=True):
        for output_binding in prepared.workflow.workflow.output_bindings:
            if not isinstance(output_binding.source, NodeOutputRef):
                continue
            terminal = next(
                (
                    entry
                    for entry in state.entries
                    if entry.template == output_binding.source.node and entry.status == "success"
                ),
                None,
            )
            if terminal is None or terminal.outcome is None:
                continue
            artifact = produced.get((target_map.target, output_binding.source.node, output_binding.source.port))
            if artifact is None:
                continue
            producer = next(
                (
                    fact.key
                    for fact in provenance
                    if fact.artifact == artifact and isinstance(fact.key, OperationOutputKey)
                ),
                None,
            )
            if producer is None:
                reject(EffectCode.MISSING)
            results.append(
                FinalOutputFact(
                    _key=_FACT_KEY,
                    target=target_map.target,
                    outcome=terminal.outcome,
                    port=output_binding.destination.port,
                    candidate=CandidateRef(artifact=artifact, target=target_map.target),
                    producer=producer,
                )
            )
    return tuple(results)


async def _cleanup_execution(handles: tuple[ImplementationHandle, ...]) -> tuple[CleanupFact, ...]:
    leases = {item.resource.resource: item.resource for item in handles if item.resource is not None}
    return tuple([await close_resource(lease) for lease in leases.values()])


def _canonical_record(
    prepared: PreparedPlan,
    invocation: InvocationId,
    states: tuple[ActivationState, ...],
    target_keys: dict[DatumId, dict[int, ActivationKey]],
    attempts: dict[ActivationKey, TaskAttemptId],
    artifacts: frozenset[ArtifactRef],
) -> CanonicalRecord:
    all_keys = {key for values in target_keys.values() for key in values.values()}
    parents = {key.parent for key in all_keys}
    memberships = tuple(
        ExpectedMembership(
            invocation=invocation,
            parent=parent,
            members=frozenset(key for key in all_keys if key.parent == parent),
            closed=True,
        )
        for parent in parents
    )
    terminals: list[TerminalFact] = []
    for state in states:
        for entry in state.entries:
            if entry.status not in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}:
                continue
            reasons = {
                "failure": frozenset({"execution_failed"}),
                "cancelled": frozenset({"cancel_requested"}),
                "lost": frozenset({"transport_lost"}),
                "blocked": frozenset({"cancel_requested" if entry.activation not in attempts else "prerequisite"}),
                "inconsistent": frozenset({"contradictory"}),
            }.get(entry.status, frozenset())
            terminals.append(
                TerminalFact(
                    activation=entry.activation,
                    attempt=attempts.get(entry.activation),
                    category=entry.status,
                    reasons=reasons,
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
        memberships=memberships,
        terminals=tuple(terminals),
        artifacts=artifacts,
        evidence=(),
        statuses=statuses,
    )
