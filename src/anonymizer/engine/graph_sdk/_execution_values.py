# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution declarations and sealed result values."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar, dataclass, field, fields, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, Protocol, TypeAlias, runtime_checkable

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    EffectRejected,
    PrivateValue,
    reject,
    require_count,
    require_instance,
    require_literal,
    require_owner,
    require_text,
)
from anonymizer.engine.graph_sdk._ingress_values import IngressLimits
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationCapability,
    ImplementationRef,
)
from anonymizer.engine.graph_sdk.context import (
    AdmittedContextPlan,
    BindingArtifactRef,
)
from anonymizer.engine.graph_sdk.observations import NodeMeasurement
from anonymizer.engine.graph_sdk.records import (
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
    TextCollectionValue,
    TransportResult,
)
from anonymizer.engine.graph_sdk.resources import (
    CleanupAssociation,
    CleanupFact,
    ResourceId,
    ResourceLease,
    ResourceOwner,
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
    Graph,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    Operation,
    OperationSpec,
    OutcomeClass,
    OutcomeSpec,
    WorkflowInputRef,
)

if TYPE_CHECKING:
    import pandas as pd

    from anonymizer.engine.detection._frame_publication import FramePublication, InvocationFrame, PublishedFrame
    from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame


ExecutionKind: TypeAlias = Literal["local", "external", "delegated"]


ArtifactRole: TypeAlias = Literal["artifact", "candidate", "evidence"]


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
class DelegatedReadiness(PrivateValue):
    """One semantic readiness obligation, physical child and local probe."""

    owner_slot: int
    request_slot: int
    probe_slot: int
    route: ChildRoute
    ready_outcome: str
    probe_outcomes: frozenset[str]

    def __post_init__(self) -> None:
        for slot in self.slots:
            require_count(slot)
        if len(set(self.slots)) != 3:
            reject(EffectCode.DUPLICATE)
        require_instance(self.route, ChildRoute)
        self.route.__post_init__()
        if self.route.parse_attempts != 1:
            reject(EffectCode.UNSUPPORTED)
        require_text(self.ready_outcome)
        if type(self.probe_outcomes) is not frozenset or not 1 <= len(self.probe_outcomes) <= 2:
            reject(EffectCode.INVALID_VALUE)
        for outcome in self.probe_outcomes:
            require_text(outcome)

    @property
    def slots(self) -> tuple[int, ...]:
        return self.owner_slot, self.request_slot, self.probe_slot


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DelegatedGenerationSelection(PrivateValue):
    """A source-owned collection selects a prefix of singleton generation slots."""

    port: str
    slots: tuple[int, ...]

    def __post_init__(self) -> None:
        require_text(getattr(self, "port", None))
        require_instance(getattr(self, "slots", None), tuple)
        for slot in self.slots:
            require_count(slot)
        if not self.slots or len(set(self.slots)) != len(self.slots):
            reject(EffectCode.INVALID_VALUE)

    def omitted(self, value: object) -> tuple[int, ...]:
        from anonymizer.engine.graph_sdk._ingress import inspect_artifact

        if type(value) is not TextCollectionValue or not inspect_artifact(value) or len(value.items) > len(self.slots):
            reject(EffectCode.CONTRADICTORY)
        if tuple((item.key, item.version) for item in value.items) != tuple((i, 1) for i in range(len(value.items))):
            reject(EffectCode.CONTRADICTORY)
        return self.slots[len(value.items) :]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DelegatedRequestContract(PrivateValue):
    """Fixed recipe declaration for bounded delegated requests."""

    mode: Literal["qualified", "opaque"]
    child_slots: tuple[int, ...]
    routes: tuple[ChildRoute, ...]
    followups: frozenset[ChildSourceKind] = frozenset({"correction", "failover"})
    usage_requirement: Literal["exact", "unknown_allowed"] = "exact"
    readiness: DelegatedReadiness | None = None
    optional_tail: tuple[int, ...] = ()
    generation_groups: tuple[tuple[int, ...], ...] = ()
    selection: DelegatedGenerationSelection | None = None
    publication_sources: tuple[tuple[NodeId, frozenset[str]], ...] = ()
    observed_request: PhysicalRequestPolicy | None = None
    _accepted_publication_sources: tuple[tuple[NodeId, frozenset[str]], ...] = field(
        init=False, repr=False, compare=False
    )
    _accepted_observed_declaration: tuple[object, ...] | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.observed_request is not None:
            require_instance(self.observed_request, PhysicalRequestPolicy)
            if self.mode != "opaque":
                reject(EffectCode.CONTRADICTORY)
        if type(self.publication_sources) is not tuple:
            reject(EffectCode.INVALID_TYPE)
        for item in self.publication_sources:
            if (
                type(item) is not tuple
                or len(item) != 2
                or not isinstance(item[0], NodeId)
                or type(item[1]) is not frozenset
                or not item[1]
                or any(type(port) is not str or not port for port in item[1])
            ):
                reject(EffectCode.INVALID_TYPE)
        if len({node for node, _ in self.publication_sources}) != len(self.publication_sources):
            reject(EffectCode.DUPLICATE)
        if not hasattr(self, "_accepted_publication_sources"):
            object.__setattr__(self, "_accepted_publication_sources", self.publication_sources)
        elif self.publication_sources != self._accepted_publication_sources:
            reject(EffectCode.FOREIGN_OWNER)
        if self.readiness is not None:
            require_instance(self.readiness, DelegatedReadiness)
            self.readiness.__post_init__()
        require_literal(self.mode, frozenset({"qualified", "opaque"}))
        require_literal(self.usage_requirement, frozenset({"exact", "unknown_allowed"}))
        require_instance(self.child_slots, tuple)
        require_instance(getattr(self, "optional_tail", None), tuple)
        require_instance(getattr(self, "generation_groups", None), tuple)
        if not hasattr(self, "selection"):
            reject(EffectCode.INVALID_TYPE)
        if self.selection is not None:
            require_instance(self.selection, DelegatedGenerationSelection)
            self.selection.__post_init__()
            if self.optional_tail or any((slot,) not in self.generation_groups for slot in self.selection.slots):
                reject(EffectCode.UNSUPPORTED)
        for group in self.generation_groups:
            require_instance(group, tuple)
            if not group:
                reject(EffectCode.MISSING)
            for slot in group:
                require_count(slot)
        if self.generation_groups and (
            tuple(slot for group in self.generation_groups for slot in group) != self.child_slots or self.optional_tail
        ):
            reject(EffectCode.UNSUPPORTED)
        if len(self.optional_tail) > 2:
            reject(EffectCode.LIMIT_EXCEEDED)
        for slot in self.optional_tail:
            require_count(slot)
        if len(set(self.optional_tail)) != len(self.optional_tail):
            reject(EffectCode.DUPLICATE)
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
        observed = self.observed_request
        readiness = self.readiness
        declaration = (
            None
            if observed is None
            else (
                observed.visibility,
                observed.pre_dispatch_control,
                observed.retry_owner,
                observed.replay,
                observed.max_attempts,
                self.child_slots,
                self.generation_groups,
                tuple((route.name, route.parse_attempts) for route in self.routes),
                self.followups,
                None
                if readiness is None
                else (
                    readiness.slots,
                    readiness.route.name,
                    readiness.route.parse_attempts,
                    readiness.ready_outcome,
                    readiness.probe_outcomes,
                ),
            )
        )
        if not hasattr(self, "_accepted_observed_declaration"):
            object.__setattr__(self, "_accepted_observed_declaration", declaration)
        elif declaration != self._accepted_observed_declaration:
            reject(EffectCode.FOREIGN_OWNER)

    @property
    def generations(self) -> tuple[tuple[int, ...], ...]:
        return self.generation_groups or (self.child_slots,)

    def snapshot(self) -> DelegatedRequestContract:
        """Retain cleanup authority independently of rejected declaration mutations."""
        return replace(
            self,
            observed_request=None if self.observed_request is None else replace(self.observed_request),
            routes=tuple(replace(route) for route in self.routes),
            readiness=None if self.readiness is None else replace(self.readiness, route=replace(self.readiness.route)),
            selection=None if self.selection is None else replace(self.selection),
        )

    @property
    def owned_slots(self) -> tuple[int, ...]:
        return self.child_slots + (() if self.readiness is None else self.readiness.slots)


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
    submission_owner: NodeId | None = None

    def __post_init__(self) -> None:
        require_instance(self.node, NodeId)
        require_literal(self.kind, frozenset({"local", "external", "delegated"}))
        if self.delegated is not None:
            require_instance(self.delegated, DelegatedRequestContract)
            self.delegated.__post_init__()
        if not hasattr(self, "submission_owner"):
            reject(EffectCode.INVALID_TYPE)
        if self.submission_owner is not None:
            require_instance(self.submission_owner, NodeId)
            if (
                self.submission_owner == self.node
                or self.delegated is not None
                or self.kind not in {"local", "delegated"}
            ):
                reject(EffectCode.CONTRADICTORY)
        if (self.kind == "delegated") != (
            self.delegated is not None or self.submission_owner is not None and self.kind == "delegated"
        ):
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
    supported_findings: frozenset[AssessmentFinding]

    def __post_init__(self) -> None:
        require_instance(self.node, NodeId)
        require_text(self.outcome)
        require_text(self.promise)
        require_text(self.evidence_port)
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
    max_assessment_facts: int
    max_port_facts: int
    max_provenance_edges: int

    def __post_init__(self) -> None:
        for value in (
            self.max_productions,
            self.max_findings_per_production,
            self.max_finding_code_bytes,
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

    def __post_init__(self) -> None:
        require_instance(self.configuration, FrozenConfig)


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


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExecutionAssessmentFact(PrivateValue):
    _key: InitVar[object]
    activation: ActivationKey
    node: NodeId
    outcome: str
    promise: str
    evidence_artifact: ArtifactRef
    finding: AssessmentFinding
    environment: AssessmentEnvironment

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _FACT_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExecutionPortFact(PrivateValue):
    _key: InitVar[object]
    activation: ActivationKey
    node: NodeId
    target: DatumId
    port: str
    artifact: ArtifactRef
    artifact_type: ArtifactType
    role: ArtifactRole

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _FACT_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class FinalOutputFact(PrivateValue):
    _key: InitVar[object]
    target: DatumId
    outcome: str
    port: str
    candidate: CandidateRef
    producer: ProvenanceKey

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _FACT_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ArtifactProvenanceFact(PrivateValue):
    _key: InitVar[object]
    key: ProvenanceKey
    artifact: ArtifactRef
    parents: frozenset[ProvenanceKey]

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _FACT_KEY)


_PLAN_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AdmittedExecutionPlan(PrivateValue):
    _key: InitVar[object]
    target_workflow: Graph
    invocation_workflow: Graph | None
    context: AdmittedContextPlan
    capabilities: tuple[ImplementationCapability, ...]
    policies: frozenset[OperationExecutionPolicy]
    map_expansions: tuple[MapExpansionDecl, ...]
    assessment_productions: tuple[EvidenceProductionDecl, ...]
    assessment_limits: AssessmentLimits
    ingress_limits: IngressLimits
    _ingress_policy: tuple[int, int, int | None, int, int | None, int | None, int | None, int] = field(
        init=False, compare=False
    )
    nodes: Mapping[NodeId, Operation] = field(init=False, compare=False)
    input_sources: Mapping[NodeInputRef, WorkflowInputRef | ContextInputRef | NodeOutputRef] = field(
        init=False, compare=False
    )
    materialization_schemas: Mapping[ArtifactType, Literal["scalar", "collection"]] = field(init=False, compare=False)

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _PLAN_KEY)
        object.__setattr__(
            self, "_ingress_policy", tuple(getattr(self.ingress_limits, item.name) for item in fields(IngressLimits))
        )
        bodies = self.context.prepared.workflow.bodies()
        object.__setattr__(
            self, "nodes", MappingProxyType({node.id: node for body in bodies for node in body.operations})
        )
        object.__setattr__(
            self,
            "input_sources",
            MappingProxyType(
                {binding.destination: binding.source for body in bodies for binding in body.input_bindings}
            ),
        )
        object.__setattr__(self, "materialization_schemas", MappingProxyType(self._compile_materialization_schemas()))

    def _validate_ingress_policy(self) -> None:
        """Preserve the response envelope selected at admission, including absent byte limits."""
        require_instance(self.ingress_limits, IngressLimits)
        self.ingress_limits.__post_init__()
        current = tuple(getattr(self.ingress_limits, item.name) for item in fields(IngressLimits))
        if getattr(self, "_ingress_policy", None) != current:
            reject(EffectCode.CONTRADICTORY)

    def _response_limits(self) -> IngressLimits:
        """Give the executor a detached copy of the issued response policy."""
        return IngressLimits(
            max_associations=self._ingress_policy[0],
            max_outputs_per_association=self._ingress_policy[1],
            max_collection_items=self._ingress_policy[2],
            max_assessments=self._ingress_policy[3],
            max_metadata_items=self._ingress_policy[4],
            max_text_bytes=self._ingress_policy[5],
            max_metadata_bytes=self._ingress_policy[6],
            max_depth=self._ingress_policy[7],
        )

    def _service_implementations(self) -> tuple[tuple[OperationExecutionPolicy, ExecutionImplementation], ...]:
        """Declared handle domain, excluding semantic work borrowed by a delegate."""
        borrowed = {
            self.context.prepared.reservation_recipe[policy.delegated.readiness.owner_slot].template
            for policy in self.policies
            if policy.delegated is not None and policy.delegated.readiness is not None
        }
        return tuple(
            (policy, item) for policy in self.policies if policy.node not in borrowed for item in policy.implementations
        )

    def _compile_materialization_schemas(self) -> dict[ArtifactType, Literal["scalar", "collection"]]:
        schemas: dict[ArtifactType, Literal["scalar", "collection"]] = {}
        context = self.context.bound_context
        if context is not None:
            for fact in context.receipt.sources:
                schemas[fact.declaration.materialization.item_type] = "scalar"
                schemas[fact.declaration.artifact_type] = (
                    "scalar" if fact.declaration.materialization.kind == "single" else "collection"
                )
        for expansion in self.map_expansions:
            operation = self.nodes[expansion.expander].operation
            artifact_type = next(
                item.artifact_type for item in operation.outputs if item.name == expansion.membership_port
            )
            schemas[expansion.item_type] = "scalar"
            schemas[artifact_type] = "collection"
        for policy in self.policies:
            if policy.delegated is None or policy.delegated.selection is None:
                continue
            selection = policy.delegated.selection
            artifact_type = next(
                port.artifact_type for port in self.nodes[policy.node].operation.outputs if port.name == selection.port
            )
            if schemas.get(artifact_type, "collection") != "collection":
                reject(EffectCode.CONTRADICTORY)
            schemas[artifact_type] = "collection"
        return schemas

    def _output_role(self, node: NodeId, outcome: OutcomeSpec, port: str) -> ArtifactRole:
        """Apply admitted occurrence-role precedence to an operation output."""
        productions = [
            item for item in self.assessment_productions if item.node == node and item.outcome == outcome.name
        ]
        promises = {item.name: item for item in outcome.evidence}
        if any(promises[item.promise].subject_port == port for item in productions) or any(
            isinstance(item.source, NodeOutputRef) and item.source.node == node and item.source.port == port
            for item in self.context.prepared.workflow.output_bindings
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
    frame: FramePublication | InvocationFrame | None = None
    failed_sources: frozenset[SemanticAssociation] = frozenset()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LocalFailure(PrivateValue):
    failure: FailureClass


LocalResult: TypeAlias = LocalCompleted | LocalFailure


@runtime_checkable
class LocalCallable(Protocol):
    async def run(self, request: tuple[AssociationInput, ...]) -> LocalResult: ...


class DelegatedRequests(Protocol):
    """Invocation-owned access to declared physical sibling slots."""

    @property
    def cancelled(self) -> bool: ...

    def require_transport(self, transport: object) -> None: ...

    async def request(
        self,
        slot: int,
        *,
        source_kind: ChildSourceKind = "initial",
        predecessor: PhysicalRequestId | None = None,
        evidence: ChildEvidence | None = None,
        association: SemanticAssociation | None = None,
    ) -> PhysicalRequestId: ...

    def bind_frame(
        self, dataframe: pd.DataFrame, *, targets: tuple[DatumId, ...] | None = None, source_limit: int | None = None
    ) -> WorkflowSourceFrame: ...

    async def enter_operation(self, slot: int, association: SemanticAssociation | None) -> AssociationInput | None: ...

    async def operation_result(self, slot: int, association: SemanticAssociation) -> AssociationResult | None: ...

    async def ready_operations(self, association: SemanticAssociation | None) -> tuple[int, ...]: ...

    async def complete_operation(self, association: SemanticAssociation, result: LocalResult) -> RuntimeOutcome: ...

    async def ready_operations_many(
        self, associations: tuple[SemanticAssociation, ...]
    ) -> tuple[tuple[int, ...], ...]: ...

    async def enter_operations(
        self, slot: int, associations: tuple[SemanticAssociation, ...]
    ) -> tuple[AssociationInput | None, ...]: ...

    async def operation_results(
        self, slot: int, associations: tuple[SemanticAssociation, ...]
    ) -> tuple[AssociationResult | None, ...]: ...

    async def complete_operations(
        self, results: tuple[tuple[SemanticAssociation, LocalResult], ...]
    ) -> tuple[RuntimeOutcome, ...]: ...

    async def close_observed_requests(
        self, associations: tuple[SemanticAssociation, ...], generated: frozenset[SemanticAssociation]
    ) -> None: ...

    async def readiness(self) -> PhysicalRequestId: ...

    async def probe(self) -> None: ...

    def parser_rejected(self, request: PhysicalRequestId) -> ChildEvidence: ...

    async def aparser_rejected(self, request: PhysicalRequestId) -> ChildEvidence: ...


@runtime_checkable
class DelegatedCallable(Protocol):
    @property
    def execution(self) -> Literal["same_loop", "worker"]: ...

    async def submit(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalResult: ...


class MonotonicClock(Protocol):
    def now_ns(self) -> int: ...


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExecutionLimits(PrivateValue):
    """Finite local work/artifacts; optional nested-item, remote and byte ceilings."""

    max_local_in_flight: int
    max_remote_outstanding: int | None
    max_runtime_artifacts: int
    max_runtime_artifact_bytes: int | None
    max_collection_items: int | None

    def __post_init__(self) -> None:
        require_count(self.max_local_in_flight)
        if self.max_remote_outstanding is not None:
            require_count(self.max_remote_outstanding)
        require_count(self.max_runtime_artifacts)
        if self.max_runtime_artifact_bytes is not None:
            require_count(self.max_runtime_artifact_bytes)
        if self.max_collection_items is not None:
            require_count(self.max_collection_items)


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
class ImplementationResourceRegistration(PrivateValue):
    """Invocation-local handle identity and its pre-dispatch resource declaration."""

    invocation: InvocationId
    implementation: ImplementationRef
    operation: OperationSpec
    configuration: FrozenConfig
    resource: ResourceId
    owner: ResourceOwner
    safe_detachment: SafeDetachment
    _lease: ResourceLease | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        require_instance(self.invocation, InvocationId)
        require_instance(self.implementation, ImplementationRef)
        require_instance(self.operation, OperationSpec)
        require_instance(self.configuration, FrozenConfig)
        require_instance(self.resource, ResourceId)
        require_literal(self.owner, frozenset({"caller", "sdk"}))
        require_literal(self.safe_detachment, frozenset({"forbidden", "independent_after_dispatch"}))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExecutionServices(PrivateValue):
    handles: tuple[ImplementationHandle, ...]
    limits: ExecutionLimits
    clock: MonotonicClock


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SubmissionFailure(PrivateValue):
    """Original callback error retained for domain projection, never qualification."""

    associations: frozenset[SemanticAssociation]
    cause: Exception


_RESULT_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ExecutionResult(PrivateValue):
    _key: InitVar[object]
    _execution: AdmittedExecutionPlan
    implementation_resources: tuple[ImplementationResourceRegistration, ...]
    _retained_resources: tuple[ResourceLease, ...] = ()
    _input_parents: tuple[tuple[DatumId, ActivationKey, str, ProvenanceKey], ...]
    _passthrough_parents: tuple[tuple[DatumId, ActivationKey, str, ProvenanceKey], ...]
    record: CanonicalRecord
    states: tuple[ActivationState, ...]
    invocation_state: ActivationState | None = None
    requests: RequestReceipt
    cleanup: tuple[CleanupFact, ...]
    artifacts: tuple[tuple[ArtifactRef, ArtifactValue], ...]
    assessments: tuple[ExecutionAssessmentFact, ...]
    ports: tuple[ExecutionPortFact, ...]
    final_outputs: tuple[FinalOutputFact, ...]
    provenance: tuple[ArtifactProvenanceFact, ...]
    cleanup_associations: tuple[CleanupAssociation, ...]
    publication_failures: tuple[PublicationFailureFact, ...]
    frames: tuple[PublishedFrame | InvocationFrame, ...] = ()
    submission_failures: tuple[SubmissionFailure, ...] = ()

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _RESULT_KEY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PublicationFailureFact(PrivateValue):
    """Occurrence-local publication rejection, independent of physical acceptance."""

    association: SemanticAssociation
    target: DatumId
    condition: RuntimeCondition
    failure: FailureClass | None
