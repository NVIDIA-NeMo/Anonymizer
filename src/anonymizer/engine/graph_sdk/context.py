# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed context declarations, binding facts, and pure context admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    PrivateValue,
    reject,
    require_count,
    require_instance,
    require_literal,
    require_text,
)
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.requests import (
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    ExternalSettlement,
    FailureClass,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    RequestAssociation,
    RequestReceipt,
    SemanticAssociation,
    StopResult,
)
from anonymizer.engine.graph_sdk.resources import CleanupFact, ResourceLease
from anonymizer.graph._values import DatumId
from anonymizer.graph.workflow import AdmittedActivationWorkflow, ArtifactType, NodeId, OperationNode, SubgraphNode

ContextRequirement: TypeAlias = Literal["required", "optional"]
ProviderExecution: TypeAlias = Literal["async", "blocking"]
SourceTerminal: TypeAlias = Literal["bound", "failed", "cancelled", "lost", "oversize", "omitted_optional"]
BindingTerminal: TypeAlias = Literal["success", "partial", "failed", "cancelled", "lost", "inconsistent"]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BindingArtifactRef(PrivateValue):
    declaration: BindingDeclarationId
    key: int
    version: int

    def __post_init__(self) -> None:
        require_instance(self.declaration, BindingDeclarationId)
        require_count(self.key)
        require_count(self.version, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContextSourceRef(PrivateValue):
    name: str
    revision: int

    def __post_init__(self) -> None:
        require_text(self.name)
        require_count(self.revision, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SelectorField(PrivateValue):
    name: str
    value: str

    def __post_init__(self) -> None:
        require_text(self.name)
        require_text(self.value, empty=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContextSelector(PrivateValue):
    fields: tuple[SelectorField, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple) or any(not isinstance(item, SelectorField) for item in self.fields):
            reject(EffectCode.INVALID_TYPE)
        names = [item.name for item in self.fields]
        if names != sorted(names):
            reject(EffectCode.INVALID_VALUE)
        if len(names) != len(set(names)):
            reject(EffectCode.DUPLICATE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RetrievalBounds(PrivateValue):
    max_items: int
    max_bytes: int
    max_requests: int

    def __post_init__(self) -> None:
        require_count(self.max_items)
        require_count(self.max_bytes)
        require_count(self.max_requests)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContextMaterialization(PrivateValue):
    kind: Literal["single", "collection"]
    item_type: ArtifactType

    def __post_init__(self) -> None:
        require_literal(self.kind, frozenset({"single", "collection"}))
        require_instance(self.item_type, ArtifactType)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class InitialContextDecl(PrivateValue):
    target: DatumId
    node: NodeId
    port: str
    artifact_type: ArtifactType
    source: ContextSourceRef
    selector: ContextSelector
    requirement: ContextRequirement
    bounds: RetrievalBounds
    materialization: ContextMaterialization

    def __post_init__(self) -> None:
        require_instance(self.target, DatumId)
        require_instance(self.node, NodeId)
        require_text(self.port)
        require_instance(self.artifact_type, ArtifactType)
        require_instance(self.source, ContextSourceRef)
        require_instance(self.selector, ContextSelector)
        require_literal(self.requirement, frozenset({"required", "optional"}))
        require_instance(self.bounds, RetrievalBounds)
        require_instance(self.materialization, ContextMaterialization)
        if self.materialization.kind == "single":
            if self.artifact_type != self.materialization.item_type or self.bounds.max_items != 1:
                reject(EffectCode.CONTRADICTORY)
        elif self.artifact_type == self.materialization.item_type or self.bounds.max_items == 0:
            reject(EffectCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContextSourceCapability(PrivateValue):
    source: ContextSourceRef
    artifact_type: ArtifactType
    uses: frozenset[Literal["initial_binding", "adaptive_retrieval"]]
    execution: ProviderExecution
    resource_owner: Literal["caller", "sdk"]
    cancellation: str
    settlement: str
    usage: str
    request: PhysicalRequestPolicy
    safe_detachment: Literal["forbidden", "independent_after_dispatch"]

    def __post_init__(self) -> None:
        require_instance(self.source, ContextSourceRef)
        require_instance(self.artifact_type, ArtifactType)
        if not isinstance(self.uses, frozenset) or any(not isinstance(item, str) for item in self.uses):
            reject(EffectCode.INVALID_TYPE)
        if not self.uses or not self.uses <= {"initial_binding", "adaptive_retrieval"}:
            reject(EffectCode.INVALID_VALUE)
        require_literal(self.execution, frozenset({"async", "blocking"}))
        require_literal(self.resource_owner, frozenset({"caller", "sdk"}))
        require_literal(self.cancellation, frozenset({"before_dispatch_only", "cooperative_ack", "unobservable"}))
        require_literal(self.settlement, frozenset({"synchronous", "explicit_ack", "uncertain_possible"}))
        require_literal(self.usage, frozenset({"exact", "upper_bound", "unknown"}))
        require_instance(self.request, PhysicalRequestPolicy)
        require_literal(self.safe_detachment, frozenset({"forbidden", "independent_after_dispatch"}))
        if self.request.retry_owner == "implementation":
            reject(EffectCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BindingLimits(PrivateValue):
    max_declarations: int
    max_sources: int
    max_capabilities: int
    max_selector_fields: int
    max_selector_bytes: int
    max_items: int
    max_bytes: int
    max_requests: int
    max_resources: int

    def __post_init__(self) -> None:
        for value in (
            self.max_declarations,
            self.max_sources,
            self.max_capabilities,
            self.max_selector_fields,
            self.max_selector_bytes,
            self.max_items,
            self.max_bytes,
            self.max_requests,
            self.max_resources,
        ):
            require_count(value)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SourceItem(PrivateValue):
    association: RequestAssociation
    key: int
    version: int
    text: str

    def __post_init__(self) -> None:
        require_instance(self.association, (BindingAssociation, SemanticAssociation))
        require_count(self.key)
        require_count(self.version, positive=True)
        require_text(self.text, empty=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SourceResponse(PrivateValue):
    source: ContextSourceRef
    items: tuple[SourceItem, ...]
    settlement: ExternalSettlement

    def __post_init__(self) -> None:
        require_instance(self.source, ContextSourceRef)
        if not isinstance(self.items, tuple) or any(not isinstance(item, SourceItem) for item in self.items):
            reject(EffectCode.INVALID_TYPE)
        require_instance(self.settlement, ExternalSettlement)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SourceFailure(PrivateValue):
    source: ContextSourceRef
    failure: FailureClass
    settlement: ExternalSettlement | None

    def __post_init__(self) -> None:
        require_instance(self.source, ContextSourceRef)
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
        if self.settlement is not None:
            require_instance(self.settlement, ExternalSettlement)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SourceLost(PrivateValue):
    source: ContextSourceRef
    settlement: ExternalSettlement | None

    def __post_init__(self) -> None:
        require_instance(self.source, ContextSourceRef)
        if self.settlement is not None:
            require_instance(self.settlement, ExternalSettlement)


SourceResult: TypeAlias = SourceResponse | SourceFailure | SourceLost


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SourceBindingFact(PrivateValue):
    identity: BindingDeclarationId
    declaration: InitialContextDecl
    terminal: SourceTerminal

    def __post_init__(self) -> None:
        require_instance(self.identity, BindingDeclarationId)
        require_instance(self.declaration, InitialContextDecl)
        require_literal(
            self.terminal,
            frozenset({"bound", "failed", "cancelled", "lost", "oversize", "omitted_optional"}),
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BoundTextArtifact(PrivateValue):
    reference: BindingArtifactRef
    target: DatumId
    node: NodeId
    port: str
    source: ContextSourceRef
    artifact_type: ArtifactType
    text: str

    def __post_init__(self) -> None:
        require_instance(self.reference, BindingArtifactRef)
        require_instance(self.target, DatumId)
        require_instance(self.node, NodeId)
        require_text(self.port)
        require_instance(self.source, ContextSourceRef)
        require_instance(self.artifact_type, ArtifactType)
        require_text(self.text, empty=True)


_RESULT_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class BindingReceipt(PrivateValue):
    binding: BindingId
    data: ValidatedDataGraph
    workflow: AdmittedActivationWorkflow
    terminal: BindingTerminal
    sources: tuple[SourceBindingFact, ...]
    artifacts: tuple[BoundTextArtifact, ...]
    requests: RequestReceipt
    cleanup: tuple[CleanupFact, ...]

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _RESULT_KEY:
            raise TypeError("binding receipts are created by the binding executor")
        for name, value in values.items():
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class BoundContext(PrivateValue):
    receipt: BindingReceipt
    artifacts: tuple[BoundTextArtifact, ...]

    def __init__(self, *, _key: object, receipt: BindingReceipt, artifacts: tuple[BoundTextArtifact, ...]) -> None:
        if _key is not _RESULT_KEY:
            raise TypeError("bound contexts are created by the binding executor")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class BindingResult(PrivateValue):
    receipt: BindingReceipt
    context: BoundContext | None

    def __init__(self, *, _key: object, receipt: BindingReceipt, context: BoundContext | None) -> None:
        if _key is not _RESULT_KEY:
            raise TypeError("binding results are created by the binding executor")
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "context", context)


@runtime_checkable
class ContextProvider(Protocol):
    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResult: ...

    async def cancel(self, request: PhysicalRequestId) -> StopResult: ...


class SdkContextFactory(Protocol):
    def __call__(self) -> ContextProvider: ...


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContextResource(PrivateValue):
    source: ContextSourceRef
    capability: ContextSourceCapability
    lease: ResourceLease | None
    factory: SdkContextFactory | None

    def __post_init__(self) -> None:
        require_instance(self.source, ContextSourceRef)
        require_instance(self.capability, ContextSourceCapability)
        if self.source != self.capability.source:
            reject(EffectCode.CONTRADICTORY)
        if (self.lease is None) == (self.factory is None):
            reject(EffectCode.CONTRADICTORY)
        if self.capability.resource_owner == "caller":
            if self.lease is None or self.lease.owner != "caller":
                reject(EffectCode.CONTRADICTORY)
        elif self.factory is None:
            reject(EffectCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class AdaptiveRetrievalDecl(PrivateValue):
    node: NodeId
    source: ContextSourceRef
    selector_ports: tuple[str, ...]
    output_port: str
    bounds: RetrievalBounds
    materialization: ContextMaterialization

    def __post_init__(self) -> None:
        require_instance(self.node, NodeId)
        require_instance(self.source, ContextSourceRef)
        if not isinstance(self.selector_ports, tuple) or any(not isinstance(item, str) for item in self.selector_ports):
            reject(EffectCode.INVALID_TYPE)
        if len(self.selector_ports) != len(set(self.selector_ports)):
            reject(EffectCode.DUPLICATE)
        if any(not item for item in self.selector_ports):
            reject(EffectCode.INVALID_VALUE)
        require_text(self.output_port)
        require_instance(self.bounds, RetrievalBounds)
        require_instance(self.materialization, ContextMaterialization)


_CONTEXT_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class AdmittedContextPlan(PrivateValue):
    prepared: PreparedPlan
    bound_context: BoundContext | None
    adaptive_retrievals: tuple[AdaptiveRetrievalDecl, ...]
    context_capabilities: tuple[ContextSourceCapability, ...]

    def __init__(self, *, _key: object, **values: object) -> None:
        if _key is not _CONTEXT_KEY:
            raise TypeError("context plans are created by admission")
        for name, value in values.items():
            object.__setattr__(self, name, value)


def admit_context_plan(
    *,
    prepared: PreparedPlan,
    bound_context: BoundContext | None,
    adaptive_retrievals: tuple[AdaptiveRetrievalDecl, ...],
    context_capabilities: tuple[ContextSourceCapability, ...],
) -> AdmittedContextPlan:
    """Admit immutable initial and adaptive context declarations without effects."""
    require_instance(prepared, PreparedPlan)
    if bound_context is not None:
        require_instance(bound_context, BoundContext)
        if bound_context.receipt.data is not prepared.data or bound_context.receipt.workflow is not prepared.workflow:
            reject(EffectCode.FOREIGN_OWNER)
    if not isinstance(adaptive_retrievals, tuple) or any(
        not isinstance(item, AdaptiveRetrievalDecl) for item in adaptive_retrievals
    ):
        reject(EffectCode.INVALID_TYPE)
    if not isinstance(context_capabilities, tuple):
        reject(EffectCode.INVALID_TYPE)
    if len(context_capabilities) > prepared.limits.max_capabilities:
        reject(EffectCode.LIMIT_EXCEEDED)
    if any(not isinstance(item, ContextSourceCapability) for item in context_capabilities):
        reject(EffectCode.INVALID_TYPE)
    if len(context_capabilities) != len(set(context_capabilities)):
        reject(EffectCode.DUPLICATE)
    nodes = _reachable_operations(prepared.workflow)
    if bound_context is not None:
        _validate_bound_context(prepared, bound_context, nodes)
    selected = {item.node: item.capability for item in prepared.implementations}
    declared_nodes = [item.node for item in adaptive_retrievals]
    if len(declared_nodes) != len(set(declared_nodes)):
        reject(EffectCode.DUPLICATE)
    for declaration in adaptive_retrievals:
        node = nodes.get(declaration.node)
        if node is None:
            reject(EffectCode.FOREIGN_OWNER)
        inputs = {item.name for item in node.operation.inputs}
        outputs = {item.name: item.artifact_type for item in node.operation.outputs}
        if not set(declaration.selector_ports) <= inputs or declaration.output_port not in outputs:
            reject(EffectCode.MISSING)
        matches = [
            item
            for item in context_capabilities
            if item.source == declaration.source
            and item.artifact_type == declaration.materialization.item_type
            and "adaptive_retrieval" in item.uses
        ]
        output_type = outputs[declaration.output_port]
        materialization_valid = (
            declaration.materialization.kind == "single"
            and output_type == declaration.materialization.item_type
            and declaration.bounds.max_items == 1
        ) or (
            declaration.materialization.kind == "collection"
            and output_type != declaration.materialization.item_type
            and declaration.bounds.max_items > 0
        )
        if (
            len(matches) != 1
            or not materialization_valid
            or declaration.bounds.max_requests > matches[0].request.max_attempts
        ):
            reject(EffectCode.UNSUPPORTED)
        capability = selected.get(declaration.node)
        if capability is None or capability.attribution != "per_task":
            reject(EffectCode.UNSUPPORTED)
        producing = [
            outcome for outcome in node.operation.outcomes if declaration.output_port in outcome.produced_ports
        ]
        if not producing or any(
            declaration.bounds.max_requests > outcome.ceiling.max_model_requests
            or declaration.bounds.max_bytes > outcome.ceiling.max_output_bytes
            for outcome in producing
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
    _validate_materialization_schema(prepared, bound_context, adaptive_retrievals, nodes)
    return AdmittedContextPlan(
        _key=_CONTEXT_KEY,
        prepared=prepared,
        bound_context=bound_context,
        adaptive_retrievals=adaptive_retrievals,
        context_capabilities=context_capabilities,
    )


def _validate_bound_context(
    prepared: PreparedPlan,
    context: BoundContext,
    nodes: dict[NodeId, OperationNode],
) -> None:
    receipt = context.receipt
    if receipt.terminal not in {"success", "partial"} or context.artifacts != receipt.artifacts:
        reject(EffectCode.CONTRADICTORY)
    identities = [item.identity for item in receipt.sources]
    if len(identities) != len(set(identities)):
        reject(EffectCode.DUPLICATE)
    targets = {item.target for item in prepared.target_occurrences}
    by_identity = {item.identity: item for item in receipt.sources}
    artifact_refs = [item.reference for item in context.artifacts]
    if len(artifact_refs) != len(set(artifact_refs)):
        reject(EffectCode.DUPLICATE)
    for fact in receipt.sources:
        declaration = fact.declaration
        operation_node = nodes.get(declaration.node)
        if declaration.target not in targets or operation_node is None:
            reject(EffectCode.FOREIGN_OWNER)
        port = next((item for item in operation_node.operation.inputs if item.name == declaration.port), None)
        context_ports = {use.port for outcome in operation_node.operation.outcomes for use in outcome.context}
        if port is None or port.artifact_type != declaration.artifact_type or declaration.port not in context_ports:
            reject(EffectCode.CONTRADICTORY)
        retained = [item for item in context.artifacts if item.reference.declaration == fact.identity]
        if fact.terminal == "bound":
            if not retained or (declaration.materialization.kind == "single" and len(retained) != 1):
                reject(EffectCode.MISSING)
        elif retained:
            reject(EffectCode.EXTRA)
        if declaration.requirement == "required" and fact.terminal != "bound":
            reject(EffectCode.CONTRADICTORY)
    for artifact in context.artifacts:
        fact = by_identity.get(artifact.reference.declaration)
        if fact is None:
            reject(EffectCode.EXTRA)
        declaration = fact.declaration
        if (
            artifact.target != declaration.target
            or artifact.node != declaration.node
            or artifact.port != declaration.port
            or artifact.source != declaration.source
            or artifact.artifact_type != declaration.materialization.item_type
        ):
            reject(EffectCode.CONTRADICTORY)


def _validate_materialization_schema(
    prepared: PreparedPlan,
    bound_context: BoundContext | None,
    adaptive: tuple[AdaptiveRetrievalDecl, ...],
    nodes: dict[NodeId, OperationNode],
) -> None:
    schemas: dict[ArtifactType, tuple[str, ArtifactType]] = {}
    initial = () if bound_context is None else tuple(item.declaration for item in bound_context.receipt.sources)
    declarations: list[tuple[ArtifactType, ContextMaterialization]] = [
        (item.artifact_type, item.materialization) for item in initial
    ]
    declarations.extend(
        (
            next(
                output.artifact_type for output in nodes[item.node].operation.outputs if output.name == item.output_port
            ),
            item.materialization,
        )
        for item in adaptive
    )
    for output_type, materialization in declarations:
        item_type = materialization.item_type
        item_schema = schemas.get(item_type)
        if item_schema is not None and item_schema[0] == "collection":
            reject(EffectCode.CONTRADICTORY)
        schemas.setdefault(item_type, ("scalar", item_type))
        candidate = ("scalar" if materialization.kind == "single" else "collection", item_type)
        existing = schemas.get(output_type)
        if existing is not None and existing != candidate:
            reject(EffectCode.CONTRADICTORY)
        schemas[output_type] = candidate
    collection_types = {item for item, schema in schemas.items() if schema[0] == "collection"}
    if any(item.artifact_type in collection_types for item in prepared.bound_inputs):
        reject(EffectCode.CONTRADICTORY)


def _create_binding_result(
    *,
    binding: BindingId,
    data: ValidatedDataGraph,
    workflow: AdmittedActivationWorkflow,
    terminal: BindingTerminal,
    sources: tuple[SourceBindingFact, ...],
    artifacts: tuple[BoundTextArtifact, ...],
    requests: RequestReceipt,
    cleanup: tuple[CleanupFact, ...],
) -> BindingResult:
    """Construct the authenticated result of an initial-binding operation."""
    receipt = BindingReceipt(
        _key=_RESULT_KEY,
        binding=binding,
        data=data,
        workflow=workflow,
        terminal=terminal,
        sources=sources,
        artifacts=artifacts,
        requests=requests,
        cleanup=cleanup,
    )
    context = (
        BoundContext(_key=_RESULT_KEY, receipt=receipt, artifacts=artifacts)
        if terminal in {"success", "partial"}
        else None
    )
    return BindingResult(_key=_RESULT_KEY, receipt=receipt, context=context)


def _reachable_operations(workflow: AdmittedActivationWorkflow) -> dict[NodeId, OperationNode]:
    operations: dict[NodeId, OperationNode] = {}
    pending = [workflow.workflow]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for node in current.nodes:
            if isinstance(node, OperationNode):
                operations[node.id] = node
            elif isinstance(node, SubgraphNode):
                pending.append(node.body)
    return operations
