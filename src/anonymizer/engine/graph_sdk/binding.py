# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Asynchronous initial context binding over the shared request reducer."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
from typing import Literal

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected, reject, require_instance
from anonymizer.engine.graph_sdk.context import (
    BindingArtifactRef,
    BindingLimits,
    BindingResult,
    BindingTerminal,
    BoundTextArtifact,
    ContextProvider,
    ContextResource,
    ContextSourceCapability,
    InitialContextDecl,
    SourceBindingFact,
    SourceFailure,
    SourceLost,
    SourceResponse,
    SourceTerminal,
    _create_binding_result,
)
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.engine.graph_sdk.requests import (
    AcceptFailure,
    AcceptResult,
    AssociationResult,
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    BindingRequestScope,
    Dispatch,
    FailureClass,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    RequestCancel,
    RequestPolicyBinding,
    RequestPurpose,
    RequestState,
    Reserve,
    ScopeCancel,
    StopAcknowledged,
    StopConfirmed,
    advance_requests,
    bind_request_policies,
    can_reserve_followup,
    initialize_requests,
    request_receipt,
)
from anonymizer.engine.graph_sdk.resources import CleanupFact, ResourceLease, close_resource
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    ContextInputRef,
    Node,
    NodeId,
    NodeInputRef,
    OperationNode,
    SubgraphNode,
    WorkflowInputRef,
)


class BindingRejected(EffectRejected):
    """A local initial-binding admission rejection."""


@dataclass(slots=True)
class _BindingWork:
    binding: BindingId
    data: ValidatedDataGraph
    workflow: AdmittedActivationWorkflow
    declarations: tuple[InitialContextDecl, ...]
    identities: tuple[BindingDeclarationId, ...]
    capabilities: tuple[ContextSourceCapability, ...]
    resources: tuple[ContextResource, ...]
    limits: BindingLimits
    cancelled: bool = False


class RunningBinding:
    """Live owner of one initial-binding operation."""

    __slots__ = ("_task", "_work", "binding")

    def __init__(self, work: _BindingWork) -> None:
        self.binding = work.binding
        self._work = work
        self._task = asyncio.create_task(_run_binding(work))

    def request_cancel(self) -> None:
        self._work.cancelled = True

    async def wait(self) -> BindingResult:
        return await asyncio.shield(self._task)


async def start_initial_binding(
    *,
    data: ValidatedDataGraph,
    workflow: AdmittedActivationWorkflow,
    declarations: tuple[InitialContextDecl, ...],
    capabilities: tuple[ContextSourceCapability, ...],
    resources: tuple[ContextResource, ...],
    limits: BindingLimits,
) -> RunningBinding:
    """Validate initial context and return its live asynchronous controller."""
    _validate_binding(data, workflow, declarations, capabilities, resources, limits)
    binding = BindingId.new()
    identities = tuple(BindingDeclarationId.new(binding=binding, ordinal=index) for index in range(len(declarations)))
    return RunningBinding(
        _BindingWork(
            binding=binding,
            data=data,
            workflow=workflow,
            declarations=declarations,
            identities=identities,
            capabilities=capabilities,
            resources=resources,
            limits=limits,
        )
    )


async def _run_binding(work: _BindingWork) -> BindingResult:
    scope = BindingRequestScope(binding=work.binding)
    state = initialize_requests(
        scope=scope,
        hard_limit=work.limits.max_requests,
        policies=frozenset(item.request for item in work.capabilities),
    )
    try:
        acquired, cleanup = _acquire_resources(work)
    except Exception:
        facts = tuple(
            SourceBindingFact(identity=identity, declaration=declaration, terminal="failed")
            for identity, declaration in zip(work.identities, work.declarations, strict=True)
        )
        return _create_binding_result(
            binding=work.binding,
            data=work.data,
            workflow=work.workflow,
            terminal="failed",
            sources=facts,
            artifacts=(),
            requests=request_receipt(state),
            cleanup=(),
        )
    facts: list[SourceBindingFact] = []
    artifacts: list[BoundTextArtifact] = []
    if work.cancelled:
        state = advance_requests(state=state, event=ScopeCancel())
        facts.extend(
            SourceBindingFact(identity=identity, declaration=declaration, terminal="cancelled")
            for identity, declaration in zip(work.identities, work.declarations, strict=True)
        )
        cleanup.extend(await _cleanup(acquired))
        return _create_binding_result(
            binding=work.binding,
            data=work.data,
            workflow=work.workflow,
            terminal="cancelled",
            sources=tuple(facts),
            artifacts=(),
            requests=request_receipt(state),
            cleanup=tuple(cleanup),
        )
    total_items = 0
    total_bytes = 0
    for identity, declaration in zip(work.identities, work.declarations, strict=True):
        state, fact, retained, item_count, byte_count = await _bind_declaration(
            work,
            scope,
            state,
            acquired,
            identity,
            declaration,
            total_items,
            total_bytes,
        )
        facts.append(fact)
        artifacts.extend(retained)
        total_items += item_count
        total_bytes += byte_count
    cleanup.extend(await _cleanup(acquired))
    terminal = _binding_terminal(facts)
    return _create_binding_result(
        binding=work.binding,
        data=work.data,
        workflow=work.workflow,
        terminal=terminal,
        sources=tuple(facts),
        artifacts=tuple(artifacts),
        requests=request_receipt(state),
        cleanup=tuple(cleanup),
    )


async def _bind_declaration(
    work: _BindingWork,
    scope: BindingRequestScope,
    state: RequestState,
    acquired: dict[object, ResourceLease],
    identity: BindingDeclarationId,
    declaration: InitialContextDecl,
    total_items: int,
    total_bytes: int,
) -> tuple[RequestState, SourceBindingFact, tuple[BoundTextArtifact, ...], int, int]:
    association = BindingAssociation(declaration=identity)
    capability = _capability(work.capabilities, declaration)
    state = bind_request_policies(
        state=state,
        binding=RequestPolicyBinding.create(association=association, policies=frozenset({capability.request})),
    )
    provider = acquired[declaration.source].handle
    if not isinstance(provider, ContextProvider):
        reject(EffectCode.INVALID_TYPE)
    purpose: RequestPurpose = "initial_binding"
    while True:
        request = PhysicalRequestId.new(scope=scope)
        state = advance_requests(
            state=state,
            event=Reserve(
                request=request,
                purpose=purpose,
                associations=frozenset({association}),
                policy=capability.request,
            ),
        )
        if not any(item.request == request for item in state.reserved):
            return state, _source_fact(identity, declaration, "failed"), (), 0, 0
        if work.cancelled:
            state = advance_requests(state=state, event=RequestCancel(request=request))
            return state, _source_fact(identity, declaration, "cancelled"), (), 0, 0
        state = advance_requests(state=state, event=Dispatch(request=request))
        retrieval = asyncio.create_task(
            provider.retrieve(
                request=request,
                association=association,
                selector=declaration.selector,
                bounds=declaration.bounds,
            )
        )
        while not retrieval.done() and not work.cancelled:
            await asyncio.sleep(0)
        if work.cancelled and not retrieval.done():
            try:
                stopped = await provider.cancel(request)
            except Exception:
                stopped = None
            retrieval.cancel()
            with suppress(asyncio.CancelledError):
                await retrieval
            if isinstance(stopped, StopConfirmed):
                state = advance_requests(state=state, event=StopAcknowledged(request=request, usage=stopped.usage))
                return state, _source_fact(identity, declaration, "cancelled"), (), 0, 0
            state = advance_requests(state=state, event=MarkLost(request=request))
            return state, _source_fact(identity, declaration, "lost"), (), 0, 0
        try:
            result = retrieval.result()
        except Exception:
            state = advance_requests(state=state, event=MarkLost(request=request))
            return state, _source_fact(identity, declaration, "lost"), (), 0, 0
        if isinstance(result, SourceResponse):
            valid_settlement = result.settlement.request == request
            valid_items = (
                result.source == declaration.source
                and all(item.association == association for item in result.items)
                and len({(item.association, item.key, item.version) for item in result.items}) == len(result.items)
                and bool(result.items)
            )
            if not valid_settlement or not valid_items:
                state = advance_requests(
                    state=state, event=AcceptFailure(request=request, failure="malformed_response")
                )
                if valid_settlement:
                    state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
                purpose = "correction"
                if _can_retry(capability, declaration, state, association, "malformed_response"):
                    continue
                return state, _source_fact(identity, declaration, _failed_terminal(declaration)), (), 0, 0
            item_bytes = sum(len(item.text.encode("utf-8")) for item in result.items)
            oversize = (
                len(result.items) > declaration.bounds.max_items
                or item_bytes > declaration.bounds.max_bytes
                or total_items + len(result.items) > work.limits.max_items
                or total_bytes + item_bytes > work.limits.max_bytes
            )
            accepted = advance_requests(
                state=state,
                event=AcceptResult(
                    request=request,
                    results=(
                        AssociationResult(
                            association=association,
                            outcome="retrieved",
                            outputs=(),
                            consumed_context_ports=frozenset(),
                        ),
                    ),
                ),
            )
            state = advance_requests(state=accepted, event=ObserveSettlement(settlement=result.settlement))
            if oversize:
                return state, _source_fact(identity, declaration, "oversize"), (), 0, 0
            retained = tuple(
                BoundTextArtifact(
                    reference=BindingArtifactRef(declaration=identity, key=item.key, version=item.version),
                    target=declaration.target,
                    node=declaration.node,
                    port=declaration.port,
                    source=declaration.source,
                    artifact_type=declaration.materialization.item_type,
                    text=item.text,
                )
                for item in result.items
            )
            return state, _source_fact(identity, declaration, "bound"), retained, len(result.items), item_bytes
        if isinstance(result, SourceFailure):
            valid_owner = result.source == declaration.source and (
                result.settlement is None or result.settlement.request == request
            )
            valid_omission = result.disposition == "failed" or (
                declaration.requirement == "optional" and result.failure == "permanent"
            )
            failure = result.failure if valid_owner and valid_omission else "malformed_response"
            state = advance_requests(state=state, event=AcceptFailure(request=request, failure=failure))
            if result.settlement is not None and result.settlement.request == request:
                state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
            if failure != "malformed_response" and result.disposition == "omitted_optional":
                return state, _source_fact(identity, declaration, "omitted_optional"), (), 0, 0
            purpose = "correction" if failure == "malformed_response" else "retry"
            if _can_retry(capability, declaration, state, association, failure):
                continue
            return state, _source_fact(identity, declaration, _failed_terminal(declaration)), (), 0, 0
        if isinstance(result, SourceLost) and result.source == declaration.source:
            state = advance_requests(state=state, event=MarkLost(request=request))
            if result.settlement is not None and result.settlement.request == request:
                state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
            return state, _source_fact(identity, declaration, "lost"), (), 0, 0
        state = advance_requests(state=state, event=AcceptFailure(request=request, failure="malformed_response"))
        purpose = "correction"
        if not _can_retry(capability, declaration, state, association, "malformed_response"):
            return state, _source_fact(identity, declaration, _failed_terminal(declaration)), (), 0, 0


def _source_fact(
    identity: BindingDeclarationId, declaration: InitialContextDecl, terminal: SourceTerminal
) -> SourceBindingFact:
    return SourceBindingFact(identity=identity, declaration=declaration, terminal=terminal)


def _failed_terminal(declaration: InitialContextDecl) -> SourceTerminal:
    del declaration
    return "failed"


def _can_retry(
    capability: ContextSourceCapability,
    declaration: InitialContextDecl,
    state: RequestState,
    association: BindingAssociation,
    failure: FailureClass,
) -> bool:
    attempts = sum(association in item.associations for item in state.dispatches)
    if attempts >= declaration.bounds.max_requests:
        return False
    purpose: Literal["retry", "correction"] = "correction" if failure == "malformed_response" else "retry"
    if purpose == "retry" and capability.request.retry_owner != "executor":
        return False
    return can_reserve_followup(
        state=state,
        purpose=purpose,
        associations=frozenset({association}),
        policy=capability.request,
    )


def _validate_binding(
    data: ValidatedDataGraph,
    workflow: AdmittedActivationWorkflow,
    declarations: tuple[InitialContextDecl, ...],
    capabilities: tuple[ContextSourceCapability, ...],
    resources: tuple[ContextResource, ...],
    limits: BindingLimits,
) -> None:
    require_instance(data, ValidatedDataGraph)
    require_instance(workflow, AdmittedActivationWorkflow)
    require_instance(limits, BindingLimits)
    for values, maximum, expected in (
        (declarations, limits.max_declarations, InitialContextDecl),
        (capabilities, limits.max_capabilities, ContextSourceCapability),
        (resources, limits.max_resources, ContextResource),
    ):
        if not isinstance(values, tuple):
            reject(EffectCode.INVALID_TYPE)
        if len(values) > maximum:
            reject(EffectCode.LIMIT_EXCEEDED)
        if any(not isinstance(item, expected) for item in values):
            reject(EffectCode.INVALID_TYPE)
    sources = {item.source for item in declarations}
    if len(sources) > limits.max_sources:
        reject(EffectCode.LIMIT_EXCEEDED)
    if sum(len(item.selector.fields) for item in declarations) > limits.max_selector_fields:
        reject(EffectCode.LIMIT_EXCEEDED)
    if (
        sum(
            len(field.name.encode()) + len(field.value.encode())
            for item in declarations
            for field in item.selector.fields
        )
        > limits.max_selector_bytes
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    keys = [(item.target, item.node, item.port) for item in declarations]
    if len(keys) != len(set(keys)) or len(capabilities) != len(set(capabilities)):
        reject(EffectCode.DUPLICATE)
    resource_sources = [item.source for item in resources]
    if len(resource_sources) != len(set(resource_sources)):
        reject(EffectCode.DUPLICATE)
    if set(resource_sources) != sources:
        reject(EffectCode.MISSING)
    schemas: dict[object, tuple[str, object]] = {}
    for declaration in declarations:
        item_type = declaration.materialization.item_type
        if schemas.get(item_type, ("scalar", item_type))[0] == "collection":
            reject(EffectCode.CONTRADICTORY)
        schemas.setdefault(item_type, ("scalar", item_type))
        candidate = (
            "scalar" if declaration.materialization.kind == "single" else "collection",
            item_type,
        )
        existing = schemas.get(declaration.artifact_type)
        if existing is not None and existing != candidate:
            reject(EffectCode.CONTRADICTORY)
        schemas[declaration.artifact_type] = candidate
    nodes = _reachable_operations(workflow)
    workflow_owners = {item.workflow for item in nodes}
    for declaration in declarations:
        if declaration.target not in data.targets:
            reject(EffectCode.FOREIGN_OWNER)
        node = nodes.get(declaration.node)
        if node is None:
            reject(EffectCode.MISSING if declaration.node.workflow in workflow_owners else EffectCode.FOREIGN_OWNER)
        inputs = {item.name: item.artifact_type for item in node.operation.inputs}
        context_ports = {use.port for outcome in node.operation.outcomes for use in outcome.context}
        if declaration.port not in inputs:
            reject(EffectCode.MISSING)
        if declaration.port not in context_ports or inputs[declaration.port] != declaration.artifact_type:
            reject(EffectCode.CONTRADICTORY)
        binding = next(
            (
                item
                for item in workflow.workflow.input_bindings
                if item.destination == NodeInputRef(node=declaration.node, port=declaration.port)
            ),
            None,
        )
        if binding is None or not isinstance(binding.source, ContextInputRef):
            reject(EffectCode.CONTRADICTORY)
        matches = [
            item
            for item in capabilities
            if item.source == declaration.source
            and item.artifact_type == declaration.materialization.item_type
            and "initial_binding" in item.uses
        ]
        if len(matches) != 1 or declaration.bounds.max_requests > matches[0].request.max_attempts:
            reject(EffectCode.UNSUPPORTED)
        resource = next(item for item in resources if item.source == declaration.source)
        if resource.capability != matches[0]:
            reject(EffectCode.CONTRADICTORY)
    context_destinations = {
        (binding.destination.node, binding.destination.port)
        for binding in workflow.workflow.input_bindings
        if isinstance(binding.source, ContextInputRef)
    }
    expected = {(target, node, port) for target in data.targets for node, port in context_destinations}
    observed = {(item.target, item.node, item.port) for item in declarations}
    if observed < expected:
        reject(EffectCode.MISSING)
    if observed > expected:
        reject(EffectCode.EXTRA)
    interface_types = {item.name: item.artifact_type for item in workflow.workflow.interface.inputs}
    collection_types = {artifact_type for artifact_type, schema in schemas.items() if schema[0] == "collection"}
    if any(
        isinstance(binding.source, WorkflowInputRef) and interface_types[binding.source.port] in collection_types
        for binding in workflow.workflow.input_bindings
    ):
        reject(EffectCode.CONTRADICTORY)


def _acquire_resources(work: _BindingWork) -> tuple[dict[object, ResourceLease], list[CleanupFact]]:
    acquired: dict[object, ResourceLease] = {}
    cleanup: list[CleanupFact] = []
    for resource in work.resources:
        if resource.lease is not None:
            acquired[resource.source] = resource.lease
            continue
        if work.cancelled:
            continue
        if resource.factory is None:
            reject(EffectCode.MISSING)
        provider = resource.factory()
        if not isinstance(provider, ContextProvider):
            reject(EffectCode.INVALID_TYPE)
        acquired[resource.source] = ResourceLease.create(
            owner="sdk",
            safe_detachment=resource.capability.safe_detachment,
            handle=provider,
        )
    return acquired, cleanup


async def _cleanup(acquired: dict[object, ResourceLease]) -> list[CleanupFact]:
    return [await close_resource(lease) for lease in acquired.values()]


def _capability(
    capabilities: tuple[ContextSourceCapability, ...], declaration: InitialContextDecl
) -> ContextSourceCapability:
    return next(
        item
        for item in capabilities
        if item.source == declaration.source and item.artifact_type == declaration.materialization.item_type
    )


def _binding_terminal(facts: list[SourceBindingFact]) -> BindingTerminal:
    if any(item.terminal == "lost" and item.declaration.requirement == "required" for item in facts):
        return "lost"
    if any(
        item.terminal in {"failed", "oversize", "cancelled"} and item.declaration.requirement == "required"
        for item in facts
    ):
        return "failed"
    if any(item.terminal != "bound" for item in facts):
        return "partial"
    return "success"


def _reachable_operations(workflow: AdmittedActivationWorkflow) -> dict[NodeId, Node]:
    operations: dict[NodeId, Node] = {}
    pending = [workflow.workflow]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        for node in current.nodes:
            operations[node.id] = node
            if isinstance(node, OperationNode):
                continue
            elif isinstance(node, SubgraphNode):
                pending.append(node.body)
    return operations
