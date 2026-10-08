# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Asynchronous initial context binding over the shared request reducer."""

from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass

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
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    RequestCancel,
    RequestPolicyBinding,
    Reserve,
    ScopeCancel,
    StopAcknowledged,
    StopConfirmed,
    advance_requests,
    bind_request_policies,
    initialize_requests,
    request_receipt,
)
from anonymizer.engine.graph_sdk.resources import CleanupFact, ResourceLease, close_resource
from anonymizer.graph.workflow import AdmittedActivationWorkflow, NodeId, OperationNode, SubgraphNode


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
        association = BindingAssociation(declaration=identity)
        capability = _capability(work.capabilities, declaration)
        state = bind_request_policies(
            state=state,
            binding=RequestPolicyBinding.create(
                association=association,
                policies=frozenset({capability.request}),
            ),
        )
        request = PhysicalRequestId.new(scope=scope)
        state = advance_requests(
            state=state,
            event=Reserve(
                request=request,
                purpose="initial_binding",
                associations=frozenset({association}),
                policy=capability.request,
            ),
        )
        if not any(item.request == request for item in state.reserved):
            facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="failed"))
            continue
        if work.cancelled:
            state = advance_requests(state=state, event=RequestCancel(request=request))
            facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="cancelled"))
            continue
        state = advance_requests(state=state, event=Dispatch(request=request))
        provider = acquired[declaration.source].handle
        if not isinstance(provider, ContextProvider):
            reject(EffectCode.INVALID_TYPE)
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
                state = advance_requests(
                    state=state,
                    event=StopAcknowledged(request=request, usage=stopped.usage),
                )
                terminal = "cancelled"
            else:
                state = advance_requests(state=state, event=MarkLost(request=request))
                terminal = "lost"
            facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal=terminal))
            continue
        try:
            result = retrieval.result()
        except Exception:
            result = SourceLost(source=declaration.source, settlement=None)
        if isinstance(result, SourceResponse):
            if result.source != declaration.source or result.settlement.request != request:
                state = advance_requests(
                    state=state,
                    event=AcceptFailure(request=request, failure="malformed_response"),
                )
                facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="failed"))
                continue
            byte_count = sum(len(item.text.encode("utf-8")) for item in result.items)
            valid_associations = all(item.association == association for item in result.items)
            unique = len({(item.association, item.key, item.version) for item in result.items}) == len(result.items)
            oversize = (
                len(result.items) > declaration.bounds.max_items
                or byte_count > declaration.bounds.max_bytes
                or total_items + len(result.items) > work.limits.max_items
                or total_bytes + byte_count > work.limits.max_bytes
            )
            wrong_cardinality = not result.items or (
                declaration.materialization.kind == "single" and len(result.items) != 1
            )
            if not valid_associations or not unique or wrong_cardinality:
                state = advance_requests(state=state, event=MarkLost(request=request))
                facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="lost"))
                continue
            state = advance_requests(
                state=state,
                event=AcceptResult(
                    request=request,
                    results=(
                        AssociationResult(
                            association=association,
                            outcome="bound",
                            outputs=(),
                            consumed_context_ports=frozenset(),
                        ),
                    ),
                ),
            )
            state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
            if oversize:
                facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="oversize"))
                continue
            total_items += len(result.items)
            total_bytes += byte_count
            artifacts.extend(
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
            facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="bound"))
        elif isinstance(result, SourceFailure):
            failure = result.failure if result.source == declaration.source else "malformed_response"
            state = advance_requests(state=state, event=AcceptFailure(request=request, failure=failure))
            if result.settlement is not None and result.settlement.request == request:
                state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
            terminal = "omitted_optional" if declaration.requirement == "optional" else "failed"
            facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal=terminal))
        elif isinstance(result, SourceLost):
            if result.source != declaration.source:
                state = advance_requests(
                    state=state,
                    event=AcceptFailure(request=request, failure="malformed_response"),
                )
                facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="failed"))
                continue
            state = advance_requests(state=state, event=MarkLost(request=request))
            if result.settlement is not None and result.settlement.request == request:
                state = advance_requests(state=state, event=ObserveSettlement(settlement=result.settlement))
            facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="lost"))
        else:
            state = advance_requests(state=state, event=MarkLost(request=request))
            facts.append(SourceBindingFact(identity=identity, declaration=declaration, terminal="lost"))
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
    for declaration in declarations:
        if declaration.target not in data.targets:
            reject(EffectCode.FOREIGN_OWNER)
        node = nodes.get(declaration.node)
        if node is None:
            reject(EffectCode.FOREIGN_OWNER)
        inputs = {item.name: item.artifact_type for item in node.operation.inputs}
        context_ports = {use.port for outcome in node.operation.outcomes for use in outcome.context}
        if declaration.port not in context_ports or inputs.get(declaration.port) != declaration.artifact_type:
            reject(EffectCode.UNSUPPORTED)
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
