# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initial context binding tests with real provider calls and receipts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    ContextSourceRef,
    InitialContextDecl,
    RetrievalBounds,
    SourceFailure,
    SourceItem,
    SourceResponse,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.requests import (
    BindingAssociation,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    StopConfirmed,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.workflow import (
    ContextInputRef,
    ContextUse,
    DynamicLimits,
    DynamicScope,
    NodeId,
    OperationNode,
    WorkflowId,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_preparation import _capability, _data, _prepare, _workflow


@dataclass
class _Provider:
    calls: int = 0

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: BindingAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse:
        self.calls += 1
        assert bounds.max_items == 1 and not selector.fields
        return SourceResponse(
            source=SOURCE,
            items=(SourceItem(association=association, key=0, version=1, text="context"),),
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                usage=ExactUsage(input_units=1, output_units=1),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: object) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


@dataclass
class _RetryProvider:
    calls: int = 0

    async def retrieve(self, *, request, association, selector, bounds):
        del selector, bounds
        self.calls += 1
        settlement = ExternalSettlement(
            request=request,
            disposition="completed",
            usage=ExactUsage(input_units=1, output_units=1),
            remote_stopped=True,
        )
        if self.calls == 1:
            return SourceFailure(source=SOURCE, failure="retryable", settlement=settlement)
        return SourceResponse(
            source=SOURCE,
            items=(SourceItem(association=association, key=0, version=1, text="retried"),),
            settlement=settlement,
        )

    async def cancel(self, request):
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


SOURCE = ContextSourceRef(name="test-source", revision=1)
OTHER_SOURCE = ContextSourceRef(name="other-source", revision=1)


def test_source_failure_rejects_incomplete_and_contradictory_values_at_construction() -> None:
    with pytest.raises(TypeError):
        cast(Any, SourceFailure)(source=SOURCE)
    with pytest.raises(TypeError):
        cast(Any, SourceFailure)(source=SOURCE, failure="permanent")
    with pytest.raises(EffectRejected) as rejected:
        SourceFailure(
            source=SOURCE,
            failure="retryable",
            settlement=None,
            disposition="omitted_optional",
        )
    assert rejected.value.code.value == "contradictory"
    assert (
        SourceFailure(
            source=SOURCE,
            failure="permanent",
            settlement=None,
            disposition="omitted_optional",
        ).disposition
        == "omitted_optional"
    )


@dataclass
class _BoundaryProvider:
    mode: str

    async def retrieve(self, *, request, association, selector, bounds):
        del selector, bounds
        items = (
            SourceItem(
                association=association,
                key=0,
                version=1,
                text="too-long" if self.mode == "oversize" else "ok",
            ),
        )
        if self.mode == "multiple":
            items += (SourceItem(association=association, key=1, version=1, text="b"),)
        elif self.mode == "duplicate":
            items += (SourceItem(association=association, key=0, version=1, text="duplicate"),)
        return SourceResponse(
            source=OTHER_SOURCE if self.mode == "malformed" else SOURCE,
            items=items,
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                usage=ExactUsage(input_units=1, output_units=1),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request):
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


def _context_workflow():
    admitted, node, artifact = _workflow(with_input=True)
    static = admitted.workflow
    operation_node = next(item for item in static.nodes if isinstance(item, OperationNode))
    outcome = replace(
        operation_node.operation.outcomes[0],
        context=frozenset({ContextUse(port="input", meaning="test", capture="whole_artifact")}),
    )
    operation = replace(operation_node.operation, outcomes=(outcome,))
    rebuilt = admit_static_workflow(
        workflow=static.workflow,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=tuple(
            replace(binding, source=ContextInputRef(port=binding.source.port)) for binding in static.input_bindings
        ),
        output_bindings=tuple(static.output_bindings),
        outcome_bindings=tuple(static.outcome_bindings),
        sequence=tuple(static.sequence),
        choices=tuple(static.choices),
        protection=(),
        limits=static.limits,
    )
    return (
        admit_activation_workflow(
            workflow=rebuilt,
            scopes=(DynamicScope(workflow=rebuilt, maps=(), joins=(), loops=()),),
            limits=DynamicLimits(
                max_maps=0,
                max_joins=0,
                max_loops=0,
                max_children_per_map=0,
                max_iterations_per_loop=0,
                max_dynamic_depth=1,
                max_activation_occurrences=1,
            ),
        ),
        node,
        artifact,
    )


def test_initial_binding_preserves_provider_text_and_receipt() -> None:
    asyncio.run(_assert_initial_binding())


def test_context_source_requires_exact_initial_declaration_before_provider_effects() -> None:
    asyncio.run(_assert_context_source_requires_exact_initial_declaration())


async def _assert_context_source_requires_exact_initial_declaration() -> None:
    workflow, node, artifact = _context_workflow()
    data = _data(1)
    target = next(iter(data.targets))
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=artifact,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    provider = _Provider()
    base = InitialContextDecl(
        target=target,
        node=node,
        port="input",
        artifact_type=artifact,
        source=SOURCE,
        selector=ContextSelector(fields=()),
        requirement="required",
        bounds=RetrievalBounds(max_items=1, max_bytes=20, max_requests=1),
        materialization=ContextMaterialization(kind="single", item_type=artifact),
    )
    absent = NodeId.new(workflow=node.workflow)
    cases = (
        ((), "missing"),
        ((base, base), "duplicate"),
        ((replace(base, port="absent"),), "missing"),
        ((replace(base, node=absent),), "missing"),
        ((replace(base, node=NodeId.new(workflow=WorkflowId.new())),), "foreign_owner"),
    )
    for declarations, code in cases:
        with pytest.raises(EffectRejected) as rejected:
            await start_initial_binding(
                data=data,
                workflow=workflow,
                declarations=declarations,
                capabilities=(capability,),
                resources=(
                    ContextResource(
                        source=SOURCE,
                        capability=capability,
                        lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                        factory=None,
                    ),
                ),
                limits=BindingLimits(
                    max_declarations=2,
                    max_sources=1,
                    max_capabilities=1,
                    max_selector_fields=0,
                    max_selector_bytes=0,
                    max_items=1,
                    max_bytes=20,
                    max_requests=1,
                    max_resources=1,
                ),
            )
        assert rejected.value.code.value == code
    assert provider.calls == 0

    prepared = _prepare(data=data, workflow=workflow, capability=_capability(workflow), bound_inputs=())
    with pytest.raises(EffectRejected) as rejected:
        admit_context_plan(
            prepared=prepared,
            bound_context=None,
            adaptive_retrievals=(),
            context_capabilities=(),
        )
    assert rejected.value.code.value == "missing"


async def _assert_initial_binding() -> None:
    workflow, node, artifact = _context_workflow()
    data = _data(1)
    target = next(iter(data.targets))
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=artifact,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    declaration = InitialContextDecl(
        target=target,
        node=node,
        port="input",
        artifact_type=artifact,
        source=SOURCE,
        selector=ContextSelector(fields=()),
        requirement="required",
        bounds=RetrievalBounds(max_items=1, max_bytes=20, max_requests=1),
        materialization=ContextMaterialization(kind="single", item_type=artifact),
    )
    provider = _Provider()
    lease = ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider)
    running = await start_initial_binding(
        data=data,
        workflow=workflow,
        declarations=(declaration,),
        capabilities=(capability,),
        resources=(ContextResource(source=SOURCE, capability=capability, lease=lease, factory=None),),
        limits=BindingLimits(
            max_declarations=1,
            max_sources=1,
            max_capabilities=1,
            max_selector_fields=0,
            max_selector_bytes=0,
            max_items=1,
            max_bytes=20,
            max_requests=1,
            max_resources=1,
        ),
    )
    result = await running.wait()
    assert provider.calls == 1
    assert result.context is not None
    assert result.context.artifacts[0].text == "context"
    assert result.receipt.data is data
    assert result.receipt.workflow is workflow
    assert result.receipt.requests.dispatched_count == 1
    assert result.receipt.requests.terminals[0].results[0].outcome == "retrieved"
    assert result.receipt.cleanup[0].disposition == "left_open"


def test_initial_binding_owns_retry_and_charges_each_request() -> None:
    asyncio.run(_assert_initial_binding_retry())


def test_initial_binding_retains_matching_settlement_for_malformed_and_oversize_responses() -> None:
    asyncio.run(_assert_initial_binding_boundary_settlements())


async def _assert_initial_binding_boundary_settlements() -> None:
    workflow, node, artifact = _context_workflow()
    data = _data(1)
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=artifact,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    for mode, expected_request, expected_source in (
        ("malformed", "failure", "failed"),
        ("duplicate", "failure", "failed"),
        ("oversize", "success", "oversize"),
        ("multiple", "success", "oversize"),
    ):
        provider = _BoundaryProvider(mode=mode)
        declaration = InitialContextDecl(
            target=next(iter(data.targets)),
            node=node,
            port="input",
            artifact_type=artifact,
            source=SOURCE,
            selector=ContextSelector(fields=()),
            requirement="required",
            bounds=RetrievalBounds(max_items=1, max_bytes=2, max_requests=1),
            materialization=ContextMaterialization(kind="single", item_type=artifact),
        )
        result = await (
            await start_initial_binding(
                data=data,
                workflow=workflow,
                declarations=(declaration,),
                capabilities=(capability,),
                resources=(
                    ContextResource(
                        source=SOURCE,
                        capability=capability,
                        lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                        factory=None,
                    ),
                ),
                limits=BindingLimits(
                    max_declarations=1,
                    max_sources=1,
                    max_capabilities=1,
                    max_selector_fields=0,
                    max_selector_bytes=0,
                    max_items=1,
                    max_bytes=2,
                    max_requests=1,
                    max_resources=1,
                ),
            )
        ).wait()
        assert result.receipt.terminal == "failed"
        assert result.receipt.sources[0].terminal == expected_source
        assert result.receipt.requests.terminals[0].category == expected_request
        if expected_request == "failure":
            assert result.receipt.requests.terminals[0].failure == "malformed_response"
        assert result.receipt.requests.settlements[0].disposition == "completed"
        assert result.receipt.requests.settlements[0].usage == ExactUsage(input_units=1, output_units=1)
        assert not result.receipt.artifacts


async def _assert_initial_binding_retry() -> None:
    workflow, node, artifact = _context_workflow()
    data = _data(1)
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=2,
    )
    capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=artifact,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    declaration = InitialContextDecl(
        target=next(iter(data.targets)),
        node=node,
        port="input",
        artifact_type=artifact,
        source=SOURCE,
        selector=ContextSelector(fields=()),
        requirement="required",
        bounds=RetrievalBounds(max_items=1, max_bytes=20, max_requests=2),
        materialization=ContextMaterialization(kind="single", item_type=artifact),
    )
    provider = _RetryProvider()
    result = await (
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=(declaration,),
            capabilities=(capability,),
            resources=(
                ContextResource(
                    source=SOURCE,
                    capability=capability,
                    lease=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider),
                    factory=None,
                ),
            ),
            limits=BindingLimits(
                max_declarations=1,
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=1,
                max_bytes=20,
                max_requests=2,
                max_resources=1,
            ),
        )
    ).wait()
    assert provider.calls == 2
    assert result.receipt.terminal == "success"
    assert result.context is not None and result.context.artifacts[0].text == "retried"
    assert [item.purpose for item in result.receipt.requests.dispatches] == ["initial_binding", "retry"]
