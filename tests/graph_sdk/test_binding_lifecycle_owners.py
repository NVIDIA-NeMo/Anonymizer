# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public binding lifecycle witnesses for neutral-only reference transitions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    InitialContextDecl,
    RetrievalBounds,
    SourceFailure,
    SourceItem,
    SourceResponse,
)
from anonymizer.engine.graph_sdk.requests import (
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    RequestAssociation,
    StopConfirmed,
)
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_binding import SOURCE, _context_workflow
from tests.graph_sdk.test_preparation import _data

OWNER_CASE_IDS = frozenset(
    {
        "materialization/latest_unresolved_optional_at_finish",
        "materialization/latest_post_finish_source_failure",
        "materialization/latest_post_finish_materialization",
    }
)


@dataclass
class _GatedOptionalProvider:
    optional_started: asyncio.Event = field(default_factory=asyncio.Event)
    release_optional: asyncio.Event = field(default_factory=asyncio.Event)
    calls: int = 0
    closes: int = 0
    sealed: bool = False

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse | SourceFailure:
        assert not self.sealed, "completed binding re-entered its provider"
        assert not selector.fields and bounds.max_items == 3
        self.calls += 1
        settlement = ExternalSettlement(
            request=request,
            disposition="completed",
            remote_stopped=True,
            usage=ExactUsage(input_units=0, output_units=0),
        )
        if self.calls == 1:
            return SourceResponse(
                source=SOURCE,
                items=(SourceItem(association=association, key=0, version=1, text="one"),),
                settlement=settlement,
            )
        assert self.calls == 2
        self.optional_started.set()
        await self.release_optional.wait()
        return SourceFailure(
            source=SOURCE,
            failure="permanent",
            disposition="omitted_optional",
            settlement=settlement,
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))

    async def close(self) -> None:
        self.closes += 1


def test_optional_request_blocks_finish_and_completed_wait_never_reenters_provider() -> None:
    cases = load_cases("effects")
    matched = [case for case in cases if case["case_id"] in OWNER_CASE_IDS]
    assert len(matched) == len(OWNER_CASE_IDS) == 3
    assert all(case["comparison_scope"] == "neutral_only" for case in matched)
    asyncio.run(_assert_binding_lifecycle())


async def _assert_binding_lifecycle() -> None:
    workflow, node, artifact_type = _context_workflow()
    data = _data(2)
    targets = tuple(data.targets)
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=artifact_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="sdk",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    declarations = tuple(
        InitialContextDecl(
            target=target,
            node=node,
            port="input",
            artifact_type=artifact_type,
            source=SOURCE,
            selector=ContextSelector(fields=()),
            requirement="required" if index == 0 else "optional",
            bounds=RetrievalBounds(max_items=3, max_bytes=12, max_requests=1),
            materialization=ContextMaterialization(kind="single", item_type=artifact_type),
            version_selection="latest",
        )
        for index, target in enumerate(targets)
    )
    provider = _GatedOptionalProvider()
    running = await start_initial_binding(
        data=data,
        workflow=workflow,
        declarations=declarations,
        capabilities=(capability,),
        resources=(ContextResource(source=SOURCE, capability=capability, lease=None, factory=lambda: provider),),
        limits=BindingLimits(
            max_declarations=2,
            max_sources=1,
            max_capabilities=1,
            max_selector_fields=0,
            max_selector_bytes=0,
            max_items=6,
            max_bytes=24,
            max_requests=2,
            max_resources=1,
        ),
    )
    waiting = asyncio.create_task(running.wait())
    try:
        await asyncio.wait_for(provider.optional_started.wait(), timeout=5)
        # Run the waiting task while the actual second provider request is gated.
        await asyncio.sleep(0)
        assert not waiting.done()
        assert provider.calls == 2 and provider.closes == 0
    finally:
        provider.release_optional.set()
    result = await asyncio.wait_for(waiting, timeout=5)
    assert result.context is not None and result.context.receipt is result.receipt
    assert result.receipt.terminal == "partial"
    assert tuple(fact.terminal for fact in result.receipt.sources) == ("bound", "omitted_optional")
    assert tuple(item.text for item in result.receipt.artifacts) == ("one",)
    assert result.context.artifacts == result.receipt.artifacts
    assert provider.calls == 2 and provider.closes == 1
    (cleanup,) = result.receipt.cleanup
    (association,) = result.receipt.cleanup_associations
    assert cleanup.resource == association.resource
    assert cleanup.owner == "sdk" and cleanup.disposition == "closed"
    assert association.targets == frozenset(targets) and association.purpose == "accounting"
    snapshot = (result.receipt.sources, result.receipt.artifacts, result.receipt.requests, result.receipt.cleanup)
    provider.sealed = True
    for _ in range(2):
        assert await running.wait() is result
        assert (
            result.receipt.sources,
            result.receipt.artifacts,
            result.receipt.requests,
            result.receipt.cleanup,
        ) == snapshot
    assert provider.calls == 2 and provider.closes == 1
