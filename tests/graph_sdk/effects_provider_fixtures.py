# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider doubles and local bridge fixtures for effects conformance."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, cast

from anonymizer.engine.graph_sdk.context import (
    ContextSelector,
    ContextSourceRef,
    RetrievalBounds,
    SourceFailure,
    SourceItem,
    SourceResponse,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.executor import (
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    BindingAssociation,
    BindingDeclarationId,
    BindingId,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    PortArtifact,
    RequestAssociation,
    SemanticAssociation,
    StopConfirmed,
    TextArtifactValue,
    UnknownUsage,
)
from tests.graph_sdk.effects_admission_fixtures import _assessment_limits, _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.effects_request_fixtures import _usage
from tests.graph_sdk.test_preparation import _data, _prepare


class _CaseProvider:
    def __init__(
        self,
        *,
        events: list[dict[str, Any]],
        sources: dict[str, ContextSourceRef],
        association_names: dict[object, str],
        request_names: dict[PhysicalRequestId, str],
        fallback_associations: list[str],
        default_source: ContextSourceRef,
    ) -> None:
        self.events = events
        self.sources = sources
        self.association_names = association_names
        self.request_names = request_names
        self.fallback_associations = fallback_associations
        self.default_source = default_source
        self.closes = 0

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: RequestAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse | SourceFailure:
        del selector, bounds
        event = self.events.pop(0)
        label = self.association_names.get(association)
        if label is None:
            label = self.fallback_associations.pop(0)
        self.association_names[association] = label
        self.request_names[request] = cast(str, event["request"])
        settlement_value = cast(dict[str, Any] | None, event.get("settlement"))
        settlement = None
        if settlement_value is not None:
            settlement = ExternalSettlement(
                request=request,
                disposition=settlement_value["disposition"],
                usage=_usage(settlement_value["usage"]),
                remote_stopped=settlement_value["remote_stopped"],
            )
        source = self.sources[cast(str, event["source"])] if "source" in event else self.default_source
        if event["kind"] == "source_failure":
            return SourceFailure(
                source=source,
                failure=event["failure"],
                settlement=settlement,
                disposition=event.get("disposition", "failed"),
            )
        assert settlement is not None
        return SourceResponse(
            source=source,
            items=tuple(
                SourceItem(
                    association=(
                        association
                        if item.get("association", event.get("association")) == label
                        else BindingAssociation(
                            declaration=BindingDeclarationId.new(binding=BindingId.new(), ordinal=0)
                        )
                    ),
                    key=item["key"],
                    version=item["version"],
                    text=item.get("text", item.get("value")),
                )
                for item in cast(list[dict[str, Any]], event["items"])
            ),
            settlement=settlement,
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))

    async def close(self) -> None:
        self.closes += 1


class _CloseFailureProvider(_CaseProvider):
    async def close(self) -> None:
        await super().close()
        raise RuntimeError("scripted close failure")


@dataclass
class _ProviderWithoutClose:
    delegate: _CaseProvider

    async def retrieve(self, **kwargs: Any) -> SourceResponse | SourceFailure:
        return await self.delegate.retrieve(**kwargs)

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        return await self.delegate.cancel(request)


@dataclass
class _BridgeLocalCallback:
    failure: str | None
    calls: int = 0
    block: bool = False
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted | LocalFailure:
        self.calls += 1
        self.started.set()
        if self.block:
            await asyncio.Event().wait()
        if self.failure is not None:
            return LocalFailure(failure=cast(Any, self.failure))
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                )
                for item in request
            )
        )


async def _assert_local_bridge_case(case: dict[str, Any]) -> None:
    prepared = _prepare(data=_data(1))
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    selected = next(iter(prepared.implementations))
    implementation = ExecutionImplementation(
        implementation=selected.capability.implementation,
        configuration=selected.capability.configuration,
        capability=selected.capability,
        request=None,
    )
    policy = OperationExecutionPolicy(
        node=selected.node,
        kind="local",
        request=None,
        safe_detachment="forbidden",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_valid_runtime_rows("local", frozenset({"ok"})),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(selected.capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        assessment_limits=_assessment_limits(),
    )
    condition = (
        next(
            cast(str, event["failure"])
            for event in cast(list[dict[str, Any]], case["events"])
            if event["kind"] == "bridge_condition" and event.get("failure") is not None
        )
        if "/failure_" in cast(str, case["case_id"])
        else None
    )
    callback = _BridgeLocalCallback(
        failure=condition,
        block=case["case_id"] == "bridges/cancel_after_start",
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=(selected.capability,),
        services=ExecutionServices(
            handles=(
                ImplementationHandle(
                    implementation=implementation.implementation,
                    operation=implementation.capability.operation,
                    configuration=implementation.configuration,
                    local=callback,
                    transport=None,
                    resource=None,
                ),
            ),
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=0,
                max_runtime_artifacts=4,
                max_runtime_artifact_bytes=64,
                max_collection_items=1,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_ZeroClock(),
        ),
    )
    if case["case_id"] == "bridges/cancel_before_start":
        running.request_cancel()
    if callback.block:
        await callback.started.wait()
        running.request_cancel()
    result = await running.wait()
    expected = cast(dict[str, Any], case["expected"])["state"]
    if case["case_id"] == "bridges/cancel_before_start":
        assert callback.calls == 0
        assert len(result.record.terminals) == 1
        assert result.record.terminals[0].attempt is None
        assert result.record.terminals[0].category == expected["closed_unstarted"]["A0"]
        assert not result.requests.dispatches and not result.artifacts
        assert all(state.complete for state in result.states)
        return
    assert callback.calls == 1
    assert len(result.record.terminals) == 1
    terminal = result.record.terminals[0]
    assert terminal.category == expected["tasks"]["T0"]
    assert terminal.attempt is not None


@dataclass
class _LateAdaptiveProvider:
    source: ContextSourceRef
    association_names: dict[object, str]
    request_names: dict[PhysicalRequestId, str]
    acknowledge_stop: bool
    started: asyncio.Event = field(default_factory=asyncio.Event)
    returned_after_cancel: bool = False

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: SemanticAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse:
        del selector, bounds
        self.association_names[association] = "A0"
        self.request_names[request] = "R0"
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.returned_after_cancel = True
        return SourceResponse(
            source=self.source,
            items=(SourceItem(association=association, key=0, version=1, text="late"),),
            settlement=ExternalSettlement(
                request=request,
                disposition="completed",
                usage=ExactUsage(input_units=0, output_units=0),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        if not self.acknowledge_stop:
            raise RuntimeError("transport did not acknowledge cancellation")
        return StopConfirmed(usage=UnknownUsage())


@dataclass
class _LateBindingCaseProvider:
    delegate: _CaseProvider
    acknowledge_stop: bool
    started: asyncio.Event = field(default_factory=asyncio.Event)

    async def retrieve(self, **kwargs: Any) -> SourceResponse | SourceFailure:
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            pass
        return await self.delegate.retrieve(**kwargs)

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        if not self.acknowledge_stop:
            raise RuntimeError("stop not acknowledged")
        return StopConfirmed(usage=UnknownUsage())

    async def close(self) -> None:
        await self.delegate.close()


@dataclass
class _ContextConsumer:
    port: str
    outputs: tuple[PortArtifact, ...] = ()
    expected_input: str | None = None
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        assert len(request) == 1
        if self.expected_input is not None:
            (item,) = request[0].inputs
            assert isinstance(item.value, TextArtifactValue)
            assert item.value.text == self.expected_input
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome="ok",
                    outputs=self.outputs,
                    consumed_context_ports=frozenset({self.port}),
                ),
            )
        )
