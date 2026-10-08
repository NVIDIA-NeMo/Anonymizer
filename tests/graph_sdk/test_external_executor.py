# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External execution tests for bounded remote-work ownership."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from anonymizer.engine.graph_sdk.capabilities import ImplementationCapability
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import PreparationConfiguration
from anonymizer.engine.graph_sdk.requests import (
    DispatchEnvelope,
    PhysicalRequestPolicy,
    StopConfirmed,
    TransportLost,
    TransportResult,
    UnknownUsage,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.test_adaptive_executor import _adaptive_rows
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow


@dataclass
class _LostTransport:
    calls: int = 0
    closed: bool = False

    async def dispatch(self, request: DispatchEnvelope) -> TransportResult:
        del request
        self.calls += 1
        return TransportLost(settlement=None)

    async def cancel(self, request: object) -> StopConfirmed:
        del request
        return StopConfirmed(usage=UnknownUsage())

    async def close(self) -> None:
        self.closed = True


def test_remote_outstanding_capacity_closes_later_work_without_overdispatch() -> None:
    asyncio.run(_assert_remote_capacity())


async def _assert_remote_capacity() -> None:
    workflow, node, _ = _workflow(requests=1)
    capability: ImplementationCapability = _capability(workflow, external=True)
    prepared = _prepare(
        data=_data(2),
        workflow=workflow,
        capability=capability,
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=2,
        ),
        limits=_limits(capabilities=1),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    request_policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=request_policy,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="external",
        request=request_policy,
        safe_detachment="independent_after_dispatch",
        implementations=(implementation,),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_adaptive_rows(),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=0,
            max_provenance_edges=0,
        ),
    )
    transport = _LostTransport()
    result = await (
        await start_execution(
            admitted=admitted,
            capabilities=(capability,),
            services=ExecutionServices(
                handles=(
                    ImplementationHandle(
                        implementation=capability.implementation,
                        operation=capability.operation,
                        configuration=capability.configuration,
                        local=None,
                        transport=transport,
                        resource=ResourceLease.create(
                            owner="sdk",
                            safe_detachment="independent_after_dispatch",
                            handle=transport,
                        ),
                    ),
                ),
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=0,
                    max_remote_outstanding=1,
                    max_runtime_artifacts=0,
                    max_runtime_artifact_bytes=0,
                    max_collection_items=0,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    assert transport.calls == 1
    assert transport.closed
    assert all(state.complete for state in result.states)
    assert sorted(item.category for item in result.record.terminals) == ["blocked", "lost"]
    assert len(result.requests.remote_outstanding) == 1
    assert result.cleanup[0].disposition == "closed"
