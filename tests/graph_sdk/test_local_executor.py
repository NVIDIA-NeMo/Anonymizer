# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-free integration tests for the local graph executor."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    OperationExecutionPolicy,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult
from tests.graph_sdk.test_preparation import _prepare


@dataclass
class _Clock:
    value: int = 0

    def now_ns(self) -> int:
        return self.value


@dataclass
class _Uppercase:
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        assert len(request) == 1
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome="ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                ),
            )
        )


def _runtime_rows() -> tuple[RuntimeOutcome, ...]:
    rows = [RuntimeOutcome(condition="result", reported_outcome="ok", failure=None, outcome="ok", category="success")]
    rows.extend(
        RuntimeOutcome(
            condition="failure",
            reported_outcome=None,
            failure=failure,
            outcome=None,
            category="failure",
        )
        for failure in (
            "rejected_before_acceptance",
            "retryable",
            "malformed_response",
            "permanent",
            "transport_unknown",
            "implementation_exception",
        )
    )
    rows.extend(
        RuntimeOutcome(
            condition=condition,
            reported_outcome=None,
            failure=None,
            outcome=None,
            category=category,
        )
        for condition, category in (
            ("cancel_before_start", "blocked"),
            ("cancel_after_start", "cancelled"),
            ("artifact_limit_exhausted", "blocked"),
            ("deadline_exhausted", "blocked"),
        )
    )
    return tuple(rows)


def test_local_execution_uses_real_callback_and_p3_state() -> None:
    asyncio.run(_assert_local_execution())


async def _assert_local_execution() -> None:
    prepared = _prepare(data=None)
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
        runtime_outcomes=_runtime_rows(),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(selected.capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=10,
            max_provenance_edges=10,
        ),
    )
    callback = _Uppercase()
    services = ExecutionServices(
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
            max_runtime_artifacts=10,
            max_runtime_artifact_bytes=100,
            max_collection_items=10,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_Clock(),
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=(selected.capability,),
        services=services,
    )
    result = await running.wait()
    assert callback.calls == 2
    assert len({state.invocation for state in result.states}) == 1
    assert all(state.complete for state in result.states)
    assert all(item.category == "success" for item in result.record.terminals)
    assert all(item.qualification == "not_assessed" for item in result.record.statuses)
