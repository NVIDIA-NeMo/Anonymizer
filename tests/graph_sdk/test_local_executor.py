# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-free integration tests for the local graph executor."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    OperationExecutionPolicy,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    prepare,
)
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult
from anonymizer.graph import ActivationLimits, InputPort
from tests.graph_sdk.implementation_fixtures import _execution_implementation, _implementation_handle, _selection
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow
from tests.graph_sdk.workflow_fixtures import _shared_body_workflow


@dataclass
class _Clock:
    value: int = 0

    def now_ns(self) -> int:
        return self.value


@dataclass
class _Uppercase:
    calls: int = 0
    second_started: asyncio.Event = field(default_factory=asyncio.Event)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        assert len(request) == 1
        if self.calls == 1:
            await self.second_started.wait()
        else:
            self.second_started.set()
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


@dataclass
class _Immediate:
    calls: int = 0

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
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
    )
    selected = next(iter(prepared.implementations))
    implementation = _execution_implementation(selected.capability, request=None)
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
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
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
        limits=ExecutionLimits(
            max_local_in_flight=2,
            max_remote_outstanding=0,
            max_runtime_artifacts=10,
            max_runtime_artifact_bytes=100,
            max_collection_items=10,
        ),
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


def test_shared_nested_body_executes_each_real_occurrence() -> None:
    asyncio.run(_assert_shared_nested_body())


async def _assert_shared_nested_body() -> None:
    body_dynamic, child, artifact_type = _workflow(with_input=True)
    body = body_dynamic
    interface = replace(
        body.interface,
        name="shared-body-root",
        inputs=(InputPort(name="outer", artifact_type=artifact_type),),
        outcomes=tuple(
            replace(outcome, ceiling=replace(outcome.ceiling, max_activations=4)) for outcome in body.interface.outcomes
        ),
    )
    workflow, first, second = _shared_body_workflow(body, interface, max_bindings=3, input_binding=("outer", "input"))
    capability = _capability(body_dynamic)
    data = _data(1)
    target = next(iter(data.targets))
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=12, max_entries=4, max_parent_depth=2),
        bound_inputs=(BoundInput(target=target, source=target, port="outer", artifact_type=artifact_type),),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        selections=(_selection(child, capability),),
        capabilities=(capability,),
        limits=_limits(slots=4),
    )
    context = admit_context_plan(prepared=prepared, bound_context=None)
    implementation = _execution_implementation(capability, request=None)
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=child,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(implementation,),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_runtime_rows(),
            ),
        ),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_assessment_facts=0,
            max_port_facts=4,
            max_provenance_edges=0,
        ),
    )
    callback = _Immediate()
    result = await (
        await start_execution(
            admitted=admitted,
            capabilities=(capability,),
            services=ExecutionServices(
                handles=(_implementation_handle(capability, local=callback, transport=None, resource=None),),
                limits=ExecutionLimits(
                    max_local_in_flight=2,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=2,
                    max_runtime_artifact_bytes=100,
                    max_collection_items=0,
                ),
                clock=_Clock(),
            ),
        )
    ).wait()
    assert callback.calls == 2, [
        (item.template, item.activation.occurrence, item.activation.parent, item.status)
        for item in result.states[0].entries
    ]
    assert result.states[0].complete
    terminals = {item.activation: item for item in result.record.terminals}
    members = {member for membership in result.record.memberships for member in membership.members}
    assert set(terminals) == members
    for entry in result.states[0].entries:
        terminal = terminals[entry.activation]
        assert terminal.category == entry.status
        assert terminal.structural == (entry.template in {first, second})
        assert (terminal.attempt is None) == terminal.structural
    child_entries = [item for item in result.states[0].entries if item.template == child]
    assert len(child_entries) == 2
    assert len({item.activation.parent for item in child_entries}) == 2
    child_inputs = [item for item in result.ports if item.node == child and item.port == "input"]
    assert len(child_inputs) == 2
    assert child_inputs[0].artifact == child_inputs[1].artifact
