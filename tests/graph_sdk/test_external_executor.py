# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""External execution tests for bounded remote-work ownership."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

import pytest

from anonymizer.engine.graph_sdk.capabilities import ImplementationCapability
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentLimits,
    DecisionLimits,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    IngressLimits,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.inspection import inspect_requests
from anonymizer.engine.graph_sdk.preparation import PreparationConfiguration
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestPolicy,
    PortArtifact,
    SemanticAssociation,
    StopConfirmed,
    TaskAttemptId,
    TextArtifactValue,
    TransportFailure,
    TransportLost,
    TransportResult,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.workflow import (
    NodeOutputRef,
    OperationNode,
    OutputBinding,
    OutputDependency,
    OutputPort,
    WorkflowOutputRef,
)
from tests.graph_sdk.implementation_fixtures import _execution_implementation
from tests.graph_sdk.test_adaptive_executor import _adaptive_rows
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _capability, _data, _limits, _prepare, _workflow
from tests.graph_sdk.workflow_construction import DynamicScope, admit_activation_workflow, admit_static_workflow


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


@dataclass
class _DelayedSuccessTransport:
    release: asyncio.Event
    first_dispatched: asyncio.Event
    calls: int = 0
    mode: str = "valid"
    envelopes: list[DispatchEnvelope] = field(default_factory=list)

    async def dispatch(self, request: DispatchEnvelope) -> TransportResult:
        self.calls += 1
        self.envelopes.append(request)
        if self.calls == 1:
            self.first_dispatched.set()
            await self.release.wait()
        if self.mode == "retry" and self.calls == 1:
            return TransportFailure(
                failure="retryable",
                settlement=ExternalSettlement(
                    request=request.request,
                    disposition="completed",
                    usage=ExactUsage(input_units=0, output_units=0),
                    remote_stopped=True,
                ),
            )
        rows = tuple(
            AssociationResult(
                association=item.association,
                outcome="ok",
                outputs=(
                    PortArtifact(
                        port="output",
                        artifact_type=request.operation.outputs[0].artifact_type,
                        artifact=None,
                        value=TextArtifactValue(text="x"),
                    ),
                )
                if self.mode.startswith("output_")
                else (),
                consumed_context_ports=frozenset(),
            )
            for item in reversed(request.associations)
        )
        if self.mode == "output_malformed" and self.calls == 1:
            first = rows[0]
            rows = (
                replace(first, outputs=(replace(first.outputs[0], value=TextArtifactValue(text=chr(0xD800))),)),
                *rows[1:],
            )
        if self.mode == "missing":
            rows = rows[:1]
        elif self.mode == "duplicate":
            rows = (rows[0], rows[0])
        elif self.mode == "extra":
            rows = (*rows, rows[0])
        elif self.mode == "foreign":
            association = rows[0].association
            assert isinstance(association, SemanticAssociation)
            rows = (
                replace(
                    rows[0],
                    association=SemanticAssociation(task=TaskAttemptId.new(activation=association.task.activation)),
                ),
                rows[1],
            )
        return TransportSuccess(
            results=rows,
            settlement=ExternalSettlement(
                request=request.request,
                disposition="completed",
                usage=ExactUsage(input_units=0, output_units=0),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: object) -> StopConfirmed:
        del request
        return StopConfirmed(usage=UnknownUsage())

    async def close(self) -> None:
        return None


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
    implementation = _execution_implementation(capability, request=request_policy)
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


def test_remote_capacity_waits_for_settlement_before_dispatching_later_work() -> None:
    asyncio.run(_assert_transient_remote_capacity())


def _capacity_plan(
    *, shared: bool, mode: str, ingress_limits: IngressLimits = IngressLimits()
) -> tuple[AdmittedExecutionPlan, ImplementationCapability]:
    workflow, node, artifact_type = _workflow(requests=2 if mode == "retry" else 1)
    if mode.startswith("output_"):
        static = workflow
        operation = replace(
            static.interface,
            outputs=(OutputPort(name="output", artifact_type=artifact_type),),
            output_dependencies=(OutputDependency(output="output", inputs=frozenset(), identity_input=None),),
            outcomes=(replace(static.interface.outcomes[0], produced_ports=frozenset({"output"})),),
        )
        rebuilt = admit_static_workflow(
            workflow=static.workflow,
            interface=operation,
            nodes=(OperationNode(id=node, operation=operation),),
            input_bindings=(),
            output_bindings=(
                OutputBinding(
                    source=NodeOutputRef(node=node, port="output"), destination=WorkflowOutputRef(port="output")
                ),
            ),
            outcome_bindings=tuple(static.outcome_bindings),
            sequence=(),
            choices=(),
            protection=(),
            limits=replace(static.limits, max_bindings=2),
        )
        workflow = admit_activation_workflow(
            workflow=rebuilt,
            scopes=(DynamicScope(workflow=rebuilt, maps=(), joins=(), loops=()),),
            limits=workflow.dynamic_limits,
        )
    capability: ImplementationCapability = _capability(workflow, external=True)
    if mode == "retry":
        capability = replace(capability, max_physical_requests_per_activation=2)
    if shared:
        capability = replace(capability, attribution="keyed_shared_request")
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
        max_attempts=2 if mode == "retry" else 1,
    )
    implementation = _execution_implementation(capability, request=request_policy)
    admitted = admit_execution_plan(
        ingress_limits=ingress_limits,
        context=context,
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=node,
                kind="external",
                request=request_policy,
                safe_detachment="independent_after_dispatch",
                implementations=(implementation,),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_adaptive_rows(),
            ),
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=2,
            max_provenance_edges=0,
        ),
    )
    return admitted, capability


async def _assert_transient_remote_capacity(*, shared: bool = False, mode: str = "valid") -> None:
    admitted, capability = _capacity_plan(shared=shared, mode=mode)
    release = asyncio.Event()
    first_dispatched = asyncio.Event()
    transport = _DelayedSuccessTransport(release=release, first_dispatched=first_dispatched, mode=mode)
    running = await start_execution(
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
                max_runtime_artifacts={"output_zero": 0, "output_limit": 1}.get(mode, 2),
                max_runtime_artifact_bytes=2,
                max_collection_items=0,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_Clock(),
        ),
    )
    await first_dispatched.wait()
    await asyncio.sleep(0)
    assert transport.calls == 1
    release.set()
    result = await running.wait()
    physical_count = 2 if mode == "retry" or not shared else 1
    assert transport.calls == physical_count
    assert [len(envelope.associations) for envelope in transport.envelopes] == (
        [2] * physical_count if shared else [1, 1]
    )
    if mode == "retry":
        assert transport.envelopes[0].associations == transport.envelopes[1].associations
        assert [envelope.purpose for envelope in transport.envelopes] == ["initial", "retry"]
    assert all(state.complete for state in result.states)
    if mode in {"valid", "retry", "output_valid"}:
        assert all(item.category == "success" for item in result.record.terminals)
    elif mode in {"output_limit", "output_malformed"}:
        assert sum(item.category == "success" for item in result.record.terminals) == 1
        assert len(result.artifacts) == len(result.ports) == len(result.final_outputs) == len(result.provenance) == 1
        assert not result.assessments
        if mode == "output_limit":
            assert len(result.publication_failures) == 1
            failure = result.publication_failures[0]
            assert failure.condition == "artifact_limit_exhausted"
            assert all(port.activation != failure.association.task.activation for port in result.ports)
        else:
            assert not result.publication_failures
            summaries = [row for terminal in result.requests.terminals for row in terminal.results]
            assert sorted(row.category for row in summaries) == ["failure", "success"]
            assert next(row for row in summaries if row.category == "failure").failure == "malformed_response"
        assert all(terminal.category == "success" for terminal in result.requests.terminals)
    elif mode == "output_zero":
        assert not result.artifacts and not result.ports and not result.provenance and not result.final_outputs
        assert len(result.publication_failures) == 2
    else:
        assert all(item.category == "inconsistent" for item in result.record.terminals)
    if mode == "output_valid":
        assert len(result.artifacts) == len(result.ports) == len(result.final_outputs) == 2
    assert result.requests.dispatched_count == physical_count
    assert len(result.requests.terminals) == result.requests.dispatched_count
    assert len(result.requests.settlements) == result.requests.dispatched_count
    assert not result.requests.remote_outstanding
    views = inspect_requests(result)
    assert len(views) == sum(len(dispatch.associations) for dispatch in result.requests.dispatches)
    assert all(view.ownership_resolved for view in views)
    assert {view.publication_failure for view in views if view.publication_failure is not None} == set(
        result.publication_failures
    )
    for view in views:
        assert isinstance(view.association, SemanticAssociation)
        assert view.operation_terminal is not None and view.operation_terminal.attempt == view.association.task
        assert not view.binding_artifacts and view.binding_terminal is None
        assert set(view.artifacts) == {
            port.artifact for port in result.ports if port.activation == view.association.task.activation
        }
        if view.acceptance is not None:
            assert not view.acceptance.artifacts
        if view.publication_failure is not None:
            assert not view.artifacts
    assert inspect_requests(result) == views and transport.calls == physical_count


@pytest.mark.parametrize("mode", ["valid", "missing", "duplicate", "extra", "foreign", "retry"])
def test_shared_request_dispatches_once_and_bridges_each_key(mode: str) -> None:
    asyncio.run(_assert_transient_remote_capacity(shared=True, mode=mode))


@pytest.mark.parametrize("mode", ["output_valid", "output_zero", "output_limit", "output_malformed"])
@pytest.mark.parametrize("shared", [True, False])
def test_independent_publication_survives_peer_rejection(mode: str, shared: bool) -> None:
    asyncio.run(_assert_transient_remote_capacity(shared=shared, mode=mode))
