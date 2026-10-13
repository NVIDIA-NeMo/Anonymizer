# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal acquisition and cleanup expectations, independent of runtime collection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.context import ContextResource, ContextSourceCapability, admit_context_plan
from anonymizer.engine.graph_sdk.executor import admit_execution_plan, start_execution
from anonymizer.engine.graph_sdk.qualification import qualify_retained
from anonymizer.engine.graph_sdk.requests import PhysicalRequestPolicy
from anonymizer.engine.graph_sdk.resources import ResourceId, ResourceLease
from anonymizer.graph._values import InvocationId
from anonymizer.graph.workflow import ArtifactType
from tests.graph_sdk.child_qualification_fixtures import result_with
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_adaptive_executor import SOURCE, _adaptive_fixture, _Provider
from tests.graph_sdk.test_delegated_execution import _execute
from tests.graph_sdk.test_delegated_readiness import _qualify
from tests.graph_sdk.test_optional_tail import optional_fixture


@dataclass
class Provider(_Provider):
    factories: int = 0
    closes: int = 0
    mode: str = "normal"
    close_failure: bool = False

    def factory(self):
        self.factories += 1
        if self.mode == "raises":
            raise RuntimeError("before transfer")
        if self.mode == "invalid":

            class Invalid:
                async def close(inner):
                    await self.close()

            return Invalid()
        return self

    async def close(self):
        self.closes += 1
        if self.close_failure:
            raise RuntimeError("close failed")


def with_context(execution, services, capabilities, resources):
    context = admit_context_plan(
        prepared=execution.context.prepared,
        bound_context=execution.context.bound_context,
        adaptive_retrievals=execution.context.adaptive_retrievals,
        context_capabilities=capabilities,
    )
    execution = admit_execution_plan(
        context=context,
        capabilities=execution.capabilities,
        policies=tuple(execution.policies),
        decisions=tuple(execution.decisions),
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
        ingress_limits=execution.ingress_limits,
    )
    return execution, replace(services, context_resources=resources)


@pytest.mark.parametrize("reverse_resources", [False, True])
@pytest.mark.parametrize("reverse_capabilities", [False, True])
@pytest.mark.parametrize("extra_mode", ["normal", "raises", "invalid"])
def test_same_source_distinct_capabilities_keep_each_transfer_and_select_declared_provider(
    reverse_resources,
    reverse_capabilities,
    extra_mode,
):
    async def run():
        a, b = Provider(), Provider(mode=extra_mode)
        execution, services, _ = _adaptive_fixture(a, capability_limit=2)
        first = replace(execution.context.context_capabilities[0], resource_owner="sdk")
        second = replace(first, artifact_type=ArtifactType(name="extra", revision=1))
        capabilities = (second, first) if reverse_capabilities else (first, second)
        resources = tuple(
            ContextResource(source=SOURCE, capability=cap, lease=None, factory=p.factory)
            for cap, p in ((first, a), (second, b))
        )
        if reverse_resources:
            resources = resources[::-1]
        execution, services = with_context(execution, services, capabilities, resources)
        running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
        result = await running.wait()
        assert (a.factories, a.calls, a.closes) == (1, 1, 1)
        assert (b.factories, b.calls, b.closes) == (1, 0, int(extra_mode != "raises"))
        assert len(result.adaptive_acquisitions) == 2
        assert sum(row.outcome == "acquired" for row in result.adaptive_acquisitions) == (
            1 if extra_mode == "raises" else 2
        )
        assert len(result.cleanup) == len(result.cleanup_associations) == (2 if extra_mode == "raises" else 3)
        assert result.requests.dispatched_count == len(result.final_outputs) == 1
        assert [row.category for row in result.record.terminals] == ["success"]

    asyncio.run(run())


async def unused(*, mode="normal", failed=False, caller=False, shared=False):
    execution, services, _ = optional_fixture()
    policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=ArtifactType(name="text", revision=1),
        uses=frozenset({"adaptive_retrieval"}),
        execution="async",
        resource_owner="caller" if caller else "sdk",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    provider = Provider(mode=mode, close_failure=failed)
    lease = ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider) if caller else None
    resource = ContextResource(
        source=SOURCE, capability=capability, lease=lease, factory=None if caller else provider.factory
    )
    resources = (resource,)
    if shared:
        other = replace(capability, artifact_type=ArtifactType(name="extra", revision=1))
        resources = (*resources, replace(resource, capability=other))
    execution, services = with_context(execution, services, tuple(row.capability for row in resources), resources)
    result = await _execute(execution, services)
    return execution, result, provider


@pytest.mark.parametrize("mode", ["normal", "raises", "invalid"])
@pytest.mark.parametrize("failed", [False, True])
def test_unused_acquisition_distinguishes_no_transfer_from_invalid_return(mode, failed):
    async def run():
        execution, result, provider = await unused(mode=mode, failed=failed)
        assert (provider.factories, provider.calls, provider.closes) == (1, 0, int(mode != "raises"))
        (row,) = result.adaptive_acquisitions
        assert row.outcome == ("not_transferred" if mode == "raises" else "acquired")
        assert (row.resource is None) == (mode == "raises")
        assert len(result.cleanup) == len(result.cleanup_associations) == int(mode != "raises")
        expected = int(not failed or mode == "raises")
        assert len(_qualify(execution, result).qualified) == expected
        admitted, current, _ = _inputs(execution, result)
        assert len(qualify_retained(admitted=admitted, result=result, current=current).qualified) == expected
        assert provider.closes == int(mode != "raises")

    asyncio.run(run())


def test_shared_caller_lease_retains_both_capabilities_and_one_left_open_cleanup():
    async def run():
        execution, result, provider = await unused(caller=True, shared=True)
        assert (provider.factories, provider.calls, provider.closes) == (0, 0, 0)
        assert len(result.adaptive_acquisitions) == 2
        assert len({row.resource for row in result.adaptive_acquisitions}) == 1
        assert len(result.cleanup) == 1 and result.cleanup[0].disposition == "left_open"
        assert len(_qualify(execution, result).qualified) == 1
        assert not provider.close_failure

    asyncio.run(run())


@pytest.mark.parametrize(
    "mutation",
    ["pair", "triple", "row", "duplicate", "scope", "owner", "detachment", "foreign", "raw_source", "raw_policy"],
)
@pytest.mark.parametrize("failed", [False, True])
def test_adaptive_membership_cannot_be_removed_or_detached(mutation, failed):
    async def run():
        execution, result, _ = await unused(failed=failed)
        (row,) = result.adaptive_acquisitions
        cleanup, associations, rows = result.cleanup, result.cleanup_associations, (row,)
        if mutation in {"pair", "triple"}:
            cleanup, associations = (), ()
        if mutation in {"triple", "row"}:
            rows = ()
        elif mutation == "duplicate":
            rows = (row, row)
        elif mutation == "scope":
            rows = (replace(row, invocation=InvocationId.new(plan=result.record.plan)),)
        elif mutation == "owner":
            rows = (replace(row, owner="caller"),)
        elif mutation == "detachment":
            rows = (replace(row, safe_detachment="independent_after_dispatch"),)
        elif mutation == "foreign":
            rows = (replace(row, resource=ResourceId.new()),)
        elif mutation.startswith("raw_"):
            cap = replace(row.capability)
            if mutation == "raw_source":
                source = replace(cap.source)
                object.__setattr__(source, "name", [])
                object.__setattr__(cap, "source", source)
            else:
                policy = replace(cap.request)
                object.__setattr__(policy, "max_attempts", [])
                object.__setattr__(cap, "request", policy)
            rows = (replace(row, capability=cap),)
        changed = result_with(result, adaptive_acquisitions=rows, cleanup=cleanup, cleanup_associations=associations)
        output = _qualify(execution, changed)
        admitted, current, _ = _inputs(execution, changed)
        assert not qualify_retained(admitted=admitted, result=changed, current=current).qualified
        assert not output.qualified
        assert all("inconsistent_attribution" in target.withholding for target in output.targets)
        assert len(_qualify(execution, result).qualified) == int(not failed)

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["list", "missing", "overflow"])
def test_raw_acquisition_container_is_bounded_before_members(kind):
    async def run():
        execution, result, _ = await unused()
        changed = result_with(result)
        if kind == "missing":
            object.__delattr__(changed, "adaptive_acquisitions")
        else:
            object.__setattr__(
                changed,
                "adaptive_acquisitions",
                [] if kind == "list" else (object(),) * (execution.context.prepared.limits.max_capabilities + 1),
            )
        with pytest.raises(EffectRejected) as error:
            _qualify(execution, changed)
        assert error.value.code == (EffectCode.LIMIT_EXCEEDED if kind == "overflow" else EffectCode.INVALID_TYPE)

    asyncio.run(run())


@pytest.mark.parametrize("reverse", [False, True])
def test_two_declared_artifact_types_share_source_without_sharing_provider(reverse):
    from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, ImplementationSelection
    from anonymizer.engine.graph_sdk.preparation import prepare
    from anonymizer.graph.workflow import (
        DynamicScope,
        InputBinding,
        NodeId,
        NodeInputRef,
        OperationNode,
        OutputPort,
        SequenceEdge,
        WorkflowInputRef,
        admit_activation_workflow,
        admit_static_workflow,
    )

    async def run():
        a, b = Provider(), Provider()
        old, services, _ = _adaptive_fixture(a, capability_limit=2)
        previous = old.context.prepared
        static = previous.workflow.workflow
        (node_a,) = static.nodes
        node_b = NodeId.new(workflow=static.workflow)
        extra = ArtifactType(name="extra", revision=1)
        operation_b = replace(node_a.operation, outputs=(OutputPort(name="context", artifact_type=extra),))
        interface = replace(
            static.interface,
            outcomes=tuple(
                replace(
                    outcome,
                    ceiling=replace(
                        outcome.ceiling,
                        max_activations=2,
                        max_model_requests=2,
                        max_input_bytes=200,
                        max_output_bytes=200,
                    ),
                )
                for outcome in static.interface.outcomes
            ),
        )
        workflow = admit_static_workflow(
            workflow=static.workflow,
            interface=interface,
            nodes=(node_a, OperationNode(id=node_b, operation=operation_b)),
            input_bindings=(
                *static.input_bindings,
                InputBinding(
                    source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=node_b, port="input")
                ),
            ),
            output_bindings=tuple(static.output_bindings),
            outcome_bindings=tuple(static.outcome_bindings),
            sequence=(SequenceEdge(before=node_b, after=node_a.id),),
            choices=(),
            protection=(),
            limits=replace(static.limits, max_nodes=2, max_bindings=4, max_sequence_edges=1),
        )
        dynamic = admit_activation_workflow(
            workflow=workflow,
            scopes=(DynamicScope(workflow=workflow, maps=(), joins=(), loops=()),),
            limits=replace(previous.workflow.limits, max_activation_occurrences=2),
        )
        (cap_a,) = old.capabilities
        cap_b = replace(cap_a, implementation=ImplementationRef(name="second", revision=1), operation=operation_b)
        capabilities = (cap_a, cap_b)
        prepared = prepare(
            data=previous.data,
            workflow=dynamic,
            activation_limits=replace(previous.activation_limits, max_events=6, max_entries=2),
            bound_inputs=tuple(previous.bound_inputs),
            state=previous.state,
            selections=tuple(
                ImplementationSelection(node=node, implementation=cap.implementation, configuration=cap.configuration)
                for node, cap in ((node_a.id, cap_a), (node_b, cap_b))
            ),
            capabilities=capabilities,
            configuration=replace(previous.configuration, hard_request_limit=2),
            limits=previous.limits,
        )
        source_a = replace(old.context.context_capabilities[0], resource_owner="sdk")
        source_b = replace(source_a, artifact_type=extra)
        (declaration,) = old.context.adaptive_retrievals
        context = admit_context_plan(
            prepared=prepared,
            bound_context=None,
            adaptive_retrievals=(
                declaration,
                replace(
                    declaration, node=node_b, materialization=replace(declaration.materialization, item_type=extra)
                ),
            ),
            context_capabilities=(source_b, source_a) if reverse else (source_a, source_b),
        )
        (policy,) = old.policies
        (implementation,) = policy.implementations
        policy_b = replace(
            policy,
            node=node_b,
            implementations=(replace(implementation, implementation=cap_b.implementation, capability=cap_b),),
        )
        execution = admit_execution_plan(
            context=context,
            capabilities=capabilities,
            policies=(policy, policy_b),
            decisions=(),
            assessment_productions=(),
            assessment_limits=replace(old.assessment_limits, max_port_facts=4, max_provenance_edges=2),
        )
        (handle,) = services.handles
        resources = tuple(
            ContextResource(source=SOURCE, capability=cap, lease=None, factory=p.factory)
            for cap, p in ((source_a, a), (source_b, b))
        )
        services = replace(
            services,
            handles=(handle, replace(handle, implementation=cap_b.implementation, operation=operation_b)),
            context_resources=resources[::-1] if reverse else resources,
            limits=replace(services.limits, max_remote_outstanding=2, max_runtime_artifacts=4),
        )
        result = await _execute(execution, services)
        assert (a.calls, b.calls, a.closes, b.closes) == (1, 1, 1, 1)
        assert result.requests.dispatched_count == 2
        assert len(result.cleanup) == len(result.cleanup_associations) == 3
        assert len(result.adaptive_acquisitions) == 2
        assert all(row.category == "success" for row in result.record.terminals)
        assert len({row.request for row in result.requests.dispatches}) == 2

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["raises", "invalid"])
def test_selected_acquisition_failure_does_not_erase_other_transferred_lease(mode):
    async def run():
        a, b = Provider(mode=mode), Provider()
        execution, services, _ = _adaptive_fixture(a, capability_limit=2)
        first = replace(execution.context.context_capabilities[0], resource_owner="sdk")
        second = replace(first, artifact_type=ArtifactType(name="unused", revision=1))
        resources = tuple(
            ContextResource(source=SOURCE, capability=cap, lease=None, factory=p.factory)
            for cap, p in ((first, a), (second, b))
        )
        execution, services = with_context(execution, services, (first, second), resources)
        result = await _execute(execution, services)
        assert (a.calls, b.calls, a.closes, b.closes) == (0, 0, int(mode == "invalid"), 1)
        assert len(result.adaptive_acquisitions) == 2
        assert result.requests.dispatched_count == 0
        assert len(result.cleanup) == (3 if mode == "invalid" else 2)
        assert [row.category for row in result.record.terminals] == ["failure"]

    asyncio.run(run())


@pytest.mark.parametrize("before_entry", [False, True])
def test_cancelled_adaptive_execution_preserves_acquisition_and_uncertain_stop_membership(before_entry):
    from anonymizer.engine.graph_sdk.requests import StopUnknown

    async def run():
        entered, release, stopped = asyncio.Event(), asyncio.Event(), asyncio.Event()

        class Held(Provider):
            async def retrieve(self, **kwargs):
                task = asyncio.current_task()
                assert task is not None
                self.task = task
                entered.set()
                await release.wait()
                return await super().retrieve(**kwargs)

            async def cancel(self, request):
                stopped.set()
                return StopUnknown()

            async def close(self):
                assert before_entry or self.task.done()
                await super().close()

        provider = Held()
        execution, services, _ = _adaptive_fixture(provider)
        capability = replace(execution.context.context_capabilities[0], resource_owner="sdk")
        resource = ContextResource(source=SOURCE, capability=capability, lease=None, factory=provider.factory)
        execution, services = with_context(execution, services, (capability,), (resource,))
        running = await start_execution(admitted=execution, capabilities=execution.capabilities, services=services)
        if before_entry:
            running.request_cancel()
        else:
            await entered.wait()
            running.request_cancel()
            await stopped.wait()
            for _ in range(2):
                observer = asyncio.create_task(running.wait())
                await asyncio.sleep(0)
                observer.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await observer
            assert not running._task.done()
            assert provider.closes == 0
            release.set()
        result = await running.wait()
        assert provider.factories == 1
        assert result.requests.dispatched_count == int(not before_entry)
        if before_entry:
            assert provider.closes == 1 and provider.calls == 0
        else:
            assert provider.closes == 0
            assert any(row.owner == "sdk" and row.disposition == "left_open" for row in result.cleanup)
            with pytest.raises(asyncio.CancelledError):
                await provider.task
            assert provider.calls == 0
        assert len(result.adaptive_acquisitions) == 1
        assert result.adaptive_acquisitions[0].outcome == "acquired"
        assert len(result.cleanup) == len(result.cleanup_associations) == 2
        assert all(state.complete for state in result.states)
        if not before_entry:
            assert result.requests.terminals[0].category == "lost"
            assert result.requests.remote_outstanding

    asyncio.run(run())
