# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable coverage for explicit static context input sources."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
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
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    BindingAssociation,
    ExactUsage,
    ExternalSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    PortArtifact,
    StopConfirmed,
    TextArtifactValue,
    TextCollectionValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    ContextInputRef,
    ContextUse,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_local_executor import _Clock, _runtime_rows
from tests.graph_sdk.test_preparation import _capability, _data, _limits

SOURCE = ContextSourceRef(name="context-source", revision=1)


@dataclass
class _ContextProvider:
    items: tuple[str, ...]
    omit: bool = False
    calls: int = 0

    async def retrieve(
        self,
        *,
        request: PhysicalRequestId,
        association: BindingAssociation,
        selector: ContextSelector,
        bounds: RetrievalBounds,
    ) -> SourceResponse | SourceFailure:
        del selector, bounds
        self.calls += 1
        settlement = ExternalSettlement(
            request=request,
            disposition="completed",
            usage=ExactUsage(input_units=0, output_units=0),
            remote_stopped=True,
        )
        if self.omit:
            return SourceFailure(
                source=SOURCE,
                failure="permanent",
                settlement=settlement,
                disposition="omitted_optional",
            )
        return SourceResponse(
            source=SOURCE,
            items=tuple(
                SourceItem(association=association, key=index, version=1, text=text)
                for index, text in enumerate(self.items)
            ),
            settlement=settlement,
        )

    async def cancel(self, request: PhysicalRequestId) -> StopConfirmed:
        del request
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))


@dataclass
class _IdentityCallback:
    calls: int = 0
    observed: list[object] = field(default_factory=list)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        value = request[0].inputs[0]
        self.observed.append(value.value)
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome="ok",
                    outputs=(
                        PortArtifact(
                            port="output",
                            artifact_type=value.artifact_type,
                            artifact=None,
                            value=value.value,
                        ),
                    ),
                    consumed_context_ports=frozenset({"context"}),
                ),
            )
        )


def _operation(artifact_type: ArtifactType) -> OperationSpec:
    return OperationSpec(
        name="context-identity",
        inputs=(InputPort(name="context", artifact_type=artifact_type),),
        outputs=(OutputPort(name="output", artifact_type=artifact_type),),
        output_dependencies=(
            OutputDependency(output="output", inputs=frozenset({"context"}), identity_input="context"),
        ),
        outcomes=(
            OutcomeSpec(
                name="ok",
                category="success",
                produced_ports=frozenset({"output"}),
                context=frozenset(
                    {ContextUse(port="context", meaning="retrieved", capture="whole_artifact")}
                ),
                evidence=frozenset(),
                state_effects=frozenset(),
                model_requirements=frozenset(),
                ceiling=ResourceCeiling(
                    max_activations=2,
                    max_model_requests=0,
                    max_input_bytes=16,
                    max_output_bytes=16,
                ),
            ),
        ),
    )


def _single_workflow(artifact_type: ArtifactType):
    owner = WorkflowId.new()
    node = NodeId.new(workflow=owner)
    operation = _operation(artifact_type)
    static = admit_static_workflow(
        workflow=owner,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=(
            InputBinding(
                source=ContextInputRef(port="context"),
                destination=NodeInputRef(node=node, port="context"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=node, port="output"),
                destination=WorkflowOutputRef(port="output"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=1,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    return (
        admit_activation_workflow(
            workflow=static,
            scopes=(DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
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
        node,
    )


def _nested_workflow(artifact_type: ArtifactType):
    operation = _operation(artifact_type)
    body_owner = WorkflowId.new()
    inner = NodeId.new(workflow=body_owner)
    body = admit_static_workflow(
        workflow=body_owner,
        interface=operation,
        nodes=(OperationNode(id=inner, operation=operation),),
        input_bindings=(
            InputBinding(
                source=ContextInputRef(port="context"),
                destination=NodeInputRef(node=inner, port="context"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=inner, port="output"),
                destination=WorkflowOutputRef(port="output"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=inner, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=1,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    owner = WorkflowId.new()
    wrapper = NodeId.new(workflow=owner)
    root = admit_static_workflow(
        workflow=owner,
        interface=operation,
        nodes=(SubgraphNode(id=wrapper, operation=operation, body=body),),
        input_bindings=(
            InputBinding(
                source=ContextInputRef(port="context"),
                destination=NodeInputRef(node=wrapper, port="context"),
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=wrapper, port="output"),
                destination=WorkflowOutputRef(port="output"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=wrapper, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=2,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2,
            max_choice_states=1,
        ),
    )
    return (
        admit_activation_workflow(
            workflow=root,
            scopes=(
                DynamicScope(workflow=root, maps=(), joins=(), loops=()),
                DynamicScope(workflow=body, maps=(), joins=(), loops=()),
            ),
            limits=DynamicLimits(
                max_maps=0,
                max_joins=0,
                max_loops=0,
                max_children_per_map=0,
                max_iterations_per_loop=0,
                max_dynamic_depth=2,
                max_activation_occurrences=2,
            ),
        ),
        wrapper,
        inner,
    )


def test_scalar_collection_and_nested_context_sources_execute_with_exact_identity() -> None:
    asyncio.run(_assert_context_execution("scalar"))
    asyncio.run(_assert_context_execution("collection"))
    asyncio.run(_assert_context_execution("nested"))


def test_optional_context_omission_closes_unstarted_without_an_attempt() -> None:
    asyncio.run(_assert_context_execution("omitted"))


async def _assert_context_execution(mode: str) -> None:
    item_type = ArtifactType(name="text", revision=1)
    input_type = ArtifactType(name="text_collection", revision=1) if mode in {"collection", "nested"} else item_type
    workflow, initial_node, implementation_node = (
        _nested_workflow(input_type) if mode == "nested" else _single_workflow(input_type)
    )
    data = _data(1)
    target = next(iter(data.targets))
    request_policy = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="executor",
        replay="idempotent",
        max_attempts=1,
    )
    source_capability = ContextSourceCapability(
        source=SOURCE,
        artifact_type=item_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=request_policy,
        safe_detachment="forbidden",
    )
    provider = _ContextProvider(items=("ab", "cd") if mode in {"collection", "nested"} else ("ab",), omit=mode == "omitted")
    declaration = InitialContextDecl(
        target=target,
        node=initial_node,
        port="context",
        artifact_type=input_type,
        source=SOURCE,
        selector=ContextSelector(fields=()),
        requirement="optional" if mode == "omitted" else "required",
        bounds=RetrievalBounds(
            max_items=2 if mode in {"collection", "nested"} else 1,
            max_bytes=8,
            max_requests=1,
        ),
        materialization=ContextMaterialization(
            kind="collection" if mode in {"collection", "nested"} else "single",
            item_type=item_type,
        ),
    )
    binding = await (
        await start_initial_binding(
            data=data,
            workflow=workflow,
            declarations=(declaration,),
            capabilities=(source_capability,),
            resources=(
                ContextResource(
                    source=SOURCE,
                    capability=source_capability,
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
                max_items=2,
                max_bytes=8,
                max_requests=1,
                max_resources=1,
            ),
        )
    ).wait()
    assert binding.context is not None
    capability = _capability(workflow)
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=8, max_entries=2, max_parent_depth=2),
        bound_inputs=(),
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=(
            ImplementationSelection(
                node=implementation_node,
                implementation=capability.implementation,
                configuration=capability.configuration,
            ),
        ),
        capabilities=(capability,),
        limits=_limits(slots=2),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=binding.context,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=None,
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=implementation_node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(implementation,),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_runtime_rows(),
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
            max_port_facts=6,
            max_provenance_edges=6,
        ),
    )
    callback = _IdentityCallback()
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
                        local=callback,
                        transport=None,
                        resource=None,
                    ),
                ),
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=6,
                    max_runtime_artifact_bytes=32,
                    max_collection_items=2,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    assert not any(type(key).__name__ == "RootInputKey" for fact in result.provenance for key in (fact.key,))
    if mode == "omitted":
        assert callback.calls == 0
        assert result.record.terminals[0].category == "blocked"
        assert result.record.terminals[0].attempt is None
        assert not result.final_outputs
        return
    assert callback.calls == 1
    expected_value = ("ab", "cd") if mode in {"collection", "nested"} else "ab"
    actual_value = callback.observed[0]
    if isinstance(actual_value, TextCollectionValue):
        assert tuple(item.value.text for item in actual_value.items) == expected_value
    else:
        assert isinstance(actual_value, TextArtifactValue)
        assert actual_value.text == expected_value
    assert result.final_outputs, (
        tuple((item.category, item.reasons) for item in result.record.terminals),
        tuple((item.port, item.role) for item in result.ports),
        tuple(type(item.key).__name__ for item in result.provenance),
    )
    final = result.final_outputs[0]
    source_artifact = next(
        fact.artifact
        for fact in result.provenance
        if type(fact.key).__name__ in {"BoundInputKey", "InitialCollectionKey"}
        and (mode == "scalar" or type(fact.key).__name__ == "InitialCollectionKey")
    )
    assert final.candidate.artifact == source_artifact
    if mode == "nested":
        assert sum(type(fact.key).__name__ == "InitialCollectionKey" for fact in result.provenance) == 1
