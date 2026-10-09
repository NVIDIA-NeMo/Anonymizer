# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Admission and execution helpers for map and loop conformance."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.binding import start_initial_binding
from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
from anonymizer.engine.graph_sdk.context import (
    BindingLimits,
    ContextMaterialization,
    ContextResource,
    ContextSelector,
    ContextSourceCapability,
    InitialContextDecl,
    RetrievalBounds,
    admit_context_plan,
)
from anonymizer.engine.graph_sdk.executor import (
    AssessmentFinding,
    AssessmentLimits,
    DecisionLimits,
    EvidenceProductionDecl,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    MapExpansionDecl,
    MapItemKey,
    OperationExecutionPolicy,
    OperationOutputKey,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.requests import (
    PhysicalRequestPolicy,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    InputPort,
    KeyedJoinDecl,
    LoopCarriedBinding,
    LoopDecl,
    LoopInitialBinding,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutputDependency,
    OutputPort,
    SequenceEdge,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.context_source_fixtures import SOURCE, _ContextProvider
from tests.graph_sdk.effects_map_fixtures import MAP_CASES, _LoopCallback, _map_fixture, _MapCallback, _MapFixture
from tests.graph_sdk.test_dynamic_executor import _capability, _operation, _outcome, _rows
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _data, _limits


def _admit_fixture(
    fixture: _MapFixture,
    *,
    map_expansions: tuple[MapExpansionDecl, ...] | None = None,
    assessment_productions: tuple[EvidenceProductionDecl, ...] | None = None,
    data: Any | None = None,
    bound_context: Any | None = None,
    baseline_port_facts: int = 1,
    baseline_provenance_edges: int = 0,
    port_fact_headroom: int = 8,
    provenance_edge_headroom: int = 8,
):
    by_node = dict(zip(fixture.implementation_nodes, fixture.capabilities, strict=True))
    data = _data(1) if data is None else data
    prepared = prepare(
        data=data,
        workflow=fixture.workflow,
        activation_limits=ActivationLimits(
            max_events=max(20, 4 * fixture.workflow.limits.max_activation_occurrences),
            max_entries=fixture.workflow.limits.max_activation_occurrences,
            max_parent_depth=fixture.workflow.limits.max_dynamic_depth + 1,
        ),
        bound_inputs=tuple(
            BoundInput(target=target, source=target, port="default", artifact_type=fixture.text_type)
            for target in data.targets
        ),
        configuration=PreparationConfiguration(
            purpose="protection" if fixture.member_assessment else "execution_only",
            required_protection_outcomes=frozenset({"ok"}) if fixture.member_assessment else frozenset(),
            hard_request_limit=None,
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=tuple(
            ImplementationSelection(
                node=node,
                implementation=capability.implementation,
                configuration=capability.configuration,
            )
            for node, capability in by_node.items()
        ),
        capabilities=fixture.capabilities,
        limits=_limits(
            capabilities=len(fixture.capabilities),
            slots=fixture.workflow.limits.max_activation_occurrences * len(data.targets),
        ),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=bound_context,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    policies = tuple(
        OperationExecutionPolicy(
            node=node,
            kind="local",
            request=None,
            safe_detachment="forbidden",
            implementations=(
                ExecutionImplementation(
                    implementation=capability.implementation,
                    configuration=capability.configuration,
                    capability=capability,
                    request=None,
                ),
            ),
            result_outcomes=frozenset(outcome.name for outcome in capability.operation.outcomes),
            runtime_outcomes=_rows(frozenset(outcome.name for outcome in capability.operation.outcomes)),
        )
        for node, capability in by_node.items()
    )
    return admit_execution_plan(
        context=context,
        capabilities=fixture.capabilities,
        policies=policies,
        decisions=(),
        assessment_productions=assessment_productions
        if assessment_productions is not None
        else (
            EvidenceProductionDecl(
                node=fixture.expander,
                outcome="expand",
                promise="assessment0",
                evidence_port="members",
                absence_queries=frozenset(),
                supported_findings=frozenset({AssessmentFinding(status="satisfied", code="assessment0")}),
            ),
            *(
                (
                    EvidenceProductionDecl(
                        node=fixture.member_implementation,
                        outcome="ok",
                        promise="member_checked",
                        evidence_port="evidence",
                        absence_queries=frozenset(),
                        supported_findings=frozenset({AssessmentFinding(status="satisfied", code="member_checked")}),
                    ),
                )
                if fixture.member_assessment
                else ()
            ),
        ),
        assessment_limits=AssessmentLimits(
            max_productions=len(assessment_productions)
            if assessment_productions is not None
            else 1 + fixture.member_assessment,
            max_findings_per_production=1,
            max_finding_code_bytes=16,
            max_absence_queries=0,
            max_assessment_facts=len(data.targets)
            * (fixture.workflow.limits.max_activation_occurrences if fixture.member_assessment else 1),
            max_port_facts=baseline_port_facts + port_fact_headroom,
            max_provenance_edges=baseline_provenance_edges + provenance_edge_headroom,
        ),
        map_expansions=(
            MapExpansionDecl(
                expander=fixture.expander,
                outcome="expand",
                membership_port="members",
                item_type=fixture.text_type,
            ),
        )
        if map_expansions is None
        else map_expansions,
    )


async def _execute_membership(
    item_count: int,
    *,
    response_mode: str = "valid",
    item_values: tuple[str, ...] | None = None,
    artifact_headroom: int = 8,
    artifact_byte_headroom: int = 32,
    max_collection_items: int = 4,
    control_only: bool = False,
    other_count: int = 0,
    cancel_before_result: bool = False,
    provenance_edge_headroom: int = 8,
    default_override: bool = False,
    member_subgraph: bool = False,
    target_count: int = 1,
    context_membership_schema: bool = False,
    context_schema_item_mismatch: bool = False,
    max_children: int = 2,
    outward_scalar: str | None = None,
    outward_identity: bool = True,
    same_key_versions: bool = False,
    item_counts: tuple[int, ...] | None = None,
    member_assessment: bool = False,
    fixture: _MapFixture | None = None,
    port_fact_headroom: int = 8,
    assessment_productions: tuple[EvidenceProductionDecl, ...] | None = None,
    map_expansions: tuple[MapExpansionDecl, ...] | None = None,
):
    fixture = fixture or _map_fixture(
        control_only=control_only,
        other_count=other_count,
        default_override=default_override,
        member_subgraph=member_subgraph,
        context_membership_schema=context_membership_schema,
        max_children=max_children,
        outward_scalar=outward_scalar,
        outward_identity=outward_identity,
        member_assessment=member_assessment,
    )
    data = _data(target_count)
    bound_context = None
    if other_count:
        target = next(iter(data.targets))
        policy = PhysicalRequestPolicy(
            visibility="dispatch_and_settlement",
            pre_dispatch_control="executor",
            retry_owner="executor",
            replay="idempotent",
            max_attempts=1,
        )
        context_item_type = fixture.other_type if context_schema_item_mismatch else fixture.text_type
        capability = ContextSourceCapability(
            source=SOURCE,
            artifact_type=context_item_type,
            uses=frozenset({"initial_binding"}),
            execution="async",
            resource_owner="caller",
            cancellation="cooperative_ack",
            settlement="explicit_ack",
            usage="exact",
            request=policy,
            safe_detachment="forbidden",
        )
        provider = _ContextProvider(items=("schema",))
        binding = await (
            await start_initial_binding(
                data=data,
                workflow=fixture.workflow,
                declarations=(
                    InitialContextDecl(
                        target=target,
                        node=fixture.expander,
                        port="schema",
                        artifact_type=fixture.collection_type if context_membership_schema else fixture.other_type,
                        source=SOURCE,
                        selector=ContextSelector(fields=()),
                        requirement="required",
                        bounds=RetrievalBounds(max_items=1, max_bytes=6, max_requests=1),
                        materialization=ContextMaterialization(kind="collection", item_type=context_item_type),
                    ),
                ),
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
                    max_bytes=6,
                    max_requests=1,
                    max_resources=1,
                ),
            )
        ).wait()
        assert binding.context is not None
        bound_context = binding.context
    baseline_port_facts = target_count * (1 + (1 if bound_context is not None else 0))
    baseline_provenance_edges = target_count * ((1 if bound_context is not None else 0) + 1 + other_count)
    admitted = _admit_fixture(
        fixture,
        assessment_productions=assessment_productions,
        map_expansions=map_expansions,
        data=data,
        bound_context=bound_context,
        baseline_port_facts=baseline_port_facts,
        baseline_provenance_edges=baseline_provenance_edges,
        port_fact_headroom=port_fact_headroom * target_count,
        provenance_edge_headroom=provenance_edge_headroom * target_count,
    )
    context_item_bytes = (
        sum(len(artifact.text.encode()) for artifact in bound_context.artifacts) if bound_context is not None else 0
    )
    baseline_artifacts = len(data.targets) + (len(bound_context.artifacts) + 1 if bound_context is not None else 0)
    baseline_bytes = (
        sum(len(datum.text.encode()) for datum in data.datums if datum.id in data.targets)
        + context_item_bytes
        + (context_item_bytes if bound_context is not None else 0)
    )
    by_node = dict(zip(fixture.implementation_nodes, fixture.capabilities, strict=True))
    started = asyncio.Event() if cancel_before_result else None
    release = asyncio.Event() if cancel_before_result else None
    callbacks = {
        node: _MapCallback(
            mode=(
                "expander"
                if node in {item.expander for item in admitted.map_expansions}
                else "default_source"
                if node == fixture.default_source
                else "operation"
            ),
            response_mode=response_mode,
            item_count=item_count,
            item_values=item_values,
            same_key_versions=same_key_versions,
            item_counts=item_counts,
            collection_type=fixture.collection_type,
            text_type=fixture.text_type,
            other_type=fixture.other_type,
            other_count=other_count,
            started=started if node == fixture.expander else None,
            release=release if node == fixture.expander else None,
            cross_ready=asyncio.Event() if node == fixture.expander and response_mode == "cross_association" else None,
            passthrough=node == fixture.member_implementation and outward_scalar is not None,
            member_assessment=any(
                item.node == node and item.evidence_port == "evidence" for item in admitted.assessment_productions
            ),
            assessment_promise=next(
                (item.promise for item in admitted.assessment_productions if item.node == node), None
            ),
            finalize_collection=any(port.name == "members" for port in by_node[node].operation.inputs)
            and any(port.name == "value" for port in by_node[node].operation.outputs),
        )
        for node in by_node
    }
    handles = tuple(
        ImplementationHandle(
            implementation=capability.implementation,
            operation=capability.operation,
            configuration=capability.configuration,
            local=callbacks[node],
            transport=None,
            resource=None,
        )
        for node, capability in by_node.items()
    )
    running = await start_execution(
        admitted=admitted,
        capabilities=fixture.capabilities,
        services=ExecutionServices(
            handles=handles,
            context_resources=(),
            limits=ExecutionLimits(
                max_local_in_flight=4,
                max_remote_outstanding=0,
                max_runtime_artifacts=baseline_artifacts + artifact_headroom,
                max_runtime_artifact_bytes=baseline_bytes + artifact_byte_headroom,
                max_collection_items=max_collection_items,
            ),
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_Clock(),
        ),
    )
    if cancel_before_result:
        assert started is not None and release is not None
        wait = asyncio.create_task(running.wait())
        await started.wait()
        running.request_cancel()
        release.set()
        result = await wait
    else:
        result = await running.wait()
    return fixture, result, callbacks


async def _execute_loop_resolution(mode: str):
    loop_bound = 1 if mode == "prior_continue" else 2
    owner = WorkflowId.new()
    starter, member, ordinary, join = (NodeId.new(workflow=owner) for _ in range(4))
    text_type = ArtifactType(name="text", revision=1)
    starter_operation = OperationSpec(
        name="starter",
        inputs=(InputPort(name="initial", artifact_type=text_type),),
        outputs=(OutputPort(name="value", artifact_type=text_type),),
        output_dependencies=(
            OutputDependency(output="value", inputs=frozenset({"initial"}), identity_input="initial"),
        ),
        outcomes=(
            _outcome("enter", produced=frozenset({"value"}), max_output_bytes=32),
            _outcome("bypass", produced=frozenset({"value"}), max_output_bytes=32),
        ),
    )
    member_operation = OperationSpec(
        name="member",
        inputs=(InputPort(name="previous", artifact_type=text_type),),
        outputs=(OutputPort(name="value", artifact_type=text_type),),
        output_dependencies=(OutputDependency(output="value", inputs=frozenset({"previous"}), identity_input=None),),
        outcomes=(
            _outcome("again", produced=frozenset({"value"}), max_output_bytes=32),
            _outcome("exit", produced=frozenset({"value"}), max_output_bytes=32),
            replace(
                _outcome("failure", produced=frozenset({"value"}), max_output_bytes=32),
                category="failure",
            ),
        ),
    )
    ordinary_operation = OperationSpec(
        name="ordinary",
        inputs=(InputPort(name="value", artifact_type=text_type),),
        outputs=(),
        output_dependencies=(),
        outcomes=(_outcome("ok"),),
    )
    join_operation = _operation("join", (_outcome("ok"),))
    interface = OperationSpec(
        name="loop-root",
        inputs=(InputPort(name="initial", artifact_type=text_type),),
        outputs=(),
        output_dependencies=(),
        outcomes=(_outcome("ok", max_activations=12, max_output_bytes=128),),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(
            OperationNode(id=starter, operation=starter_operation),
            OperationNode(id=member, operation=member_operation),
            OperationNode(id=ordinary, operation=ordinary_operation),
            OperationNode(id=join, operation=join_operation),
        ),
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="initial"),
                destination=NodeInputRef(node=starter, port="initial"),
            ),
            InputBinding(
                source=NodeOutputRef(node=starter, port="value"),
                destination=NodeInputRef(node=member, port="previous"),
            ),
            InputBinding(
                source=NodeOutputRef(node=member, port="value"),
                destination=NodeInputRef(node=ordinary, port="value"),
            ),
        ),
        output_bindings=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=join, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(
            SequenceEdge(before=starter, after=member),
            SequenceEdge(before=member, after=ordinary),
            SequenceEdge(before=ordinary, after=join),
        ),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=4,
            max_bindings=4,
            max_sequence_edges=3,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(
            DynamicScope(
                workflow=static,
                maps=(),
                joins=(
                    KeyedJoinDecl(
                        source=starter,
                        join=join,
                        accepted_categories=frozenset({"success"}),
                        reduction="all_by_key",
                    ),
                ),
                loops=(
                    LoopDecl(
                        starter=starter,
                        member=member,
                        join=join,
                        enter_outcomes=frozenset({"enter"}),
                        bypass_outcomes=frozenset({"bypass"}),
                        continue_outcomes=frozenset({"again"}),
                        exit_outcomes=frozenset({"exit", "failure"}),
                        initial=(
                            LoopInitialBinding(
                                source=NodeOutputRef(node=starter, port="value"),
                                destination=NodeInputRef(node=member, port="previous"),
                            ),
                        ),
                        carried=(
                            LoopCarriedBinding(
                                source=NodeOutputRef(node=member, port="value"),
                                destination=NodeInputRef(node=member, port="previous"),
                            ),
                        ),
                        max_iterations=loop_bound,
                    ),
                ),
            ),
        ),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=1,
            max_loops=1,
            max_children_per_map=0,
            max_iterations_per_loop=loop_bound,
            max_dynamic_depth=1,
            max_activation_occurrences=loop_bound + 3,
        ),
    )
    operations = {
        starter: starter_operation,
        member: member_operation,
        ordinary: ordinary_operation,
        join: join_operation,
    }
    capabilities = tuple(_capability(operation, index) for index, operation in enumerate(operations.values()))
    by_node = dict(zip(operations, capabilities, strict=True))
    data = _data(1)
    target = next(iter(data.targets))
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(
            max_events=24,
            max_entries=loop_bound + 3,
            max_parent_depth=2,
        ),
        bound_inputs=(BoundInput(target=target, source=target, port="initial", artifact_type=text_type),),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=tuple(
            ImplementationSelection(
                node=node,
                implementation=by_node[node].implementation,
                configuration=by_node[node].configuration,
            )
            for node in operations
        ),
        capabilities=capabilities,
        limits=_limits(capabilities=4, slots=loop_bound + 3),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
        adaptive_retrievals=(),
        context_capabilities=(),
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=capabilities,
        policies=tuple(
            OperationExecutionPolicy(
                node=node,
                kind="local",
                request=None,
                safe_detachment="forbidden",
                implementations=(
                    ExecutionImplementation(
                        implementation=by_node[node].implementation,
                        configuration=by_node[node].configuration,
                        capability=by_node[node],
                        request=None,
                    ),
                ),
                result_outcomes=frozenset(outcome.name for outcome in operation.outcomes),
                runtime_outcomes=tuple(
                    replace(row, category="failure")
                    if row.condition == "result" and row.reported_outcome == "failure"
                    else row
                    for row in _rows(frozenset(outcome.name for outcome in operation.outcomes))
                ),
            )
            for node, operation in operations.items()
        ),
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=12,
            max_provenance_edges=10,
        ),
    )
    callbacks = {
        node: _LoopCallback(
            role="starter" if node == starter else "member" if node == member else "ordinary",
            mode=mode,
            text_type=text_type,
        )
        for node in operations
    }
    handles = tuple(
        ImplementationHandle(
            implementation=by_node[node].implementation,
            operation=operation,
            configuration=by_node[node].configuration,
            local=callbacks[node],
            transport=None,
            resource=None,
        )
        for node, operation in operations.items()
    )
    result = await (
        await start_execution(
            admitted=admitted,
            capabilities=capabilities,
            services=ExecutionServices(
                handles=handles,
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=4,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=8,
                    max_runtime_artifact_bytes=128,
                    max_collection_items=1,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    return result, callbacks, starter, member, ordinary


async def _assert_loop_scalar_source_resolution(
    case_id: str,
    mode: str,
    expected_resolution: str,
) -> None:
    case = MAP_CASES[case_id]
    result, callbacks, starter, member, ordinary = await _execute_loop_resolution(mode)
    assert case["expected"]["state"]["resolution"] == expected_resolution
    entries = result.states[0].entries
    ordinary_entry = next(entry for entry in entries if entry.template == ordinary)
    starter_entry = next(entry for entry in entries if entry.template == starter)

    if mode == "exit":
        exited = next(
            entry
            for entry in entries
            if entry.template == member and entry.activation.iteration == 1 and entry.outcome == "exit"
        )
        assert ordinary_entry.status == "success"
        assert len(callbacks[ordinary].calls) == 1
        ordinary_input = callbacks[ordinary].calls[0][0].inputs[0]
        producer = next(
            fact
            for fact in result.provenance
            if isinstance(fact.key, OperationOutputKey)
            and fact.key.activation == exited.activation
            and fact.key.port == "value"
        )
        assert ordinary_input.artifact == producer.artifact
        assert ordinary_input.value == TextArtifactValue(text="target-0:0:1")
    else:
        assert ordinary_entry.status == "blocked"
        assert not callbacks[ordinary].calls

    if mode == "bypass":
        assert starter_entry.outcome == "bypass"
        assert not any(entry.template == member for entry in entries)
    elif mode == "prior_continue":
        iterations = [entry for entry in entries if entry.template == member]
        assert [(entry.activation.iteration, entry.outcome) for entry in iterations] == [(0, "again")]
    elif mode == "failure":
        iterations = [entry for entry in entries if entry.template == member]
        assert [(entry.activation.iteration, entry.status, entry.outcome) for entry in iterations] == [
            (0, "failure", "failure")
        ]
    elif mode == "overflow":
        iterations = sorted(
            (entry for entry in entries if entry.template == member),
            key=lambda entry: cast(int, entry.activation.iteration),
        )
        assert [(entry.activation.iteration, entry.outcome) for entry in iterations] == [
            (0, "again"),
            (1, "again"),
        ]


async def _assert_map_subgraph_item_binding() -> None:
    case = MAP_CASES["map/subgraph_item_binding"]
    fixture, result, callbacks = await _execute_membership(
        2,
        item_values=("left", "right"),
        member_subgraph=True,
    )
    expected = case["expected"]["state"]["publication"]
    map_items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    assert len(map_items) == len(expected["inputs"]) == 2
    by_member = {fact.key.member: fact for fact in map_items}
    calls = [call[0] for call in callbacks[fixture.member_implementation].calls]
    assert len(calls) == 2
    assert {
        cast(SemanticAssociation, call.association).task.activation.parent: (
            call.inputs[0].artifact,
            call.inputs[0].value,
        )
        for call in calls
    } == {
        member: (fact.artifact, TextArtifactValue(text=("left", "right")[fact.key.item_key]))
        for member, fact in by_member.items()
    }
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "closed"
    assert expansion.members == frozenset(by_member)


async def _assert_map_collection_schema_minimum_conflict() -> None:
    case = MAP_CASES["map/context_minimum_conflict"]
    with pytest.raises(EffectRejected) as error:
        await _execute_membership(
            1,
            other_count=1,
            context_membership_schema=True,
        )
    assert error.value.code == EffectCode.CONTRADICTORY
    assert error.value.code.value == case["expected"]["code"]


async def _assert_map_collection_item_schema_conflict() -> None:
    case = MAP_CASES["map/collection_item_schema_conflict"]
    with pytest.raises(EffectRejected) as error:
        await _execute_membership(
            1,
            other_count=1,
            context_membership_schema=True,
            context_schema_item_mismatch=True,
        )
    assert error.value.code == EffectCode.CONTRADICTORY
    assert error.value.code.value == case["expected"]["code"]
