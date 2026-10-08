# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate frozen map effects cases through the real graph executor."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
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
    BoundInputKey,
    DecisionLimits,
    EvidenceProductionDecl,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    InitialCollectionKey,
    LocalAssessmentResult,
    LocalCompleted,
    LocalFailure,
    MapExpansionDecl,
    MapItemKey,
    OperationExecutionPolicy,
    OperationOutputKey,
    RootInputKey,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PhysicalRequestPolicy,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph._values import ContractViolation, ValidationCode
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    ContextInputRef,
    ContextUse,
    DynamicLimits,
    DynamicScope,
    EvidencePromise,
    InputBinding,
    InputPort,
    KeyedJoinDecl,
    LoopCarriedBinding,
    LoopDecl,
    LoopInitialBinding,
    MapDecl,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutputBinding,
    OutputDependency,
    OutputPort,
    SequenceEdge,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_activation import _loop_workflow
from tests.graph_sdk.test_context_source_execution import SOURCE, _ContextProvider
from tests.graph_sdk.test_dynamic_executor import _capability, _operation, _outcome, _rows
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _data, _limits

CORPUS = Path(__file__).parent / "reference" / "effects_v1_cases.json"
MAP_CASES = {item["case_id"]: item for item in json.loads(CORPUS.read_bytes()) if item["family"] == "map"}


@dataclass(frozen=True)
class _MapFixture:
    workflow: Any
    expander: NodeId
    member: NodeId
    member_implementation: NodeId
    join: NodeId
    default_source: NodeId | None
    text_type: ArtifactType
    collection_type: ArtifactType
    other_type: ArtifactType
    implementation_nodes: tuple[NodeId, ...]
    capabilities: tuple[Any, ...]


def _map_fixture(
    *,
    max_children: int = 2,
    control_only: bool = False,
    other_count: int = 0,
    default_override: bool = False,
    outward_scalar: str | None = None,
    member_subgraph: bool = False,
    context_membership_schema: bool = False,
    outward_identity: bool = True,
    outward_value_depends_on_default: bool = True,
    context_item_override: bool = False,
) -> _MapFixture:
    owner = WorkflowId.new()
    expander, member, join = (NodeId.new(workflow=owner) for _ in range(3))
    default_source = NodeId.new(workflow=owner) if default_override else None
    ordinary = NodeId.new(workflow=owner) if outward_scalar == "ordinary" else None
    text_type = ArtifactType(name="text", revision=1)
    collection_type = ArtifactType(name="members_t", revision=1)
    other_type = ArtifactType(name="other_t", revision=1)
    context_type = collection_type if context_membership_schema else other_type
    other_ports = tuple(OutputPort(name=f"other{index}", artifact_type=other_type) for index in range(other_count))
    expander_operation = OperationSpec(
        name="expander",
        inputs=(
            InputPort(name="default", artifact_type=text_type),
            *((InputPort(name="schema", artifact_type=context_type),) if other_count else ()),
        ),
        outputs=(OutputPort(name="members", artifact_type=collection_type), *other_ports),
        output_dependencies=(
            OutputDependency(output="members", inputs=frozenset({"default"}), identity_input=None),
            *(
                OutputDependency(output=port.name, inputs=frozenset({"default"}), identity_input=None)
                for port in other_ports
            ),
        ),
        outcomes=(
            replace(
                _outcome(
                    "expand",
                    produced=frozenset({"members", *(port.name for port in other_ports)}),
                    max_activations=max_children + 2,
                    max_output_bytes=128,
                ),
                evidence=frozenset(
                    {
                        EvidencePromise(
                            name="assessment0",
                            meaning="map membership accepted",
                            subject_port="members",
                            consumed_ports=frozenset({"default"}),
                            coverage=frozenset(),
                        )
                    }
                ),
                context=(
                    frozenset({ContextUse(port="schema", meaning="retrieved", capture="whole_artifact")})
                    if other_count
                    else frozenset()
                ),
            ),
        ),
    )
    member_operation = OperationSpec(
        name="member",
        inputs=() if control_only else (InputPort(name="item", artifact_type=text_type),),
        outputs=(OutputPort(name="value", artifact_type=text_type),) if outward_scalar else (),
        output_dependencies=(
            OutputDependency(
                output="value",
                inputs=frozenset({"item"}),
                identity_input="item" if outward_identity else None,
            ),
        )
        if outward_scalar
        else (),
        outcomes=(
            replace(
                _outcome(
                    "ok",
                    produced=frozenset({"value"}) if outward_scalar else frozenset(),
                    max_output_bytes=32 if outward_scalar else 0,
                ),
                context=(
                    frozenset({ContextUse(port="item", meaning="retrieved", capture="whole_artifact")})
                    if context_item_override
                    else frozenset()
                ),
            ),
        ),
    )
    member_node: OperationNode | SubgraphNode = OperationNode(id=member, operation=member_operation)
    member_implementation = member
    if member_subgraph:
        body_owner = WorkflowId.new()
        body_child = NodeId.new(workflow=body_owner)
        member_implementation = body_child
        body = admit_static_workflow(
            workflow=body_owner,
            interface=member_operation,
            nodes=(OperationNode(id=body_child, operation=member_operation),),
            input_bindings=(
                InputBinding(
                    source=WorkflowInputRef(port="item"),
                    destination=NodeInputRef(node=body_child, port="item"),
                ),
            ),
            output_bindings=(),
            outcome_bindings=(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=body_child, outcome="ok"),
                    destination=WorkflowOutcomeRef(outcome="ok"),
                ),
            ),
            sequence=(),
            choices=(),
            protection=(),
            limits=WorkflowLimits(
                max_nodes=1,
                max_bindings=2,
                max_sequence_edges=0,
                max_choices=0,
                max_branch_members=0,
                max_subgraph_depth=1,
                max_choice_states=1,
            ),
        )
        member_node = SubgraphNode(id=member, operation=body.interface, body=body)
    join_operation = OperationSpec(
        name="join",
        inputs=(InputPort(name="value", artifact_type=text_type),) if outward_scalar == "join" else (),
        outputs=(),
        output_dependencies=(),
        outcomes=(_outcome("ok"),),
    )
    ordinary_operation = OperationSpec(
        name="ordinary",
        inputs=(InputPort(name="value", artifact_type=text_type),),
        outputs=(),
        output_dependencies=(),
        outcomes=(_outcome("ok"),),
    )
    default_operation = OperationSpec(
        name="default-source",
        inputs=(),
        outputs=(OutputPort(name="default", artifact_type=text_type),),
        output_dependencies=(OutputDependency(output="default", inputs=frozenset(), identity_input=None),),
        outcomes=(_outcome("available", produced=frozenset({"default"})),),
    )
    interface = OperationSpec(
        name="map-root",
        inputs=(
            InputPort(name="default", artifact_type=text_type),
            *((InputPort(name="member_context", artifact_type=text_type),) if context_item_override else ()),
            *((InputPort(name="schema", artifact_type=context_type),) if other_count else ()),
        ),
        outputs=(
            OutputPort(name="members", artifact_type=collection_type),
            *((OutputPort(name="value", artifact_type=text_type),) if outward_scalar == "workflow_output" else ()),
        ),
        output_dependencies=(
            OutputDependency(output="members", inputs=frozenset({"default"}), identity_input=None),
            *(
                (
                    OutputDependency(
                        output="value",
                        inputs=frozenset({"default"}) if outward_value_depends_on_default else frozenset(),
                        identity_input="default" if outward_identity else None,
                    ),
                )
                if outward_scalar == "workflow_output"
                else ()
            ),
        ),
        outcomes=(
            replace(
                _outcome(
                    "ok",
                    produced=frozenset({"members", *(("value",) if outward_scalar == "workflow_output" else ())}),
                    max_activations=max_children + (8 if default_source is not None or outward_scalar else 4),
                    max_output_bytes=512 if default_source is not None or outward_scalar else 128,
                ),
                evidence=frozenset(
                    {
                        EvidencePromise(
                            name="assessment0",
                            meaning="map membership accepted",
                            subject_port="members",
                            consumed_ports=frozenset({"default"}),
                            coverage=frozenset(),
                        )
                    }
                ),
                context=(
                    frozenset(
                        {
                            *(
                                (ContextUse(port="schema", meaning="retrieved", capture="whole_artifact"),)
                                if other_count
                                else ()
                            ),
                            *(
                                (ContextUse(port="member_context", meaning="retrieved", capture="whole_artifact"),)
                                if context_item_override
                                else ()
                            ),
                        }
                    )
                ),
            ),
        ),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(
            OperationNode(id=expander, operation=expander_operation),
            member_node,
            OperationNode(id=join, operation=join_operation),
            *((OperationNode(id=default_source, operation=default_operation),) if default_source is not None else ()),
            *((OperationNode(id=ordinary, operation=ordinary_operation),) if ordinary is not None else ()),
        ),
        input_bindings=(
            *(
                ()
                if control_only
                else (
                    InputBinding(
                        source=(
                            ContextInputRef(port="member_context")
                            if context_item_override
                            else NodeOutputRef(node=default_source, port="default")
                            if default_source is not None
                            else WorkflowInputRef(port="default")
                        ),
                        destination=NodeInputRef(node=member, port="item"),
                    ),
                )
            ),
            *(
                (
                    InputBinding(
                        source=NodeOutputRef(node=member, port="value"),
                        destination=NodeInputRef(
                            node=join if outward_scalar == "join" else cast(NodeId, ordinary),
                            port="value",
                        ),
                    ),
                )
                if outward_scalar in {"join", "ordinary"}
                else ()
            ),
            InputBinding(
                source=WorkflowInputRef(port="default"),
                destination=NodeInputRef(node=expander, port="default"),
            ),
            *(
                (
                    InputBinding(
                        source=ContextInputRef(port="schema"),
                        destination=NodeInputRef(node=expander, port="schema"),
                    ),
                )
                if other_count
                else ()
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=expander, port="members"),
                destination=WorkflowOutputRef(port="members"),
            ),
            *(
                (
                    OutputBinding(
                        source=NodeOutputRef(node=member, port="value"),
                        destination=WorkflowOutputRef(port="value"),
                    ),
                )
                if outward_scalar == "workflow_output"
                else ()
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=join, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(
            SequenceEdge(before=expander, after=member),
            SequenceEdge(before=member, after=join),
            *((SequenceEdge(before=default_source, after=join),) if default_source is not None else ()),
            *(
                (SequenceEdge(before=member, after=ordinary), SequenceEdge(before=ordinary, after=join))
                if ordinary is not None
                else ()
            ),
        ),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=3 + int(default_source is not None) + int(ordinary is not None) + int(member_subgraph),
            max_bindings=(5 if other_count else 4) + int(outward_scalar is not None),
            max_sequence_edges=2 + int(default_source is not None) + (2 if ordinary is not None else 0),
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2 if member_subgraph else 1,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(
            DynamicScope(
                workflow=static,
                maps=(
                    MapDecl(
                        expander=expander,
                        member=member,
                        expansion_outcomes=frozenset({"expand"}),
                        max_children=max_children,
                        item_input=None if control_only else "item",
                    ),
                ),
                joins=(
                    KeyedJoinDecl(
                        source=expander,
                        join=join,
                        accepted_categories=frozenset({"success"}),
                        reduction="all_by_key",
                    ),
                ),
                loops=(),
            ),
            *(
                (DynamicScope(workflow=member_node.body, maps=(), joins=(), loops=()),)
                if isinstance(member_node, SubgraphNode)
                else ()
            ),
        ),
        limits=DynamicLimits(
            max_maps=1,
            max_joins=1,
            max_loops=0,
            max_children_per_map=max_children,
            max_iterations_per_loop=0,
            max_dynamic_depth=2 if member_subgraph else 1,
            max_activation_occurrences=(2 * max_children if member_subgraph else max_children)
            + 2
            + int(default_source is not None)
            + int(ordinary is not None),
        ),
    )
    operations = (
        expander_operation,
        member_operation,
        join_operation,
        *((default_operation,) if default_source is not None else ()),
        *((ordinary_operation,) if ordinary is not None else ()),
    )
    implementation_nodes = (
        expander,
        member_implementation,
        join,
        *((default_source,) if default_source is not None else ()),
        *((ordinary,) if ordinary is not None else ()),
    )
    capabilities = tuple(_capability(operation, index) for index, operation in enumerate(operations))
    return _MapFixture(
        workflow,
        expander,
        member,
        member_implementation,
        join,
        default_source,
        text_type,
        collection_type,
        other_type,
        implementation_nodes,
        capabilities,
    )


@dataclass
class _MapCallback:
    mode: str
    response_mode: str
    item_count: int
    item_values: tuple[str, ...] | None
    collection_type: ArtifactType
    text_type: ArtifactType
    other_type: ArtifactType
    other_count: int
    started: asyncio.Event | None = None
    release: asyncio.Event | None = None
    calls: list[tuple[AssociationInput, ...]] = field(default_factory=list)
    cross_associations: list[SemanticAssociation] = field(default_factory=list)
    cross_ready: asyncio.Event | None = None
    passthrough: bool = False

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted | LocalFailure:
        self.calls.append(request)
        association = request[0].association
        assert isinstance(association, SemanticAssociation)
        if self.mode == "expander" and self.response_mode == "cross_association":
            assert self.cross_ready is not None
            self.cross_associations.append(association)
            if len(self.cross_associations) == 2:
                self.cross_ready.set()
            await self.cross_ready.wait()
            association = next(item for item in self.cross_associations if item != association)
        if self.mode == "expander" and self.started is not None and self.release is not None:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
        outputs: tuple[PortArtifact, ...] = ()
        outcome = "ok"
        if self.mode == "default_source":
            return LocalFailure(failure="permanent")
        if self.mode == "expander":
            outcome = "expand"
            membership = PortArtifact(
                port="members",
                artifact_type=self.text_type if self.response_mode == "wrong_type" else self.collection_type,
                artifact=None,
                value=TextCollectionValue(
                    items=tuple(
                        TextCollectionItem(
                            key=index,
                            version=1,
                            value=TextArtifactValue(
                                text=(self.item_values[index] if self.item_values is not None else f"item-{index}")
                            ),
                        )
                        for index in range(self.item_count)
                    )
                ),
            )
            other_outputs = tuple(
                PortArtifact(
                    port=f"other{index}",
                    artifact_type=self.other_type,
                    artifact=None,
                    value=TextCollectionValue(items=()),
                )
                for index in range(self.other_count)
            )
            if self.response_mode == "missing":
                outputs = other_outputs
            elif self.response_mode == "duplicate":
                outputs = (membership, membership, *other_outputs)
            else:
                outputs = (membership, *other_outputs)
        elif self.passthrough:
            value = request[0].inputs[0]
            outputs = (
                PortArtifact(
                    port="value",
                    artifact_type=self.text_type,
                    artifact=None,
                    value=value.value,
                ),
            )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=association,
                    outcome=outcome,
                    outputs=outputs,
                    consumed_context_ports=frozenset({"schema"}) if self.other_count else frozenset(),
                ),
            ),
            assessments=(
                LocalAssessmentResult(
                    association=association,
                    promise="assessment0",
                    evidence_port="members",
                    finding=AssessmentFinding(status="satisfied", code="assessment0"),
                ),
            )
            if self.mode == "expander"
            else (),
        )


@dataclass
class _LoopCallback:
    role: str
    mode: str
    text_type: ArtifactType
    calls: list[tuple[AssociationInput, ...]] = field(default_factory=list)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls.append(request)
        association = request[0].association
        assert isinstance(association, SemanticAssociation)
        outputs: tuple[PortArtifact, ...] = ()
        outcome = "ok"
        if self.role == "starter":
            outcome = "bypass" if self.mode == "bypass" else "enter"
            value = request[0].inputs[0]
            outputs = (PortArtifact(port="value", artifact_type=self.text_type, artifact=None, value=value.value),)
        elif self.role == "member":
            iteration = association.task.activation.iteration
            assert iteration is not None
            outcome = (
                "failure" if self.mode == "failure" else "exit" if self.mode == "exit" and iteration == 1 else "again"
            )
            value = request[0].inputs[0]
            assert isinstance(value.value, TextArtifactValue)
            outputs = (
                PortArtifact(
                    port="value",
                    artifact_type=self.text_type,
                    artifact=None,
                    value=TextArtifactValue(text=f"{value.value.text}:{iteration}"),
                ),
            )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=association,
                    outcome=outcome,
                    outputs=outputs,
                    consumed_context_ports=frozenset(),
                ),
            ),
        )


def _admit_fixture(
    fixture: _MapFixture,
    *,
    map_expansions: tuple[MapExpansionDecl, ...] | None = None,
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
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
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
        assessment_productions=(
            EvidenceProductionDecl(
                node=fixture.expander,
                outcome="expand",
                promise="assessment0",
                evidence_port="members",
                absence_queries=frozenset(),
                supported_findings=frozenset({AssessmentFinding(status="satisfied", code="assessment0")}),
            ),
        ),
        assessment_limits=AssessmentLimits(
            max_productions=1,
            max_findings_per_production=1,
            max_finding_code_bytes=16,
            max_absence_queries=0,
            max_assessment_facts=len(data.targets),
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
):
    fixture = _map_fixture(
        control_only=control_only,
        other_count=other_count,
        default_override=default_override,
        member_subgraph=member_subgraph,
        context_membership_schema=context_membership_schema,
        max_children=max_children,
        outward_scalar=outward_scalar,
        outward_identity=outward_identity,
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
        data=data,
        bound_context=bound_context,
        baseline_port_facts=baseline_port_facts,
        baseline_provenance_edges=baseline_provenance_edges,
        port_fact_headroom=8 * target_count,
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
                if node == fixture.expander
                else "default_source"
                if node == fixture.default_source
                else "operation"
            ),
            response_mode=response_mode,
            item_count=item_count,
            item_values=item_values,
            collection_type=fixture.collection_type,
            text_type=fixture.text_type,
            other_type=fixture.other_type,
            other_count=other_count,
            started=started if node == fixture.expander else None,
            release=release if node == fixture.expander else None,
            cross_ready=asyncio.Event() if node == fixture.expander and response_mode == "cross_association" else None,
            passthrough=node == fixture.member_implementation and outward_scalar is not None,
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


@pytest.mark.parametrize(
    ("case_id", "mode", "expected_resolution"),
    (
        ("map/loop_exit", "exit", "member:1:exit"),
        ("map/loop_bypass", "bypass", "blocked:bypass"),
        ("map/loop_prior_continue", "prior_continue", "blocked:no_exit"),
        ("map/loop_failure", "failure", "blocked:no_exit"),
        ("map/loop_overflow", "overflow", "blocked:no_exit"),
    ),
)
def test_loop_scalar_source_resolution_matches_frozen_case(
    case_id: str,
    mode: str,
    expected_resolution: str,
) -> None:
    asyncio.run(_assert_loop_scalar_source_resolution(case_id, mode, expected_resolution))


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


def test_map_operation_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_operation"]
    fixture = _map_fixture()
    assert case["expected"] == {"status": "accepted"}
    assert fixture.workflow.scopes[0].maps[0].member == fixture.member


def test_map_subgraph_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_subgraph"]
    fixture = _map_fixture(member_subgraph=True)
    member = next(node for node in fixture.workflow.workflow.nodes if node.id == fixture.member)
    assert isinstance(member, SubgraphNode)
    assert case["expected"] == {"status": "accepted"}


def test_map_items_are_projected_into_real_subgraph_members() -> None:
    asyncio.run(_assert_map_subgraph_item_binding())


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


def test_control_only_map_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_control_only"]
    fixture = _map_fixture(control_only=True)
    assert case["expected"] == {"status": "accepted"}
    assert fixture.workflow.scopes[0].maps[0].item_input is None
    _admit_fixture(fixture)


def test_map_dynamic_item_dependency_overrides_static_default_summary() -> None:
    case = MAP_CASES["map/default_override_dependency"]
    fixture = _map_fixture(default_override=True)
    _admit_fixture(fixture)
    assert case["expected"] == {"status": "accepted"}


def test_boolean_collection_limit_rejects_at_typed_constructor() -> None:
    case = MAP_CASES["map/collection_items_invalid_limit"]
    with pytest.raises(EffectRejected) as error:
        ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=0,
            max_runtime_artifacts=1,
            max_runtime_artifact_bytes=1,
            max_collection_items=True,
        )
    assert error.value.code == EffectCode.INVALID_TYPE
    assert error.value.code.value == case["expected"]["code"]


def test_map_collection_schema_minimum_conflict_rejects_execution_admission() -> None:
    asyncio.run(_assert_map_collection_schema_minimum_conflict())


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


def test_map_collection_item_schema_conflict_rejects_execution_admission() -> None:
    asyncio.run(_assert_map_collection_item_schema_conflict())


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


def test_false_map_item_identity_summary_rejects_execution_admission() -> None:
    case = MAP_CASES["map/false_identity_summary"]
    false_identity = _map_fixture(max_children=1, outward_scalar="workflow_output")
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(false_identity)
    assert error.value.code == EffectCode.CONTRADICTORY
    assert error.value.code.value == case["expected"]["code"]

    nonidentity = _map_fixture(
        max_children=1,
        outward_scalar="workflow_output",
        outward_identity=False,
    )
    _admit_fixture(nonidentity)


def test_map_item_dependency_summary_mismatch_rejects_execution_admission() -> None:
    case = MAP_CASES["map/dependency_summary_mismatch"]
    fixture = _map_fixture(
        max_children=1,
        default_override=True,
        outward_scalar="workflow_output",
        outward_identity=False,
        outward_value_depends_on_default=False,
    )
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(fixture)
    assert error.value.code == EffectCode.CONTRADICTORY
    assert error.value.code.value == case["expected"]["code"]


def test_bound_context_cannot_override_a_dynamic_map_item() -> None:
    asyncio.run(_assert_bound_context_map_item_conflict())


async def _assert_bound_context_map_item_conflict() -> None:
    case = MAP_CASES["map/context_override_conflict"]
    fixture = _map_fixture(context_item_override=True)
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
        artifact_type=fixture.text_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=policy,
        safe_detachment="forbidden",
    )
    provider = _ContextProvider(items=("context",))
    binding = await (
        await start_initial_binding(
            data=data,
            workflow=fixture.workflow,
            declarations=(
                InitialContextDecl(
                    target=target,
                    node=fixture.member,
                    port="item",
                    artifact_type=fixture.text_type,
                    source=SOURCE,
                    selector=ContextSelector(fields=()),
                    requirement="required",
                    bounds=RetrievalBounds(max_items=1, max_bytes=7, max_requests=1),
                    materialization=ContextMaterialization(kind="single", item_type=fixture.text_type),
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
                max_bytes=7,
                max_requests=1,
                max_resources=1,
            ),
        )
    ).wait()
    assert binding.context is not None
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(fixture, data=data, bound_context=binding.context)
    assert error.value.code == EffectCode.CONTRADICTORY
    assert error.value.code.value == case["expected"]["code"]


@pytest.mark.parametrize(
    ("case_id", "max_children", "destination", "accepted"),
    (
        ("map/max_zero_scalar_join", 0, "join", True),
        ("map/max_one_scalar_ordinary", 1, "ordinary", True),
        ("map/max_one_scalar_workflow", 1, "workflow_output", True),
        ("map/max_two_scalar_join", 2, "join", False),
        ("map/max_two_scalar_ordinary", 2, "ordinary", False),
        ("map/max_two_scalar_workflow_output", 2, "workflow_output", False),
    ),
)
def test_scalar_map_cardinality_admission_matches_frozen_case(
    case_id: str,
    max_children: int,
    destination: str,
    accepted: bool,
) -> None:
    case = MAP_CASES[case_id]
    if accepted:
        _map_fixture(max_children=max_children, outward_scalar=destination)
        assert case["expected"] == {"status": "accepted"}
        return

    with pytest.raises(ContractViolation) as error:
        _map_fixture(max_children=max_children, outward_scalar=destination)
    assert error.value.code is ValidationCode.UNSUPPORTED
    assert error.value.code.value == case["expected"]["code"]


@pytest.mark.parametrize(
    ("case_id", "declarations", "code"),
    (
        ("map/missing_expansion", (), EffectCode.MISSING),
        ("map/item_type_mismatch", "wrong_item", EffectCode.CONTRADICTORY),
    ),
)
def test_map_expansion_admission_rejections(
    case_id: str,
    declarations: tuple[MapExpansionDecl, ...] | str,
    code: EffectCode,
) -> None:
    case = MAP_CASES[case_id]
    fixture = _map_fixture()
    if declarations == "wrong_item":
        declarations = (
            MapExpansionDecl(
                expander=fixture.expander,
                outcome="expand",
                membership_port="members",
                item_type=ArtifactType(name="bytes", revision=1),
            ),
        )
    with pytest.raises(EffectRejected) as error:
        _admit_fixture(fixture, map_expansions=cast(tuple[MapExpansionDecl, ...], declarations))
    assert error.value.code == code
    assert case["expected"] == {"status": "rejected", "code": code.value}


@pytest.mark.parametrize(
    ("case_id", "foreign", "code"),
    (
        ("map/duplicate_map_source", False, ValidationCode.DUPLICATE),
        ("map/foreign_before_duplicate", True, ValidationCode.FOREIGN_OWNER),
    ),
)
def test_map_source_admission_precedence_matches_frozen_case(
    case_id: str,
    foreign: bool,
    code: ValidationCode,
) -> None:
    case = MAP_CASES[case_id]
    fixture = _map_fixture()
    scope = fixture.workflow.scopes[0]
    declaration = scope.maps[0]
    duplicate = replace(
        declaration,
        expander=NodeId.new(workflow=WorkflowId.new()) if foreign else declaration.expander,
    )

    with pytest.raises(ContractViolation) as error:
        admit_activation_workflow(
            workflow=fixture.workflow.workflow,
            scopes=(replace(scope, maps=(declaration, duplicate)),),
            limits=replace(fixture.workflow.limits, max_maps=2),
        )

    assert error.value.code is code
    assert case["expected"]["status"] == "rejected"
    assert case["expected"]["code"] == code.value


@pytest.mark.parametrize("case_id", ("map/map_loop_duplicate_source", "map/duplicate_loop_source"))
def test_duplicate_dynamic_source_roles_reject_before_shape_checks(case_id: str) -> None:
    case = MAP_CASES[case_id]
    workflow, starter, member, _ = _loop_workflow(2)
    scope = workflow.scopes[0]
    maps = scope.maps
    loops = scope.loops
    if case_id == "map/map_loop_duplicate_source":
        maps = (
            MapDecl(
                expander=starter,
                member=member,
                expansion_outcomes=frozenset({"again"}),
                max_children=1,
            ),
        )
    else:
        loops = (scope.loops[0], scope.loops[0])

    with pytest.raises(ContractViolation) as error:
        admit_activation_workflow(
            workflow=workflow.workflow,
            scopes=(replace(scope, maps=maps, loops=loops),),
            limits=replace(
                workflow.limits,
                max_maps=len(maps),
                max_loops=len(loops),
                max_children_per_map=1,
                max_activation_occurrences=8,
            ),
        )
    assert error.value.code == ValidationCode.DUPLICATE
    assert error.value.code.value == case["expected"]["code"]


def test_map_provenance_capacity_rejects_before_execution_at_one_over() -> None:
    rejected = MAP_CASES["map/provenance_one_over"]
    accepted = MAP_CASES["map/bounds_exact"]
    fixture = _map_fixture()

    with pytest.raises(EffectRejected) as error:
        _admit_fixture(fixture, baseline_provenance_edges=1, provenance_edge_headroom=1)

    assert error.value.code == EffectCode.LIMIT_EXCEEDED
    assert rejected["boundary"] == "map_execution_preflight"
    assert rejected["events"] == []
    assert rejected["expected"] == {"status": "rejected", "code": error.value.code.value}

    plan = _admit_fixture(fixture, baseline_provenance_edges=1, provenance_edge_headroom=2)
    assert plan.assessment_limits.max_provenance_edges == 3
    assert accepted["expected"]["status"] == "accepted"


@pytest.mark.parametrize("item_count", (0, 1, 2))
def test_map_harness_publishes_real_occurrences(item_count: int) -> None:
    asyncio.run(_assert_map_membership(item_count))


async def _assert_map_membership(item_count: int) -> None:
    fixture, result, callbacks = await _execute_membership(item_count)
    state = result.states[0]
    expansion = next(iter(state.expansions))
    assert expansion.status == "closed"
    assert len(expansion.members) == item_count
    map_items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    assert len(map_items) == item_count
    assert {fact.key.item_key for fact in map_items} == set(range(item_count))
    parent = next(
        fact.key
        for fact in result.provenance
        if isinstance(fact.key, OperationOutputKey)
        and fact.key.activation == expansion.parent
        and fact.key.port == "members"
    )
    assert all(fact.parents == frozenset({parent}) for fact in map_items)
    membership = next(fact for fact in result.provenance if fact.key == parent)
    assert len(membership.parents) == 1
    assert isinstance(next(iter(membership.parents)), RootInputKey)
    member_inputs = [call[0] for call in callbacks[fixture.member].calls]
    assert [item.inputs[0].value for item in member_inputs] == [
        TextArtifactValue(text=f"item-{index}") for index in range(item_count)
    ]
    item_artifacts = {fact.key.member: fact.artifact for fact in map_items}
    assert all(
        isinstance(item.association, SemanticAssociation)
        and item.inputs[0].artifact == item_artifacts[item.association.task.activation]
        for item in member_inputs
    )
    assert {entry.template for entry in state.entries if entry.activation in expansion.members} == (
        {fixture.member} if item_count else set()
    )


@pytest.mark.parametrize(
    ("case_id", "response_mode", "other_count"),
    (
        ("map/missing_membership_port", "missing", 1),
        ("map/wrong_membership_type", "wrong_type", 0),
        ("map/duplicate_membership_port", "duplicate", 0),
    ),
)
def test_malformed_map_results_publish_no_partial_facts(case_id: str, response_mode: str, other_count: int) -> None:
    asyncio.run(_assert_malformed_map_result(case_id, response_mode, other_count))


async def _assert_malformed_map_result(case_id: str, response_mode: str, other_count: int) -> None:
    case = MAP_CASES[case_id]
    fixture, result, callbacks = await _execute_membership(1, response_mode=response_mode, other_count=other_count)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "failure"
    assert expander.outcome is None
    assert case["expected"]["state"]["terminal"] == "malformed_response"
    _assert_only_setup_baseline(fixture, result, callbacks)


def test_cross_returned_map_association_fails_before_publication() -> None:
    asyncio.run(_assert_cross_returned_map_association())


async def _assert_cross_returned_map_association() -> None:
    case = MAP_CASES["map/wrong_parent"]
    fixture, result, callbacks = await _execute_membership(
        1,
        response_mode="cross_association",
        target_count=2,
    )
    callback = callbacks[fixture.expander]
    assert len(callback.calls) == len(callback.cross_associations) == 2
    assert callback.calls[0][0].association != callback.calls[1][0].association
    assert case["boundary"] == "local_callback"
    for state in result.states:
        expander = next(entry for entry in state.entries if entry.template == fixture.expander)
        assert expander.status == "failure"
        assert expander.outcome is None
        expansion = next(iter(state.expansions))
        assert expansion.parent == expander.activation
        assert expansion.status == "failed"
        assert not expansion.members
    assert not result.assessments
    assert not any(isinstance(fact.key, (OperationOutputKey, MapItemKey)) for fact in result.provenance)
    assert not callbacks[fixture.member].calls
    assert case["expected"]["state"]["terminal"] == "malformed_response"


@pytest.mark.parametrize("destination", ("join", "ordinary", "workflow_output"))
@pytest.mark.parametrize("maximum", (0, 1))
def test_map_scalar_source_resolution_uses_the_unique_dynamic_member(destination: str, maximum: int) -> None:
    asyncio.run(_assert_map_scalar_source_resolution(destination, maximum))


async def _assert_map_scalar_source_resolution(destination: str, maximum: int) -> None:
    case = MAP_CASES[f"map/resolve_{destination}_max_{maximum}"]
    fixture, result, callbacks = await _execute_membership(
        maximum,
        item_values=("a",) if maximum else (),
        max_children=maximum,
        outward_scalar=destination,
        outward_identity=destination != "workflow_output",
    )
    expected_resolution = case["expected"]["state"]["resolution"]
    if destination == "workflow_output":
        values = [fact for fact in result.final_outputs if fact.port == "value"]
        assert len(values) == maximum
        if maximum:
            expansion = next(iter(result.states[0].expansions))
            assert len(expansion.members) == 1
            member_activation = next(iter(expansion.members))
            member = next(
                entry
                for entry in result.states[0].entries
                if entry.template == fixture.member and entry.activation == member_activation
            )
            output = next(
                fact
                for fact in result.provenance
                if isinstance(fact.key, OperationOutputKey)
                and fact.key.activation == member.activation
                and fact.key.port == "value"
            )
            assert values[0].producer == output.key
            assert values[0].candidate.artifact == output.artifact
            assert values[0].outcome == "ok"
        assert expected_resolution == ("member:0" if maximum else "blocked:workflow_output")
        return

    destination_node = fixture.join
    if destination == "ordinary":
        destination_node = next(
            node
            for node in fixture.implementation_nodes
            if node not in {fixture.expander, fixture.member_implementation, fixture.join}
        )
    calls = callbacks[destination_node].calls
    assert len(calls) == maximum
    if maximum:
        assert calls[0][0].inputs[0].value == TextArtifactValue(text="a")
        assert expected_resolution == "member:0"
    else:
        destination_entry = next(entry for entry in result.states[0].entries if entry.template == destination_node)
        assert destination_entry.status == "blocked"
        assert expected_resolution == f"blocked:{destination}"


@pytest.mark.parametrize("destination", ("join", "ordinary", "workflow_output"))
def test_map_scalar_source_resolution_rejects_ambiguous_fanout(destination: str) -> None:
    case = MAP_CASES[f"map/resolve_{destination}_max_2"]
    with pytest.raises(ContractViolation) as error:
        _map_fixture(max_children=2, outward_scalar=destination)
    assert error.value.code == ValidationCode.UNSUPPORTED
    assert error.value.code.value == case["expected"]["code"]


@pytest.mark.parametrize("destination", ("join", "ordinary", "workflow_output"))
def test_map_scalar_source_resolution_blocks_an_empty_max_one_expansion(destination: str) -> None:
    asyncio.run(_assert_empty_max_one_map_scalar_resolution(destination))


async def _assert_empty_max_one_map_scalar_resolution(destination: str) -> None:
    case = MAP_CASES[f"map/resolve_{destination}_max_1_empty"]
    fixture, result, callbacks = await _execute_membership(
        0,
        item_values=(),
        max_children=1,
        outward_scalar=destination,
        outward_identity=destination != "workflow_output",
    )
    if destination == "workflow_output":
        assert not any(fact.port == "value" for fact in result.final_outputs)
    else:
        destination_node = fixture.join
        if destination == "ordinary":
            destination_node = next(
                node
                for node in fixture.implementation_nodes
                if node not in {fixture.expander, fixture.member_implementation, fixture.join}
            )
        assert not callbacks[destination_node].calls
        entry = next(item for item in result.states[0].entries if item.template == destination_node)
        assert entry.status == "blocked"
    assert case["expected"]["state"]["resolution"] == f"blocked:{destination}"


@pytest.mark.parametrize("destination", ("join", "ordinary", "workflow_output"))
@pytest.mark.parametrize(("item_count", "response_mode"), ((2, "valid"), (1, "missing")))
def test_map_scalar_source_blocks_terminal_expansion_without_unique_member(
    destination: str,
    item_count: int,
    response_mode: str,
) -> None:
    asyncio.run(_assert_terminal_map_without_scalar(destination, item_count, response_mode))


async def _assert_terminal_map_without_scalar(destination: str, item_count: int, response_mode: str) -> None:
    fixture, result, callbacks = await _execute_membership(
        item_count,
        response_mode=response_mode,
        max_children=1,
        outward_scalar=destination,
        outward_identity=destination != "workflow_output",
    )
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == ("overflow" if response_mode == "valid" else "failed")
    assert not expansion.members
    if destination == "workflow_output":
        assert not any(fact.port == "value" for fact in result.final_outputs)
        return
    destination_node = fixture.join
    if destination == "ordinary":
        destination_node = next(
            node
            for node in fixture.implementation_nodes
            if node not in {fixture.expander, fixture.member_implementation, fixture.join}
        )
    assert not callbacks[destination_node].calls
    entry = next(item for item in result.states[0].entries if item.template == destination_node)
    expected_status = "inconsistent" if destination == "join" and response_mode == "valid" else "blocked"
    assert entry.status == expected_status


def test_map_overflow_publishes_collection_without_item_facts() -> None:
    asyncio.run(_assert_map_overflow())


async def _assert_map_overflow() -> None:
    case = MAP_CASES["map/membership_one_over"]
    fixture, result, _ = await _execute_membership(3)
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "overflow"
    assert not expansion.members
    assert case["expected"]["state"]["terminal"] == "overflow"
    outputs = [fact for fact in result.provenance if isinstance(fact.key, OperationOutputKey)]
    assert len(outputs) == 1
    assert outputs[0].key.port == "members"
    assert not any(isinstance(fact.key, MapItemKey) for fact in result.provenance)
    assert len(result.artifacts) == 2  # one captured root plus the accepted collection
    assert len(result.assessments) == 1


def test_overflow_collection_storage_exact_matches_frozen_publication() -> None:
    asyncio.run(_assert_overflow_collection_storage_exact())


async def _assert_overflow_collection_storage_exact() -> None:
    case = MAP_CASES["map/overflow_collection_storage_exact"]
    fixture, result, callbacks = await _execute_membership(
        3,
        item_values=("a", "b", "c"),
        max_collection_items=3,
    )
    expected = case["expected"]["state"]["publication"]
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "overflow"
    assert not expansion.members
    root = next(fact for fact in result.provenance if isinstance(fact.key, RootInputKey))
    staged = [(artifact, value) for artifact, value in result.artifacts if artifact != root.artifact]
    assert len(staged) == len(expected["artifacts"]) == 1
    collection = cast(TextCollectionValue, staged[0][1])
    assert [item.value.text for item in collection.items] == ["a", "b", "c"]
    assert len(result.assessments) == len(expected["assessments"]) == 1
    output = next(fact for fact in result.provenance if isinstance(fact.key, OperationOutputKey))
    assert output.parents == frozenset({root.key})
    assert not any(isinstance(fact.key, MapItemKey) for fact in result.provenance)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "success"


def test_control_only_map_activates_members_without_item_facts() -> None:
    asyncio.run(_assert_control_only_map())


def test_dynamic_map_items_override_an_unavailable_static_default() -> None:
    asyncio.run(_assert_dynamic_map_items_override_default())


async def _assert_dynamic_map_items_override_default() -> None:
    await _assert_map_result_publication(
        "map/default_override_no_fallback",
        ("actual",),
        artifact_headroom=8,
        artifact_byte_headroom=32,
        other_count=0,
        default_override=True,
    )


async def _assert_control_only_map() -> None:
    case = MAP_CASES["map/control_only_members"]
    fixture, result, callbacks = await _execute_membership(2, control_only=True)
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "closed"
    assert len(expansion.members) == 2
    assert len(callbacks[fixture.member].calls) == 2
    assert all(not call[0].inputs for call in callbacks[fixture.member].calls)
    assert not any(isinstance(fact.key, MapItemKey) for fact in result.provenance)
    expected = case["expected"]["state"]["publication"]
    assert len(result.artifacts) - 1 == len(expected["artifacts"]) == 1


@pytest.mark.parametrize(
    ("case_id", "values", "artifact_headroom", "artifact_byte_headroom", "other_count"),
    (
        ("map/membership_0", (), 8, 32, 0),
        ("map/membership_1", ("a",), 8, 32, 0),
        ("map/membership_2", ("a", "b"), 8, 32, 0),
        ("map/membership_with_0_other_outputs", ("a", "b"), 8, 32, 0),
        ("map/membership_with_1_other_outputs", ("a", "b"), 8, 32, 1),
        ("map/membership_with_2_other_outputs", ("a", "b"), 8, 32, 2),
        ("map/collection_items_exact", ("a", "b"), 8, 32, 0),
        ("map/bounds_exact", ("aa", "bb"), 3, 8, 0),
    ),
)
def test_map_result_publication_matches_frozen_case(
    case_id: str,
    values: tuple[str, ...],
    artifact_headroom: int,
    artifact_byte_headroom: int,
    other_count: int,
) -> None:
    asyncio.run(
        _assert_map_result_publication(
            case_id,
            values,
            artifact_headroom,
            artifact_byte_headroom,
            other_count,
        )
    )


async def _assert_map_result_publication(
    case_id: str,
    values: tuple[str, ...],
    artifact_headroom: int,
    artifact_byte_headroom: int,
    other_count: int,
    default_override: bool = False,
) -> None:
    case = MAP_CASES[case_id]
    fixture, result, callbacks = await _execute_membership(
        len(values),
        item_values=values,
        artifact_headroom=artifact_headroom,
        artifact_byte_headroom=artifact_byte_headroom,
        other_count=other_count,
        default_override=default_override,
        provenance_edge_headroom=cast(dict[str, int], case["declaration"]["limits"])["max_provenance_edges"],
    )
    expected = case["expected"]["state"]["publication"]
    root = next(fact for fact in result.provenance if isinstance(fact.key, RootInputKey))
    baseline_provenance = [
        fact for fact in result.provenance if isinstance(fact.key, (RootInputKey, BoundInputKey, InitialCollectionKey))
    ]
    baseline_artifacts = {fact.artifact for fact in baseline_provenance}
    staged_provenance = [fact for fact in result.provenance if fact not in baseline_provenance]
    staged_artifacts = [(artifact, value) for artifact, value in result.artifacts if artifact not in baseline_artifacts]
    collection = next(value for _, value in staged_artifacts if isinstance(value, TextCollectionValue))
    items = [value for _, value in staged_artifacts if isinstance(value, TextArtifactValue)]
    assert [item.value.text for item in collection.items] == list(values)
    assert sorted(item.text for item in items) == sorted(values)
    assert len(staged_artifacts) == len(expected["artifacts"])
    assert len(result.assessments) == len(expected["assessments"]) == 1
    assert result.assessments[0].finding.code == "assessment0"
    staged_ports = [fact for fact in result.ports if fact.port not in {"default", "schema"}]
    assert len(staged_ports) == len(expected["ports"])
    assert len(staged_provenance) == len(expected["provenance"])
    outputs = {fact.key.port: fact for fact in staged_provenance if isinstance(fact.key, OperationOutputKey)}
    assert set(outputs) == {"members", *(f"other{index}" for index in range(other_count))}
    output = outputs["members"]
    assert output.parents == frozenset({root.key})
    values_by_artifact = dict(staged_artifacts)
    for index in range(other_count):
        other = outputs[f"other{index}"]
        assert other.parents == frozenset({root.key})
        assert values_by_artifact[other.artifact] == TextCollectionValue(items=())
    map_items = [fact for fact in staged_provenance if isinstance(fact.key, MapItemKey)]
    assert all(fact.parents == frozenset({output.key}) for fact in map_items)
    item_artifacts = {fact.key.member: fact.artifact for fact in map_items}
    member_inputs = [call[0] for call in callbacks[fixture.member].calls]
    assert [item.inputs[0].value for item in member_inputs] == [TextArtifactValue(text=value) for value in values]
    assert all(
        isinstance(item.association, SemanticAssociation)
        and item.inputs[0].artifact == item_artifacts[item.association.task.activation]
        for item in member_inputs
    )
    if default_override:
        assert fixture.default_source is not None
        assert len(callbacks[fixture.default_source].calls) == 1
    expander_ports = {fact.port: fact for fact in staged_ports if fact.node == fixture.expander}
    assert set(expander_ports) == set(outputs)
    assert all(expander_ports[port].artifact == fact.artifact for port, fact in outputs.items())
    item_ports = [fact for fact in staged_ports if fact.node == fixture.member]
    assert {fact.activation: fact.artifact for fact in item_ports} == item_artifacts
    assert all(fact.port == "item" and fact.artifact_type == fixture.text_type for fact in item_ports)
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == "closed"
    assert len(expansion.members) == len(values)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "success"


@pytest.mark.parametrize(
    ("case_id", "artifact_headroom", "max_collection_items"),
    (
        ("map/artifact_count_one_over", 2, 4),
        ("map/collection_items_one_over", 8, 1),
        ("map/overflow_collection_storage_one_over", 8, 2),
        ("map/artifact_bytes_one_over", 8, 4),
    ),
)
def test_map_storage_limits_rollback_publication(
    case_id: str,
    artifact_headroom: int,
    max_collection_items: int,
) -> None:
    asyncio.run(_assert_map_storage_limit(case_id, artifact_headroom, max_collection_items))


async def _assert_map_storage_limit(case_id: str, artifact_headroom: int, max_collection_items: int) -> None:
    case = MAP_CASES[case_id]
    item_count = 3 if case_id == "map/overflow_collection_storage_one_over" else 2
    item_values = ("aa", "bb") if case_id == "map/artifact_bytes_one_over" else None
    fixture, result, callbacks = await _execute_membership(
        item_count,
        item_values=item_values,
        artifact_headroom=artifact_headroom,
        artifact_byte_headroom=7 if case_id == "map/artifact_bytes_one_over" else 32,
        max_collection_items=max_collection_items,
    )
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "blocked"
    assert expander.outcome is None
    assert case["expected"]["state"]["terminal"] == "artifact_limit"
    _assert_only_setup_baseline(fixture, result, callbacks)


@pytest.mark.parametrize(
    "case_id",
    ("map/prospective_transition_rejected", "map/caller_transition_verdict_rejected"),
)
def test_map_result_after_parent_cancellation_is_not_published(case_id: str) -> None:
    asyncio.run(_assert_cancelled_map_result(case_id))


async def _assert_cancelled_map_result(case_id: str) -> None:
    case = MAP_CASES[case_id]
    fixture, result, callbacks = await _execute_membership(1, cancel_before_result=True)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "cancelled"
    assert case["expected"]["state"]["terminal"] == "transition_rejected"
    _assert_only_setup_baseline(fixture, result, callbacks)


def _assert_only_setup_baseline(
    fixture: _MapFixture,
    result: Any,
    callbacks: dict[NodeId, _MapCallback],
) -> None:
    assert not result.assessments
    assert all(isinstance(fact.key, (RootInputKey, BoundInputKey, InitialCollectionKey)) for fact in result.provenance)
    assert not any(isinstance(fact.key, (OperationOutputKey, MapItemKey)) for fact in result.provenance)
    assert {artifact for artifact, _ in result.artifacts} == {fact.artifact for fact in result.provenance}
    assert all(port.node == fixture.expander and port.port in {"default", "schema"} for port in result.ports)
    assert {port.artifact for port in result.ports} <= {fact.artifact for fact in result.provenance}
    requests = result.requests
    assert requests.dispatched_count == 0
    assert not requests.dispatches
    assert not requests.denials
    assert not requests.terminals
    assert not requests.settlements
    assert not requests.defects
    assert not requests.local_in_flight
    assert not requests.remote_outstanding
    assert not requests.cancel_requested
    expansions = result.states[0].expansions
    assert len(expansions) == 1
    expansion = next(iter(expansions))
    assert expansion.status == "failed"
    assert not expansion.members
    parent = next(entry.activation for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expansion.parent == parent
    assert not callbacks[fixture.member].calls


@pytest.mark.parametrize(
    ("count", "mode", "status"), [(0, "valid", "closed"), (2, "valid", "overflow"), (1, "missing", "failed")]
)
def test_canonical_membership_retains_expansions_without_instantiated_children(
    count: int, mode: str, status: str
) -> None:
    _, result, _ = asyncio.run(_execute_membership(count, response_mode=mode, max_children=1, outward_scalar="join"))
    expansion = next(iter(result.states[0].expansions))
    assert expansion.status == status
    assert not expansion.members
    memberships = [item for item in result.record.memberships if item.parent == expansion.parent]
    assert len(memberships) == 1
    assert memberships[0].closed
    assert not memberships[0].members
    assert not any(entry.activation.parent == expansion.parent for entry in result.states[0].entries)
    assert {item.activation for item in result.record.terminals} == {
        member for membership in result.record.memberships for member in membership.members
    }
