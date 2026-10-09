# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed map workflow fixtures and operation callbacks."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any, cast

from anonymizer.engine.graph_sdk.executor import (
    AssessmentFinding,
    LocalAssessmentResult,
    LocalCompleted,
    LocalFailure,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
)
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
    ProtectionRequirement,
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
from tests.graph_sdk.reference.corpora import load_cases
from tests.graph_sdk.test_dynamic_executor import _capability, _outcome

MAP_CASES = {item["case_id"]: item for item in load_cases("effects") if item["family"] == "map"}


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
    member_assessment: bool = False


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
    member_assessment: bool = False,
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
    if member_assessment:
        assert not control_only and not outward_scalar and not member_subgraph
        member_operation = replace(
            member_operation,
            inputs=(*member_operation.inputs, InputPort(name="subject", artifact_type=text_type)),
            outputs=(OutputPort(name="evidence", artifact_type=text_type),),
            output_dependencies=(
                OutputDependency(output="evidence", inputs=frozenset({"subject"}), identity_input=None),
            ),
            outcomes=(
                replace(
                    member_operation.outcomes[0],
                    produced_ports=frozenset({"evidence"}),
                    ceiling=replace(member_operation.outcomes[0].ceiling, max_output_bytes=32),
                    evidence=frozenset(
                        {
                            EvidencePromise(
                                name="member_checked",
                                meaning="member accepted",
                                subject_port="subject",
                                consumed_ports=frozenset({"subject"}),
                                coverage=frozenset(),
                            )
                        }
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
                    max_output_bytes=512 if default_source is not None or outward_scalar or member_assessment else 128,
                ),
                evidence=frozenset(
                    {
                        EvidencePromise(
                            name="assessment0",
                            meaning="map membership accepted",
                            subject_port="members",
                            consumed_ports=frozenset({"default"}),
                            coverage=frozenset(),
                        ),
                        *(
                            (
                                EvidencePromise(
                                    name="member_checked",
                                    meaning="member accepted",
                                    subject_port="default",
                                    consumed_ports=frozenset({"default"}),
                                    coverage=frozenset(),
                                ),
                            )
                            if member_assessment
                            else ()
                        ),
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
                (
                    InputBinding(
                        source=WorkflowInputRef(port="default"), destination=NodeInputRef(node=member, port="subject")
                    ),
                )
                if member_assessment
                else ()
            ),
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
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning="map membership accepted",
                subject_port="members",
                consumed_ports=frozenset({"default"}),
                coverage=frozenset(),
            ),
            ProtectionRequirement(
                outcome="ok",
                meaning="member accepted",
                subject_port="default",
                consumed_ports=frozenset({"default"}),
                coverage=frozenset(),
            ),
        )
        if member_assessment
        else (),
        limits=WorkflowLimits(
            max_nodes=3 + int(default_source is not None) + int(ordinary is not None) + int(member_subgraph),
            max_bindings=(5 if other_count else 4) + int(outward_scalar is not None) + int(member_assessment),
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
        member_assessment,
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
    same_key_versions: bool = False
    item_counts: tuple[int, ...] | None = None
    member_assessment: bool = False
    finalize_collection: bool = False
    assessment_promise: str | None = None

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
                            key=7 if self.same_key_versions else index,
                            version=index + 1 if self.same_key_versions else 1,
                            value=TextArtifactValue(
                                text=(self.item_values[index] if self.item_values is not None else f"item-{index}")
                            ),
                        )
                        for index in range(
                            self.item_counts[len(self.calls) - 1] if self.item_counts is not None else self.item_count
                        )
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
        elif self.finalize_collection:
            values = [item.value for item in request[0].inputs]
            assert all(isinstance(value, TextCollectionValue) for value in values)
            outputs = (
                PortArtifact(
                    port="value",
                    artifact_type=self.text_type,
                    artifact=None,
                    value=TextArtifactValue(
                        text=",".join(
                            item.value.text
                            for value in values
                            if isinstance(value, TextCollectionValue)
                            for item in value.items
                        )
                    ),
                ),
            )
        elif self.member_assessment:
            outputs = (
                PortArtifact(
                    port="evidence",
                    artifact_type=self.text_type,
                    artifact=None,
                    value=TextArtifactValue(text="checked"),
                ),
            )
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
                    promise=self.assessment_promise or "assessment0",
                    evidence_port="members",
                    finding=AssessmentFinding(status="satisfied", code=self.assessment_promise or "assessment0"),
                ),
            )
            if self.mode == "expander"
            else (
                (
                    LocalAssessmentResult(
                        association=association,
                        promise=self.assessment_promise or "member_checked",
                        evidence_port="evidence",
                        finding=AssessmentFinding(status="satisfied", code=self.assessment_promise or "member_checked"),
                    ),
                )
                if self.member_assessment
                else ()
            ),
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
