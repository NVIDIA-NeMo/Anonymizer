# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate frozen map effects cases through the real graph executor."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
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
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
)
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    InputPort,
    KeyedJoinDecl,
    MapDecl,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
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
    join: NodeId
    text_type: ArtifactType
    collection_type: ArtifactType
    capabilities: tuple[Any, ...]


def _map_fixture(*, max_children: int = 2) -> _MapFixture:
    owner = WorkflowId.new()
    expander, member, join = (NodeId.new(workflow=owner) for _ in range(3))
    text_type = ArtifactType(name="text", revision=1)
    collection_type = ArtifactType(name="members_t", revision=1)
    expander_operation = OperationSpec(
        name="expander",
        inputs=(InputPort(name="default", artifact_type=text_type),),
        outputs=(OutputPort(name="members", artifact_type=collection_type),),
        output_dependencies=(OutputDependency(output="members", inputs=frozenset({"default"}), identity_input=None),),
        outcomes=(
            _outcome(
                "expand",
                produced=frozenset({"members"}),
                max_activations=max_children + 2,
                max_output_bytes=128,
            ),
        ),
    )
    member_operation = OperationSpec(
        name="member",
        inputs=(InputPort(name="item", artifact_type=text_type),),
        outputs=(),
        output_dependencies=(),
        outcomes=(_outcome("ok"),),
    )
    join_operation = _operation("join", (_outcome("ok"),))
    interface = OperationSpec(
        name="map-root",
        inputs=(InputPort(name="default", artifact_type=text_type),),
        outputs=(),
        output_dependencies=(),
        outcomes=(_outcome("ok", max_activations=max_children + 4, max_output_bytes=128),),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(
            OperationNode(id=expander, operation=expander_operation),
            OperationNode(id=member, operation=member_operation),
            OperationNode(id=join, operation=join_operation),
        ),
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="default"),
                destination=NodeInputRef(node=member, port="item"),
            ),
            InputBinding(
                source=WorkflowInputRef(port="default"),
                destination=NodeInputRef(node=expander, port="default"),
            ),
        ),
        output_bindings=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=join, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(SequenceEdge(before=expander, after=member), SequenceEdge(before=member, after=join)),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=3,
            max_bindings=3,
            max_sequence_edges=2,
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
                maps=(
                    MapDecl(
                        expander=expander,
                        member=member,
                        expansion_outcomes=frozenset({"expand"}),
                        max_children=max_children,
                        item_input="item",
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
        ),
        limits=DynamicLimits(
            max_maps=1,
            max_joins=1,
            max_loops=0,
            max_children_per_map=max_children,
            max_iterations_per_loop=0,
            max_dynamic_depth=1,
            max_activation_occurrences=max_children + 2,
        ),
    )
    operations = (expander_operation, member_operation, join_operation)
    capabilities = tuple(_capability(operation, index) for index, operation in enumerate(operations))
    return _MapFixture(workflow, expander, member, join, text_type, collection_type, capabilities)


@dataclass
class _MapCallback:
    mode: str
    item_count: int
    collection_type: ArtifactType
    text_type: ArtifactType
    calls: list[tuple[AssociationInput, ...]] = field(default_factory=list)

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls.append(request)
        outputs: tuple[PortArtifact, ...] = ()
        outcome = "ok"
        if self.mode == "expander":
            outcome = "expand"
            outputs = (
                PortArtifact(
                    port="members",
                    artifact_type=self.collection_type,
                    artifact=None,
                    value=TextCollectionValue(
                        items=tuple(
                            TextCollectionItem(
                                key=index,
                                version=1,
                                value=TextArtifactValue(text=f"item-{index}"),
                            )
                            for index in range(self.item_count)
                        )
                    ),
                ),
            )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=request[0].association,
                    outcome=outcome,
                    outputs=outputs,
                    consumed_context_ports=frozenset(),
                ),
            )
        )


def _admit_fixture(
    fixture: _MapFixture,
    *,
    map_expansions: tuple[MapExpansionDecl, ...] | None = None,
):
    by_node = dict(zip((fixture.expander, fixture.member, fixture.join), fixture.capabilities, strict=True))
    data = _data(1)
    target = next(iter(data.targets))
    prepared = prepare(
        data=data,
        workflow=fixture.workflow,
        activation_limits=ActivationLimits(max_events=20, max_entries=4, max_parent_depth=2),
        bound_inputs=(BoundInput(target=target, source=target, port="default", artifact_type=fixture.text_type),),
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
        limits=_limits(capabilities=3, slots=4),
    )
    context = admit_context_plan(
        prepared=prepared,
        bound_context=None,
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
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=8,
            max_provenance_edges=4,
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


async def _execute_membership(item_count: int):
    fixture = _map_fixture()
    admitted = _admit_fixture(fixture)
    by_node = dict(zip((fixture.expander, fixture.member, fixture.join), fixture.capabilities, strict=True))
    callbacks = {
        node: _MapCallback(
            mode="expander" if node == fixture.expander else "operation",
            item_count=item_count,
            collection_type=fixture.collection_type,
            text_type=fixture.text_type,
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
    result = await (
        await start_execution(
            admitted=admitted,
            capabilities=fixture.capabilities,
            services=ExecutionServices(
                handles=handles,
                context_resources=(),
                limits=ExecutionLimits(
                    max_local_in_flight=4,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=8,
                    max_runtime_artifact_bytes=128,
                    max_collection_items=4,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    return fixture, result, callbacks


def test_map_operation_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_operation"]
    fixture = _map_fixture()
    assert case["expected"] == {"status": "accepted"}
    assert fixture.workflow.scopes[0].maps[0].member == fixture.member


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
