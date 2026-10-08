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
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
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
    join: NodeId
    text_type: ArtifactType
    collection_type: ArtifactType
    other_type: ArtifactType
    capabilities: tuple[Any, ...]


def _map_fixture(*, max_children: int = 2, control_only: bool = False, other_count: int = 0) -> _MapFixture:
    owner = WorkflowId.new()
    expander, member, join = (NodeId.new(workflow=owner) for _ in range(3))
    text_type = ArtifactType(name="text", revision=1)
    collection_type = ArtifactType(name="members_t", revision=1)
    other_type = ArtifactType(name="other_t", revision=1)
    other_ports = tuple(OutputPort(name=f"other{index}", artifact_type=other_type) for index in range(other_count))
    expander_operation = OperationSpec(
        name="expander",
        inputs=(
            InputPort(name="default", artifact_type=text_type),
            *((InputPort(name="schema", artifact_type=other_type),) if other_count else ()),
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
        outputs=(),
        output_dependencies=(),
        outcomes=(_outcome("ok"),),
    )
    join_operation = _operation("join", (_outcome("ok"),))
    interface = OperationSpec(
        name="map-root",
        inputs=(
            InputPort(name="default", artifact_type=text_type),
            *((InputPort(name="schema", artifact_type=other_type),) if other_count else ()),
        ),
        outputs=(OutputPort(name="members", artifact_type=collection_type),),
        output_dependencies=(OutputDependency(output="members", inputs=frozenset({"default"}), identity_input=None),),
        outcomes=(
            replace(
                _outcome(
                    "ok",
                    produced=frozenset({"members"}),
                    max_activations=max_children + 4,
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
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(
            OperationNode(id=expander, operation=expander_operation),
            OperationNode(id=member, operation=member_operation),
            OperationNode(id=join, operation=join_operation),
        ),
        input_bindings=(
            *(
                ()
                if control_only
                else (
                    InputBinding(
                        source=WorkflowInputRef(port="default"),
                        destination=NodeInputRef(node=member, port="item"),
                    ),
                )
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
        ),
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
            max_bindings=5 if other_count else 4,
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
    return _MapFixture(workflow, expander, member, join, text_type, collection_type, other_type, capabilities)


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

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls.append(request)
        association = request[0].association
        assert isinstance(association, SemanticAssociation)
        if self.mode == "expander" and self.started is not None and self.release is not None:
            self.started.set()
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                await self.release.wait()
        outputs: tuple[PortArtifact, ...] = ()
        outcome = "ok"
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
    by_node = dict(zip((fixture.expander, fixture.member, fixture.join), fixture.capabilities, strict=True))
    data = _data(1) if data is None else data
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
            max_assessment_facts=1,
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
):
    fixture = _map_fixture(control_only=control_only, other_count=other_count)
    data = _data(1)
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
                        artifact_type=fixture.other_type,
                        source=SOURCE,
                        selector=ContextSelector(fields=()),
                        requirement="required",
                        bounds=RetrievalBounds(max_items=1, max_bytes=6, max_requests=1),
                        materialization=ContextMaterialization(kind="collection", item_type=fixture.text_type),
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
    baseline_port_facts = 1 + (1 if bound_context is not None else 0)
    baseline_provenance_edges = 1 if bound_context is not None else 0
    admitted = _admit_fixture(
        fixture,
        data=data,
        bound_context=bound_context,
        baseline_port_facts=baseline_port_facts,
        baseline_provenance_edges=baseline_provenance_edges,
    )
    baseline_artifacts = len(data.targets) + (len(bound_context.artifacts) if bound_context is not None else 0)
    baseline_bytes = sum(len(datum.text.encode()) for datum in data.datums if datum.id in data.targets) + (
        6 if bound_context is not None else 0
    )
    by_node = dict(zip((fixture.expander, fixture.member, fixture.join), fixture.capabilities, strict=True))
    started = asyncio.Event() if cancel_before_result else None
    release = asyncio.Event() if cancel_before_result else None
    callbacks = {
        node: _MapCallback(
            mode="expander" if node == fixture.expander else "operation",
            response_mode=response_mode,
            item_count=item_count,
            item_values=item_values,
            collection_type=fixture.collection_type,
            text_type=fixture.text_type,
            other_type=fixture.other_type,
            other_count=other_count,
            started=started if node == fixture.expander else None,
            release=release if node == fixture.expander else None,
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


def test_map_operation_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_operation"]
    fixture = _map_fixture()
    assert case["expected"] == {"status": "accepted"}
    assert fixture.workflow.scopes[0].maps[0].member == fixture.member


def test_control_only_map_declaration_admits_through_production() -> None:
    case = MAP_CASES["map/admit_control_only"]
    fixture = _map_fixture(control_only=True)
    assert case["expected"] == {"status": "accepted"}
    assert fixture.workflow.scopes[0].maps[0].item_input is None
    _admit_fixture(fixture)


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
    fixture, result, _ = await _execute_membership(1, response_mode=response_mode, other_count=other_count)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "failure"
    assert expander.outcome is None
    assert case["expected"]["state"]["terminal"] == "malformed_response"
    _assert_only_setup_baseline(fixture, result)


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
) -> None:
    case = MAP_CASES[case_id]
    fixture, result, callbacks = await _execute_membership(
        len(values),
        item_values=values,
        artifact_headroom=artifact_headroom,
        artifact_byte_headroom=artifact_byte_headroom,
        other_count=other_count,
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
    output = next(fact for fact in staged_provenance if isinstance(fact.key, OperationOutputKey))
    assert output.parents == frozenset({root.key})
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
    output_port = next(fact for fact in staged_ports if fact.node == fixture.expander)
    assert output_port.port == "members"
    assert output_port.artifact == output.artifact
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
    fixture, result, _ = await _execute_membership(
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
    _assert_only_setup_baseline(fixture, result)


@pytest.mark.parametrize(
    "case_id",
    ("map/prospective_transition_rejected", "map/caller_transition_verdict_rejected"),
)
def test_map_result_after_parent_cancellation_is_not_published(case_id: str) -> None:
    asyncio.run(_assert_cancelled_map_result(case_id))


async def _assert_cancelled_map_result(case_id: str) -> None:
    case = MAP_CASES[case_id]
    fixture, result, _ = await _execute_membership(1, cancel_before_result=True)
    expander = next(entry for entry in result.states[0].entries if entry.template == fixture.expander)
    assert expander.status == "cancelled"
    assert case["expected"]["state"]["terminal"] == "transition_rejected"
    _assert_only_setup_baseline(fixture, result)


def _assert_only_setup_baseline(fixture: _MapFixture, result: Any) -> None:
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
