# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real nested map and loop execution through the P3 activation authority."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from anonymizer.engine.graph_sdk.capabilities import (
    ImplementationCapability,
    ImplementationRef,
    ImplementationSelection,
)
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.evidence import admit_qualification, evidence_revision_view
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
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.qualification import qualify
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
    OutcomeSpec,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
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
from tests.graph_sdk.test_evidence import _qualification_limits
from tests.graph_sdk.test_local_executor import _Clock
from tests.graph_sdk.test_preparation import _config, _data, _limits


def _outcome(
    name: str,
    *,
    produced: frozenset[str] = frozenset(),
    max_activations: int = 1,
    max_output_bytes: int = 0,
) -> OutcomeSpec:
    return OutcomeSpec(
        name=name,
        category="success",
        produced_ports=produced,
        context=frozenset(),
        evidence=frozenset(),
        state_effects=frozenset(),
        model_requirements=frozenset(),
        ceiling=ResourceCeiling(
            max_activations=max_activations,
            max_model_requests=0,
            max_input_bytes=0,
            max_output_bytes=max_output_bytes,
        ),
    )


def _operation(name: str, outcomes: tuple[OutcomeSpec, ...]) -> OperationSpec:
    return OperationSpec(name=name, inputs=(), outputs=(), output_dependencies=(), outcomes=outcomes)


def _capability(operation: OperationSpec, ordinal: int) -> ImplementationCapability:
    return ImplementationCapability(
        implementation=ImplementationRef(name=f"dynamic-{ordinal}", revision=1),
        operation=operation,
        configuration=_config(),
        effect="local",
        attribution="per_task",
        request_visibility="none",
        pre_dispatch_control="none",
        retry_owner="none",
        error_reporting="typed_terminal",
        cancellation="before_dispatch_only",
        settlement="synchronous",
        usage="exact",
        resource_lifetime="stateless",
        max_physical_requests_per_activation=0,
    )


def _rows(outcomes: frozenset[str]) -> tuple[RuntimeOutcome, ...]:
    values = [
        RuntimeOutcome(condition="result", reported_outcome=name, failure=None, outcome=name, category="success")
        for name in sorted(outcomes)
    ]
    values.extend(
        RuntimeOutcome(condition="failure", reported_outcome=None, failure=failure, outcome=None, category="failure")
        for failure in (
            "rejected_before_acceptance",
            "retryable",
            "malformed_response",
            "permanent",
            "transport_unknown",
            "implementation_exception",
        )
    )
    values.extend(
        RuntimeOutcome(condition=condition, reported_outcome=None, failure=None, outcome=None, category=category)
        for condition, category in (
            ("cancel_before_start", "blocked"),
            ("cancel_after_start", "cancelled"),
            ("artifact_limit_exhausted", "blocked"),
            ("deadline_exhausted", "blocked"),
        )
    )
    return tuple(values)


@dataclass
class _DynamicCallback:
    mode: str
    collection_type: ArtifactType
    text_type: ArtifactType
    calls: int = 0
    seen: list[str] | None = None
    same_key_versions: bool = False

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        self.calls += 1
        association = request[0].association
        assert isinstance(association, SemanticAssociation)
        activation = association.task.activation
        outcome = "again" if self.mode == "loop" and activation.iteration == 0 else "stop"
        outputs: tuple[PortArtifact, ...] = ()
        if self.mode == "expand":
            outcome = "expand"
            outputs = (
                PortArtifact(
                    port="items",
                    artifact_type=self.collection_type,
                    artifact=None,
                    value=TextCollectionValue(
                        items=tuple(
                            TextCollectionItem(
                                key=7 if self.same_key_versions else index,
                                version=index + 1 if self.same_key_versions else 1,
                                value=TextArtifactValue(text=f"item-{index}"),
                            )
                            for index in range(2)
                        )
                    ),
                ),
                PortArtifact(
                    port="default",
                    artifact_type=self.text_type,
                    artifact=None,
                    value=TextArtifactValue(text="unused-default"),
                ),
            )
        elif self.mode in {"join", "root_join"}:
            outcome = "ok"
        elif self.mode == "starter":
            outcome = "again"
            value = request[0].inputs[0].value
            assert isinstance(value, TextArtifactValue)
            if self.seen is not None:
                self.seen.append(value.text)
            outputs = (
                PortArtifact(
                    port="value",
                    artifact_type=self.text_type,
                    artifact=None,
                    value=value,
                ),
            )
        elif self.mode == "loop":
            value = request[0].inputs[0].value
            assert isinstance(value, TextArtifactValue)
            if self.seen is not None:
                self.seen.append(value.text)
            outputs = (
                PortArtifact(
                    port="value",
                    artifact_type=self.text_type,
                    artifact=None,
                    value=TextArtifactValue(text=f"{value.text}:{activation.iteration}"),
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


@pytest.mark.parametrize("same_key_versions", [False, True])
def test_nested_two_by_two_map_loop_executes_all_occurrences(same_key_versions: bool) -> None:
    asyncio.run(_assert_nested_two_by_two(same_key_versions=same_key_versions))


async def _assert_nested_two_by_two(*, same_key_versions: bool = False) -> None:
    body_owner = WorkflowId.new()
    starter, loop_member, loop_join = (NodeId.new(workflow=body_owner) for _ in range(3))
    text_type = ArtifactType(name="text", revision=1)
    control = (
        _outcome("again", produced=frozenset({"value"}), max_output_bytes=100),
        _outcome("stop", produced=frozenset({"value"}), max_output_bytes=100),
    )
    starter_operation = OperationSpec(
        name="starter",
        inputs=(InputPort(name="item", artifact_type=text_type),),
        outputs=(OutputPort(name="value", artifact_type=text_type),),
        output_dependencies=(OutputDependency(output="value", inputs=frozenset({"item"}), identity_input="item"),),
        outcomes=control,
    )
    member_operation = OperationSpec(
        name="loop-member",
        inputs=(InputPort(name="previous", artifact_type=text_type),),
        outputs=(OutputPort(name="value", artifact_type=text_type),),
        output_dependencies=(OutputDependency(output="value", inputs=frozenset({"previous"}), identity_input=None),),
        outcomes=control,
    )
    join_operation = _operation("loop-join", (_outcome("ok"),))
    body_interface = OperationSpec(
        name="body",
        inputs=(InputPort(name="item", artifact_type=text_type),),
        outputs=(),
        output_dependencies=(),
        outcomes=(_outcome("ok", max_activations=4, max_output_bytes=200),),
    )
    body = admit_static_workflow(
        workflow=body_owner,
        interface=body_interface,
        nodes=(
            OperationNode(id=starter, operation=starter_operation),
            OperationNode(id=loop_member, operation=member_operation),
            OperationNode(id=loop_join, operation=join_operation),
        ),
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="item"),
                destination=NodeInputRef(node=starter, port="item"),
            ),
            InputBinding(
                source=NodeOutputRef(node=starter, port="value"),
                destination=NodeInputRef(node=loop_member, port="previous"),
            ),
        ),
        output_bindings=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=loop_join, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(SequenceEdge(before=starter, after=loop_member), SequenceEdge(before=loop_member, after=loop_join)),
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
    body_scope = DynamicScope(
        workflow=body,
        maps=(),
        joins=(
            KeyedJoinDecl(
                source=starter,
                join=loop_join,
                accepted_categories=frozenset({"success"}),
                reduction="all_by_key",
            ),
        ),
        loops=(
            LoopDecl(
                starter=starter,
                member=loop_member,
                join=loop_join,
                enter_outcomes=frozenset({"again"}),
                bypass_outcomes=frozenset({"stop"}),
                continue_outcomes=frozenset({"again"}),
                exit_outcomes=frozenset({"stop"}),
                initial=(
                    LoopInitialBinding(
                        source=NodeOutputRef(node=starter, port="value"),
                        destination=NodeInputRef(node=loop_member, port="previous"),
                    ),
                ),
                carried=(
                    LoopCarriedBinding(
                        source=NodeOutputRef(node=loop_member, port="value"),
                        destination=NodeInputRef(node=loop_member, port="previous"),
                    ),
                ),
                max_iterations=2,
            ),
        ),
    )

    root_owner = WorkflowId.new()
    expander, member, root_join = (NodeId.new(workflow=root_owner) for _ in range(3))
    collection_type = ArtifactType(name="item-collection", revision=1)
    expand_operation = OperationSpec(
        name="expand",
        inputs=(),
        outputs=(
            OutputPort(name="items", artifact_type=collection_type),
            OutputPort(name="default", artifact_type=text_type),
        ),
        output_dependencies=(
            OutputDependency(output="items", inputs=frozenset(), identity_input=None),
            OutputDependency(output="default", inputs=frozenset(), identity_input=None),
        ),
        outcomes=(_outcome("expand", produced=frozenset({"items", "default"}), max_output_bytes=120),),
    )
    expand_body_owner = WorkflowId.new()
    expand_child = NodeId.new(workflow=expand_body_owner)
    expand_interface = OperationSpec(
        name="expand-body",
        inputs=expand_operation.inputs,
        outputs=expand_operation.outputs,
        output_dependencies=expand_operation.output_dependencies,
        outcomes=tuple(
            OutcomeSpec(
                name=outcome.name,
                category=outcome.category,
                produced_ports=outcome.produced_ports,
                context=outcome.context,
                evidence=outcome.evidence,
                state_effects=outcome.state_effects,
                model_requirements=outcome.model_requirements,
                ceiling=ResourceCeiling(
                    max_activations=2,
                    max_model_requests=outcome.ceiling.max_model_requests,
                    max_input_bytes=outcome.ceiling.max_input_bytes,
                    max_output_bytes=outcome.ceiling.max_output_bytes,
                ),
            )
            for outcome in expand_operation.outcomes
        ),
    )
    expand_body = admit_static_workflow(
        workflow=expand_body_owner,
        interface=expand_interface,
        nodes=(OperationNode(id=expand_child, operation=expand_operation),),
        input_bindings=(),
        output_bindings=tuple(
            OutputBinding(
                source=NodeOutputRef(node=expand_child, port=port.name),
                destination=WorkflowOutputRef(port=port.name),
            )
            for port in expand_operation.outputs
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=expand_child, outcome="expand"),
                destination=WorkflowOutcomeRef(outcome="expand"),
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
    root_join_operation = _operation("root-join", (_outcome("ok"),))
    root_interface = _operation("root", (_outcome("ok", max_activations=13, max_output_bytes=400),))
    root = admit_static_workflow(
        workflow=root_owner,
        interface=root_interface,
        nodes=(
            SubgraphNode(id=expander, operation=expand_interface, body=expand_body),
            SubgraphNode(id=member, operation=body.interface, body=body),
            OperationNode(id=root_join, operation=root_join_operation),
        ),
        input_bindings=(
            InputBinding(
                source=NodeOutputRef(node=expander, port="default"),
                destination=NodeInputRef(node=member, port="item"),
            ),
        ),
        output_bindings=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=root_join, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(SequenceEdge(before=expander, after=member), SequenceEdge(before=member, after=root_join)),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=8,
            max_bindings=2,
            max_sequence_edges=2,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
        workflow=root,
        scopes=(
            DynamicScope(
                workflow=root,
                maps=(
                    MapDecl(
                        expander=expander,
                        member=member,
                        expansion_outcomes=frozenset({"expand"}),
                        max_children=2,
                        item_input="item",
                    ),
                ),
                joins=(
                    KeyedJoinDecl(
                        source=expander,
                        join=root_join,
                        accepted_categories=frozenset({"success"}),
                        reduction="all_by_key",
                    ),
                ),
                loops=(),
            ),
            body_scope,
            DynamicScope(workflow=expand_body, maps=(), joins=(), loops=()),
        ),
        limits=DynamicLimits(
            max_maps=1,
            max_joins=2,
            max_loops=1,
            max_children_per_map=2,
            max_iterations_per_loop=2,
            max_dynamic_depth=2,
            max_activation_occurrences=13,
        ),
    )
    operations = {
        expand_child: expand_operation,
        root_join: root_join_operation,
        starter: starter_operation,
        loop_member: member_operation,
        loop_join: join_operation,
    }
    capabilities = tuple(_capability(operation, index) for index, operation in enumerate(operations.values()))
    by_operation = dict(zip(operations, capabilities, strict=True))
    selections = tuple(
        ImplementationSelection(
            node=node,
            implementation=by_operation[node].implementation,
            configuration=by_operation[node].configuration,
        )
        for node in operations
    )
    prepared = prepare(
        data=_data(1),
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=44, max_entries=13, max_parent_depth=4),
        bound_inputs=(),
        configuration=PreparationConfiguration(
            purpose="execution_only", required_protection_outcomes=frozenset(), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=selections,
        capabilities=capabilities,
        limits=_limits(capabilities=5, slots=13),
    )
    context = admit_context_plan(prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=())
    implementations = {
        node: ExecutionImplementation(
            implementation=capability.implementation,
            configuration=capability.configuration,
            capability=capability,
            request=None,
        )
        for node, capability in by_operation.items()
    }
    policies = tuple(
        OperationExecutionPolicy(
            node=node,
            kind="local",
            request=None,
            safe_detachment="forbidden",
            implementations=(implementations[node],),
            result_outcomes=frozenset(item.name for item in operation.outcomes),
            runtime_outcomes=_rows(frozenset(item.name for item in operation.outcomes)),
        )
        for node, operation in operations.items()
    )
    admitted = admit_execution_plan(
        context=context,
        capabilities=capabilities,
        policies=policies,
        decisions=(),
        assessment_productions=(),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=20,
            max_provenance_edges=10,
        ),
        map_expansions=(
            MapExpansionDecl(
                expander=expander,
                outcome="expand",
                membership_port="items",
                item_type=text_type,
            ),
        ),
    )
    modes = {
        expand_child: "expand",
        root_join: "root_join",
        starter: "starter",
        loop_member: "loop",
        loop_join: "join",
    }
    callbacks = {
        node: _DynamicCallback(
            mode=mode, collection_type=collection_type, text_type=text_type, same_key_versions=same_key_versions
        )
        for node, mode in modes.items()
    }
    observed_items: list[str] = []
    observed_carries: list[str] = []
    callbacks[starter].seen = observed_items
    callbacks[loop_member].seen = observed_carries
    handles = tuple(
        ImplementationHandle(
            implementation=by_operation[node].implementation,
            operation=operation,
            configuration=by_operation[node].configuration,
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
                    max_runtime_artifact_bytes=300,
                    max_collection_items=2,
                ),
                decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
                clock=_Clock(),
            ),
        )
    ).wait()
    assert result.states[0].complete, [
        (
            {
                expander: "expander",
                expand_child: "expand-child",
                member: "member",
                root_join: "root-join",
                starter: "starter",
                loop_member: "loop-member",
                loop_join: "loop-join",
            }.get(entry.template, "unknown"),
            entry.activation.occurrence,
            entry.status,
            entry.outcome,
        )
        for entry in sorted(result.states[0].entries, key=lambda item: item.activation.occurrence)
    ]
    assert callbacks[expand_child].calls == 1
    assert callbacks[starter].calls == 2
    assert callbacks[loop_member].calls == 4
    assert callbacks[loop_join].calls == 2
    assert callbacks[root_join].calls == 1
    assert sorted(observed_items) == ["item-0", "item-1"]
    assert sorted(observed_carries) == ["item-0", "item-0:0", "item-1", "item-1:0"]
    map_items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    assert len(map_items) == 2
    assert {fact.key.item_key for fact in map_items if isinstance(fact.key, MapItemKey)} == (
        {7} if same_key_versions else {0, 1}
    )
    assert len({fact.artifact.key for fact in map_items}) == (1 if same_key_versions else 2)
    assert {fact.artifact.version for fact in map_items} == ({1, 2} if same_key_versions else {1})
    assert all(len(fact.parents) == 1 for fact in map_items)
    qualification = admit_qualification(
        execution=admitted, productions=(), limits=_qualification_limits(max_port_facts=32, max_provenance_edges=32)
    )
    latest = {ref.key: ref for ref, _ in sorted(result.artifacts, key=lambda pair: pair[0].version)}
    current = evidence_revision_view(
        admitted=qualification,
        result=result,
        artifacts=tuple(latest.values()),
        absences=(),
        configurations=(),
        state=admitted.context.prepared.state,
    )
    qualified = qualify(admitted=qualification, result=result, current=current, submissions=())
    assert all(status.qualification == "not_assessed" for status in qualified.record.statuses)
    assert all(target.withholding == frozenset({"execution_only"}) for target in qualified.targets), [
        target.withholding for target in qualified.targets
    ]
    assert len(result.states[0].entries) == 13
    assert len({item.activation for item in result.states[0].entries}) == 13
