# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Boundary tests for pure bounded workflow activation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from anonymizer.graph._values import ActivationKey, ContractViolation, InvocationId, PlanId, ValidationCode
from anonymizer.graph.activation import (
    ActivationLimits,
    ActivationSeed,
    CloseUnstarted,
    ObserveMembership,
    ObserveTerminal,
    Select,
    Start,
    advance_activation,
    initialize_activation,
)
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    DynamicLimits,
    DynamicScope,
    KeyedJoinDecl,
    LoopDecl,
    MapDecl,
    NodeId,
    NodeOutcomeRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeClass,
    OutcomeSpec,
    ResourceCeiling,
    SequenceEdge,
    SubgraphNode,
    WorkflowId,
    WorkflowLimits,
    WorkflowOutcomeRef,
    admit_activation_workflow,
    admit_static_workflow,
)


def _outcome(name: str, category: OutcomeClass) -> OutcomeSpec:
    return OutcomeSpec(
        name=name,
        category=category,
        produced_ports=frozenset(),
        context=frozenset(),
        evidence=frozenset(),
        state_effects=frozenset(),
        model_requirements=frozenset(),
        ceiling=ResourceCeiling(max_activations=1, max_model_requests=0, max_input_bytes=0, max_output_bytes=0),
    )


def _interface_outcome(source: OutcomeSpec, activations: int) -> OutcomeSpec:
    return OutcomeSpec(
        name=source.name,
        category=source.category,
        produced_ports=source.produced_ports,
        context=source.context,
        evidence=source.evidence,
        state_effects=source.state_effects,
        model_requirements=source.model_requirements,
        ceiling=ResourceCeiling(
            max_activations=activations, max_model_requests=0, max_input_bytes=0, max_output_bytes=0
        ),
    )


def _single() -> tuple[AdmittedActivationWorkflow, NodeId]:
    owner = WorkflowId.new()
    node_id = NodeId.new(workflow=owner)
    outcomes = (_outcome("ok", "success"), _outcome("fail", "failure"))
    operation = OperationSpec(name="operation", inputs=(), outputs=(), output_dependencies=(), outcomes=outcomes)
    interface = OperationSpec(name="interface", inputs=(), outputs=(), output_dependencies=(), outcomes=outcomes)
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(OperationNode(id=node_id, operation=operation),),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node_id, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in outcomes
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
    dynamic = admit_activation_workflow(
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
    )
    return dynamic, node_id


def _initialized(*, max_events: int = 3):
    workflow, node = _single()
    invocation = InvocationId.new(plan=PlanId.new())
    key = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    seed = ActivationSeed(template=node, activation=key)
    state = initialize_activation(
        workflow=workflow,
        invocation=invocation,
        reservations=frozenset({seed}),
        limits=ActivationLimits(max_events=max_events, max_entries=1, max_parent_depth=1),
    )
    return state, seed


def test_exact_capacity_supports_select_start_terminal() -> None:
    state, seed = _initialized()
    state = advance_activation(state=state, event=Select(seeds=frozenset({seed})))
    assert next(iter(state.entries)).status == "ready"
    state = advance_activation(state=state, event=Start(activation=seed.activation))
    assert next(iter(state.entries)).status == "running"
    state = advance_activation(
        state=state,
        event=ObserveTerminal(activation=seed.activation, outcome="ok", category="success"),
    )
    assert state.events_applied == 3
    assert state.complete
    assert next(iter(state.entries)).status == "success"


@pytest.mark.parametrize("max_events", [0, 1, 2])
def test_initialization_rejects_insufficient_settlement_capacity(max_events: int) -> None:
    with pytest.raises(ContractViolation) as raised:
        _initialized(max_events=max_events)
    assert raised.value.code is ValidationCode.LIMIT_EXCEEDED


def test_unconditional_initialization_limit_precedes_foreign_owner() -> None:
    workflow, node = _single()
    invocation = InvocationId.new(plan=PlanId.new())
    foreign = InvocationId.new(plan=PlanId.new())
    key = ActivationKey(invocation=foreign, occurrence=0, parent=None, iteration=None)
    with pytest.raises(ContractViolation) as raised:
        initialize_activation(
            workflow=workflow,
            invocation=invocation,
            reservations=frozenset({ActivationSeed(template=node, activation=key)}),
            limits=ActivationLimits(max_events=0, max_entries=0, max_parent_depth=0),
        )
    assert raised.value.code is ValidationCode.LIMIT_EXCEEDED


def test_abnormal_terminal_has_no_semantic_outcome() -> None:
    state, seed = _initialized()
    state = advance_activation(state=state, event=Select(seeds=frozenset({seed})))
    state = advance_activation(state=state, event=Start(activation=seed.activation))
    state = advance_activation(
        state=state,
        event=ObserveTerminal(activation=seed.activation, outcome=None, category="lost"),
    )
    entry = next(iter(state.entries))
    assert (entry.status, entry.outcome) == ("lost", None)


def test_rejected_transition_preserves_frozen_input() -> None:
    state, seed = _initialized()
    with pytest.raises(ContractViolation) as raised:
        advance_activation(state=state, event=Start(activation=seed.activation))
    assert raised.value.code is ValidationCode.MISSING
    assert not state.entries and state.events_applied == 0
    with pytest.raises(FrozenInstanceError):
        state.events_applied = 1  # type: ignore[misc]


def test_unstarted_close_is_explicit_and_terminal() -> None:
    state, seed = _initialized()
    state = advance_activation(state=state, event=Select(seeds=frozenset({seed})))
    state = advance_activation(state=state, event=CloseUnstarted(activation=seed.activation, category="blocked"))
    assert next(iter(state.entries)).status == "blocked"


def test_closed_map_join_is_conjunctive_over_every_child() -> None:
    owner = WorkflowId.new()
    expander, member, join = (NodeId.new(workflow=owner) for _ in range(3))
    outcomes = (_outcome("ok", "success"), _outcome("fail", "failure"))
    operation = OperationSpec(name="node", inputs=(), outputs=(), output_dependencies=(), outcomes=outcomes)
    interface = OperationSpec(
        name="interface",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(_interface_outcome(outcome, 4) for outcome in outcomes),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=tuple(OperationNode(id=node, operation=operation) for node in (expander, member, join)),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=join, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in outcomes
        ),
        sequence=(SequenceEdge(before=expander, after=member), SequenceEdge(before=member, after=join)),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=3,
            max_bindings=2,
            max_sequence_edges=2,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    dynamic = admit_activation_workflow(
        workflow=static,
        scopes=(
            DynamicScope(
                workflow=static,
                maps=(
                    MapDecl(
                        expander=expander,
                        member=member,
                        expansion_outcomes=frozenset({"ok"}),
                        max_children=2,
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
            max_children_per_map=2,
            max_iterations_per_loop=0,
            max_dynamic_depth=1,
            max_activation_occurrences=4,
        ),
    )
    invocation = InvocationId.new(plan=PlanId.new())
    expander_key = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    join_key = ActivationKey(invocation=invocation, occurrence=1, parent=None, iteration=None)
    child_keys = tuple(
        ActivationKey(invocation=invocation, occurrence=index + 2, parent=expander_key, iteration=None)
        for index in range(2)
    )
    seeds = frozenset(
        {
            ActivationSeed(template=expander, activation=expander_key),
            ActivationSeed(template=join, activation=join_key),
            *(ActivationSeed(template=member, activation=key) for key in child_keys),
        }
    )
    fixed = frozenset(seed for seed in seeds if seed.template in {expander, join})

    duplicate_reservations = frozenset(
        {
            ActivationSeed(template=expander, activation=expander_key),
            ActivationSeed(template=join, activation=expander_key),
            *(ActivationSeed(template=member, activation=key) for key in child_keys),
        }
    )
    with pytest.raises(ContractViolation) as raised:
        initialize_activation(
            workflow=dynamic,
            invocation=invocation,
            reservations=duplicate_reservations,
            limits=ActivationLimits(max_events=13, max_entries=4, max_parent_depth=2),
        )
    assert raised.value.code is ValidationCode.DUPLICATE

    def initialized():
        return initialize_activation(
            workflow=dynamic,
            invocation=invocation,
            reservations=seeds,
            limits=ActivationLimits(max_events=13, max_entries=4, max_parent_depth=2),
        )

    failed_empty = advance_activation(state=initialized(), event=Select(seeds=fixed))
    failed_empty = advance_activation(state=failed_empty, event=Start(activation=expander_key))
    failed_empty = advance_activation(
        state=failed_empty,
        event=ObserveTerminal(activation=expander_key, outcome="fail", category="failure"),
    )
    assert next(iter(failed_empty.expansions)) == next(
        expansion for expansion in failed_empty.expansions if expansion.status == "failed" and not expansion.members
    )
    assert next(entry for entry in failed_empty.entries if entry.activation == join_key).status == "blocked"

    failed_partial = advance_activation(state=initialized(), event=Select(seeds=fixed))
    failed_partial = advance_activation(
        state=failed_partial,
        event=ObserveMembership(parent=expander_key, members=frozenset({child_keys[0]}), closed=False),
    )
    failed_partial = advance_activation(state=failed_partial, event=Start(activation=expander_key))
    failed_partial = advance_activation(
        state=failed_partial,
        event=ObserveTerminal(activation=expander_key, outcome=None, category="lost"),
    )
    partial_expansion = next(iter(failed_partial.expansions))
    assert (partial_expansion.status, partial_expansion.members) == ("failed", frozenset({child_keys[0]}))
    assert next(entry for entry in failed_partial.entries if entry.activation == join_key).status == "blocked"

    state = initialized()
    state = advance_activation(state=state, event=Select(seeds=fixed))
    state = advance_activation(state=state, event=Start(activation=expander_key))
    state = advance_activation(
        state=state, event=ObserveTerminal(activation=expander_key, outcome="ok", category="success")
    )
    state = advance_activation(
        state=state, event=ObserveMembership(parent=expander_key, members=frozenset(child_keys), closed=True)
    )
    for key, outcome, category in (
        (child_keys[0], "ok", "success"),
        (child_keys[1], "fail", "failure"),
    ):
        state = advance_activation(state=state, event=Start(activation=key))
        state = advance_activation(
            state=state,
            event=ObserveTerminal(activation=key, outcome=outcome, category=category),  # type: ignore[arg-type]
        )
    assert next(entry for entry in state.entries if entry.activation == join_key).status == "blocked"

    successful = initialized()
    successful = advance_activation(state=successful, event=Select(seeds=fixed))
    successful = advance_activation(state=successful, event=Start(activation=expander_key))
    successful = advance_activation(
        state=successful,
        event=ObserveTerminal(activation=expander_key, outcome="ok", category="success"),
    )
    successful = advance_activation(
        state=successful,
        event=ObserveMembership(parent=expander_key, members=frozenset(child_keys), closed=True),
    )
    for key in child_keys:
        successful = advance_activation(state=successful, event=Start(activation=key))
        successful = advance_activation(
            state=successful,
            event=ObserveTerminal(activation=key, outcome="ok", category="success"),
        )
    successful = advance_activation(state=successful, event=Start(activation=join_key))
    assert next(entry for entry in successful.entries if entry.activation == join_key).status == "running"
    successful = advance_activation(
        state=successful,
        event=ObserveTerminal(activation=join_key, outcome="ok", category="success"),
    )
    assert next(entry for entry in successful.entries if entry.activation == join_key).status == "success"


def _loop_workflow(bound: int):
    owner = WorkflowId.new()
    starter, member, join = (NodeId.new(workflow=owner) for _ in range(3))
    control = (_outcome("again", "success"), _outcome("stop", "success"))
    terminal = (_outcome("ok", "success"), _outcome("fail", "failure"))
    control_op = OperationSpec(name="control", inputs=(), outputs=(), output_dependencies=(), outcomes=control)
    terminal_op = OperationSpec(name="join", inputs=(), outputs=(), output_dependencies=(), outcomes=terminal)
    interface = OperationSpec(
        name="interface",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(_interface_outcome(outcome, max(3, bound + 2)) for outcome in terminal),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(
            OperationNode(id=starter, operation=control_op),
            OperationNode(id=member, operation=control_op),
            OperationNode(id=join, operation=terminal_op),
        ),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=join, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in terminal
        ),
        sequence=(SequenceEdge(before=starter, after=member), SequenceEdge(before=member, after=join)),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=3,
            max_bindings=2,
            max_sequence_edges=2,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    loop = LoopDecl(
        starter=starter,
        member=member,
        join=join,
        enter_outcomes=frozenset({"again"}),
        bypass_outcomes=frozenset({"stop"}),
        continue_outcomes=frozenset({"again"}),
        exit_outcomes=frozenset({"stop"}),
        initial=(),
        carried=(),
        max_iterations=bound,
    )
    dynamic = admit_activation_workflow(
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
                loops=(loop,),
            ),
        ),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=1,
            max_loops=1,
            max_children_per_map=0,
            max_iterations_per_loop=bound,
            max_dynamic_depth=1,
            max_activation_occurrences=bound + 2,
        ),
    )
    return dynamic, starter, member, join


def test_loop_admits_consecutive_iterations_and_releases_join_on_exit() -> None:
    workflow, starter, member, join = _loop_workflow(2)
    invocation = InvocationId.new(plan=PlanId.new())
    starter_key = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    join_key = ActivationKey(invocation=invocation, occurrence=1, parent=None, iteration=None)
    member_keys = tuple(
        ActivationKey(invocation=invocation, occurrence=index + 2, parent=starter_key, iteration=index)
        for index in range(2)
    )
    seeds = frozenset(
        {
            ActivationSeed(template=starter, activation=starter_key),
            ActivationSeed(template=join, activation=join_key),
            *(ActivationSeed(template=member, activation=key) for key in member_keys),
        }
    )
    state = initialize_activation(
        workflow=workflow,
        invocation=invocation,
        reservations=seeds,
        limits=ActivationLimits(max_events=12, max_entries=4, max_parent_depth=2),
    )
    state = advance_activation(
        state=state,
        event=Select(seeds=frozenset(seed for seed in seeds if seed.template in {starter, join})),
    )
    state = advance_activation(state=state, event=Start(activation=starter_key))
    state = advance_activation(
        state=state, event=ObserveTerminal(activation=starter_key, outcome="again", category="success")
    )
    for index, key in enumerate(member_keys):
        state = advance_activation(state=state, event=Start(activation=key))
        state = advance_activation(
            state=state,
            event=ObserveTerminal(
                activation=key,
                outcome="stop" if index == 1 else "again",
                category="success",
            ),
        )
    assert next(expansion for expansion in state.expansions if expansion.parent == starter_key).status == "closed"
    assert next(entry for entry in state.entries if entry.activation == join_key).status == "ready"


def test_loop_enter_at_zero_closes_overflow_without_member() -> None:
    workflow, starter, _, join = _loop_workflow(0)
    invocation = InvocationId.new(plan=PlanId.new())
    starter_key = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    join_key = ActivationKey(invocation=invocation, occurrence=1, parent=None, iteration=None)
    seeds = frozenset(
        {
            ActivationSeed(template=starter, activation=starter_key),
            ActivationSeed(template=join, activation=join_key),
        }
    )
    state = initialize_activation(
        workflow=workflow,
        invocation=invocation,
        reservations=seeds,
        limits=ActivationLimits(max_events=6, max_entries=2, max_parent_depth=1),
    )
    state = advance_activation(state=state, event=Select(seeds=seeds))
    state = advance_activation(state=state, event=Start(activation=starter_key))
    state = advance_activation(
        state=state, event=ObserveTerminal(activation=starter_key, outcome="again", category="success")
    )
    assert next(iter(state.expansions)).status == "overflow"
    assert next(entry for entry in state.entries if entry.activation == join_key).status == "inconsistent"


def test_subgraph_start_materializes_body_and_derives_parent_terminal() -> None:
    child_owner = WorkflowId.new()
    child_node = NodeId.new(workflow=child_owner)
    outcomes = (_outcome("ok", "success"), _outcome("fail", "failure"))
    child_operation = OperationSpec(name="child", inputs=(), outputs=(), output_dependencies=(), outcomes=outcomes)
    child_interface = OperationSpec(
        name="child-interface",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(_interface_outcome(outcome, 1) for outcome in outcomes),
    )
    body = admit_static_workflow(
        workflow=child_owner,
        interface=child_interface,
        nodes=(OperationNode(id=child_node, operation=child_operation),),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=child_node, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in outcomes
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
    root_owner = WorkflowId.new()
    parent_node = NodeId.new(workflow=root_owner)
    parent_operation = OperationSpec(
        name="parent",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(_interface_outcome(outcome, 1) for outcome in outcomes),
    )
    root_interface = OperationSpec(
        name="root-interface",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(_interface_outcome(outcome, 2) for outcome in outcomes),
    )
    root = admit_static_workflow(
        workflow=root_owner,
        interface=root_interface,
        nodes=(SubgraphNode(id=parent_node, operation=parent_operation, body=body),),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=parent_node, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in outcomes
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=2,
            max_bindings=2,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
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
    )
    invocation = InvocationId.new(plan=PlanId.new())
    parent_key = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    child_key = ActivationKey(invocation=invocation, occurrence=1, parent=parent_key, iteration=None)
    parent_seed = ActivationSeed(template=parent_node, activation=parent_key)
    child_seed = ActivationSeed(template=child_node, activation=child_key)
    state = initialize_activation(
        workflow=workflow,
        invocation=invocation,
        reservations=frozenset({parent_seed, child_seed}),
        limits=ActivationLimits(max_events=6, max_entries=2, max_parent_depth=2),
    )
    state = advance_activation(state=state, event=Select(seeds=frozenset({parent_seed})))
    state = advance_activation(state=state, event=Start(activation=parent_key))
    assert next(entry for entry in state.entries if entry.activation == child_key).status == "ready"
    with pytest.raises(ContractViolation) as raised:
        advance_activation(
            state=state,
            event=ObserveTerminal(activation=parent_key, outcome="ok", category="success"),
        )
    assert raised.value.code is ValidationCode.CONTRADICTORY
    state = advance_activation(state=state, event=Start(activation=child_key))
    state = advance_activation(
        state=state, event=ObserveTerminal(activation=child_key, outcome="ok", category="success")
    )
    parent = next(entry for entry in state.entries if entry.activation == parent_key)
    assert (parent.status, parent.outcome, state.complete) == ("success", "ok", True)


def test_initialization_allows_shared_body_under_distinct_parent_templates() -> None:
    body_workflow, child = _single()
    body = body_workflow.workflow
    owner = WorkflowId.new()
    first_parent, second_parent = (NodeId.new(workflow=owner) for _ in range(2))
    interface = OperationSpec(
        name="shared-body-root",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(_interface_outcome(outcome, 4) for outcome in body.interface.outcomes),
    )
    root = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(
            SubgraphNode(id=first_parent, operation=body.interface, body=body),
            SubgraphNode(id=second_parent, operation=body.interface, body=body),
        ),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=second_parent, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in body.interface.outcomes
        ),
        sequence=(SequenceEdge(before=first_parent, after=second_parent),),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=4,
            max_bindings=2,
            max_sequence_edges=1,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
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
            max_activation_occurrences=4,
        ),
    )
    invocation = InvocationId.new(plan=PlanId.new())
    first_key = ActivationKey(invocation=invocation, occurrence=0, parent=None, iteration=None)
    second_key = ActivationKey(invocation=invocation, occurrence=1, parent=None, iteration=None)
    first_child = ActivationKey(invocation=invocation, occurrence=2, parent=first_key, iteration=None)
    second_child = ActivationKey(invocation=invocation, occurrence=3, parent=second_key, iteration=None)
    reservations = frozenset(
        {
            ActivationSeed(template=first_parent, activation=first_key),
            ActivationSeed(template=second_parent, activation=second_key),
            ActivationSeed(template=child, activation=first_child),
            ActivationSeed(template=child, activation=second_child),
        }
    )
    state = initialize_activation(
        workflow=workflow,
        invocation=invocation,
        reservations=reservations,
        limits=ActivationLimits(max_events=12, max_entries=4, max_parent_depth=2),
    )
    assert not state.entries


def test_nested_two_by_two_map_loop_reserves_twelve_distinct_keys() -> None:
    body_workflow, loop_starter, loop_member, loop_join = _loop_workflow(2)
    body = body_workflow.workflow
    owner = WorkflowId.new()
    expander, member, join = (NodeId.new(workflow=owner) for _ in range(3))
    outer_outcomes = (_outcome("ok", "success"), _outcome("fail", "failure"))
    outer_operation = OperationSpec(
        name="outer", inputs=(), outputs=(), output_dependencies=(), outcomes=outer_outcomes
    )
    interface = OperationSpec(
        name="interface",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(_interface_outcome(outcome, 12) for outcome in outer_outcomes),
    )
    static = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(
            OperationNode(id=expander, operation=outer_operation),
            SubgraphNode(id=member, operation=body.interface, body=body),
            OperationNode(id=join, operation=outer_operation),
        ),
        input_bindings=(),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=join, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in outer_outcomes
        ),
        sequence=(SequenceEdge(before=expander, after=member), SequenceEdge(before=member, after=join)),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=7,
            max_bindings=2,
            max_sequence_edges=2,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=2,
            max_choice_states=1,
        ),
    )
    root_scope = DynamicScope(
        workflow=static,
        maps=(
            MapDecl(
                expander=expander,
                member=member,
                expansion_outcomes=frozenset({"ok"}),
                max_children=2,
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
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(root_scope, body_workflow.scopes[0]),
        limits=DynamicLimits(
            max_maps=1,
            max_joins=2,
            max_loops=1,
            max_children_per_map=2,
            max_iterations_per_loop=2,
            max_dynamic_depth=2,
            max_activation_occurrences=12,
        ),
    )
    assert workflow.activation_upper_bound == 12

    invocation = InvocationId.new(plan=PlanId.new())
    occurrence = 0

    def key(*, parent: ActivationKey | None, iteration: int | None = None) -> ActivationKey:
        nonlocal occurrence
        result = ActivationKey(
            invocation=invocation,
            occurrence=occurrence,
            parent=parent,
            iteration=iteration,
        )
        occurrence += 1
        return result

    expander_key = key(parent=None)
    seeds = {
        ActivationSeed(template=expander, activation=expander_key),
        ActivationSeed(template=join, activation=key(parent=None)),
    }
    for _ in range(2):
        member_key = key(parent=expander_key)
        starter_key = key(parent=member_key)
        seeds.update(
            {
                ActivationSeed(template=member, activation=member_key),
                ActivationSeed(template=loop_starter, activation=starter_key),
                ActivationSeed(template=loop_join, activation=key(parent=member_key)),
                ActivationSeed(template=loop_member, activation=key(parent=starter_key, iteration=0)),
                ActivationSeed(template=loop_member, activation=key(parent=starter_key, iteration=1)),
            }
        )
    state = initialize_activation(
        workflow=workflow,
        invocation=invocation,
        reservations=frozenset(seeds),
        limits=ActivationLimits(max_events=37, max_entries=12, max_parent_depth=4),
    )
    assert len(state.reservations) == 12
    with pytest.raises(ContractViolation) as raised:
        initialize_activation(
            workflow=workflow,
            invocation=invocation,
            reservations=frozenset(seeds),
            limits=ActivationLimits(max_events=37, max_entries=12, max_parent_depth=3),
        )
    assert raised.value.code is ValidationCode.LIMIT_EXCEEDED
