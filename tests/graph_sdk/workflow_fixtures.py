# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named graph topologies shared by activation, preparation and execution tests."""

from __future__ import annotations

from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    AdmittedWorkflow,
    ContextInputRef,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutputBinding,
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


def _shared_body_workflow(
    body: AdmittedWorkflow,
    interface: OperationSpec,
    *,
    max_bindings: int,
    input_binding: tuple[str, str] | None = None,
) -> tuple[AdmittedActivationWorkflow, NodeId, NodeId]:
    owner = WorkflowId.new()
    first, second = (NodeId.new(workflow=owner) for _ in range(2))
    root = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=tuple(SubgraphNode(id=parent, operation=body.interface, body=body) for parent in (first, second)),
        input_bindings=tuple(
            InputBinding(
                source=WorkflowInputRef(port=input_binding[0]),
                destination=NodeInputRef(node=parent, port=input_binding[1]),
            )
            for parent in (first, second)
        )
        if input_binding is not None
        else (),
        output_bindings=(),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=second, outcome=outcome.name),
                destination=WorkflowOutcomeRef(outcome=outcome.name),
            )
            for outcome in body.interface.outcomes
        ),
        sequence=(SequenceEdge(before=first, after=second),),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=4,
            max_bindings=max_bindings,
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
    return workflow, first, second


def _single_node_workflow(
    *,
    workflow: WorkflowId,
    interface: OperationSpec,
    node: OperationNode | SubgraphNode,
    inputs: tuple[WorkflowInputRef | ContextInputRef, ...],
    outputs: tuple[str, ...],
    outcomes: tuple[str, ...],
    limits: WorkflowLimits,
) -> AdmittedWorkflow:
    """Expose one node's named ports and outcomes without edges or protection."""
    return admit_static_workflow(
        workflow=workflow,
        interface=interface,
        nodes=(node,),
        input_bindings=tuple(
            InputBinding(source=source, destination=NodeInputRef(node=node.id, port=source.port)) for source in inputs
        ),
        output_bindings=tuple(
            OutputBinding(source=NodeOutputRef(node=node.id, port=port), destination=WorkflowOutputRef(port=port))
            for port in outputs
        ),
        outcome_bindings=tuple(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node.id, outcome=outcome), destination=WorkflowOutcomeRef(outcome=outcome)
            )
            for outcome in outcomes
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=limits,
    )
