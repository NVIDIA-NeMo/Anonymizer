# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parent text publication using the existing declared child recipe."""

from __future__ import annotations

from dataclasses import replace

from anonymizer.engine.graph_sdk.capabilities import ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, admit_execution_plan
from anonymizer.engine.graph_sdk.preparation import prepare
from anonymizer.graph.workflow import (
    DynamicScope,
    InputBinding,
    NodeInputRef,
    NodeOutputRef,
    WorkflowInputRef,
    admit_activation_workflow,
    admit_static_workflow,
)


def with_parent_publication(execution: AdmittedExecutionPlan) -> AdmittedExecutionPlan:
    prepared = execution.context.prepared
    base = prepared.workflow
    raw = base.workflow
    owner = next(item for item in execution.policies if item.delegated is not None)
    assessor = next(item for item in execution.policies if item.kind == "local")
    owner_op = execution.nodes[owner.node].operation
    assessor_op = execution.nodes[assessor.node].operation
    parent_op = replace(
        assessor_op,
        name="delegated-generation",
        outcomes=tuple(replace(item, produced_ports=frozenset({"context"})) for item in owner_op.outcomes),
    )
    assessor_op = replace(
        assessor_op,
        inputs=(*assessor_op.inputs, replace(assessor_op.inputs[0], name="source")),
        outputs=tuple(replace(port, name="assessment") for port in assessor_op.outputs),
        outcomes=tuple(
            replace(
                outcome,
                produced_ports=frozenset({"assessment"}),
                evidence=frozenset(
                    replace(promise, subject_port="input", consumed_ports=frozenset({"source"}))
                    for promise in outcome.evidence
                ),
            )
            for outcome in assessor_op.outcomes
        ),
        output_dependencies=tuple(
            replace(item, output="assessment", inputs=frozenset({"source"}), identity_input=None)
            for item in assessor_op.output_dependencies
        ),
    )
    nodes = tuple(
        replace(node, operation=parent_op if node.id == owner.node else assessor_op)
        if node.id in {owner.node, assessor.node}
        else node
        for node in raw.nodes
    )
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=raw.interface,
        nodes=nodes,
        input_bindings=(
            InputBinding(
                source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=owner.node, port="input")
            ),
            InputBinding(
                source=NodeOutputRef(node=owner.node, port="context"),
                destination=NodeInputRef(node=assessor.node, port="input"),
            ),
            InputBinding(
                source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=assessor.node, port="source")
            ),
        ),
        output_bindings=tuple(
            replace(binding, source=NodeOutputRef(node=owner.node, port="context")) for binding in raw.output_bindings
        ),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=tuple(raw.sequence),
        choices=(),
        protection=tuple(raw.protection_requirements),
        limits=replace(raw.limits, max_bindings=5),
    )
    workflow = admit_activation_workflow(
        workflow=static, scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),), limits=base.limits
    )
    by_node = {node.id: node.operation for node in nodes}
    capabilities = tuple(
        replace(implementation.capability, operation=by_node[policy.node])
        for policy in execution.policies
        for implementation in policy.implementations
    )
    updated = prepare(
        data=prepared.data,
        workflow=workflow,
        capabilities=capabilities,
        activation_limits=prepared.activation_limits,
        selections=tuple(
            ImplementationSelection(
                node=policy.node,
                implementation=policy.implementations[0].implementation,
                configuration=policy.implementations[0].configuration,
            )
            for policy in execution.policies
        ),
        limits=prepared.limits,
        configuration=prepared.configuration,
        bound_inputs=tuple(prepared.bound_inputs),
        state=prepared.state,
    )
    assert owner.delegated is not None
    leaves = {prepared.reservation_recipe[index].template for index in owner.delegated.child_slots}
    contract = replace(
        owner.delegated, child_slots=tuple(slot.index for slot in updated.reservation_recipe if slot.template in leaves)
    )
    policies = tuple(
        replace(
            policy,
            implementations=tuple(
                replace(
                    implementation,
                    capability=next(cap for cap in capabilities if cap.implementation == implementation.implementation),
                )
                for implementation in policy.implementations
            ),
            delegated=contract if policy.node == owner.node else None,
        )
        for policy in execution.policies
    )
    return admit_execution_plan(
        context=admit_context_plan(
            prepared=updated, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=policies,
        decisions=(),
        assessment_productions=tuple(
            replace(item, evidence_port="assessment") for item in execution.assessment_productions
        ),
        assessment_limits=execution.assessment_limits,
    )
