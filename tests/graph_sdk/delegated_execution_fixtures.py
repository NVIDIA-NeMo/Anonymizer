# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parent text publication using the existing declared child recipe."""

from __future__ import annotations

from dataclasses import replace

from anonymizer.engine.graph_sdk.capabilities import ImplementationRef, ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, DelegatedReadiness, admit_execution_plan
from anonymizer.engine.graph_sdk.preparation import prepare
from anonymizer.engine.graph_sdk.requests import ChildRoute
from anonymizer.graph.workflow import (
    DynamicScope,
    InputBinding,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    SequenceEdge,
    WorkflowInputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows


def with_parent_publication(execution: AdmittedExecutionPlan, *, readiness: bool = False) -> AdmittedExecutionPlan:
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
    source_policies = tuple(execution.policies)
    sequence = tuple(raw.sequence)
    interface = raw.interface
    readiness_nodes = ()
    if readiness:
        physical = next(item for item in execution.policies if item.kind == "external")
        empty = execution.nodes[physical.node].operation
        readiness_nodes = tuple(NodeId.new(workflow=raw.workflow) for _ in range(3))
        additions = []
        for index, node in enumerate(readiness_nodes):
            local = index == 2
            names = ("available", "async_required") if local else ("ready",)
            operation = replace(
                empty,
                name="probe" if local else "readiness",
                outcomes=tuple(
                    replace(
                        empty.outcomes[0],
                        name=name,
                        ceiling=replace(empty.outcomes[0].ceiling, max_model_requests=0 if local else 1),
                    )
                    for name in names
                ),
            )
            source = assessor if local else physical
            impl = source.implementations[0]
            impl = replace(
                impl,
                implementation=ImplementationRef(name=f"readiness-{index}", revision=1),
                capability=replace(
                    impl.capability,
                    implementation=ImplementationRef(name=f"readiness-{index}", revision=1),
                    operation=operation,
                    usage="exact",
                ),
            )
            additions.append(
                replace(
                    source,
                    node=node,
                    implementations=(impl,),
                    result_outcomes=frozenset(names),
                    runtime_outcomes=_valid_runtime_rows("local" if local else "external", frozenset(names)),
                )
            )
            nodes += (OperationNode(id=node, operation=operation),)
        source_policies += tuple(additions)
        semantic, _, probe = readiness_nodes
        sequence += (
            SequenceEdge(before=semantic, after=probe),
            SequenceEdge(before=probe, after=physical.node),
            *(SequenceEdge(before=node, after=assessor.node) for node in readiness_nodes),
        )
        interface = replace(
            interface,
            outcomes=tuple(
                replace(
                    outcome,
                    ceiling=replace(
                        outcome.ceiling,
                        max_activations=len(nodes),
                        max_model_requests=outcome.ceiling.max_model_requests + 2,
                        max_input_bytes=100 * len(nodes),
                        max_output_bytes=100 * len(nodes),
                    ),
                )
                for outcome in interface.outcomes
            ),
        )
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=interface,
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
        sequence=sequence,
        choices=(),
        protection=tuple(raw.protection_requirements),
        limits=replace(raw.limits, max_bindings=5, max_nodes=len(nodes), max_sequence_edges=len(sequence)),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),),
        limits=replace(base.limits, max_activation_occurrences=len(nodes)),
    )
    by_node = {node.id: node.operation for node in nodes}
    capabilities = tuple(
        replace(implementation.capability, operation=by_node[policy.node])
        for policy in source_policies
        for implementation in policy.implementations
    )
    updated = prepare(
        data=prepared.data,
        workflow=workflow,
        capabilities=capabilities,
        activation_limits=replace(prepared.activation_limits, max_entries=len(nodes), max_events=3 * len(nodes)),
        selections=tuple(
            ImplementationSelection(
                node=policy.node,
                implementation=policy.implementations[0].implementation,
                configuration=policy.implementations[0].configuration,
            )
            for policy in source_policies
        ),
        limits=replace(prepared.limits, max_capabilities=len(capabilities)),
        configuration=prepared.configuration,
        bound_inputs=tuple(prepared.bound_inputs),
        state=prepared.state,
    )
    assert owner.delegated is not None
    leaves = tuple(prepared.reservation_recipe[index].template for index in owner.delegated.child_slots)
    indices = {slot.template: slot.index for slot in updated.reservation_recipe}
    contract = replace(owner.delegated, child_slots=tuple(indices[node] for node in leaves))
    if readiness:
        indices = {slot.template: slot.index for slot in updated.reservation_recipe}
        contract = replace(
            contract,
            readiness=DelegatedReadiness(
                owner_slot=indices[readiness_nodes[0]],
                request_slot=indices[readiness_nodes[1]],
                probe_slot=indices[readiness_nodes[2]],
                route=ChildRoute(name="health", parse_attempts=1),
                ready_outcome="ready",
                probe_outcomes=frozenset({"available", "async_required"}),
            ),
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
        for policy in source_policies
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


def with_multiple_sources(execution: AdmittedExecutionPlan, count: int = 2) -> AdmittedExecutionPlan:
    """Reuse the admitted operation recipe for distinct source occurrences."""
    from tests.graph_sdk.test_preparation import _data

    prepared = execution.context.prepared
    data = _data(count)
    original = next(iter(prepared.data.targets))
    updated = prepare(
        data=data,
        workflow=prepared.workflow,
        capabilities=execution.capabilities,
        activation_limits=prepared.activation_limits,
        selections=tuple(
            ImplementationSelection(
                node=item.node,
                implementation=item.capability.implementation,
                configuration=item.capability.configuration,
            )
            for item in prepared.implementations
        ),
        limits=replace(
            prepared.limits,
            max_bound_inputs=count * len(prepared.bound_inputs),
            max_total_activation_slots=count * prepared.total_activation_slots,
        ),
        configuration=replace(prepared.configuration, hard_request_limit=None),
        bound_inputs=tuple(
            replace(item, target=target, source=target if item.source == original else item.source)
            for target in data.targets
            for item in prepared.bound_inputs
        ),
        state=prepared.state,
    )
    return admit_execution_plan(
        context=admit_context_plan(
            prepared=updated, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=execution.capabilities,
        policies=tuple(execution.policies),
        decisions=(),
        assessment_productions=execution.assessment_productions,
        assessment_limits=replace(
            execution.assessment_limits,
            max_assessment_facts=count * execution.assessment_limits.max_assessment_facts,
            max_port_facts=count * execution.assessment_limits.max_port_facts,
            max_provenance_edges=count * execution.assessment_limits.max_provenance_edges,
        ),
    )
