# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static and dynamic workflow admission and substitution."""

from __future__ import annotations

from typing import Any

from anonymizer.graph._values import ValidationCode
from anonymizer.graph._workflow_composition import _has_cycle, _validate_paths
from anonymizer.graph._workflow_values import (
    _ADMISSION_KEY,
    AdmittedActivationWorkflow,
    AdmittedWorkflow,
    ArtifactType,
    ChoiceDecl,
    ContextInputRef,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    MapItemPort,
    Node,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutputBinding,
    ProtectionRequirement,
    SequenceEdge,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    _reject,
    _tuple_of,
)


def admit_activation_workflow(
    *, workflow: AdmittedWorkflow, scopes: tuple[DynamicScope, ...], limits: DynamicLimits
) -> AdmittedActivationWorkflow:
    """Validate dynamic roles and compute a finite occurrence bound."""
    if not isinstance(workflow, AdmittedWorkflow) or not isinstance(limits, DynamicLimits):
        _reject(ValidationCode.INVALID_TYPE)
    _tuple_of(scopes, DynamicScope)

    reachable: list[AdmittedWorkflow] = []
    pending = [workflow]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        reachable.append(current)
        pending.extend(node.body for node in current.nodes if isinstance(node, SubgraphNode))
    by_workflow = {id(scope.workflow): scope for scope in scopes}
    if len(by_workflow) != len(scopes):
        _reject(ValidationCode.DUPLICATE)
    if len(scopes) != len(reachable) or set(by_workflow) != {id(item) for item in reachable}:
        _reject(ValidationCode.MISSING)
    for body in reachable:
        for endpoint in body.map_item_ports:
            matches = [
                item
                for scope in scopes
                for item in scope.maps
                if item.expander == endpoint.expander
                and item.member == endpoint.member
                and item.item_input == endpoint.item_input
                and endpoint.expansion_outcome in item.expansion_outcomes
            ]
            if len(matches) != 1:
                _reject(ValidationCode.MISSING if not matches else ValidationCode.DUPLICATE)

    total_maps = sum(len(scope.maps) for scope in scopes)
    total_joins = sum(len(scope.joins) for scope in scopes)
    total_loops = sum(len(scope.loops) for scope in scopes)
    if (
        total_maps > limits.max_maps
        or total_joins > limits.max_joins
        or total_loops > limits.max_loops
        or any(item.max_children > limits.max_children_per_map for scope in scopes for item in scope.maps)
        or any(item.max_iterations > limits.max_iterations_per_loop for scope in scopes for item in scope.loops)
    ):
        _reject(ValidationCode.LIMIT_EXCEEDED)

    for scope in scopes:
        nodes = {node.id: node for node in scope.workflow.nodes}
        owned = set(nodes)
        declared_joins = [join.join for join in scope.joins]
        aggregate_sources = [item.expander for item in scope.maps] + [item.starter for item in scope.loops]
        sources = set(aggregate_sources)
        members = [item.member for item in scope.maps] + [item.member for item in scope.loops]
        aggregate_joins = declared_joins
        referenced = (
            {node for item in scope.maps for node in (item.expander, item.member)}
            | {node for item in scope.joins for node in (item.source, item.join)}
            | {node for item in scope.loops for node in (item.starter, item.member, item.join)}
        )
        if any(node.workflow != scope.workflow.workflow for node in referenced):
            _reject(ValidationCode.FOREIGN_OWNER)
        if (
            len(members) != len(set(members))
            or len(aggregate_joins) != len(set(aggregate_joins))
            or len(aggregate_sources) != len(sources)
        ):
            _reject(ValidationCode.DUPLICATE)
        if not referenced <= owned:
            _reject(ValidationCode.MISSING)
        join_sources = {item.source for item in scope.joins}
        if join_sources != sources or len(join_sources) != len(scope.joins):
            _reject(ValidationCode.MISSING)
        choice_members = {
            member for choice in scope.workflow.choices for branch in choice.branches for member in branch.members
        }
        if choice_members & (set(members) | set(aggregate_joins)):
            _reject(ValidationCode.OVERLAP)
        outcomes = {node_id: {outcome.name for outcome in node.operation.outcomes} for node_id, node in nodes.items()}
        if any(not item.expansion_outcomes <= outcomes[item.expander] for item in scope.maps):
            _reject(ValidationCode.MISSING)
        for item in scope.maps:
            if item.max_children > 1:
                outward_inputs = any(
                    isinstance(binding.source, NodeOutputRef)
                    and binding.source.node == item.member
                    and binding.destination.node != item.member
                    for binding in scope.workflow.input_bindings
                )
                outward_outputs = any(
                    isinstance(binding.source, NodeOutputRef) and binding.source.node == item.member
                    for binding in scope.workflow.output_bindings
                )
                if outward_inputs or outward_outputs:
                    _reject(ValidationCode.UNSUPPORTED)
            if item.item_input is None:
                continue
            member_inputs = {port.name for port in nodes[item.member].operation.inputs}
            if item.item_input not in member_inputs:
                _reject(ValidationCode.MISSING)
            bindings = [
                binding
                for binding in scope.workflow.input_bindings
                if binding.destination == NodeInputRef(node=item.member, port=item.item_input)
            ]
            if len(bindings) != 1:
                _reject(ValidationCode.MISSING)
        for item in scope.loops:
            if item.enter_outcomes | item.bypass_outcomes != outcomes[item.starter]:
                _reject(ValidationCode.MISSING)
            if item.continue_outcomes | item.exit_outcomes != outcomes[item.member]:
                _reject(ValidationCode.MISSING)
            member_operation = nodes[item.member].operation
            input_names = {port.name for port in member_operation.inputs}
            initial_destinations = [binding.destination.port for binding in item.initial]
            carried_destinations = [binding.destination.port for binding in item.carried]
            if any(binding.destination.node != item.member for binding in (*item.initial, *item.carried)):
                _reject(ValidationCode.FOREIGN_OWNER)
            if len(initial_destinations) != len(set(initial_destinations)) or len(carried_destinations) != len(
                set(carried_destinations)
            ):
                _reject(ValidationCode.DUPLICATE)
            if set(initial_destinations) != input_names or set(carried_destinations) != input_names:
                _reject(ValidationCode.MISSING)
            static_bindings = {
                (binding.source, binding.destination)
                for binding in scope.workflow.input_bindings
                if binding.destination.node == item.member
            }
            if {(binding.source, binding.destination) for binding in item.initial} != static_bindings:
                _reject(ValidationCode.CONTRADICTORY)
            output_types = {port.name: port.artifact_type for port in member_operation.outputs}
            input_types = {port.name: port.artifact_type for port in member_operation.inputs}
            if any(
                binding.source.node != item.member
                or binding.source.port not in output_types
                or output_types[binding.source.port] != input_types[binding.destination.port]
                for binding in item.carried
            ):
                _reject(ValidationCode.CONTRADICTORY)
        joins_by_source = {item.source: item for item in scope.joins}
        if any(joins_by_source[item.starter].join != item.join for item in scope.loops):
            _reject(ValidationCode.CONTRADICTORY)
        edges = {(edge.before, edge.after) for edge in scope.workflow.sequence}
        reachable_pairs = set(edges)
        changed = True
        while changed:
            changed = False
            additions = {
                (left, right)
                for left, middle in reachable_pairs
                for candidate, right in reachable_pairs
                if middle == candidate and left != right
            } - reachable_pairs
            if additions:
                reachable_pairs.update(additions)
                changed = True
        if any(
            (item.expander, item.member) not in reachable_pairs
            or (item.member, next(join.join for join in scope.joins if join.source == item.expander))
            not in reachable_pairs
            for item in scope.maps
        ) or any(
            (item.starter, item.member) not in reachable_pairs or (item.member, item.join) not in reachable_pairs
            for item in scope.loops
        ):
            _reject(ValidationCode.CONTRADICTORY)

    scope_depth: dict[int, int] = {id(workflow): 1}
    pending_depth = [workflow]
    while pending_depth:
        current = pending_depth.pop()
        depth = scope_depth[id(current)]
        for node in current.nodes:
            if isinstance(node, SubgraphNode):
                scope_depth[id(node.body)] = depth + 1
                pending_depth.append(node.body)
    dynamic_depth = max(scope_depth.values(), default=1)
    if dynamic_depth > limits.max_dynamic_depth:
        _reject(ValidationCode.LIMIT_EXCEEDED)

    bounds: dict[int, int] = {}
    stack: list[tuple[AdmittedWorkflow, bool]] = [(workflow, False)]
    while stack:
        current, visited = stack.pop()
        if id(current) in bounds:
            continue
        children = [node.body for node in current.nodes if isinstance(node, SubgraphNode)]
        if not visited:
            stack.append((current, True))
            stack.extend((child, False) for child in children)
            continue
        scope = by_workflow[id(current)]
        factors = {item.member: item.max_children for item in scope.maps}
        factors.update({item.member: item.max_iterations for item in scope.loops})
        bound = 0
        for node in current.nodes:
            node_bound = 1 + (bounds[id(node.body)] if isinstance(node, SubgraphNode) else 0)
            bound += factors.get(node.id, 1) * node_bound
            if bound > limits.max_activation_occurrences:
                _reject(ValidationCode.LIMIT_EXCEEDED)
        bounds[id(current)] = bound
    upper_bound = bounds[id(workflow)]
    if upper_bound > limits.max_activation_occurrences:
        _reject(ValidationCode.LIMIT_EXCEEDED)
    if any(
        outcome.ceiling.max_activations < bounds[id(item)] for item in reachable for outcome in item.interface.outcomes
    ):
        _reject(ValidationCode.CONTRADICTORY)
    ordered = tuple(by_workflow[id(item)] for item in reachable)
    return AdmittedActivationWorkflow(
        _key=_ADMISSION_KEY,
        workflow=workflow,
        scopes=ordered,
        limits=limits,
        activation_upper_bound=upper_bound,
        dynamic_depth=dynamic_depth,
    )


def substitute_activation_workflow(
    *, workflow: AdmittedActivationWorkflow, target: NodeId, replacement: AdmittedWorkflow
) -> AdmittedActivationWorkflow:
    """Substitute a static body and readmit the dynamic wrapper."""
    if not isinstance(workflow, AdmittedActivationWorkflow):
        _reject(ValidationCode.INVALID_TYPE)
    updated = substitute(workflow=workflow.workflow, target=target, replacement=replacement)
    root_scope = next(scope for scope in workflow.scopes if scope.workflow is workflow.workflow)
    admitted_scopes = [
        DynamicScope(workflow=updated, maps=root_scope.maps, joins=root_scope.joins, loops=root_scope.loops)
    ]
    existing = {id(scope.workflow): scope for scope in workflow.scopes}
    pending = [node.body for node in updated.nodes if isinstance(node, SubgraphNode)]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        prior = existing.get(id(current))
        admitted_scopes.append(
            prior if prior is not None else DynamicScope(workflow=current, maps=(), joins=(), loops=())
        )
        pending.extend(node.body for node in current.nodes if isinstance(node, SubgraphNode))
    return admit_activation_workflow(
        workflow=updated,
        scopes=tuple(admitted_scopes),
        limits=workflow.limits,
    )


def admit_static_workflow(
    *,
    workflow: WorkflowId,
    interface: OperationSpec,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    protection: tuple[ProtectionRequirement, ...],
    limits: WorkflowLimits,
) -> AdmittedWorkflow:
    """Validate and normalize a pure static workflow declaration."""
    _validate_admission_types(
        workflow,
        interface,
        nodes,
        input_bindings,
        output_bindings,
        outcome_bindings,
        sequence,
        choices,
        protection,
        limits,
    )
    if any(choice.selector in branch.members for choice in choices for branch in choice.branches):
        _reject(ValidationCode.INVALID_VALUE)
    expanded_count = sum(1 + (node.body.expanded_node_count if isinstance(node, SubgraphNode) else 0) for node in nodes)
    depth = max((1 + _workflow_depth(node.body) if isinstance(node, SubgraphNode) else 1 for node in nodes), default=1)
    choice_states = 1
    node_operations = {node.id: node.operation for node in nodes}
    for choice in choices:
        operation = node_operations.get(choice.selector)
        choice_states *= len(operation.outcomes) if operation is not None else 1
    raw_bindings = len(input_bindings) + len(output_bindings) + len(outcome_bindings)
    raw_bindings += len(
        {
            port
            for declaration in (
                *(promise for outcome in interface.outcomes for promise in outcome.evidence),
                *protection,
            )
            for port in (declaration.subject_port, *declaration.consumed_ports)
            if isinstance(port, MapItemPort)
        }
    )
    branch_members = sum(len(branch.members) for choice in choices for branch in choice.branches)
    if (
        len(nodes) > limits.max_nodes
        or raw_bindings > limits.max_bindings
        or len(sequence) > limits.max_sequence_edges
        or len(choices) > limits.max_choices
        or branch_members > limits.max_branch_members
        or expanded_count > limits.max_nodes
        or depth > limits.max_subgraph_depth
        or choice_states > limits.max_choice_states
    ):
        _reject(ValidationCode.LIMIT_EXCEEDED)
    _validate_owners(workflow, nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices)
    _validate_duplicates(nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices, protection)
    incompatible_binding = _validate_references(
        interface, nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices, protection
    )
    _validate_choice_overlaps(choices)
    _validate_cycles(nodes, sequence)
    _validate_choice_reachability(sequence, choices)
    for node in nodes:
        if isinstance(node, SubgraphNode):
            if not _compatible(node.operation, node.body.interface, allow_narrower=True):
                _reject(ValidationCode.CONTRADICTORY)
    if incompatible_binding:
        _reject(ValidationCode.CONTRADICTORY)
    _validate_paths(
        interface, nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices, check_cycles=False
    )
    eligible, unmet = _protection(interface, protection)
    return AdmittedWorkflow(
        _key=_ADMISSION_KEY,
        workflow=workflow,
        interface=interface,
        nodes=frozenset(nodes),
        input_bindings=frozenset(input_bindings),
        output_bindings=frozenset(output_bindings),
        outcome_bindings=frozenset(outcome_bindings),
        sequence=frozenset(sequence),
        choices=frozenset(choices),
        protection_requirements=frozenset(protection),
        protection_eligible_outcomes=eligible,
        unmet_protection=unmet,
        limits=limits,
        expanded_node_count=expanded_count,
    )


def substitute(*, workflow: AdmittedWorkflow, target: NodeId, replacement: AdmittedWorkflow) -> AdmittedWorkflow:
    """Replace one operation with a compatible, separately owned workflow."""
    if (
        not isinstance(workflow, AdmittedWorkflow)
        or not isinstance(target, NodeId)
        or not isinstance(replacement, AdmittedWorkflow)
    ):
        _reject(ValidationCode.INVALID_TYPE)
    if target.workflow != workflow.workflow:
        _reject(ValidationCode.FOREIGN_OWNER)
    if replacement.workflow == workflow.workflow:
        _reject(ValidationCode.FOREIGN_OWNER)
    target_node = next((node for node in workflow.nodes if node.id == target), None)
    if target_node is None:
        _reject(ValidationCode.MISSING)
    if not _compatible(target_node.operation, replacement.interface, allow_narrower=True):
        _reject(ValidationCode.CONTRADICTORY)
    nodes = tuple(
        SubgraphNode(id=target, operation=target_node.operation, body=replacement) if node.id == target else node
        for node in workflow.nodes
    )
    return admit_static_workflow(
        workflow=workflow.workflow,
        interface=workflow.interface,
        nodes=nodes,
        input_bindings=tuple(workflow.input_bindings),
        output_bindings=tuple(workflow.output_bindings),
        outcome_bindings=tuple(workflow.outcome_bindings),
        sequence=tuple(workflow.sequence),
        choices=tuple(workflow.choices),
        protection=tuple(workflow.protection_requirements),
        limits=workflow.limits,
    )


def _validate_admission_types(
    workflow: object,
    interface: object,
    nodes: object,
    input_bindings: object,
    output_bindings: object,
    outcome_bindings: object,
    sequence: object,
    choices: object,
    protection: object,
    limits: object,
) -> None:
    if (
        not isinstance(workflow, WorkflowId)
        or not isinstance(interface, OperationSpec)
        or not isinstance(limits, WorkflowLimits)
    ):
        _reject(ValidationCode.INVALID_TYPE)
    _tuple_of(nodes, (OperationNode, SubgraphNode))
    _tuple_of(input_bindings, InputBinding)
    _tuple_of(output_bindings, OutputBinding)
    _tuple_of(outcome_bindings, OutcomeBinding)
    _tuple_of(sequence, SequenceEdge)
    _tuple_of(choices, ChoiceDecl)
    _tuple_of(protection, ProtectionRequirement)


def _workflow_depth(workflow: AdmittedWorkflow) -> int:
    maximum = 1
    pending = [(workflow, 1)]
    while pending:
        current, depth = pending.pop()
        maximum = max(maximum, depth)
        pending.extend((node.body, depth + 1) for node in current.nodes if isinstance(node, SubgraphNode))
    return maximum


def _node_references(
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
) -> list[NodeId]:
    references: list[NodeId] = []
    for binding in (*input_bindings, *output_bindings):
        if isinstance(binding.source, NodeOutputRef):
            references.append(binding.source.node)
        if isinstance(binding, InputBinding):
            references.append(binding.destination.node)
    references.extend(binding.source.node for binding in outcome_bindings)
    references.extend(node for edge in sequence for node in (edge.before, edge.after))
    for choice in choices:
        references.append(choice.selector)
        references.extend(member for branch in choice.branches for member in branch.members)
    return references


def _validate_owners(
    workflow: WorkflowId,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
) -> None:
    if any(node.id.workflow != workflow for node in nodes) or any(
        reference.workflow != workflow
        for reference in _node_references(input_bindings, output_bindings, outcome_bindings, sequence, choices)
    ):
        _reject(ValidationCode.FOREIGN_OWNER)
    if any(isinstance(node, SubgraphNode) and node.body.workflow == workflow for node in nodes):
        _reject(ValidationCode.FOREIGN_OWNER)


def _duplicates(values: tuple[Any, ...] | list[Any]) -> bool:
    return len(values) != len(set(values))


def _validate_duplicates(
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    protection: tuple[ProtectionRequirement, ...],
) -> None:
    if (
        _duplicates([node.id for node in nodes])
        or _duplicates([binding.destination for binding in input_bindings])
        or _duplicates(
            [binding.source.port for binding in input_bindings if isinstance(binding.source, ContextInputRef)]
        )
        or _duplicates([binding.destination for binding in output_bindings])
        or _duplicates([binding.source for binding in outcome_bindings])
        or _duplicates(list(sequence))
        or _duplicates([choice.selector for choice in choices])
        or _duplicates(list(protection))
    ):
        _reject(ValidationCode.DUPLICATE)
    if any(_duplicates(list(choice.branches)) for choice in choices):
        _reject(ValidationCode.DUPLICATE)


def _port_type(operation: OperationSpec, port: str, *, output: bool) -> ArtifactType | None:
    ports = operation.outputs if output else operation.inputs
    return next((item.artifact_type for item in ports if item.name == port), None)


def _source_type(
    source: WorkflowInputRef | ContextInputRef | NodeOutputRef,
    interface: OperationSpec,
    nodes: dict[NodeId, OperationSpec],
) -> ArtifactType | None:
    if isinstance(source, (WorkflowInputRef, ContextInputRef)):
        return _port_type(interface, source.port, output=False)
    operation = nodes.get(source.node)
    return None if operation is None else _port_type(operation, source.port, output=True)


def _validate_references(
    interface: OperationSpec,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    protection: tuple[ProtectionRequirement, ...],
) -> bool:
    operations = {node.id: node.operation for node in nodes}
    interface_outputs = {port.name for port in interface.outputs}
    interface_outcomes = {outcome.name for outcome in interface.outcomes}
    incompatible_binding = False
    endpoints = {
        port
        for outcome in interface.outcomes
        for promise in outcome.evidence
        for port in (promise.subject_port, *promise.consumed_ports)
        if isinstance(port, MapItemPort)
    } | {
        port
        for requirement in protection
        for port in (requirement.subject_port, *requirement.consumed_ports)
        if isinstance(port, MapItemPort)
    }
    for endpoint in endpoints:
        if not nodes:
            _reject(ValidationCode.MISSING)
        endpoint.validate_in(workflow=nodes[0].id.workflow, nodes=nodes)
    for binding in input_bindings:
        destination = operations.get(binding.destination.node)
        source_type = _source_type(binding.source, interface, operations)
        destination_type = (
            None if destination is None else _port_type(destination, binding.destination.port, output=False)
        )
        if source_type is None or destination_type is None:
            _reject(ValidationCode.MISSING)
        if source_type != destination_type:
            incompatible_binding = True
    workflow_ports = {binding.source.port for binding in input_bindings if isinstance(binding.source, WorkflowInputRef)}
    context_ports = {binding.source.port for binding in input_bindings if isinstance(binding.source, ContextInputRef)}
    if workflow_ports & context_ports:
        _reject(ValidationCode.CONTRADICTORY)
    for binding in input_bindings:
        if not isinstance(binding.source, ContextInputRef):
            continue
        destination = operations[binding.destination.node]
        marked = {use.port for outcome in destination.outcomes for use in outcome.context}
        if binding.destination.port not in marked:
            _reject(ValidationCode.CONTRADICTORY)
    for binding in output_bindings:
        source_type = _source_type(binding.source, interface, operations)
        destination_type = _port_type(interface, binding.destination.port, output=True)
        if source_type is None or destination_type is None:
            _reject(ValidationCode.MISSING)
        if source_type != destination_type:
            incompatible_binding = True
    for binding in outcome_bindings:
        operation = operations.get(binding.source.node)
        if operation is None or binding.source.outcome not in {outcome.name for outcome in operation.outcomes}:
            _reject(ValidationCode.MISSING)
        if binding.destination.outcome not in interface_outcomes:
            _reject(ValidationCode.MISSING)
    if any(edge.before not in operations or edge.after not in operations for edge in sequence):
        _reject(ValidationCode.MISSING)
    for choice in choices:
        selector = operations.get(choice.selector)
        if selector is None:
            _reject(ValidationCode.MISSING)
        selector_outcomes = {outcome.name for outcome in selector.outcomes}
        for branch in choice.branches:
            if not branch.outcomes <= selector_outcomes or not branch.members <= operations.keys():
                _reject(ValidationCode.MISSING)
    for requirement in protection:
        if (
            requirement.outcome not in interface_outcomes
            or isinstance(requirement.subject_port, str)
            and requirement.subject_port not in ({port.name for port in interface.inputs} | interface_outputs)
            or any(
                isinstance(port, str) and port not in {item.name for item in interface.inputs}
                for port in requirement.consumed_ports
            )
            or requirement.candidate_port is not None
            and requirement.candidate_port not in interface_outputs
        ):
            _reject(ValidationCode.MISSING)
        if requirement.candidate_port is not None and requirement.candidate_port not in next(
            outcome.produced_ports for outcome in interface.outcomes if outcome.name == requirement.outcome
        ):
            _reject(ValidationCode.MISSING)
    required_inputs = {(node.id, port.name) for node in nodes for port in node.operation.inputs}
    if {binding.destination for binding in input_bindings} != {
        NodeInputRef(node=node, port=port) for node, port in required_inputs
    }:
        _reject(ValidationCode.MISSING)
    if {binding.destination.port for binding in output_bindings} != interface_outputs:
        _reject(ValidationCode.MISSING)
    return incompatible_binding


def _closure(start: NodeId, edges: frozenset[SequenceEdge]) -> frozenset[NodeId]:
    reached: set[NodeId] = {start}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.before in reached and edge.after not in reached:
                reached.add(edge.after)
                changed = True
    return frozenset(reached)


def _validate_choice_overlaps(choices: tuple[ChoiceDecl, ...]) -> None:
    all_members: set[NodeId] = set()
    for choice in choices:
        branch_members: set[NodeId] = set()
        branch_outcomes: set[str] = set()
        for branch in choice.branches:
            if branch_members & branch.members or branch_outcomes & branch.outcomes:
                _reject(ValidationCode.OVERLAP)
            branch_members.update(branch.members)
            branch_outcomes.update(branch.outcomes)
        if all_members & branch_members:
            _reject(ValidationCode.OVERLAP)
        all_members.update(branch_members)


def _validate_choice_reachability(sequence: tuple[SequenceEdge, ...], choices: tuple[ChoiceDecl, ...]) -> None:
    edges = frozenset(sequence)
    for choice in choices:
        reached = _closure(choice.selector, edges)
        if any(member not in reached for branch in choice.branches for member in branch.members):
            _reject(ValidationCode.CONTRADICTORY)


def _validate_cycles(nodes: tuple[Node, ...], sequence: tuple[SequenceEdge, ...]) -> None:
    if _has_cycle(frozenset(node.id for node in nodes), frozenset(sequence)):
        _reject(ValidationCode.CYCLE)


def _compatible(target: OperationSpec, replacement: OperationSpec, *, allow_narrower: bool) -> bool:
    if (
        target.inputs != replacement.inputs
        or target.outputs != replacement.outputs
        or frozenset(target.output_dependencies) != frozenset(replacement.output_dependencies)
    ):
        return False
    target_outcomes = {outcome.name: outcome for outcome in target.outcomes}
    replacement_outcomes = {outcome.name: outcome for outcome in replacement.outcomes}
    if target_outcomes.keys() != replacement_outcomes.keys():
        return False
    for name, expected in target_outcomes.items():
        actual = replacement_outcomes[name]
        if (
            expected.category != actual.category
            or expected.produced_ports != actual.produced_ports
            or expected.context != actual.context
            or expected.evidence != actual.evidence
            or expected.state_effects != actual.state_effects
            or expected.model_requirements != actual.model_requirements
        ):
            return False
        expected_ceiling = expected.ceiling
        actual_ceiling = actual.ceiling
        if allow_narrower:
            if any(
                replacement_value > target_value
                for replacement_value, target_value in zip(
                    (
                        actual_ceiling.max_activations,
                        actual_ceiling.max_model_requests,
                        actual_ceiling.max_input_bytes,
                        actual_ceiling.max_output_bytes,
                    ),
                    (
                        expected_ceiling.max_activations,
                        expected_ceiling.max_model_requests,
                        expected_ceiling.max_input_bytes,
                        expected_ceiling.max_output_bytes,
                    ),
                    strict=True,
                )
            ):
                return False
        elif expected_ceiling != actual_ceiling:
            return False
    return True


def _protection(
    interface: OperationSpec, protection: tuple[ProtectionRequirement, ...]
) -> tuple[frozenset[str], frozenset[ProtectionRequirement]]:
    outcomes = {outcome.name: outcome for outcome in interface.outcomes}
    affected = {requirement.outcome for requirement in protection}
    unmet: set[ProtectionRequirement] = set()
    for requirement in protection:
        matched = any(
            promise.meaning == requirement.meaning
            and promise.subject_port == requirement.subject_port
            and requirement.consumed_ports <= promise.consumed_ports
            and requirement.coverage <= promise.coverage
            for promise in outcomes[requirement.outcome].evidence
        )
        if not matched:
            unmet.add(requirement)
    ineligible = {requirement.outcome for requirement in unmet}
    eligible = affected - ineligible
    return frozenset(eligible), frozenset(unmet)
