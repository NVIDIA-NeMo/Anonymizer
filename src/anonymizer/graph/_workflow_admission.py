# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static and dynamic workflow admission and substitution."""

from __future__ import annotations

from typing import Any

from anonymizer.graph._values import ValidationCode
from anonymizer.graph._workflow_analysis import (
    _bounded_analysis,
    _difference,
    _equal,
    _intersection,
    _subset,
    _union,
    _visits,
)
from anonymizer.graph._workflow_composition import _has_cycle, _validate_paths
from anonymizer.graph._workflow_values import (
    _ADMISSION_KEY,
    ArtifactType,
    ChoiceDecl,
    ContextInputRef,
    DynamicLimits,
    Graph,
    InputBinding,
    KeyedJoinDecl,
    LoopDecl,
    MapDecl,
    Node,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    Operation,
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
    _analysis_limit,
    _map_item_ports,
    _reject,
    _tuple_of,
    _validate_scalars,
)


def _validate_controls(workflow: Graph) -> None:
    """Check finite control roles on the same body accepted by static admission."""
    limits = workflow.dynamic_limits
    reachable = workflow.bodies()
    scopes = reachable
    for body in _visits(reachable):
        for endpoint in _visits(body.map_item_ports):
            matches = [
                item
                for scope in _visits(scopes)
                for item in _visits(scope.maps)
                if _equal(item.expander, endpoint.expander)
                and _equal(item.member, endpoint.member)
                and _equal(item.item_input, endpoint.item_input)
                and endpoint.expansion_outcome in item.expansion_outcomes
            ]
            if not _equal(len(matches), 1):
                _reject(ValidationCode.MISSING if not matches else ValidationCode.DUPLICATE)

    total_maps = sum(len(scope.maps) for scope in _visits(scopes))
    total_joins = sum(len(scope.joins) for scope in _visits(scopes))
    total_loops = sum(len(scope.loops) for scope in _visits(scopes))
    if (
        total_maps > limits.max_maps
        or total_joins > limits.max_joins
        or total_loops > limits.max_loops
        or any(
            item.max_children > limits.max_children_per_map for scope in _visits(scopes) for item in _visits(scope.maps)
        )
        or any(
            item.max_iterations > limits.max_iterations_per_loop
            for scope in _visits(scopes)
            for item in _visits(scope.loops)
        )
    ):
        _reject(ValidationCode.LIMIT_EXCEEDED)

    for scope in _visits(scopes):
        nodes = {node.id: node for node in _visits(scope.operations)}
        owned = set(_visits(nodes))
        declared_joins = [join.join for join in _visits(scope.joins)]
        aggregate_sources = [item.expander for item in _visits(scope.maps)] + [
            item.starter for item in _visits(scope.loops)
        ]
        sources = set(_visits(aggregate_sources))
        members = [item.member for item in _visits(scope.maps)] + [item.member for item in _visits(scope.loops)]
        aggregate_joins = declared_joins
        referenced = _union(
            _union(
                {node for item in _visits(scope.maps) for node in _visits((item.expander, item.member))},
                {node for item in _visits(scope.joins) for node in _visits((item.source, item.join))},
            ),
            {node for item in _visits(scope.loops) for node in _visits((item.starter, item.member, item.join))},
        )
        if any(not _equal(node.workflow, scope.workflow) for node in _visits(referenced)):
            _reject(ValidationCode.FOREIGN_OWNER)
        if (
            not _equal(len(members), len(set(_visits(members))))
            or not _equal(len(aggregate_joins), len(set(_visits(aggregate_joins))))
            or not _equal(len(aggregate_sources), len(sources))
        ):
            _reject(ValidationCode.DUPLICATE)
        if not _subset(referenced, owned):
            _reject(ValidationCode.MISSING)
        join_sources = {item.source for item in _visits(scope.joins)}
        if not _equal(join_sources, sources) or not _equal(len(join_sources), len(scope.joins)):
            _reject(ValidationCode.MISSING)
        choice_members = {
            member
            for choice in _visits(scope.choices)
            for branch in _visits(choice.branches)
            for member in _visits(branch.members)
        }
        if _intersection(choice_members, _union(set(_visits(members)), set(_visits(aggregate_joins)))):
            _reject(ValidationCode.OVERLAP)
        outcomes = {
            node_id: {outcome.name for outcome in _visits(node.operation.outcomes)}
            for node_id, node in _visits(nodes.items())
        }
        if any(not _subset(item.expansion_outcomes, outcomes[item.expander]) for item in _visits(scope.maps)):
            _reject(ValidationCode.MISSING)
        for item in _visits(scope.maps):
            if item.max_children > 1:
                outward_inputs = any(
                    isinstance(binding.source, NodeOutputRef)
                    and _equal(binding.source.node, item.member)
                    and not _equal(binding.destination.node, item.member)
                    for binding in _visits(scope.input_bindings)
                )
                outward_outputs = any(
                    isinstance(binding.source, NodeOutputRef) and _equal(binding.source.node, item.member)
                    for binding in _visits(scope.output_bindings)
                )
                if outward_inputs or outward_outputs:
                    _reject(ValidationCode.UNSUPPORTED)
            if item.item_input is None:
                continue
            member_inputs = {port.name for port in _visits(nodes[item.member].operation.inputs)}
            if item.item_input not in member_inputs:
                _reject(ValidationCode.MISSING)
            bindings = [
                binding
                for binding in _visits(scope.input_bindings)
                if _equal(binding.destination, NodeInputRef(node=item.member, port=item.item_input))
            ]
            if not _equal(len(bindings), 1):
                _reject(ValidationCode.MISSING)
        for item in _visits(scope.loops):
            if not _equal(_union(item.enter_outcomes, item.bypass_outcomes), outcomes[item.starter]):
                _reject(ValidationCode.MISSING)
            if not _equal(_union(item.continue_outcomes, item.exit_outcomes), outcomes[item.member]):
                _reject(ValidationCode.MISSING)
            member_operation = nodes[item.member].operation
            input_names = {port.name for port in _visits(member_operation.inputs)}
            initial_destinations = [binding.destination.port for binding in _visits(item.initial)]
            carried_destinations = [binding.destination.port for binding in _visits(item.carried)]
            if any(
                not _equal(binding.destination.node, item.member)
                for binding in _visits((*_visits(item.initial), *_visits(item.carried)))
            ):
                _reject(ValidationCode.FOREIGN_OWNER)
            if not _equal(len(initial_destinations), len(set(_visits(initial_destinations)))) or not _equal(
                len(carried_destinations), len(set(_visits(carried_destinations)))
            ):
                _reject(ValidationCode.DUPLICATE)
            if not _equal(set(_visits(initial_destinations)), input_names) or not _equal(
                set(_visits(carried_destinations)), input_names
            ):
                _reject(ValidationCode.MISSING)
            static_bindings = {
                (binding.source, binding.destination)
                for binding in _visits(scope.input_bindings)
                if _equal(binding.destination.node, item.member)
            }
            if not _equal(
                {(binding.source, binding.destination) for binding in _visits(item.initial)}, static_bindings
            ):
                _reject(ValidationCode.CONTRADICTORY)
            output_types = {port.name: port.artifact_type for port in _visits(member_operation.outputs)}
            input_types = {port.name: port.artifact_type for port in _visits(member_operation.inputs)}
            if any(
                not _equal(binding.source.node, item.member)
                or binding.source.port not in output_types
                or not _equal(output_types[binding.source.port], input_types[binding.destination.port])
                for binding in _visits(item.carried)
            ):
                _reject(ValidationCode.CONTRADICTORY)
        joins_by_source = {item.source: item for item in _visits(scope.joins)}
        if any(not _equal(joins_by_source[item.starter].join, item.join) for item in _visits(scope.loops)):
            _reject(ValidationCode.CONTRADICTORY)
        edges = {(edge.before, edge.after) for edge in _visits(scope.sequence)}
        reachable_pairs = set(_visits(edges))
        changed = True
        while changed:
            changed = False
            additions = _difference(
                {
                    (left, right)
                    for left, middle in _visits(reachable_pairs)
                    for candidate, right in _visits(reachable_pairs)
                    if _equal(middle, candidate) and (not _equal(left, right))
                },
                reachable_pairs,
            )
            if additions:
                reachable_pairs.update(_visits(additions))
                changed = True
        if any(
            (item.expander, item.member) not in reachable_pairs
            or (item.member, next(join.join for join in _visits(scope.joins) if _equal(join.source, item.expander)))
            not in reachable_pairs
            for item in _visits(scope.maps)
        ) or any(
            (item.starter, item.member) not in reachable_pairs or (item.member, item.join) not in reachable_pairs
            for item in _visits(scope.loops)
        ):
            _reject(ValidationCode.CONTRADICTORY)

    if workflow.dynamic_depth > limits.max_dynamic_depth or any(
        body.activation_upper_bound > limits.max_activation_occurrences for body in _visits(reachable)
    ):
        _reject(ValidationCode.LIMIT_EXCEEDED)
    if any(
        outcome.ceiling.max_activations < body.activation_upper_bound
        for body in _visits(reachable)
        for outcome in _visits(body.interface.outcomes)
    ):
        _reject(ValidationCode.CONTRADICTORY)


@_bounded_analysis(_analysis_limit)
def compile_graph(
    *,
    workflow: WorkflowId,
    interface: OperationSpec,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    protection: tuple[ProtectionRequirement, ...],
    limits: WorkflowLimits,
    dynamic_limits: DynamicLimits | None = None,
) -> Graph:
    """Compile typed operation and control nodes into one admitted topology."""
    body = _compile_body(
        workflow=workflow,
        interface=interface,
        nodes=nodes,
        input_bindings=input_bindings,
        output_bindings=output_bindings,
        outcome_bindings=outcome_bindings,
        sequence=sequence,
        protection=protection,
        limits=limits,
        dynamic_limits=dynamic_limits,
    )
    _validate_controls(body)
    return body


@_bounded_analysis(_analysis_limit)
def _compile_body(
    *,
    workflow: WorkflowId,
    interface: OperationSpec,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    protection: tuple[ProtectionRequirement, ...],
    limits: WorkflowLimits,
    dynamic_limits: DynamicLimits | None = None,
) -> Graph:
    """Compile operations and controls into one admitted graph body."""
    _tuple_of(nodes, (OperationNode, SubgraphNode, ChoiceDecl, MapDecl, LoopDecl, KeyedJoinDecl))
    graph_nodes = nodes
    choices = tuple(node for node in _visits(nodes) if isinstance(node, ChoiceDecl))
    nodes = tuple(node for node in _visits(nodes) if isinstance(node, (OperationNode, SubgraphNode)))
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
    for binding in _visits(input_bindings):
        _validate_scalars(strings=(binding.destination.port,))
    if any(choice.selector in branch.members for choice in _visits(choices) for branch in _visits(choice.branches)):
        _reject(ValidationCode.INVALID_VALUE)
    expanded_count = sum(
        1 + (node.body.expanded_node_count if isinstance(node, SubgraphNode) else 0) for node in _visits(nodes)
    )
    depth = max(
        (1 + node.body.dynamic_depth if isinstance(node, SubgraphNode) else 1 for node in _visits(nodes)), default=1
    )
    choice_states = 1
    node_operations = {node.id: node.operation for node in _visits(nodes)}
    for choice in _visits(choices):
        operation = node_operations.get(choice.selector)
        choice_states *= len(operation.outcomes) if operation is not None else 1
    raw_bindings = len(input_bindings) + len(output_bindings) + len(outcome_bindings)
    declarations = (
        *_visits((promise for outcome in _visits(interface.outcomes) for promise in _visits(outcome.evidence))),
        *_visits(protection),
    )
    raw_bindings += len(set(_map_item_ports(_visits(declarations))))
    branch_members = sum(len(branch.members) for choice in _visits(choices) for branch in _visits(choice.branches))
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
    for node in _visits(nodes):
        if isinstance(node, SubgraphNode):
            if not _compatible(node.operation, node.body.interface, allow_narrower=True):
                _reject(ValidationCode.CONTRADICTORY)
    if incompatible_binding:
        _reject(ValidationCode.CONTRADICTORY)
    _validate_paths(
        interface, nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices, check_cycles=False
    )
    eligible, unmet = _protection(interface, protection)
    if dynamic_limits is None:
        if any(isinstance(node, (MapDecl, LoopDecl, KeyedJoinDecl)) for node in graph_nodes):
            _reject(ValidationCode.MISSING)
        dynamic_limits = DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=limits.max_subgraph_depth,
            max_activation_occurrences=limits.max_nodes,
        )
    if not isinstance(dynamic_limits, DynamicLimits):
        _reject(ValidationCode.INVALID_TYPE)
    if _duplicates(graph_nodes):
        _reject(ValidationCode.DUPLICATE)
    admitted = Graph(
        _key=_ADMISSION_KEY,
        workflow=workflow,
        interface=interface,
        nodes=tuple(_visits(graph_nodes)),
        input_bindings=frozenset(_visits(input_bindings)),
        output_bindings=frozenset(_visits(output_bindings)),
        outcome_bindings=frozenset(_visits(outcome_bindings)),
        sequence=frozenset(_visits(sequence)),
        protection_requirements=frozenset(_visits(protection)),
        protection_eligible_outcomes=eligible,
        unmet_protection=unmet,
        limits=limits,
        expanded_node_count=expanded_count,
        dynamic_limits=dynamic_limits,
    )
    return admitted


@_bounded_analysis(_analysis_limit)
def substitute(*, workflow: Graph, target: NodeId, replacement: Graph) -> Graph:
    """Replace a graph operation and check all resulting control obligations."""
    body = _substitute_body(workflow=workflow, target=target, replacement=replacement)
    _validate_controls(body)
    return body


@_bounded_analysis(_analysis_limit)
def _substitute_body(*, workflow: Graph, target: NodeId, replacement: Graph) -> Graph:
    """Replace one operation with a compatible, separately owned workflow."""
    if not isinstance(workflow, Graph) or not isinstance(target, NodeId) or not isinstance(replacement, Graph):
        _reject(ValidationCode.INVALID_TYPE)
    if not _equal(target.workflow, workflow.workflow):
        _reject(ValidationCode.FOREIGN_OWNER)
    if _equal(replacement.workflow, workflow.workflow):
        _reject(ValidationCode.FOREIGN_OWNER)
    indexed_nodes = {node.id: node for node in _visits(workflow.operations)}
    target_node = indexed_nodes.get(target)
    if target_node is None:
        _reject(ValidationCode.MISSING)
    if not _compatible(target_node.operation, replacement.interface, allow_narrower=True):
        _reject(ValidationCode.CONTRADICTORY)
    nodes = tuple(
        SubgraphNode(id=target, operation=target_node.operation, body=replacement) if _equal(node.id, target) else node
        for node in _visits(workflow.operations)
    )
    return _compile_body(
        workflow=workflow.workflow,
        interface=workflow.interface,
        nodes=(*nodes, *workflow.choices, *workflow.maps, *workflow.loops, *workflow.joins),
        input_bindings=tuple(_visits(workflow.input_bindings)),
        output_bindings=tuple(_visits(workflow.output_bindings)),
        outcome_bindings=tuple(_visits(workflow.outcome_bindings)),
        sequence=tuple(_visits(workflow.sequence)),
        protection=tuple(_visits(workflow.protection_requirements)),
        limits=workflow.limits,
        dynamic_limits=workflow.dynamic_limits,
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


def _node_references(
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
) -> list[NodeId]:
    references: list[NodeId] = []
    for binding in _visits((*_visits(input_bindings), *_visits(output_bindings))):
        if isinstance(binding.source, NodeOutputRef):
            references.append(binding.source.node)
        if isinstance(binding, InputBinding):
            references.append(binding.destination.node)
    references.extend(binding.source.node for binding in _visits(outcome_bindings))
    references.extend(node for edge in _visits(sequence) for node in _visits((edge.before, edge.after)))
    for choice in _visits(choices):
        references.append(choice.selector)
        references.extend(member for branch in _visits(choice.branches) for member in _visits(branch.members))
    return references


def _validate_owners(
    workflow: WorkflowId,
    nodes: tuple[Operation, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
) -> None:
    if any(not _equal(node.id.workflow, workflow) for node in _visits(nodes)) or any(
        not _equal(reference.workflow, workflow)
        for reference in _visits(_node_references(input_bindings, output_bindings, outcome_bindings, sequence, choices))
    ):
        _reject(ValidationCode.FOREIGN_OWNER)
    if any(isinstance(node, SubgraphNode) and _equal(node.body.workflow, workflow) for node in _visits(nodes)):
        _reject(ValidationCode.FOREIGN_OWNER)


def _duplicates(values: tuple[Any, ...] | list[Any]) -> bool:
    return not _equal(len(values), len(set(_visits(values))))


def _validate_duplicates(
    nodes: tuple[Operation, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    protection: tuple[ProtectionRequirement, ...],
) -> None:
    if (
        _duplicates([node.id for node in _visits(nodes)])
        or _duplicates([binding.destination for binding in _visits(input_bindings)])
        or _duplicates(
            [binding.source.port for binding in _visits(input_bindings) if isinstance(binding.source, ContextInputRef)]
        )
        or _duplicates([binding.destination for binding in _visits(output_bindings)])
        or _duplicates([binding.source for binding in _visits(outcome_bindings)])
        or _duplicates(list(_visits(sequence)))
        or _duplicates([choice.selector for choice in _visits(choices)])
        or _duplicates(list(_visits(protection)))
    ):
        _reject(ValidationCode.DUPLICATE)
    if any(_duplicates(list(_visits(choice.branches))) for choice in _visits(choices)):
        _reject(ValidationCode.DUPLICATE)


def _port_type(operation: OperationSpec, port: str, *, output: bool) -> ArtifactType | None:
    ports = operation.outputs if output else operation.inputs
    return next((item.artifact_type for item in _visits(ports) if _equal(item.name, port)), None)


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
    nodes: tuple[Operation, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    protection: tuple[ProtectionRequirement, ...],
) -> bool:
    operations = {node.id: node.operation for node in _visits(nodes)}
    interface_outputs = {port.name for port in _visits(interface.outputs)}
    interface_outcomes = {outcome.name for outcome in _visits(interface.outcomes)}
    incompatible_binding = False
    endpoints = _union(
        set(
            _map_item_ports(promise for outcome in _visits(interface.outcomes) for promise in _visits(outcome.evidence))
        ),
        set(_map_item_ports(_visits(protection))),
    )
    for endpoint in _visits(endpoints):
        if not nodes:
            _reject(ValidationCode.MISSING)
        endpoint.validate_in(workflow=nodes[0].id.workflow, nodes=nodes)
    for binding in _visits(input_bindings):
        destination = operations.get(binding.destination.node)
        source_type = _source_type(binding.source, interface, operations)
        destination_type = (
            None if destination is None else _port_type(destination, binding.destination.port, output=False)
        )
        if source_type is None or destination_type is None:
            _reject(ValidationCode.MISSING)
        if not _equal(source_type, destination_type):
            incompatible_binding = True
    workflow_ports = {
        binding.source.port for binding in _visits(input_bindings) if isinstance(binding.source, WorkflowInputRef)
    }
    context_ports = {
        binding.source.port for binding in _visits(input_bindings) if isinstance(binding.source, ContextInputRef)
    }
    if _intersection(workflow_ports, context_ports):
        _reject(ValidationCode.CONTRADICTORY)
    for binding in _visits(input_bindings):
        if not isinstance(binding.source, ContextInputRef):
            continue
        destination = operations[binding.destination.node]
        marked = {use.port for outcome in _visits(destination.outcomes) for use in _visits(outcome.context)}
        if binding.destination.port not in marked:
            _reject(ValidationCode.CONTRADICTORY)
    for binding in _visits(output_bindings):
        source_type = _source_type(binding.source, interface, operations)
        destination_type = _port_type(interface, binding.destination.port, output=True)
        if source_type is None or destination_type is None:
            _reject(ValidationCode.MISSING)
        if not _equal(source_type, destination_type):
            incompatible_binding = True
    for binding in _visits(outcome_bindings):
        operation = operations.get(binding.source.node)
        if operation is None or binding.source.outcome not in {outcome.name for outcome in _visits(operation.outcomes)}:
            _reject(ValidationCode.MISSING)
        if binding.destination.outcome not in interface_outcomes:
            _reject(ValidationCode.MISSING)
    if any(edge.before not in operations or edge.after not in operations for edge in _visits(sequence)):
        _reject(ValidationCode.MISSING)
    for choice in _visits(choices):
        selector = operations.get(choice.selector)
        if selector is None:
            _reject(ValidationCode.MISSING)
        selector_outcomes = {outcome.name for outcome in _visits(selector.outcomes)}
        for branch in _visits(choice.branches):
            if not _subset(branch.outcomes, selector_outcomes) or not _subset(branch.members, operations.keys()):
                _reject(ValidationCode.MISSING)
    for requirement in _visits(protection):
        if (
            requirement.outcome not in interface_outcomes
            or isinstance(requirement.subject_port, str)
            and requirement.subject_port
            not in (_union({port.name for port in _visits(interface.inputs)}, interface_outputs))
            or any(
                isinstance(port, str) and port not in {item.name for item in _visits(interface.inputs)}
                for port in _visits(requirement.consumed_ports)
            )
            or requirement.candidate_port is not None
            and requirement.candidate_port not in interface_outputs
        ):
            _reject(ValidationCode.MISSING)
        if requirement.candidate_port is not None and requirement.candidate_port not in next(
            outcome.produced_ports
            for outcome in _visits(interface.outcomes)
            if _equal(outcome.name, requirement.outcome)
        ):
            _reject(ValidationCode.MISSING)
    required_inputs = {(node.id, port.name) for node in _visits(nodes) for port in _visits(node.operation.inputs)}
    if not _equal(
        {binding.destination for binding in _visits(input_bindings)},
        {NodeInputRef(node=node, port=port) for node, port in _visits(required_inputs)},
    ):
        _reject(ValidationCode.MISSING)
    if not _equal({binding.destination.port for binding in _visits(output_bindings)}, interface_outputs):
        _reject(ValidationCode.MISSING)
    return incompatible_binding


def _closure(start: NodeId, edges: frozenset[SequenceEdge]) -> frozenset[NodeId]:
    reached: set[NodeId] = {start}
    while True:
        additions = {edge.after for edge in _visits(edges) if edge.before in reached}
        if _subset(additions, reached):
            return frozenset(_visits(reached))
        reached.update(_visits(additions))


def _validate_choice_overlaps(choices: tuple[ChoiceDecl, ...]) -> None:
    all_members: set[NodeId] = set()
    for choice in _visits(choices):
        branch_members: set[NodeId] = set()
        branch_outcomes: set[str] = set()
        for branch in _visits(choice.branches):
            if _intersection(branch_members, branch.members) or _intersection(branch_outcomes, branch.outcomes):
                _reject(ValidationCode.OVERLAP)
            branch_members.update(_visits(branch.members))
            branch_outcomes.update(_visits(branch.outcomes))
        if _intersection(all_members, branch_members):
            _reject(ValidationCode.OVERLAP)
        all_members.update(_visits(branch_members))


def _validate_choice_reachability(sequence: tuple[SequenceEdge, ...], choices: tuple[ChoiceDecl, ...]) -> None:
    edges = frozenset(_visits(sequence))
    for choice in _visits(choices):
        reached = _closure(choice.selector, edges)
        if any(member not in reached for branch in _visits(choice.branches) for member in _visits(branch.members)):
            _reject(ValidationCode.CONTRADICTORY)


def _validate_cycles(nodes: tuple[Operation, ...], sequence: tuple[SequenceEdge, ...]) -> None:
    if _has_cycle(frozenset(node.id for node in _visits(nodes)), frozenset(_visits(sequence))):
        _reject(ValidationCode.CYCLE)


def _compatible(target: OperationSpec, replacement: OperationSpec, *, allow_narrower: bool) -> bool:
    if (
        not _equal(target.inputs, replacement.inputs)
        or not _equal(target.outputs, replacement.outputs)
        or not _equal(
            frozenset(_visits(target.output_dependencies)), frozenset(_visits(replacement.output_dependencies))
        )
    ):
        return False
    target_outcomes = {outcome.name: outcome for outcome in _visits(target.outcomes)}
    replacement_outcomes = {outcome.name: outcome for outcome in _visits(replacement.outcomes)}
    if not _equal(target_outcomes.keys(), replacement_outcomes.keys()):
        return False
    for name, expected in _visits(target_outcomes.items()):
        actual = replacement_outcomes[name]
        if (
            not _equal(expected.category, actual.category)
            or not _equal(expected.produced_ports, actual.produced_ports)
            or not _equal(expected.context, actual.context)
            or not _equal(expected.evidence, actual.evidence)
            or not _equal(expected.state_effects, actual.state_effects)
            or not _equal(expected.model_requirements, actual.model_requirements)
        ):
            return False
        expected_ceiling = expected.ceiling
        actual_ceiling = actual.ceiling
        if allow_narrower:
            if any(
                target_value is not None and (replacement_value is None or replacement_value > target_value)
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
        elif not _equal(expected_ceiling, actual_ceiling):
            return False
    return True


def _protection(
    interface: OperationSpec, protection: tuple[ProtectionRequirement, ...]
) -> tuple[frozenset[str], frozenset[ProtectionRequirement]]:
    outcomes = {outcome.name: outcome for outcome in _visits(interface.outcomes)}
    affected = {requirement.outcome for requirement in _visits(protection)}
    unmet: set[ProtectionRequirement] = set()
    for requirement in _visits(protection):
        matches = [
            _equal(promise.meaning, requirement.meaning)
            and _equal(promise.subject_port, requirement.subject_port)
            and _subset(requirement.consumed_ports, promise.consumed_ports)
            and _subset(requirement.coverage, promise.coverage)
            for promise in _visits(outcomes[requirement.outcome].evidence)
        ]
        if not any(matches):
            unmet.add(requirement)
    ineligible = {requirement.outcome for requirement in _visits(unmet)}
    eligible = _difference(affected, ineligible)
    return frozenset(_visits(eligible)), frozenset(_visits(unmet))
