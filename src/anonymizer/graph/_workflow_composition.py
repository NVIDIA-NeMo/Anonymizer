# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow path and interface composition."""

from __future__ import annotations

import itertools
from typing import TypeAlias, cast

from anonymizer.graph._values import ContractViolation, ValidationCode
from anonymizer.graph._workflow_analysis import (
    _bounded_analysis,
    _difference,
    _equal,
    _subset,
    _union,
    _visited_pop,
    _visits,
)
from anonymizer.graph._workflow_values import (
    AdmittedWorkflow,
    ChoiceDecl,
    ContextInputRef,
    ContextUse,
    EvidencePort,
    EvidencePromise,
    InputBinding,
    MapItemPort,
    ModelRequirement,
    Node,
    NodeId,
    NodeInputRef,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputBinding,
    SequenceEdge,
    StateEffect,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    _analysis_limit,
    _map_item_ports,
    _reachable_workflows,
    _reject,
    _tuple_of,
)

Vertex: TypeAlias = tuple[str, NodeId | None, str]


def _walk_endpoints(
    start: Vertex, edges: set[tuple[Vertex, Vertex]], *, backward: bool, endpoint_kind: str
) -> frozenset[str]:
    pending = [start]
    seen: set[Vertex] = set()
    endpoints: set[str] = set()
    while pending:
        current = _visited_pop(pending)
        if current in seen:
            continue
        seen.add(current)
        if _equal(current[0], endpoint_kind):
            endpoints.add(current[2])
        for source, destination in _visits(edges):
            if backward and _equal(destination, current):
                pending.append(source)
            elif not backward and _equal(source, current):
                pending.append(destination)
    return frozenset(_visits(endpoints))


def _project_one(endpoints: frozenset[str]) -> str:
    if not endpoints:
        _reject(ValidationCode.MISSING)
    if len(endpoints) > 1:
        _reject(ValidationCode.CONTRADICTORY)
    return next(iter(endpoints))


def _walk_interface_inputs(start: Vertex, edges: set[tuple[Vertex, Vertex]], *, backward: bool) -> frozenset[str]:
    return frozenset(
        _visits(
            _union(
                _walk_endpoints(start, edges, backward=backward, endpoint_kind="wi"),
                _walk_endpoints(start, edges, backward=backward, endpoint_kind="ci"),
            )
        )
    )


def _has_cycle(selected: frozenset[NodeId], edges: frozenset[SequenceEdge]) -> bool:
    remaining = set(_visits(selected))
    while remaining:
        blocked = {edge.after for edge in _visits(edges) if edge.before in remaining}
        roots = _difference(remaining, blocked)
        if not roots:
            return True
        remaining.difference_update(_visits(roots))
    return False


def _selected_nodes(
    node_ids: frozenset[NodeId], choices: tuple[ChoiceDecl, ...], assignment: dict[NodeId, OutcomeSpec]
) -> frozenset[NodeId]:
    branch_members = {
        member
        for choice in _visits(choices)
        for branch in _visits(choice.branches)
        for member in _visits(branch.members)
    }
    selected = set(_visits(_difference(node_ids, branch_members)))
    while True:
        additions = {
            member
            for choice in _visits(choices)
            if choice.selector in selected
            for branch in _visits(choice.branches)
            if assignment[choice.selector].name in branch.outcomes
            for member in _visits(branch.members)
        }
        if _subset(additions, selected):
            return frozenset(_visits(selected))
        selected.update(_visits(additions))


@_bounded_analysis(_analysis_limit)
def validate_dynamic_input_summaries(
    workflow: AdmittedWorkflow,
    replacements: tuple[InputBinding, ...],
) -> None:
    """Recheck retained interface summaries after nonidentity dynamic input replacement."""
    if not isinstance(workflow, AdmittedWorkflow):
        _reject(ValidationCode.INVALID_TYPE)
    _tuple_of(replacements, InputBinding)
    destinations = [item.destination for item in _visits(replacements)]
    if not _equal(len(destinations), len(set(_visits(destinations)))):
        _reject(ValidationCode.DUPLICATE)
    reachable = list(_reachable_workflows([workflow]))
    owners = {item.workflow: item for item in _visits(reachable)}
    grouped: dict[WorkflowId, list[InputBinding]] = {}
    for replacement in _visits(replacements):
        if replacement.destination.node.workflow not in owners:
            _reject(ValidationCode.FOREIGN_OWNER)
        grouped.setdefault(replacement.destination.node.workflow, []).append(replacement)
    for owner, values in _visits(grouped.items()):
        current = owners[owner]
        replaced = {item.destination for item in _visits(values)}
        if any(
            not _equal(
                sum((_equal(binding.destination, destination) for binding in _visits(current.input_bindings))), 1
            )
            for destination in _visits(replaced)
        ):
            _reject(ValidationCode.MISSING)
        inputs = tuple(
            binding for binding in _visits(current.input_bindings) if binding.destination not in replaced
        ) + tuple(_visits(values))
        _validate_paths(
            current.interface,
            tuple(_visits(current.nodes)),
            inputs,
            tuple(_visits(current.output_bindings)),
            tuple(_visits(current.outcome_bindings)),
            tuple(_visits(current.sequence)),
            tuple(_visits(current.choices)),
            check_cycles=False,
            nonidentity_destinations=frozenset(_visits(replaced)),
        )


def _validate_paths(
    interface: OperationSpec,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    *,
    check_cycles: bool = True,
    nonidentity_destinations: frozenset[NodeInputRef] = frozenset(),
) -> None:
    operations = {node.id: node.operation for node in _visits(nodes)}
    node_ids = frozenset(_visits(operations))
    edges = frozenset(_visits(sequence))
    if check_cycles and _has_cycle(node_ids, edges):
        _reject(ValidationCode.CYCLE)
    outcome_options = _outcome_options(nodes, outcome_bindings, choices, interface)
    ids = tuple(_visits(operations))
    reached_interface_outcomes: set[str] = set()
    for outcomes in _visits(itertools.product(*_visits(outcome_options))):
        assignment = dict(_visits(zip(ids, outcomes, strict=True)))
        selected = _selected_nodes(node_ids, choices, assignment)
        selected_edges = frozenset(
            edge for edge in _visits(edges) if edge.before in selected and edge.after in selected
        )
        if check_cycles and _has_cycle(selected, selected_edges):
            _reject(ValidationCode.CYCLE)
        reachable: set[NodeId] = set()
        while True:
            additions: set[NodeId] = set()
            for node in _visits(_difference(selected, reachable)):
                predecessors = {edge.before for edge in _visits(selected_edges) if _equal(edge.after, node)}
                if not _subset(predecessors, reachable):
                    continue
                bindings = [binding for binding in _visits(input_bindings) if _equal(binding.destination.node, node)]
                available = [
                    isinstance(binding.source, (WorkflowInputRef, ContextInputRef))
                    or (
                        binding.source.node in reachable
                        and binding.source.port in assignment[binding.source.node].produced_ports
                    )
                    for binding in _visits(bindings)
                ]
                if all(available):
                    additions.add(node)
            if not additions:
                break
            reachable.update(_visits(additions))
        if not _equal(reachable, set(_visits(selected))):
            _reject(ValidationCode.MISSING)
        outgoing = {edge.before for edge in _visits(selected_edges) if edge.after in reachable}
        sinks = _difference(reachable, outgoing)
        if not _equal(len(sinks), 1):
            _reject(ValidationCode.MISSING)
        sink = next(iter(sinks))
        sink_outcome = assignment[sink]
        mappings = [
            binding
            for binding in _visits(outcome_bindings)
            if _equal(binding.source.node, sink) and _equal(binding.source.outcome, sink_outcome.name)
        ]
        if not _equal(len(mappings), 1):
            _reject(ValidationCode.MISSING)
        external_outcome_name = mappings[0].destination.outcome
        reached_interface_outcomes.add(external_outcome_name)
        external_outcome = next(
            outcome for outcome in _visits(interface.outcomes) if _equal(outcome.name, external_outcome_name)
        )
        _validate_composition_path(
            interface,
            external_outcome,
            reachable,
            assignment,
            operations,
            input_bindings,
            output_bindings,
            nonidentity_destinations,
        )
    if not _equal(reached_interface_outcomes, {outcome.name for outcome in _visits(interface.outcomes)}):
        _reject(ValidationCode.MISSING)


def _project_consumed_ports(
    ports: frozenset[EvidencePort],
    *,
    node: NodeId,
    identity_edges: set[tuple[Vertex, Vertex]],
    dependency_edges: set[tuple[Vertex, Vertex]],
    declared: frozenset[MapItemPort],
    assignment: dict[NodeId, OutcomeSpec],
) -> frozenset[EvidencePort]:
    """Summarize external dependencies while retaining exact local evidence inputs."""
    projected: set[EvidencePort] = set()
    for port in _visits(ports):
        try:
            projected.add(
                _project_evidence_port(
                    port,
                    node=node,
                    input_port=True,
                    edges=identity_edges,
                    declared=declared,
                    assignment=assignment,
                )
            )
        except ContractViolation as error:
            if error.code is not ValidationCode.MISSING or not isinstance(port, str):
                raise
            roots = _walk_interface_inputs(("in", node, port), dependency_edges, backward=True)
            if not roots:
                raise
            projected.update(_visits(roots))
    return frozenset(projected)


def _project_evidence_port(
    port: EvidencePort,
    *,
    node: NodeId,
    input_port: bool,
    edges: set[tuple[Vertex, Vertex]],
    declared: frozenset[MapItemPort],
    assignment: dict[NodeId, OutcomeSpec],
    allow_output_alias: bool = False,
) -> EvidencePort:
    """Project a local port by scalar identity or its declared dynamic endpoint."""
    if isinstance(port, MapItemPort):
        return port.lifted(node)
    dynamic = {
        endpoint
        for endpoint in _visits(declared)
        if not endpoint.path
        and _equal(endpoint.member, node)
        and _equal(endpoint.item_input, port)
        and endpoint.expander in assignment
        and _equal(assignment[endpoint.expander].name, endpoint.expansion_outcome)
    }
    if dynamic:
        if not _equal(len(dynamic), 1) or not input_port:
            _reject(ValidationCode.CONTRADICTORY)
        return next(iter(dynamic))
    if input_port:
        inputs = _walk_interface_inputs(("in", node, port), edges, backward=True)
        if inputs or not allow_output_alias:
            return _project_one(inputs)
        identity = _union(edges, {(destination, source) for source, destination in _visits(edges)})
        return _project_one(_walk_endpoints(("in", node, port), identity, backward=False, endpoint_kind="wo"))
    return _project_one(_walk_endpoints(("out", node, port), edges, backward=False, endpoint_kind="wo"))


def _validate_composition_path(
    interface: OperationSpec,
    external_outcome: OutcomeSpec,
    reachable: set[NodeId],
    assignment: dict[NodeId, OutcomeSpec],
    operations: dict[NodeId, OperationSpec],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    nonidentity_destinations: frozenset[NodeInputRef] = frozenset(),
) -> None:
    identity_edges: set[tuple[Vertex, Vertex]] = set()
    dependency_edges: set[tuple[Vertex, Vertex]] = set()
    for binding in _visits(input_bindings):
        if binding.destination.node not in reachable:
            continue
        source = (
            ("wi", None, binding.source.port)
            if isinstance(binding.source, WorkflowInputRef)
            else (
                ("ci", None, binding.source.port)
                if isinstance(binding.source, ContextInputRef)
                else ("out", binding.source.node, binding.source.port)
            )
        )
        destination = ("in", binding.destination.node, binding.destination.port)
        if binding.destination not in nonidentity_destinations:
            identity_edges.add((source, destination))
        dependency_edges.add((source, destination))
    produced_external: set[str] = set()
    for binding in _visits(output_bindings):
        if isinstance(binding.source, WorkflowInputRef):
            source = ("wi", None, binding.source.port)
            exists = True
        else:
            source = ("out", binding.source.node, binding.source.port)
            exists = (
                binding.source.node in reachable
                and binding.source.port in assignment[binding.source.node].produced_ports
            )
        if exists:
            destination = ("wo", None, binding.destination.port)
            identity_edges.add((source, destination))
            dependency_edges.add((source, destination))
            produced_external.add(binding.destination.port)
    for node in _visits(reachable):
        operation = operations[node]
        produced = assignment[node].produced_ports
        for dependency in _visits(operation.output_dependencies):
            if dependency.output not in produced:
                continue
            output_vertex = ("out", node, dependency.output)
            for input_port in _visits(dependency.inputs):
                dependency_edges.add((("in", node, input_port), output_vertex))
            if dependency.identity_input is not None:
                identity_edges.add((("in", node, dependency.identity_input), output_vertex))
    contexts: set[ContextUse] = set()
    evidence: set[EvidencePromise] = set()
    state_effects: set[StateEffect] = set()
    models: set[ModelRequirement] = set()
    ceiling: list[int | None] = [0, 0, 0, 0]
    declared_items = frozenset(_map_item_ports(_visits(external_outcome.evidence)))
    for node in _visits(reachable):
        operation = operations[node]
        outcome = assignment[node]
        input_names = {port.name for port in _visits(operation.inputs)}
        for use in _visits(outcome.context):
            endpoint = _project_one(
                _walk_interface_inputs(
                    ("in" if use.port in input_names else "out", node, use.port),
                    identity_edges,
                    backward=True,
                )
            )
            contexts.add(ContextUse(port=endpoint, meaning=use.meaning, capture=use.capture))
        for promise in _visits(outcome.evidence):
            consumed = _project_consumed_ports(
                promise.consumed_ports,
                node=node,
                identity_edges=identity_edges,
                dependency_edges=dependency_edges,
                declared=declared_items,
                assignment=assignment,
            )
            subject = _project_evidence_port(
                promise.subject_port,
                node=node,
                input_port=promise.subject_port in input_names,
                edges=identity_edges,
                declared=declared_items,
                assignment=assignment,
                allow_output_alias=True,
            )
            evidence.add(
                EvidencePromise(
                    name=promise.name,
                    meaning=promise.meaning,
                    subject_port=subject,
                    consumed_ports=consumed,
                    coverage=promise.coverage,
                )
            )
        state_effects.update(_visits(outcome.state_effects))
        models.update(_visits(outcome.model_requirements))
        values = outcome.ceiling
        ceiling[0] = cast(int, ceiling[0]) + values.max_activations
        ceiling[1] = cast(int, ceiling[1]) + values.max_model_requests
        for index, bound in ((2, values.max_input_bytes), (3, values.max_output_bytes)):
            prior = ceiling[index]
            ceiling[index] = None if prior is None or bound is None else prior + bound
    dependencies = {item.output: item for item in _visits(interface.output_dependencies)}
    for output in _visits(produced_external):
        vertex = ("wo", None, output)
        influence = _walk_interface_inputs(vertex, dependency_edges, backward=True)
        identity = _walk_interface_inputs(vertex, identity_edges, backward=True)
        if len(identity) > 1:
            _reject(ValidationCode.CONTRADICTORY)
        derived_identity = next(iter(identity)) if identity else None
        declared = dependencies[output]
        if not _equal(declared.inputs, influence) or not _equal(declared.identity_input, derived_identity):
            _reject(ValidationCode.CONTRADICTORY)
    if (
        not _equal(external_outcome.produced_ports, produced_external)
        or not _equal(external_outcome.context, contexts)
        or not _equal(external_outcome.evidence, evidence)
        or not _equal(external_outcome.state_effects, state_effects)
        or not _equal(external_outcome.model_requirements, models)
    ):
        _reject(ValidationCode.CONTRADICTORY)
    actual = external_outcome.ceiling
    if any(
        supplied is not None and (derived is None or supplied < derived)
        for supplied, derived in zip(
            (actual.max_activations, actual.max_model_requests, actual.max_input_bytes, actual.max_output_bytes),
            ceiling,
            strict=True,
        )
    ):
        _reject(ValidationCode.CONTRADICTORY)


def _outcome_options(
    nodes: tuple[Node, ...],
    mappings: tuple[OutcomeBinding, ...],
    choices: tuple[ChoiceDecl, ...],
    interface: OperationSpec,
) -> list[tuple[OutcomeSpec, ...]]:
    """Quotient only names that no path consumer can observe.

    Selectors, exposed outcomes and subgraph interfaces retain all names.
    Dynamic evidence endpoints can observe an expander's name; these graphs
    conservatively retain the complete product. All remaining outcome fields
    participate in the signature, including failure labels and resource cost.
    """
    observed = _union(
        {binding.source.node for binding in _visits(mappings)}, {choice.selector for choice in _visits(choices)}
    )
    operations = [interface, *_visits((node.operation for node in _visits(nodes)))]
    dynamic_endpoints = [
        isinstance(port, MapItemPort)
        for operation in _visits(operations)
        for outcome in _visits(operation.outcomes)
        for promise in _visits(outcome.evidence)
        for port in _visits((promise.subject_port, *_visits(promise.consumed_ports)))
    ]
    dynamic = any(dynamic_endpoints)
    options: list[tuple[OutcomeSpec, ...]] = []
    for node in _visits(nodes):
        if dynamic or isinstance(node, SubgraphNode) or node.id in observed:
            options.append(node.operation.outcomes)
            continue
        representatives: dict[tuple[object, ...], OutcomeSpec] = {}
        for outcome in _visits(node.operation.outcomes):
            key = _outcome_signature(outcome)
            representatives.setdefault(key, outcome)
        options.append(tuple(_visits(representatives.values())))
    return options


def _outcome_signature(outcome: OutcomeSpec) -> tuple[object, ...]:
    """Build a metered key retaining every outcome field except its name."""
    return (
        outcome.category,
        frozenset(port for port in _visits(outcome.produced_ports)),
        frozenset((use.port, use.meaning, use.capture) for use in _visits(outcome.context)),
        frozenset(
            (
                promise.name,
                promise.meaning,
                promise.subject_port,
                frozenset(port for port in _visits(promise.consumed_ports)),
                frozenset((atom.kind, atom.name) for atom in _visits(promise.coverage)),
            )
            for promise in _visits(outcome.evidence)
        ),
        frozenset((effect.kind, effect.name) for effect in _visits(outcome.state_effects)),
        frozenset((model.capability, model.revision) for model in _visits(outcome.model_requirements)),
        outcome.ceiling,
    )
