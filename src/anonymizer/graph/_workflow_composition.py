# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow path and interface composition."""

from __future__ import annotations

import itertools
from typing import TypeAlias

from anonymizer.graph._values import ValidationCode
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
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current[0] == endpoint_kind:
            endpoints.add(current[2])
        for source, destination in edges:
            if backward and destination == current:
                pending.append(source)
            elif not backward and source == current:
                pending.append(destination)
    return frozenset(endpoints)


def _project_one(endpoints: frozenset[str]) -> str:
    if not endpoints:
        _reject(ValidationCode.MISSING)
    if len(endpoints) > 1:
        _reject(ValidationCode.CONTRADICTORY)
    return next(iter(endpoints))


def _walk_interface_inputs(start: Vertex, edges: set[tuple[Vertex, Vertex]], *, backward: bool) -> frozenset[str]:
    return _walk_endpoints(start, edges, backward=backward, endpoint_kind="wi") | _walk_endpoints(
        start, edges, backward=backward, endpoint_kind="ci"
    )


def _has_cycle(selected: frozenset[NodeId], edges: frozenset[SequenceEdge]) -> bool:
    remaining = set(selected)
    while remaining:
        roots = {
            node for node in remaining if not any(edge.after == node and edge.before in remaining for edge in edges)
        }
        if not roots:
            return True
        remaining -= roots
    return False


def _selected_nodes(
    node_ids: frozenset[NodeId], choices: tuple[ChoiceDecl, ...], assignment: dict[NodeId, OutcomeSpec]
) -> frozenset[NodeId]:
    branch_members = {member for choice in choices for branch in choice.branches for member in branch.members}
    selected = set(node_ids - branch_members)
    changed = True
    while changed:
        changed = False
        for choice in choices:
            if choice.selector not in selected:
                continue
            outcome = assignment[choice.selector].name
            for branch in choice.branches:
                if outcome in branch.outcomes:
                    before = len(selected)
                    selected.update(branch.members)
                    changed |= len(selected) != before
    return frozenset(selected)


def validate_dynamic_input_summaries(
    workflow: AdmittedWorkflow,
    replacements: tuple[InputBinding, ...],
) -> None:
    """Recheck retained interface summaries after nonidentity dynamic input replacement."""
    if not isinstance(workflow, AdmittedWorkflow):
        _reject(ValidationCode.INVALID_TYPE)
    _tuple_of(replacements, InputBinding)
    destinations = [item.destination for item in replacements]
    if len(destinations) != len(set(destinations)):
        _reject(ValidationCode.DUPLICATE)
    reachable: list[AdmittedWorkflow] = []
    pending = [workflow]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        reachable.append(current)
        pending.extend(item.body for item in current.nodes if isinstance(item, SubgraphNode))
    owners = {item.workflow: item for item in reachable}
    grouped: dict[WorkflowId, list[InputBinding]] = {}
    for replacement in replacements:
        if replacement.destination.node.workflow not in owners:
            _reject(ValidationCode.FOREIGN_OWNER)
        grouped.setdefault(replacement.destination.node.workflow, []).append(replacement)
    for owner, values in grouped.items():
        current = owners[owner]
        replaced = {item.destination for item in values}
        if any(
            sum(binding.destination == destination for binding in current.input_bindings) != 1
            for destination in replaced
        ):
            _reject(ValidationCode.MISSING)
        inputs = tuple(binding for binding in current.input_bindings if binding.destination not in replaced) + tuple(
            values
        )
        _validate_paths(
            current.interface,
            tuple(current.nodes),
            inputs,
            tuple(current.output_bindings),
            tuple(current.outcome_bindings),
            tuple(current.sequence),
            tuple(current.choices),
            check_cycles=False,
            nonidentity_destinations=frozenset(replaced),
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
    operations = {node.id: node.operation for node in nodes}
    node_ids = frozenset(operations)
    edges = frozenset(sequence)
    if check_cycles and _has_cycle(node_ids, edges):
        _reject(ValidationCode.CYCLE)
    outcome_options = [operation.outcomes for operation in operations.values()]
    ids = tuple(operations)
    reached_interface_outcomes: set[str] = set()
    for outcomes in itertools.product(*outcome_options):
        assignment = dict(zip(ids, outcomes, strict=True))
        selected = _selected_nodes(node_ids, choices, assignment)
        selected_edges = frozenset(edge for edge in edges if edge.before in selected and edge.after in selected)
        if check_cycles and _has_cycle(selected, selected_edges):
            _reject(ValidationCode.CYCLE)
        reachable: set[NodeId] = set()
        changed = True
        while changed:
            changed = False
            for node in selected - reachable:
                predecessors = {edge.before for edge in selected_edges if edge.after == node}
                if not predecessors <= reachable:
                    continue
                bindings = [binding for binding in input_bindings if binding.destination.node == node]
                if all(
                    isinstance(binding.source, (WorkflowInputRef, ContextInputRef))
                    or (
                        binding.source.node in reachable
                        and binding.source.port in assignment[binding.source.node].produced_ports
                    )
                    for binding in bindings
                ):
                    reachable.add(node)
                    changed = True
        if reachable != set(selected):
            _reject(ValidationCode.MISSING)
        sinks = {
            node
            for node in reachable
            if not any(edge.before == node and edge.after in reachable for edge in selected_edges)
        }
        if len(sinks) != 1:
            _reject(ValidationCode.MISSING)
        sink = next(iter(sinks))
        sink_outcome = assignment[sink]
        mappings = [
            binding
            for binding in outcome_bindings
            if binding.source.node == sink and binding.source.outcome == sink_outcome.name
        ]
        if len(mappings) != 1:
            _reject(ValidationCode.MISSING)
        external_outcome_name = mappings[0].destination.outcome
        reached_interface_outcomes.add(external_outcome_name)
        external_outcome = next(outcome for outcome in interface.outcomes if outcome.name == external_outcome_name)
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
    if reached_interface_outcomes != {outcome.name for outcome in interface.outcomes}:
        _reject(ValidationCode.MISSING)


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
        for endpoint in declared
        if not endpoint.path
        and endpoint.member == node
        and endpoint.item_input == port
        and endpoint.expander in assignment
        and assignment[endpoint.expander].name == endpoint.expansion_outcome
    }
    if dynamic:
        if len(dynamic) != 1 or not input_port:
            _reject(ValidationCode.CONTRADICTORY)
        return next(iter(dynamic))
    if input_port:
        inputs = _walk_interface_inputs(("in", node, port), edges, backward=True)
        if inputs or not allow_output_alias:
            return _project_one(inputs)
        identity = edges | {(destination, source) for source, destination in edges}
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
    for binding in input_bindings:
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
    for binding in output_bindings:
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
    for node in reachable:
        operation = operations[node]
        produced = assignment[node].produced_ports
        for dependency in operation.output_dependencies:
            if dependency.output not in produced:
                continue
            output_vertex = ("out", node, dependency.output)
            for input_port in dependency.inputs:
                dependency_edges.add((("in", node, input_port), output_vertex))
            if dependency.identity_input is not None:
                identity_edges.add((("in", node, dependency.identity_input), output_vertex))
    contexts: set[ContextUse] = set()
    evidence: set[EvidencePromise] = set()
    state_effects: set[StateEffect] = set()
    models: set[ModelRequirement] = set()
    ceiling = [0, 0, 0, 0]
    declared_items = frozenset(
        port
        for promise in external_outcome.evidence
        for port in (promise.subject_port, *promise.consumed_ports)
        if isinstance(port, MapItemPort)
    )
    for node in reachable:
        operation = operations[node]
        outcome = assignment[node]
        input_names = {port.name for port in operation.inputs}
        for use in outcome.context:
            endpoint = _project_one(
                _walk_interface_inputs(
                    ("in" if use.port in input_names else "out", node, use.port),
                    identity_edges,
                    backward=True,
                )
            )
            contexts.add(ContextUse(port=endpoint, meaning=use.meaning, capture=use.capture))
        for promise in outcome.evidence:
            consumed = frozenset(
                _project_evidence_port(
                    port,
                    node=node,
                    input_port=True,
                    edges=identity_edges,
                    declared=declared_items,
                    assignment=assignment,
                )
                for port in promise.consumed_ports
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
        state_effects.update(outcome.state_effects)
        models.update(outcome.model_requirements)
        values = outcome.ceiling
        ceiling[0] += values.max_activations
        ceiling[1] += values.max_model_requests
        ceiling[2] += values.max_input_bytes
        ceiling[3] += values.max_output_bytes
    dependencies = {item.output: item for item in interface.output_dependencies}
    for output in produced_external:
        vertex = ("wo", None, output)
        influence = _walk_interface_inputs(vertex, dependency_edges, backward=True)
        identity = _walk_interface_inputs(vertex, identity_edges, backward=True)
        if len(identity) > 1:
            _reject(ValidationCode.CONTRADICTORY)
        derived_identity = next(iter(identity)) if identity else None
        declared = dependencies[output]
        if declared.inputs != influence or declared.identity_input != derived_identity:
            _reject(ValidationCode.CONTRADICTORY)
    if (
        external_outcome.produced_ports != produced_external
        or external_outcome.context != contexts
        or external_outcome.evidence != evidence
        or external_outcome.state_effects != state_effects
        or external_outcome.model_requirements != models
    ):
        _reject(ValidationCode.CONTRADICTORY)
    actual = external_outcome.ceiling
    if any(
        supplied < derived
        for supplied, derived in zip(
            (actual.max_activations, actual.max_model_requests, actual.max_input_bytes, actual.max_output_bytes),
            ceiling,
            strict=True,
        )
    ):
        _reject(ValidationCode.CONTRADICTORY)
