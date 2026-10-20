# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile retained artifact provenance for qualification."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from anonymizer.engine.graph_sdk._activation_topology import _publication_outcome
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_topology import _source_activation
from anonymizer.engine.graph_sdk._workflow_values import NodeOutputRef, SubgraphNode, WorkflowInputRef
from anonymizer.engine.graph_sdk.executor import (
    ArtifactProvenanceFact,
    BoundInputKey,
    InitialCollectionKey,
    MapItemKey,
    OperationOutputKey,
    ProvenanceKey,
    RootInputKey,
)
from anonymizer.engine.graph_sdk.requests import (
    TextCollectionValue,
)

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk.qualification import _Qualification


class _ProvenanceValidator:
    """Validate provenance against the invocation's reconciliation indexes."""

    def __init__(self, context: _Qualification) -> None:
        self.context = context

    def validate_provenance(self) -> None:
        self._validate_input_parents()
        self._validate_passthrough_parents()
        for fact in self.context.result.provenance:
            self._producer(fact)
        self._provenance_acyclic()
        outputs = [(item.target, item.port) for item in self.context.result.final_outputs]
        if len(outputs) != len(set(outputs)):
            reject(EffectCode.DUPLICATE)
        for output in self.context.result.final_outputs:
            producer = self.context.facts.provenance.get(output.producer)
            if producer is None:
                reject(EffectCode.MISSING)
            if producer.key.target != output.target:
                reject(EffectCode.FOREIGN_OWNER)
            bindings = [
                item for item in self.context.prepared.workflow.output_bindings if item.destination.port == output.port
            ]
            if len(bindings) != 1:
                reject(EffectCode.MISSING)
            source = bindings[0].source
            if isinstance(source, NodeOutputRef):
                if not isinstance(output.producer, OperationOutputKey):
                    reject(EffectCode.CONTRADICTORY)
                source_entry = self.context.facts.entries[output.producer.activation]
                if source_entry.template != source.node or output.producer.port != source.port:
                    reject(EffectCode.CONTRADICTORY)
            elif isinstance(source, WorkflowInputRef) and output.producer != RootInputKey(
                target=output.target, port=source.port
            ):
                reject(EffectCode.CONTRADICTORY)
            expected = _publication_outcome(
                self.context.prepared.workflow,
                {
                    entry.template: entry
                    for key, entry in self.context.facts.entries.items()
                    if key.parent is None and self.context.facts.owners.get(key) == output.target
                },
            )
            if expected != output.outcome:
                reject(EffectCode.CONTRADICTORY)
            if output.outcome not in {
                item.name
                for item in self.context.prepared.workflow.interface.outcomes
                if output.port in item.produced_ports
            }:
                reject(EffectCode.CONTRADICTORY)

    def _validate_input_parents(self) -> None:
        expected = {
            (fact.target, fact.activation, fact.port)
            for fact in self.context.result.ports
            if fact.activation in self.context.facts.entries
            and not isinstance(self.context.nodes[self.context.facts.entries[fact.activation].template], SubgraphNode)
            and fact.port
            in {
                item.name
                for item in self.context.nodes[self.context.facts.entries[fact.activation].template].operation.inputs
            }
        }
        if expected - self.context.facts.input_parents.keys():
            reject(EffectCode.MISSING)
        for parent in self.context.facts.input_parents.values():
            if parent not in self.context.facts.provenance:
                reject(EffectCode.MISSING)
            if isinstance(parent, BoundInputKey):
                bound = self.context.admitted.execution.context.bound_context
                if bound is None:
                    reject(EffectCode.MISSING)
                if parent.binding_artifact.declaration.binding != bound.receipt.binding:
                    reject(EffectCode.FOREIGN_OWNER)
                if not any(item.identity == parent.binding_artifact.declaration for item in bound.receipt.sources):
                    reject(EffectCode.MISSING)

    def _validate_passthrough_parents(self) -> None:
        expected = set()
        for fact in self.context.result.provenance:
            key = fact.key
            if not isinstance(key, OperationOutputKey) or key.activation not in self.context.facts.entries:
                continue
            node = self.context.nodes[self.context.facts.entries[key.activation].template]
            if isinstance(node, SubgraphNode) and any(
                binding.destination.port == key.port and isinstance(binding.source, WorkflowInputRef)
                for binding in node.body.output_bindings
            ):
                expected.add((key.target, key.activation, key.port))
        if expected - self.context.facts.passthrough_parents.keys():
            reject(EffectCode.MISSING)
        if self.context.facts.passthrough_parents.keys() - expected:
            reject(EffectCode.CONTRADICTORY)
        if any(
            parent not in self.context.facts.provenance for parent in self.context.facts.passthrough_parents.values()
        ):
            reject(EffectCode.MISSING)

    def _provenance_acyclic(self) -> None:
        remaining = {key: len(fact.parents) for key, fact in self.context.facts.provenance.items()}
        children: dict[ProvenanceKey, list[ProvenanceKey]] = {key: [] for key in remaining}
        for key, fact in self.context.facts.provenance.items():
            for parent in fact.parents:
                children[parent].append(key)
        ready = [key for key, count in remaining.items() if count == 0]
        visited = 0
        while ready:
            key = ready.pop()
            visited += 1
            for child in children[key]:
                remaining[child] -= 1
                if remaining[child] == 0:
                    ready.append(child)
        if visited != len(remaining):
            reject(EffectCode.CONTRADICTORY)

    def _producer(self, fact: ArtifactProvenanceFact) -> None:
        key = fact.key
        if isinstance(key, OperationOutputKey):
            entry = self.context.facts.entries.get(key.activation)
            if entry is None:
                reject(EffectCode.MISSING)
            outcome = next(
                (item for item in self.context.nodes[entry.template].operation.outcomes if item.name == entry.outcome),
                None,
            )
            if outcome is None or key.port not in outcome.produced_ports:
                reject(EffectCode.CONTRADICTORY)
            node = self.context.nodes[entry.template]
            if isinstance(node, SubgraphNode):
                # Admission allows one binding per output port and materialization records exactly its source
                # (a passthrough's retained parent, or the body node's output) as the single parent.
                source = next(item for item in node.body.output_bindings if item.destination.port == key.port).source
                parent = self.context.facts.provenance[next(iter(fact.parents))]
                if not isinstance(source, WorkflowInputRef):
                    parent_key = cast(OperationOutputKey, parent.key)
                    state = self.context.result.states[self.context.order.index(key.target)]
                    expected_activation = _source_activation(state, key.activation, source.node)
                    source_entry = self.context.facts.entries.get(parent_key.activation)
                    if (
                        source_entry is None
                        or parent_key.activation != expected_activation
                        or parent_key.target != key.target
                        or source_entry.template != source.node
                        or parent_key.port != source.port
                    ):
                        reject(EffectCode.CONTRADICTORY)
                if parent.artifact != fact.artifact:
                    reject(EffectCode.CONTRADICTORY)
            else:
                dependency = next(item for item in node.operation.output_dependencies if item.output == key.port)
                input_keys = [(key.target, key.activation, name) for name in dependency.inputs]
                if any(item not in self.context.facts.input_parents for item in input_keys):
                    reject(EffectCode.MISSING)
                if dependency.identity_input is not None:
                    identity_parent = self.context.facts.input_parents[
                        key.target, key.activation, dependency.identity_input
                    ]
                    if fact.artifact != self.context.facts.provenance[identity_parent].artifact:
                        reject(EffectCode.CONTRADICTORY)
        elif isinstance(key, RootInputKey):
            if fact.parents or not any(
                item.target == key.target and item.port == key.port for item in self.context.prepared.bound_inputs
            ):
                reject(EffectCode.CONTRADICTORY)
        elif isinstance(key, (BoundInputKey, InitialCollectionKey)):
            self._bound_producer(fact, key)
        elif isinstance(key, MapItemKey):
            self._map_producer(fact, key)

    def _bound_producer(self, fact: ArtifactProvenanceFact, key: BoundInputKey | InitialCollectionKey) -> None:
        bound = self.context.admitted.execution.context.bound_context
        if bound is None:
            reject(EffectCode.CONTRADICTORY)
        declaration = key.binding_artifact.declaration if isinstance(key, BoundInputKey) else key.declaration
        if declaration.binding != bound.receipt.binding:
            reject(EffectCode.FOREIGN_OWNER)
        # Bound and initial-collection facts are materialized from the admitted bound artifacts of their own
        # declaration (with exactly those parents), so only the collection payload is re-joined here.
        if isinstance(key, InitialCollectionKey):
            items = [item for item in bound.artifacts if item.reference.declaration == declaration]
            value = self.context.facts.values[fact.artifact]
            if not isinstance(value, TextCollectionValue) or {(item.key, item.version) for item in value.items} != {
                (item.reference.key, item.reference.version) for item in items
            }:
                reject(EffectCode.CONTRADICTORY)

    def _map_producer(self, fact: ArtifactProvenanceFact, key: MapItemKey) -> None:
        # Map item facts are materialized for committed members of this expander in the same target state, from
        # the first matching map declaration, with the item zipped (in occurrence order) from the published
        # collection. The parent join stays: it reads the caller-declared publication retained by the plan.
        expander = self.context.facts.entries[key.expander]
        declaration = next(
            (
                item
                for scope in self.context.prepared.workflow.bodies()
                for item in scope.maps
                if item.expander == expander.template
            ),
            None,
        )
        publication = next(
            (
                item
                for item in self.context.admitted.execution.map_expansions
                if item.expander == expander.template and item.outcome == expander.outcome
            ),
            None,
        )
        if declaration is None or publication is None:
            reject(EffectCode.MISSING)
        parent = OperationOutputKey(activation=key.expander, target=key.target, port=publication.membership_port)
        if fact.parents != frozenset({parent}):
            reject(EffectCode.CONTRADICTORY)
        collection = self.context.facts.values[self.context.facts.provenance[parent].artifact]
        if not isinstance(collection, TextCollectionValue):
            reject(EffectCode.CONTRADICTORY)
        state = self.context.result.states[self.context.order.index(key.target)]
        expansion = next((item for item in state.expansions if item.parent == key.expander), None)
        if expansion is None or key.member not in expansion.members:
            reject(EffectCode.MISSING)
