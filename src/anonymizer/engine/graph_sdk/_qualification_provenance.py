# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reconcile retained artifact provenance for qualification."""

from __future__ import annotations

from typing import TYPE_CHECKING

from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_topology import _source_activation
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
from anonymizer.graph.workflow import (
    NodeOutputRef,
    SubgraphNode,
    WorkflowInputRef,
)

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk.qualification import _Qualification


class _ProvenanceValidator:
    """Validate provenance against the invocation's reconciliation indexes."""

    def __init__(self, context: _Qualification) -> None:
        self.context = context

    def validate_provenance(self) -> None:
        if len(self.context.result.provenance) > self.context.admitted.limits.max_port_facts:
            reject(EffectCode.LIMIT_EXCEEDED)
        for fact in self.context.result.provenance:
            if fact.key in self.context.provenance:
                reject(EffectCode.DUPLICATE)
            if (
                fact.artifact.invocation != self.context.result.record.invocation
                or fact.key.target not in self.context.order
            ):
                reject(EffectCode.FOREIGN_OWNER)
            if fact.artifact not in self.context.result.record.artifacts:
                reject(EffectCode.MISSING)
            self.context.provenance[fact.key] = fact
        self._validate_input_parents()
        self._validate_passthrough_parents()
        for fact in self.context.result.provenance:
            if not fact.parents <= self.context.provenance.keys():
                reject(EffectCode.MISSING)
            self._producer(fact)
        self._validate_artifact_allocations()
        self._provenance_acyclic()
        outputs = [(item.target, item.port) for item in self.context.result.final_outputs]
        if len(outputs) != len(set(outputs)):
            reject(EffectCode.DUPLICATE)
        for output in self.context.result.final_outputs:
            if output.target not in self.context.order or output.candidate.target != output.target:
                reject(EffectCode.FOREIGN_OWNER)
            producer = self.context.provenance.get(output.producer)
            if producer is None:
                reject(EffectCode.MISSING)
            if producer.key.target != output.target:
                reject(EffectCode.FOREIGN_OWNER)
            if producer.artifact != output.candidate.artifact:
                reject(EffectCode.CONTRADICTORY)
            bindings = [
                item
                for item in self.context.prepared.workflow.workflow.output_bindings
                if item.destination.port == output.port
            ]
            if len(bindings) != 1:
                reject(EffectCode.MISSING)
            source = bindings[0].source
            if isinstance(source, NodeOutputRef):
                if not isinstance(output.producer, OperationOutputKey):
                    reject(EffectCode.CONTRADICTORY)
                source_entry = self.context.entries[output.producer.activation]
                if source_entry.template != source.node or output.producer.port != source.port:
                    reject(EffectCode.CONTRADICTORY)
            elif isinstance(source, WorkflowInputRef) and output.producer != RootInputKey(
                target=output.target, port=source.port
            ):
                reject(EffectCode.CONTRADICTORY)
            actual_outcomes = {
                binding.destination.outcome
                for binding in self.context.prepared.workflow.workflow.outcome_bindings
                for key, entry in self.context.entries.items()
                if self.context.owners[key] == output.target
                and entry.template == binding.source.node
                and entry.status == "success"
                and entry.outcome == binding.source.outcome
            }
            if actual_outcomes != {output.outcome}:
                reject(EffectCode.CONTRADICTORY)
            if output.outcome not in {
                item.name
                for item in self.context.prepared.workflow.workflow.interface.outcomes
                if output.port in item.produced_ports
            }:
                reject(EffectCode.CONTRADICTORY)

    def _validate_artifact_allocations(self) -> None:
        allocated: dict[tuple[object, ...], int] = {}
        owners: dict[int, tuple[object, ...]] = {}
        for fact in self.context.result.provenance:
            key = fact.key
            if isinstance(key, BoundInputKey):
                lineage = (BoundInputKey, key.binding_artifact.declaration, key.binding_artifact.key)
                version = key.binding_artifact.version
            elif isinstance(key, MapItemKey):
                lineage = (MapItemKey, key.expander, key.target, key.item_key)
                version = key.item_version
            elif isinstance(key, OperationOutputKey):
                node = self.context.nodes[self.context.entries[key.activation].template]
                if isinstance(node, SubgraphNode):
                    continue
                dependency = next(item for item in node.operation.output_dependencies if item.output == key.port)
                if dependency.identity_input is not None:
                    continue
                lineage = (OperationOutputKey, key)
                version = 1
            else:
                lineage = (type(key), key)
                version = 1
            if (
                fact.artifact.version != version
                or allocated.setdefault(lineage, fact.artifact.key) != fact.artifact.key
                or owners.setdefault(fact.artifact.key, lineage) != lineage
            ):
                reject(EffectCode.CONTRADICTORY)

    def _validate_input_parents(self) -> None:
        for target, activation, port, parent in self.context.result._input_parents:
            if (
                activation not in self.context.entries
                or target != self.context.owners[activation]
                or parent.target != target
            ):
                reject(EffectCode.FOREIGN_OWNER)
            key = (target, activation, port)
            if key in self.context.input_parents:
                reject(EffectCode.DUPLICATE)
            self.context.input_parents[key] = parent
        expected = {
            (fact.target, fact.activation, fact.port)
            for fact in self.context.result.ports
            if fact.activation in self.context.entries
            and not isinstance(self.context.nodes[self.context.entries[fact.activation].template], SubgraphNode)
            and fact.port
            in {
                item.name
                for item in self.context.nodes[self.context.entries[fact.activation].template].operation.inputs
            }
        }
        if expected - self.context.input_parents.keys():
            reject(EffectCode.MISSING)
        if self.context.input_parents.keys() - expected:
            reject(EffectCode.CONTRADICTORY)
        for (_, activation, port), parent in self.context.input_parents.items():
            source = self.context.provenance.get(parent)
            if source is None:
                reject(EffectCode.MISSING)
            if source.artifact != self.context.facts.ports[activation, port].artifact:
                reject(EffectCode.CONTRADICTORY)
            if isinstance(parent, BoundInputKey):
                bound = self.context.admitted.execution.context.bound_context
                if bound is None:
                    reject(EffectCode.MISSING)
                declaration = next(
                    (
                        item.declaration
                        for item in bound.receipt.sources
                        if item.identity == parent.binding_artifact.declaration
                    ),
                    None,
                )
                if declaration is None:
                    reject(EffectCode.FOREIGN_OWNER)
                if declaration.version_selection == "latest" and parent.binding_artifact.version != max(
                    (
                        item.reference.version
                        for item in bound.artifacts
                        if item.reference.declaration == parent.binding_artifact.declaration
                        and item.reference.key == parent.binding_artifact.key
                    ),
                    default=0,
                ):
                    reject(EffectCode.CONTRADICTORY)

    def _validate_passthrough_parents(self) -> None:
        for target, activation, port, parent in self.context.result._passthrough_parents:
            if (
                activation not in self.context.entries
                or target != self.context.owners[activation]
                or parent.target != target
            ):
                reject(EffectCode.FOREIGN_OWNER)
            key = (target, activation, port)
            if key in self.context.passthrough_parents:
                reject(EffectCode.DUPLICATE)
            self.context.passthrough_parents[key] = parent
        expected = set()
        for fact in self.context.result.provenance:
            key = fact.key
            if not isinstance(key, OperationOutputKey) or key.activation not in self.context.entries:
                continue
            node = self.context.nodes[self.context.entries[key.activation].template]
            if isinstance(node, SubgraphNode) and any(
                binding.destination.port == key.port and isinstance(binding.source, WorkflowInputRef)
                for binding in node.body.output_bindings
            ):
                expected.add((key.target, key.activation, key.port))
        if expected - self.context.passthrough_parents.keys():
            reject(EffectCode.MISSING)
        if self.context.passthrough_parents.keys() - expected:
            reject(EffectCode.CONTRADICTORY)
        if any(parent not in self.context.provenance for parent in self.context.passthrough_parents.values()):
            reject(EffectCode.MISSING)

    def _provenance_acyclic(self) -> None:
        remaining = {key: len(fact.parents) for key, fact in self.context.provenance.items()}
        children: dict[ProvenanceKey, list[ProvenanceKey]] = {key: [] for key in remaining}
        for key, fact in self.context.provenance.items():
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
            entry = self.context.entries.get(key.activation)
            if entry is None:
                reject(EffectCode.MISSING)
            if self.context.owners[key.activation] != key.target:
                reject(EffectCode.FOREIGN_OWNER)
            port = self.context.facts.ports.get((key.activation, key.port))
            if port is None or port.artifact != fact.artifact or port.node != entry.template:
                reject(EffectCode.CONTRADICTORY)
            outcome = next(
                (item for item in self.context.nodes[entry.template].operation.outcomes if item.name == entry.outcome),
                None,
            )
            if outcome is None or key.port not in outcome.produced_ports:
                reject(EffectCode.CONTRADICTORY)
            node = self.context.nodes[entry.template]
            if isinstance(node, SubgraphNode):
                bindings = [item for item in node.body.output_bindings if item.destination.port == key.port]
                if len(bindings) != 1 or len(fact.parents) != 1:
                    reject(EffectCode.CONTRADICTORY)
                source = bindings[0].source
                parent = self.context.provenance[next(iter(fact.parents))]
                if isinstance(source, WorkflowInputRef):
                    expected_key = self.context.passthrough_parents[key.target, key.activation, key.port]
                    if parent.key != expected_key:
                        reject(EffectCode.CONTRADICTORY)
                else:
                    if not isinstance(parent.key, OperationOutputKey):
                        reject(EffectCode.CONTRADICTORY)
                    state = self.context.result.states[self.context.order.index(key.target)]
                    expected_activation = _source_activation(state, key.activation, source.node)
                    source_entry = self.context.entries.get(parent.key.activation)
                    if (
                        source_entry is None
                        or parent.key.activation != expected_activation
                        or parent.key.target != key.target
                        or source_entry.template != source.node
                        or parent.key.port != source.port
                    ):
                        reject(EffectCode.CONTRADICTORY)
                if parent.artifact != fact.artifact or fact.decision != parent.decision:
                    reject(EffectCode.CONTRADICTORY)
            else:
                dependency = next(item for item in node.operation.output_dependencies if item.output == key.port)
                input_keys = [(key.target, key.activation, name) for name in dependency.inputs]
                if any(item not in self.context.input_parents for item in input_keys):
                    reject(EffectCode.MISSING)
                expected = frozenset(self.context.input_parents[item] for item in input_keys)
                if dependency.identity_input is not None:
                    identity_parent = self.context.input_parents[key.target, key.activation, dependency.identity_input]
                    if fact.artifact != self.context.provenance[identity_parent].artifact:
                        reject(EffectCode.CONTRADICTORY)
                if fact.parents != expected or fact.decision != any(
                    item.node == entry.template for item in self.context.admitted.execution.decisions
                ):
                    reject(EffectCode.CONTRADICTORY)
        elif isinstance(key, RootInputKey):
            if (
                fact.parents
                or fact.decision
                or not any(
                    item.target == key.target and item.port == key.port for item in self.context.prepared.bound_inputs
                )
            ):
                reject(EffectCode.CONTRADICTORY)
        elif isinstance(key, (BoundInputKey, InitialCollectionKey)):
            self._bound_producer(fact, key)
        elif isinstance(key, MapItemKey):
            self._map_producer(fact, key)

    def _bound_producer(self, fact: ArtifactProvenanceFact, key: BoundInputKey | InitialCollectionKey) -> None:
        bound = self.context.admitted.execution.context.bound_context
        if bound is None or fact.decision:
            reject(EffectCode.CONTRADICTORY)
        declaration = key.binding_artifact.declaration if isinstance(key, BoundInputKey) else key.declaration
        source = next((item for item in bound.receipt.sources if item.identity == declaration), None)
        if source is None:
            reject(EffectCode.MISSING)
        if (source.declaration.target, source.declaration.node, source.declaration.port) != (
            key.target,
            key.node,
            key.port,
        ):
            reject(EffectCode.FOREIGN_OWNER)
        items = [item for item in bound.artifacts if item.reference.declaration == declaration]
        if isinstance(key, BoundInputKey):
            if fact.parents or not any(
                item.reference == key.binding_artifact
                and (item.target, item.node, item.port) == (key.target, key.node, key.port)
                for item in items
            ):
                reject(EffectCode.CONTRADICTORY)
        else:
            expected = frozenset(
                BoundInputKey(target=key.target, node=key.node, port=key.port, binding_artifact=item.reference)
                for item in items
            )
            if source.declaration.materialization.kind != "collection" or fact.parents != expected:
                reject(EffectCode.CONTRADICTORY)
            value = dict(self.context.result.artifacts)[fact.artifact]
            if not isinstance(value, TextCollectionValue) or {(item.key, item.version) for item in value.items} != {
                (item.reference.key, item.reference.version) for item in items
            }:
                reject(EffectCode.CONTRADICTORY)

    def _map_producer(self, fact: ArtifactProvenanceFact, key: MapItemKey) -> None:
        member = self.context.entries.get(key.member)
        expander = self.context.entries.get(key.expander)
        if member is None or expander is None:
            reject(EffectCode.MISSING)
        if (
            key.member.parent != key.expander
            or self.context.owners[key.member] != key.target
            or self.context.owners[key.expander] != key.target
        ):
            reject(EffectCode.FOREIGN_OWNER)
        declaration = next(
            (
                item
                for scope in self.context.prepared.workflow.scopes
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
        if declaration.member != member.template or declaration.item_input != key.port or fact.decision:
            reject(EffectCode.CONTRADICTORY)
        parent = OperationOutputKey(activation=key.expander, target=key.target, port=publication.membership_port)
        if fact.parents != frozenset({parent}):
            reject(EffectCode.CONTRADICTORY)
        collection = dict(self.context.result.artifacts)[self.context.provenance[parent].artifact]
        if not isinstance(collection, TextCollectionValue):
            reject(EffectCode.CONTRADICTORY)
        if (key.item_key, key.item_version) not in {(item.key, item.version) for item in collection.items}:
            reject(EffectCode.MISSING)
        state = self.context.result.states[self.context.order.index(key.target)]
        expansion = next((item for item in state.expansions if item.parent == key.expander), None)
        if expansion is None or key.member not in expansion.members:
            reject(EffectCode.MISSING)
        children = sorted(expansion.members, key=lambda item: item.occurrence)
        index = children.index(key.member)
        if index >= len(collection.items) or (key.item_key, key.item_version) != (
            collection.items[index].key,
            collection.items[index].version,
        ):
            reject(EffectCode.CONTRADICTORY)
