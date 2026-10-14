# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input and dynamic membership materialization."""

from __future__ import annotations

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
)
from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts
from anonymizer.engine.graph_sdk._execution_topology import (
    _is_mapped_item_input,
    _loop_input_source,
    _source_activation,
)
from anonymizer.engine.graph_sdk._execution_values import (
    _FACT_KEY,
    AdmittedExecutionPlan,
    ArtifactProvenanceFact,
    BoundInputKey,
    ExecutionLimits,
    InitialCollectionKey,
    MapItemKey,
    OperationOutputKey,
    ProvenanceKey,
    RootInputKey,
)
from anonymizer.engine.graph_sdk.context import (
    BoundTextArtifact,
)
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    AssociationInput,
    AssociationResult,
    BindingDeclarationId,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionItem,
    TextCollectionValue,
)
from anonymizer.graph._activation_topology import _depth as _activation_depth
from anonymizer.graph._values import (
    ActivationKey,
    ArtifactRef,
    DatumId,
    InvocationId,
)
from anonymizer.graph.activation import (
    ActivationState,
    ObserveMembership,
    ObserveOverflow,
    advance_activation,
)
from anonymizer.graph.workflow import (
    ContextInputRef,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    SubgraphNode,
    WorkflowInputRef,
)


def _input_source(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    node: NodeId,
    activation: ActivationKey,
    state: ActivationState,
    port: str,
    subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
    facts: _ExecutionFacts,
) -> ArtifactProvenanceFact:
    source = admitted.input_sources.get(NodeInputRef(node=node, port=port))
    if not isinstance(admitted.nodes[node], SubgraphNode):
        source = _loop_input_source(admitted, state, node, activation, port, source)
    direct = facts.source(target, activation, port)
    if direct is None and _is_mapped_item_input(admitted, state, node, activation, port):
        reject(EffectCode.MISSING)
    direct = direct or facts.source(target, node, port)
    if direct is not None:
        return direct
    parent = None
    if isinstance(source, (WorkflowInputRef, ContextInputRef)):
        if activation.parent is not None:
            parent = subgraph_input_parents.get((target, activation.parent, source.port))
        if parent is None and isinstance(source, WorkflowInputRef):
            parent = RootInputKey(target=target, port=source.port)
    elif isinstance(source, NodeOutputRef):
        occurrence = _source_activation(state, activation, source.node)
        if occurrence is not None:
            parent = OperationOutputKey(activation=occurrence, target=target, port=source.port)
    found = facts.provenance.get(parent) if parent is not None else None
    if found is None:
        reject(EffectCode.MISSING)
    return found


def _operation_inputs(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    node: NodeId,
    activation: ActivationKey,
    state: ActivationState,
    association: SemanticAssociation,
    subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
    facts: _ExecutionFacts,
) -> tuple[tuple[AssociationInput, ...], dict[str, ProvenanceKey]]:
    ports: list[PortArtifact] = []
    parents: dict[str, ProvenanceKey] = {}
    for port in admitted.nodes[node].operation.inputs:
        source = _input_source(admitted, target, node, activation, state, port.name, subgraph_input_parents, facts)
        parents[port.name] = source.key
        ports.append(
            PortArtifact(
                port=port.name,
                artifact_type=port.artifact_type,
                artifact=source.artifact,
                value=facts.values[source.artifact],
            )
        )
    return (AssociationInput(association=association, inputs=tuple(ports)),), parents


def _observe_dynamic_membership(
    admitted: AdmittedExecutionPlan,
    state: ActivationState,
    target: DatumId,
    activation: ActivationKey,
    node: NodeId,
    outcome: str | None,
    collection: ArtifactValue,
    facts: _ExecutionFacts,
    limits: ExecutionLimits,
) -> ActivationState:
    scope = next(item for item in state.workflow.scopes if node.workflow == item.workflow.workflow)
    declaration = next((item for item in scope.maps if item.expander == node), None)
    if declaration is None or outcome not in declaration.expansion_outcomes:
        return state
    expansion = next(item for item in admitted.map_expansions if item.expander == node and item.outcome == outcome)
    if not isinstance(collection, TextCollectionValue):
        reject(EffectCode.INVALID_TYPE)
    observed_count = len(collection.items)
    if observed_count > declaration.max_children:
        return advance_activation(
            state=state,
            event=ObserveOverflow(parent=activation, observed_count=observed_count),
        )
    members = sorted(
        (
            seed.activation
            for seed in state.reservations
            if seed.template == declaration.member and seed.activation.parent == activation
        ),
        key=lambda item: item.occurrence,
    )
    selected = members[:observed_count]
    if declaration.item_input is not None:
        if len(facts.values) + len(selected) > limits.max_runtime_artifacts:
            reject(EffectCode.LIMIT_EXCEEDED)
        if limits.max_runtime_artifact_bytes is not None and (
            sum(_artifact_bytes(item) for item in facts.values.values()) + collection.shape.content_bytes
            > limits.max_runtime_artifact_bytes
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        if sum(len(item.parents) for item in facts.provenance.values()) + len(selected) > (
            admitted.assessment_limits.max_provenance_edges
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        parent_key = OperationOutputKey(
            activation=activation,
            target=target,
            port=expansion.membership_port,
        )
        if parent_key not in facts.provenance:
            reject(EffectCode.MISSING)
        version_keys: dict[int, int] = {}
        for member, item in zip(selected, collection.items, strict=True):
            if item.key not in version_keys:
                version_keys[item.key] = facts.next_artifact
                facts.next_artifact += 1
            reference = ArtifactRef(
                invocation=activation.invocation,
                key=version_keys[item.key],
                version=item.version,
            )
            facts.values[reference] = item.value

            key = MapItemKey(
                expander=activation,
                member=member,
                target=target,
                port=declaration.item_input,
                item_key=item.key,
                item_version=item.version,
            )
            facts.add_provenance(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY,
                    key=key,
                    artifact=reference,
                    parents=frozenset({parent_key}),
                    decision=False,
                )
            )
    return advance_activation(
        state=state,
        event=ObserveMembership(parent=activation, members=frozenset(selected), closed=True),
    )


def _membership_value(
    admitted: AdmittedExecutionPlan,
    node: NodeId,
    outcome: str | None,
    results: tuple[AssociationResult, ...],
) -> TextCollectionValue | None:
    declaration = next(
        (item for item in admitted.map_expansions if item.expander == node and item.outcome == outcome), None
    )
    if declaration is None:
        return None
    selected = [
        output.value for result in results for output in result.outputs if output.port == declaration.membership_port
    ]
    if len(selected) != 1 or not isinstance(selected[0], TextCollectionValue):
        reject(EffectCode.MISSING)
    return selected[0]


def _materialize_subgraph_inputs(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    state: ActivationState,
    activation: ActivationKey,
    node: NodeId,
    subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
    facts: _ExecutionFacts,
) -> None:
    subgraph_input_parents.update(
        {
            (target, activation, port.name): _input_source(
                admitted, target, node, activation, state, port.name, subgraph_input_parents, facts
            ).key
            for port in admitted.nodes[node].operation.inputs
        }
    )


def _materialize_structure(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    state: ActivationState,
    facts: _ExecutionFacts,
    input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
    limits: ExecutionLimits,
) -> ActivationState:
    """Close structural outputs and membership in the same publication transaction."""
    while True:
        for entry in sorted(
            state.entries, key=lambda item: (-_activation_depth(item.activation), item.activation.occurrence)
        ):
            if entry.status not in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}:
                continue
            declaration = admitted.nodes[entry.template]
            if not isinstance(declaration, SubgraphNode):
                continue
            if entry.outcome is None:
                continue
            for binding in declaration.body.output_bindings:
                destination = (target, entry.activation, binding.destination.port)
                if facts.artifact(*destination) is not None:
                    continue
                source_key: ProvenanceKey
                if isinstance(binding.source, WorkflowInputRef):
                    parent = input_parents.get((target, entry.activation, binding.source.port))
                    if parent is None:
                        continue
                    source_key = parent
                else:
                    source_activation = _source_activation(state, entry.activation, binding.source.node)
                    source_entry = next(
                        (
                            child
                            for child in state.entries
                            if child.activation == source_activation and child.status == "success"
                        ),
                        None,
                    )
                    if source_entry is None:
                        continue
                    source_key = OperationOutputKey(
                        activation=source_entry.activation,
                        target=target,
                        port=binding.source.port,
                    )
                source_fact = facts.provenance.get(source_key)
                if source_fact is None:
                    continue

                if isinstance(binding.source, WorkflowInputRef):
                    facts.passthrough_parents[destination] = source_key
                key = OperationOutputKey(
                    activation=entry.activation,
                    target=target,
                    port=binding.destination.port,
                )
                output_type = next(
                    item.artifact_type
                    for item in declaration.operation.outputs
                    if item.name == binding.destination.port
                )
                facts.add_output(
                    key,
                    entry.template,
                    source_fact.artifact,
                    output_type,
                    frozenset({source_key}),
                    (
                        "decision"
                        if source_fact.decision
                        else "candidate"
                        if any(
                            isinstance(output.source, NodeOutputRef)
                            and output.source.node == entry.template
                            and output.source.port == binding.destination.port
                            for output in admitted.context.prepared.workflow.workflow.output_bindings
                        )
                        else "artifact"
                    ),
                )
        expansions = {item.parent: item for item in state.expansions}
        pending = next(
            (
                (entry, expansion)
                for entry in sorted(state.entries, key=lambda item: item.activation.occurrence)
                if entry.status == "success"
                and isinstance(admitted.nodes[entry.template], SubgraphNode)
                and (entry.activation not in expansions or expansions[entry.activation].status == "pending")
                for expansion in admitted.map_expansions
                if expansion.expander == entry.template and expansion.outcome == entry.outcome
            ),
            None,
        )
        if pending is None:
            return state
        entry, expansion = pending
        reference = facts.artifact(target, entry.activation, expansion.membership_port)
        if reference is None or reference not in facts.values:
            reject(EffectCode.MISSING)
        state = _observe_dynamic_membership(
            admitted,
            state,
            target,
            entry.activation,
            entry.template,
            entry.outcome,
            facts.values[reference],
            facts,
            limits,
        )


def _materialize_bound_context(
    admitted: AdmittedExecutionPlan,
    invocation: InvocationId,
    facts: _ExecutionFacts,
    limits: ExecutionLimits,
) -> None:
    context = admitted.context.bound_context
    if context is None:
        return
    declarations = {item.identity: item.declaration for item in context.receipt.sources}
    groups: dict[BindingDeclarationId, list[BoundTextArtifact]] = {}
    for artifact in context.artifacts:
        groups.setdefault(artifact.reference.declaration, []).append(artifact)
    for identity, raw_items in groups.items():
        declaration = declarations[identity]
        items = sorted(raw_items, key=lambda item: (item.reference.key, item.reference.version))
        if (
            limits.max_collection_items is not None
            and declaration.materialization.kind == "collection"
            and len(items) > limits.max_collection_items
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        item_refs: list[ArtifactRef] = []
        parents: set[ProvenanceKey] = set()
        version_keys: dict[int, int] = {}
        for item in items:
            if item.reference.key not in version_keys:
                version_keys[item.reference.key] = facts.next_artifact
                facts.next_artifact += 1
            reference = ArtifactRef(
                invocation=invocation, key=version_keys[item.reference.key], version=item.reference.version
            )
            facts.values[reference] = TextArtifactValue(text=item.text)
            key = BoundInputKey(
                target=item.target,
                node=item.node,
                port=item.port,
                binding_artifact=item.reference,
            )
            facts.add_provenance(
                ArtifactProvenanceFact(_key=_FACT_KEY, key=key, artifact=reference, parents=frozenset(), decision=False)
            )
            item_refs.append(reference)
            parents.add(key)
        if declaration.materialization.kind == "single":
            if declaration.version_selection == "exact_one" and len(item_refs) != 1:
                reject(EffectCode.CONTRADICTORY)

        else:
            reference = ArtifactRef(invocation=invocation, key=facts.next_artifact, version=1)
            facts.next_artifact += 1
            facts.values[reference] = TextCollectionValue(
                items=tuple(
                    TextCollectionItem(
                        key=item.reference.key,
                        version=item.reference.version,
                        value=TextArtifactValue(text=item.text),
                    )
                    for item in items
                )
            )
            key = InitialCollectionKey(
                target=declaration.target,
                node=declaration.node,
                port=declaration.port,
                declaration=identity,
            )
            facts.add_provenance(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY, key=key, artifact=reference, parents=frozenset(parents), decision=False
                )
            )

    if len(facts.values) > limits.max_runtime_artifacts or (
        limits.max_runtime_artifact_bytes is not None
        and sum(_artifact_bytes(item) for item in facts.values.values()) > limits.max_runtime_artifact_bytes
    ):
        reject(EffectCode.LIMIT_EXCEEDED)


def _artifact_bytes(value: ArtifactValue) -> int:
    return value.shape.content_bytes
