# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Input and dynamic membership materialization."""

from __future__ import annotations

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    EffectRejected,
    reject,
)
from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts
from anonymizer.engine.graph_sdk._execution_topology import (
    _is_mapped_item_input,
    _is_subgraph_node,
    _loop_input_source,
    _operation_owner,
    _source_activation,
    _subgraph_owner,
)
from anonymizer.engine.graph_sdk._execution_values import (
    _FACT_KEY,
    AdmittedExecutionPlan,
    ArtifactProvenanceFact,
    BoundInputKey,
    ExecutionLimits,
    ExecutionPortFact,
    InitialCollectionKey,
    MapItemKey,
    OperationOutputKey,
    ProvenanceKey,
    RootInputKey,
    RuntimeOutcome,
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
    AdmittedWorkflow,
    ContextInputRef,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    WorkflowInputRef,
)


def _operation_inputs(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    node: NodeId,
    activation: ActivationKey,
    state: ActivationState,
    association: SemanticAssociation,
    root_inputs: dict[tuple[DatumId, str], ArtifactRef],
    subgraph_inputs: dict[tuple[DatumId, ActivationKey, str], ArtifactRef],
    subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef],
    values: dict[ArtifactRef, ArtifactValue],
    provenance: list[ArtifactProvenanceFact],
) -> tuple[tuple[AssociationInput, ...], dict[str, ProvenanceKey]]:
    workflow, operation_node = _operation_owner(admitted.context.prepared.workflow.workflow, node)
    operation = operation_node.operation
    ports: list[PortArtifact] = []
    parents: dict[str, ProvenanceKey] = {}
    for port in operation.inputs:
        source = next(
            (
                binding.source
                for binding in workflow.input_bindings
                if binding.destination == NodeInputRef(node=node, port=port.name)
            ),
            None,
        )
        source = _loop_input_source(admitted, state, node, activation, port.name, source)
        reference: ArtifactRef | None = produced.get((target, activation, port.name))
        if reference is not None:
            item_parent = next(
                (
                    fact.key
                    for fact in provenance
                    if fact.artifact == reference
                    and isinstance(fact.key, MapItemKey)
                    and fact.key.member == activation
                    and fact.key.port == port.name
                ),
                None,
            )
            if item_parent is not None:
                parents[port.name] = item_parent
        mapped_item = _is_mapped_item_input(admitted, state, node, activation, port.name)
        if reference is None and mapped_item:
            reject(EffectCode.MISSING)
        if reference is None:
            reference = produced.get((target, node, port.name))
        if reference is None and isinstance(source, (WorkflowInputRef, ContextInputRef)):
            if activation.parent is not None:
                reference = subgraph_inputs.get((target, activation.parent, source.port))
                parent = subgraph_input_parents.get((target, activation.parent, source.port))
                if parent is not None:
                    parents[port.name] = parent
            if reference is None and isinstance(source, WorkflowInputRef):
                reference = root_inputs.get((target, source.port))
            if reference is not None and isinstance(source, WorkflowInputRef):
                parents.setdefault(port.name, RootInputKey(target=target, port=source.port))
        elif reference is None and isinstance(source, NodeOutputRef):
            source_activation = _source_activation(state, activation, source.node)
            if source_activation is not None:
                reference = produced.get((target, source_activation, source.port))
                if reference is not None:
                    parents[port.name] = OperationOutputKey(
                        activation=source_activation, target=target, port=source.port
                    )
        if reference is None:
            reject(EffectCode.MISSING)
        if port.name not in parents:
            candidates = [
                fact.key
                for fact in provenance
                if fact.artifact == reference
                and isinstance(fact.key, (BoundInputKey, InitialCollectionKey))
                and fact.key.target == target
                and fact.key.node == node
                and fact.key.port == port.name
            ]
            if len(candidates) != 1:
                reject(EffectCode.MISSING)
            parents[port.name] = candidates[0]
        ports.append(
            PortArtifact(
                port=port.name,
                artifact_type=port.artifact_type,
                artifact=reference,
                value=values[reference],
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
        if (
            sum(_artifact_bytes(item) for item in facts.values.values())
            + sum(len(item.value.text.encode()) for item in collection.items)
            > limits.max_runtime_artifact_bytes
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        if sum(len(item.parents) for item in facts.provenance) + len(selected) > (
            admitted.assessment_limits.max_provenance_edges
        ):
            reject(EffectCode.LIMIT_EXCEEDED)
        parent_key = OperationOutputKey(
            activation=activation,
            target=target,
            port=expansion.membership_port,
        )
        if not any(item.key == parent_key for item in facts.provenance):
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
            facts.produced[(target, member, declaration.item_input)] = reference
            key = MapItemKey(
                expander=activation,
                member=member,
                target=target,
                port=declaration.item_input,
                item_key=item.key,
                item_version=item.version,
            )
            facts.provenance.append(
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


def _valid_dynamic_membership_result(
    admitted: AdmittedExecutionPlan,
    node: NodeId,
    mapping: RuntimeOutcome,
    results: tuple[AssociationResult, ...],
) -> bool:
    dynamic_map = next(
        (
            declaration
            for dynamic_scope in admitted.context.prepared.workflow.scopes
            for declaration in dynamic_scope.maps
            if declaration.expander == node
        ),
        None,
    )
    if dynamic_map is None or mapping.outcome not in dynamic_map.expansion_outcomes:
        return True
    declaration = next(
        item for item in admitted.map_expansions if item.expander == node and item.outcome == mapping.outcome
    )
    selected = [output for result in results for output in result.outputs if output.port == declaration.membership_port]
    return len(selected) == 1 and isinstance(selected[0].value, TextCollectionValue)


def _dynamic_membership_value(
    admitted: AdmittedExecutionPlan,
    node: NodeId,
    outcome: str | None,
    results: tuple[AssociationResult, ...],
) -> ArtifactValue | None:
    declaration = next(
        (item for item in admitted.map_expansions if item.expander == node and item.outcome == outcome),
        None,
    )
    if declaration is None:
        return None
    selected = [
        output.value for result in results for output in result.outputs if output.port == declaration.membership_port
    ]
    if len(selected) != 1:
        reject(EffectCode.MISSING)
    return selected[0]


def _materialize_subgraph_inputs(
    admitted: AdmittedExecutionPlan,
    target: DatumId,
    state: ActivationState,
    activation: ActivationKey,
    node: NodeId,
    root_inputs: dict[tuple[DatumId, str], ArtifactRef],
    subgraph_inputs: dict[tuple[DatumId, ActivationKey, str], ArtifactRef],
    subgraph_input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef],
    provenance: list[ArtifactProvenanceFact],
) -> None:
    owner, declaration = _subgraph_owner(admitted.context.prepared.workflow.workflow, node)
    for port in declaration.operation.inputs:
        source = next(
            (
                binding.source
                for binding in owner.input_bindings
                if binding.destination == NodeInputRef(node=node, port=port.name)
            ),
            None,
        )
        reference: ArtifactRef | None = produced.get((target, activation, port.name))
        parent: ProvenanceKey | None = None
        if reference is None and isinstance(source, ContextInputRef):
            reference = produced.get((target, node, port.name))
            if reference is not None:
                parent = next(
                    (fact.key for fact in provenance if fact.artifact == reference),
                    None,
                )
        if reference is not None:
            parent = parent or next(
                (
                    fact.key
                    for fact in provenance
                    if fact.artifact == reference
                    and isinstance(fact.key, MapItemKey)
                    and fact.key.member == activation
                    and fact.key.port == port.name
                ),
                None,
            )
        mapped_item = _is_mapped_item_input(admitted, state, node, activation, port.name)
        if reference is None and mapped_item:
            reject(EffectCode.MISSING)
        if reference is None and isinstance(source, (WorkflowInputRef, ContextInputRef)):
            if activation.parent is not None:
                reference = subgraph_inputs.get((target, activation.parent, source.port))
                parent = subgraph_input_parents.get((target, activation.parent, source.port))
            if reference is None and isinstance(source, WorkflowInputRef):
                reference = root_inputs.get((target, source.port))
                if reference is not None:
                    parent = RootInputKey(target=target, port=source.port)
        elif reference is None and isinstance(source, NodeOutputRef):
            source_activation = _source_activation(state, activation, source.node)
            if source_activation is not None:
                reference = produced.get((target, source_activation, source.port))
                parent = OperationOutputKey(activation=source_activation, target=target, port=source.port)
        if reference is None or parent is None:
            reject(EffectCode.MISSING)
        key = (target, activation, port.name)
        subgraph_inputs[key] = reference
        subgraph_input_parents[key] = parent


def _materialize_subgraph_outputs(
    root: AdmittedWorkflow,
    target: DatumId,
    state: ActivationState,
    facts: _ExecutionFacts,
    input_parents: dict[tuple[DatumId, ActivationKey, str], ProvenanceKey],
) -> None:
    for entry in sorted(
        state.entries, key=lambda item: (-_activation_depth(item.activation), item.activation.occurrence)
    ):
        if entry.status not in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}:
            continue
        try:
            _, declaration = _subgraph_owner(root, entry.template)
        except EffectRejected:
            continue
        if entry.outcome is None:
            continue
        for binding in declaration.body.output_bindings:
            destination = (target, entry.activation, binding.destination.port)
            if destination in facts.produced:
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
            source_fact = next((item for item in facts.provenance if item.key == source_key), None)
            if source_fact is None:
                continue
            facts.produced[destination] = source_fact.artifact
            if isinstance(binding.source, WorkflowInputRef):
                facts.passthrough_parents[destination] = source_key
            key = OperationOutputKey(
                activation=entry.activation,
                target=target,
                port=binding.destination.port,
            )
            facts.provenance.append(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY,
                    key=key,
                    artifact=source_fact.artifact,
                    parents=frozenset({source_key}),
                    decision=source_fact.decision,
                )
            )
            output_type = next(
                item.artifact_type for item in declaration.operation.outputs if item.name == binding.destination.port
            )
            facts.ports.append(
                ExecutionPortFact(
                    _key=_FACT_KEY,
                    activation=entry.activation,
                    node=entry.template,
                    target=target,
                    port=binding.destination.port,
                    artifact=source_fact.artifact,
                    artifact_type=output_type,
                    role=(
                        "decision"
                        if source_fact.decision
                        else "candidate"
                        if any(
                            isinstance(output.source, NodeOutputRef)
                            and output.source.node == entry.template
                            and output.source.port == binding.destination.port
                            for output in root.output_bindings
                        )
                        else "artifact"
                    ),
                )
            )


def _bridge_structural_map_membership(
    admitted: AdmittedExecutionPlan,
    state: ActivationState,
    target: DatumId,
    facts: _ExecutionFacts,
    limits: ExecutionLimits,
) -> tuple[ActivationState, bool]:
    root = admitted.context.prepared.workflow.workflow
    for entry in sorted(state.entries, key=lambda item: item.activation.occurrence):
        if entry.status != "success" or entry.outcome is None or not _is_subgraph_node(root, entry.template):
            continue
        dynamic_map = next(
            (
                declaration
                for scope in state.workflow.scopes
                for declaration in scope.maps
                if declaration.expander == entry.template and entry.outcome in declaration.expansion_outcomes
            ),
            None,
        )
        if dynamic_map is None:
            continue
        expansion_state = next(
            (item for item in state.expansions if item.parent == entry.activation),
            None,
        )
        if expansion_state is not None and expansion_state.status != "pending":
            continue
        expansion = next(
            item
            for item in admitted.map_expansions
            if item.expander == entry.template and item.outcome == entry.outcome
        )
        reference = facts.produced.get((target, entry.activation, expansion.membership_port))
        if reference is None or reference not in facts.values:
            reject(EffectCode.MISSING)
        updated = _observe_dynamic_membership(
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
        return updated, True
    return state, False


def _materialize_bound_context(
    admitted: AdmittedExecutionPlan,
    invocation: InvocationId,
    values: dict[ArtifactRef, ArtifactValue],
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef],
    provenance: list[ArtifactProvenanceFact],
    next_artifact: int,
    limits: ExecutionLimits,
) -> int:
    context = admitted.context.bound_context
    if context is None:
        return next_artifact
    declarations = {item.identity: item.declaration for item in context.receipt.sources}
    groups: dict[BindingDeclarationId, list[BoundTextArtifact]] = {}
    for artifact in context.artifacts:
        groups.setdefault(artifact.reference.declaration, []).append(artifact)
    for identity, raw_items in groups.items():
        declaration = declarations[identity]
        items = sorted(raw_items, key=lambda item: (item.reference.key, item.reference.version))
        if declaration.materialization.kind == "collection" and len(items) > limits.max_collection_items:
            reject(EffectCode.LIMIT_EXCEEDED)
        item_refs: list[ArtifactRef] = []
        parents: set[ProvenanceKey] = set()
        version_keys: dict[int, int] = {}
        for item in items:
            if item.reference.key not in version_keys:
                version_keys[item.reference.key] = next_artifact
                next_artifact += 1
            reference = ArtifactRef(
                invocation=invocation, key=version_keys[item.reference.key], version=item.reference.version
            )
            values[reference] = TextArtifactValue(text=item.text)
            key = BoundInputKey(
                target=item.target,
                node=item.node,
                port=item.port,
                binding_artifact=item.reference,
            )
            provenance.append(
                ArtifactProvenanceFact(_key=_FACT_KEY, key=key, artifact=reference, parents=frozenset(), decision=False)
            )
            item_refs.append(reference)
            parents.add(key)
        destination = (declaration.target, declaration.node, declaration.port)
        if declaration.materialization.kind == "single":
            if declaration.version_selection == "exact_one" and len(item_refs) != 1:
                reject(EffectCode.CONTRADICTORY)
            produced[destination] = max(item_refs, key=lambda item: item.version)
        else:
            reference = ArtifactRef(invocation=invocation, key=next_artifact, version=1)
            next_artifact += 1
            values[reference] = TextCollectionValue(
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
            provenance.append(
                ArtifactProvenanceFact(
                    _key=_FACT_KEY, key=key, artifact=reference, parents=frozenset(parents), decision=False
                )
            )
            produced[destination] = reference
    if len(values) > limits.max_runtime_artifacts or sum(_artifact_bytes(item) for item in values.values()) > (
        limits.max_runtime_artifact_bytes
    ):
        reject(EffectCode.LIMIT_EXCEEDED)
    return next_artifact


def _artifact_bytes(value: ArtifactValue) -> int:
    if isinstance(value, TextArtifactValue):
        return len(value.text.encode())
    return sum(len(item.value.text.encode()) for item in value.items)
