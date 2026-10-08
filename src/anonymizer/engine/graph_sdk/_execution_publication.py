# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output publication, provenance and assessment capture."""

from __future__ import annotations

from typing import Literal

from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
)
from anonymizer.engine.graph_sdk._execution_materialization import _artifact_bytes
from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts
from anonymizer.engine.graph_sdk._execution_topology import (
    _is_subgraph_node,
    _node_operation,
    _operation_owner,
    _source_activation,
)
from anonymizer.engine.graph_sdk._execution_values import (
    _FACT_KEY,
    AdmittedExecutionPlan,
    ArtifactProvenanceFact,
    AssessmentEnvironment,
    ExecutionAssessmentFact,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionPortFact,
    ExecutionServices,
    FinalOutputFact,
    LocalAssessmentResult,
    MapItemKey,
    OperationOutputKey,
    ProvenanceKey,
    RootInputKey,
    RuntimeOutcome,
)
from anonymizer.engine.graph_sdk.preparation import PreparedPlan, StateRevisionView
from anonymizer.engine.graph_sdk.records import (
    AbsenceRef,
    CandidateRef,
    CanonicalRecord,
    ExpectedMembership,
    TargetStatus,
    TerminalFact,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionValue,
)
from anonymizer.graph._values import (
    ActivationKey,
    ArtifactRef,
    DatumId,
    InvocationId,
    TaskAttemptId,
)
from anonymizer.graph.activation import (
    ActivationState,
)
from anonymizer.graph.workflow import (
    ArtifactType,
    NodeId,
    OperationSpec,
    OutputDependency,
    WorkflowInputRef,
)


def _accept_outputs(
    admitted: AdmittedExecutionPlan,
    invocation: InvocationId,
    target: DatumId,
    activation: ActivationKey,
    node: NodeId,
    operation: OperationSpec,
    mapping: RuntimeOutcome,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    input_parents: dict[str, ProvenanceKey],
    results: tuple[AssociationResult, ...],
    facts: _ExecutionFacts,
    limits: ExecutionLimits,
) -> tuple[Literal["valid", "malformed", "limit"], tuple[ArtifactRef, ...]]:
    if not _validate_output_shape(admitted, operation, mapping, association, inputs, results):
        return "malformed", ()
    assert mapping.outcome is not None
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    expected = {item.name: item.artifact_type for item in operation.outputs if item.name in outcome.produced_ports}
    outputs = results[0].outputs
    if len(outputs) != len(expected) or {item.port for item in outputs} != set(expected):
        return "malformed", ()
    if any(item.artifact is not None or item.artifact_type != expected[item.port] for item in outputs):
        return "malformed", ()
    dependencies = {item.output: item for item in operation.output_dependencies if item.output in expected}
    new_outputs = [item for item in outputs if dependencies[item.port].identity_input is None]
    if len(facts.values) + len(new_outputs) > limits.max_runtime_artifacts:
        return "limit", ()
    if any(
        isinstance(item.value, TextCollectionValue) and len(item.value.items) > limits.max_collection_items
        for item in outputs
    ):
        return "limit", ()
    if (
        sum(_artifact_bytes(item) for item in facts.values.values())
        + sum(_artifact_bytes(item.value) for item in new_outputs)
        > limits.max_runtime_artifact_bytes
    ):
        return "limit", ()
    input_artifacts = {item.port: item.artifact for item in inputs[0].inputs if item.artifact is not None}
    planned_parents: dict[str, frozenset[ProvenanceKey]] = {}
    for output in outputs:
        dependency = dependencies[output.port]
        if not dependency.inputs <= input_artifacts.keys():
            return "malformed", ()
        parent_keys = {input_parents[name] for name in dependency.inputs if name in input_parents}
        if len(parent_keys) != len(dependency.inputs):
            return "malformed", ()
        planned_parents[output.port] = frozenset(parent_keys)
        if dependency.identity_input is not None:
            aliased = input_artifacts.get(dependency.identity_input)
            if aliased is None or facts.values[aliased] != output.value:
                return "malformed", ()
    limits_assessment = admitted.assessment_limits
    if len(facts.ports) + len(outputs) > limits_assessment.max_port_facts:
        return "limit", ()
    current_edges = sum(len(item.parents) for item in facts.provenance)
    if current_edges + sum(len(item) for item in planned_parents.values()) > limits_assessment.max_provenance_edges:
        return "limit", ()
    created: list[ArtifactRef] = []
    decision_output = any(item.node == node for item in admitted.decisions)
    for output in outputs:
        identity_input = dependencies[output.port].identity_input
        if identity_input is None:
            reference = ArtifactRef(invocation=invocation, key=facts.next_artifact, version=1)
            facts.next_artifact += 1
            facts.values[reference] = output.value
            created.append(reference)
        else:
            reference = input_artifacts[identity_input]
        facts.produced[(target, activation, output.port)] = reference
        key = OperationOutputKey(activation=activation, target=target, port=output.port)
        facts.provenance.append(
            ArtifactProvenanceFact(
                _key=_FACT_KEY,
                key=key,
                artifact=reference,
                parents=planned_parents[output.port],
                decision=decision_output,
            )
        )
        facts.ports.append(
            ExecutionPortFact(
                _key=_FACT_KEY,
                activation=activation,
                node=node,
                target=target,
                port=output.port,
                artifact=reference,
                artifact_type=output.artifact_type,
                role=admitted._output_role(node, outcome, output.port),
            )
        )
    return "valid", tuple(created)


def _validate_output_shape(
    admitted: AdmittedExecutionPlan,
    operation: OperationSpec,
    mapping: RuntimeOutcome,
    association: SemanticAssociation,
    inputs: tuple[AssociationInput, ...],
    results: tuple[AssociationResult, ...],
) -> bool:
    if len(results) != 1 or results[0].association != association or mapping.outcome is None:
        return False
    outcome = next((item for item in operation.outcomes if item.name == mapping.outcome), None)
    if outcome is None:
        return False
    expected = {item.name: item.artifact_type for item in operation.outputs if item.name in outcome.produced_ports}
    outputs = results[0].outputs
    present_inputs = {item.port for item in inputs[0].inputs}
    schemas = _materialization_schemas(admitted)
    return (
        len(outputs) == len(expected)
        and {item.port for item in outputs} == set(expected)
        and all(
            item.artifact is None
            and item.artifact_type == expected[item.port]
            and (
                isinstance(item.value, TextCollectionValue)
                if schemas.get(item.artifact_type) == "collection"
                else isinstance(item.value, TextArtifactValue)
            )
            for item in outputs
        )
        and results[0].consumed_context_ports
        == frozenset(use.port for use in outcome.context if use.port in present_inputs)
    )


def _identity_input(dependency: OutputDependency) -> str:
    value = dependency.identity_input
    if value is None:
        reject(EffectCode.CONTRADICTORY)
    return value


def _materialization_schemas(admitted: AdmittedExecutionPlan) -> dict[ArtifactType, Literal["scalar", "collection"]]:
    schemas: dict[ArtifactType, Literal["scalar", "collection"]] = {}
    context = admitted.context.bound_context
    if context is not None:
        for fact in context.receipt.sources:
            schemas[fact.declaration.materialization.item_type] = "scalar"
            schemas[fact.declaration.artifact_type] = (
                "scalar" if fact.declaration.materialization.kind == "single" else "collection"
            )
    for declaration in admitted.context.adaptive_retrievals:
        operation = _operation_owner(
            admitted.context.prepared.workflow.workflow,
            declaration.node,
        )[1].operation
        artifact_type = next(item.artifact_type for item in operation.outputs if item.name == declaration.output_port)
        schemas[declaration.materialization.item_type] = "scalar"
        schemas[artifact_type] = "scalar" if declaration.materialization.kind == "single" else "collection"
    for declaration in admitted.map_expansions:
        operation = _node_operation(admitted.context.prepared.workflow.workflow, declaration.expander)
        artifact_type = next(
            item.artifact_type for item in operation.outputs if item.name == declaration.membership_port
        )
        schemas[declaration.item_type] = "scalar"
        schemas[artifact_type] = "collection"
    return schemas


def _capture_assessments(
    admitted: AdmittedExecutionPlan,
    implementation: ExecutionImplementation,
    association: SemanticAssociation,
    activation: ActivationKey,
    node: NodeId,
    mapping: RuntimeOutcome,
    returned: tuple[LocalAssessmentResult, ...],
    target: DatumId,
    services: ExecutionServices,
    facts: _ExecutionFacts,
) -> bool:
    if mapping.outcome is None:
        return not returned
    declarations = [
        item for item in admitted.assessment_productions if item.node == node and item.outcome == mapping.outcome
    ]
    if len(returned) != len(declarations):
        return False
    if len(facts.assessments) + len(declarations) > admitted.assessment_limits.max_assessment_facts:
        return False
    absence_map = dict(services.absence_revisions)
    operation = implementation.capability.operation
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    reads = {item for item in outcome.state_effects if item.kind == "read"}
    state_view = StateRevisionView(
        revisions=frozenset(item for item in admitted.context.prepared.state.revisions if item.effect in reads)
    )
    for declaration in declarations:
        matches = [
            item
            for item in returned
            if item.association == association
            and item.promise == declaration.promise
            and item.evidence_port == declaration.evidence_port
            and item.finding in declaration.supported_findings
        ]
        artifact = facts.produced.get((target, activation, declaration.evidence_port))
        if len(matches) != 1 or artifact is None:
            return False
        facts.assessments.append(
            ExecutionAssessmentFact(
                _key=_FACT_KEY,
                activation=activation,
                node=node,
                outcome=mapping.outcome,
                promise=declaration.promise,
                evidence_artifact=artifact,
                finding=matches[0].finding,
                environment=AssessmentEnvironment(
                    configuration=implementation.configuration,
                    state=state_view,
                    absences=frozenset(
                        AbsenceRef(
                            invocation=activation.invocation,
                            query=query,
                            scope_revision=absence_map[query],
                        )
                        for query in declaration.absence_queries
                    ),
                ),
            )
        )
    return True


def _mark_assessment_subjects(
    admitted: AdmittedExecutionPlan,
    node: NodeId,
    mapping: RuntimeOutcome,
    target: DatumId,
    activation: ActivationKey,
    facts: _ExecutionFacts,
) -> None:
    if mapping.outcome is None:
        return
    declarations = [
        item for item in admitted.assessment_productions if item.node == node and item.outcome == mapping.outcome
    ]
    if not declarations:
        return
    operation = _node_operation(admitted.context.prepared.workflow.workflow, node)
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    promises = {item.name: item for item in outcome.evidence}
    for port in {promises[item.promise].subject_port for item in declarations}:
        indexes = [
            index
            for index, fact in enumerate(facts.ports)
            if fact.activation == activation and fact.node == node and fact.target == target and fact.port == port
        ]
        if len(indexes) != 1 or facts.ports[indexes[0]].role == "decision":
            reject(EffectCode.CONTRADICTORY)
        fact = facts.ports[indexes[0]]
        item_subject = any(
            (owner, occurrence, name) == (target, activation, port) and isinstance(producer, MapItemKey)
            for owner, occurrence, name, producer in facts.input_parents
        )
        facts.ports[indexes[0]] = ExecutionPortFact(
            _key=_FACT_KEY,
            activation=fact.activation,
            node=fact.node,
            target=fact.target,
            port=fact.port,
            artifact=fact.artifact,
            artifact_type=fact.artifact_type,
            role="artifact" if item_subject else "candidate",
        )


def _validate_assessment_returns(
    admitted: AdmittedExecutionPlan,
    association: SemanticAssociation,
    node: NodeId,
    mapping: RuntimeOutcome,
    returned: tuple[LocalAssessmentResult, ...],
) -> bool:
    if mapping.outcome is None:
        return not returned
    declarations = [
        item for item in admitted.assessment_productions if item.node == node and item.outcome == mapping.outcome
    ]
    if len(returned) != len(declarations):
        return False
    return all(
        len(
            [
                item
                for item in returned
                if item.association == association
                and item.promise == declaration.promise
                and item.evidence_port == declaration.evidence_port
                and item.finding in declaration.supported_findings
            ]
        )
        == 1
        for declaration in declarations
    )


def _final_outputs(
    prepared: PreparedPlan,
    states: tuple[ActivationState, ...],
    produced: dict[tuple[DatumId, ActivationKey | NodeId, str], ArtifactRef],
    provenance: list[ArtifactProvenanceFact],
) -> tuple[FinalOutputFact, ...]:
    results: list[FinalOutputFact] = []
    for target_map, state in zip(prepared.target_occurrences, states, strict=True):
        for output_binding in prepared.workflow.workflow.output_bindings:
            root_anchor = next(
                (entry.activation for entry in state.entries if entry.activation.parent is None),
                None,
            )
            if root_anchor is None:
                continue
            producer: ProvenanceKey
            if isinstance(output_binding.source, WorkflowInputRef):
                producer = RootInputKey(target=target_map.target, port=output_binding.source.port)
                source_fact = next((fact for fact in provenance if fact.key == producer), None)
                if source_fact is None:
                    continue
                artifact = source_fact.artifact
            else:
                source_activation = _source_activation(state, root_anchor, output_binding.source.node)
                terminal = next(
                    (
                        entry
                        for entry in state.entries
                        if entry.activation == source_activation and entry.status == "success"
                    ),
                    None,
                )
                if terminal is None or terminal.outcome is None:
                    continue
                artifact = produced.get((target_map.target, terminal.activation, output_binding.source.port))
                if artifact is None:
                    continue
                producer = OperationOutputKey(
                    activation=terminal.activation,
                    target=target_map.target,
                    port=output_binding.source.port,
                )
                if not any(fact.key == producer and fact.artifact == artifact for fact in provenance):
                    reject(EffectCode.MISSING)
            workflow_outcome = None
            for binding in prepared.workflow.workflow.outcome_bindings:
                outcome_activation = _source_activation(state, root_anchor, binding.source.node)
                outcome_entry = next(
                    (
                        entry
                        for entry in state.entries
                        if entry.activation == outcome_activation
                        and entry.status == "success"
                        and entry.outcome == binding.source.outcome
                    ),
                    None,
                )
                if outcome_entry is not None:
                    workflow_outcome = binding.destination.outcome
                    break
            if workflow_outcome is None:
                continue
            results.append(
                FinalOutputFact(
                    _key=_FACT_KEY,
                    target=target_map.target,
                    outcome=workflow_outcome,
                    port=output_binding.destination.port,
                    candidate=CandidateRef(artifact=artifact, target=target_map.target),
                    producer=producer,
                )
            )
    return tuple(results)


def _canonical_record(
    prepared: PreparedPlan,
    invocation: InvocationId,
    states: tuple[ActivationState, ...],
    target_keys: dict[DatumId, dict[int, ActivationKey]],
    attempts: dict[ActivationKey, TaskAttemptId],
    artifacts: frozenset[ArtifactRef],
    cancelled_unstarted: frozenset[ActivationKey],
) -> CanonicalRecord:
    del target_keys
    all_entries = [entry for state in states for entry in state.entries]
    root_members = frozenset(entry.activation for entry in all_entries if entry.activation.parent is None)
    memberships: list[ExpectedMembership] = [
        ExpectedMembership(
            invocation=invocation,
            parent=None,
            members=root_members,
            closed=all(state.complete for state in states),
        )
    ]
    for state in states:
        child_parents = {entry.activation.parent for entry in state.entries if entry.activation.parent is not None}
        child_parents.update(expansion.parent for expansion in state.expansions)
        for parent in sorted(child_parents, key=lambda item: item.occurrence):
            assert parent is not None
            expansion = next((item for item in state.expansions if item.parent == parent), None)
            parent_entry = next((item for item in state.entries if item.activation == parent), None)
            memberships.append(
                ExpectedMembership(
                    invocation=invocation,
                    parent=parent,
                    members=frozenset(entry.activation for entry in state.entries if entry.activation.parent == parent),
                    closed=(
                        expansion.status != "pending"
                        if expansion is not None
                        else parent_entry is not None
                        and parent_entry.status
                        in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}
                    ),
                )
            )
    terminals: list[TerminalFact] = []
    for state in states:
        for entry in state.entries:
            if entry.status not in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}:
                continue
            structural = _is_subgraph_node(prepared.workflow.workflow, entry.template)
            reasons = {
                "failure": frozenset({"execution_failed"}),
                "cancelled": frozenset({"cancel_requested"}),
                "lost": frozenset({"transport_lost"}),
                "blocked": frozenset(
                    {"cancel_requested" if entry.activation in cancelled_unstarted else "prerequisite"}
                ),
                "inconsistent": frozenset({"contradictory"}),
            }.get(entry.status, frozenset())
            terminals.append(
                TerminalFact(
                    activation=entry.activation,
                    attempt=attempts.get(entry.activation),
                    category=entry.status,
                    reasons=reasons,
                    structural=structural,
                )
            )
    statuses = tuple(
        TargetStatus(
            target=target,
            completion="closed" if state.complete else "pending",
            qualification="not_assessed",
            artifact_available=bool(artifacts),
            protection_available=False,
        )
        for target, state in zip((item.target for item in prepared.target_occurrences), states, strict=True)
    )
    return CanonicalRecord(
        plan=prepared.plan,
        invocation=invocation,
        graph=prepared.data.graph,
        targets=prepared.data.targets,
        memberships=tuple(memberships),
        terminals=tuple(terminals),
        artifacts=artifacts,
        evidence=(),
        statuses=statuses,
    )
