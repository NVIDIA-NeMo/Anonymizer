# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output publication, provenance and assessment capture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Literal

from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE, EntitySpansValue
from anonymizer.engine.graph_sdk._effect_values import (
    EffectCode,
    reject,
)
from anonymizer.engine.graph_sdk._execution_materialization import _artifact_bytes
from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts, _OperationJob
from anonymizer.engine.graph_sdk._execution_topology import (
    _source_activation,
)
from anonymizer.engine.graph_sdk._execution_values import (
    _FACT_KEY,
    AdmittedExecutionPlan,
    AssessmentEnvironment,
    EvidenceProductionDecl,
    ExecutionAssessmentFact,
    ExecutionLimits,
    ExecutionServices,
    FinalOutputFact,
    LocalAssessmentResult,
    LocalCompleted,
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
    UnusedRetryCapacity,
)
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    SemanticAssociation,
    TextArtifactValue,
    TextCollectionValue,
)
from anonymizer.engine.rewrite.artifacts import REWRITE_TYPES
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
    NodeId,
    OperationSpec,
    OutputDependency,
    SubgraphNode,
    WorkflowInputRef,
)

if TYPE_CHECKING:
    from anonymizer.engine.detection._frame_publication import InvocationFrame


def _accept_outputs(
    admitted: AdmittedExecutionPlan,
    job: _OperationJob[DatumId],
    mapping: RuntimeOutcome,
    results: tuple[AssociationResult, ...],
    facts: _ExecutionFacts,
    limits: ExecutionLimits,
) -> Literal["valid", "malformed", "limit"]:
    operation = job.implementation.capability.operation
    inputs, input_parents = job.inputs, job.input_parents
    if not _validate_output_shape(admitted, operation, mapping, job.association, inputs, results):
        return "malformed"
    assert mapping.outcome is not None
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    expected = {item.name: item.artifact_type for item in operation.outputs if item.name in outcome.produced_ports}
    outputs = results[0].outputs
    dependencies = {item.output: item for item in operation.output_dependencies if item.output in expected}
    new_outputs = [item for item in outputs if dependencies[item.port].identity_input is None]
    if len(facts.values) + len(new_outputs) > limits.max_runtime_artifacts:
        return "limit"
    if limits.max_collection_items is not None and any(
        item.value.shape.items > limits.max_collection_items for item in outputs
    ):
        return "limit"
    if limits.max_runtime_artifact_bytes is not None and (
        sum(_artifact_bytes(item) for item in facts.values.values())
        + sum(_artifact_bytes(item.value) for item in new_outputs)
        > limits.max_runtime_artifact_bytes
    ):
        return "limit"
    input_artifacts = {item.port: item.artifact for item in inputs[0].inputs if item.artifact is not None}
    planned_parents: dict[str, frozenset[ProvenanceKey]] = {}
    for output in outputs:
        dependency = dependencies[output.port]
        if not dependency.inputs <= input_artifacts.keys():
            return "malformed"
        parent_keys = {input_parents[name] for name in dependency.inputs if name in input_parents}
        if len(parent_keys) != len(dependency.inputs):
            return "malformed"
        planned_parents[output.port] = frozenset(parent_keys)
        if dependency.identity_input is not None:
            aliased = input_artifacts.get(dependency.identity_input)
            if aliased is None or facts.values[aliased] != output.value:
                return "malformed"
    limits_assessment = admitted.assessment_limits
    if len(facts.ports) + len(outputs) > limits_assessment.max_port_facts:
        return "limit"
    current_edges = sum(len(item.parents) for item in facts.provenance.values())
    if current_edges + sum(len(item) for item in planned_parents.values()) > limits_assessment.max_provenance_edges:
        return "limit"
    for output in outputs:
        identity_input = dependencies[output.port].identity_input
        if identity_input is None:
            reference = ArtifactRef(invocation=job.activation.invocation, key=facts.next_artifact, version=1)
            facts.next_artifact += 1
            facts.values[reference] = output.value
        else:
            reference = input_artifacts[identity_input]
        key = OperationOutputKey(activation=job.activation, target=job.scope, port=output.port)
        facts.add_output(
            key,
            job.node,
            reference,
            output.artifact_type,
            planned_parents[output.port],
            admitted._output_role(job.node, outcome, output.port),
        )
    return "valid"


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
    schemas = admitted.materialization_schemas
    dependencies = {item.output: item for item in operation.output_dependencies}
    for output in outputs:
        if type(output.value) is EntitySpansValue:
            dependency = dependencies.get(output.port)
            if dependency is not None and any(
                item.port == dependency.identity_input and item.value == output.value for item in inputs[0].inputs
            ):
                continue
            if dependency is None or not any(
                item.port in dependency.inputs
                and item.artifact is not None
                and type(item.value) is TextArtifactValue
                and output.value.matches_source(item.artifact, item.value.text)
                for item in inputs[0].inputs
            ):
                return False
    return (
        len(outputs) == len(expected)
        and {item.port for item in outputs} == set(expected)
        and all(
            item.artifact is None
            and item.artifact_type == expected[item.port]
            and (
                REWRITE_TYPES.get(type(item.value)) == item.artifact_type
                if item.artifact_type in REWRITE_TYPES.values()
                else type(item.value) is EntitySpansValue
                if item.artifact_type == ENTITY_SPANS_TYPE
                else isinstance(item.value, TextCollectionValue)
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


def _assessment_results(
    admitted: AdmittedExecutionPlan,
    node: NodeId,
    mapping: RuntimeOutcome,
    association: SemanticAssociation,
    returned: tuple[LocalAssessmentResult, ...],
) -> tuple[tuple[EvidenceProductionDecl, LocalAssessmentResult], ...] | None:
    declarations = tuple(
        item for item in admitted.assessment_productions if item.node == node and item.outcome == mapping.outcome
    )
    if len(returned) != len(declarations):
        return None
    selected = []
    for declaration in declarations:
        matches = [
            item
            for item in returned
            if item.association == association
            and item.promise == declaration.promise
            and item.evidence_port == declaration.evidence_port
            and item.finding in declaration.supported_findings
        ]
        if len(matches) != 1:
            return None
        selected.append((declaration, matches[0]))
    return tuple(selected)


def _capture_assessments(
    admitted: AdmittedExecutionPlan,
    job: _OperationJob[DatumId],
    mapping: RuntimeOutcome,
    returned: tuple[tuple[EvidenceProductionDecl, LocalAssessmentResult], ...],
    services: ExecutionServices,
    facts: _ExecutionFacts,
) -> bool:
    implementation, activation, node, target = job.implementation, job.activation, job.node, job.scope
    if mapping.outcome is None:
        return not returned
    if len(facts.assessments) + len(returned) > admitted.assessment_limits.max_assessment_facts:
        return False
    absence_map = dict(services.absence_revisions)
    operation = implementation.capability.operation
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    promises = {item.name: item for item in outcome.evidence}
    reads = {item for item in outcome.state_effects if item.kind == "read"}
    state_view = StateRevisionView(
        revisions=frozenset(item for item in admitted.context.prepared.state.revisions if item.effect in reads)
    )
    for declaration, result in returned:
        artifact = facts.artifact(target, activation, declaration.evidence_port)
        if artifact is None:
            return False
        facts.assessments[(activation, declaration.promise)] = ExecutionAssessmentFact(
            _key=_FACT_KEY,
            activation=activation,
            node=node,
            outcome=mapping.outcome,
            promise=declaration.promise,
            evidence_artifact=artifact,
            finding=result.finding,
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
        port = promises[declaration.promise].subject_port
        if not isinstance(port, str):
            reject(EffectCode.CONTRADICTORY)
        fact = facts.ports.get((activation, port))
        if fact is None or fact.node != node or fact.target != target or fact.role == "decision":
            reject(EffectCode.CONTRADICTORY)
        item_subject = isinstance(job.input_parents.get(port), MapItemKey)
        facts.ports[activation, port] = replace(fact, _key=_FACT_KEY, role="artifact" if item_subject else "candidate")
    return True


def _final_outputs(
    prepared: PreparedPlan,
    states: tuple[ActivationState, ...],
    facts: _ExecutionFacts,
) -> tuple[FinalOutputFact, ...]:
    results: list[FinalOutputFact] = []
    for target_map, state in zip(prepared.target_occurrences, states, strict=True):
        for output_binding in prepared.workflow.output_bindings:
            root_anchor = next(
                (entry.activation for entry in state.entries if entry.activation.parent is None),
                None,
            )
            if root_anchor is None:
                continue
            producer: ProvenanceKey
            if isinstance(output_binding.source, WorkflowInputRef):
                producer = RootInputKey(target=target_map.target, port=output_binding.source.port)
                source_fact = next((fact for fact in facts.provenance.values() if fact.key == producer), None)
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
                artifact = facts.artifact(target_map.target, terminal.activation, output_binding.source.port)
                if artifact is None:
                    continue
                producer = OperationOutputKey(
                    activation=terminal.activation,
                    target=target_map.target,
                    port=output_binding.source.port,
                )
                if not any(fact.key == producer and fact.artifact == artifact for fact in facts.provenance.values()):
                    reject(EffectCode.MISSING)
            workflow_outcome = None
            for binding in prepared.workflow.outcome_bindings:
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
    admitted: AdmittedExecutionPlan,
    invocation: InvocationId,
    states: tuple[ActivationState, ...],
    target_keys: dict[DatumId, dict[int, ActivationKey]],
    attempts: dict[ActivationKey, TaskAttemptId],
    artifacts: frozenset[ArtifactRef],
    cancelled_unstarted: frozenset[ActivationKey],
    blocked_reasons: Mapping[ActivationKey, str | UnusedRetryCapacity],
    *,
    invocation_state: ActivationState | None = None,
) -> CanonicalRecord:
    del target_keys
    prepared = admitted.context.prepared
    target_states = states
    states = states + (() if invocation_state is None else (invocation_state,))
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
            structural = isinstance(admitted.nodes[entry.template], SubgraphNode)
            cause = blocked_reasons.get(entry.activation)
            unused = (
                cause
                if isinstance(cause, UnusedRetryCapacity) and entry.activation not in cancelled_unstarted
                else None
            )
            reason = (
                "unused_retry_capacity" if unused is not None else cause if isinstance(cause, str) else "prerequisite"
            )
            reasons = {
                "failure": frozenset({"execution_failed"}),
                "cancelled": frozenset({"cancel_requested"}),
                "lost": frozenset({"transport_lost"}),
                "blocked": frozenset({"cancel_requested" if entry.activation in cancelled_unstarted else reason}),
                "inconsistent": frozenset({"contradictory"}),
            }.get(entry.status, frozenset())
            terminals.append(
                TerminalFact(
                    activation=entry.activation,
                    attempt=attempts.get(entry.activation),
                    category=entry.status,
                    reasons=reasons,
                    structural=structural,
                    unused_retry=unused,
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
        for target, state in zip((item.target for item in prepared.target_occurrences), target_states, strict=True)
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


def _accept_invocation_frame(result: object) -> InvocationFrame | None:
    """Validate a source-free compatibility frame before canonical publication."""
    from anonymizer.engine.detection._frame_publication import InvocationFrame

    if (
        type(result) is not LocalCompleted
        or type(getattr(result, "results", None)) is not tuple
        or getattr(result, "results", None) != ()
        or type(getattr(result, "assessments", None)) is not tuple
        or getattr(result, "assessments", None) != ()
        or type(getattr(result, "failed_sources", None)) is not frozenset
        or getattr(result, "failed_sources", None) != frozenset()
        or type(getattr(result, "frame", None)) is not InvocationFrame
    ):
        return None
    assert isinstance(result, LocalCompleted)
    frame = result.frame
    assert isinstance(frame, InvocationFrame)
    try:
        return frame.accept()
    except (AttributeError, TypeError, ValueError):
        return None
