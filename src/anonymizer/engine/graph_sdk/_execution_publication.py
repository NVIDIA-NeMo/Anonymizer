# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output publication, provenance and assessment capture."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING, Literal, TypeGuard, cast

from anonymizer.engine.graph_sdk._activation_topology import _publication_outcome
from anonymizer.engine.graph_sdk._activation_values import ActivationState
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
    FinalOutputFact,
    LocalAssessmentResult,
    LocalCompleted,
    MapItemKey,
    OperationOutputKey,
    ProvenanceKey,
    RootInputKey,
    RuntimeOutcome,
)
from anonymizer.engine.graph_sdk._graph_values import ActivationKey, ArtifactRef, DatumId, InvocationId, TaskAttemptId
from anonymizer.engine.graph_sdk._workflow_values import NodeId, OperationSpec, SubgraphNode, WorkflowInputRef
from anonymizer.engine.graph_sdk.detection_artifacts import ENTITY_SPANS_TYPE, EntitySpansValue
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.records import (
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
from anonymizer.engine.graph_sdk.rewrite_artifacts import REWRITE_TYPES

if TYPE_CHECKING:
    from anonymizer.engine.harness.frames import AggregateFramePublication, FramePublication, InvocationFrame


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
    if facts.artifact_count + len(new_outputs) > limits.max_runtime_artifacts:
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
        # Two inputs bound to one upstream output share a single parent key.
        if not dependency.inputs <= input_parents.keys():
            return "malformed"
        planned_parents[output.port] = frozenset(input_parents[name] for name in dependency.inputs)
        if dependency.identity_input is not None:
            aliased = input_artifacts.get(dependency.identity_input)
            if aliased is None or facts.values[aliased] != output.value:
                return "malformed"
    limits_assessment = admitted.assessment_limits
    if facts.port_count + len(outputs) > limits_assessment.max_port_facts:
        return "limit"
    current_edges = facts.provenance_edge_count
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
    facts: _ExecutionFacts,
) -> bool:
    implementation, activation, node, target = job.implementation, job.activation, job.node, job.scope
    if mapping.outcome is None:
        return not returned
    if len(facts.assessments) + len(returned) > admitted.assessment_limits.max_assessment_facts:
        return False
    operation = implementation.capability.operation
    outcome = next(item for item in operation.outcomes if item.name == mapping.outcome)
    promises = {item.name: item for item in outcome.evidence}
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
            environment=AssessmentEnvironment(configuration=implementation.configuration),
        )
        port = promises[declaration.promise].subject_port
        if not isinstance(port, str):
            reject(EffectCode.CONTRADICTORY)
        fact = facts.ports.get((activation, port))
        if fact is None or fact.node != node or fact.target != target:
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
        # The selected sink decides the workflow outcome; see _publication_outcome
        # for the withheld candidate of a sink that did not succeed.
        workflow_outcome = _publication_outcome(
            prepared.workflow, {entry.template: entry for entry in state.entries if entry.activation.parent is None}
        )
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
                source_fact = facts.provenance.get(producer)
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
                if (fact := facts.provenance.get(producer)) is None or fact.artifact != artifact:
                    reject(EffectCode.MISSING)
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


# The SDK's only dispatch on concrete detection frame types.
def _is_invocation_frame(frame: object) -> TypeGuard[InvocationFrame]:
    from anonymizer.engine.harness.frames import InvocationFrame

    return type(frame) is InvocationFrame


def _is_publication_frame(frame: object) -> TypeGuard[FramePublication]:
    from anonymizer.engine.harness.frames import FramePublication

    return type(frame) is FramePublication


def _accept_invocation_frame(result: object) -> InvocationFrame | None:
    """Validate a source-free compatibility frame before canonical publication."""
    if (
        type(result) is not LocalCompleted
        or type(getattr(result, "results", None)) is not tuple
        or getattr(result, "results", None) != ()
        or type(getattr(result, "assessments", None)) is not tuple
        or getattr(result, "assessments", None) != ()
        or type(getattr(result, "failed_sources", None)) is not frozenset
        or getattr(result, "failed_sources", None) != frozenset()
        or not _is_invocation_frame(getattr(result, "frame", None))
    ):
        return None
    frame = cast("InvocationFrame", result.frame)
    try:
        return frame.accept()
    except (AttributeError, TypeError, ValueError):
        return None


def _aggregate_frame(frame: object, internal: frozenset[ActivationKey]) -> AggregateFramePublication | None:
    """Keep an accepted owner frame apart from the owner's internal observations."""
    from anonymizer.engine.harness.frames import AggregateFramePublication

    accepted = _accept_invocation_frame(LocalCompleted(results=(), frame=cast("InvocationFrame", frame)))
    return None if accepted is None else AggregateFramePublication(accepted, internal)
