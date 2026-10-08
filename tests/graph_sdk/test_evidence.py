# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence authentication through real local execution and retained facts."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from typing import cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, ImplementationRef, ImplementationSelection
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    QualificationLimits,
    VerifiedEvidence,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import (
    AdmittedExecutionPlan,
    AssessmentFinding,
    AssessmentLimits,
    DecisionDeclaration,
    DecisionLimits,
    DecisionOutcome,
    DecisionResponse,
    EvidenceProductionDecl,
    ExecutionAssessmentFact,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    OperationExecutionPolicy,
    RequestTransport,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    StateRevision,
    StateRevisionView,
    prepare,
)
from anonymizer.engine.graph_sdk.records import AbsenceRef, CandidateRef, DecisionRef
from anonymizer.engine.graph_sdk.requests import PhysicalRequestPolicy
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.graph._values import ArtifactRef, InvocationId
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    CoverageAtom,
    DynamicScope,
    InputBinding,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OutcomeBinding,
    OutputBinding,
    ProtectionRequirement,
    SequenceEdge,
    StateEffect,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.test_adaptive_executor import _adaptive_workflow, _AssessingLocal
from tests.graph_sdk.test_decision_scheduling import _Callback
from tests.graph_sdk.test_effects_production_conformance import _valid_runtime_rows, _ZeroClock
from tests.graph_sdk.test_preparation import _capability, _data, _limits


def _qualification_limits(**changes: int) -> QualificationLimits:
    return QualificationLimits(**{item.name: changes.get(item.name, 16) for item in fields(QualificationLimits)})


async def _execute_assessment(
    *,
    environment: bool = False,
    candidate_input: bool = False,
    alias_output: bool = False,
    decision_input: bool = False,
    coverage: frozenset[CoverageAtom] = frozenset(),
    target_count: int = 1,
    nested: bool = False,
    rename_ports: bool = False,
    data: ValidatedDataGraph | None = None,
    execution_only: bool = False,
    finding: AssessmentFinding | None = None,
    resource: ResourceLease | None = None,
    external: tuple[RequestTransport, ResourceLease] | None = None,
) -> tuple[AdmittedExecutionPlan, ExecutionResult]:
    assert not (decision_input and external is not None)
    predecessor = decision_input or external is not None
    request_policy = (
        PhysicalRequestPolicy(
            visibility="dispatch_and_settlement",
            pre_dispatch_control="executor",
            retry_owner="executor",
            replay="idempotent",
            max_attempts=2,
        )
        if external is not None
        else None
    )
    evidence_port = "assessment" if candidate_input else "context"
    base, node, artifact = _adaptive_workflow(
        assessment=True, requests=0, distinct_subject=candidate_input, alias_evidence=candidate_input or alias_output
    )
    raw = base.workflow
    read = StateEffect(kind="read", name="assessment-policy")
    operation = replace(
        raw.interface,
        outcomes=tuple(
            replace(
                outcome,
                state_effects=frozenset({read}) if environment else frozenset(),
                evidence=frozenset(replace(promise, coverage=coverage) for promise in outcome.evidence),
            )
            for outcome in raw.interface.outcomes
        ),
    )
    decision_node = NodeId.new(workflow=raw.workflow)
    nodes = (OperationNode(id=node, operation=operation),)
    bindings = tuple(raw.input_bindings)
    if predecessor:
        decision_operation = replace(
            operation,
            name="approve-input",
            outcomes=tuple(
                replace(
                    outcome,
                    evidence=frozenset(),
                    state_effects=frozenset(),
                    ceiling=replace(outcome.ceiling, max_model_requests=2 if external is not None else 0),
                )
                for outcome in operation.outcomes
            ),
            output_dependencies=tuple(
                replace(dependency, identity_input="input") for dependency in operation.output_dependencies
            ),
        )
        nodes = (OperationNode(id=decision_node, operation=decision_operation), *nodes)
        bindings = (
            InputBinding(
                source=WorkflowInputRef(port="input"), destination=NodeInputRef(node=decision_node, port="input")
            ),
            InputBinding(
                source=NodeOutputRef(node=decision_node, port=evidence_port),
                destination=NodeInputRef(node=node, port="input"),
            ),
        )
    interface = replace(
        operation,
        outcomes=tuple(
            replace(
                outcome,
                ceiling=replace(
                    outcome.ceiling,
                    max_activations=len(nodes),
                    max_model_requests=2 if external is not None else 0,
                    max_input_bytes=100 * len(nodes),
                    max_output_bytes=100 * len(nodes),
                ),
            )
            for outcome in operation.outcomes
        ),
    )
    root_input = "document" if rename_ports else "input"
    root_output = "protected" if rename_ports else evidence_port
    if rename_ports:
        interface = replace(
            interface,
            inputs=tuple(replace(port, name=root_input) for port in interface.inputs),
            outputs=tuple(replace(port, name=root_output) for port in interface.outputs),
            output_dependencies=tuple(
                replace(
                    item,
                    output=root_output,
                    inputs=frozenset({root_input}),
                    identity_input=root_input if item.identity_input is not None else None,
                )
                for item in interface.output_dependencies
            ),
            outcomes=tuple(
                replace(
                    outcome,
                    produced_ports=frozenset({root_output}),
                    evidence=frozenset(
                        replace(
                            promise,
                            subject_port=root_input if candidate_input else root_output,
                            consumed_ports=frozenset({root_input}),
                        )
                        for promise in outcome.evidence
                    ),
                )
                for outcome in interface.outcomes
            ),
        )
        bindings = tuple(
            replace(binding, source=WorkflowInputRef(port=root_input))
            if isinstance(binding.source, WorkflowInputRef)
            else binding
            for binding in bindings
        )
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=interface,
        nodes=nodes,
        input_bindings=bindings,
        output_bindings=tuple(
            replace(binding, destination=WorkflowOutputRef(port=root_output)) for binding in raw.output_bindings
        ),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(SequenceEdge(before=decision_node, after=node),) if predecessor else (),
        choices=tuple(raw.choices),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning="test assessment",
                subject_port=root_input if candidate_input else root_output,
                consumed_ports=frozenset({root_input}),
                coverage=coverage,
            ),
        ),
        limits=replace(raw.limits, max_nodes=len(nodes), max_bindings=4, max_sequence_edges=1),
    )
    scopes = (DynamicScope(workflow=static, maps=(), loops=(), joins=()),)
    if nested:
        body = static
        owner = WorkflowId.new()
        container = NodeId.new(workflow=owner)
        outer_interface = replace(
            body.interface,
            outcomes=tuple(
                replace(outcome, ceiling=replace(outcome.ceiling, max_activations=outcome.ceiling.max_activations + 1))
                for outcome in body.interface.outcomes
            ),
        )
        static = admit_static_workflow(
            workflow=owner,
            interface=outer_interface,
            nodes=(SubgraphNode(id=container, operation=body.interface, body=body),),
            input_bindings=(
                InputBinding(
                    source=WorkflowInputRef(port=root_input), destination=NodeInputRef(node=container, port=root_input)
                ),
            ),
            output_bindings=(
                OutputBinding(
                    source=NodeOutputRef(node=container, port=root_output),
                    destination=WorkflowOutputRef(port=root_output),
                ),
            ),
            outcome_bindings=tuple(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=container, outcome=item.name),
                    destination=WorkflowOutcomeRef(outcome=item.name),
                )
                for item in body.interface.outcomes
            ),
            sequence=(),
            choices=(),
            protection=tuple(body.protection_requirements),
            limits=replace(body.limits, max_nodes=len(nodes) + 1, max_subgraph_depth=2),
        )
        scopes = (*scopes, DynamicScope(workflow=static, maps=(), loops=(), joins=()))
    count = len(nodes) + int(nested)
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=scopes,
        limits=replace(base.limits, max_activation_occurrences=count, max_dynamic_depth=2 if nested else 1),
    )
    data = data if data is not None else _data(target_count)
    target_count = len(data.targets)
    capabilities = tuple(
        replace(
            _capability(workflow, external=external is not None and item.id == decision_node),
            operation=item.operation,
            max_physical_requests_per_activation=2 if external is not None and item.id == decision_node else 0,
            resource_lifetime="executor_owned"
            if external is not None and item.id == decision_node
            else "stateless"
            if resource is None
            else "caller_owned"
            if resource.owner == "caller"
            else "executor_owned",
            implementation=ImplementationRef(name=f"evidence-fixture-{index}", revision=1),
        )
        for index, item in enumerate(nodes)
    )
    prepared = prepare(
        data=data,
        workflow=workflow,
        capabilities=capabilities,
        activation_limits=ActivationLimits(
            max_events=3 * count, max_entries=count, max_parent_depth=2 if nested else 1
        ),
        selections=tuple(
            ImplementationSelection(
                node=item.id, implementation=capability.implementation, configuration=capability.configuration
            )
            for item, capability in zip(nodes, capabilities, strict=True)
        ),
        limits=_limits(capabilities=len(nodes)),
        configuration=PreparationConfiguration(
            purpose="execution_only" if execution_only else "protection",
            required_protection_outcomes=frozenset() if execution_only else frozenset({"ok"}),
            hard_request_limit=2 * target_count if external is not None else None,
        ),
        bound_inputs=tuple(
            BoundInput(target=target, source=target, port=root_input, artifact_type=artifact) for target in data.targets
        ),
        state=StateRevisionView(
            revisions=frozenset({StateRevision(effect=read, revision=3)}) if environment else frozenset()
        ),
    )
    finding = finding if finding is not None else AssessmentFinding(status="satisfied", code="observed")
    declaration = EvidenceProductionDecl(
        node=node,
        outcome="ok",
        promise="checked",
        evidence_port=evidence_port,
        absence_queries=frozenset({7}) if environment else frozenset(),
        supported_findings=frozenset({finding}),
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=tuple(
            OperationExecutionPolicy(
                node=item.id,
                kind=("external" if external is not None else "decision") if item.id == decision_node else "local",
                request=request_policy if item.id == decision_node else None,
                safe_detachment="forbidden",
                implementations=(
                    ExecutionImplementation(
                        implementation=capability.implementation,
                        configuration=capability.configuration,
                        capability=capability,
                        request=request_policy if item.id == decision_node else None,
                    ),
                ),
                result_outcomes=frozenset() if item.id == decision_node and decision_input else frozenset({"ok"}),
                runtime_outcomes=_valid_runtime_rows(
                    ("external" if external is not None else "decision") if item.id == decision_node else "local",
                    frozenset() if item.id == decision_node and decision_input else frozenset({"ok"}),
                ),
            )
            for item, capability in zip(nodes, capabilities, strict=True)
        ),
        decisions=(
            DecisionDeclaration(
                node=decision_node,
                artifact_port="input",
                outcomes=(DecisionOutcome(decision="approve", outcome="ok"),),
                max_lifetime_ns=10,
            ),
        )
        if decision_input
        else (),
        assessment_productions=(declaration,),
        assessment_limits=AssessmentLimits(
            max_productions=1,
            max_findings_per_production=1,
            max_finding_code_bytes=20,
            max_absence_queries=1 if environment else 0,
            max_assessment_facts=target_count,
            max_port_facts=(3 + 2 * int(predecessor) + int(nested)) * target_count,
            max_provenance_edges=(2 + int(predecessor) + int(nested)) * target_count,
        ),
    )
    services = ExecutionServices(
        handles=tuple(
            ImplementationHandle(
                implementation=capability.implementation,
                operation=capability.operation,
                configuration=capability.configuration,
                local=None
                if external is not None and item.id == decision_node
                else _Callback(decision=True)
                if item.id == decision_node
                else _AssessingLocal(
                    finding=finding, evidence_port=evidence_port, alias_evidence=candidate_input or alias_output
                ),
                transport=external[0] if external is not None and item.id == decision_node else None,
                resource=external[1] if external is not None and item.id == decision_node else resource,
            )
            for item, capability in zip(nodes, capabilities, strict=True)
        ),
        context_resources=(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=1 if external is not None else 0,
            max_runtime_artifacts=2 * target_count,
            max_runtime_artifact_bytes=100,
            max_collection_items=0,
        ),
        decision_limits=DecisionLimits(
            max_pending=target_count if decision_input else 0, max_lifetime_ns=10 if decision_input else 0
        ),
        clock=_ZeroClock(),
        absence_revisions=((7, 4),) if environment else (),
    )
    running = await start_execution(admitted=admitted, capabilities=capabilities, services=services)
    if decision_input:
        async with asyncio.timeout(2):
            submitted = set()
            while len(submitted) < target_count:
                for wait in running.pending_decisions():
                    if wait.wait not in submitted:
                        running.submit_decision(
                            DecisionResponse(
                                wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve"
                            )
                        )
                        submitted.add(wait.wait)
                await asyncio.sleep(0)
    return admitted, await asyncio.wait_for(running.wait(), 2)


@pytest.fixture
def assessment_execution() -> tuple[AdmittedExecutionPlan, ExecutionResult]:
    return asyncio.run(_execute_assessment())


def test_verify_derives_exact_subject_dependencies_and_retained_environment(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    assert verified.subject == result.final_outputs[0].candidate
    assert verified.reference.artifact == fact.evidence_artifact
    input_fact = next(item for item in result.ports if item.port == "input")
    assert verified.consumed_by_port == (("input", input_fact.artifact),)
    assert verified.reference.consumed == frozenset({input_fact.artifact})
    assert verified.environment is fact.environment
    assert verified.finding is fact.finding
    assert verified.coverage == verified.promise.coverage
    assert verified.subject == CandidateRef(artifact=fact.evidence_artifact, target=input_fact.target)
    assert verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),)) == (
        verified,
    )


def test_copied_fields_do_not_authenticate_an_assessment(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    original = result.assessments[0]
    copied = object.__new__(ExecutionAssessmentFact)
    for item in fields(original):
        object.__setattr__(copied, item.name, getattr(original, item.name))
    assert copied == original and copied is not original
    with pytest.raises(EffectRejected) as error:
        verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=copied),))
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_submission_duplicates_and_cross_invocation_facts_reject(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    submission = AssessmentSubmission(fact=result.assessments[0])
    with pytest.raises(EffectRejected) as error:
        verify_evidence(admitted=admitted, result=result, submissions=(submission, submission))
    assert error.value.code is EffectCode.DUPLICATE
    _, foreign = asyncio.run(_execute_assessment())
    with pytest.raises(EffectRejected) as error:
        verify_evidence(
            admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=foreign.assessments[0]),)
        )
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_submission_limit_precedes_member_validation(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_submissions=0),
    )
    with pytest.raises(EffectRejected) as error:
        verify_evidence(
            admitted=admitted, result=result, submissions=cast(tuple[AssessmentSubmission, ...], (object(),))
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_factories_reject_direct_construction_and_boolean_limits() -> None:
    with pytest.raises(TypeError):
        VerifiedEvidence(_key=object())
    with pytest.raises(EffectRejected) as error:
        replace(_qualification_limits(), max_submissions=True)
    assert error.value.code is EffectCode.INVALID_TYPE


def test_explicit_current_view_and_stale_precedes_missing(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    artifacts = tuple(reference for reference, _ in result.artifacts)
    configurations = ((fact.node, fact.environment.configuration),)
    state = StateRevisionView(revisions=frozenset())
    current = evidence_revision_view(
        admitted=admitted, result=result, artifacts=artifacts, absences=(), configurations=configurations, state=state
    )
    assert evidence_validity(evidence=verified, current=current) == "current"
    missing = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(item for item in artifacts if item != fact.evidence_artifact),
        absences=(),
        configurations=configurations,
        state=state,
    )
    assert evidence_validity(evidence=verified, current=missing) == "unknown"
    changed = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=(),
        absences=(),
        configurations=((fact.node, FrozenConfig(fields=())),),
        state=state,
    )
    assert evidence_validity(evidence=verified, current=changed) == "stale"
    assert verified.environment is fact.environment
    assert evidence_validity(evidence=verified, current=current) == "current"


def test_revision_duplicates_and_foreign_result_binding_reject(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    reference = fact.evidence_artifact
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=(reference, reference),
            absences=(),
            configurations=(),
            state=StateRevisionView(revisions=frozenset()),
        )
    assert error.value.code is EffectCode.DUPLICATE
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    other_execution, other_result = asyncio.run(_execute_assessment())
    other_admitted = admit_qualification(
        execution=other_execution, productions=other_execution.assessment_productions, limits=_qualification_limits()
    )
    other_view = evidence_revision_view(
        admitted=other_admitted,
        result=other_result,
        artifacts=(),
        absences=(),
        configurations=(),
        state=StateRevisionView(revisions=frozenset()),
    )
    with pytest.raises(EffectRejected) as error:
        evidence_validity(evidence=verified, current=other_view)
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_revision_aggregate_limit_is_checked_before_members(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_revision_entries=0),
    )
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=cast(tuple, (object(),)),
            absences=(),
            configurations=(),
            state=StateRevisionView(revisions=frozenset()),
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


@pytest.mark.parametrize("factory", ["verify", "view"])
def test_same_prepared_plan_does_not_authenticate_a_different_execution(
    factory: str, assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult]
) -> None:
    execution, result = assessment_execution
    other = admit_execution_plan(
        context=execution.context,
        capabilities=execution.capabilities,
        policies=tuple(execution.policies),
        decisions=tuple(execution.decisions),
        assessment_productions=execution.assessment_productions,
        assessment_limits=execution.assessment_limits,
        map_expansions=execution.map_expansions,
    )
    assert other is not execution and other.context.prepared is execution.context.prepared
    admitted = admit_qualification(
        execution=other, productions=other.assessment_productions, limits=_qualification_limits()
    )
    with pytest.raises(EffectRejected) as error:
        if factory == "verify":
            verify_evidence(
                admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=result.assessments[0]),)
            )
        else:
            evidence_revision_view(
                admitted=admitted,
                result=result,
                artifacts=(),
                absences=(),
                configurations=(),
                state=StateRevisionView(revisions=frozenset()),
            )
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_absence_capture_does_not_consume_a_port_dependency_slot() -> None:
    execution, result = asyncio.run(_execute_assessment(environment=True))
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_consumed_per_assessment=1),
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    assert len(verified.consumed_by_port) == 1
    assert {(item.query, item.scope_revision) for item in verified.environment.absences} == {(7, 4)}
    assert {(item.effect.name, item.revision) for item in verified.environment.state.revisions} == {
        ("assessment-policy", 3)
    }
    with pytest.raises(EffectRejected) as error:
        admit_qualification(
            execution=execution,
            productions=execution.assessment_productions,
            limits=_qualification_limits(max_consumed_per_assessment=0),
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


@pytest.mark.parametrize(
    ("dependency", "expected"),
    [
        ("none", "current"),
        ("absence-missing", "unknown"),
        ("absence-changed", "stale"),
        ("state-missing", "unknown"),
        ("state-changed", "stale"),
        ("input-missing", "unknown"),
        ("candidate-missing", "unknown"),
        ("changed-and-missing", "stale"),
    ],
)
def test_validity_selectively_tracks_real_environment(dependency: str, expected: str) -> None:
    execution, result = asyncio.run(_execute_assessment(environment=True))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    artifacts = tuple(reference for reference, _ in result.artifacts)
    absences = tuple(fact.environment.absences)
    state = fact.environment.state
    if dependency == "absence-missing":
        absences = ()
    if dependency in {"absence-changed", "changed-and-missing"}:
        absences = tuple(replace(item, scope_revision=item.scope_revision + 1) for item in absences)
    if dependency in {"state-missing", "changed-and-missing"}:
        state = StateRevisionView(revisions=frozenset())
    if dependency == "state-changed":
        state = StateRevisionView(
            revisions=frozenset(replace(item, revision=item.revision + 1) for item in state.revisions)
        )
    if dependency == "input-missing":
        consumed = next(item.artifact for item in result.ports if item.port == "input")
        artifacts = tuple(item for item in artifacts if item != consumed)
    if dependency == "candidate-missing":
        artifacts = tuple(item for item in artifacts if item != verified.subject.artifact)
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=artifacts,
        absences=absences,
        configurations=((fact.node, fact.environment.configuration),),
        state=state,
    )
    assert evidence_validity(evidence=verified, current=current) == expected


def test_missing_current_artifact_precedes_unsupported_absence(
    assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult],
) -> None:
    execution, result = assessment_execution
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    missing = ArtifactRef(invocation=result.record.invocation, key=999, version=1)
    unsupported = AbsenceRef(invocation=result.record.invocation, query=999, scope_revision=1)
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=(missing,),
            absences=(unsupported,),
            configurations=(),
            state=StateRevisionView(revisions=frozenset()),
        )
    assert error.value.code is EffectCode.MISSING


@pytest.mark.parametrize(
    ("defect", "code"),
    [("foreign", EffectCode.FOREIGN_OWNER), ("duplicate", EffectCode.DUPLICATE), ("missing", EffectCode.MISSING)],
)
def test_environment_identity_errors_precede_configuration_contradiction(defect: str, code: EffectCode) -> None:
    execution, result = asyncio.run(_execute_assessment(environment=True))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    original = fact.environment
    absence = next(iter(original.absences))
    absences = frozenset()
    if defect == "foreign":
        absences = frozenset({replace(absence, invocation=InvocationId.new(plan=result.record.plan))})
    if defect == "duplicate":
        absences = frozenset({absence, replace(absence, scope_revision=absence.scope_revision + 1)})
    # Corrupt the retained fact solely to exercise validation of simultaneous defects.
    object.__setattr__(fact, "environment", replace(original, configuration=FrozenConfig(fields=()), absences=absences))
    try:
        with pytest.raises(EffectRejected) as error:
            verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
        assert error.value.code is code
    finally:
        object.__setattr__(fact, "environment", original)


@pytest.mark.parametrize("bound", ["max_port_facts", "max_provenance_edges"])
def test_result_bounds_are_exact_and_count_actual_edges(
    bound: str, assessment_execution: tuple[AdmittedExecutionPlan, ExecutionResult]
) -> None:
    execution, result = assessment_execution
    count = len(result.ports) if bound == "max_port_facts" else sum(len(item.parents) for item in result.provenance)
    assert count > 0
    for limit in (count, count - 1):
        admitted = admit_qualification(
            execution=execution,
            productions=execution.assessment_productions,
            limits=_qualification_limits(**{bound: limit}),
        )
        if limit == count:
            assert verify_evidence(
                admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=result.assessments[0]),)
            )
        else:
            with pytest.raises(EffectRejected) as error:
                verify_evidence(
                    admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=result.assessments[0]),)
                )
            assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_consumed_candidate_retains_target_and_alias_identity() -> None:
    execution, result = asyncio.run(_execute_assessment(candidate_input=True))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    assert verified.consumed_by_port == (("input", verified.subject),)
    assert verified.subject == result.final_outputs[0].candidate
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        absences=(),
        configurations=((fact.node, fact.environment.configuration),),
        state=fact.environment.state,
    )
    assert evidence_validity(evidence=verified, current=current) == "current"


def test_coverage_capacity_accepts_exact_and_rejects_one_over() -> None:
    coverage = frozenset({CoverageAtom(kind="field", name="person")})
    execution, result = asyncio.run(_execute_assessment(coverage=coverage))
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=_qualification_limits(max_coverage_atoms=1),
    )
    (verified,) = verify_evidence(
        admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=result.assessments[0]),)
    )
    assert verified.coverage == coverage
    with pytest.raises(EffectRejected) as error:
        admit_qualification(
            execution=execution,
            productions=execution.assessment_productions,
            limits=_qualification_limits(max_coverage_atoms=0),
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_fixed_point_outer_bound_precedes_missing_production() -> None:
    execution, _ = asyncio.run(_execute_assessment(target_count=2))
    with pytest.raises(EffectRejected) as error:
        admit_qualification(execution=execution, productions=(), limits=_qualification_limits(max_fixed_point_steps=1))
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


def test_consumed_decision_is_derived_from_an_actual_resumed_producer() -> None:
    execution, result = asyncio.run(_execute_assessment(decision_input=True))
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    fact = result.assessments[0]
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=(AssessmentSubmission(fact=fact),))
    ((port, decision),) = verified.consumed_by_port
    assert port == "input" and isinstance(decision, DecisionRef)
    producers = [item for item in result.provenance if item.decision]
    assert len(producers) == 1 and producers[0].artifact == decision.artifact
    assert len(result.record.terminals) == 2
    assert all(item.category == "success" for item in result.record.terminals)
    artifacts = tuple(reference for reference, _ in result.artifacts)
    for missing in (False, True):
        selected = tuple(item for item in artifacts if not missing or item != decision.artifact)
        current = evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=selected,
            absences=(),
            configurations=((fact.node, fact.environment.configuration),),
            state=fact.environment.state,
        )
        assert evidence_validity(evidence=verified, current=current) == ("unknown" if missing else "current")
