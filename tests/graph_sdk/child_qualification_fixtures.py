# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical fixed-recipe receipts; no delegated runtime execution is claimed."""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Literal

from anonymizer.engine.graph_sdk._execution_values import _FACT_KEY, _RESULT_KEY, ImplementationResourceRegistration
from anonymizer.engine.graph_sdk.capabilities import ImplementationRef
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.executor import (
    ArtifactProvenanceFact,
    AssessmentEnvironment,
    AssessmentFinding,
    AssessmentLimits,
    DelegatedRequestContract,
    EvidenceProductionDecl,
    ExecutionAssessmentFact,
    ExecutionPortFact,
    ExecutionResult,
    FinalOutputFact,
    OperationExecutionPolicy,
    OperationOutputKey,
    RootInputKey,
    admit_execution_plan,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparationConfiguration, StateRevisionView, prepare
from anonymizer.engine.graph_sdk.records import (
    CandidateRef,
    CanonicalRecord,
    ExpectedMembership,
    TargetStatus,
    TerminalFact,
)
from anonymizer.engine.graph_sdk.requests import (
    AcceptChildEvidence,
    AcceptFailure,
    AcceptResult,
    AssociationResult,
    ChildEvidence,
    ChildRoute,
    Dispatch,
    ExactUsage,
    ExternalSettlement,
    InvocationRequestScope,
    ObserveSettlement,
    PhysicalRequestId,
    PhysicalRequestPolicy,
    RequestChain,
    RequestChainId,
    RequestChild,
    RequestPolicyBinding,
    Reserve,
    SemanticAssociation,
    TextArtifactValue,
    UnknownUsage,
    advance_requests,
    bind_request_chain,
    bind_request_policies,
    initialize_requests,
    request_receipt,
)
from anonymizer.engine.graph_sdk.resources import CleanupFact, ResourceId, _cleanup_association
from anonymizer.graph._values import ActivationKey, ArtifactRef, InvocationId, TaskAttemptId
from anonymizer.graph.activation import (
    ActivationLimits,
    ActivationSeed,
    ObserveTerminal,
    Select,
    Start,
    advance_activation,
    initialize_activation,
)
from anonymizer.graph.workflow import (
    DynamicScope,
    NodeId,
    OperationNode,
    ProtectionRequirement,
    SequenceEdge,
    admit_activation_workflow,
    admit_static_workflow,
)
from tests.graph_sdk.effects_admission_fixtures import _valid_runtime_rows
from tests.graph_sdk.implementation_fixtures import _execution_implementation, _selection
from tests.graph_sdk.test_adaptive_executor import _adaptive_workflow
from tests.graph_sdk.test_preparation import _capability, _data, _limits


def child_fixture(
    *,
    failover: bool = False,
    usage: Literal["exact", "unknown_allowed"] = "unknown_allowed",
    exact: bool = False,
    interleaved: bool = False,
    alternate_outcome: bool = False,
    ordinary_slot: bool = False,
    simple: bool = False,
    stateless: bool = False,
    unready: bool = False,
    generations: int = 0,
):
    count = generations or (1 if simple else 4 if interleaved else 3 if failover else 2)
    base, assessor, artifact_type = _adaptive_workflow(assessment=True, requests=0)
    raw = base.workflow
    assessment_op = next(iter(raw.nodes)).operation
    owner = NodeId.new(workflow=raw.workflow)
    leaves = tuple(NodeId.new(workflow=raw.workflow) for _ in range(count))
    ordinary = (NodeId.new(workflow=raw.workflow),) if ordinary_slot else ()
    empty = replace(
        assessment_op,
        name="request",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=tuple(
            replace(
                outcome,
                produced_ports=frozenset(),
                evidence=frozenset(),
                ceiling=replace(outcome.ceiling, max_model_requests=1),
            )
            for outcome in assessment_op.outcomes
        ),
    )
    if alternate_outcome:
        empty = replace(empty, outcomes=(*empty.outcomes, replace(empty.outcomes[0], name="alternate")))
    owner_op = replace(
        empty,
        outcomes=tuple(
            replace(outcome, ceiling=replace(outcome.ceiling, max_model_requests=count)) for outcome in empty.outcomes
        ),
    )
    nodes = (
        OperationNode(id=assessor, operation=assessment_op),
        OperationNode(id=owner, operation=owner_op),
        *(OperationNode(id=node, operation=empty) for node in (*leaves, *ordinary)),
    )
    interface = replace(
        raw.interface,
        outcomes=tuple(
            replace(
                outcome,
                ceiling=replace(
                    outcome.ceiling,
                    max_activations=len(nodes),
                    max_model_requests=2 * count + len(ordinary),
                    max_input_bytes=100 * len(nodes),
                    max_output_bytes=100 * len(nodes),
                ),
            )
            for outcome in raw.interface.outcomes
        ),
    )
    static = admit_static_workflow(
        workflow=raw.workflow,
        interface=interface,
        nodes=nodes,
        input_bindings=tuple(raw.input_bindings),
        output_bindings=tuple(raw.output_bindings),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(
            *tuple(SequenceEdge(before=node, after=assessor) for node in (owner, *leaves, *ordinary)),
            *((SequenceEdge(before=owner, after=leaves[0]),) if unready else ()),
        ),
        choices=(),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning="test assessment",
                subject_port="context",
                consumed_ports=frozenset({"input"}),
                coverage=frozenset(),
            ),
        ),
        limits=replace(raw.limits, max_nodes=len(nodes), max_sequence_edges=len(nodes)),
    )
    workflow = admit_activation_workflow(
        workflow=static,
        scopes=(DynamicScope(workflow=static, maps=(), loops=(), joins=()),),
        limits=replace(base.limits, max_activation_occurrences=len(nodes)),
    )
    capabilities = tuple(
        replace(
            _capability(workflow, external=index > 0),
            operation=node.operation,
            implementation=ImplementationRef(name=f"child-fixture-{index}", revision=1),
            max_physical_requests_per_activation=count if index == 1 else int(index > 1),
            retry_owner="implementation" if index == 1 else "executor" if index > 1 else "none",
            usage="unknown" if index > 0 and usage != "exact" else "exact",
            resource_lifetime="stateless" if stateless else "executor_owned" if index > 0 else "stateless",
        )
        for index, node in enumerate(nodes)
    )
    data = _data(1)
    target = next(iter(data.targets))
    prepared = prepare(
        data=data,
        workflow=workflow,
        capabilities=capabilities,
        activation_limits=ActivationLimits(max_events=3 * len(nodes), max_entries=len(nodes), max_parent_depth=1),
        selections=tuple(_selection(node.id, cap) for node, cap in zip(nodes, capabilities, strict=True)),
        limits=_limits(capabilities=len(capabilities)),
        configuration=PreparationConfiguration(
            purpose="protection", required_protection_outcomes=frozenset({"ok"}), hard_request_limit=None
        ),
        bound_inputs=(BoundInput(target=target, source=target, port="input", artifact_type=artifact_type),),
        state=StateRevisionView(revisions=frozenset()),
    )
    request = PhysicalRequestPolicy(
        visibility="dispatch_and_settlement",
        pre_dispatch_control="executor",
        retry_owner="implementation",
        replay="idempotent",
        max_attempts=count,
    )
    contract = DelegatedRequestContract(
        mode="qualified",
        child_slots=tuple(slot.index for slot in prepared.reservation_recipe if slot.template in leaves),
        generation_groups=tuple((slot.index,) for slot in prepared.reservation_recipe if slot.template in leaves)
        if generations
        else (),
        routes=(ChildRoute(name="primary", parse_attempts=2), ChildRoute(name="fallback", parse_attempts=1))
        if failover
        else (ChildRoute(name="primary", parse_attempts=1 if simple or generations else 2),),
        usage_requirement=usage,
    )
    policies = []
    for index, (node, cap) in enumerate(zip(nodes, capabilities, strict=True)):
        physical = (
            request if index == 1 else replace(request, retry_owner="executor", max_attempts=1) if index > 1 else None
        )
        policies.append(
            OperationExecutionPolicy(
                node=node.id,
                kind="delegated" if index == 1 else "external" if index > 1 else "local",
                request=physical,
                safe_detachment="forbidden",
                implementations=(_execution_implementation(cap, request=physical),),
                result_outcomes=frozenset(outcome.name for outcome in node.operation.outcomes),
                runtime_outcomes=_valid_runtime_rows(
                    "external" if index else "local", frozenset(outcome.name for outcome in node.operation.outcomes)
                ),
                delegated=contract if index == 1 else None,
            )
        )
    finding = AssessmentFinding(status="satisfied", code="observed")
    execution = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=tuple(policies),
        decisions=(),
        assessment_productions=(
            EvidenceProductionDecl(
                node=assessor,
                outcome="ok",
                promise="checked",
                evidence_port="context",
                absence_queries=frozenset(),
                supported_findings=frozenset({finding}),
            ),
        ),
        assessment_limits=AssessmentLimits(
            max_productions=1,
            max_findings_per_production=1,
            max_finding_code_bytes=20,
            max_absence_queries=0,
            max_assessment_facts=1,
            max_port_facts=8,
            max_provenance_edges=8,
        ),
    )
    if generations:
        return execution, None
    invocation = InvocationId.new(plan=prepared.plan)
    keys = {
        slot.template: ActivationKey(invocation=invocation, occurrence=slot.index, parent=None, iteration=None)
        for slot in prepared.reservation_recipe
    }
    seeds = frozenset(ActivationSeed(template=node, activation=key) for node, key in keys.items())
    state = initialize_activation(
        workflow=workflow, invocation=invocation, reservations=seeds, limits=prepared.activation_limits
    )
    state = advance_activation(state=state, event=Select(seeds=seeds))
    attempts = {node: TaskAttemptId.new(activation=key) for node, key in keys.items()}
    terminals = []
    for node in (owner, *leaves, *ordinary, assessor):
        key = keys[node]
        category = "failure" if node in (leaves[:2] if interleaved else leaves[:-1]) else "success"
        state = advance_activation(state=state, event=Start(activation=key))
        state = advance_activation(
            state=state,
            event=ObserveTerminal(activation=key, outcome=None if category == "failure" else "ok", category=category),
        )
        terminals.append(
            TerminalFact(
                activation=key,
                attempt=attempts[node],
                category=category,
                reasons=frozenset({"execution_failed"}) if category == "failure" else frozenset(),
            )
        )
    scope = InvocationRequestScope(invocation=invocation)
    association = SemanticAssociation(task=attempts[owner])
    chain = RequestChain(
        chain=RequestChainId.new(scope=scope),
        associations=frozenset({association}),
        policy=request,
        role="generation",
        routes=contract.routes,
        followups=contract.followups,
    )
    requests = initialize_requests(scope=scope, hard_limit=None, policies=frozenset({request}))
    requests = bind_request_policies(
        state=requests, binding=RequestPolicyBinding.create(association=association, policies=frozenset({request}))
    )
    requests = bind_request_chain(state=requests, chain=chain)
    chains: tuple[RequestChain, ...] = (
        (chain, replace(chain, chain=RequestChainId.new(scope=scope))) if interleaved else (chain,)
    )
    for additional in chains[1:]:
        requests = bind_request_chain(state=requests, chain=additional)
    predecessors: dict[int, PhysicalRequestId] = {}
    proofs: dict[int, ChildEvidence] = {}
    for index, node in enumerate(leaves):
        chain_index = index % 2 if interleaved else 0
        chain = chains[chain_index]
        prior = predecessors.get(chain_index)
        proof = proofs.get(chain_index)
        physical = PhysicalRequestId.new(scope=scope)
        source = "initial" if prior is None else "failover" if failover and index == 2 else "correction"
        child = RequestChild(
            chain=chain.chain,
            role="generation",
            route="fallback" if source == "failover" else "primary",
            source_kind=source,
            predecessor=prior,
            evidence=proof,
            leaf=attempts[node],
        )
        requests = advance_requests(
            state=requests,
            event=Reserve(
                request=physical, purpose=source, associations=chain.associations, policy=request, child=child
            ),
        )
        requests = advance_requests(state=requests, event=Dispatch(request=physical))
        requests = advance_requests(
            state=requests,
            event=ObserveSettlement(
                settlement=ExternalSettlement(
                    request=physical,
                    disposition="completed",
                    remote_stopped=True,
                    usage=ExactUsage(input_units=1, output_units=1) if exact else UnknownUsage(),
                )
            ),
        )
        if index >= (2 if interleaved else count - 1):
            requests = advance_requests(
                state=requests,
                event=AcceptResult(
                    request=physical,
                    results=(
                        AssociationResult(
                            association=association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                        ),
                    ),
                ),
            )
        else:
            requests = advance_requests(
                state=requests, event=AcceptFailure(request=physical, failure="malformed_response")
            )
            proof = ChildEvidence(chain=chain.chain, request=physical, kind="parser_rejected")
            requests = advance_requests(state=requests, event=AcceptChildEvidence(evidence=proof))
            if failover and index == 1:
                proof = ChildEvidence(chain=chain.chain, request=physical, kind="generation_exhausted")
                requests = advance_requests(state=requests, event=AcceptChildEvidence(evidence=proof))
        predecessors[chain_index] = physical
        if proof is not None:
            proofs[chain_index] = proof
    root = ArtifactRef(invocation=invocation, key=0, version=1)
    output = ArtifactRef(invocation=invocation, key=1, version=1)
    root_key = RootInputKey(target=target, port="input")
    output_key = OperationOutputKey(activation=keys[assessor], target=target, port="context")
    record = CanonicalRecord(
        plan=prepared.plan,
        invocation=invocation,
        graph=data.graph,
        targets=data.targets,
        memberships=(
            ExpectedMembership(invocation=invocation, parent=None, members=frozenset(keys.values()), closed=True),
        ),
        terminals=tuple(terminals),
        artifacts=frozenset({root, output}),
        evidence=(),
        statuses=(
            TargetStatus(
                target=target,
                completion="closed",
                qualification="unknown",
                artifact_available=True,
                protection_available=False,
            ),
        ),
    )
    # The fixture author supplies a canonical retained record. Sealed fact constructors
    # do not execute a delegated operation or bypass the production service gate.
    # Declared SDK lease identities and closed facts are literal oracle inputs too.
    registrations = tuple(
        ImplementationResourceRegistration(
            invocation=invocation,
            implementation=item.implementation,
            operation=item.capability.operation,
            configuration=item.configuration,
            resource=ResourceId.new(),
            owner="sdk",
            safe_detachment=policy.safe_detachment,
        )
        for policy in execution.policies
        for item in policy.implementations
        if item.capability.resource_lifetime != "stateless"
    )
    result = ExecutionResult(
        _key=_RESULT_KEY,
        _execution=execution,
        implementation_resources=registrations,
        adaptive_acquisitions=(),
        record=record,
        states=(state,),
        requests=request_receipt(state=requests),
        cleanup=tuple(CleanupFact(resource=row.resource, owner="sdk", disposition="closed") for row in registrations),
        pending_decisions=(),
        artifacts=((root, TextArtifactValue(text="source")), (output, TextArtifactValue(text="candidate"))),
        assessments=(
            ExecutionAssessmentFact(
                _key=_FACT_KEY,
                activation=keys[assessor],
                node=assessor,
                outcome="ok",
                promise="checked",
                evidence_artifact=output,
                finding=finding,
                environment=AssessmentEnvironment(
                    configuration=capabilities[0].configuration, state=prepared.state, absences=frozenset()
                ),
            ),
        ),
        ports=(
            ExecutionPortFact(
                _key=_FACT_KEY,
                activation=keys[assessor],
                node=assessor,
                target=target,
                port="input",
                artifact=root,
                artifact_type=artifact_type,
                role="artifact",
            ),
            ExecutionPortFact(
                _key=_FACT_KEY,
                activation=keys[assessor],
                node=assessor,
                target=target,
                port="context",
                artifact=output,
                artifact_type=artifact_type,
                role="candidate",
            ),
        ),
        final_outputs=(
            FinalOutputFact(
                _key=_FACT_KEY,
                target=target,
                outcome="ok",
                port="context",
                candidate=CandidateRef(artifact=output, target=target),
                producer=output_key,
            ),
        ),
        provenance=(
            ArtifactProvenanceFact(_key=_FACT_KEY, key=root_key, artifact=root, parents=frozenset(), decision=False),
            ArtifactProvenanceFact(
                _key=_FACT_KEY, key=output_key, artifact=output, parents=frozenset({root_key}), decision=False
            ),
        ),
        _input_parents=((target, keys[assessor], "input", root_key),),
        _passthrough_parents=(),
        cleanup_associations=tuple(
            _cleanup_association(resource=row.resource, targets=data.targets, purpose="accounting")
            for row in registrations
        ),
        publication_failures=(),
    )
    return execution, result


def result_with(result: ExecutionResult, **changes: object) -> ExecutionResult:
    return ExecutionResult(
        _key=_RESULT_KEY, **{**{field.name: getattr(result, field.name) for field in fields(result)}, **changes}
    )
