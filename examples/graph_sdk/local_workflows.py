# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-free public Graph SDK examples; run this file from an installed wheel environment."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, replace
from time import monotonic_ns

from anonymizer import graph as g


class Clock:
    def now_ns(self) -> int:
        return monotonic_ns()


@dataclass
class Transform:
    calls: int = 0

    async def run(self, request: tuple[g.AssociationInput, ...]) -> g.LocalResult:
        self.calls += 1
        item = request[0]
        value = item.inputs[0].value
        assert isinstance(value, g.TextArtifactValue)
        return g.LocalCompleted(
            results=(
                g.AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        g.PortArtifact(
                            port="candidate",
                            artifact_type=item.inputs[0].artifact_type,
                            artifact=None,
                            value=g.TextArtifactValue(text=value.text.replace("Alice", "[NAME]")),
                        ),
                    ),
                ),
            )
        )


class Decision(Transform):
    async def run(self, request: tuple[g.AssociationInput, ...]) -> g.LocalDecisionWait:
        self.calls += 1
        item = request[0]
        assert isinstance(item.association, g.SemanticAssociation)
        assert item.inputs[0].artifact is not None
        return g.LocalDecisionWait(association=item.association, artifact=item.inputs[0].artifact)


@dataclass
class Verifier:
    calls: int = 0

    async def run(self, request: tuple[g.AssociationInput, ...]) -> g.LocalCompleted:
        self.calls += 1
        item = request[0]
        source = item.inputs[0]
        assert isinstance(source.value, g.TextArtifactValue)
        assert isinstance(item.association, g.SemanticAssociation)
        finding = g.AssessmentFinding(
            status="satisfied" if "Alice" not in source.value.text else "unsatisfied", code="name-check"
        )
        return g.LocalCompleted(
            results=(
                g.AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        g.PortArtifact(
                            port="candidate",
                            artifact_type=source.artifact_type,
                            artifact=None,
                            value=source.value,
                        ),
                        g.PortArtifact(
                            port="assessment",
                            artifact_type=source.artifact_type,
                            artifact=None,
                            value=g.TextArtifactValue(text=finding.code),
                        ),
                    ),
                ),
            ),
            assessments=(
                g.LocalAssessmentResult(
                    association=item.association, promise="checked", evidence_port="assessment", finding=finding
                ),
            ),
        )


def runtime_outcomes() -> tuple[g.RuntimeOutcome, ...]:
    return (
        g.RuntimeOutcome(condition="result", reported_outcome="ok", failure=None, outcome="ok", category="success"),
        *(
            g.RuntimeOutcome(
                condition="failure", reported_outcome=None, failure=failure, outcome=None, category="failure"
            )
            for failure in (
                "rejected_before_acceptance",
                "retryable",
                "malformed_response",
                "permanent",
                "transport_unknown",
                "implementation_exception",
            )
        ),
        *(
            g.RuntimeOutcome(condition=condition, reported_outcome=None, failure=None, outcome=None, category=category)
            for condition, category in (
                ("cancel_before_start", "blocked"),
                ("cancel_after_start", "cancelled"),
                ("artifact_limit_exhausted", "blocked"),
                ("deadline_exhausted", "blocked"),
            )
        ),
    )


def build(
    *, verified: bool, decision: bool = False, texts: tuple[str, ...] = ("Alice uses alice@example.test",)
) -> tuple[g.AdmittedExecutionPlan, g.ExecutionServices, Transform, Verifier]:
    """Declare a transform and, optionally, a separate verifier with exact candidate evidence."""
    assert not (verified and decision)
    data = g.DataGraph.new()
    targets = []
    for text_value in texts:
        data, target = data.add_text(text_value)
        targets.append(target)
    count = len(targets)
    validated = data.validate(
        targets=tuple(targets),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=g.DataLimits(
            max_datums=count, max_targets=count, max_text_bytes=1024 * count, max_declarations=0, max_group_members=0
        ),
    )
    text = g.ArtifactType(name="text", revision=1)
    owner = g.WorkflowId.new()
    transform_node, verifier_node = g.NodeId.new(workflow=owner), g.NodeId.new(workflow=owner)
    coverage = frozenset({g.CoverageAtom(kind="field", name="name")})
    promise = g.EvidencePromise(
        name="checked", meaning="name-redacted", subject_port="candidate", consumed_ports=frozenset(), coverage=coverage
    )
    outcome = g.OutcomeSpec(
        name="ok",
        category="success",
        produced_ports=frozenset({"candidate"}),
        context=frozenset(),
        evidence=frozenset(),
        state_effects=frozenset(),
        model_requirements=frozenset(),
        ceiling=g.ResourceCeiling(max_activations=1, max_model_requests=0, max_input_bytes=1024, max_output_bytes=1024),
    )
    transform = g.OperationSpec(
        name="custom-redaction",
        inputs=(g.InputPort(name="input", artifact_type=text),),
        outputs=(g.OutputPort(name="candidate", artifact_type=text),),
        output_dependencies=(
            g.OutputDependency(
                output="candidate", inputs=frozenset({"input"}), identity_input="input" if decision else None
            ),
        ),
        outcomes=(outcome,),
    )
    verifier = replace(
        transform,
        name="custom-verifier",
        outputs=(*transform.outputs, g.OutputPort(name="assessment", artifact_type=text)),
        output_dependencies=(
            g.OutputDependency(output="candidate", inputs=frozenset({"input"}), identity_input="input"),
            g.OutputDependency(output="assessment", inputs=frozenset(), identity_input=None),
        ),
        outcomes=(
            replace(
                outcome,
                produced_ports=frozenset({"candidate", "assessment"}),
                evidence=frozenset({replace(promise, subject_port="input")}),
            ),
        ),
    )
    nodes = (g.OperationNode(id=transform_node, operation=transform),)
    bindings = (
        g.InputBinding(
            source=g.WorkflowInputRef(port="input"), destination=g.NodeInputRef(node=transform_node, port="input")
        ),
    )
    sequence = ()
    last = transform_node
    if verified:
        nodes += (g.OperationNode(id=verifier_node, operation=verifier),)
        bindings += (
            g.InputBinding(
                source=g.NodeOutputRef(node=transform_node, port="candidate"),
                destination=g.NodeInputRef(node=verifier_node, port="input"),
            ),
        )
        sequence = (g.SequenceEdge(before=transform_node, after=verifier_node),)
        last = verifier_node
    interface = replace(
        transform,
        name="redaction-workflow",
        outcomes=(
            replace(
                outcome,
                ceiling=replace(
                    outcome.ceiling,
                    max_activations=len(nodes),
                    max_input_bytes=1024 * len(nodes),
                    max_output_bytes=1024 * len(nodes),
                ),
                evidence=frozenset({promise}) if verified else frozenset(),
            ),
        ),
    )
    static = g.admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=nodes,
        input_bindings=bindings,
        output_bindings=(
            g.OutputBinding(
                source=g.NodeOutputRef(node=last, port="candidate"), destination=g.WorkflowOutputRef(port="candidate")
            ),
        ),
        outcome_bindings=(
            g.OutcomeBinding(
                source=g.NodeOutcomeRef(node=last, outcome="ok"), destination=g.WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=sequence,
        choices=(),
        protection=(
            g.ProtectionRequirement(
                outcome="ok",
                meaning="name-redacted",
                subject_port="candidate",
                consumed_ports=frozenset(),
                coverage=coverage,
            ),
        )
        if verified
        else (),
        limits=g.WorkflowLimits(
            max_nodes=2,
            max_bindings=8,
            max_sequence_edges=1,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    workflow = g.admit_activation_workflow(
        workflow=static,
        scopes=(g.DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
        limits=g.DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=1,
            max_activation_occurrences=2,
        ),
    )
    config = g.FrozenConfig(fields=())
    capabilities = tuple(
        g.ImplementationCapability(
            implementation=g.ImplementationRef(name=node.operation.name, revision=1),
            operation=node.operation,
            configuration=config,
            effect="local",
            attribution="per_task",
            request_visibility="none",
            pre_dispatch_control="none",
            retry_owner="none",
            error_reporting="typed_terminal",
            cancellation="before_dispatch_only",
            settlement="synchronous",
            usage="exact",
            resource_lifetime="stateless",
            max_physical_requests_per_activation=0,
        )
        for node in nodes
    )
    prepared = g.prepare(
        data=validated,
        workflow=workflow,
        activation_limits=g.ActivationLimits(max_events=8, max_entries=2, max_parent_depth=1),
        bound_inputs=tuple(
            g.BoundInput(target=target, port="input", source=target, artifact_type=text) for target in targets
        ),
        configuration=g.PreparationConfiguration(
            purpose="protection" if verified else "execution_only",
            required_protection_outcomes=frozenset({"ok"}) if verified else frozenset(),
            hard_request_limit=0,
        ),
        state=g.StateRevisionView.from_revisions(revisions=()),
        selections=tuple(
            g.ImplementationSelection(node=node.id, implementation=capability.implementation, configuration=config)
            for node, capability in zip(nodes, capabilities, strict=True)
        ),
        capabilities=capabilities,
        limits=g.PreparationLimits(
            max_bound_inputs=count,
            max_state_revisions=0,
            max_selections=2,
            max_config_depth=2,
            max_config_atoms=4,
            max_total_activation_slots=2 * count,
            max_capabilities=2,
        ),
    )
    context = g.admit_context_plan(
        prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
    )
    policies = tuple(
        g.OperationExecutionPolicy(
            node=node.id,
            kind="decision" if decision else "local",
            request=None,
            safe_detachment="forbidden",
            implementations=(
                g.ExecutionImplementation(
                    implementation=capability.implementation, configuration=config, capability=capability, request=None
                ),
            ),
            result_outcomes=frozenset() if decision else frozenset({"ok"}),
            runtime_outcomes=tuple(
                row
                for row in runtime_outcomes()
                if row.condition != "result" and row.failure in {None, "permanent", "implementation_exception"}
            )
            if decision
            else runtime_outcomes(),
        )
        for node, capability in zip(nodes, capabilities, strict=True)
    )
    productions = (
        (
            g.EvidenceProductionDecl(
                node=verifier_node,
                outcome="ok",
                promise="checked",
                evidence_port="assessment",
                absence_queries=frozenset(),
                supported_findings=frozenset(
                    {
                        g.AssessmentFinding(status="satisfied", code="name-check"),
                        g.AssessmentFinding(status="unsatisfied", code="name-check"),
                    }
                ),
            ),
        )
        if verified
        else ()
    )
    admitted = g.admit_execution_plan(
        context=context,
        capabilities=capabilities,
        policies=policies,
        decisions=(
            g.DecisionDeclaration(
                node=transform_node,
                artifact_port="input",
                outcomes=(g.DecisionOutcome(decision="approve", outcome="ok"),),
                max_lifetime_ns=30_000_000_000,
            ),
        )
        if decision
        else (),
        assessment_productions=productions,
        assessment_limits=g.AssessmentLimits(
            max_productions=1,
            max_findings_per_production=2,
            max_finding_code_bytes=32,
            max_absence_queries=0,
            max_assessment_facts=count,
            max_port_facts=8 * count,
            max_provenance_edges=8 * count,
        ),
    )
    transform_callback, verifier_callback = Decision() if decision else Transform(), Verifier()
    callbacks = (transform_callback, verifier_callback) if verified else (transform_callback,)
    services = g.ExecutionServices(
        handles=tuple(
            g.ImplementationHandle(
                implementation=capability.implementation,
                operation=capability.operation,
                configuration=config,
                local=callback,
                transport=None,
                resource=None,
            )
            for capability, callback in zip(capabilities, callbacks, strict=True)
        ),
        context_resources=(),
        limits=g.ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=0,
            max_runtime_artifacts=8 * count,
            max_runtime_artifact_bytes=8192 * count,
            max_collection_items=0,
        ),
        decision_limits=g.DecisionLimits(
            max_pending=1 if decision else 0, max_lifetime_ns=30_000_000_000 if decision else 0
        ),
        clock=Clock(),
    )
    return admitted, services, transform_callback, verifier_callback


def qualification_inputs(admitted: g.AdmittedExecutionPlan, result: g.ExecutionResult):
    count = len(admitted.context.prepared.data.targets)
    qualification = g.admit_qualification(
        execution=admitted,
        productions=admitted.assessment_productions,
        limits=g.QualificationLimits(
            max_productions=1,
            max_submissions=count,
            max_port_facts=8 * count,
            max_consumed_per_assessment=8,
            max_coverage_atoms=4,
            max_verified_evidence=count,
            max_required_decisions=0,
            max_fixed_point_steps=8,
            max_revision_entries=16 * count,
            max_absence_revisions=0,
            max_provenance_edges=16 * count,
        ),
    )
    current = g.evidence_revision_view(
        admitted=qualification,
        result=result,
        artifacts=tuple(ref for ref, _ in result.artifacts),
        absences=(),
        configurations=tuple(dict.fromkeys((item.node, item.environment.configuration) for item in result.assessments)),
        state=admitted.context.prepared.state,
    )
    return qualification, current


async def run_example(*, verified: bool) -> dict[str, object]:
    admitted, services, transform, verifier = build(verified=verified)
    running = await g.start_execution(admitted=admitted, capabilities=admitted.capabilities, services=services)
    result = await running.wait()
    qualification, current = qualification_inputs(admitted, result)
    qualified = g.qualify(admitted=qualification, result=result, current=current)
    source = g.bind_projection_source(execution=result, qualification=qualified, continuity=None)
    protected = g.project_protected(source=source)
    selected = tuple(
        g.ArtifactSelection(target=item.target, artifact=item.candidate.artifact) for item in result.final_outputs
    )
    content = g.project_content(source=source, kind="custom" if verified else "execution_only", artifacts=selected)
    calls = (transform.calls, verifier.calls)
    assert g.project_protected(source=source) == protected
    assert g.project_content(source=source, kind=content.kind, artifacts=selected) == content
    assert (transform.calls, verifier.calls) == calls
    assert calls == (1, int(verified))
    assert len(protected.items) == int(verified)
    value = content.items[0].content
    assert isinstance(value, g.TextArtifactValue) and value.text == "[NAME] uses alice@example.test"
    return {
        "verified": verified,
        "callback_counts": calls,
        "text": value.text,
        "qualification": qualified.record.statuses[0].qualification,
        "protected_items": len(protected.items),
        "projection_reexecuted": False,
    }


async def run_controls() -> dict[str, object]:
    admitted, services, transform, _ = build(verified=False)
    try:
        await g.start_execution(admitted=admitted, capabilities=(), services=services)
    except g.PreparationRejected as error:
        negative = error.code.value
    else:
        raise AssertionError("missing capabilities must reject before callbacks")
    assert transform.calls == 0
    try:
        g.execute_sync(admitted=admitted, capabilities=admitted.capabilities, services=services)
    except g.NestedEventLoopError:
        pass
    else:
        raise AssertionError("execute_sync must reject an active event loop")
    assert transform.calls == 0
    decision_plan, decision_services, callback, _ = build(verified=False, decision=True)
    running = await g.start_execution(
        admitted=decision_plan, capabilities=decision_plan.capabilities, services=decision_services
    )
    async with asyncio.timeout(5):
        while not running.pending_decisions():
            await asyncio.sleep(0)
        wait = running.pending_decisions()[0]
        running.submit_decision(
            g.DecisionResponse(wait=wait.wait, workflow=wait.workflow, artifact=wait.artifact, decision="approve")
        )
        result = await running.wait()
    assert callback.calls == 1 and not result.pending_decisions
    assert all(item.category == "success" for item in result.record.terminals)
    return {
        "negative_admission": negative,
        "negative_callbacks": transform.calls,
        "nested_loop_rejected": True,
        "decision_callbacks": callback.calls,
        "same_process_resume": True,
    }


def run_sync_example() -> dict[str, object]:
    admitted, services, callback, _ = build(verified=False)
    result = g.execute_sync(admitted=admitted, capabilities=admitted.capabilities, services=services)
    assert callback.calls == 1 and result.final_outputs
    return {"sync_callbacks": callback.calls, "sync_completed": True}


class MemoryContext(g.RemoteContext):
    """A caller-owned provider; replace storage access with an async client if needed."""

    def __init__(self, source: g.ContextSourceRef) -> None:
        self.source = source
        self.calls = 0

    async def retrieve(
        self,
        *,
        request: g.PhysicalRequestId,
        association: g.RequestAssociation,
        selector: g.ContextSelector,
        bounds: g.RetrievalBounds,
    ) -> g.SourceResponse:
        self.calls += 1
        assert not selector.fields and bounds.max_items == 2
        return g.SourceResponse(
            source=self.source,
            items=tuple(
                g.SourceItem(association=association, key=0, version=version, text=f"subject-{self.calls}-v{version}")
                for version in (2, 1)
            ),
            settlement=g.ExternalSettlement(
                request=request,
                disposition="completed",
                usage=g.ExactUsage(input_units=0, output_units=0),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: g.PhysicalRequestId) -> g.StopConfirmed:
        return g.StopConfirmed(usage=g.ExactUsage(input_units=0, output_units=0))


async def run_context_example() -> dict[str, object]:
    admitted, _, _, _ = build(verified=False)
    prepared = admitted.context.prepared
    raw = prepared.workflow.workflow
    node = next(iter(raw.nodes))
    assert isinstance(node, g.OperationNode)
    use = g.ContextUse(port="input", meaning="subject-context", capture="whole_artifact")
    operation = replace(
        node.operation, outcomes=tuple(replace(item, context=frozenset({use})) for item in node.operation.outcomes)
    )
    static = g.admit_static_workflow(
        workflow=raw.workflow,
        interface=replace(raw.interface, outcomes=operation.outcomes),
        nodes=(g.OperationNode(id=node.id, operation=operation),),
        input_bindings=(
            g.InputBinding(
                source=g.ContextInputRef(port="input"), destination=g.NodeInputRef(node=node.id, port="input")
            ),
        ),
        output_bindings=tuple(raw.output_bindings),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(),
        choices=(),
        protection=(),
        limits=raw.limits,
    )
    workflow = g.admit_activation_workflow(
        workflow=static,
        scopes=(g.DynamicScope(workflow=static, maps=(), joins=(), loops=()),),
        limits=prepared.workflow.limits,
    )
    source = g.ContextSourceRef(name="memory-context", revision=1)
    provider = MemoryContext(source)
    artifact_type = operation.inputs[0].artifact_type
    capability = g.ContextSourceCapability(
        source=source,
        artifact_type=artifact_type,
        uses=frozenset({"initial_binding"}),
        execution="async",
        resource_owner="caller",
        cancellation="cooperative_ack",
        settlement="explicit_ack",
        usage="exact",
        request=g.PhysicalRequestPolicy(
            visibility="dispatch_and_settlement",
            pre_dispatch_control="executor",
            retry_owner="executor",
            replay="idempotent",
            max_attempts=1,
        ),
        safe_detachment="forbidden",
    )
    declaration = g.InitialContextDecl(
        target=next(iter(prepared.data.targets)),
        node=node.id,
        port="input",
        artifact_type=artifact_type,
        source=source,
        selector=g.ContextSelector(fields=()),
        requirement="required",
        bounds=g.RetrievalBounds(max_items=2, max_bytes=128, max_requests=1),
        materialization=g.ContextMaterialization(kind="single", item_type=artifact_type),
        version_selection="latest",
    )
    lease = g.ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=provider)
    saved = []
    for _ in range(2):
        running = await g.start_initial_binding(
            data=prepared.data,
            workflow=workflow,
            declarations=(declaration,),
            capabilities=(capability,),
            resources=(g.ContextResource(source=source, capability=capability, lease=lease, factory=None),),
            limits=g.BindingLimits(
                max_declarations=1,
                max_sources=1,
                max_capabilities=1,
                max_selector_fields=0,
                max_selector_bytes=0,
                max_items=2,
                max_bytes=128,
                max_requests=1,
                max_resources=1,
            ),
        )
        bound = await running.wait()
        assert bound.context is not None and bound.receipt.requests.dispatched_count == 1
        saved.append(tuple(item.text for item in bound.context.artifacts))
    assert provider.calls == 2 and saved[0] != saved[1]
    assert set(saved[0]) == {"subject-1-v1", "subject-1-v2"}
    return {
        "remote_context_calls": provider.calls,
        "retained_versions": saved,
        "ownership": "caller",
        "version_selection": "latest",
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "examples": [asyncio.run(run_example(verified=False)), asyncio.run(run_example(verified=True))],
                "controls": asyncio.run(run_controls()),
                "sync": run_sync_example(),
                "context": asyncio.run(run_context_example()),
            },
            sort_keys=True,
        )
    )
