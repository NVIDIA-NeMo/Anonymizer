# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual DD detection/Redact development profile, using the single graph executor.

Fixed generation counts are explicit admission declarations, not public input
limits. Source-bound Redact assessments and shared request receipts are qualified.
Variable chunk closure and retry chains remain incomplete.
The public facade does not use this profile.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from time import monotonic_ns
from typing import Any, cast

import pandas as pd
from data_designer.config import ModelConfig
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import COL_AUGMENTED_ENTITIES, COL_RAW_DETECTED, COL_TEXT, COL_VALIDATION_DECISIONS
from anonymizer.engine.detection._graph_invocation import DetectionRedactInvocation
from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.graph_sdk._execution_delegated import WorkerDelegatedSubmitter
from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
from anonymizer.engine.graph_sdk.blocking import BlockingAffinity, BlockingPool
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationCapability,
    ImplementationRef,
    ImplementationSelection,
)
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.evidence import QualificationLimits, admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    DelegatedReadiness,
    DelegatedRequestContract,
    EvidenceProductionDecl,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    LocalCompleted,
    OperationExecutionPolicy,
    RuntimeOutcome,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    PreparationLimits,
    StateRevisionView,
    prepare,
)
from anonymizer.engine.graph_sdk.qualification import QualificationResult, qualify_retained
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, ChildRoute, PhysicalRequestPolicy
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.engine.ndd._graph_requests import NddPhysicalTransport
from anonymizer.engine.replace._graph_redact import (
    REDACT_EVIDENCE_TYPE,
    REDACT_MATCH,
    REDACT_MEANING,
    REDACT_MISMATCH,
    RedactAssessment,
)
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    DynamicLimits,
    DynamicScope,
    EvidencePromise,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ProtectionRequirement,
    ResourceCeiling,
    SequenceEdge,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)


class _Clock:
    def now_ns(self) -> int:
        return monotonic_ns()


@dataclass(frozen=True)
class _ProfileLocal:
    outcome: str

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        if self.outcome == "async_required" and not ensure_async_engine_loop().is_running():
            raise RuntimeError("DD async engine is not running")
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association,
                    outcome=self.outcome,
                    outputs=(),
                    consumed_context_ports=frozenset(),
                )
                for item in request
            )
        )


def _runtime_rows(names: tuple[str, ...], external: bool) -> tuple[RuntimeOutcome, ...]:
    rows = [
        RuntimeOutcome(
            condition="result",
            reported_outcome=name,
            failure=None,
            outcome=name,
            category="success",
        )
        for name in names
    ]
    rows += [
        RuntimeOutcome(condition="failure", reported_outcome=None, failure=failure, outcome=None, category="failure")
        for failure in (
            "rejected_before_acceptance",
            "retryable",
            "malformed_response",
            "permanent",
            "transport_unknown",
            "implementation_exception",
        )
    ]
    conditions = {
        "cancel_before_start": "blocked",
        "cancel_after_start": "cancelled",
        "artifact_limit_exhausted": "blocked",
        "deadline_exhausted": "failure",
    }
    if external:
        conditions.update(
            cancel_after_dispatch="cancelled",
            lost="lost",
            request_inconsistent="inconsistent",
            budget_exhausted="blocked",
            request_limit_exhausted="blocked",
        )
    rows += [
        RuntimeOutcome(
            condition=cast(Any, condition),
            reported_outcome=None,
            failure=None,
            outcome=None,
            category=cast(Any, category),
        )
        for condition, category in conditions.items()
    ]
    return tuple(rows)


async def execute_detection_redact(
    source: pd.DataFrame,
    *,
    detection: EntityDetectionWorkflow,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    detection_options: Mapping[str, Any],
    validation_chunks: int,
    pool: BlockingPool,
    limits: ExecutionLimits,
) -> ExecutionResult:
    """Admit and execute the named real-DD profile without a second workflow runner."""
    if type(validation_chunks) is not int or validation_chunks < 1 or source.empty:
        raise ValueError("development profile requires nonempty sources and declared validator calls")
    generations = (
        (COL_RAW_DETECTED, 0),
        *((COL_VALIDATION_DECISIONS, i) for i in range(validation_chunks)),
        (COL_AUGMENTED_ENTITIES, 0),
    )
    graph = DataGraph.new()
    targets = []
    for text in source[COL_TEXT]:
        graph, target = graph.add_text(text)
        targets.append(target)
    count = len(generations)
    size = len(targets)
    byte_limit = limits.max_runtime_artifact_bytes
    data = graph.validate(
        targets=tuple(targets),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(
            max_datums=size, max_targets=size, max_text_bytes=byte_limit, max_declarations=0, max_group_members=0
        ),
    )
    workflow_id = WorkflowId.new()
    owner, readiness, readiness_request, probe, *leaves = [NodeId.new(workflow=workflow_id) for _ in range(count + 4)]
    complete = NodeId.new(workflow=workflow_id)
    locals_ = {probe, complete}
    text_type = ArtifactType(name="anonymizer.text", revision=1)
    ceiling = ResourceCeiling(
        max_activations=1, max_model_requests=1, max_input_bytes=byte_limit, max_output_bytes=byte_limit
    )

    def operation(name: str, outcomes: tuple[str, ...], requests: int, io: bool = False) -> OperationSpec:
        return OperationSpec(
            name=name,
            inputs=(InputPort(name="source", artifact_type=text_type),) if io else (),
            outputs=(
                OutputPort(name="redacted", artifact_type=text_type),
                OutputPort(name="spans", artifact_type=ENTITY_SPANS_TYPE),
            )
            if io
            else (),
            output_dependencies=tuple(
                OutputDependency(output=port, inputs=frozenset({"source"}), identity_input=None)
                for port in ("redacted", "spans")
            )
            if io
            else (),
            outcomes=tuple(
                OutcomeSpec(
                    name=value,
                    category="success",
                    produced_ports=frozenset({"redacted", "spans"}) if io else frozenset(),
                    context=frozenset(),
                    evidence=frozenset(),
                    state_effects=frozenset(),
                    model_requirements=frozenset(),
                    ceiling=replace(ceiling, max_model_requests=requests),
                )
                for value in outcomes
            ),
        )

    specs = [
        operation("dd-detection-redact", ("ok",), count, True),
        operation("dd-readiness", ("ready",), 1),
        operation("dd-readiness-request", ("ready",), 1),
        operation("dd-async-probe", ("async_required",), 0),
        *(operation(f"dd-generation-{index}", ("ok",), 1) for index in range(count)),
        operation("dd-frame-complete", ("ok",), 0),
    ]
    specs[-1] = replace(
        specs[-1],
        name="redact-assessment",
        inputs=(
            InputPort(name="source", artifact_type=text_type),
            InputPort(name="candidate", artifact_type=text_type),
            InputPort(name="spans", artifact_type=ENTITY_SPANS_TYPE),
        ),
        outputs=(OutputPort(name="assessment", artifact_type=REDACT_EVIDENCE_TYPE),),
        output_dependencies=(
            OutputDependency(
                output="assessment", inputs=frozenset({"source", "candidate", "spans"}), identity_input=None
            ),
        ),
        outcomes=tuple(
            replace(
                outcome,
                produced_ports=frozenset({"assessment"}),
                evidence=frozenset(
                    {
                        EvidencePromise(
                            name="redaction",
                            meaning=REDACT_MEANING,
                            subject_port="candidate",
                            consumed_ports=frozenset({"source", "candidate", "spans"}),
                            coverage=frozenset(),
                        )
                    }
                ),
            )
            for outcome in specs[-1].outcomes
        ),
    )
    ids = [owner, readiness, readiness_request, probe, *leaves, complete]
    nodes = tuple(OperationNode(id=node, operation=spec) for node, spec in zip(ids, specs, strict=True))
    interface = replace(
        specs[0],
        name="dd-detection-redact-profile",
        outputs=(specs[0].outputs[0],),
        output_dependencies=(specs[0].output_dependencies[0],),
        outcomes=(
            replace(
                specs[0].outcomes[0],
                produced_ports=frozenset({"redacted"}),
                evidence=frozenset(
                    {
                        replace(
                            next(iter(specs[-1].outcomes[0].evidence)),
                            subject_port="redacted",
                            consumed_ports=frozenset({"source"}),
                        )
                    }
                ),
                ceiling=replace(
                    ceiling,
                    max_activations=len(nodes),
                    max_model_requests=2 * count + 2,
                    max_input_bytes=byte_limit * len(nodes),
                    max_output_bytes=byte_limit * len(nodes),
                ),
            ),
        ),
    )
    raw = admit_static_workflow(
        workflow=workflow_id,
        interface=interface,
        nodes=nodes,
        input_bindings=(
            InputBinding(source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=owner, port="source")),
            InputBinding(
                source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=complete, port="source")
            ),
            InputBinding(
                source=NodeOutputRef(node=owner, port="redacted"),
                destination=NodeInputRef(node=complete, port="candidate"),
            ),
            InputBinding(
                source=NodeOutputRef(node=owner, port="spans"), destination=NodeInputRef(node=complete, port="spans")
            ),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=owner, port="redacted"), destination=WorkflowOutputRef(port="redacted")
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=complete, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=(
            SequenceEdge(before=readiness, after=probe),
            *(SequenceEdge(before=probe, after=leaf) for leaf in leaves),
            *(SequenceEdge(before=node, after=complete) for node in ids if node != complete),
        ),
        choices=(),
        protection=(
            ProtectionRequirement(
                outcome="ok",
                meaning=REDACT_MEANING,
                subject_port="redacted",
                consumed_ports=frozenset({"source"}),
                coverage=frozenset(),
            ),
        ),
        limits=WorkflowLimits(
            max_nodes=len(nodes),
            max_bindings=6,
            max_sequence_edges=count + len(nodes),
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    workflow = admit_activation_workflow(
        workflow=raw,
        scopes=(DynamicScope(workflow=raw, maps=(), joins=(), loops=()),),
        limits=DynamicLimits(
            max_maps=0,
            max_joins=0,
            max_loops=0,
            max_children_per_map=0,
            max_iterations_per_loop=0,
            max_dynamic_depth=1,
            max_activation_occurrences=len(nodes),
        ),
    )
    capabilities = tuple(
        ImplementationCapability(
            implementation=ImplementationRef(name=spec.name, revision=1),
            operation=spec,
            configuration=FrozenConfig(fields=()),
            effect="local" if node in locals_ else "external",
            attribution="per_task",
            request_visibility="none" if node in locals_ else "dispatch_and_settlement",
            pre_dispatch_control="none" if node in locals_ else "executor",
            retry_owner="none" if node in locals_ else "implementation" if node == owner else "executor",
            error_reporting="typed_terminal",
            cancellation="before_dispatch_only",
            settlement="synchronous" if node in locals_ else "explicit_ack",
            usage="exact",
            resource_lifetime="caller_owned"
            if node == owner
            else "stateless"
            if node in {*locals_, readiness}
            else "executor_owned",
            max_physical_requests_per_activation=count if node == owner else 0 if node in locals_ else 1,
        )
        for node, spec in zip(ids, specs, strict=True)
    )
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=3 * len(nodes), max_entries=len(nodes), max_parent_depth=1),
        bound_inputs=tuple(
            BoundInput(target=target, source=target, port="source", artifact_type=text_type) for target in targets
        ),
        configuration=PreparationConfiguration(
            purpose="protection", required_protection_outcomes=frozenset({"ok"}), hard_request_limit=None
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=tuple(
            ImplementationSelection(node=node, implementation=cap.implementation, configuration=cap.configuration)
            for node, cap in zip(ids, capabilities, strict=True)
        ),
        capabilities=capabilities,
        limits=PreparationLimits(
            max_bound_inputs=size,
            max_state_revisions=0,
            max_selections=len(nodes),
            max_config_depth=1,
            max_config_atoms=2 * len(nodes),
            max_total_activation_slots=size * len(nodes),
            max_capabilities=len(nodes),
        ),
    )
    slots = {slot.template: slot.index for slot in prepared.reservation_recipe}
    policies = []
    for node, cap in zip(ids, capabilities, strict=True):
        request = (
            None
            if node in locals_
            else PhysicalRequestPolicy(
                visibility="dispatch_and_settlement",
                pre_dispatch_control="executor",
                retry_owner="implementation" if node == owner else "executor",
                replay="idempotent",
                max_attempts=count if node == owner else 1,
            )
        )
        delegated = (
            DelegatedRequestContract(
                mode="qualified",
                child_slots=tuple(slots[leaf] for leaf in leaves),
                generation_groups=tuple((slots[leaf],) for leaf in leaves),
                routes=(ChildRoute(name="primary", parse_attempts=1),),
                followups=frozenset(),
                readiness=DelegatedReadiness(
                    owner_slot=slots[readiness],
                    request_slot=slots[readiness_request],
                    probe_slot=slots[probe],
                    route=ChildRoute(name="readiness", parse_attempts=1),
                    ready_outcome="ready",
                    probe_outcomes=frozenset({"async_required"}),
                ),
            )
            if node == owner
            else None
        )
        names = tuple(outcome.name for outcome in cap.operation.outcomes)
        policies.append(
            OperationExecutionPolicy(
                node=node,
                kind="local" if node in locals_ else "delegated" if node == owner else "external",
                request=request,
                safe_detachment="forbidden",
                implementations=(
                    ExecutionImplementation(
                        implementation=cap.implementation,
                        configuration=cap.configuration,
                        capability=cap,
                        request=request,
                    ),
                ),
                result_outcomes=frozenset(names),
                runtime_outcomes=_runtime_rows(names, node not in locals_),
                delegated=delegated,
            )
        )
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=capabilities,
        policies=tuple(policies),
        decisions=(),
        assessment_productions=(
            EvidenceProductionDecl(
                node=complete,
                outcome="ok",
                promise="redaction",
                evidence_port="assessment",
                absence_queries=frozenset(),
                supported_findings=frozenset({REDACT_MATCH, REDACT_MISMATCH}),
            ),
        ),
        assessment_limits=AssessmentLimits(
            max_productions=1,
            max_findings_per_production=2,
            max_finding_code_bytes=64,
            max_absence_queries=0,
            max_assessment_facts=size,
            max_port_facts=size * 8,
            max_provenance_edges=size * 8,
        ),
    )
    loop = ensure_async_engine_loop()

    async def create_resource() -> tuple[NddPhysicalTransport, ForeignLoopTransport]:
        physical = NddPhysicalTransport()
        transport = ForeignLoopTransport.create(
            factory=lambda: physical, owner="sdk", gate_timeout_ns=1_000_000_000, exclusive=True
        )
        return physical, transport

    physical, transport = await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(create_resource(), loop))
    handed_to_runtime = False
    try:
        consumer = DetectionRedactInvocation(
            source,
            tuple(targets),
            detection,
            model_configs,
            selected_models,
            physical,
            dict(zip(generations, (slots[leaf] for leaf in leaves), strict=True)),
            detection_options,
            "redacted",
            "ok",
        )
        worker = WorkerDelegatedSubmitter(
            callback=consumer._run,
            pool=pool,
            affinity=BlockingAffinity(cross_thread=True, concurrency_safe=False, close_on_event_loop=False),
            borrowed_loop=loop,
        )
        lease = ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=worker)
        handles = tuple(
            ImplementationHandle(
                implementation=cap.implementation,
                operation=cap.operation,
                configuration=cap.configuration,
                local=RedactAssessment()
                if node == complete
                else _ProfileLocal("async_required")
                if node == probe
                else None,
                transport=transport if node not in {owner, *locals_, readiness} else None,
                delegated=worker if node == owner else None,
                resource=lease
                if node == owner
                else transport.lease
                if node not in {owner, *locals_, readiness}
                else None,
            )
            for node, cap in zip(ids, capabilities, strict=True)
            if node != readiness
        )
        services = ExecutionServices(
            handles=handles,
            context_resources=(),
            limits=limits,
            decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
            clock=_Clock(),
        )
        running = await start_execution(admitted=admitted, capabilities=capabilities, services=services)
        handed_to_runtime = True
        completion = asyncio.create_task(running.wait())
        try:
            return await asyncio.shield(completion)
        except asyncio.CancelledError:
            running.request_cancel()
            while not completion.done():
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError:
                    continue
            raise
    finally:
        if not handed_to_runtime and not physical.closed:
            await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(physical.close(), loop))


def qualify_detection_redact(result: ExecutionResult) -> QualificationResult:
    """Reconcile this profile's source checks and physical receipts before release."""
    execution = result._execution
    prepared = execution.context.prepared
    limits = execution.assessment_limits
    admitted = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=QualificationLimits(
            max_productions=limits.max_productions,
            max_submissions=limits.max_assessment_facts,
            max_port_facts=limits.max_port_facts,
            max_consumed_per_assessment=3,
            max_coverage_atoms=0,
            max_verified_evidence=limits.max_assessment_facts,
            max_required_decisions=0,
            max_fixed_point_steps=len(prepared.target_occurrences) + 1,
            max_revision_entries=limits.max_port_facts + prepared.limits.max_capabilities,
            max_absence_revisions=0,
            max_provenance_edges=limits.max_provenance_edges,
        ),
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(reference for reference, _ in result.artifacts),
        absences=(),
        configurations=tuple(
            (policy.node, policy.implementations[0].configuration)
            for policy in execution.policies
            if any(production.node == policy.node for production in execution.assessment_productions)
        ),
        state=prepared.state,
    )
    return qualify_retained(admitted=admitted, result=result, current=current)
