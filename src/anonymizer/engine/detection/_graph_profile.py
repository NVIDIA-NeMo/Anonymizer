# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DD detection/Redact consumers of the single graph executor.

The public facade selects execution-only publication with unknown physical
request visibility. The separate qualified development profile retains explicit
child-request admission and its narrower backend conformance requirements.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Future
from contextvars import copy_context
from dataclasses import dataclass, replace
from functools import partial
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic_ns
from typing import Any, cast

import pandas as pd
from data_designer.config import ModelConfig
from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Annotate, Hash, Redact, ReplaceMethod
from anonymizer.engine.constants import COL_AUGMENTED_ENTITIES, COL_RAW_DETECTED, COL_TEXT, COL_VALIDATION_DECISIONS
from anonymizer.engine.detection._graph_invocation import DetectionRedactInvocation
from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE
from anonymizer.engine.detection.detection_workflow import (
    _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL,
    EntityDetectionWorkflow,
)
from anonymizer.engine.graph_sdk._artifact_shape import content_bytes
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
    DelegatedGenerationSelection,
    DelegatedReadiness,
    DelegatedRequestContract,
    EvidenceProductionDecl,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    IngressLimits,
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
    RunningExecution,
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
    local_configuration,
    redact_configuration,
)
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
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
    compile_graph,
)
from anonymizer.interface.errors import AnonymizerWorkflowError


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


def run_detection_redact(
    source: pd.DataFrame,
    *,
    detection: EntityDetectionWorkflow,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    detection_options: Mapping[str, Any],
    limits: ExecutionLimits | None = None,
    replace_method: ReplaceMethod | None = None,
    replacement: ReplacementWorkflow | None = None,
    replacement_models: ReplaceModelSelection | None = None,
    opaque: bool = False,
) -> ReplacementResult:
    """Run the admitted profile synchronously with one owned callback worker.

    Counts follow the selected source and declared work. Callers may explicitly
    supply tighter count or byte limits. A notebook's active loop
    stays untouched; the host thread runs the same authoritative async executor,
    while actual DD clients remain on DD's existing foreign loop.
    """
    if (replacement is not None or opaque) and limits is not None:
        raise ValueError("Opaque injected execution cannot guarantee caller-supplied execution limits")
    selected_method = replace_method.model_copy(deep=True) if replace_method is not None else Redact()
    if replacement is None and not isinstance(selected_method, (Redact, Hash, Annotate)):
        raise ValueError("Substitute requires a replacement workflow")
    if replacement is None and not opaque and not isinstance(selected_method, Redact):
        raise ValueError("The qualified detection profile requires Redact")
    local_config = (
        local_configuration(selected_method)
        if replacement is None and isinstance(selected_method, (Redact, Hash, Annotate))
        else None
    )
    selected_limits = replace(limits) if limits is not None else None

    async def operation(
        pool: BlockingPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> ReplacementResult:
        if replacement is not None:
            from anonymizer.engine.detection._opaque_redact import _DetectionInvocation, execute_injected_redact

            assert replacement_models is not None and replace_method is not None
            result = await execute_injected_redact(
                _DetectionInvocation(
                    source,
                    detection,
                    partial(
                        replacement.run,
                        replace_method=replace_method,
                        model_configs=model_configs,
                        selected_models=replacement_models,
                        preview_num_records=detection_options.get("preview_num_records"),
                    ),
                    model_configs,
                    selected_models,
                    detection_options,
                    handoff=handoff,
                ),
                pool,
            )
            return project_detection_redact(result)
        if opaque:
            from anonymizer.engine.detection._opaque_redact import execute_injected_redact

            assert local_config is not None
            result = await execute_injected_redact(
                DetectionRedactInvocation(
                    source,
                    (),
                    detection,
                    model_configs,
                    selected_models,
                    local_config,
                    None,
                    {},
                    detection_options,
                    "redacted",
                    "ok",
                ),
                pool,
            )
            return project_detection_redact(result)
        result = await execute_detection_redact(
            source,
            detection=detection,
            model_configs=model_configs,
            selected_models=selected_models,
            detection_options=detection_options,
            replace_method=cast(Redact, selected_method),
            pool=pool,
            limits=selected_limits,
        )
        return project_detection_redact(result)

    return _run_sync_profile(operation, caller_callbacks=replacement is not None)


def _run_sync_profile(
    operation: Callable[
        [BlockingPool, Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure]],
        Awaitable[ReplacementResult],
    ],
    *,
    caller_callbacks: bool,
) -> ReplacementResult:
    """Host one graph invocation while preserving synchronous caller-thread callbacks."""
    started = Event()
    cancellation: list[tuple[asyncio.AbstractEventLoop, asyncio.Task[Any]]] = []
    callbacks: Queue[tuple[Callable[[], LocalCompleted | LocalFailure], Future[LocalCompleted | LocalFailure]]] = Queue(
        maxsize=1
    )

    def handoff(callback: Callable[[], LocalCompleted | LocalFailure]) -> LocalCompleted | LocalFailure:
        returned: Future[LocalCompleted | LocalFailure] = Future()
        callbacks.put((callback, returned))
        return returned.result()

    def serve_callback() -> None:
        try:
            callback, returned = callbacks.get(timeout=0.01)
        except Empty:
            return
        if interrupted is not None:
            returned.set_result(LocalFailure(failure="implementation_exception"))
            return
        try:
            returned.set_result(callback())
        except Exception as error:
            returned.set_exception(error)
        except BaseException:
            if cancellation:
                loop, task = cancellation[0]
                loop.call_soon_threadsafe(task.cancel)
            returned.set_result(LocalFailure(failure="implementation_exception"))
            raise

    async def run() -> ReplacementResult:
        task = asyncio.current_task()
        assert task is not None
        cancellation.append((asyncio.get_running_loop(), task))
        started.set()
        pool = BlockingPool(max_workers=1)
        try:
            return await operation(pool, handoff)
        finally:
            await pool.close()

    completion: Future[ReplacementResult] = Future()

    def host() -> None:
        try:
            completion.set_result(asyncio.run(run()))
        except BaseException as error:
            completion.set_exception(error)
        finally:
            started.set()

    context = copy_context()
    thread = Thread(target=lambda: context.run(host), name="anonymizer-graph-host")
    thread.start()
    interrupted: BaseException | None = None
    try:
        if caller_callbacks:
            while not completion.done():
                serve_callback()
        return completion.result()
    except BaseException as error:
        interrupted = error
        raise
    finally:
        # Cancellation only requests a stop. Keep the host and pool alive until
        # the existing executor has drained the actual callback and client work.
        while thread.is_alive():
            try:
                if interrupted is not None:
                    started.wait()
                    if cancellation:
                        loop, task = cancellation[0]
                        if not loop.is_closed():
                            try:
                                loop.call_soon_threadsafe(task.cancel)
                            except RuntimeError:
                                if not loop.is_closed():
                                    raise
                serve_callback()
                thread.join(timeout=0.01)
            except KeyboardInterrupt as error:
                interrupted = error
        if interrupted is not None:
            failure = completion.exception()
            owner = (
                getattr(failure, "_anonymizer_execution", None)
                if failure is not None
                else completion.result()._execution
            )
            if owner is not None:
                setattr(interrupted, "_anonymizer_execution", owner)
            raise interrupted


async def execute_detection_redact(
    source: pd.DataFrame,
    *,
    detection: EntityDetectionWorkflow,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    detection_options: Mapping[str, Any],
    validation_chunks: int | None = None,
    replace_method: Redact | None = None,
    pool: BlockingPool,
    limits: ExecutionLimits | None = None,
) -> ExecutionResult:
    """Admit and execute the named real-DD profile without a second workflow runner."""
    redaction = redact_configuration(replace_method if replace_method is not None else Redact())
    if limits is not None:
        limits = replace(limits)
    preview = detection_options.get("preview_num_records")
    if preview is not None and (type(preview) is not int or preview < 0):
        raise ValueError("preview_num_records must be a nonnegative integer")
    selected_source = source if preview is None else source.iloc[:preview]
    if validation_chunks is None:
        chunk_size = detection_options.get(
            "validation_max_entities_per_call", _DEFAULT_VALIDATION_MAX_ENTITIES_PER_CALL
        )
        if type(chunk_size) is not int or chunk_size < 1:
            raise ValueError("validation_max_entities_per_call must be a positive integer")
        # parse_raw_entities admits only nonempty, in-range, nonoverlapping spans.
        # Thus detector-only validation has at most one candidate per character.
        # This reserves capacity from existing input, without limiting public input.
        characters = max((len(text) for text in selected_source[COL_TEXT]), default=0)
        validation_chunks = max(1, (characters + chunk_size - 1) // chunk_size)
    if type(validation_chunks) is not int or validation_chunks < 1:
        raise ValueError("validator capacity must be a positive integer")
    generations = (
        (COL_RAW_DETECTED, 0),
        *((COL_VALIDATION_DECISIONS, i) for i in range(validation_chunks)),
        (COL_AUGMENTED_ENTITIES, 0),
    )
    run_config = detection._adapter._data_designer.run_config
    parser_capacity = (run_config.max_conversation_correction_steps + 1) * (run_config.max_conversation_restarts + 1)
    group_sizes = (parser_capacity,) + (1,) * validation_chunks + (parser_capacity,)
    if len(selected_source) == 0:
        generations, group_sizes = (), ()
    graph = DataGraph.new()
    targets = []
    for text in selected_source[COL_TEXT]:
        graph, target = graph.add_text(text)
        targets.append(target)
    count = sum(group_sizes)
    size = len(targets)
    if limits is None:
        limits = ExecutionLimits(
            max_local_in_flight=max(1, size + 1),
            max_remote_outstanding=max(1, size * count + 2),
            # Each source retains one root, three producer outputs and one assessment.
            max_runtime_artifacts=5 * size,
            max_runtime_artifact_bytes=None,
            max_collection_items=max(
                validation_chunks, max((len(text) for text in selected_source[COL_TEXT]), default=0)
            ),
        )
    byte_limit = limits.max_runtime_artifact_bytes
    span_capacity = sum(len(text) for text in selected_source[COL_TEXT])
    chunk_capacity = size * validation_chunks
    data = graph.validate(
        targets=tuple(targets),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(
            max_datums=size,
            max_targets=size,
            max_text_bytes=(
                sum(content_bytes(text) for text in selected_source[COL_TEXT]) if byte_limit is None else byte_limit
            ),
            max_declarations=0,
            max_group_members=0,
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
                OutputPort(
                    name="validation_chunks",
                    artifact_type=ArtifactType(name="anonymizer.validation-chunks", revision=1),
                ),
            )
            if io
            else (),
            output_dependencies=tuple(
                OutputDependency(output=port, inputs=frozenset({"source"}), identity_input=None)
                for port in ("redacted", "spans", "validation_chunks")
            )
            if io
            else (),
            outcomes=tuple(
                OutcomeSpec(
                    name=value,
                    category="success",
                    produced_ports=frozenset({"redacted", "spans", "validation_chunks"}) if io else frozenset(),
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
    if not targets:
        specs = [operation("dd-empty-detection", ("ok",), 1), *specs[1:4], operation("dd-empty-complete", ("ok",), 0)]
        ids = [owner, readiness, readiness_request, probe, complete]
    nodes = tuple(OperationNode(id=node, operation=spec) for node, spec in zip(ids, specs, strict=True))
    if not targets:
        interface = replace(
            specs[0],
            name="dd-empty-detection-profile",
            outcomes=(
                replace(
                    specs[0].outcomes[0],
                    ceiling=replace(
                        ceiling,
                        max_activations=len(nodes),
                        max_model_requests=3,
                        max_input_bytes=None if byte_limit is None else byte_limit * len(nodes),
                        max_output_bytes=None if byte_limit is None else byte_limit * len(nodes),
                    ),
                ),
            ),
        )
    else:
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
                        max_input_bytes=None if byte_limit is None else byte_limit * len(nodes),
                        max_output_bytes=None if byte_limit is None else byte_limit * len(nodes),
                    ),
                ),
            ),
        )
    raw = compile_graph(
        workflow=workflow_id,
        interface=interface,
        nodes=nodes,
        input_bindings=()
        if not targets
        else (
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
        output_bindings=()
        if not targets
        else (
            OutputBinding(
                source=NodeOutputRef(node=owner, port="redacted"), destination=WorkflowOutputRef(port="redacted")
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=complete, outcome="ok"),
                destination=WorkflowOutcomeRef(outcome="ok"),
            ),
        ),
        sequence=(
            SequenceEdge(before=readiness, after=probe),
            *(SequenceEdge(before=probe, after=leaf) for leaf in leaves),
            *(SequenceEdge(before=node, after=complete) for node in ids if node != complete),
        ),
        protection=()
        if not targets
        else (
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
    workflow = raw
    capabilities = tuple(
        ImplementationCapability(
            implementation=ImplementationRef(name=spec.name, revision=1),
            operation=spec,
            configuration=redaction if node in {owner, complete} else FrozenConfig(fields=()),
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
            max_physical_requests_per_activation=max(1, count) if node == owner else 0 if node in locals_ else 1,
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
            purpose="protection" if targets else "execution_only",
            required_protection_outcomes=frozenset({"ok"}) if targets else frozenset(),
            hard_request_limit=None,
            invocation_owned=not targets,
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
            max_config_depth=3,
            # Producer and assessor each add five atoms, counted in selection and capability.
            max_config_atoms=2 * len(nodes) + 20,
            max_total_activation_slots=max(size, 1) * len(nodes),
            max_capabilities=len(nodes),
        ),
    )
    slots = {slot.template: slot.index for slot in prepared.reservation_recipe}
    groups = []
    position = 0
    for length in group_sizes:
        groups.append(tuple(slots[leaf] for leaf in leaves[position : position + length]))
        position += length
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
                max_attempts=max(1, count) if node == owner else 1,
            )
        )
        delegated = (
            DelegatedRequestContract(
                mode="qualified",
                child_slots=tuple(slots[leaf] for leaf in leaves),
                generation_groups=tuple(groups),
                selection=DelegatedGenerationSelection(
                    port="validation_chunks", slots=tuple(group[0] for group in groups[1:-1])
                )
                if targets
                else None,
                routes=(ChildRoute(name="primary", parse_attempts=parser_capacity if targets else 1),),
                followups=frozenset({"correction", "conversation_restart"}),
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
        assessment_productions=()
        if not targets
        else (
            EvidenceProductionDecl(
                node=complete,
                outcome="ok",
                promise="redaction",
                evidence_port="assessment",
                absence_queries=frozenset(),
                supported_findings=frozenset({REDACT_MATCH, REDACT_MISMATCH}),
            ),
        ),
        ingress_limits=IngressLimits(
            max_associations=max(1, size),
            max_outputs_per_association=3,
            # Final nonoverlapping spans are bounded by source characters;
            # validation membership has one entry per admitted chunk per row.
            max_collection_items=max(1, span_capacity + chunk_capacity),
            max_assessments=max(1, size),
            # Per row: association/outcome (2), three port/type names (6),
            # span-source reference (1); three scalar fields per span and
            # key/version per validation member. The assessment response is smaller.
            max_metadata_items=max(9, 9 * size + 3 * span_capacity + 2 * chunk_capacity),
            max_text_bytes=None,
            max_metadata_bytes=None,
            max_depth=10,
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
        try:
            transport = ForeignLoopTransport.create(
                factory=lambda: physical, owner="sdk", gate_timeout_ns=1_000_000_000, exclusive=True
            )
            return physical, transport
        except BaseException:
            await physical.close()
            raise

    acquisition = asyncio.wrap_future(asyncio.run_coroutine_threadsafe(create_resource(), loop))
    transport = None
    handed_to_runtime = False
    interrupted: asyncio.CancelledError | None = None
    try:
        # Cancelling this observer must not cancel or abandon the foreign factory.
        # Own its actual completion before either transferring or closing the lease.
        while True:
            try:
                physical, transport = await asyncio.shield(acquisition)
                break
            except asyncio.CancelledError as error:
                if acquisition.cancelled():
                    raise
                if interrupted is None:
                    interrupted = error
            except BaseException as error:
                if interrupted is not None:
                    raise interrupted from error
                raise
        if interrupted is not None:
            raise interrupted
        consumer = DetectionRedactInvocation(
            source,
            tuple(targets),
            detection,
            model_configs,
            selected_models,
            redaction,
            physical,
            dict(zip(generations, groups, strict=True)),
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
                local=(RedactAssessment(configuration=cap.configuration) if targets else _ProfileLocal("ok"))
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
        return await _await_profile_execution(running)
    finally:
        if not handed_to_runtime and transport is not None:
            await transport.close()


async def _await_profile_execution(running: RunningExecution) -> ExecutionResult:
    """Retain the actual owner until cancellation and callback settlement finish."""
    completion = asyncio.create_task(running.wait())
    try:
        return await asyncio.shield(completion)
    except asyncio.CancelledError as error:
        running.request_cancel()
        while not completion.done():
            try:
                await asyncio.shield(completion)
            except asyncio.CancelledError:
                continue
        setattr(error, "_anonymizer_execution", completion.result())
        raise


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


def project_detection_redact(result: ExecutionResult) -> ReplacementResult:
    """Project canonical content under its selected purpose, preserving original errors."""
    for failure in result.submission_failures:
        # Failure projection does not upgrade eligibility or alter physical history.
        # Keeping the canonical execution on the raised error also retains any
        # forbidden-detachment lease while the caller handles the failure.
        setattr(failure.cause, "_anonymizer_execution", result)
        raise failure.cause
    from anonymizer.engine.detection._frame_publication import InvocationFrame

    if (
        result._execution.context.prepared.configuration.purpose == "execution_only"
        and len(result.frames) == 1
        and type(result.frames[0]) is InvocationFrame
    ):
        frame = result.frames[0]
        if result.invocation_state is None or any(
            entry.status != "success" for entry in result.invocation_state.entries
        ):
            raise AnonymizerWorkflowError("Opaque invocation did not complete")
        return ReplacementResult(
            dataframe=frame.project(result), failed_records=list(frame.failed_records), _execution=result
        )
    if result._execution.context.prepared.configuration.purpose == "execution_only" and len(result.frames) == 1:
        frame = result.frames[0]
        return ReplacementResult(
            dataframe=frame.project(result), failed_records=list(frame.failed_records), _execution=result
        )
    qualified = qualify_detection_redact(result)
    if len(result.frames) != 1 or len(qualified.qualified) != len(result.final_outputs):
        error = AnonymizerWorkflowError("Detection/Redact execution did not qualify all returned sources")
        setattr(error, "_anonymizer_execution", result)
        raise error
    frame = result.frames[0]
    return ReplacementResult(
        dataframe=frame.project(result), failed_records=list(frame.failed_records), _execution=result
    )
