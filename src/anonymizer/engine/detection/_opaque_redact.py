# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-frame compatibility consumers under nonqualifying graph execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any

import pandas as pd
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import ReplaceMethod
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.detection._frame_publication import InvocationFrame
from anonymizer.engine.detection._graph_invocation import DetectionRedactInvocation
from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.graph_sdk._artifact_shape import content_bytes
from anonymizer.engine.graph_sdk._execution_delegated import WorkerDelegatedSubmitter
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.blocking import BlockingAffinity, BlockingPool
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationCapability,
    ImplementationRef,
    ImplementationSelection,
)
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    DelegatedRequestContract,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionResult,
    ExecutionServices,
    ImplementationHandle,
    IngressLimits,
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
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
from anonymizer.engine.graph_sdk.requests import AssociationInput
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.engine.replace.replace_runner import ReplacementResult, ReplacementWorkflow
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    DynamicLimits,
    DynamicScope,
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
    ResourceCeiling,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_activation_workflow,
    admit_static_workflow,
)


@dataclass(frozen=True)
class _InjectedRedact:
    source: pd.DataFrame
    detection: EntityDetectionWorkflow
    replacement: ReplacementWorkflow
    model_configs: list[ModelConfig]
    selected_models: DetectionModelSelection
    replacement_models: ReplaceModelSelection
    detection_options: Mapping[str, Any]
    replace_method: ReplaceMethod
    handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure] | None = None

    def run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        return (
            self._run(inputs, requests) if self.handoff is None else self.handoff(lambda: self._run(inputs, requests))
        )

    def _run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        del inputs
        if requests.cancelled:
            return LocalFailure(failure="implementation_exception")
        started = perf_counter()
        detected = self.detection.run(
            self.source,
            model_configs=self.model_configs,
            selected_models=self.selected_models,
            **self.detection_options,
        )
        if requests.cancelled:
            return LocalFailure(failure="implementation_exception")
        detection_elapsed = perf_counter() - started
        started = perf_counter()
        replaced = self.replacement.run(
            detected.dataframe,
            replace_method=self.replace_method,
            model_configs=self.model_configs,
            selected_models=self.replacement_models,
            preview_num_records=self.detection_options.get("preview_num_records"),
        )
        return LocalCompleted(
            results=(),
            frame=InvocationFrame(
                dataframe=replaced.dataframe.copy(),
                detected_dataframe=detected.dataframe.copy(),
                detection_failures=tuple(detected.failed_records),
                replacement_failures=tuple(replaced.failed_records),
                detection_elapsed=detection_elapsed,
                replacement_elapsed=perf_counter() - started,
                retained_owner=getattr(replaced, "_execution", None),
            ),
        )


@dataclass(frozen=True)
class _EvaluationInvocation:
    """One judge callback with an independently retained prior execution owner."""

    source: pd.DataFrame
    evaluate: Callable[[DelegatedRequests], ReplacementResult | LocalFailure]
    retained_owner: object | None
    handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure]

    def run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        def evaluate() -> LocalCompleted | LocalFailure:
            if requests.cancelled:
                return LocalFailure(failure="implementation_exception")
            started = perf_counter()
            result = self.evaluate(requests)
            if isinstance(result, LocalFailure):
                return result
            return LocalCompleted(
                results=(),
                frame=InvocationFrame(
                    dataframe=result.dataframe,
                    detected_dataframe=self.source,
                    detection_failures=(),
                    replacement_failures=tuple(result.failed_records),
                    detection_elapsed=0.0,
                    replacement_elapsed=perf_counter() - started,
                    retained_owner=(self.retained_owner, result._execution),
                ),
            )

        return self.handoff(evaluate)


def run_evaluation(
    source: pd.DataFrame,
    evaluate: Callable[[DelegatedRequests], ReplacementResult | LocalFailure],
    *,
    retained_owner: object | None,
) -> ReplacementResult:
    """Evaluate a retained public frame under fresh, nonqualifying graph authority."""
    from anonymizer.engine.detection._graph_profile import _run_sync_profile, project_detection_redact

    async def operation(
        pool: BlockingPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> ReplacementResult:
        result = await execute_injected_redact(_EvaluationInvocation(source, evaluate, retained_owner, handoff), pool)
        return project_detection_redact(result)

    return _run_sync_profile(operation, caller_callbacks=True)


async def execute_injected_redact(
    callback: _InjectedRedact | DetectionRedactInvocation | _EvaluationInvocation, pool: BlockingPool
) -> ExecutionResult:
    from anonymizer.engine.detection._graph_profile import _await_profile_execution, _Clock, _runtime_rows

    authenticated = isinstance(callback, DetectionRedactInvocation)
    preview = (
        None if isinstance(callback, _EvaluationInvocation) else callback.detection_options.get("preview_num_records")
    )
    source = callback.source if preview is None else callback.source.iloc[:preview]
    # Evaluation owns a whole public frame; it makes no semantic source-association claim.
    texts = () if isinstance(callback, _EvaluationInvocation) else tuple(source[COL_TEXT])
    graph = DataGraph.new()
    targets = []
    for text in texts:
        graph, target = graph.add_text(text)
        if authenticated:
            targets.append(target)
    if authenticated:
        callback = replace(callback, targets=tuple(targets))
    size = len(targets)
    characters = max((len(text) for text in texts), default=0)
    total_characters = sum(len(text) for text in texts)
    text_type = ArtifactType(name="anonymizer.text", revision=1)
    outputs = (
        (
            OutputPort(name="redacted", artifact_type=text_type),
            OutputPort(name="spans", artifact_type=ENTITY_SPANS_TYPE),
        )
        if size
        else ()
    )
    data = graph.validate(
        targets=tuple(targets),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(
            max_datums=len(texts),
            max_targets=size,
            max_text_bytes=sum(content_bytes(text) for text in texts),
            max_declarations=0,
            max_group_members=0,
        ),
    )
    owner = WorkflowId.new()
    node = NodeId.new(workflow=owner)
    operation = OperationSpec(
        name="opaque-evaluation"
        if isinstance(callback, _EvaluationInvocation)
        else "dd-detection-replace"
        if authenticated
        else "opaque-detection-replace",
        inputs=(InputPort(name="source", artifact_type=text_type),) if size else (),
        outputs=outputs,
        output_dependencies=tuple(
            OutputDependency(output=port.name, inputs=frozenset({"source"}), identity_input=None) for port in outputs
        ),
        outcomes=(
            OutcomeSpec(
                name="ok",
                category="success",
                produced_ports=frozenset(port.name for port in outputs),
                context=frozenset(),
                evidence=frozenset(),
                state_effects=frozenset(),
                model_requirements=frozenset(),
                ceiling=ResourceCeiling(
                    max_activations=1, max_model_requests=None, max_input_bytes=None, max_output_bytes=None
                ),
            ),
        ),
    )
    interface = replace(
        operation,
        outputs=outputs[:1],
        output_dependencies=operation.output_dependencies[:1],
        outcomes=tuple(
            replace(outcome, produced_ports=frozenset(port.name for port in outputs[:1]))
            for outcome in operation.outcomes
        ),
    )
    raw = admit_static_workflow(
        workflow=owner,
        interface=interface,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=(
            InputBinding(source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=node, port="source")),
        )
        if size
        else (),
        output_bindings=tuple(
            OutputBinding(
                source=NodeOutputRef(node=node, port=port.name), destination=WorkflowOutputRef(port=port.name)
            )
            for port in interface.outputs
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        sequence=(),
        choices=(),
        protection=(),
        limits=WorkflowLimits(
            max_nodes=1,
            max_bindings=4,
            max_sequence_edges=0,
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
            max_activation_occurrences=1,
        ),
    )
    capability = ImplementationCapability(
        implementation=ImplementationRef(name=operation.name, revision=1),
        operation=operation,
        configuration=callback.redact_configuration
        if isinstance(callback, DetectionRedactInvocation)
        else FrozenConfig(fields=()),
        effect="external",
        attribution="per_task" if authenticated else "aggregate_only",
        request_visibility="none",
        pre_dispatch_control="none",
        retry_owner="implementation",
        error_reporting="typed_terminal",
        cancellation="unobservable",
        settlement="uncertain_possible",
        usage="unknown",
        resource_lifetime="caller_owned",
        max_physical_requests_per_activation=None,
    )
    prepared = prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=3, max_entries=1, max_parent_depth=1),
        bound_inputs=tuple(
            BoundInput(target=target, source=target, port="source", artifact_type=text_type) for target in targets
        ),
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
            invocation_owned=not targets,
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=(
            ImplementationSelection(
                node=node, implementation=capability.implementation, configuration=capability.configuration
            ),
        ),
        capabilities=(capability,),
        limits=PreparationLimits(
            max_bound_inputs=size,
            max_state_revisions=0,
            max_selections=1,
            max_config_depth=3,
            max_config_atoms=12,
            max_total_activation_slots=max(1, size),
            max_capabilities=1,
        ),
    )
    implementation = ExecutionImplementation(
        implementation=capability.implementation,
        configuration=capability.configuration,
        capability=capability,
        request=None,
    )
    execution = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=(capability,),
        policies=(
            OperationExecutionPolicy(
                node=node,
                kind="delegated",
                request=None,
                safe_detachment="forbidden",
                implementations=(implementation,),
                result_outcomes=frozenset({"ok"}),
                runtime_outcomes=_runtime_rows(("ok",), True),
                delegated=DelegatedRequestContract(
                    mode="opaque", child_slots=(), routes=(), followups=frozenset(), usage_requirement="unknown_allowed"
                ),
            ),
        ),
        decisions=(),
        assessment_productions=(),
        ingress_limits=IngressLimits(
            max_associations=max(1, size),
            max_outputs_per_association=2,
            # Ingress counts spans across the batch; publication checks each artifact.
            max_collection_items=max(1, total_characters),
            max_assessments=1,
            max_metadata_items=max(7, 7 * size + 3 * total_characters),
            max_text_bytes=None,
            max_metadata_bytes=None,
            max_depth=10,
        ),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=3 * size,
            max_provenance_edges=2 * size,
        ),
    )
    worker = WorkerDelegatedSubmitter(
        callback=callback._run if isinstance(callback, DetectionRedactInvocation) else callback.run,
        pool=pool,
        affinity=BlockingAffinity(cross_thread=True, concurrency_safe=False, close_on_event_loop=False),
        borrowed_loop=asyncio.get_running_loop(),
    )
    services = ExecutionServices(
        handles=(
            ImplementationHandle(
                implementation=capability.implementation,
                operation=operation,
                configuration=capability.configuration,
                local=None,
                transport=None,
                delegated=worker,
                resource=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=worker),
            ),
        ),
        context_resources=(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=None,
            max_runtime_artifacts=3 * size,
            max_runtime_artifact_bytes=None,
            max_collection_items=characters,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_Clock(),
    )
    return await _await_profile_execution(
        await start_execution(admitted=execution, capabilities=(capability,), services=services)
    )
