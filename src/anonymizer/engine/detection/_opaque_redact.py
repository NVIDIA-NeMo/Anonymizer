# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Existing injected callbacks under one nonqualifying graph invocation."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import pandas as pd
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection, ReplaceModelSelection
from anonymizer.config.replace_strategies import Redact
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.detection._frame_publication import InvocationFrame
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
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    PreparationConfiguration,
    PreparationLimits,
    StateRevisionView,
    prepare,
)
from anonymizer.engine.graph_sdk.requests import AssociationInput
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.engine.replace.replace_runner import ReplacementWorkflow
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    DynamicLimits,
    DynamicScope,
    NodeId,
    NodeOutcomeRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeSpec,
    ResourceCeiling,
    WorkflowId,
    WorkflowLimits,
    WorkflowOutcomeRef,
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
    replace_method: Redact

    def run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        del inputs
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


async def execute_injected_redact(callback: _InjectedRedact, pool: BlockingPool) -> ExecutionResult:
    from anonymizer.engine.detection._graph_profile import _await_profile_execution, _Clock, _runtime_rows

    graph = DataGraph.new()
    for text in callback.source[COL_TEXT]:
        graph, _ = graph.add_text(text)
    data = graph.validate(
        targets=(),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(
            max_datums=len(callback.source),
            max_targets=0,
            max_text_bytes=sum(content_bytes(text) for text in callback.source[COL_TEXT]),
            max_declarations=0,
            max_group_members=0,
        ),
    )
    owner = WorkflowId.new()
    node = NodeId.new(workflow=owner)
    operation = OperationSpec(
        name="injected-detection-redact",
        inputs=(),
        outputs=(),
        output_dependencies=(),
        outcomes=(
            OutcomeSpec(
                name="ok",
                category="success",
                produced_ports=frozenset(),
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
    raw = admit_static_workflow(
        workflow=owner,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=(),
        output_bindings=(),
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
            max_bindings=1,
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
        implementation=ImplementationRef(name="injected-detection-redact", revision=1),
        operation=operation,
        configuration=FrozenConfig(fields=()),
        effect="external",
        attribution="aggregate_only",
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
        bound_inputs=(),
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
            invocation_owned=True,
        ),
        state=StateRevisionView(revisions=frozenset()),
        selections=(
            ImplementationSelection(
                node=node, implementation=capability.implementation, configuration=capability.configuration
            ),
        ),
        capabilities=(capability,),
        limits=PreparationLimits(
            max_bound_inputs=0,
            max_state_revisions=0,
            max_selections=1,
            max_config_depth=1,
            max_config_atoms=2,
            max_total_activation_slots=1,
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
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=0,
            max_provenance_edges=0,
        ),
    )
    worker = WorkerDelegatedSubmitter(
        callback=callback.run,
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
            max_remote_outstanding=0,
            max_runtime_artifacts=0,
            max_runtime_artifact_bytes=None,
            max_collection_items=0,
        ),
        decision_limits=DecisionLimits(max_pending=0, max_lifetime_ns=0),
        clock=_Clock(),
    )
    return await _await_profile_execution(
        await start_execution(admitted=execution, capabilities=(capability,), services=services)
    )
