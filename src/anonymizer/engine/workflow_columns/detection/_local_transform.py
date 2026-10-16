# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone registered transforms use one local, source-free graph invocation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pandas as pd

from anonymizer.engine.detection._frame_publication import InvocationFrame
from anonymizer.engine.detection._graph_profile import (
    _Clock,
    _run_sync_profile,
    _runtime_rows,
    project_detection_redact,
)
from anonymizer.engine.detection._opaque_redact import _prepare_frame_plan
from anonymizer.engine.graph_sdk.blocking import BlockingPool
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, ImplementationCapability, ImplementationRef
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DecisionLimits,
    ExecutionImplementation,
    ExecutionLimits,
    ExecutionServices,
    ImplementationHandle,
    IngressLimits,
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.requests import AssociationInput
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.graph.workflow import (
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
    compile_graph,
)


class _CallbackCancelled(Exception):
    def __init__(self, cause: asyncio.CancelledError) -> None:
        self.cause = cause


class _Transform:
    def __init__(
        self,
        callback: Callable[[], dict[str, Any]],
        column: str,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> None:
        self.callback, self.column, self.handoff = callback, column, handoff
        self.cancellation: asyncio.CancelledError | None = None

    async def run(self, request: tuple[AssociationInput, ...]) -> LocalCompleted | LocalFailure:
        if request:
            raise ValueError("source-free transform received semantic inputs")

        def apply() -> LocalCompleted:
            try:
                row = self.callback()
            except asyncio.CancelledError as error:
                raise _CallbackCancelled(error) from error
            frame = pd.DataFrame({self.column: [row]})
            return LocalCompleted(
                results=(),
                frame=InvocationFrame(
                    dataframe=frame,
                    detected_dataframe=frame,
                    detection_failures=(),
                    replacement_failures=(),
                    detection_elapsed=0.0,
                    replacement_elapsed=0.0,
                ),
            )

        try:
            return self.handoff(apply)
        except _CallbackCancelled as error:
            self.cancellation = error.cause
            raise error.cause


def run_transform(callback: Callable[[], dict[str, Any]], *, name: str, column: str) -> dict[str, Any]:
    """Project only the canonically accepted dictionary, preserving native object cells."""
    owner = WorkflowId.new()
    node = NodeId.new(workflow=owner)
    operation = OperationSpec(
        name=name,
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
                model_requirements=frozenset(),
                ceiling=ResourceCeiling(
                    max_activations=1, max_model_requests=0, max_input_bytes=None, max_output_bytes=None
                ),
            ),
        ),
    )
    raw = compile_graph(
        workflow=owner,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=(),
        output_bindings=(),
        sequence=(),
        protection=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
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
    capability = ImplementationCapability(
        implementation=ImplementationRef(name=name, revision=1),
        operation=operation,
        configuration=FrozenConfig(fields=()),
        effect="local",
        attribution="per_task",
        request_visibility="none",
        pre_dispatch_control="none",
        retry_owner="none",
        error_reporting="typed_terminal",
        cancellation="cooperative_ack",
        settlement="synchronous",
        usage="exact",
        resource_lifetime="stateless",
        max_physical_requests_per_activation=0,
    )
    prepared = _prepare_frame_plan(
        DataGraph.new().validate(
            targets=(),
            source_relations=(),
            contexts=(),
            dependencies=(),
            coherence=(),
            atomic=(),
            output_regions=(),
            limits=DataLimits(max_datums=0, max_targets=0, max_text_bytes=0, max_declarations=0, max_group_members=0),
        ),
        raw,
        (capability,),
        (),
        invocation_owned=True,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="local",
        request=None,
        safe_detachment="forbidden",
        implementations=(
            ExecutionImplementation(
                implementation=capability.implementation,
                configuration=capability.configuration,
                capability=capability,
                request=None,
            ),
        ),
        result_outcomes=frozenset({"ok"}),
        runtime_outcomes=_runtime_rows(("ok",), False),
    )
    admitted = admit_execution_plan(
        context=admit_context_plan(
            prepared=prepared, bound_context=None, adaptive_retrievals=(), context_capabilities=()
        ),
        capabilities=(capability,),
        policies=(policy,),
        decisions=(),
        assessment_productions=(),
        ingress_limits=IngressLimits(
            max_associations=1,
            max_outputs_per_association=1,
            max_collection_items=1,
            max_assessments=1,
            max_metadata_items=1,
            max_text_bytes=None,
            max_metadata_bytes=None,
            max_depth=1,
        ),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_assessment_facts=0,
            max_port_facts=0,
            max_provenance_edges=0,
        ),
    )

    async def execute(
        pool: BlockingPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> ReplacementResult:
        del pool
        from anonymizer.engine.detection._graph_profile import _await_profile_execution

        producer = _Transform(callback, column, handoff)
        handle = ImplementationHandle(
            implementation=capability.implementation,
            operation=operation,
            configuration=capability.configuration,
            local=producer,
            transport=None,
            resource=None,
        )
        result = await _await_profile_execution(
            await start_execution(
                admitted=admitted,
                capabilities=(capability,),
                services=ExecutionServices(
                    handles=(handle,),
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
                ),
            )
        )
        if producer.cancellation is not None:
            assert result.invocation_state is not None
            assert all(entry.status == "cancelled" for entry in result.invocation_state.entries)
            setattr(producer.cancellation, "_anonymizer_execution", result)
            raise producer.cancellation
        return project_detection_redact(result)

    return _run_sync_profile(execute, caller_callbacks=True).dataframe.at[0, column]
