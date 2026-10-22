# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-frame compatibility consumers under nonqualifying graph execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, cast

import pandas as pd
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import COL_TEXT
from anonymizer.engine.detection.detection_workflow import EntityDetectionWorkflow
from anonymizer.engine.detection.run_arguments import detection_run_arguments
from anonymizer.engine.graph_sdk._artifact_shape import content_bytes
from anonymizer.engine.graph_sdk._authoring_defaults import (
    _compile_graph,
    _execution_limits,
    _ingress_limits,
    _operation_spec,
    _outcome_spec,
    _validate_targets,
)
from anonymizer.engine.graph_sdk._execution_delegated import SubmitterPool, drive
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk._workflow_values import (
    ArtifactType,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    OperationNode,
    OutcomeBinding,
    ResourceCeiling,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
)
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
)
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    ExecutionResult,
    LocalCompleted,
    LocalFailure,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
)
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, SemanticAssociation
from anonymizer.engine.harness import kernel
from anonymizer.engine.harness.frames import InvocationFrame
from anonymizer.engine.ndd._stage_binding import completed
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.rewrite.rewrite_workflow import RewriteArguments, RewriteResult, RewriteWorkflow
from anonymizer.engine.source_text import source_text
from anonymizer.interface.errors import AnonymizerWorkflowError


@dataclass(frozen=True)
class _DetectionInvocation:
    source: pd.DataFrame
    detection: EntityDetectionWorkflow
    transform: Callable[[pd.DataFrame], RewriteResult]
    model_configs: list[ModelConfig]
    selected_models: DetectionModelSelection
    detection_options: Mapping[str, Any]
    name: str = "opaque-detection-replace"
    semantic_slots: Mapping[str, int] | None = None
    semantic_targets: tuple[Any, ...] = ()
    semantic_transform_slot: int | None = None
    semantic_entry_slot: int | None = None

    def run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        if requests.cancelled:
            return LocalFailure(failure="implementation_exception")
        started = perf_counter()
        arguments: dict[str, Any] = {
            "model_configs": self.model_configs,
            "selected_models": self.selected_models,
            **self.detection_options,
        }
        # Slots exist only for builtin runners, which the graph lowers directly; caller runners run opaque.
        if self.semantic_slots is not None:
            from anonymizer.engine.detection._semantic_detection import complete_detection_entry, lower_detection

            drive(complete_detection_entry(requests, inputs))
            source = requests.bind_frame(
                self.source, targets=self.semantic_targets, source_limit=len(self.semantic_targets)
            )
            detected = lower_detection(
                self.detection,
                self.source,
                arguments,
                requests,
                source,
                dict(self.semantic_slots),
                self.semantic_entry_slot,
            )
        else:
            detected = self.detection.run(self.source, **detection_run_arguments(self.detection.run, arguments))
        if requests.cancelled:
            return LocalFailure(failure="implementation_exception")
        detection_elapsed = perf_counter() - started
        started = perf_counter()
        replaced = self.transform(detected.dataframe)
        selected_positions = detected.source_positions
        if self.semantic_transform_slot is not None:
            transform_slot = self.semantic_transform_slot

            async def finish() -> None:
                from anonymizer.engine.detection._semantic_detection import complete_empty_stage

                if not source.associations:
                    item = await requests.enter_operation(transform_slot, None)
                    if item is None:
                        raise AnonymizerWorkflowError("Empty detection transform is not ready")
                    await complete_empty_stage(requests, [item])
                for position in selected_positions:
                    item = await requests.enter_operation(transform_slot, source.associations[position])
                    if item is None:
                        raise AnonymizerWorkflowError("Detection transform is not ready")
                    await requests.complete_operation(
                        cast(SemanticAssociation, item.association),
                        completed(item.association),
                    )

            drive(finish())
        succeeded = (
            {source.associations[position] for position in selected_positions}
            if self.semantic_slots is not None
            else set()
        )
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                )
                for item in inputs
                if item.association in succeeded
            ),
            failed_sources=frozenset(set(source.associations) - succeeded)
            if self.semantic_slots is not None
            else frozenset(),
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
    """One compatibility callback with an independently retained prior execution owner."""

    source: pd.DataFrame
    evaluate: Callable[[DelegatedRequests], ReplacementResult | LocalFailure]
    retained_owner: object | None
    name: str = "opaque-evaluation"

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

        return evaluate()


def run_detection_rewrite(
    source: pd.DataFrame,
    *,
    detection: EntityDetectionWorkflow,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    detection_options: Mapping[str, Any],
    rewrite: Callable[[pd.DataFrame], RewriteResult],
    semantic_arguments: RewriteArguments | None = None,
    semantic_runner: RewriteWorkflow | None = None,
    combined: bool = False,
) -> ReplacementResult:
    """Admit native Rewrite independently of opaque detection callbacks."""
    from anonymizer.engine.harness.kernel import _run_sync_profile, project_detection_redact

    selected = (
        len(source)
        if detection_options.get("preview_num_records") is None
        else min(len(source), detection_options["preview_num_records"])
    )
    if semantic_arguments is not None and selected:
        from anonymizer.engine.rewrite._semantic_profile import run_semantic_rewrite
        from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow

        return run_semantic_rewrite(
            source,
            detection=detection,
            adapter=semantic_runner._adapter if semantic_runner is not None else detection._adapter,
            runner=semantic_runner,
            combined=isinstance(semantic_runner, CombinedRewriteWorkflow) if semantic_runner is not None else combined,
            arguments=semantic_arguments,
            detection_arguments=detection_run_arguments(
                detection.run,
                {"model_configs": model_configs, "selected_models": selected_models, **detection_options},
            ),
        )

    async def operation(
        pool: SubmitterPool,
    ) -> ReplacementResult:
        result = await execute_injected_redact(
            _DetectionInvocation(
                source,
                detection,
                rewrite,
                model_configs,
                selected_models,
                detection_options,
                name="opaque-detection-rewrite",
            ),
            pool,
        )
        return project_detection_redact(result)

    return _run_sync_profile(operation)


def run_evaluation(
    source: pd.DataFrame,
    evaluate: Callable[[DelegatedRequests], ReplacementResult | LocalFailure],
    *,
    retained_owner: object | None,
    name: str = "opaque-evaluation",
) -> ReplacementResult:
    """Evaluate a retained public frame under fresh, nonqualifying graph authority."""
    from anonymizer.engine.harness.kernel import _run_sync_profile, project_detection_redact

    async def operation(
        pool: SubmitterPool,
    ) -> ReplacementResult:
        result = await execute_injected_redact(_EvaluationInvocation(source, evaluate, retained_owner, name), pool)
        return project_detection_redact(result)

    return _run_sync_profile(operation)


async def execute_injected_redact(
    callback: _DetectionInvocation | _EvaluationInvocation, pool: SubmitterPool
) -> ExecutionResult:
    from anonymizer.engine.ndd._route import builtin_detection

    # Builtin runners are lowered by this graph; a caller's runner runs opaque, as in the release.
    semantic_detection = not isinstance(callback, _EvaluationInvocation) and builtin_detection(callback.detection)
    preview = (
        None if isinstance(callback, _EvaluationInvocation) else callback.detection_options.get("preview_num_records")
    )
    source = callback.source if preview is None else callback.source.iloc[:preview]
    # Evaluation owns a whole public frame; it makes no semantic source-association claim.
    texts = () if isinstance(callback, _EvaluationInvocation) else tuple(source_text(text) for text in source[COL_TEXT])
    graph = DataGraph.new()
    targets = []
    for text in texts:
        graph, target = graph.add_text(text)
        if semantic_detection:
            targets.append(target)
    size = len(targets)
    characters = max((len(text) for text in texts), default=0)
    total_characters = sum(len(text) for text in texts)
    text_type = ArtifactType(name="anonymizer.text", revision=1)
    data = _validate_targets(
        graph,
        targets=tuple(targets),
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
    operation = _operation_spec(
        name=callback.name,
        inputs=(InputPort(name="source", artifact_type=text_type),) if size else (),
        outputs=(),
        output_dependencies=(),
        outcomes=(
            _outcome_spec(
                name="ok",
                category="success",
                produced_ports=frozenset(),
                ceiling=ResourceCeiling(
                    max_activations=1, max_model_requests=None, max_input_bytes=None, max_output_bytes=None
                ),
            ),
        ),
    )
    raw = _compile_graph(
        workflow=owner,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
        input_bindings=(
            InputBinding(source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=node, port="source")),
        )
        if size
        else (),
        output_bindings=(),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=node, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
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
    capability, policy = kernel._frame_implementation(node, operation, FrozenConfig(fields=()), aggregate=True)
    detection_ids = {}
    detection_config = None
    child_capabilities, child_policies = (), ()
    if semantic_detection and not isinstance(callback, _EvaluationInvocation):
        from anonymizer.engine.detection._semantic_detection import (
            detection_configuration,
            extend_detection_graph,
            reparent_detection_policies,
        )

        detection_config = detection_configuration(
            {
                "model_configs": callback.model_configs,
                "selected_models": callback.selected_models,
                **callback.detection_options,
            }
        )
        raw, child_capabilities, child_policies, detection_ids = extend_detection_graph(
            raw,
            node,
            detection_config,
            latent=callback.detection_options.get("tag_latent_entities", True),
            source_free=not size,
        )
    parent_capabilities, parent_policies = (capability,), (policy,)
    if detection_ids:
        parent_capabilities, parent_policies = reparent_detection_policies(
            parent_capabilities, parent_policies, detection_ids["entry"]
        )
    prepared = kernel._prepare_frame_plan(
        data,
        raw,
        (*parent_capabilities, *child_capabilities),
        tuple(BoundInput(target=target, source=target, port="source", artifact_type=text_type) for target in targets),
        invocation_owned=not targets,
    )
    if detection_ids:
        from anonymizer.engine.detection._semantic_detection import detection_slots

        callback = replace(
            callback,
            semantic_slots=detection_slots(prepared, detection_ids),
            semantic_entry_slot=next(
                slot.index for slot in prepared.reservation_recipe if slot.template == detection_ids["entry"]
            )
            if not size
            else None,
            semantic_transform_slot=next(slot.index for slot in prepared.reservation_recipe if slot.template == node),
            semantic_targets=tuple(targets),
        )
    return await kernel._execute_frame_plan(
        prepared=prepared,
        capabilities=(*parent_capabilities, *child_capabilities),
        callback=callback.run,
        pool=pool,
        borrowed_loop=asyncio.get_running_loop(),
        limits=_execution_limits(
            max_local_in_flight=1,
            max_runtime_artifacts=max(1, 11 * size),
            max_collection_items=characters,
        ),
        policies=(*parent_policies, *child_policies),
        ingress_limits=_ingress_limits(
            max_associations=max(1, size),
            max_outputs_per_association=2,
            # Ingress counts spans across the batch; publication checks each artifact.
            max_collection_items=max(1, total_characters),
            max_assessments=1,
            max_metadata_items=max(7, 7 * size + 3 * total_characters),
            max_depth=10,
        ),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_assessment_facts=0,
            max_port_facts=20 * size,
            max_provenance_edges=20 * size,
        ),
    )
