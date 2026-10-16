# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Whole-frame compatibility consumers under nonqualifying graph execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from contextvars import copy_context
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Any, cast

import pandas as pd
from data_designer.config import ModelConfig

from anonymizer.config.models import DetectionModelSelection
from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.detection._frame_publication import FramePublication, FrameStages, InvocationFrame
from anonymizer.engine.detection._graph_invocation import DetectionRedactInvocation
from anonymizer.engine.detection.artifacts import ENTITY_SPANS_TYPE
from anonymizer.engine.detection.detection_workflow import _DETECTION_RUN, EntityDetectionWorkflow
from anonymizer.engine.graph_sdk._artifact_shape import content_bytes
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_delegated import WorkerDelegatedSubmitter
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.blocking import BlockingAffinity, BlockingPool
from anonymizer.engine.graph_sdk.capabilities import (
    FrozenConfig,
    ImplementationCapability,
    ImplementationRef,
    ImplementationSelection,
    config_size,
)
from anonymizer.engine.graph_sdk.context import admit_context_plan
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits, ValidatedDataGraph
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
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
    admit_execution_plan,
    start_execution,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
    PreparationConfiguration,
    PreparationLimits,
    PreparedPlan,
    prepare,
)
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, SemanticAssociation
from anonymizer.engine.graph_sdk.resources import ResourceLease
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    ArtifactType,
    Graph,
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
    compile_graph,
)
from anonymizer.interface.errors import AnonymizerWorkflowError


@dataclass(frozen=True)
class _DetectionInvocation:
    source: pd.DataFrame
    detection: EntityDetectionWorkflow
    transform: Callable[[pd.DataFrame], ReplacementResult | RewriteResult]
    model_configs: list[ModelConfig]
    selected_models: DetectionModelSelection
    detection_options: Mapping[str, Any]
    name: str = "opaque-detection-replace"
    semantic_slots: Mapping[str, int] | None = None
    semantic_loop: asyncio.AbstractEventLoop | None = None
    semantic_targets: tuple[Any, ...] = ()
    semantic_transform_slot: int | None = None
    semantic_configuration: FrozenConfig | None = None
    semantic_adapter: object | None = None
    semantic_lowerings: tuple[Any, Any] | None = None
    semantic_entry_slot: int | None = None
    semantic_optional: bool = False
    handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure] | None = None
    replacement_receiver: Any | None = None
    replacement_arguments: dict[str, Any] | None = None
    map_slots: Mapping[str, int] | None = None
    map_adapter: object | None = None
    map_lowering: object | None = None
    map_entry_method: object | None = None
    replacement_entry_method: object | None = None
    map_configuration: FrozenConfig | None = None
    map_is_submission: bool = False

    def run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        context = copy_context()
        return (
            self._run(inputs, requests)
            if self.handoff is None
            else self.handoff(lambda: context.run(self._run, inputs, requests))
        )

    def _run(self, inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        if requests.cancelled:
            return LocalFailure(failure="implementation_exception")
        started = perf_counter()
        arguments: dict[str, Any] = {
            "model_configs": self.model_configs,
            "selected_models": self.selected_models,
            **self.detection_options,
        }
        if self.semantic_slots is not None:
            if self.semantic_loop is None:
                raise AnonymizerWorkflowError("Detection binding has no owning loop")
            from anonymizer.engine.detection._semantic_detection import bind_detection, complete_detection_entry

            if not self.semantic_optional:
                asyncio.run_coroutine_threadsafe(
                    complete_detection_entry(requests, inputs), self.semantic_loop
                ).result()
            source = requests.bind_frame(
                self.source, targets=self.semantic_targets, source_limit=len(self.semantic_targets)
            )
            with bind_detection(
                self.detection,
                self.source,
                arguments,
                requests,
                source,
                dict(self.semantic_slots),
                self.semantic_loop,
                self.semantic_configuration,
                self.semantic_adapter,
                self.semantic_lowerings,
                inputs if self.semantic_optional and inputs else None,
                entry_slot=self.semantic_entry_slot,
            ) as binding:
                detected = self.detection.run(self.source, **arguments)
            if self.semantic_optional and not binding.claim.locked():
                asyncio.run_coroutine_threadsafe(
                    complete_detection_entry(requests, inputs, "bypass", self.semantic_entry_slot), self.semantic_loop
                ).result()
        else:
            if self.map_slots is not None:
                source = requests.bind_frame(
                    self.source, targets=self.semantic_targets, source_limit=len(self.semantic_targets)
                )
            detected = self.detection.run(self.source, **arguments)
        if requests.cancelled:
            return LocalFailure(failure="implementation_exception")
        detection_elapsed = perf_counter() - started
        started = perf_counter()
        map_binding = None
        if self.map_slots is not None:
            from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE
            from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
            from anonymizer.engine.replace._semantic_map import bind_map, map_configuration

            if self.replacement_receiver is None or self.replacement_arguments is None or self.semantic_loop is None:
                reject(EffectCode.MISSING)
            helper = self.replacement_receiver._llm_workflow
            options = self.replacement_arguments
            map_slots = self.map_slots
            loop = self.semantic_loop
            if (
                helper._adapter is not self.map_adapter
                or helper._generate_columns != self.map_lowering
                or helper.generate_map_only != self.map_entry_method
                or self.replacement_receiver.run != self.replacement_entry_method
            ):
                reject(EffectCode.FOREIGN_OWNER)
            map_arguments = dict(
                model_configs=options["model_configs"],
                selected_models=options["selected_models"],
                instructions=options["replace_method"].instructions,
                entities_column=COL_ENTITIES_BY_VALUE,
                preview_num_records=options["preview_num_records"],
            )
            if map_configuration(map_arguments) != self.map_configuration:
                reject(EffectCode.FOREIGN_OWNER)
            active = tuple(source.associations[position] for position in detected.source_positions)

            async def enter_map():
                return (
                    await requests.enter_operations(map_slots["entry"], active)
                    if active
                    else (
                        (await requests.enter_operation(map_slots["entry"], None),) if not source.associations else ()
                    )
                )

            entry_inputs = asyncio.run_coroutine_threadsafe(enter_map(), loop).result()
            with bind_map(
                helper,
                detected.dataframe,
                map_arguments,
                requests,
                WorkflowSourceFrame.capture(detected.dataframe, associations=active, invocation_bound=True),
                {name: slot for name, slot in self.map_slots.items() if name != "entry"},
                self.semantic_loop,
                apply=True,
                entry_inputs=entry_inputs,
            ) as map_binding:
                map_binding.replacement_receiver = self.replacement_receiver
                map_binding.replacement_arguments = options
                map_binding.replacement_configuration = options["replace_method"].model_dump_json()
                replaced = self.transform(detected.dataframe)
            if not map_binding.claim.locked():
                map_binding.bypass()
            else:
                map_binding.authenticate_result(
                    LlmReplaceResult(
                        replaced.dataframe,
                        replaced.failed_records,
                        tuple(position for position, _, _ in map_binding.terminal),
                    )
                )
        else:
            replaced = self.transform(detected.dataframe)
        if self.semantic_transform_slot is not None:
            if self.semantic_loop is None:
                raise AnonymizerWorkflowError("Detection binding has no owning loop")
            transform_slot = self.semantic_transform_slot

            async def finish() -> None:
                if not source.associations:
                    item = await requests.enter_operation(transform_slot, None)
                    if item is None:
                        raise AnonymizerWorkflowError("Empty detection transform is not ready")
                    await binding.complete_empty_stage([item])
                selected_positions = (
                    tuple(detected.source_positions[position] for position, _, _ in map_binding.terminal)
                    if map_binding is not None and map_binding.claim.locked()
                    else detected.source_positions
                )
                for position in selected_positions:
                    item = await requests.enter_operation(transform_slot, source.associations[position])
                    if item is None:
                        raise AnonymizerWorkflowError("Detection transform is not ready")
                    await requests.complete_operation(
                        cast(SemanticAssociation, item.association),
                        LocalCompleted(
                            results=(
                                AssociationResult(
                                    association=item.association,
                                    outcome="ok",
                                    outputs=(),
                                    consumed_context_ports=frozenset(),
                                ),
                            )
                        ),
                    )

            asyncio.run_coroutine_threadsafe(finish(), self.semantic_loop).result()
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association,
                    outcome="builtin" if self.map_is_submission else "ok",
                    outputs=(),
                    consumed_context_ports=frozenset(),
                )
                for item in inputs
                if item.association
                in {
                    source.associations[position]
                    for position in (
                        tuple(detected.source_positions[row] for row, _, _ in map_binding.terminal)
                        if map_binding is not None and map_binding.claim.locked()
                        else detected.source_positions
                    )
                }
            )
            if self.semantic_slots is not None or self.map_slots is not None
            else (),
            failed_sources=frozenset(
                set(source.associations)
                - {
                    source.associations[position]
                    for position in (
                        tuple(detected.source_positions[row] for row, _, _ in map_binding.terminal)
                        if map_binding is not None and map_binding.claim.locked()
                        else detected.source_positions
                    )
                }
            )
            if self.semantic_slots is not None or self.map_slots is not None
            else frozenset(),
            frame=FramePublication.capture(
                replaced.dataframe,
                associations=tuple(association for _, association, _ in map_binding.terminal),
                text_ports=(
                    (
                        COL_REPLACED_TEXT,
                        "protected",
                    ),
                ),
                json_ports=(
                    (
                        COL_REPLACEMENT_MAP,
                        "map",
                    ),
                ),
                failed_records=tuple(detected.failed_records) + tuple(replaced.failed_records),
                opaque_source=self.source,
                stages=FrameStages(
                    detected.dataframe,
                    tuple(detected.failed_records),
                    tuple(replaced.failed_records),
                    detection_elapsed,
                    perf_counter() - started,
                ),
            )
            if map_binding is not None and map_binding.claim.locked() and source.associations
            else InvocationFrame(
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
    handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure]

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

        return self.handoff(evaluate)


def builtin_rewrite_runner(runner: RewriteWorkflow, *, combined: bool, supplied: bool) -> bool:
    from anonymizer.engine.rewrite.rewrite_workflow import _REWRITE_RUN

    del combined, supplied
    method = _REWRITE_RUN
    return (
        isinstance(runner, RewriteWorkflow)
        and getattr(runner.run, "__func__", None) is method
        and callable(getattr(runner._adapter, "run_workflow", None))
    )


def run_detection_rewrite(
    source: pd.DataFrame,
    *,
    detection: EntityDetectionWorkflow,
    model_configs: list[ModelConfig],
    selected_models: DetectionModelSelection,
    detection_options: Mapping[str, Any],
    rewrite: Callable[[pd.DataFrame], RewriteResult],
    semantic_arguments: dict[str, Any] | None = None,
    semantic_runner: RewriteWorkflow | None = None,
    combined: bool = False,
) -> ReplacementResult:
    """Admit native Rewrite independently of opaque detection callbacks."""
    from anonymizer.engine.detection._graph_profile import _run_sync_profile, project_detection_redact

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
            detection_arguments={
                "model_configs": model_configs,
                "selected_models": selected_models,
                **detection_options,
            },
        )

    async def operation(
        pool: BlockingPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
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
                handoff=handoff,
            ),
            pool,
        )
        return project_detection_redact(result)

    return _run_sync_profile(operation, caller_callbacks=True)


def run_evaluation(
    source: pd.DataFrame,
    evaluate: Callable[[DelegatedRequests], ReplacementResult | LocalFailure],
    *,
    retained_owner: object | None,
    name: str = "opaque-evaluation",
) -> ReplacementResult:
    """Evaluate a retained public frame under fresh, nonqualifying graph authority."""
    from anonymizer.engine.detection._graph_profile import _run_sync_profile, project_detection_redact

    async def operation(
        pool: BlockingPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> ReplacementResult:
        result = await execute_injected_redact(
            _EvaluationInvocation(source, evaluate, retained_owner, handoff, name), pool
        )
        return project_detection_redact(result)

    return _run_sync_profile(operation, caller_callbacks=True)


async def execute_injected_redact(
    callback: _DetectionInvocation | DetectionRedactInvocation | _EvaluationInvocation, pool: BlockingPool
) -> ExecutionResult:

    expected_adapter = (
        None if isinstance(callback, _EvaluationInvocation) else getattr(callback.detection, "_adapter", None)
    )
    expected_lowerings = (
        None
        if isinstance(callback, _EvaluationInvocation)
        or not issubclass(type(callback.detection), EntityDetectionWorkflow)
        else (callback.detection.detect_and_validate_entities, callback.detection.identify_latent_entities)
    )
    authenticated = isinstance(callback, DetectionRedactInvocation)
    bindable_detection = not isinstance(callback, _EvaluationInvocation) and issubclass(
        type(callback.detection), EntityDetectionWorkflow
    )
    builtin_detection = (
        not isinstance(callback, _EvaluationInvocation)
        and getattr(callback.detection.run, "__func__", None) is _DETECTION_RUN
    )
    from anonymizer.config.replace_strategies import Substitute
    from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow

    bindable_map = (
        isinstance(callback, _DetectionInvocation)
        and callback.replacement_arguments is not None
        and isinstance(callback.replacement_arguments["replace_method"], Substitute)
        and isinstance(getattr(callback.replacement_receiver, "_llm_workflow", None), LlmReplaceWorkflow)
    )
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
        if authenticated or bindable_detection or bindable_map:
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
        if authenticated and size
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
        name="dd-detection-replace" if isinstance(callback, DetectionRedactInvocation) else callback.name,
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
    raw = compile_graph(
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
    capability, policy = _frame_implementation(
        node,
        operation,
        callback.redact_configuration if isinstance(callback, DetectionRedactInvocation) else FrozenConfig(fields=()),
        aggregate=not authenticated,
    )
    detection_ids = {}
    detection_config = None
    child_capabilities, child_policies = (), ()
    if (bindable_detection or authenticated) and not isinstance(callback, _EvaluationInvocation):
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
            optional=not builtin_detection,
            source_free=not size,
        )
    parent_capabilities, parent_policies = (capability,), (policy,)
    if detection_ids:
        parent_capabilities, parent_policies = reparent_detection_policies(
            parent_capabilities, parent_policies, detection_ids["entry"]
        )
    map_ids = {}
    native_ids = {}
    if isinstance(callback, _DetectionInvocation) and callback.replacement_receiver is not None:
        from anonymizer.config.replace_strategies import Substitute
        from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE
        from anonymizer.engine.replace._semantic_map import extend_map_graph, map_configuration
        from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow

        helper = getattr(callback.replacement_receiver, "_llm_workflow", None)
        options = callback.replacement_arguments
        if options is None:
            reject(EffectCode.MISSING)
        if isinstance(options["replace_method"], Substitute) and isinstance(helper, LlmReplaceWorkflow):
            args = dict(
                model_configs=options["model_configs"],
                selected_models=options["selected_models"],
                instructions=options["replace_method"].instructions,
                entities_column=COL_ENTITIES_BY_VALUE,
                preview_num_records=options["preview_num_records"],
            )
            config = map_configuration(args)
            raw, map_caps, map_policies, map_ids = extend_map_graph(
                raw,
                node,
                config,
                submission_owner=detection_ids["entry"] if detection_ids else node,
                predecessor=detection_ids["final"] if detection_ids else None,
                source_free=not size,
                apply=True,
                optional=True,
            )
            child_capabilities = (*child_capabilities, *map_caps)
            child_policies = (*child_policies, *map_policies)
            child_policies = tuple(
                replace(
                    policy,
                    delegated=replace(
                        cast(DelegatedRequestContract, policy.delegated),
                        publication_sources=tuple(
                            (map_ids[name], frozenset({"map", "protected"})) for name in ("accept", "empty")
                        ),
                    ),
                )
                if policy.node == detection_ids.get("entry", node)
                else policy
                for policy in child_policies
            )
            if not detection_ids:
                from anonymizer.engine.detection._semantic_detection import reparent_detection_policies

                parent_capabilities, parent_policies = reparent_detection_policies(
                    parent_capabilities, parent_policies, map_ids["entry"]
                )
                entry_operation = next(node.operation for node in raw.operations if node.id == map_ids["entry"])
                entry_operation = replace(
                    entry_operation,
                    outcomes=tuple(
                        replace(outcome, ceiling=replace(outcome.ceiling, max_model_requests=None))
                        for outcome in entry_operation.outcomes
                    ),
                )
                raw = compile_graph(
                    workflow=raw.workflow,
                    interface=raw.interface,
                    nodes=tuple(
                        replace(item, operation=entry_operation)
                        if isinstance(item, OperationNode) and item.id == map_ids["entry"]
                        else item
                        for item in raw.nodes
                    ),
                    input_bindings=tuple(raw.input_bindings),
                    output_bindings=tuple(raw.output_bindings),
                    outcome_bindings=tuple(raw.outcome_bindings),
                    sequence=tuple(raw.sequence),
                    protection=tuple(raw.protection_requirements),
                    limits=raw.limits,
                    dynamic_limits=raw.dynamic_limits,
                )
                entry_cap, entry_policy = _frame_implementation(map_ids["entry"], entry_operation, config)
                entry_policy = replace(
                    entry_policy,
                    delegated=replace(
                        cast(DelegatedRequestContract, entry_policy.delegated),
                        publication_sources=tuple(
                            (map_ids[name], frozenset({"map", "protected"})) for name in ("accept", "empty")
                        ),
                    ),
                )
                child_capabilities = tuple(
                    entry_cap if policy.node == map_ids["entry"] else cap
                    for cap, policy in zip(child_capabilities, child_policies, strict=True)
                )
                child_policies = tuple(
                    entry_policy
                    if policy.node == map_ids["entry"]
                    else replace(policy, submission_owner=map_ids["entry"])
                    for policy in child_policies
                )
                callback = replace(callback, map_is_submission=True)
            from anonymizer.engine.replace._native_map_requests import extend_native_map_requests, native_map_capacity

            capacity = native_map_capacity(helper)
            if size and capacity is not None:
                raw, physical_caps, physical_policies, native_ids = extend_native_map_requests(raw, capacity)
                child_capabilities = (*child_capabilities, *physical_caps)
                child_policies = (*child_policies, *physical_policies)
            callback = replace(
                callback,
                map_adapter=helper._adapter,
                map_lowering=helper._generate_columns,
                map_entry_method=helper.generate_map_only,
                replacement_entry_method=cast(Any, callback.replacement_receiver).run,
                map_configuration=config,
            )
    prepared = _prepare_frame_plan(
        data,
        raw,
        (*parent_capabilities, *child_capabilities),
        tuple(BoundInput(target=target, source=target, port="source", artifact_type=text_type) for target in targets),
        invocation_owned=not targets,
    )
    if detection_ids:
        from anonymizer.engine.detection._semantic_detection import detection_slots

        if isinstance(callback, DetectionRedactInvocation):
            callback = replace(
                callback,
                semantic_slots=detection_slots(prepared, detection_ids),
                semantic_loop=asyncio.get_running_loop(),
                semantic_configuration=detection_config,
                semantic_adapter=expected_adapter,
                semantic_lowerings=expected_lowerings,
                semantic_optional=not builtin_detection,
                semantic_entry_slot=next(
                    slot.index for slot in prepared.reservation_recipe if slot.template == detection_ids["entry"]
                )
                if not size
                else None,
                semantic_transform_slot=next(
                    slot.index for slot in prepared.reservation_recipe if slot.template == node
                ),
            )
        else:
            callback = replace(
                callback,
                semantic_slots=detection_slots(prepared, detection_ids),
                semantic_loop=asyncio.get_running_loop(),
                semantic_configuration=detection_config,
                semantic_adapter=expected_adapter,
                semantic_lowerings=expected_lowerings,
                semantic_entry_slot=next(
                    slot.index for slot in prepared.reservation_recipe if slot.template == detection_ids["entry"]
                )
                if not size
                else None,
                semantic_optional=not builtin_detection,
                semantic_targets=tuple(targets),
                semantic_transform_slot=next(
                    slot.index for slot in prepared.reservation_recipe if slot.template == node
                ),
            )
    if map_ids:
        callback = replace(
            callback,
            semantic_loop=asyncio.get_running_loop(),
            semantic_targets=tuple(targets),
            semantic_transform_slot=next(slot.index for slot in prepared.reservation_recipe if slot.template == node),
            map_slots={
                name: next(slot.index for slot in prepared.reservation_recipe if slot.template == node)
                for name, node in map_ids.items()
            },
        )
    return await _execute_frame_plan(
        prepared=prepared,
        capabilities=(*parent_capabilities, *child_capabilities),
        callback=callback.run,
        pool=pool,
        borrowed_loop=asyncio.get_running_loop(),
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=None,
            max_runtime_artifacts=max(1, (24 if map_ids else 11) * size),
            max_runtime_artifact_bytes=None,
            max_collection_items=characters,
        ),
        policies=(*parent_policies, *child_policies),
        ingress_limits=IngressLimits(
            max_associations=max(1, size),
            max_outputs_per_association=4 if map_ids else 2,
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
            max_assessment_facts=0,
            max_port_facts=(48 if map_ids else 20) * size,
            max_provenance_edges=(100 if map_ids else 20) * size,
        ),
        native_request_nodes=native_ids,
    )


def _prepare_frame_plan(
    data: ValidatedDataGraph,
    raw: Graph,
    capabilities: tuple[ImplementationCapability, ...],
    bound_inputs: tuple[BoundInput, ...],
    *,
    invocation_owned: bool = False,
) -> PreparedPlan:
    """Shared fixed-operation preparation for existing whole-frame consumers."""
    bodies = raw.bodies()
    nodes = tuple(node for body in bodies for node in body.operations)
    count = raw.activation_upper_bound
    workflow = raw
    sizes = [config_size(cap.configuration) for cap in capabilities]
    return prepare(
        data=data,
        workflow=workflow,
        activation_limits=ActivationLimits(max_events=4 * count, max_entries=count, max_parent_depth=len(bodies) + 1),
        bound_inputs=bound_inputs,
        configuration=PreparationConfiguration(
            purpose="execution_only",
            required_protection_outcomes=frozenset(),
            hard_request_limit=None,
            invocation_owned=invocation_owned,
        ),
        selections=tuple(
            ImplementationSelection(
                node=next(
                    node.id for node in nodes if isinstance(node, OperationNode) and node.operation == cap.operation
                ),
                implementation=cap.implementation,
                configuration=cap.configuration,
            )
            for cap in capabilities
        ),
        capabilities=capabilities,
        limits=PreparationLimits(
            max_bound_inputs=len(bound_inputs),
            max_selections=len(capabilities),
            max_config_depth=max(size[0] for size in sizes),
            max_config_atoms=sum(size[1] for size in sizes) * 2,
            max_total_activation_slots=count * max(1, len(data.targets)),
            max_capabilities=len(capabilities),
        ),
    )


async def _execute_frame_plan(
    *,
    prepared: PreparedPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
    callback: Callable[[tuple[AssociationInput, ...], DelegatedRequests], LocalCompleted | LocalFailure],
    pool: BlockingPool,
    borrowed_loop: asyncio.AbstractEventLoop,
    limits: ExecutionLimits,
    ingress_limits: IngressLimits,
    assessment_limits: AssessmentLimits,
    assessment_productions: tuple[EvidenceProductionDecl, ...] = (),
    native_request_nodes: dict[str, NodeId] | None = None,
) -> ExecutionResult:
    """Run a single grouped submission with its admitted observed operations."""

    physical = None
    transport = None
    if native_request_nodes:
        from data_designer.engine.dataset_builders.utils.async_concurrency import ensure_async_engine_loop

        from anonymizer.engine.graph_sdk._foreign_transport import ForeignLoopTransport
        from anonymizer.engine.ndd._graph_requests import NddPhysicalTransport
        from anonymizer.engine.replace._native_map_requests import bind_native_map_policy

        policies = bind_native_map_policy(prepared, policies, native_request_nodes)

        async def create_resource():
            value = NddPhysicalTransport()
            try:
                return value, ForeignLoopTransport.create(
                    factory=lambda: value, owner="sdk", gate_timeout_ns=1_000_000_000, exclusive=True
                )
            except BaseException:
                await value.close()
                raise

        acquisition = asyncio.wrap_future(
            asyncio.run_coroutine_threadsafe(create_resource(), ensure_async_engine_loop())
        )
        interrupted = None
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
            await transport.close()
            raise interrupted
    try:
        return await _execute_frame_plan_owned(
            prepared=prepared,
            capabilities=capabilities,
            policies=policies,
            callback=callback,
            pool=pool,
            borrowed_loop=borrowed_loop,
            limits=limits,
            ingress_limits=ingress_limits,
            assessment_limits=assessment_limits,
            assessment_productions=assessment_productions,
            native_request_nodes=native_request_nodes,
            physical=physical,
            transport=transport,
        )
    finally:
        if transport is not None and not transport._claimed:
            await transport.close()


async def _execute_frame_plan_owned(
    *,
    prepared: PreparedPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
    callback: Callable[..., Any],
    pool: BlockingPool,
    borrowed_loop: asyncio.AbstractEventLoop,
    limits: ExecutionLimits,
    ingress_limits: IngressLimits,
    assessment_limits: AssessmentLimits,
    assessment_productions: tuple[EvidenceProductionDecl, ...],
    native_request_nodes: dict[str, NodeId] | None,
    physical: Any,
    transport: Any,
) -> ExecutionResult:
    from anonymizer.engine.detection._graph_profile import _await_profile_execution, _Clock, _ProfileLocal

    execution = admit_execution_plan(
        context=admit_context_plan(prepared=prepared, bound_context=None),
        capabilities=capabilities,
        policies=policies,
        assessment_productions=assessment_productions,
        ingress_limits=ingress_limits,
        assessment_limits=assessment_limits,
    )

    def dispatch(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests):
        if not native_request_nodes:
            return callback(inputs, requests)
        from anonymizer.engine.ndd._graph_requests import NddRequestBridge
        from anonymizer.engine.replace._native_map_requests import _BRIDGE

        leaves = tuple(
            slot.index
            for slot in prepared.reservation_recipe
            if slot.template in {node for name, node in native_request_nodes.items() if name.startswith("leaf-")}
        )
        token = _BRIDGE.set(NddRequestBridge(requests, {(COL_REPLACEMENT_MAP, 0): leaves}, physical))
        try:
            return callback(inputs, requests)
        finally:
            _BRIDGE.reset(token)

    worker = WorkerDelegatedSubmitter(
        callback=dispatch,
        pool=pool,
        affinity=BlockingAffinity(cross_thread=True, concurrency_safe=False, close_on_event_loop=False),
        borrowed_loop=borrowed_loop,
    )
    handles = tuple(
        ImplementationHandle(
            implementation=cap.implementation,
            operation=cap.operation,
            configuration=cap.configuration,
            local=_ProfileLocal("async_required")
            if native_request_nodes and policy.node == native_request_nodes["probe"]
            else None,
            transport=transport
            if native_request_nodes
            and policy.node
            in {node for name, node in native_request_nodes.items() if name not in {"readiness", "probe"}}
            else None,
            delegated=worker if policy.delegated is not None else None,
            resource=ResourceLease.create(owner="caller", safe_detachment="forbidden", handle=worker)
            if policy.delegated is not None
            else transport.lease
            if native_request_nodes
            and policy.node
            in {node for name, node in native_request_nodes.items() if name not in {"readiness", "probe"}}
            else None,
        )
        for policy, cap in zip(policies, capabilities, strict=True)
        if not native_request_nodes or policy.node != native_request_nodes["readiness"]
    )
    return await _await_profile_execution(
        await start_execution(
            admitted=execution,
            capabilities=capabilities,
            services=ExecutionServices(
                handles=handles,
                limits=limits,
                clock=_Clock(),
            ),
        )
    )


def _frame_implementation(
    node: NodeId,
    operation: OperationSpec,
    configuration: FrozenConfig,
    *,
    submission_owner: NodeId | None = None,
    local: bool = False,
    aggregate: bool = False,
) -> tuple[ImplementationCapability, OperationExecutionPolicy]:
    from anonymizer.engine.detection._graph_profile import _runtime_rows

    capability = ImplementationCapability(
        implementation=ImplementationRef(name=operation.name, revision=1),
        operation=operation,
        configuration=configuration,
        effect="local" if local else "external",
        attribution="aggregate_only" if aggregate else "per_task",
        request_visibility="none",
        pre_dispatch_control="none",
        retry_owner="none" if local else "implementation",
        error_reporting="typed_terminal",
        cancellation="cooperative_ack" if local else "unobservable",
        settlement="synchronous" if local else "uncertain_possible",
        usage="exact" if local else "unknown",
        resource_lifetime="caller_owned" if submission_owner is None else "stateless",
        max_physical_requests_per_activation=0 if local else None,
    )
    policy = OperationExecutionPolicy(
        node=node,
        kind="local" if local else "delegated",
        request=None,
        safe_detachment="forbidden",
        implementations=(
            ExecutionImplementation(
                implementation=capability.implementation,
                configuration=configuration,
                capability=capability,
                request=None,
            ),
        ),
        result_outcomes=frozenset(outcome.name for outcome in operation.outcomes),
        runtime_outcomes=_runtime_rows(tuple(outcome.name for outcome in operation.outcomes), not local),
        delegated=DelegatedRequestContract(
            mode="opaque", child_slots=(), routes=(), followups=frozenset(), usage_requirement="unknown_allowed"
        )
        if submission_owner is None
        else None,
        submission_owner=submission_owner,
    )
    return capability, policy
