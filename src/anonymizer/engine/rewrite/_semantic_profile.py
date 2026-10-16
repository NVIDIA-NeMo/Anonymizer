# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One source-bound invocation for detection and the shared Rewrite repair tail."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextvars import copy_context
from copy import deepcopy
from dataclasses import replace
from time import perf_counter
from typing import Any, cast

import pandas as pd
from pydantic import TypeAdapter

from anonymizer.config.rewrite import EvaluationCriteria
from anonymizer.engine.constants import COL_REWRITTEN_TEXT, COL_REWRITTEN_TEXT_INITIAL, COL_TEXT
from anonymizer.engine.detection._frame_publication import FramePublication, FrameStages
from anonymizer.engine.detection._opaque_redact import _execute_frame_plan, _frame_implementation, _prepare_frame_plan
from anonymizer.engine.detection._semantic_detection import detection_configuration
from anonymizer.engine.detection.detection_workflow import (
    _DETECTION_RUN,
    EntityDetectionResult,
    EntityDetectionWorkflow,
)
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.blocking import BlockingPool
from anonymizer.engine.graph_sdk.capabilities import (
    ConfigField,
    ConfigText,
    FrozenConfig,
)
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    QualificationLimits,
    admit_qualification,
    evidence_revision_view,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    EvidenceProductionDecl,
    ExecutionLimits,
    ExecutionResult,
    IngressLimits,
    LocalCompleted,
    LocalFailure,
)
from anonymizer.engine.graph_sdk.preparation import (
    BoundInput,
)
from anonymizer.engine.graph_sdk.qualification import qualify_retained
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.rewrite._semantic_evaluation import (
    EVALUATION_FAIL,
    EVALUATION_MEANING,
    EVALUATION_PASS,
    EVALUATION_PROMISE,
    RewriteEvaluationBinding,
)
from anonymizer.engine.rewrite.artifacts import (
    REWRITE_TYPES,
    EvaluationMetricsValue,
    PrivacyAnswersValue,
    QualityAnswersValue,
    QualityComparisonValue,
    RewriteCandidateValue,
    RewriteControlValue,
    RewriteQAValue,
)
from anonymizer.engine.rewrite.combined_rewrite_workflow import CombinedRewriteWorkflow, _evaluation_state
from anonymizer.engine.rewrite.rewrite_workflow import RewriteResult, RewriteWorkflow
from anonymizer.engine.row_partitioning import resolve_source_positions
from anonymizer.graph.workflow import (
    ArtifactType,
    ChoiceBranch,
    ChoiceDecl,
    DynamicLimits,
    EvidencePromise,
    InputBinding,
    InputPort,
    KeyedJoinDecl,
    LoopCarriedBinding,
    LoopDecl,
    LoopInitialBinding,
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
    SequenceEdge,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    compile_graph,
)


def _configuration(arguments: dict[str, Any], detection_arguments: dict[str, Any] | None) -> FrozenConfig:
    fields = [ConfigField(name="evaluation", value=ConfigText(value=arguments["evaluation"].model_dump_json()))]
    if detection_arguments is not None:
        fields.append(
            ConfigField(
                name="environment",
                value=ConfigText(
                    value=TypeAdapter(dict[str, Any])
                    .dump_json({"rewrite": arguments, "detection": detection_arguments})
                    .decode()
                ),
            )
        )
    return FrozenConfig(fields=tuple(sorted(fields, key=lambda field: field.name)))


def run_builtin_rewrite(runner: RewriteWorkflow, source: pd.DataFrame, arguments: dict[str, Any]) -> RewriteResult:
    """Execute the caller's actual builtin runner through the shared graph control."""
    from anonymizer.engine.detection._graph_profile import _run_sync_profile, project_detection_redact

    captured: list[RewriteResult] = []
    count = max(1, len(source))
    allowance = arguments["evaluation"].max_repair_iterations

    opaque = getattr(runner._adapter.run_workflow, "__func__", None) is not NddAdapter.run_workflow

    async def operation(
        pool: BlockingPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> ReplacementResult:
        result = await execute_initial_evaluation(
            source,
            adapter=runner._adapter,
            combined=isinstance(runner, CombinedRewriteWorkflow),
            arguments=arguments,
            configuration=_configuration(arguments, None),
            pool=pool,
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=None,
                max_runtime_artifacts=count * (12 + 6 * allowance),
                max_runtime_artifact_bytes=None,
                max_collection_items=None,
            ),
            runner=runner,
            opaque=opaque,
            handoff=handoff if opaque else None,
            on_result=captured.append,
        )
        projected = project_detection_redact(result)
        return projected

    projected = _run_sync_profile(operation, caller_callbacks=opaque)
    result = RewriteResult(projected.dataframe, projected.failed_records, captured[0].source_positions)
    object.__setattr__(result, "_execution", projected._execution)
    return result


class _ObservedCombinedRunner(RewriteWorkflow):
    """Adapt already generated column outcomes; it issues no physical work."""

    def run(self, dataframe: pd.DataFrame, **arguments: Any) -> RewriteResult:
        binding = self._semantic_binding
        if binding is None:
            reject(EffectCode.FOREIGN_OWNER)
        binding.claim_lowering(self, dataframe, arguments)

        def require_observation(selected: RewriteEvaluationBinding, row: dict[str, Any]) -> None:
            columns = tuple(value for value in selected.state.__dict__.values() if isinstance(value, str))
            if any(column not in row for column in columns):
                reject(EffectCode.MISSING)
            if type(row[selected.state.needs_repair]) is not bool:
                reject(EffectCode.INVALID_TYPE)

        for row in dataframe.to_dict("records"):
            require_observation(binding, row)

        async def observe() -> None:
            for association, row in zip(binding.source.associations, dataframe.to_dict("records"), strict=True):
                await binding.after(binding.state.rewritten_text, association, row, prefix=True)
                await binding.after(binding.state.needs_repair, association, row)
                for iteration, slots in binding.iteration_slots.items():
                    ready = await binding.requests.ready_operations(association)
                    if slots["repair"] not in ready:
                        continue
                    selected = binding.for_iteration(iteration + 1)
                    require_observation(selected, row)
                    await selected.after(selected.repair_columns[1], association, row)
                    await selected.after(selected.state.needs_repair, association, row)

        asyncio.run_coroutine_threadsafe(observe(), binding.loop).result()
        return RewriteResult(dataframe, [], tuple(range(len(dataframe))))


def project_combined_observations(row: dict[str, Any], criteria: EvaluationCriteria) -> dict[str, Any]:
    """Project generated column observations through the shared admitted control.

    Standalone column execution has no invocation hooks. Its observed decisions
    carry no QA assessment; absent stages cannot produce a terminal observation.
    If original text is unavailable, the empty datum only anchors this observed
    row's control scope; it does not authenticate an original text or protection.
    """
    from anonymizer.engine.detection._graph_profile import _run_sync_profile, project_detection_redact

    source = pd.DataFrame([{**row, COL_TEXT: row.get(COL_TEXT, "")}])
    runner = _ObservedCombinedRunner(cast(NddAdapter, None))
    arguments = {"evaluation": criteria}

    async def operation(
        pool: BlockingPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> ReplacementResult:
        del handoff
        result = await execute_initial_evaluation(
            source,
            adapter=runner._adapter,
            combined=True,
            arguments=arguments,
            configuration=_configuration(arguments, None),
            pool=pool,
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=None,
                max_runtime_artifacts=12 + 6 * criteria.max_repair_iterations,
                max_runtime_artifact_bytes=None,
                max_collection_items=None,
            ),
            runner=runner,
            opaque=True,
            observed_columns=True,
        )
        return project_detection_redact(result)

    projected = _run_sync_profile(operation, caller_callbacks=False)
    values = projected.dataframe.iloc[0].to_dict()
    projected_row = {**row, **values}
    if COL_TEXT not in row:
        projected_row.pop(COL_TEXT, None)
    return projected_row


def run_semantic_rewrite(
    source: pd.DataFrame,
    *,
    detection: EntityDetectionWorkflow,
    adapter: NddAdapter,
    combined: bool,
    arguments: dict[str, Any],
    detection_arguments: dict[str, Any],
    runner: RewriteWorkflow | None = None,
) -> ReplacementResult:
    """Run detection and the shared Rewrite tail under one admitted invocation."""
    from anonymizer.engine.detection._graph_profile import _run_sync_profile, project_detection_redact

    arguments = deepcopy(arguments)
    configuration = _configuration(arguments, detection_arguments)
    selected = (
        len(source)
        if arguments.get("preview_num_records") is None
        else min(len(source), arguments["preview_num_records"])
    )
    allowance = arguments["evaluation"].max_repair_iterations
    limits = ExecutionLimits(
        max_local_in_flight=1,
        max_remote_outstanding=None,
        max_runtime_artifacts=max(1, selected * (20 + 6 * allowance)),
        max_runtime_artifact_bytes=None,
        max_collection_items=None,
    )

    opaque = getattr(adapter.run_workflow, "__func__", None) is not NddAdapter.run_workflow
    opaque_detection = getattr(detection.run, "__func__", None) is not _DETECTION_RUN

    async def operation(
        pool: BlockingPool,
        handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure],
    ) -> ReplacementResult:
        result = await execute_initial_evaluation(
            source,
            adapter=adapter,
            combined=combined,
            arguments=arguments,
            configuration=configuration,
            pool=pool,
            limits=limits,
            detection=detection,
            detection_arguments=detection_arguments,
            runner=runner,
            opaque=opaque,
            handoff=handoff if opaque else None,
            detection_handoff=handoff if opaque_detection else None,
        )
        return project_detection_redact(result)

    return _run_sync_profile(operation, caller_callbacks=opaque or opaque_detection)


async def execute_initial_evaluation(
    source: pd.DataFrame,
    *,
    adapter: NddAdapter,
    combined: bool,
    arguments: dict[str, Any],
    configuration: FrozenConfig,
    pool: BlockingPool,
    limits: ExecutionLimits,
    detection: EntityDetectionWorkflow | None = None,
    detection_arguments: dict[str, Any] | None = None,
    runner: RewriteWorkflow | None = None,
    opaque: bool = False,
    on_result: Callable[[RewriteResult], None] | None = None,
    observed_columns: bool = False,
    handoff: Callable[[Callable[[], LocalCompleted | LocalFailure]], LocalCompleted | LocalFailure] | None = None,
    detection_handoff: Callable[[Callable[[], Any]], Any] | None = None,
) -> ExecutionResult:
    """Execute Rewrite lowerings under one admitted source-bound owner.

    Native generators publish typed stage outcomes. Opaque adapters publish
    observed decisions without QA assessments or inferred physical requests.
    Final projection uses the admitted terminal occurrence in both cases.
    """
    if source.empty or limits.max_runtime_artifact_bytes is not None or limits.max_remote_outstanding is not None:
        # The DD and arbitrary-object boundaries do not expose enforceable
        # internal request cardinality or hidden cell sizes.
        reject(EffectCode.UNSUPPORTED)
    preview = arguments.get("preview_num_records")
    selected_source = source if preview is None else source.iloc[:preview]
    arguments = deepcopy(arguments)
    criteria = EvaluationCriteria.model_validate_json(arguments["evaluation"].model_dump_json())
    expected_configuration = _configuration(arguments, detection_arguments)
    if configuration != expected_configuration:
        reject(EffectCode.CONTRADICTORY)
    arguments["evaluation"] = criteria
    graph = DataGraph.new()
    targets = []
    for text in selected_source[COL_TEXT]:
        graph, target = graph.add_text(text)
        targets.append(target)
    size = len(targets)
    allowance = 0 if opaque and combined and not observed_columns else arguments["evaluation"].max_repair_iterations
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
            max_text_bytes=sum(len(t.encode("utf-8")) for t in selected_source[COL_TEXT]),
            max_declarations=0,
            max_group_members=0,
        ),
    )
    raw, capabilities, policies, ids, body_ids = _tail(configuration, allowance, opaque=opaque)
    detection_ids = {}
    detection_config = (
        detection_configuration(detection_arguments or {})
        if detection is not None and issubclass(type(detection), EntityDetectionWorkflow)
        else None
    )
    detection_adapter = getattr(detection, "_adapter", None) if detection is not None else None
    detection_lowerings = (
        (detection.detect_and_validate_entities, detection.identify_latent_entities)
        if detection is not None and issubclass(type(detection), EntityDetectionWorkflow)
        else None
    )
    optional_detection = (
        detection is not None
        and issubclass(type(detection), EntityDetectionWorkflow)
        and getattr(detection.run, "__func__", None) is not _DETECTION_RUN
    )
    if detection is not None and issubclass(type(detection), EntityDetectionWorkflow):
        from anonymizer.engine.detection._semantic_detection import (
            extend_detection_graph,
            reparent_detection_policies,
        )

        raw, detection_capabilities, detection_policies, detection_ids = extend_detection_graph(
            raw,
            ids["prefix"],
            detection_configuration(detection_arguments or {}),
            latent=(detection_arguments or {}).get("tag_latent_entities", True),
            optional=optional_detection,
        )
        capabilities, policies = reparent_detection_policies(capabilities, policies, detection_ids["entry"])
        capabilities = (*capabilities, *detection_capabilities)
        policies = (*policies, *detection_policies)
    map_ids = {}
    map_entry = None
    map_configuration_snapshot = None
    map_factory = None
    if not combined and callable(getattr(adapter, "run_workflow", None)):
        import anonymizer.engine.rewrite.rewrite_workflow as rewrite_module
        from anonymizer.engine.constants import COL_ENTITIES_BY_VALUE
        from anonymizer.engine.replace._semantic_map import extend_map_graph, map_configuration

        map_factory = rewrite_module.LlmReplaceWorkflow
        map_arguments: dict[str, Any] = dict(
            model_configs=arguments.get("model_configs", []),
            selected_models=arguments.get("replace_model_selection"),
            instructions=None,
            entities_column=COL_ENTITIES_BY_VALUE,
            preview_num_records=None,
        )
        map_configuration_snapshot = map_configuration(map_arguments)
        if not detection_ids:
            from anonymizer.engine.detection._semantic_detection import reparent_detection_policies

            map_entry = NodeId.new(workflow=raw.workflow)
            op = OperationSpec(
                name="rewrite-map-entry",
                inputs=(InputPort(name="source", artifact_type=ArtifactType(name="anonymizer.text", revision=1)),),
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
                            max_activations=1, max_model_requests=None, max_input_bytes=None, max_output_bytes=None
                        ),
                    ),
                ),
            )
            cap, policy = _frame_implementation(map_entry, op, map_configuration_snapshot)
            raw = compile_graph(
                workflow=raw.workflow,
                interface=replace(
                    raw.interface,
                    outcomes=tuple(
                        replace(o, ceiling=replace(o.ceiling, max_activations=o.ceiling.max_activations + 1))
                        for o in raw.interface.outcomes
                    ),
                ),
                nodes=(*raw.nodes, OperationNode(id=map_entry, operation=op)),
                input_bindings=(
                    *raw.input_bindings,
                    InputBinding(
                        source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=map_entry, port="source")
                    ),
                ),
                output_bindings=tuple(raw.output_bindings),
                outcome_bindings=tuple(raw.outcome_bindings),
                sequence=(*raw.sequence, SequenceEdge(before=map_entry, after=ids["prefix"])),
                protection=tuple(raw.protection_requirements),
                limits=replace(
                    raw.limits,
                    max_nodes=raw.limits.max_nodes + 1,
                    max_bindings=raw.limits.max_bindings + 1,
                    max_sequence_edges=raw.limits.max_sequence_edges + 1,
                ),
                dynamic_limits=replace(
                    raw.dynamic_limits, max_activation_occurrences=raw.dynamic_limits.max_activation_occurrences + 1
                ),
            )
            capabilities, policies = reparent_detection_policies(capabilities, policies, map_entry)
            capabilities = (*capabilities, cap)
            policies = (*policies, policy)
        raw, map_caps, map_policies, map_ids = extend_map_graph(
            raw,
            ids["prefix"],
            map_configuration_snapshot,
            submission_owner=detection_ids["entry"] if detection_ids else cast(NodeId, map_entry),
            predecessor=detection_ids["final"] if detection_ids else map_entry,
            optional=True,
        )
        capabilities = (*capabilities, *map_caps)
        policies = (*policies, *map_policies)
    text = ArtifactType(name="anonymizer.text", revision=1)
    prepared = _prepare_frame_plan(
        data,
        raw,
        capabilities,
        tuple(BoundInput(target=t, source=t, port="source", artifact_type=text) for t in targets),
    )
    names = {node: name for name, node in ids.items() if name != "member"}
    body_names = {node: name for name, node in body_ids.items()}
    slots: dict[str, int] = {}
    iteration_slots: dict[int, dict[str, int]] = {}
    for slot in prepared.reservation_recipe:
        if slot.template in names:
            slots[names[slot.template]] = slot.index
        elif slot.template in body_names:
            ancestor = slot
            while ancestor.iteration is None and ancestor.parent_index is not None:
                ancestor = prepared.reservation_recipe[ancestor.parent_index]
            assert ancestor.iteration is not None
            iteration_slots.setdefault(ancestor.iteration, {})[body_names[slot.template]] = slot.index
    loop = asyncio.get_running_loop()

    def callback(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        started = perf_counter()
        original_frame = requests.bind_frame(source, targets=tuple(targets), source_limit=len(selected_source))
        if map_entry is not None:
            from anonymizer.engine.detection._semantic_detection import complete_detection_entry

            asyncio.run_coroutine_threadsafe(complete_detection_entry(requests, inputs), loop).result()
        if detection_ids and not optional_detection:
            from anonymizer.engine.detection._semantic_detection import complete_detection_entry

            asyncio.run_coroutine_threadsafe(complete_detection_entry(requests, inputs), loop).result()

        def detect() -> EntityDetectionResult | None:
            if detection is None:
                return None
            native = getattr(detection.run, "__func__", None) is _DETECTION_RUN
            arguments_for_detection = {
                **({"source_frame": original_frame} if native else {}),
                **(detection_arguments or {}),
            }
            if detection_ids:
                from anonymizer.engine.detection._semantic_detection import (
                    bind_detection,
                    complete_detection_entry,
                    detection_slots,
                )

                with bind_detection(
                    detection,
                    source,
                    arguments_for_detection,
                    requests,
                    original_frame,
                    detection_slots(prepared, detection_ids),
                    loop,
                    detection_config,
                    detection_adapter,
                    detection_lowerings,
                    inputs if optional_detection else None,
                ) as binding:
                    result = detection.run(source, **arguments_for_detection)
                if optional_detection and not binding.claim.locked():
                    asyncio.run_coroutine_threadsafe(
                        complete_detection_entry(requests, inputs, "bypass"), loop
                    ).result()
                return result
            return detection.run(source, **arguments_for_detection)

        if detection_handoff is None or handoff is not None:
            detected = detect()
        else:
            context = copy_context()
            detected = detection_handoff(lambda: context.run(detect))
        if requests.cancelled:
            return LocalFailure(failure="implementation_exception")
        if _configuration(arguments, detection_arguments) != configuration:
            reject(EffectCode.CONTRADICTORY)
        detected_frame = detected.dataframe if detected is not None else selected_source
        positions = (
            resolve_source_positions(source, detected_frame, detected.source_positions)
            if detected is not None
            else tuple(range(len(selected_source)))
        )
        active = tuple(original_frame.associations[index] for index in positions)
        selected_runner = runner or (CombinedRewriteWorkflow if combined else RewriteWorkflow)(adapter)

        def prepare_maps(entity_rows: pd.DataFrame, passthrough_rows: pd.DataFrame):
            import anonymizer.engine.rewrite.rewrite_workflow as rewrite_module
            from anonymizer.engine.replace._semantic_map import bind_map, map_configuration
            from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceWorkflow
            from anonymizer.engine.row_partitioning import ROW_ORDER_COL

            if (
                rewrite_module.LlmReplaceWorkflow is not map_factory
                or map_configuration(map_arguments) != map_configuration_snapshot
            ):
                reject(EffectCode.FOREIGN_OWNER)
            map_slots = {
                name: next(slot.index for slot in prepared.reservation_recipe if slot.template == node)
                for name, node in map_ids.items()
            }
            map_result = None
            surviving = []
            for rows, passthrough in ((passthrough_rows, True), (entity_rows, False)):
                if rows.empty:
                    continue
                associations = tuple(active[position] for position in rows[ROW_ORDER_COL])
                entry_inputs = asyncio.run_coroutine_threadsafe(
                    requests.enter_operations(map_slots["entry"], associations), loop
                ).result()
                helper = LlmReplaceWorkflow(adapter) if passthrough else map_factory(adapter=adapter)
                native_helper = helper if isinstance(helper, LlmReplaceWorkflow) else LlmReplaceWorkflow(adapter)
                if any(item is None for item in entry_inputs):
                    reject(EffectCode.REQUEST_CAUSALITY)
                entry_inputs = cast(tuple[AssociationInput, ...], entry_inputs)
                with bind_map(
                    native_helper,
                    rows,
                    map_arguments,
                    requests,
                    WorkflowSourceFrame.capture(rows, associations=associations, invocation_bound=True),
                    {k: v for k, v in map_slots.items() if k != "entry"},
                    loop,
                    entry_inputs=entry_inputs,
                ) as mapping:
                    if passthrough:
                        result = mapping.run(native_helper, rows, map_arguments)
                    else:
                        result = helper.generate_map_only(
                            rows,
                            model_configs=map_arguments["model_configs"],
                            selected_models=map_arguments["selected_models"],
                        )
                    native = mapping.claim.locked()
                    if native:
                        mapping.authenticate_result(result)
                    if not native:
                        mapping.bypass()
                if passthrough:
                    surviving.extend(associations)
                else:
                    map_result = result
                    selected_positions = (
                        resolve_source_positions(rows, result.dataframe, result.source_positions)
                        if native
                        else tuple(range(len(rows)))
                    )
                    surviving.extend(associations[index] for index in selected_positions)

            async def begin_map_prefix() -> None:
                for association in surviving:
                    await requests.enter_operation(slots["prefix"], association)

            asyncio.run_coroutine_threadsafe(begin_map_prefix(), loop).result()
            if not observed_columns:
                asyncio.run_coroutine_threadsafe(binding.passthrough(detected_frame), loop).result()
            return map_result

        binding = RewriteEvaluationBinding(
            requests,
            WorkflowSourceFrame.capture(detected_frame, associations=active, invocation_bound=True),
            slots,
            loop,
            _evaluation_state(0 if combined else None, COL_REWRITTEN_TEXT_INITIAL if combined else COL_REWRITTEN_TEXT),
            iteration_slots=iteration_slots,
            combined=combined,
            opaque=opaque,
            observed_columns=observed_columns,
            prefix_is_child=bool(detection_ids) or map_entry is not None,
            criteria=arguments["evaluation"],
            receiver=selected_runner,
            adapter=adapter,
            entry_frame=detected_frame,
            entry_arguments=deepcopy(arguments),
        )
        binding._map_prepare = prepare_maps if map_ids else None
        detection_elapsed = perf_counter() - started
        started = perf_counter()
        if detection_ids and not map_ids:

            async def begin_prefix() -> None:
                for association in active:
                    await requests.enter_operation(slots["prefix"], association)

            asyncio.run_coroutine_threadsafe(begin_prefix(), loop).result()
        if not observed_columns and not map_ids:
            asyncio.run_coroutine_threadsafe(binding.passthrough(detected_frame), loop).result()
        binding.authorize_entry()
        with selected_runner._bind_semantic(binding):
            result = selected_runner.run(detected_frame, **arguments)
        if on_result is not None:
            on_result(result)
        if result.source_positions is None:
            reject(EffectCode.MISSING)
        positions = resolve_source_positions(detected_frame, result.dataframe, result.source_positions)
        successful = tuple(binding.source.associations[index] for index in positions)
        selected, projected = asyncio.run_coroutine_threadsafe(
            binding.finish(result.dataframe, successful), loop
        ).result()
        publication = FramePublication.capture(
            projected,
            associations=tuple(selected[association] for association in successful),
            text_ports=(),
            observation_ports=((COL_REWRITTEN_TEXT, "candidate"),),
            failed_records=tuple((detected.failed_records if detected is not None else []) + result.failed_records),
            opaque_source=source,
            stages=FrameStages(
                detected_frame,
                tuple(detected.failed_records) if detected is not None else (),
                tuple(result.failed_records),
                detection_elapsed,
                perf_counter() - started,
            )
            if detected is not None
            else None,
        )
        return LocalCompleted(
            results=tuple(
                AssociationResult(
                    association=item.association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                )
                for item in inputs
                if item.association in successful
            ),
            failed_sources=frozenset(set(original_frame.associations) - set(successful)),
            frame=publication,
        )

    def dispatch(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted | LocalFailure:
        if handoff is None:
            return callback(inputs, requests)
        context = copy_context()
        return handoff(lambda: context.run(callback, inputs, requests))

    result = await _execute_frame_plan(
        prepared=prepared,
        capabilities=tuple(capabilities),
        policies=tuple(policies),
        callback=dispatch,
        pool=pool,
        borrowed_loop=loop,
        limits=replace(limits, max_runtime_artifacts=limits.max_runtime_artifacts + (20 * size if map_ids else 0)),
        ingress_limits=IngressLimits(
            max_associations=size,
            max_outputs_per_association=8,
            max_collection_items=None if limits.max_collection_items is None else limits.max_collection_items * size,
            max_assessments=1,
            max_metadata_items=None
            if limits.max_collection_items is None
            else size * (20 + 10 * limits.max_collection_items),
            max_text_bytes=limits.max_runtime_artifact_bytes,
            max_metadata_bytes=limits.max_runtime_artifact_bytes,
            max_depth=8,
        ),
        assessment_productions=tuple(
            EvidenceProductionDecl(
                node=node,
                outcome=outcome,
                promise=EVALUATION_PROMISE,
                evidence_port="metrics",
                supported_findings=frozenset({EVALUATION_PASS, EVALUATION_FAIL}),
            )
            for node in (() if opaque else (ids["initial"], ids["final"]))
            for outcome in ("ok",)
        ),
        assessment_limits=AssessmentLimits(
            max_productions=2,
            max_findings_per_production=2,
            max_finding_code_bytes=32,
            max_assessment_facts=2 * size,
            max_port_facts=((8 + 6 * allowance) * 20 + (48 if map_ids else 0)) * size,
            max_provenance_edges=((8 + 6 * allowance) * 64 + (100 if map_ids else 0)) * size,
        ),
    )

    if opaque:
        return result
    qualification = admit_qualification(
        execution=result._execution,
        productions=result._execution.assessment_productions,
        limits=QualificationLimits(
            max_productions=2,
            max_submissions=size,
            max_port_facts=((8 + 6 * allowance) * 20 + (48 if map_ids else 0)) * size,
            max_consumed_per_assessment=8,
            max_coverage_atoms=0,
            max_verified_evidence=size,
            max_fixed_point_steps=max(1, size),
            max_revision_entries=(8 + 6 * allowance) * 20 * size,
            max_provenance_edges=((8 + 6 * allowance) * 64 + (100 if map_ids else 0)) * size,
        ),
    )
    current = evidence_revision_view(
        admitted=qualification,
        result=result,
        artifacts=tuple(reference for reference, _ in result.artifacts),
        configurations=tuple(dict((fact.node, fact.environment.configuration) for fact in result.assessments).items()),
    )
    qualify_retained(admitted=qualification, result=result, current=current)
    verify_evidence(
        admitted=qualification,
        result=result,
        submissions=tuple(AssessmentSubmission(fact=fact) for fact in result.assessments),
    )
    return result


def _tail(configuration: FrozenConfig, allowance: int, *, opaque: bool = False):
    """Declare one initial evaluator and one carried repair/evaluation body."""
    text = ArtifactType(name="anonymizer.text", revision=1)
    qa, answers, privacy, comparison, metrics = (
        REWRITE_TYPES[t]
        for t in (
            RewriteQAValue,
            QualityAnswersValue,
            PrivacyAnswersValue,
            QualityComparisonValue,
            EvaluationMetricsValue,
        )
    )
    if opaque:
        metrics = REWRITE_TYPES[RewriteControlValue]
    candidate = REWRITE_TYPES[RewriteCandidateValue]
    carried = {
        "source": text,
        "baseline": candidate,
        "qa": qa,
        "candidate": candidate,
        "privacy": privacy,
        "answers": answers,
        "comparison": comparison,
        "metrics": metrics,
    }
    if opaque:
        carried = {
            key: value for key, value in carried.items() if key not in {"qa", "privacy", "answers", "comparison"}
        }
    root, body = WorkflowId.new(), WorkflowId.new()
    ids = {
        name: NodeId.new(workflow=root)
        for name in ("prefix", "quality", "privacy", "comparison", "metrics", "member", "join", "initial", "final")
    }
    body_ids = {name: NodeId.new(workflow=body) for name in ("repair", "quality", "privacy", "comparison", "metrics")}
    capabilities, policies = [], []

    def operation(name, inputs, outputs, *, identity=(), control=False, local=False, ceiling=1):
        return OperationSpec(
            name=name,
            inputs=tuple(InputPort(name="in_" + k if k in outputs else k, artifact_type=v) for k, v in inputs.items()),
            outputs=tuple(OutputPort(name=k, artifact_type=v) for k, v in outputs.items()),
            output_dependencies=tuple(
                OutputDependency(
                    output=k,
                    inputs=frozenset({"in_" + k})
                    if k in identity
                    else frozenset("in_" + i if i in outputs else i for i in inputs),
                    identity_input="in_" + k if k in identity else None,
                )
                for k in outputs
            ),
            outcomes=tuple(
                OutcomeSpec(
                    name=n,
                    category="success",
                    produced_ports=frozenset(outputs),
                    context=frozenset(),
                    evidence=frozenset(),
                    model_requirements=frozenset(),
                    ceiling=ResourceCeiling(
                        max_activations=ceiling,
                        max_model_requests=0 if local else None,
                        max_input_bytes=None,
                        max_output_bytes=None,
                    ),
                )
                for n in (
                    ("again", "stop", "exhausted")
                    if control
                    else ("ok", "skipped")
                    if name.endswith(("prefix", "quality", "privacy", "comparison"))
                    else ("ok",)
                )
            ),
        )

    def node(owner_ids, name, inputs, outputs, **kwargs):
        op = operation("rewrite-" + ("iteration-" if owner_ids is body_ids else "") + name, inputs, outputs, **kwargs)
        if name == "metrics" and owner_ids is ids:
            op = replace(op, outcomes=(*op.outcomes, replace(op.outcomes[1], name="skipped")))
        if name in {"initial", "final"} and not opaque:
            promise = EvidencePromise(
                name=EVALUATION_PROMISE,
                meaning=EVALUATION_MEANING,
                subject_port="in_candidate",
                consumed_ports=frozenset(port.name for port in op.inputs),
                coverage=frozenset(),
            )
            op = replace(
                op,
                output_dependencies=tuple(
                    replace(dep, inputs=promise.consumed_ports, identity_input=None) if dep.output == "metrics" else dep
                    for dep in op.output_dependencies
                ),
                outcomes=tuple(replace(outcome, evidence=frozenset({promise})) for outcome in op.outcomes),
            )
            if name == "initial":
                op = replace(op, outcomes=(*op.outcomes, replace(op.outcomes[0], name="skipped", evidence=frozenset())))
        if name == "initial" and opaque:
            op = replace(op, outcomes=(*op.outcomes, replace(op.outcomes[0], name="skipped")))
        cap, policy = _frame_implementation(
            owner_ids[name],
            op,
            configuration,
            submission_owner=None if name == "prefix" else ids["prefix"],
            local=kwargs.get("local", False),
        )
        capabilities.append(cap)
        policies.append(policy)
        return OperationNode(id=owner_ids[name], operation=op)

    def evaluation(owner_ids, first, first_inputs, first_outputs):
        nodes = [
            node(owner_ids, first, first_inputs, first_outputs),
            *(
                []
                if opaque
                else [
                    node(owner_ids, "quality", {"candidate": candidate, "qa": qa}, {"answers": answers}),
                    node(owner_ids, "privacy", {"candidate": candidate, "qa": qa}, {"privacy": privacy}),
                    node(owner_ids, "comparison", {"qa": qa, "answers": answers}, {"comparison": comparison}),
                ]
            ),
            node(
                owner_ids,
                "metrics",
                {k: v for k, v in carried.items() if k != "metrics"},
                carried,
                identity=tuple(k for k in carried if k != "metrics"),
                control=True,
                local=True,
            ),
        ]
        bindings, edges = [], set()
        for current in nodes:
            name = next(k for k, v in owner_ids.items() if v == current.id)
            for port in current.operation.inputs:
                key = port.name.removeprefix("in_")
                if name == first:
                    origin = WorkflowInputRef(port="in_" + key if first == "repair" else key)
                elif key in {"source", "metrics"}:
                    origin = WorkflowInputRef(port="in_" + key if first == "repair" else key)
                elif key in {"qa", "baseline"} and first == "repair":
                    origin = WorkflowInputRef(port="in_" + key if first == "repair" else key)
                else:
                    predecessor = (
                        "quality"
                        if key == "answers"
                        else "privacy"
                        if key == "privacy"
                        else "comparison"
                        if key == "comparison"
                        else first
                    )
                    origin = NodeOutputRef(node=owner_ids[predecessor], port=key)
                    edges.add((owner_ids[predecessor], current.id))
                bindings.append(InputBinding(source=origin, destination=NodeInputRef(node=current.id, port=port.name)))
        return nodes, bindings, edges

    # Initial metrics have no prior metrics input; each repair body consumes all carried values.
    root_nodes, root_bindings, root_edges = evaluation(
        ids,
        "prefix",
        {"source": text},
        {"candidate": candidate, "baseline": candidate, **({} if opaque else {"qa": qa})},
    )
    body_nodes, body_bindings, body_edges = evaluation(body_ids, "repair", carried, {"candidate": candidate})
    body_interface = operation(
        "rewrite-repair-evaluate", carried, carried, identity=("source", "baseline", "qa"), control=True, ceiling=5
    )

    def admit(owner, interface, nodes, bindings, edges, outputs, outcomes, controls=()):
        return compile_graph(
            workflow=owner,
            interface=interface,
            nodes=(*nodes, *controls),
            input_bindings=tuple(bindings),
            output_bindings=tuple(outputs),
            outcome_bindings=tuple(outcomes),
            sequence=tuple(SequenceEdge(before=a, after=b) for a, b in sorted(edges, key=lambda e: repr(e))),
            protection=(),
            dynamic_limits=DynamicLimits(
                max_maps=0,
                max_joins=1,
                max_loops=1,
                max_children_per_map=0,
                max_iterations_per_loop=allowance,
                max_dynamic_depth=3,
                max_activation_occurrences=max(5, 8 + 6 * allowance),
            ),
            limits=WorkflowLimits(
                max_nodes=20,
                max_bindings=100,
                max_sequence_edges=30,
                max_choices=1,
                max_branch_members=4,
                max_subgraph_depth=2,
                max_choice_states=3,
            ),
        )

    body_raw = admit(
        body,
        body_interface,
        body_nodes,
        body_bindings,
        body_edges,
        [
            OutputBinding(source=NodeOutputRef(node=body_ids["metrics"], port=k), destination=WorkflowOutputRef(port=k))
            for k in carried
        ],
        [
            OutcomeBinding(
                source=NodeOutcomeRef(node=body_ids["metrics"], outcome=n), destination=WorkflowOutcomeRef(outcome=n)
            )
            for n in ("again", "stop", "exhausted")
        ],
    )
    root_nodes.append(SubgraphNode(id=ids["member"], operation=body_interface, body=body_raw))
    root_nodes.extend(
        [
            node(ids, "join", {"metrics": metrics}, {}, control=True, local=True),
            node(ids, "initial", carried, carried, identity=tuple(carried), local=True),
            node(ids, "final", carried, carried, identity=tuple(carried), local=True),
        ]
    )
    root_bindings.extend(
        InputBinding(
            source=NodeOutputRef(node=ids["metrics"], port=k),
            destination=NodeInputRef(node=ids["member"], port="in_" + k),
        )
        for k in carried
    )
    root_bindings.extend(
        InputBinding(
            source=NodeOutputRef(node=ids[src], port=port),
            destination=NodeInputRef(node=ids[dst], port="in_" + port),
        )
        for src, dst in (("metrics", "initial"), ("member", "final"))
        for port in carried
    )
    root_bindings.append(
        InputBinding(
            source=NodeOutputRef(node=ids["metrics"], port="metrics"),
            destination=NodeInputRef(node=ids["join"], port="metrics"),
        )
    )
    root_edges.update(
        (ids[a], ids[b]) for a, b in (("metrics", "member"), ("member", "join"), ("join", "final"), ("join", "initial"))
    )
    interface = operation(
        "rewrite-tail",
        {"source": text},
        {"initial_candidate": candidate, "repaired_candidate": candidate},
        ceiling=max(14, 8 + 6 * allowance),
    )
    interface = replace(
        interface,
        outcomes=tuple(
            replace(
                interface.outcomes[0],
                name=n,
                produced_ports=frozenset({p}),
                evidence=frozenset()
                if n == "skipped" or opaque
                else frozenset(
                    {
                        EvidencePromise(
                            name=EVALUATION_PROMISE,
                            meaning=EVALUATION_MEANING,
                            subject_port=p,
                            consumed_ports=frozenset({"source"}),
                            coverage=frozenset(),
                        )
                    }
                ),
            )
            for n, p in (
                ("initial", "initial_candidate"),
                ("final", "repaired_candidate"),
                ("skipped", "initial_candidate"),
            )
        ),
    )
    loop = LoopDecl(
        starter=ids["metrics"],
        member=ids["member"],
        join=ids["join"],
        enter_outcomes=frozenset({"again"}),
        bypass_outcomes=frozenset({"stop", "exhausted", "skipped"}),
        continue_outcomes=frozenset({"again"}),
        exit_outcomes=frozenset({"stop", "exhausted"}),
        initial=tuple(
            LoopInitialBinding(
                source=NodeOutputRef(node=ids["metrics"], port=k),
                destination=NodeInputRef(node=ids["member"], port="in_" + k),
            )
            for k in carried
        ),
        carried=tuple(
            LoopCarriedBinding(
                source=NodeOutputRef(node=ids["member"], port=k),
                destination=NodeInputRef(node=ids["member"], port="in_" + k),
            )
            for k in carried
        ),
        max_iterations=allowance,
    )
    raw = admit(
        root,
        interface,
        root_nodes,
        root_bindings,
        root_edges,
        [
            OutputBinding(
                source=NodeOutputRef(node=ids[n], port="candidate"),
                destination=WorkflowOutputRef(port="initial_candidate" if n == "initial" else "repaired_candidate"),
            )
            for n in ("initial", "final")
        ],
        [
            *[
                OutcomeBinding(
                    source=NodeOutcomeRef(node=ids[n], outcome="ok"), destination=WorkflowOutcomeRef(outcome=n)
                )
                for n in ("initial", "final")
            ],
            OutcomeBinding(
                source=NodeOutcomeRef(node=ids["initial"], outcome="skipped"),
                destination=WorkflowOutcomeRef(outcome="skipped"),
            ),
        ],
        (
            loop,
            KeyedJoinDecl(
                source=ids["metrics"],
                join=ids["join"],
                accepted_categories=frozenset({"success"}),
                reduction="all_by_key",
            ),
            ChoiceDecl(
                selector=ids["join"],
                branches=(
                    ChoiceBranch(outcomes=frozenset({"stop", "exhausted"}), members=frozenset({ids["initial"]})),
                    ChoiceBranch(outcomes=frozenset({"again"}), members=frozenset({ids["final"]})),
                ),
            ),
        ),
    )
    return raw, tuple(capabilities), tuple(policies), ids, body_ids
