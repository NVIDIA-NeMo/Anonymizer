# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct replacement entries use the same admitted map stages as composed entries."""

from __future__ import annotations

import asyncio
from contextvars import copy_context
from dataclasses import replace
from typing import Any, cast

import pandas as pd

from anonymizer.engine.constants import COL_REPLACED_TEXT, COL_REPLACEMENT_MAP, COL_TEXT
from anonymizer.engine.detection._semantic_detection import complete_detection_entry
from anonymizer.engine.graph_sdk._authoring_defaults import (
    _compile_graph,
    _execution_limits,
    _ingress_limits,
    _operation_spec,
    _outcome_spec,
    _validate_targets,
)
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk._workflow_values import (
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    OperationNode,
    OutcomeBinding,
    ResourceCeiling,
    SequenceEdge,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
)
from anonymizer.engine.graph_sdk.blocking import BlockingPool
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    DelegatedRequestContract,
    LocalCompleted,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, SemanticAssociation
from anonymizer.engine.harness.frames import FramePublication, InvocationFrame
from anonymizer.engine.harness.kernel import (
    _execute_frame_plan,
    _frame_implementation,
    _prepare_frame_plan,
    _run_sync_profile,
    project_detection_redact,
)
from anonymizer.engine.harness.retention import attach_execution
from anonymizer.engine.ndd._route import native_map
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd._stage_binding import completed
from anonymizer.engine.replace._semantic_map import _TEXT, MapBinding, extend_map_graph, map_configuration
from anonymizer.engine.replace.llm_replace_workflow import LlmReplaceResult, LlmReplaceWorkflow
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.source_text import source_text


async def _enter_empty(requests: DelegatedRequests, slot: int) -> tuple[AssociationInput, ...]:
    item = await requests.enter_operation(slot, None)
    if item is None:
        reject(EffectCode.REQUEST_CAUSALITY)
    return (item,)


def run_direct_map(
    receiver: LlmReplaceWorkflow,
    dataframe: pd.DataFrame,
    arguments: dict[str, Any],
    *,
    apply: bool = False,
) -> LlmReplaceResult:
    captured = []
    configuration = map_configuration(arguments)
    opaque = not native_map(receiver)

    async def execute(pool: BlockingPool, handoff: Any) -> ReplacementResult:
        graph = DataGraph.new()
        targets = []
        texts = dataframe[COL_TEXT] if COL_TEXT in dataframe else [""] * len(dataframe)
        for text in texts:
            graph, target = graph.add_text(source_text(text))
            targets.append(target)
        size = len(targets)
        data = _validate_targets(
            graph,
            targets=tuple(targets),
            limits=DataLimits(
                max_datums=size,
                max_targets=size,
                max_text_bytes=sum(len(text.encode()) for text in texts),
                max_declarations=0,
                max_group_members=0,
            ),
        )
        workflow = WorkflowId.new()
        entry = NodeId.new(workflow=workflow)
        owner = NodeId.new(workflow=workflow)
        operation = _operation_spec(
            name="replacement-entry",
            inputs=(InputPort(name="source", artifact_type=_TEXT),) if size else (),
            outcomes=(
                _outcome_spec(
                    name="ok",
                    category="success",
                    ceiling=ResourceCeiling(
                        max_activations=1, max_model_requests=None, max_input_bytes=None, max_output_bytes=None
                    ),
                ),
            ),
        )
        projection = replace(operation, name="replacement-projection")
        raw = _compile_graph(
            workflow=workflow,
            interface=replace(
                operation,
                outcomes=(
                    replace(operation.outcomes[0], ceiling=replace(operation.outcomes[0].ceiling, max_activations=2)),
                ),
            ),
            nodes=(OperationNode(id=entry, operation=operation), OperationNode(id=owner, operation=projection)),
            input_bindings=tuple(
                InputBinding(source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=node, port="source"))
                for node in (entry, owner)
            )
            if size
            else (),
            outcome_bindings=(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=owner, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
                ),
            ),
            sequence=(SequenceEdge(before=entry, after=owner),),
            limits=WorkflowLimits(
                max_nodes=2,
                max_bindings=3,
                max_sequence_edges=1,
                max_choices=0,
                max_branch_members=0,
                max_subgraph_depth=1,
                max_choice_states=1,
            ),
        )
        cap, policy = _frame_implementation(entry, operation, configuration, aggregate=not size)
        projection_cap, projection_policy = _frame_implementation(
            owner, projection, configuration, submission_owner=entry, local=True
        )
        raw, child_caps, child_policies, ids = extend_map_graph(
            raw,
            owner,
            configuration,
            submission_owner=entry,
            predecessor=entry,
            source_free=not size,
            apply=apply,
            optional=True,
        )
        policy = replace(
            policy,
            delegated=replace(
                cast(DelegatedRequestContract, policy.delegated),
                publication_sources=tuple(
                    (ids[name], frozenset({"map", "protected"} if apply else {"map"})) for name in ("accept", "empty")
                ),
            ),
        )
        capabilities = (cap, projection_cap, *child_caps)
        policies = (policy, projection_policy, *child_policies)
        from anonymizer.engine.replace._native_map_requests import extend_native_map_requests, native_map_capacity

        native_ids = {}
        capacity = native_map_capacity(receiver, arguments)
        preview = arguments.get("preview_num_records")
        # A preview narrower than the frame (a caller detector's extra rows) leaves sources DD never generates;
        # those run through the opaque DD call, as in the release.
        if size and capacity is not None and (preview is None or preview >= size):
            raw, physical_caps, physical_policies, native_ids = extend_native_map_requests(raw, capacity)
            capabilities = (*capabilities, *physical_caps)
            policies = (*policies, *physical_policies)
        prepared = _prepare_frame_plan(
            data,
            raw,
            capabilities,
            tuple(BoundInput(target=t, source=t, port="source", artifact_type=_TEXT) for t in targets),
            invocation_owned=not size,
        )
        slots = {
            name: next(slot.index for slot in prepared.reservation_recipe if slot.template == node)
            for name, node in ids.items()
        }
        owner_slot = next(slot.index for slot in prepared.reservation_recipe if slot.template == owner)
        loop = asyncio.get_running_loop()

        def callback(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
            bound_frame = dataframe if COL_TEXT in dataframe else dataframe.assign(**{COL_TEXT: [""] * size})
            issued = requests.bind_frame(bound_frame, targets=tuple(targets), source_limit=size)
            source = WorkflowSourceFrame.capture(dataframe, associations=issued.associations, invocation_bound=True)
            asyncio.run_coroutine_threadsafe(complete_detection_entry(requests, inputs), loop).result()
            map_entry_slot = slots["entry"]
            entry_inputs = asyncio.run_coroutine_threadsafe(
                requests.enter_operations(map_entry_slot, source.associations)
                if size
                else _enter_empty(requests, map_entry_slot),
                loop,
            ).result()
            if any(item is None for item in entry_inputs):
                reject(EffectCode.REQUEST_CAUSALITY)
            entry_inputs = cast(tuple[AssociationInput, ...], entry_inputs)
            binding = MapBinding(
                requests,
                source,
                {k: v for k, v in slots.items() if k != "entry"},
                loop,
                receiver,
                dataframe,
                arguments,
                apply=apply,
                entry_inputs=entry_inputs,
            )
            result = binding.lower()
            captured.append(result)
            successful = tuple(
                source.associations[position] for position in cast(tuple[int, ...], result.source_positions)
            )

            async def finish() -> None:
                entered = (
                    await requests.enter_operations(owner_slot, successful)
                    if successful
                    else ((await requests.enter_operation(owner_slot, None),) if not size else ())
                )
                if any(item is None for item in entered):
                    reject(EffectCode.REQUEST_CAUSALITY)
                entered = cast(tuple[AssociationInput, ...], entered)
                await requests.complete_operations(
                    tuple(
                        (
                            cast(SemanticAssociation, item.association),
                            completed(item.association),
                        )
                        for item in entered
                    )
                )

            asyncio.run_coroutine_threadsafe(finish(), loop).result()
            return LocalCompleted(
                results=tuple(
                    AssociationResult(
                        association=item.association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                    )
                    for item in inputs
                    if item.association in successful
                ),
                failed_sources=frozenset(set(source.associations) - set(successful)),
                frame=FramePublication.capture(
                    result.dataframe,
                    associations=tuple(association for _, association, _ in binding.terminal),
                    text_ports=((COL_REPLACED_TEXT, "protected"),) if apply else (),
                    json_ports=((COL_REPLACEMENT_MAP, "map"),),
                    failed_records=tuple(result.failed_records),
                    opaque_source=dataframe,
                )
                if size
                else InvocationFrame(
                    result.dataframe, result.dataframe, tuple(result.failed_records), (), 0.0, 0.0, None
                ),
            )

        def dispatch(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
            context = copy_context()
            return handoff(lambda: context.run(callback, inputs, requests)) if opaque else callback(inputs, requests)

        result = await _execute_frame_plan(
            prepared=prepared,
            capabilities=capabilities,
            policies=policies,
            callback=dispatch,
            pool=pool,
            borrowed_loop=loop,
            limits=_execution_limits(
                max_local_in_flight=1,
                max_runtime_artifacts=max(1, 20 * size),
            ),
            ingress_limits=_ingress_limits(
                max_associations=max(1, size),
                max_outputs_per_association=4,
                max_assessments=1,
                max_depth=10,
            ),
            assessment_limits=AssessmentLimits(
                max_productions=0,
                max_findings_per_production=0,
                max_finding_code_bytes=0,
                max_assessment_facts=0,
                max_port_facts=40 * size,
                max_provenance_edges=80 * size,
            ),
            native_request_nodes=native_ids,
        )
        return project_detection_redact(result)

    projected = _run_sync_profile(execute, caller_callbacks=opaque)
    result = LlmReplaceResult(projected.dataframe, projected.failed_records, captured[0].source_positions)
    attach_execution(result, projected._execution)
    return result
