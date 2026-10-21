# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detection stage lowerings and terminal projection owned by admitted graph control."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from contextvars import copy_context
from dataclasses import dataclass, replace
from inspect import signature
from typing import Any, cast

import pandas as pd
from pydantic import TypeAdapter

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection.detection_workflow import (
    _DETECTION_RUN,
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
    _materialize_final_entities,
)
from anonymizer.engine.detection.entity_label_examples import resolve_entity_label_config
from anonymizer.engine.graph_sdk._authoring_defaults import (
    _compile_graph,
    _execution_limits,
    _ingress_limits,
    _operation_spec,
    _outcome_spec,
    _validate_targets,
)
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_delegated import SubmitterPool, drive
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk._workflow_values import (
    ArtifactType,
    Graph,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OutcomeBinding,
    OutputBinding,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SequenceEdge,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
)
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, ImplementationCapability
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    ExecutionResult,
    LocalCompleted,
    LocalFailure,
    OperationExecutionPolicy,
)
from anonymizer.engine.graph_sdk.preparation import BoundInput, PreparedPlan
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.harness.frames import FramePublication
from anonymizer.engine.harness.retention import attach_execution
from anonymizer.engine.ndd._route import native_adapter
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd._stage_binding import StageBinding, completed
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.row_partitioning import resolve_source_positions
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.engine.source_text import same_source_text, source_text
from anonymizer.interface.errors import AnonymizerWorkflowError
from anonymizer.measurement import stage_timer

_TEXT = ArtifactType(name="anonymizer.text", revision=1)
_ENTITIES = ArtifactType(name="anonymizer.entity-observation-json", revision=1)


def detection_configuration(arguments: dict[str, Any]) -> FrozenConfig:
    from anonymizer.engine.graph_sdk.capabilities import ConfigField, ConfigText

    normalized = signature(_DETECTION_RUN).bind(None, None, **arguments)
    normalized.apply_defaults()
    values = {
        key: value
        for key, value in normalized.arguments.items()
        # privacy_goal is a Rewrite argument admitted by the facade; a caller detector may mutate its own copy
        # before delegating to builtin detection, so it is not part of the detection configuration.
        if key not in {"self", "dataframe", "source_frame", "privacy_goal"}
    }
    return FrozenConfig(
        fields=(
            ConfigField(
                name="detection", value=ConfigText(value=TypeAdapter(dict[str, Any]).dump_json(values).decode())
            ),
        )
    )


def extend_detection_graph(
    raw: Graph,
    owner: NodeId,
    configuration: FrozenConfig,
    *,
    latent: bool,
    physical_sink: NodeId | None = None,
    source_free: bool = False,
) -> tuple[Graph, tuple[ImplementationCapability, ...], tuple[OperationExecutionPolicy, ...], dict[str, NodeId]]:
    """Compose detection children into an existing submission owner's graph."""
    from anonymizer.engine.harness.kernel import _frame_implementation

    names = ("core", "latent", "final") if latent else ("core", "final")
    ids = {name: NodeId.new(workflow=raw.workflow) for name in (*names, "entry")}
    entry = _operation_spec(
        name="detection-entry",
        inputs=() if source_free else (InputPort(name="source", artifact_type=_TEXT),),
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
    entry_cap, entry_policy = _frame_implementation(ids["entry"], entry, configuration)
    nodes = [OperationNode(id=ids["entry"], operation=entry)]
    bindings = [
        InputBinding(source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=ids["entry"], port="source"))
    ]
    if source_free:
        bindings = []
    edges = [SequenceEdge(before=ids["entry"], after=ids["core"]), SequenceEdge(before=ids["final"], after=owner)]
    capabilities, policies = [entry_cap], [entry_policy]
    for ordinal, name in enumerate(names):
        inputs = {} if source_free else {"source": _TEXT, **({"detected": _ENTITIES} if ordinal else {})}
        outputs = (
            {}
            if source_free
            else {
                **({"source": _TEXT} if physical_sink is None else {}),
                "entities" if name == "final" else "detected": _ENTITIES,
            }
        )
        if physical_sink is not None and name == "final":
            inputs.pop("source", None)
        operation = _operation_spec(
            name="detection-" + name,
            inputs=tuple(InputPort(name="in_" + key, artifact_type=value) for key, value in inputs.items()),
            outputs=tuple(OutputPort(name=key, artifact_type=value) for key, value in outputs.items()),
            output_dependencies=tuple(
                OutputDependency(
                    output=key,
                    inputs=frozenset({"in_source"}) if key == "source" else frozenset("in_" + k for k in inputs),
                    identity_input="in_source" if key == "source" else None,
                )
                for key in outputs
            ),
            outcomes=(
                _outcome_spec(
                    name="ok",
                    category="success",
                    produced_ports=frozenset(outputs),
                    ceiling=ResourceCeiling(
                        max_activations=1,
                        max_model_requests=0 if name == "final" or physical_sink is not None else None,
                        max_input_bytes=None,
                        max_output_bytes=None,
                    ),
                ),
            ),
        )
        nodes.append(OperationNode(id=ids[name], operation=operation))
        for key in inputs:
            origin = (
                WorkflowInputRef(port="source")
                if ordinal == 0 or (physical_sink is not None and key == "source")
                else NodeOutputRef(node=ids[names[ordinal - 1]], port=key)
            )
            bindings.append(InputBinding(source=origin, destination=NodeInputRef(node=ids[name], port="in_" + key)))
        if ordinal:
            edges.append(SequenceEdge(before=ids[names[ordinal - 1]], after=ids[name]))
        cap, policy = _frame_implementation(
            ids[name],
            operation,
            configuration,
            submission_owner=ids["entry"],
            local=name == "final" or physical_sink is not None,
        )
        capabilities.append(cap)
        policies.append(policy)
    if physical_sink is not None:
        nodes = nodes[1:]
        bindings = bindings[1:]
        edges = [*edges[2:], SequenceEdge(before=ids["final"], after=physical_sink)]
        capabilities = capabilities[1:]
        policies = [replace(policy, submission_owner=owner) for policy in policies[1:]]
        ids.pop("entry")
    interface = replace(
        raw.interface,
        outcomes=tuple(
            replace(
                outcome,
                ceiling=replace(
                    outcome.ceiling,
                    max_activations=outcome.ceiling.max_activations + len(names) + 1,
                    max_input_bytes=None,
                    max_output_bytes=None,
                ),
            )
            for outcome in raw.interface.outcomes
        ),
    )
    extended = _compile_graph(
        workflow=raw.workflow,
        interface=interface,
        nodes=(*raw.nodes, *nodes),
        input_bindings=(*raw.input_bindings, *bindings),
        output_bindings=tuple(raw.output_bindings),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(*raw.sequence, *edges),
        protection=tuple(raw.protection_requirements),
        limits=replace(
            raw.limits,
            max_nodes=raw.limits.max_nodes + len(names) + 1,
            max_bindings=raw.limits.max_bindings + 2 * len(names) + 1,
            max_sequence_edges=raw.limits.max_sequence_edges + len(names) + 2,
        ),
        dynamic_limits=replace(
            raw.dynamic_limits,
            max_activation_occurrences=raw.dynamic_limits.max_activation_occurrences + len(names) + 1,
        ),
    )
    return extended, tuple(capabilities), tuple(policies), ids


def detection_slots(prepared: PreparedPlan, ids: dict[str, NodeId]) -> dict[str, int]:
    return {
        name: slot.index
        for name, node in ids.items()
        if name != "entry"
        for slot in prepared.reservation_recipe
        if slot.template == node
    }


def reparent_detection_policies(
    capabilities: tuple[ImplementationCapability, ...], policies: tuple[OperationExecutionPolicy, ...], entry: NodeId
) -> tuple[tuple[ImplementationCapability, ...], tuple[OperationExecutionPolicy, ...]]:
    remapped_caps, remapped_policies = [], []
    for cap, policy in zip(capabilities, policies, strict=True):
        cap = replace(cap, resource_lifetime="stateless")
        remapped_caps.append(cap)
        remapped_policies.append(
            replace(
                policy,
                delegated=None,
                submission_owner=entry,
                implementations=tuple(
                    replace(implementation, capability=cap) for implementation in policy.implementations
                ),
            )
        )
    return tuple(remapped_caps), tuple(remapped_policies)


async def complete_detection_entry(
    requests: DelegatedRequests,
    inputs: tuple[AssociationInput, ...],
    outcome: str = "ok",
    entry_slot: int | None = None,
) -> None:
    if not inputs and entry_slot is not None:
        item = await requests.enter_operation(entry_slot, None)
        if item is None:
            reject(EffectCode.REQUEST_CAUSALITY)
        inputs = (item,)
    await requests.complete_operations(
        tuple(
            (
                cast(SemanticAssociation, item.association),
                completed(item.association, outcome=outcome),
            )
            for item in inputs
        )
    )


async def complete_empty_stage(requests: DelegatedRequests, bound: list[AssociationInput]) -> None:
    for item in bound:
        outcome = await requests.complete_operation(
            cast(SemanticAssociation, item.association),
            completed(item.association),
        )
        if outcome.condition != "result" or outcome.category != "success":
            raise AnonymizerWorkflowError("Empty detection semantic result was rejected")


@dataclass
class DetectionBinding(StageBinding):
    """The admitted graph's detection children for one builtin detection lowering."""

    rejected = "Detection semantic result was rejected"

    requests: DelegatedRequests
    source: WorkflowSourceFrame
    slots: dict[str, int]
    receiver: EntityDetectionWorkflow
    arguments: dict[str, Any]
    entry: pd.DataFrame
    entry_slot: int | None = None

    def lower(self) -> EntityDetectionResult:
        """Run the builtin detection stages against the graph's children; called only by the admitted graph."""
        from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow

        borrow = source_worker_borrow()
        receiver, frame, arguments = self.receiver, self.entry, self.arguments
        if self.entry_slot is not None:
            self.call(self.complete_empty_entry())
        borrow.authorize_sources_sync(self.source.associations)
        label_config = resolve_entity_label_config(
            entity_labels=arguments["entity_labels"],
            excluded_entity_labels=arguments["excluded_entity_labels"],
            entity_label_examples=arguments["entity_label_examples"],
        )
        selected = frame
        positions = tuple(range(len(frame)))
        failures = []
        native = native_adapter(receiver._adapter)
        common = {
            key: arguments[key]
            for key in (
                "model_configs",
                "selected_models",
                "gliner_detection_threshold",
                "entity_labels",
                "excluded_entity_labels",
                "entity_label_examples",
                "data_summary",
                "preview_num_records",
            )
        }
        common["_resolved_label_config"] = label_config
        while True:
            active = tuple(
                self.source.associations[index] for index in positions if index < len(self.source.associations)
            )
            ready = self.call(self.ready(active))
            if not ready:
                break
            name, bound = ready
            if name == "final" and not self.source.associations:
                selected = selected.copy()
                if COL_DETECTED_ENTITIES in selected:
                    selected[COL_FINAL_ENTITIES] = selected[COL_DETECTED_ENTITIES].apply(
                        lambda raw: _materialize_final_entities(raw, allowed_labels=None)
                    )
                    if arguments["compute_grouped_entities"] is not False:
                        selected[COL_ENTITIES_BY_VALUE] = selected[COL_FINAL_ENTITIES].apply(_build_entities_by_value)
                self.call(complete_empty_stage(self.requests, bound))
                return EntityDetectionResult(selected, failures, ())
            if name == "final":
                selected = selected.copy()
                canonical = []
                for item in bound:
                    detected = next(port.value for port in item.inputs if port.port == "in_detected")
                    value = _materialize_final_entities(
                        json.loads(detected.text),
                        allowed_labels=set(label_config.labels) if label_config.strict_labels else None,
                        excluded_entity_labels=set(arguments["excluded_entity_labels"] or ()),
                    )
                    canonical.append(value)
                self.call(self.complete(bound, selected, canonical))
                final = self.call(self.results(active))
                if COL_DETECTED_ENTITIES in selected:
                    selected[COL_FINAL_ENTITIES] = [
                        json.loads(
                            next(
                                cast(TextArtifactValue, port.value).text
                                for port in item.outputs
                                if port.port == "entities"
                            )
                        )
                        for item in final
                    ]
                    if arguments["compute_grouped_entities"] is not False:
                        selected[COL_ENTITIES_BY_VALUE] = selected[COL_FINAL_ENTITIES].apply(_build_entities_by_value)
                return EntityDetectionResult(selected, failures, positions)
            stage_source = (
                WorkflowSourceFrame.capture(
                    selected, associations=active, invocation_bound=True, source_limit=len(active)
                )
                if native
                else arguments.get("source_frame")
                if name == "core"
                else None
            )
            if name == "core":
                result = receiver.detect_and_validate_entities(
                    selected,
                    **common,
                    **{
                        key: arguments[key]
                        for key in (
                            "validation_max_entities_per_call",
                            "validation_excerpt_window_chars",
                            "validation_single_chunk_full_text",
                        )
                    },
                    **({"source_frame": stage_source} if stage_source is not None else {}),
                )
            else:
                result = receiver.identify_latent_entities(
                    selected,
                    **common,
                    privacy_goal=arguments["privacy_goal"],
                    **({"source_frame": stage_source} if stage_source is not None else {}),
                )
            if not self.source.associations and len(result.dataframe):
                raise AnonymizerWorkflowError("Empty detection returned unexpected sources")
            surviving = resolve_source_positions(selected, result.dataframe, result.source_positions)
            expected_text = [self.source.dataframe[COL_TEXT].iloc[positions[index]] for index in surviving]
            returned_text = result.dataframe[COL_TEXT].tolist()
            if len(returned_text) != len(expected_text) or not all(
                same_source_text(returned, source)
                for returned, source in zip(returned_text, expected_text, strict=True)
            ):
                raise AnonymizerWorkflowError("Detection stage changed its authenticated source text")
            failures.extend(result.failed_records)
            self.call(
                self.complete_stage(bound, result.dataframe, surviving)
                if self.source.associations
                else complete_empty_stage(self.requests, bound)
            )
            positions = tuple(positions[index] for index in surviving)
            selected = result.dataframe
        if len(selected) and self.source.associations:
            raise AnonymizerWorkflowError("Detection has no admitted terminal result")
        return EntityDetectionResult(selected, failures, positions)

    def call(self, operation: Any) -> Any:

        return drive(operation)

    async def complete_empty_entry(self) -> None:
        if self.entry_slot is None:
            reject(EffectCode.FOREIGN_OWNER)
        item = await self.requests.enter_operation(self.entry_slot, None)
        if item is None:
            reject(EffectCode.REQUEST_CAUSALITY)
        await complete_empty_stage(self.requests, [item])

    async def ready(self, associations: tuple[Any, ...]) -> tuple[str, list[AssociationInput]] | None:
        if not associations and self.source.associations:
            return None
        entered = await self.enter_ready(associations, whole=True)
        return None if entered is None else (entered[0], list(entered[1]))

    def not_ready(self, *, empty: bool) -> None:
        if empty:
            reject(EffectCode.REQUEST_CAUSALITY)
        raise AnonymizerWorkflowError("Detection stage is not ready")

    def partially_ready(self) -> None:
        raise AnonymizerWorkflowError("Detection stage has incomplete source readiness")

    async def complete_stage(
        self, bound: list[AssociationInput], frame: pd.DataFrame, positions: tuple[int, ...]
    ) -> None:
        rows = {position: ordinal for ordinal, position in enumerate(positions)}
        results = []
        for index, item in enumerate(bound):
            if index not in rows:
                results.append(
                    (cast(SemanticAssociation, item.association), LocalFailure(failure="implementation_exception"))
                )
                continue
            value = (
                EntitiesSchema.from_raw(frame[COL_DETECTED_ENTITIES].iloc[rows[index]]).model_dump_json()
                if COL_DETECTED_ENTITIES in frame
                else EntitiesSchema().model_dump_json()
            )
            results.append((cast(SemanticAssociation, item.association), self.publication(item, "detected", value)))
        await self.complete_results(results)

    async def complete(self, bound: list[AssociationInput], frame: pd.DataFrame, values: list[dict[str, Any]]) -> None:
        del frame
        await self.complete_results(
            (cast(SemanticAssociation, item.association), self.publication(item, "entities", json.dumps(value)))
            for item, value in zip(bound, values, strict=True)
        )

    async def publish(self, item: AssociationInput, port: str, value: str) -> None:
        result = await self.requests.complete_operation(
            cast(SemanticAssociation, item.association),
            self.publication(item, port, value),
        )
        self.check_outcomes((result,))
        if result.condition != "result" or result.category != "success":
            raise AnonymizerWorkflowError(self.rejected)

    def publication(self, item: AssociationInput, port: str, value: str) -> LocalCompleted:
        source = next((port for port in item.inputs if port.port == "in_source"), None)
        return completed(
            item.association,
            (
                *(
                    (PortArtifact(port="source", artifact_type=_TEXT, artifact=None, value=source.value),)
                    if source is not None
                    else ()
                ),
                PortArtifact(port=port, artifact_type=_ENTITIES, artifact=None, value=TextArtifactValue(text=value)),
            ),
        )

    async def results(self, associations: tuple[Any, ...]) -> list[AssociationResult]:
        results = await self.requests.operation_results(self.slots["final"], associations)
        if any(result is None for result in results):
            raise AnonymizerWorkflowError("Detection final result is missing")
        return cast(list[AssociationResult], list(results))


def detection_arguments(
    receiver: EntityDetectionWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]
) -> dict[str, Any]:
    """Normalize builtin detection arguments once, at entry, with the release signature's defaults."""
    normalized = signature(_DETECTION_RUN).bind(receiver, frame, **arguments)
    normalized.apply_defaults()
    values = {key: value for key, value in normalized.arguments.items() if key not in {"self", "dataframe"}}
    if values["tag_latent_entities"] and values["privacy_goal"] is None:
        raise ValueError("privacy_goal is required when tag_latent_entities=True (rewrite mode)")
    if values["source_frame"] is not None and not values["source_frame"].matches(frame):
        raise AnonymizerWorkflowError("Workflow source frame does not match submitted data")
    return values


def lower_detection(
    receiver: EntityDetectionWorkflow,
    frame: pd.DataFrame,
    arguments: dict[str, Any],
    requests: DelegatedRequests,
    source: WorkflowSourceFrame,
    slots: dict[str, int],
    entry_slot: int | None = None,
) -> EntityDetectionResult:
    """Lower builtin detection directly into the admitted graph's detection children."""
    values = detection_arguments(receiver, frame, arguments)
    with stage_timer(
        "EntityDetectionWorkflow.run", input_row_count=len(frame), tag_latent_entities=values["tag_latent_entities"]
    ) as measurement:
        result = DetectionBinding(requests, source, slots, receiver, values, frame, entry_slot).lower()
        measurement.update(output_row_count=len(result.dataframe), failed_record_count=len(result.failed_records))
    return result


def run_detection(
    receiver: EntityDetectionWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]
) -> EntityDetectionResult:
    """The builtin runner's own entry: one standalone invocation, also when a caller's runner delegates to it."""
    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW

    arguments = detection_arguments(receiver, frame, arguments)
    token = _WORKER_BORROW.set(None)
    try:
        return run_direct_detection(receiver, frame, arguments)
    finally:
        _WORKER_BORROW.reset(token)


def run_direct_detection(
    receiver: EntityDetectionWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]
) -> EntityDetectionResult:
    from anonymizer.engine.harness.kernel import _run_sync_profile, project_detection_redact

    if frame.empty or arguments.get("preview_num_records") == 0:
        return run_empty_detection(receiver, frame, arguments)
    captured = []

    async def operation(pool: SubmitterPool, handoff: Callable[..., Any]) -> ReplacementResult:
        result = await execute_detection(receiver, frame, arguments, pool, captured, handoff)
        return project_detection_redact(result)

    projected = _run_sync_profile(operation)
    result = EntityDetectionResult(projected.dataframe, projected.failed_records, captured[0].source_positions)
    attach_execution(result, projected._execution)
    return result


async def execute_detection(
    receiver: EntityDetectionWorkflow,
    frame: pd.DataFrame,
    arguments: dict[str, Any],
    pool: SubmitterPool,
    captured: list[EntityDetectionResult],
    handoff: Callable[..., Any],
) -> ExecutionResult:
    from anonymizer.engine.harness.kernel import _execute_frame_plan, _frame_implementation, _prepare_frame_plan

    preview = arguments.get("preview_num_records")
    selected = frame if preview is None else frame.iloc[:preview]
    texts = tuple(source_text(value) for value in selected[COL_TEXT])
    graph = DataGraph.new()
    targets = []
    for text in texts:
        graph, target = graph.add_text(text)
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
    owner = NodeId.new(workflow=workflow)
    operation = _operation_spec(
        name="detection-lowering",
        inputs=(InputPort(name="source", artifact_type=_TEXT),),
        outputs=(OutputPort(name="returned_source", artifact_type=_TEXT),),
        output_dependencies=(
            OutputDependency(output="returned_source", inputs=frozenset({"source"}), identity_input="source"),
        ),
        outcomes=(
            _outcome_spec(
                name="ok",
                category="success",
                produced_ports=frozenset({"returned_source"}),
                ceiling=ResourceCeiling(
                    max_activations=1, max_model_requests=None, max_input_bytes=None, max_output_bytes=None
                ),
            ),
        ),
    )
    raw = _compile_graph(
        workflow=workflow,
        interface=operation,
        nodes=(OperationNode(id=owner, operation=operation),),
        input_bindings=(
            InputBinding(source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=owner, port="source")),
        ),
        output_bindings=(
            OutputBinding(
                source=NodeOutputRef(node=owner, port="returned_source"),
                destination=WorkflowOutputRef(port="returned_source"),
            ),
        ),
        outcome_bindings=(
            OutcomeBinding(
                source=NodeOutcomeRef(node=owner, outcome="ok"), destination=WorkflowOutcomeRef(outcome="ok")
            ),
        ),
        limits=WorkflowLimits(
            max_nodes=1,
            max_bindings=3,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        ),
    )
    configuration = detection_configuration(arguments)
    raw, child_caps, child_policies, ids = extend_detection_graph(
        raw, owner, configuration, latent=arguments["tag_latent_entities"]
    )
    projection_ports = {"final_entities": COL_FINAL_ENTITIES}
    if arguments["compute_grouped_entities"] is not False:
        projection_ports["grouped_entities"] = COL_ENTITIES_BY_VALUE
    operation = replace(
        operation,
        inputs=(*operation.inputs, InputPort(name="in_entities", artifact_type=_ENTITIES)),
        outputs=(*operation.outputs, *(OutputPort(name=name, artifact_type=_ENTITIES) for name in projection_ports)),
        output_dependencies=(
            *operation.output_dependencies,
            *(
                OutputDependency(
                    output=name,
                    inputs=frozenset({"in_entities"}),
                    identity_input="in_entities" if name == "final_entities" else None,
                )
                for name in projection_ports
            ),
        ),
        outcomes=tuple(
            replace(outcome, produced_ports=outcome.produced_ports | frozenset(projection_ports))
            for outcome in operation.outcomes
        ),
    )
    raw = _compile_graph(
        workflow=raw.workflow,
        interface=raw.interface,
        nodes=tuple(
            replace(node, operation=operation) if isinstance(node, OperationNode) and node.id == owner else node
            for node in raw.nodes
        ),
        input_bindings=(
            *raw.input_bindings,
            InputBinding(
                source=NodeOutputRef(node=ids["final"], port="entities"),
                destination=NodeInputRef(node=owner, port="in_entities"),
            ),
        ),
        output_bindings=tuple(raw.output_bindings),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=tuple(raw.sequence),
        protection=tuple(raw.protection_requirements),
        limits=replace(raw.limits, max_bindings=raw.limits.max_bindings + 1),
        dynamic_limits=raw.dynamic_limits,
    )
    cap, policy = _frame_implementation(owner, operation, configuration)
    parent_caps, parent_policies = reparent_detection_policies((cap,), (policy,), ids["entry"])
    capabilities = (*parent_caps, *child_caps)
    prepared = _prepare_frame_plan(
        data,
        raw,
        capabilities,
        tuple(BoundInput(target=target, source=target, port="source", artifact_type=_TEXT) for target in targets),
        invocation_owned=not targets,
    )
    slots = detection_slots(prepared, ids)
    loop = asyncio.get_running_loop()

    def callback(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        source = requests.bind_frame(frame, targets=tuple(targets), source_limit=size)
        drive(complete_detection_entry(requests, inputs))
        result = DetectionBinding(requests, source, slots, receiver, arguments, frame).lower()
        captured.append(result)
        successful = tuple(source.associations[index] for index in result.source_positions)
        terminal = []

        async def finish() -> None:
            owner_slot = next(slot.index for slot in prepared.reservation_recipe if slot.template == owner)
            for association in successful:
                item = await requests.enter_operation(owner_slot, association)
                if item is None:
                    raise AnonymizerWorkflowError("Detection projection is not ready")
                terminal.append(item.association)
                await requests.complete_operation(
                    cast(SemanticAssociation, item.association),
                    completed(
                        item.association,
                        (
                            PortArtifact(
                                port="returned_source",
                                artifact_type=_TEXT,
                                artifact=None,
                                value=item.inputs[0].value,
                            ),
                            *(
                                PortArtifact(
                                    port=name,
                                    artifact_type=_ENTITIES,
                                    artifact=None,
                                    value=TextArtifactValue(
                                        text=next(
                                            cast(TextArtifactValue, port.value).text
                                            for port in item.inputs
                                            if port.port == "in_entities"
                                        )
                                        if name == "final_entities"
                                        else json.dumps(
                                            _build_entities_by_value(
                                                json.loads(
                                                    next(
                                                        cast(TextArtifactValue, port.value).text
                                                        for port in item.inputs
                                                        if port.port == "in_entities"
                                                    )
                                                )
                                            )
                                        )
                                    ),
                                )
                                for name in projection_ports
                            ),
                        ),
                    ),
                )

        drive(finish())
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
                associations=tuple(terminal),
                text_ports=((COL_TEXT, "returned_source"),),
                failed_records=tuple(result.failed_records),
                opaque_source=frame,
                json_ports=tuple(
                    (column, name) for name, column in projection_ports.items() if column in result.dataframe
                ),
            ),
        )

    def dispatch(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        context = copy_context()
        return handoff(lambda: context.run(callback, inputs, requests))

    return await _execute_frame_plan(
        prepared=prepared,
        capabilities=capabilities,
        policies=(*parent_policies, *child_policies),
        callback=dispatch,
        pool=pool,
        borrowed_loop=loop,
        limits=_execution_limits(
            max_local_in_flight=1,
            max_runtime_artifacts=max(1, 12 * size),
        ),
        ingress_limits=_ingress_limits(
            max_associations=max(1, size),
            max_outputs_per_association=4,
            max_assessments=1,
            max_depth=8,
        ),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_assessment_facts=0,
            max_port_facts=24 * size,
            max_provenance_edges=40 * size,
        ),
    )


def run_empty_detection(
    receiver: EntityDetectionWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]
) -> EntityDetectionResult:
    """Preserve zero-cardinality adapter callbacks under nonqualifying admission."""
    from anonymizer.engine.harness.frames import InvocationFrame
    from anonymizer.engine.harness.kernel import (
        _execute_frame_plan,
        _frame_implementation,
        _prepare_frame_plan,
        _run_sync_profile,
        project_detection_redact,
    )

    configuration = detection_configuration(arguments)
    workflow = WorkflowId.new()
    node = NodeId.new(workflow=workflow)
    operation = _operation_spec(
        name="empty-detection-lowering",
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
    raw = _compile_graph(
        workflow=workflow,
        interface=operation,
        nodes=(OperationNode(id=node, operation=operation),),
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
    raw, child_caps, child_policies, ids = extend_detection_graph(
        raw, node, configuration, latent=arguments["tag_latent_entities"], source_free=True
    )
    cap, policy = _frame_implementation(node, operation, configuration, aggregate=True)
    parent_caps, parent_policies = reparent_detection_policies((cap,), (policy,), ids["entry"])
    capabilities = (*parent_caps, *child_caps)
    policies = (*parent_policies, *child_policies)
    data = _validate_targets(
        DataGraph.new(),
        targets=(),
        limits=DataLimits(max_datums=0, max_targets=0, max_text_bytes=0, max_declarations=0, max_group_members=0),
    )
    prepared = _prepare_frame_plan(data, raw, capabilities, (), invocation_owned=True)

    def callback(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        del inputs
        if requests.cancelled:
            reject(EffectCode.FOREIGN_OWNER)
        source = requests.bind_frame(frame, targets=(), source_limit=0)
        binding = DetectionBinding(
            requests,
            source,
            detection_slots(prepared, ids),
            receiver,
            arguments,
            frame,
            entry_slot=next(slot.index for slot in prepared.reservation_recipe if slot.template == ids["entry"]),
        )
        selected = binding.lower()
        owner_slot = next(slot.index for slot in prepared.reservation_recipe if slot.template == node)
        item = drive(requests.enter_operation(owner_slot, None))
        if item is None:
            reject(EffectCode.REQUEST_CAUSALITY)
        drive(complete_empty_stage(requests, [item]))
        return LocalCompleted(
            results=(),
            frame=InvocationFrame(
                selected.dataframe, selected.dataframe, tuple(selected.failed_records), (), 0.0, 0.0, None
            ),
        )

    async def execute(pool: SubmitterPool, handoff: Callable[..., Any]) -> ReplacementResult:
        def dispatch(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
            context = copy_context()
            return handoff(lambda: context.run(callback, inputs, requests))

        result = await _execute_frame_plan(
            prepared=prepared,
            capabilities=capabilities,
            policies=policies,
            callback=dispatch,
            pool=pool,
            borrowed_loop=asyncio.get_running_loop(),
            limits=_execution_limits(
                max_local_in_flight=1,
                max_runtime_artifacts=1,
            ),
            ingress_limits=_ingress_limits(
                max_associations=1,
                max_outputs_per_association=1,
                max_assessments=1,
                max_depth=8,
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
        return project_detection_redact(result)

    projected = _run_sync_profile(execute)
    result = EntityDetectionResult(projected.dataframe, projected.failed_records, ())
    attach_execution(result, projected._execution)
    return result
