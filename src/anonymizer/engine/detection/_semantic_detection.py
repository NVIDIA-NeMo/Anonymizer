# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detection stage lowerings and terminal projection owned by admitted graph control."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, replace
from inspect import signature
from threading import Lock
from typing import Any, cast

import pandas as pd
from pydantic import TypeAdapter

from anonymizer.engine.constants import COL_DETECTED_ENTITIES, COL_ENTITIES_BY_VALUE, COL_FINAL_ENTITIES, COL_TEXT
from anonymizer.engine.detection._frame_publication import FramePublication
from anonymizer.engine.detection.detection_workflow import (
    _DETECTION_RUN,
    EntityDetectionResult,
    EntityDetectionWorkflow,
    _build_entities_by_value,
    _materialize_final_entities,
)
from anonymizer.engine.detection.entity_label_examples import resolve_entity_label_config
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests
from anonymizer.engine.graph_sdk.blocking import BlockingPool
from anonymizer.engine.graph_sdk.capabilities import FrozenConfig, ImplementationCapability
from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import (
    AssessmentLimits,
    ExecutionLimits,
    ExecutionResult,
    IngressLimits,
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
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.replace_runner import ReplacementResult
from anonymizer.engine.row_partitioning import resolve_source_positions
from anonymizer.engine.schemas import EntitiesSchema
from anonymizer.graph.workflow import (
    ArtifactType,
    ChoiceBranch,
    ChoiceDecl,
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
    SequenceEdge,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    compile_graph,
)
from anonymizer.interface.errors import AnonymizerWorkflowError

_TEXT = ArtifactType(name="anonymizer.text", revision=1)
_ENTITIES = ArtifactType(name="anonymizer.entity-observation-json", revision=1)
_BINDING: ContextVar[DetectionBinding | None] = ContextVar("detection_binding", default=None)


def detection_configuration(arguments: dict[str, Any]) -> FrozenConfig:
    from anonymizer.engine.graph_sdk.capabilities import ConfigField, ConfigText

    normalized = signature(_DETECTION_RUN).bind(None, None, **arguments)
    normalized.apply_defaults()
    values = {
        key: value
        for key, value in normalized.arguments.items()
        if key not in {"self", "dataframe", "source_frame", "request_bridge"}
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
    optional: bool = False,
    source_free: bool = False,
) -> tuple[Graph, tuple[ImplementationCapability, ...], tuple[OperationExecutionPolicy, ...], dict[str, NodeId]]:
    """Compose detection children into an existing submission owner's graph."""
    from anonymizer.engine.detection._opaque_redact import _frame_implementation

    names = ("core", "latent", "final") if latent else ("core", "final")
    ids = {name: NodeId.new(workflow=raw.workflow) for name in (*names, "entry")}
    entry = OperationSpec(
        name="detection-entry",
        inputs=() if source_free else (InputPort(name="source", artifact_type=_TEXT),),
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
    if optional:
        entry = replace(entry, outcomes=(*entry.outcomes, replace(entry.outcomes[0], name="bypass")))
    entry_cap, entry_policy = _frame_implementation(ids["entry"], entry, configuration)
    nodes = [OperationNode(id=ids["entry"], operation=entry)]
    bindings = [
        InputBinding(source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=ids["entry"], port="source"))
    ]
    if source_free:
        bindings = []
    edges = [SequenceEdge(before=ids["entry"], after=ids["core"]), SequenceEdge(before=ids["final"], after=owner)]
    if optional:
        edges.append(SequenceEdge(before=ids["entry"], after=owner))
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
        operation = OperationSpec(
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
                OutcomeSpec(
                    name="ok",
                    category="success",
                    produced_ports=frozenset(outputs),
                    context=frozenset(),
                    evidence=frozenset(),
                    state_effects=frozenset(),
                    model_requirements=frozenset(),
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
    extended = compile_graph(
        workflow=raw.workflow,
        interface=interface,
        nodes=(
            *raw.nodes,
            *nodes,
            *(
                (
                    ChoiceDecl(
                        selector=ids["entry"],
                        branches=(
                            ChoiceBranch(outcomes=frozenset({"ok"}), members=frozenset(ids[name] for name in names)),
                        ),
                    ),
                )
                if optional
                else ()
            ),
        ),
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
            max_choices=raw.limits.max_choices + int(optional),
            max_branch_members=raw.limits.max_branch_members + (len(names) if optional else 0),
            max_choice_states=raw.limits.max_choice_states * (2 if optional else 1),
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
                LocalCompleted(
                    results=(
                        AssociationResult(
                            association=item.association,
                            outcome=outcome,
                            outputs=(),
                            consumed_context_ports=frozenset(),
                        ),
                    )
                ),
            )
            for item in inputs
        )
    )


@dataclass
class DetectionBinding:
    requests: DelegatedRequests
    source: WorkflowSourceFrame
    slots: dict[str, int]
    loop: asyncio.AbstractEventLoop
    receiver: EntityDetectionWorkflow
    arguments: dict[str, Any]
    entry: pd.DataFrame
    adapter: object
    claim: Lock
    configuration: FrozenConfig
    lowerings: tuple[Callable[..., EntityDetectionResult], Callable[..., EntityDetectionResult]]
    entry_inputs: tuple[AssociationInput, ...] | None = None
    entry_slot: int | None = None

    def run(
        self, receiver: EntityDetectionWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]
    ) -> EntityDetectionResult:
        from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow

        borrow = source_worker_borrow()
        supplied_source = arguments.get("source_frame")
        if (
            supplied_source is not None
            and supplied_source.associations
            and supplied_source.associations != self.source.associations
        ):
            reject(EffectCode.FOREIGN_OWNER)
        if (
            receiver is not self.receiver
            or receiver._adapter is not self.adapter
            or frame is not self.entry
            or not self.source.matches(frame)
            or detection_configuration(arguments) != self.configuration
            or arguments.get("privacy_goal") is not self.arguments.get("privacy_goal")
            or arguments.get("source_frame") is not self.arguments.get("source_frame")
            or arguments.get("request_bridge") is not self.arguments.get("request_bridge")
            or borrow is not self.requests
            or borrow.owner.loop is not self.loop
            or borrow.owner.closed
            or not self.claim.acquire(blocking=False)
        ):
            reject(EffectCode.FOREIGN_OWNER)
        self.authenticate_stage(receiver, arguments)
        if self.entry_inputs is not None:
            self.call(complete_detection_entry(self.requests, self.entry_inputs))
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
        native = getattr(receiver._adapter.run_workflow, "__func__", None) is NddAdapter.run_workflow
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
        core_lowering, latent_lowering = self.lowerings
        while True:
            self.authenticate_stage(receiver, arguments)
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
                self.call(self.complete_empty_stage(bound))
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
                result = core_lowering(
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
                    **(
                        {"request_bridge": arguments["request_bridge"]}
                        if arguments["request_bridge"] is not None
                        else {}
                    ),
                )
            else:
                result = latent_lowering(
                    selected,
                    **common,
                    privacy_goal=arguments["privacy_goal"],
                    **({"source_frame": stage_source} if stage_source is not None else {}),
                )
            self.authenticate_stage(receiver, arguments)
            if not self.source.associations and len(result.dataframe):
                raise AnonymizerWorkflowError("Empty detection returned unexpected sources")
            surviving = resolve_source_positions(selected, result.dataframe, result.source_positions)
            expected_text = [self.source.dataframe[COL_TEXT].iloc[positions[index]] for index in surviving]
            if result.dataframe[COL_TEXT].tolist() != expected_text:
                raise AnonymizerWorkflowError("Detection stage changed its authenticated source text")
            failures.extend(result.failed_records)
            self.call(
                self.complete_stage(bound, result.dataframe, surviving)
                if self.source.associations
                else self.complete_empty_stage(bound)
            )
            positions = tuple(positions[index] for index in surviving)
            selected = result.dataframe
        if len(selected) and self.source.associations:
            raise AnonymizerWorkflowError("Detection has no admitted terminal result")
        return EntityDetectionResult(selected, failures, positions)

    def authenticate_stage(self, receiver: EntityDetectionWorkflow, arguments: dict[str, Any]) -> None:
        if (
            receiver is not self.receiver
            or receiver._adapter is not self.adapter
            or detection_configuration(arguments) != self.configuration
            or receiver.detect_and_validate_entities != self.lowerings[0]
            or receiver.identify_latent_entities != self.lowerings[1]
            or not self.source.matches(self.entry)
        ):
            reject(EffectCode.FOREIGN_OWNER)

    def call(self, operation: Any) -> Any:
        from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow

        return asyncio.run_coroutine_threadsafe(operation, source_worker_borrow().loop).result()

    async def complete_empty_entry(self) -> None:
        if self.entry_slot is None:
            reject(EffectCode.FOREIGN_OWNER)
        item = await self.requests.enter_operation(self.entry_slot, None)
        if item is None:
            reject(EffectCode.REQUEST_CAUSALITY)
        await self.complete_empty_stage([item])

    async def complete_empty_stage(self, bound: list[AssociationInput]) -> None:
        for item in bound:
            outcome = await self.requests.complete_operation(
                cast(SemanticAssociation, item.association),
                LocalCompleted(
                    results=(
                        AssociationResult(
                            association=item.association, outcome="ok", outputs=(), consumed_context_ports=frozenset()
                        ),
                    )
                ),
            )
            if outcome.condition != "result" or outcome.category != "success":
                raise AnonymizerWorkflowError("Empty detection semantic result was rejected")

    async def ready(self, associations: tuple[Any, ...]) -> tuple[str, list[AssociationInput]] | None:
        if not self.source.associations:
            ready = await self.requests.ready_operations(None)
            for name, slot in self.slots.items():
                if slot in ready:
                    item = await self.requests.enter_operation(slot, None)
                    if item is None:
                        reject(EffectCode.REQUEST_CAUSALITY)
                    return name, [item]
            return None
        readiness = await self.requests.ready_operations_many(associations)
        for name, slot in self.slots.items():
            ready = tuple(
                association for association, slots in zip(associations, readiness, strict=True) if slot in slots
            )
            if ready:
                if len(ready) != len(associations):
                    raise AnonymizerWorkflowError("Detection stage has incomplete source readiness")
                entered = await self.requests.enter_operations(slot, ready)
                if any(item is None for item in entered):
                    raise AnonymizerWorkflowError("Detection stage is not ready")
                return name, cast(list[AssociationInput], list(entered))
        return None

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
        outcomes = await self.requests.complete_operations(tuple(results))
        self.require_publication_capacity(outcomes)
        if len(outcomes) != len(results) or any(
            outcome.condition != "result" or outcome.category != "success"
            for (_, result), outcome in zip(results, outcomes)
            if type(result) is LocalCompleted
        ):
            raise AnonymizerWorkflowError("Detection semantic result was rejected")

    async def complete(self, bound: list[AssociationInput], frame: pd.DataFrame, values: list[dict[str, Any]]) -> None:
        del frame
        results = tuple(
            (cast(SemanticAssociation, item.association), self.publication(item, "entities", json.dumps(value)))
            for item, value in zip(bound, values, strict=True)
        )
        outcomes = await self.requests.complete_operations(results)
        self.require_publication_capacity(outcomes)
        if len(outcomes) != len(results) or any(
            result.condition != "result" or result.category != "success" for result in outcomes
        ):
            raise AnonymizerWorkflowError("Detection semantic result was rejected")

    async def publish(self, item: AssociationInput, port: str, value: str) -> None:
        result = await self.requests.complete_operation(
            cast(SemanticAssociation, item.association),
            self.publication(item, port, value),
        )
        self.require_publication_capacity((result,))
        if result.condition != "result" or result.category != "success":
            raise AnonymizerWorkflowError("Detection semantic result was rejected")

    def require_publication_capacity(self, outcomes: tuple[Any, ...]) -> None:
        if self.arguments["request_bridge"] is not None and any(
            outcome.condition == "artifact_limit_exhausted" for outcome in outcomes
        ):
            raise AnonymizerWorkflowError("Detection/Redact execution did not qualify all returned sources")

    def publication(self, item: AssociationInput, port: str, value: str) -> LocalCompleted:
        source = next((port for port in item.inputs if port.port == "in_source"), None)
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome="ok",
                    consumed_context_ports=frozenset(),
                    outputs=(
                        *(
                            (PortArtifact(port="source", artifact_type=_TEXT, artifact=None, value=source.value),)
                            if self.arguments["request_bridge"] is None and source is not None
                            else ()
                        ),
                        PortArtifact(
                            port=port, artifact_type=_ENTITIES, artifact=None, value=TextArtifactValue(text=value)
                        ),
                    ),
                ),
            )
        )

    async def results(self, associations: tuple[Any, ...]) -> list[AssociationResult]:
        results = await self.requests.operation_results(self.slots["final"], associations)
        if any(result is None for result in results):
            raise AnonymizerWorkflowError("Detection final result is missing")
        return cast(list[AssociationResult], list(results))


@contextmanager
def bind_detection(
    receiver: EntityDetectionWorkflow,
    frame: pd.DataFrame,
    arguments: dict[str, Any],
    requests: DelegatedRequests,
    source: WorkflowSourceFrame,
    slots: dict[str, int],
    loop: asyncio.AbstractEventLoop,
    configuration: FrozenConfig | None = None,
    adapter: object | None = None,
    lowerings: tuple[Callable[..., EntityDetectionResult], Callable[..., EntityDetectionResult]] | None = None,
    entry_inputs: tuple[AssociationInput, ...] | None = None,
    entry_slot: int | None = None,
) -> Iterator[DetectionBinding]:
    normalized = signature(_DETECTION_RUN).bind(receiver, frame, **arguments)
    normalized.apply_defaults()
    values = {key: value for key, value in normalized.arguments.items() if key not in {"self", "dataframe"}}
    binding = DetectionBinding(
        requests,
        source,
        slots,
        loop,
        receiver,
        values,
        frame,
        receiver._adapter if adapter is None else adapter,
        Lock(),
        detection_configuration(values) if configuration is None else configuration,
        (receiver.detect_and_validate_entities, receiver.identify_latent_entities) if lowerings is None else lowerings,
        entry_inputs,
        entry_slot,
    )
    token = _BINDING.set(binding)
    try:
        yield binding
    finally:
        _BINDING.reset(token)


def run_detection(
    receiver: EntityDetectionWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]
) -> EntityDetectionResult:
    if arguments["tag_latent_entities"] and arguments["privacy_goal"] is None:
        raise ValueError("privacy_goal is required when tag_latent_entities=True (rewrite mode)")
    if arguments["request_bridge"] is not None and (
        arguments["source_frame"] is None or arguments["tag_latent_entities"]
    ):
        raise ValueError("Graph request binding requires a source frame and detection-only workflow")
    supplied_source = arguments.get("source_frame")
    if supplied_source is not None and not supplied_source.matches(frame):
        raise AnonymizerWorkflowError("Workflow source frame does not match submitted data")
    binding = _BINDING.get()
    if binding is not None:
        if (
            supplied_source is not None
            and supplied_source.associations
            and supplied_source.associations != binding.source.associations
        ):
            reject(EffectCode.FOREIGN_OWNER)
        return binding.run(receiver, frame, arguments)
    return run_direct_detection(receiver, frame, arguments)


def run_direct_detection(
    receiver: EntityDetectionWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]
) -> EntityDetectionResult:
    from anonymizer.engine.detection._graph_profile import _run_sync_profile, project_detection_redact

    if frame.empty or arguments.get("preview_num_records") == 0:
        return run_empty_detection(receiver, frame, arguments)
    captured = []
    opaque = getattr(receiver._adapter.run_workflow, "__func__", None) is not NddAdapter.run_workflow

    async def operation(pool: BlockingPool, handoff: Callable[..., Any]) -> ReplacementResult:
        result = await execute_detection(receiver, frame, arguments, pool, captured, handoff if opaque else None)
        return project_detection_redact(result)

    projected = _run_sync_profile(operation, caller_callbacks=opaque)
    result = EntityDetectionResult(projected.dataframe, projected.failed_records, captured[0].source_positions)
    object.__setattr__(result, "_execution", projected._execution)
    return result


async def execute_detection(
    receiver: EntityDetectionWorkflow,
    frame: pd.DataFrame,
    arguments: dict[str, Any],
    pool: BlockingPool,
    captured: list[EntityDetectionResult],
    handoff: Callable[..., Any] | None,
) -> ExecutionResult:
    from anonymizer.engine.detection._opaque_redact import (
        _execute_frame_plan,
        _frame_implementation,
        _prepare_frame_plan,
    )

    expected_adapter = receiver._adapter
    expected_lowerings = (receiver.detect_and_validate_entities, receiver.identify_latent_entities)
    preview = arguments.get("preview_num_records")
    selected = frame if preview is None else frame.iloc[:preview]
    graph = DataGraph.new()
    targets = []
    for text in selected[COL_TEXT]:
        graph, target = graph.add_text(text)
        targets.append(target)
    size = len(targets)
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
            max_text_bytes=sum(len(text.encode()) for text in selected[COL_TEXT]),
            max_declarations=0,
            max_group_members=0,
        ),
    )
    workflow = WorkflowId.new()
    owner = NodeId.new(workflow=workflow)
    operation = OperationSpec(
        name="detection-lowering",
        inputs=(InputPort(name="source", artifact_type=_TEXT),),
        outputs=(OutputPort(name="returned_source", artifact_type=_TEXT),),
        output_dependencies=(
            OutputDependency(output="returned_source", inputs=frozenset({"source"}), identity_input="source"),
        ),
        outcomes=(
            OutcomeSpec(
                name="ok",
                category="success",
                produced_ports=frozenset({"returned_source"}),
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
    raw = compile_graph(
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
        sequence=(),
        protection=(),
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
    raw = compile_graph(
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
        asyncio.run_coroutine_threadsafe(complete_detection_entry(requests, inputs), loop).result()
        with bind_detection(
            receiver,
            frame,
            arguments,
            requests,
            source,
            slots,
            loop,
            configuration,
            expected_adapter,
            expected_lowerings,
        ):
            binding = _BINDING.get()
            if binding is None:
                reject(EffectCode.FOREIGN_OWNER)
            result = binding.run(receiver, frame, arguments)
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
                    LocalCompleted(
                        results=(
                            AssociationResult(
                                association=item.association,
                                outcome="ok",
                                consumed_context_ports=frozenset(),
                                outputs=(
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
                    ),
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
        return (
            callback(inputs, requests) if handoff is None else handoff(lambda: context.run(callback, inputs, requests))
        )

    return await _execute_frame_plan(
        prepared=prepared,
        capabilities=capabilities,
        policies=(*parent_policies, *child_policies),
        callback=dispatch,
        pool=pool,
        borrowed_loop=loop,
        limits=ExecutionLimits(
            max_local_in_flight=1,
            max_remote_outstanding=None,
            max_runtime_artifacts=max(1, 12 * size),
            max_runtime_artifact_bytes=None,
            max_collection_items=None,
        ),
        ingress_limits=IngressLimits(
            max_associations=max(1, size),
            max_outputs_per_association=4,
            max_collection_items=None,
            max_assessments=1,
            max_metadata_items=None,
            max_text_bytes=None,
            max_metadata_bytes=None,
            max_depth=8,
        ),
        assessment_limits=AssessmentLimits(
            max_productions=0,
            max_findings_per_production=0,
            max_finding_code_bytes=0,
            max_absence_queries=0,
            max_assessment_facts=0,
            max_port_facts=24 * size,
            max_provenance_edges=40 * size,
        ),
    )


def run_empty_detection(
    receiver: EntityDetectionWorkflow, frame: pd.DataFrame, arguments: dict[str, Any]
) -> EntityDetectionResult:
    """Preserve zero-cardinality adapter callbacks under nonqualifying admission."""
    from anonymizer.engine.detection._frame_publication import InvocationFrame
    from anonymizer.engine.detection._graph_profile import _run_sync_profile, project_detection_redact
    from anonymizer.engine.detection._opaque_redact import (
        _execute_frame_plan,
        _frame_implementation,
        _prepare_frame_plan,
    )

    configuration = detection_configuration(arguments)
    workflow = WorkflowId.new()
    node = NodeId.new(workflow=workflow)
    operation = OperationSpec(
        name="empty-detection-lowering",
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
    raw = compile_graph(
        workflow=workflow,
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
    expected_adapter = receiver._adapter
    expected_lowerings = (receiver.detect_and_validate_entities, receiver.identify_latent_entities)
    raw, child_caps, child_policies, ids = extend_detection_graph(
        raw, node, configuration, latent=arguments["tag_latent_entities"], source_free=True
    )
    cap, policy = _frame_implementation(node, operation, configuration, aggregate=True)
    parent_caps, parent_policies = reparent_detection_policies((cap,), (policy,), ids["entry"])
    capabilities = (*parent_caps, *child_caps)
    policies = (*parent_policies, *child_policies)
    data = DataGraph.new().validate(
        targets=(),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(max_datums=0, max_targets=0, max_text_bytes=0, max_declarations=0, max_group_members=0),
    )
    prepared = _prepare_frame_plan(data, raw, capabilities, (), invocation_owned=True)
    loop = None

    def callback(inputs: tuple[AssociationInput, ...], requests: DelegatedRequests) -> LocalCompleted:
        del inputs
        if loop is None or requests.cancelled:
            reject(EffectCode.FOREIGN_OWNER)
        source = requests.bind_frame(frame, targets=(), source_limit=0)
        with bind_detection(
            receiver,
            frame,
            arguments,
            requests,
            source,
            detection_slots(prepared, ids),
            loop,
            configuration,
            expected_adapter,
            expected_lowerings,
            entry_slot=next(slot.index for slot in prepared.reservation_recipe if slot.template == ids["entry"]),
        ) as binding:
            selected = binding.run(receiver, frame, arguments)
        owner_slot = next(slot.index for slot in prepared.reservation_recipe if slot.template == node)
        item = asyncio.run_coroutine_threadsafe(requests.enter_operation(owner_slot, None), loop).result()
        if item is None:
            reject(EffectCode.REQUEST_CAUSALITY)
        asyncio.run_coroutine_threadsafe(binding.complete_empty_stage([item]), loop).result()
        return LocalCompleted(
            results=(),
            frame=InvocationFrame(
                selected.dataframe, selected.dataframe, tuple(selected.failed_records), (), 0.0, 0.0, None
            ),
        )

    async def execute(pool: BlockingPool, handoff: Callable[..., Any]) -> ReplacementResult:
        nonlocal loop
        loop = asyncio.get_running_loop()

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
            limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=None,
                max_runtime_artifacts=1,
                max_runtime_artifact_bytes=None,
                max_collection_items=None,
            ),
            ingress_limits=IngressLimits(
                max_associations=1,
                max_outputs_per_association=1,
                max_collection_items=None,
                max_assessments=1,
                max_metadata_items=None,
                max_text_bytes=None,
                max_metadata_bytes=None,
                max_depth=8,
            ),
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
        return project_detection_redact(result)

    projected = _run_sync_profile(execute, caller_callbacks=True)
    result = EntityDetectionResult(projected.dataframe, projected.failed_records, ())
    object.__setattr__(result, "_execution", projected._execution)
    return result
