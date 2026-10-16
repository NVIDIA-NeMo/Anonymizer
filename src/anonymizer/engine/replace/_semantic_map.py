# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound replacement selection, generation and acceptance under graph control."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Any, cast

import pandas as pd
from pydantic import TypeAdapter

from anonymizer.engine.constants import COL_REPLACEMENT_MAP, COL_REPLACEMENT_MAP_SOURCE
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests, LocalCompleted
from anonymizer.engine.graph_sdk.capabilities import ConfigField, ConfigText, FrozenConfig
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    AssociationResult,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd.adapter import NddAdapter
from anonymizer.engine.replace.llm_replace_workflow import (
    _INTERNAL_COLUMNS,
    REPLACEMENT_MAP_SOURCE_LLM,
    LlmReplaceResult,
    LlmReplaceWorkflow,
    _create_entity_examples,
    _enrich_entities_for_template,
    _filter_replacement_map_to_input_entities,
)
from anonymizer.engine.replace.strategies import apply_replacement_map
from anonymizer.engine.row_partitioning import ROW_ORDER_COL, merge_and_reorder, resolve_source_positions
from anonymizer.engine.schemas import EntitiesByValueSchema, EntityReplacementMapSchema
from anonymizer.graph.workflow import (
    ArtifactType,
    ChoiceBranch,
    ChoiceDecl,
    Graph,
    InputBinding,
    InputPort,
    NodeId,
    NodeInputRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeSpec,
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SequenceEdge,
    WorkflowInputRef,
    compile_graph,
)
from anonymizer.interface.errors import AnonymizerWorkflowError

_TEXT = ArtifactType(name="anonymizer.text", revision=1)
_MAP = ArtifactType(name="anonymizer.replacement-map-json", revision=1)
_ENTITIES = ArtifactType(name="anonymizer.entity-observation-json", revision=1)
_BINDING: ContextVar[MapBinding | None] = ContextVar("replacement_map_binding", default=None)


def map_configuration(arguments: dict[str, Any]) -> FrozenConfig:
    return FrozenConfig(
        fields=(
            ConfigField(
                name="replacement", value=ConfigText(value=TypeAdapter(dict[str, Any]).dump_json(arguments).decode())
            ),
        )
    )


def extend_map_graph(
    raw: Graph,
    owner: NodeId,
    configuration: FrozenConfig,
    *,
    submission_owner: NodeId,
    predecessor: NodeId | None = None,
    source_free: bool = False,
    apply: bool = False,
    optional: bool = False,
) -> tuple[Graph, tuple[Any, ...], tuple[Any, ...], dict[str, NodeId]]:
    from anonymizer.engine.detection._opaque_redact import _frame_implementation

    ids = {name: NodeId.new(workflow=raw.workflow) for name in ("select", "generate", "accept", "empty")}
    nodes, bindings, edges, capabilities, policies = [], [], [], [], []
    for name in ids:
        inputs = (
            {}
            if source_free
            else {
                "source": _TEXT,
                **({"entities": _ENTITIES} if name != "select" else {}),
                **({"raw_map": _MAP} if name == "accept" else {}),
            }
        )
        outputs = (
            {}
            if source_free
            else {
                "source": _TEXT,
                **({"entities": _ENTITIES} if name in {"select", "generate"} else {}),
                **({"raw_map": _MAP} if name == "generate" else {}),
                **({"map": _MAP} if name in {"accept", "empty"} else {}),
                **({"protected": _TEXT} if apply and name in {"accept", "empty"} else {}),
            }
        )
        operation = OperationSpec(
            name="replacement-" + name,
            inputs=tuple(InputPort(name="in_" + key, artifact_type=value) for key, value in inputs.items()),
            outputs=tuple(OutputPort(name=key, artifact_type=value) for key, value in outputs.items()),
            output_dependencies=tuple(
                OutputDependency(
                    output=key,
                    inputs=frozenset({"in_source"}) if key == "source" else frozenset("in_" + key for key in inputs),
                    identity_input="in_source" if key == "source" else None,
                )
                for key in outputs
            ),
            outcomes=tuple(
                OutcomeSpec(
                    name=outcome,
                    category="success",
                    produced_ports=frozenset(outputs),
                    context=frozenset(),
                    evidence=frozenset(),
                    model_requirements=frozenset(),
                    ceiling=ResourceCeiling(
                        max_activations=1,
                        max_model_requests=None if name == "generate" else 0,
                        max_input_bytes=None,
                        max_output_bytes=None,
                    ),
                )
                for outcome in (
                    ("entities", "empty", "bypass")
                    if name == "select" and optional
                    else ("entities", "empty")
                    if name == "select"
                    else ("ok",)
                )
            ),
        )
        nodes.append(OperationNode(id=ids[name], operation=operation))
        for key in inputs:
            origin = (
                WorkflowInputRef(port="source")
                if name == "select"
                else NodeOutputRef(node=ids["generate" if name == "accept" else "select"], port=key)
            )
            bindings.append(InputBinding(source=origin, destination=NodeInputRef(node=ids[name], port="in_" + key)))
        cap, policy = _frame_implementation(
            ids[name], operation, configuration, submission_owner=submission_owner, local=name != "generate"
        )
        capabilities.append(cap)
        policies.append(policy)
    edges.extend(
        SequenceEdge(before=ids[a], after=ids[b])
        for a, b in (("select", "generate"), ("generate", "accept"), ("select", "empty"))
    )
    edges.extend(SequenceEdge(before=ids[name], after=owner) for name in ("select", "accept", "empty"))
    if predecessor is not None:
        edges.append(SequenceEdge(before=predecessor, after=ids["select"]))
    if optional:
        entry = NodeId.new(workflow=raw.workflow)
        ids["entry"] = entry
        op = replace(
            nodes[0].operation,
            name="replacement-call",
            outputs=(),
            output_dependencies=(),
            outcomes=tuple(
                replace(nodes[0].operation.outcomes[0], name=name, produced_ports=frozenset())
                for name in ("builtin", "bypass")
            ),
        )
        nodes.append(OperationNode(id=entry, operation=op))
        if not source_free:
            bindings.append(
                InputBinding(
                    source=WorkflowInputRef(port="source"), destination=NodeInputRef(node=entry, port="in_source")
                )
            )
        cap, policy = _frame_implementation(entry, op, configuration, submission_owner=submission_owner, local=True)
        capabilities.append(cap)
        policies.append(policy)
        if predecessor is not None:
            edges.remove(SequenceEdge(before=predecessor, after=ids["select"]))
            edges.append(SequenceEdge(before=predecessor, after=entry))
        edges.extend((SequenceEdge(before=entry, after=ids["select"]), SequenceEdge(before=entry, after=owner)))
    raw = compile_graph(
        workflow=raw.workflow,
        interface=replace(
            raw.interface,
            outcomes=tuple(
                replace(
                    outcome,
                    ceiling=replace(
                        outcome.ceiling,
                        max_activations=outcome.ceiling.max_activations + 4 + int(optional),
                        max_input_bytes=None,
                        max_output_bytes=None,
                    ),
                )
                for outcome in raw.interface.outcomes
            ),
        ),
        nodes=(
            *raw.nodes,
            *nodes,
            ChoiceDecl(
                selector=ids["select"],
                branches=(
                    ChoiceBranch(outcomes=frozenset({"entities"}), members=frozenset({ids["generate"], ids["accept"]})),
                    ChoiceBranch(outcomes=frozenset({"empty"}), members=frozenset({ids["empty"]})),
                ),
            ),
        ),
        input_bindings=(*raw.input_bindings, *bindings),
        output_bindings=tuple(raw.output_bindings),
        outcome_bindings=tuple(raw.outcome_bindings),
        sequence=(*raw.sequence, *edges),
        protection=tuple(raw.protection_requirements),
        limits=replace(
            raw.limits,
            max_nodes=raw.limits.max_nodes + 4 + int(optional),
            max_bindings=raw.limits.max_bindings + 13,
            max_sequence_edges=raw.limits.max_sequence_edges + 10,
            max_choices=raw.limits.max_choices + 1,
            max_branch_members=raw.limits.max_branch_members + 3,
            max_choice_states=raw.limits.max_choice_states * (6 if optional else 2),
        ),
        dynamic_limits=replace(
            raw.dynamic_limits,
            max_activation_occurrences=raw.dynamic_limits.max_activation_occurrences + 4 + int(optional),
        ),
    )
    return raw, tuple(capabilities), tuple(policies), ids


@dataclass
class MapBinding:
    requests: DelegatedRequests
    source: WorkflowSourceFrame
    slots: dict[str, int]
    loop: asyncio.AbstractEventLoop
    receiver: LlmReplaceWorkflow
    entry: pd.DataFrame
    arguments: dict[str, Any]
    adapter: Any
    lowering: Any
    configuration: FrozenConfig
    apply: bool = False
    claim: Lock = field(default_factory=Lock)
    terminal: tuple[Any, ...] = ()
    entry_inputs: tuple[AssociationInput, ...] = ()
    _entry_owner: MapBinding | None = field(default=None, repr=False)
    replacement_receiver: Any | None = None
    replacement_arguments: dict[str, Any] | None = None
    replacement_configuration: str | None = None
    adapter_run: Any | None = None
    _associations: tuple[SemanticAssociation, ...] = field(init=False, repr=False)
    _positions: dict[SemanticAssociation, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._associations = self.source.associations
        self._positions = {association: position for position, association in enumerate(self._associations)}
        if len(self._positions) != len(self._associations):
            reject(EffectCode.DUPLICATE)
        if self._entry_owner is None:
            self._entry_owner = self
        if self.adapter_run is None:
            self.adapter_run = self.adapter.run_workflow

    def bypass(self) -> None:
        self.authenticate()
        self.call(
            self.complete(
                tuple(
                    (
                        cast(SemanticAssociation, item.association),
                        LocalCompleted(
                            results=(
                                AssociationResult(
                                    association=item.association,
                                    outcome="bypass",
                                    outputs=(),
                                    consumed_context_ports=frozenset(),
                                ),
                            )
                        ),
                    )
                    for item in self.entry_inputs
                )
            )
        )
        ready = self.call(self.ready(self.source.associations))
        if ready is None or ready[0] != "select":
            reject(EffectCode.REQUEST_CAUSALITY)
        _, bound, sources = ready
        results = []
        for item, association in zip(bound, sources, strict=True):
            row = self.source.dataframe.iloc[self._positions[association]]
            entities = EntitiesByValueSchema.from_raw(row[self.arguments["entities_column"]]).model_dump_json()
            results.append(
                (cast(SemanticAssociation, item.association), self.publish(item, outcome="bypass", entities=entities))
            )
        if not sources:
            results = [(cast(SemanticAssociation, bound[0].association), self.publish(bound[0], outcome="bypass"))]
        self.call(self.complete(results))

    def authenticate_result(self, result: LlmReplaceResult) -> None:
        self.authenticate()
        if result.source_positions != tuple(position for position, _, _ in self.terminal) or len(
            result.dataframe
        ) != len(self.terminal):
            reject(EffectCode.FOREIGN_OWNER)
        start = 0
        while start < len(self.terminal):
            name = self.terminal[start][2]
            end = next(
                (
                    index
                    for index in range(start + 1, min(start + 64, len(self.terminal)))
                    if self.terminal[index][2] != name
                ),
                min(start + 64, len(self.terminal)),
            )
            accepted_rows = self.call(
                self.requests.operation_results(
                    self.slots[name], tuple(self._associations[position] for position, _, _ in self.terminal[start:end])
                )
            )
            if len(accepted_rows) != end - start:
                reject(EffectCode.MISSING)
            for index, accepted in enumerate(accepted_rows, start):
                if accepted is None:
                    reject(EffectCode.MISSING)
                canonical = next(port.value.text for port in accepted.outputs if port.port == "map")
                if json.dumps(result.dataframe[COL_REPLACEMENT_MAP].iloc[index]) != canonical:
                    reject(EffectCode.FOREIGN_OWNER)
            start = end

    def authenticate(self) -> None:
        from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow

        borrow = source_worker_borrow()
        if (
            self._entry_owner is not self
            or borrow is not self.requests
            or borrow.owner.loop is not self.loop
            or borrow.owner.closed
            or self.requests.cancelled
            or self.source.associations != self._associations
            or self.receiver._adapter is not self.adapter
            or self.adapter.run_workflow != self.adapter_run
            or self.receiver._generate_columns != self.lowering
            or not self.source.matches(self.entry)
            or map_configuration(self.arguments) != self.configuration
        ):
            reject(EffectCode.FOREIGN_OWNER)
        borrow.authorize_sources_sync(self.source.associations)

    def call(self, operation: Any) -> Any:
        return asyncio.run_coroutine_threadsafe(operation, self.loop).result()

    async def ready(
        self, associations: tuple[Any, ...]
    ) -> tuple[str, tuple[AssociationInput, ...], tuple[Any, ...]] | None:
        ready = (
            await self.requests.ready_operations_many(associations)
            if associations
            else (await self.requests.ready_operations(None),)
        )
        for name, slot in self.slots.items():
            selected = (
                tuple(a for a, slots in zip(associations, ready, strict=True) if slot in slots) if associations else ()
            )
            if selected or not associations and slot in ready[0]:
                entered = (
                    await self.requests.enter_operations(slot, selected)
                    if selected
                    else (await self.requests.enter_operation(slot, None),)
                )
                if any(item is None for item in entered):
                    reject(EffectCode.REQUEST_CAUSALITY)
                return name, cast(tuple[AssociationInput, ...], tuple(entered)), selected
        return None

    def publish(
        self,
        item: AssociationInput,
        *,
        outcome: str = "ok",
        entities: str | None = None,
        raw_map: str | None = None,
        accepted: str | None = None,
        protected: str | None = None,
    ) -> LocalCompleted:
        outputs = []
        source = next((port.value for port in item.inputs if port.port == "in_source"), None)
        if source is not None:
            outputs.append(PortArtifact(port="source", artifact_type=_TEXT, artifact=None, value=source))
            for name, value, kind in (
                ("entities", entities, _ENTITIES),
                ("raw_map", raw_map, _MAP),
                ("map", accepted, _MAP),
                ("protected", protected, _TEXT),
            ):
                if value is not None:
                    outputs.append(
                        PortArtifact(port=name, artifact_type=kind, artifact=None, value=TextArtifactValue(text=value))
                    )
        return LocalCompleted(
            results=(
                AssociationResult(
                    association=item.association,
                    outcome=outcome,
                    outputs=tuple(outputs),
                    consumed_context_ports=frozenset(),
                ),
            )
        )

    async def complete(self, results: Any) -> None:
        outcomes = await self.requests.complete_operations(tuple(results))
        if any(
            outcome.condition != "result" or outcome.category != "success"
            for (_, result), outcome in zip(results, outcomes, strict=True)
            if isinstance(result, LocalCompleted)
        ):
            raise AnonymizerWorkflowError("Replacement semantic result was rejected")

    def run(self, receiver: LlmReplaceWorkflow, dataframe: pd.DataFrame, arguments: dict[str, Any]) -> LlmReplaceResult:
        if (
            receiver is not self.receiver
            or dataframe is not self.entry
            or map_configuration(arguments) != self.configuration
            or arguments["model_configs"] is not self.arguments["model_configs"]
            or arguments["selected_models"] is not self.arguments["selected_models"]
            or self.claim is None
            or not self.claim.acquire(blocking=False)
        ):
            reject(EffectCode.FOREIGN_OWNER)
        self.authenticate()
        if self.entry_inputs:
            self.call(
                self.complete(
                    tuple(
                        (
                            cast(SemanticAssociation, item.association),
                            LocalCompleted(
                                results=(
                                    AssociationResult(
                                        association=item.association,
                                        outcome="builtin",
                                        outputs=(),
                                        consumed_context_ports=frozenset(),
                                    ),
                                )
                            ),
                        )
                        for item in self.entry_inputs
                    )
                )
            )
        working = dataframe.copy()
        parsed = working[arguments["entities_column"]].apply(EntitiesByValueSchema.from_raw)
        from anonymizer.engine.constants import (
            COL_ENTITIES_FOR_REPLACE,
            COL_ENTITIES_FOR_REPLACE_JSON,
            COL_ENTITY_EXAMPLES,
        )

        working[COL_ENTITY_EXAMPLES] = parsed.apply(_create_entity_examples)
        working[COL_ENTITIES_FOR_REPLACE] = parsed.apply(_enrich_entities_for_template)
        working[COL_ENTITIES_FOR_REPLACE_JSON] = working[COL_ENTITIES_FOR_REPLACE].apply(json.dumps)
        working[ROW_ORDER_COL] = range(len(working))
        empty_partition = None
        if len(working) and all(entity.entities_by_value for entity in parsed):
            # The accepted concat includes the original empty passthrough
            # partition; its extension dtypes affect native all-null columns.
            empty_partition = working.iloc[:0].copy()
            empty_partition[COL_REPLACEMENT_MAP] = pd.Series([], dtype=object)
            empty_partition[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM
        active = self.source.associations
        successful = []
        failures = []
        terminal = []
        generated = {}
        generated_sources = set()
        while True:
            self.authenticate()
            ready = self.call(self.ready(active))
            if ready is None:
                break
            name, bound, selected_sources = ready
            positions = [self._positions[association] for association in selected_sources] if active else []
            selected = working.iloc[positions].copy() if active else working
            if name == "select":
                results = (
                    [
                        (
                            cast(SemanticAssociation, item.association),
                            self.publish(
                                item,
                                outcome="entities" if parsed.iloc[position].entities_by_value else "empty",
                                entities=parsed.iloc[position].model_dump_json(),
                            ),
                        )
                        for item, position in zip(bound, positions, strict=True)
                    ]
                    if active
                    else [(cast(SemanticAssociation, bound[0].association), self.publish(bound[0], outcome="empty"))]
                )
            elif name == "generate":
                from anonymizer.engine.replace._native_map_requests import _BRIDGE

                native = getattr(self.adapter.run_workflow, "__func__", None) is NddAdapter.run_workflow
                stage_source = (
                    WorkflowSourceFrame.capture(selected, associations=selected_sources, invocation_bound=True)
                    if native
                    else None
                )
                preview = arguments["preview_num_records"]
                result = self.lowering(
                    selected,
                    model_configs=arguments["model_configs"],
                    selected_models=arguments["selected_models"],
                    instructions=arguments["instructions"],
                    preview_num_records=min(preview, len(selected)) if preview is not None else None,
                    **({"source_frame": stage_source} if stage_source is not None else {}),
                    **({"request_bridge": _BRIDGE.get()} if native and _BRIDGE.get() is not None else {}),
                )
                generated_sources.update(selected_sources)
                self.authenticate()
                surviving = resolve_source_positions(selected, result.dataframe, result.source_positions)
                failures.extend(result.failed_records)
                generated.update(
                    (positions[index], result.dataframe.iloc[[row]].copy()) for row, index in enumerate(surviving)
                )
                by_position = {index: row for row, index in enumerate(surviving)}
                from anonymizer.engine.graph_sdk._execution_values import LocalFailure

                results = []
                for index, item in enumerate(bound):
                    if index not in by_position:
                        results.append(
                            (
                                cast(SemanticAssociation, item.association),
                                LocalFailure(failure="implementation_exception"),
                            )
                        )
                    else:
                        row = result.dataframe.iloc[by_position[index]]
                        value = row.get(COL_REPLACEMENT_MAP, {"replacements": []})
                        if hasattr(value, "model_dump"):
                            value = value.model_dump()
                        if not isinstance(value, dict):
                            value = _filter_replacement_map_to_input_entities(
                                raw_map=value,
                                parsed_entities=EntitiesByValueSchema.from_raw(
                                    selected.iloc[index][arguments["entities_column"]]
                                ),
                            )
                        value = EntityReplacementMapSchema.model_validate(value).model_dump(mode="python")
                        entities = next(port.value.text for port in item.inputs if port.port == "in_entities")
                        results.append(
                            (
                                cast(SemanticAssociation, item.association),
                                self.publish(item, entities=entities, raw_map=json.dumps(value)),
                            )
                        )
            else:
                results = []
                for index, item in enumerate(bound):
                    if not active:
                        results.append((cast(SemanticAssociation, item.association), self.publish(item)))
                        continue
                    position = positions[index]
                    one = generated[position].copy() if name == "accept" else working.iloc[[position]].copy()
                    row = one.iloc[0]
                    raw = next(
                        (json.loads(port.value.text) for port in item.inputs if port.port == "in_raw_map"),
                        {"replacements": []},
                    )
                    entities = EntitiesByValueSchema.model_validate_json(
                        next(port.value.text for port in item.inputs if port.port == "in_entities")
                    )
                    accepted_map = _filter_replacement_map_to_input_entities(
                        raw_map=raw, parsed_entities=entities, record_id=str(row.get("_anonymizer_record_id", ""))
                    )
                    one[COL_REPLACEMENT_MAP] = [accepted_map]
                    one[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM
                    one[ROW_ORDER_COL] = position
                    if self.apply:
                        one = apply_replacement_map(one)
                    from anonymizer.engine.constants import COL_REPLACED_TEXT

                    results.append(
                        (
                            cast(SemanticAssociation, item.association),
                            self.publish(
                                item,
                                accepted=json.dumps(accepted_map),
                                protected=one[COL_REPLACED_TEXT].iloc[0] if self.apply else None,
                            ),
                        )
                    )
                    successful.append(one)
                    terminal.append((position, item.association, name))
            self.call(self.complete(results))
        from anonymizer.engine.replace._native_map_requests import _BRIDGE

        if _BRIDGE.get() is not None:
            self.call(self.requests.close_observed_requests(self.source.associations, frozenset(generated_sources)))
        combined = (
            merge_and_reorder(*successful, *((empty_partition,) if empty_partition is not None else ()))
            if successful
            else working.iloc[:0].drop(columns=[ROW_ORDER_COL], errors="ignore")
        )
        if not active:
            combined = working.copy().drop(columns=[ROW_ORDER_COL], errors="ignore")
            combined[COL_REPLACEMENT_MAP] = pd.Series([], index=combined.index, dtype=object)
            combined[COL_REPLACEMENT_MAP_SOURCE] = REPLACEMENT_MAP_SOURCE_LLM
        combined = combined.drop(columns=list(_INTERNAL_COLUMNS), errors="ignore")
        self.terminal = tuple(sorted(terminal))
        return LlmReplaceResult(combined, failures, tuple(position for position, _, _ in self.terminal))


@contextmanager
def bind_map(
    receiver: LlmReplaceWorkflow,
    dataframe: pd.DataFrame,
    arguments: dict[str, Any],
    requests: DelegatedRequests,
    source: WorkflowSourceFrame,
    slots: dict[str, int],
    loop: asyncio.AbstractEventLoop,
    *,
    apply: bool = False,
    entry_inputs: tuple[AssociationInput, ...] = (),
) -> Iterator[MapBinding]:
    binding = MapBinding(
        requests,
        source,
        slots,
        loop,
        receiver,
        dataframe,
        arguments,
        receiver._adapter,
        receiver._generate_columns,
        map_configuration(arguments),
        apply,
        Lock(),
        entry_inputs=entry_inputs,
    )
    token = _BINDING.set(binding)
    try:
        yield binding
    finally:
        _BINDING.reset(token)


def run_map(receiver: LlmReplaceWorkflow, dataframe: pd.DataFrame, arguments: dict[str, Any]) -> LlmReplaceResult:
    binding = _BINDING.get()
    if binding is not None:
        if type(binding) is not MapBinding:
            reject(EffectCode.FOREIGN_OWNER)
        return binding.run(receiver, dataframe, arguments)
    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW

    if _WORKER_BORROW.get() is not None:
        reject(EffectCode.FOREIGN_OWNER)
    from anonymizer.engine.replace._semantic_map_profile import run_direct_map

    return run_direct_map(receiver, dataframe, arguments, invoke_entry=False)
