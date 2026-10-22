# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Source-bound replacement selection, generation and acceptance under graph control."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from typing import Any, cast

import pandas as pd
from pydantic import TypeAdapter

from anonymizer.engine.constants import (
    COL_ENTITIES_FOR_REPLACE,
    COL_ENTITIES_FOR_REPLACE_JSON,
    COL_ENTITY_EXAMPLES,
    COL_REPLACEMENT_MAP,
    COL_REPLACEMENT_MAP_SOURCE,
)
from anonymizer.engine.graph_sdk._authoring_defaults import _compile_graph, _operation_spec, _outcome_spec
from anonymizer.engine.graph_sdk._effect_values import EffectCode, reject
from anonymizer.engine.graph_sdk._execution_delegated import drive
from anonymizer.engine.graph_sdk._execution_values import DelegatedRequests, LocalCompleted
from anonymizer.engine.graph_sdk._workflow_values import (
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
    OutputDependency,
    OutputPort,
    ResourceCeiling,
    SequenceEdge,
    WorkflowInputRef,
)
from anonymizer.engine.graph_sdk.capabilities import ConfigField, ConfigText, FrozenConfig
from anonymizer.engine.graph_sdk.requests import (
    AssociationInput,
    PortArtifact,
    SemanticAssociation,
    TextArtifactValue,
)
from anonymizer.engine.ndd._route import native_adapter
from anonymizer.engine.ndd._source_frame import WorkflowSourceFrame
from anonymizer.engine.ndd._stage_binding import StageBinding, completed
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

_TEXT = ArtifactType(name="anonymizer.text", revision=1)
_MAP = ArtifactType(name="anonymizer.replacement-map-json", revision=1)
_ENTITIES = ArtifactType(name="anonymizer.entity-observation-json", revision=1)


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
    from anonymizer.engine.harness.kernel import _frame_implementation

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
        operation = _operation_spec(
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
                _outcome_spec(
                    name=outcome,
                    category="success",
                    produced_ports=frozenset(outputs),
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
    raw = _compile_graph(
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
class MapBinding(StageBinding):
    """The admitted graph's map stages for one builtin replacement-map lowering."""

    rejected = "Replacement semantic result was rejected"

    requests: DelegatedRequests
    source: WorkflowSourceFrame
    slots: dict[str, int]
    receiver: LlmReplaceWorkflow
    entry: pd.DataFrame
    arguments: dict[str, Any]
    apply: bool = False
    entry_inputs: tuple[AssociationInput, ...] = ()
    terminal: tuple[Any, ...] = ()
    _positions: dict[SemanticAssociation, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._positions = {association: position for position, association in enumerate(self.source.associations)}
        if len(self._positions) != len(self.source.associations):
            reject(EffectCode.DUPLICATE)

    def bypass(self) -> None:
        """Account the map stages as bypassed when a caller's map helper ran opaque in their place."""
        self.authorize()
        self.call(
            self.complete_results(
                tuple(
                    (cast(SemanticAssociation, item.association), completed(item.association, outcome="bypass"))
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
        self.call(self.complete_results(results))

    def authorize(self) -> None:
        from anonymizer.engine.graph_sdk._execution_delegated import source_worker_borrow

        if self.requests.cancelled:
            reject(EffectCode.FOREIGN_OWNER)
        source_worker_borrow().authorize_sources_sync(self.source.associations)

    def call(self, operation: Any) -> Any:
        return drive(operation)

    async def ready(
        self, associations: tuple[Any, ...]
    ) -> tuple[str, tuple[AssociationInput, ...], tuple[Any, ...]] | None:
        return await self.enter_ready(associations)

    def not_ready(self, *, empty: bool) -> None:
        del empty
        reject(EffectCode.REQUEST_CAUSALITY)

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
        return completed(item.association, outputs, outcome)

    def lower(self) -> LlmReplaceResult:
        """Run the builtin map stages against the graph's children; called only by the admitted graph."""
        dataframe, arguments = self.entry, self.arguments
        self.authorize()
        if self.entry_inputs:
            self.call(
                self.complete_results(
                    tuple(
                        (cast(SemanticAssociation, item.association), completed(item.association, outcome="builtin"))
                        for item in self.entry_inputs
                    )
                )
            )
        working = dataframe.copy()
        parsed = working[arguments["entities_column"]].apply(EntitiesByValueSchema.from_raw)
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
            self.authorize()
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
                native = native_adapter(self.receiver._adapter)
                stage_source = (
                    WorkflowSourceFrame.capture(selected, associations=selected_sources, invocation_bound=True)
                    if native
                    else None
                )
                preview = arguments["preview_num_records"]
                result = self.receiver._generate_columns(
                    selected,
                    model_configs=arguments["model_configs"],
                    selected_models=arguments["selected_models"],
                    instructions=arguments["instructions"],
                    preview_num_records=min(preview, len(selected)) if preview is not None else None,
                    **({"source_frame": stage_source} if stage_source is not None else {}),
                )
                generated_sources.update(selected_sources)
                self.authorize()
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
            self.call(self.complete_results(results))
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


def run_map(receiver: LlmReplaceWorkflow, dataframe: pd.DataFrame, arguments: dict[str, Any]) -> LlmReplaceResult:
    """The builtin helper's own entry: one standalone invocation, also when a caller's helper delegates to it."""
    from anonymizer.engine.graph_sdk._execution_delegated import _WORKER_BORROW
    from anonymizer.engine.replace._semantic_map_profile import run_direct_map

    token = _WORKER_BORROW.set(None)
    try:
        return run_direct_map(receiver, dataframe, arguments)
    finally:
        _WORKER_BORROW.reset(token)
