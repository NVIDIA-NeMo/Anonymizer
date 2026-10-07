# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable product comparison for every frozen activation reference trace."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import pytest

from anonymizer.graph._values import ActivationKey, ContractViolation, InvocationId, PlanId
from anonymizer.graph.activation import (
    ActivationLimits,
    ActivationSeed,
    ActivationState,
    CloseUnstarted,
    ObserveMembership,
    ObserveOverflow,
    ObserveTerminal,
    Select,
    Start,
    advance_activation,
    initialize_activation,
)
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    AdmittedWorkflow,
    ArtifactType,
    ChoiceBranch,
    ChoiceDecl,
    DynamicLimits,
    DynamicScope,
    InputBinding,
    InputPort,
    KeyedJoinDecl,
    LoopCarriedBinding,
    LoopDecl,
    LoopInitialBinding,
    MapDecl,
    NodeId,
    NodeInputRef,
    NodeOutcomeRef,
    NodeOutputRef,
    OperationNode,
    OperationSpec,
    OutcomeBinding,
    OutcomeClass,
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
    admit_activation_workflow,
    admit_static_workflow,
)

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]
Object: TypeAlias = dict[str, Json]
Scope: TypeAlias = tuple[str, ...]
Identity: TypeAlias = tuple[Scope, str]

CASES_PATH = Path(__file__).parent / "reference" / "activation_v1_cases.json"
CASES = cast(list[Object], json.loads(CASES_PATH.read_bytes()))
ARTIFACT = ArtifactType(name="reference-artifact", revision=1)
ARTIFACT_V2 = ArtifactType(name="reference-artifact", revision=2)


def _object(value: Json) -> Object:
    if not isinstance(value, dict):
        raise TypeError("expected reference object")
    return cast(Object, value)


def _array(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise TypeError("expected reference array")
    return value


def _scope(value: Json) -> Scope:
    return tuple(cast(str, item) for item in _array(value))


def _case(case_id: str) -> Object:
    return next(case for case in CASES if case["case_id"] == case_id)


def _executions() -> list[tuple[str, Object]]:
    executions: list[tuple[str, Object]] = []
    for case in CASES:
        case_id = cast(str, case["case_id"])
        executions.append((case_id, case))
        for index, trace_value in enumerate(_array(case["traces"])):
            trace = _object(trace_value)
            traced_case = dict(case)
            traced_case["declaration"] = trace["declaration"]
            traced_case["boundary"] = trace["boundary"]
            traced_case["events"] = trace["events"]
            traced_case["expected"] = trace["expected"]
            executions.append((f"{case_id}::{trace['name']}-{index}", cast(Object, traced_case)))
    return executions


EXECUTIONS = _executions()
COMMUTED_CASES = [
    case
    for case in CASES
    if any(_object(trace)["name"] == "commute_independent_siblings" for trace in _array(case["traces"]))
]


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _support_case(case: Object) -> Object:
    case_id = cast(str, case["case_id"])
    family = cast(str, case["family"])
    if family == "sequence_mutations" and case_id != "sequence_mutations/010/named_missing_result":
        return _case("sequence_single/000/base")
    if family == "choice" and int(case_id.split("/")[1]) >= 4:
        return _case("choice/000/ok_forward")
    if family == "subgraph" and int(case_id.split("/")[1]) in {4, 5, 6, 7, 9}:
        return _case("subgraph/000/body_1_success")
    if family == "map" and int(case_id.split("/")[1]) >= 12:
        return _case("map/007/bound_2_size_2_ss")
    if family == "join" and int(case_id.split("/")[1]) >= 21:
        return _case("join/005/children_2_success-success")
    if family == "loop" and 8 <= int(case_id.split("/")[1]) <= 15:
        return _case("loop/006/bound_2_executed_2_stop")
    if family == "terminal_coverage" and int(case_id.split("/")[1]) >= 12:
        return _case("terminal_coverage/011/closed_2_2")
    if case_id == "precedence/000/overflow_type_before_value":
        return _case("map/007/bound_2_size_2_ss")
    if case_id == "precedence/001/event_limit_before_foreign":
        return _case("sequence_single/000/base")
    if case_id == "precedence/002/foreign_before_duplicate":
        return _case("sequence_linked_pair/000-000/base")
    if case_id == "precedence/003/duplicate_before_missing":
        return _case("sequence_independent_siblings/000-000/base")
    if case_id == "precedence/004/missing_before_contradictory":
        return _case("map/002/bound_1_size_1_s")
    return case


@dataclass
class _BuiltScope:
    workflow: AdmittedWorkflow
    nodes: dict[str, NodeId]


class _Adapter:
    def __init__(self, case: Object) -> None:
        self.case = case
        self.declaration = _object(case["declaration"])
        self.support = _object(_support_case(case)["declaration"])
        self.plan = PlanId.new()
        self.invocations: dict[str, InvocationId] = {}
        self.keys: dict[tuple[str, str], ActivationKey] = {}
        self.seed_records = [_object(value) for value in _array(self.declaration["seeds"])]
        self.support_seeds = [_object(value) for value in _array(self.support["seeds"])]
        self.seed_by_key = {cast(str, seed["key"]): seed for seed in self.seed_records}
        self.support_seed_by_key = {cast(str, seed["key"]): seed for seed in self.support_seeds}
        self.node_ids: dict[Identity, NodeId] = {}
        self.node_identity: dict[NodeId, Identity] = {}
        self.scopes: dict[Scope, _BuiltScope] = {}
        self.activation_labels: dict[ActivationKey, str] = {}
        self.workflow: AdmittedActivationWorkflow | None = None

    def invocation(self, label: str) -> InvocationId:
        if label not in self.invocations:
            self.invocations[label] = InvocationId.new(plan=self.plan)
        return self.invocations[label]

    def _support_identity(self, key: str) -> Identity:
        seed = self.support_seed_by_key[key]
        return (_scope(seed["scope"]), cast(str, seed["template"]))

    def _target_identity(self, seed: Object) -> Identity:
        return (_scope(seed["scope"]), cast(str, seed["template"]))

    def _records(self, identity: Identity) -> list[Object]:
        scope, template = identity
        return [
            item
            for value in _array(self.support["outcomes"])
            if (item := _object(value))["template"] == template and _scope(item["scope"]) == scope
        ]

    def _operation(
        self,
        identity: Identity,
        *,
        inputs: tuple[str, ...],
        ceiling: int = 0,
        dependency_inputs: bool = True,
        expose_outputs: bool = True,
        outcome_names: frozenset[str] | None = None,
    ) -> OperationSpec:
        records = self._records(identity)
        if outcome_names is not None:
            records = [record for record in records if record["name"] in outcome_names]
        outputs = sorted({cast(str, port) for record in records for port in _array(record["produced_ports"])})
        return OperationSpec(
            name="/".join((*identity[0], identity[1])),
            inputs=tuple(InputPort(name=name, artifact_type=ARTIFACT) for name in inputs),
            outputs=tuple(OutputPort(name=name, artifact_type=ARTIFACT) for name in outputs if expose_outputs),
            output_dependencies=tuple(
                OutputDependency(
                    output=name,
                    inputs=frozenset(inputs) if dependency_inputs else frozenset(),
                    identity_input=None,
                )
                for name in outputs
                if expose_outputs
            ),
            outcomes=tuple(
                OutcomeSpec(
                    name=cast(str, record["name"]),
                    category=cast(OutcomeClass, record["category"]),
                    produced_ports=frozenset(cast(str, port) for port in _array(record["produced_ports"]))
                    if expose_outputs
                    else frozenset(),
                    context=frozenset(),
                    evidence=frozenset(),
                    state_effects=frozenset(),
                    model_requirements=frozenset(),
                    ceiling=ResourceCeiling(
                        max_activations=ceiling,
                        max_model_requests=0,
                        max_input_bytes=0,
                        max_output_bytes=0,
                    ),
                )
                for record in records
            ),
        )

    def _aggregate_scope(self, aggregate: Object) -> Scope:
        if aggregate["kind"] == "loop":
            return _scope(aggregate["member_scope"])
        return self._support_identity(cast(str, aggregate["parent"]))[0]

    def _scope_templates(self, scope: Scope) -> set[str]:
        templates = {
            identity[1] for seed in self.support_seeds if (identity := self._target_identity(seed))[0] == scope
        }
        for value in _array(self.support["aggregates"]):
            aggregate = _object(value)
            if aggregate["kind"] == "loop" and _scope(aggregate["member_scope"]) == scope:
                templates.add(cast(str, aggregate["member_template"]))
            elif aggregate["kind"] == "map" and self._aggregate_scope(aggregate) == scope:
                members = _array(aggregate["members"])
                templates.add(self._support_identity(cast(str, members[0]))[1] if members else "N1")
        return templates

    def _node_for_seed(self, seed: Object) -> NodeId:
        identity = self._target_identity(seed)
        if identity in self.node_ids:
            return self.node_ids[identity]
        label = identity[1]
        candidates = [node for (scope, template), node in self.node_ids.items() if template == label and not scope]
        if not candidates:
            candidates = [node for (_, template), node in self.node_ids.items() if template == label]
        if not candidates:
            raise KeyError(identity)
        return candidates[0]

    def _input_names(self, identity: Identity) -> tuple[str, ...]:
        names: set[str] = set()
        for value in _array(self.support["input_dependencies"]):
            dependency = _object(value)
            destination = self._support_identity(cast(str, dependency["destination"]))
            if destination == identity:
                names.add(cast(str, dependency["destination_port"]))
        for value in _array(self.support["aggregates"]):
            aggregate = _object(value)
            if (
                aggregate["kind"] == "loop"
                and (_scope(aggregate["member_scope"]), cast(str, aggregate["member_template"])) == identity
            ):
                binding = aggregate["initial_binding"] or aggregate["carried_binding"]
                if isinstance(binding, dict):
                    names.add(cast(str, binding["destination_port"]))
        return tuple(sorted(names))

    def _subgraph_bodies(self) -> dict[Identity, Scope]:
        result: dict[Identity, Scope] = {}
        for value in _array(self.support["subgraphs"]):
            subgraph = _object(value)
            parent = self._support_identity(cast(str, subgraph["parent"]))
            result[parent] = (*parent[0], parent[1])
        return result

    def _build_static_precedence(self) -> AdmittedWorkflow:
        """Construct the two frozen static-ordering probes from their raw facts."""
        owner = WorkflowId.new()
        left = NodeId.new(workflow=owner)
        right = NodeId.new(workflow=owner)
        empty_interface = OperationSpec(
            name="precedence-interface",
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
                        max_activations=0,
                        max_model_requests=0,
                        max_input_bytes=0,
                        max_output_bytes=0,
                    ),
                ),
            ),
        )
        code = cast(str, self.case["case_id"])
        choices: tuple[ChoiceDecl, ...] = ()
        input_bindings: tuple[InputBinding, ...] = ()
        if code.endswith("overlap_before_cycle"):
            left_operation = self._operation(((), "N0"), inputs=())
            right_operation = left_operation
            choices = (
                ChoiceDecl(
                    selector=right,
                    branches=(
                        ChoiceBranch(outcomes=frozenset({"ok"}), members=frozenset({left})),
                        ChoiceBranch(outcomes=frozenset({"fail"}), members=frozenset({left})),
                    ),
                ),
            )
        else:
            left_operation = OperationSpec(
                name="precedence-source",
                inputs=(),
                outputs=(OutputPort(name="result", artifact_type=ARTIFACT),),
                output_dependencies=(OutputDependency(output="result", inputs=frozenset(), identity_input=None),),
                outcomes=(
                    OutcomeSpec(
                        name="ok",
                        category="success",
                        produced_ports=frozenset({"result"}),
                        context=frozenset(),
                        evidence=frozenset(),
                        state_effects=frozenset(),
                        model_requirements=frozenset(),
                        ceiling=ResourceCeiling(
                            max_activations=0,
                            max_model_requests=0,
                            max_input_bytes=0,
                            max_output_bytes=0,
                        ),
                    ),
                ),
            )
            right_operation = OperationSpec(
                name="precedence-destination",
                inputs=(InputPort(name="input", artifact_type=ARTIFACT_V2),),
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
                            max_activations=0,
                            max_model_requests=0,
                            max_input_bytes=0,
                            max_output_bytes=0,
                        ),
                    ),
                ),
            )
            input_bindings = (
                InputBinding(
                    source=NodeOutputRef(node=left, port="result"),
                    destination=NodeInputRef(node=right, port="input"),
                ),
            )
        return admit_static_workflow(
            workflow=owner,
            interface=empty_interface,
            nodes=(
                OperationNode(id=left, operation=left_operation),
                OperationNode(id=right, operation=right_operation),
            ),
            input_bindings=input_bindings,
            output_bindings=(),
            outcome_bindings=(),
            sequence=(SequenceEdge(before=left, after=right), SequenceEdge(before=right, after=left)),
            choices=choices,
            protection=(),
            limits=WorkflowLimits(
                max_nodes=2,
                max_bindings=len(input_bindings),
                max_sequence_edges=2,
                max_choices=len(choices),
                max_branch_members=2 if choices else 0,
                max_subgraph_depth=1,
                max_choice_states=4,
            ),
        )

    def build_static(self) -> AdmittedWorkflow:
        if cast(str, self.case["case_id"]) in {
            "precedence/005/overlap_before_cycle",
            "precedence/006/cycle_before_contradictory",
        }:
            return self._build_static_precedence()
        subgraph_bodies = self._subgraph_bodies()
        scopes = {self._target_identity(seed)[0] for seed in self.support_seeds} | set(subgraph_bodies.values())
        for scope in sorted(scopes, key=len, reverse=True):
            owner = WorkflowId.new()
            for template in self._scope_templates(scope):
                node = NodeId.new(workflow=owner)
                self.node_ids[scope, template] = node
                self.node_identity[node] = (scope, template)

            sequence_pairs: set[tuple[Identity, Identity]] = set()
            for value in _array(self.support["edges"]):
                before, after = cast(list[str], value)
                sequence_pairs.add((self._support_identity(before), self._support_identity(after)))
            choices: list[ChoiceDecl] = []
            for value in _array(self.support["choices"]):
                choice = _object(value)
                selector_identity = self._support_identity(cast(str, choice["selector"]))
                if selector_identity[0] != scope:
                    continue
                branches: list[ChoiceBranch] = []
                for branch_value in _array(choice["branches"]):
                    branch = _object(branch_value)
                    members = [self._support_identity(cast(str, key)) for key in _array(branch["members"])]
                    branches.append(
                        ChoiceBranch(
                            outcomes=frozenset({cast(str, branch["outcome"])}),
                            members=frozenset(self.node_ids[member] for member in members),
                        )
                    )
                    sequence_pairs.update((selector_identity, member) for member in members)
                choices.append(ChoiceDecl(selector=self.node_ids[selector_identity], branches=tuple(branches)))
            for value in _array(self.support["aggregates"]):
                aggregate = _object(value)
                aggregate_scope = self._aggregate_scope(aggregate)
                if aggregate_scope != scope:
                    continue
                if aggregate["kind"] == "map":
                    source = self._support_identity(cast(str, aggregate["parent"]))
                    join = self._support_identity(cast(str, aggregate["join"]))
                    member = (
                        (
                            scope,
                            cast(str, self.support_seed_by_key[cast(str, _array(aggregate["members"])[0])]["template"]),
                        )
                        if _array(aggregate["members"])
                        else (scope, "N1")
                    )
                else:
                    source = self._support_identity(cast(str, aggregate["starter"]))
                    join = self._support_identity(cast(str, aggregate["join"]))
                    member = (scope, cast(str, aggregate["member_template"]))
                sequence_pairs.update(((source, member), (member, join)))

            dependencies = [_object(value) for value in _array(self.support["input_dependencies"])]
            input_bindings: list[InputBinding] = []
            for dependency in dependencies:
                source_identity = self._support_identity(cast(str, dependency["source"]))
                destination_identity = self._support_identity(cast(str, dependency["destination"]))
                if destination_identity[0] != scope:
                    continue
                input_bindings.append(
                    InputBinding(
                        source=NodeOutputRef(
                            node=self.node_ids[source_identity], port=cast(str, dependency["source_port"])
                        ),
                        destination=NodeInputRef(
                            node=self.node_ids[destination_identity],
                            port=cast(str, dependency["destination_port"]),
                        ),
                    )
                )
            for value in _array(self.support["aggregates"]):
                aggregate = _object(value)
                if aggregate["kind"] != "loop" or self._aggregate_scope(aggregate) != scope:
                    continue
                member = (scope, cast(str, aggregate["member_template"]))
                destination_port = self._input_names(member)[0]
                binding = InputBinding(
                    source=WorkflowInputRef(port=destination_port),
                    destination=NodeInputRef(node=self.node_ids[member], port=destination_port),
                )
                if binding not in input_bindings:
                    input_bindings.append(binding)

            nodes: list[OperationNode | SubgraphNode] = []
            for template in sorted(self._scope_templates(scope)):
                identity = (scope, template)
                node_id = self.node_ids[identity]
                body_scope = subgraph_bodies.get(identity)
                if body_scope is not None:
                    body = self.scopes[body_scope].workflow
                    nodes.append(SubgraphNode(id=node_id, operation=body.interface, body=body))
                    if body.interface.inputs:
                        for port in body.interface.inputs:
                            input_bindings.append(
                                InputBinding(
                                    source=WorkflowInputRef(port=port.name),
                                    destination=NodeInputRef(node=node_id, port=port.name),
                                )
                            )
                else:
                    nodes.append(
                        OperationNode(
                            id=node_id,
                            operation=self._operation(identity, inputs=self._input_names(identity)),
                        )
                    )

            local_pairs = {(left, right) for left, right in sequence_pairs if left[0] == scope and right[0] == scope}
            outgoing = {left for left, _ in local_pairs}
            sinks = [
                (scope, template) for template in self._scope_templates(scope) if (scope, template) not in outgoing
            ]
            if not sinks:
                sinks = [(scope, sorted(self._scope_templates(scope))[-1])]
            parent_identity = (scope[:-1], scope[-1]) if scope else None
            interface_records = (
                self._records(parent_identity)
                if parent_identity is not None
                else [record for sink in sinks for record in self._records(sink)]
            )
            unique_interface_records = {cast(str, record["name"]): record for record in interface_records}
            interface_inputs = tuple(
                sorted(
                    {
                        port.name
                        for binding in input_bindings
                        if isinstance(binding.source, WorkflowInputRef)
                        for port in (InputPort(name=binding.source.port, artifact_type=ARTIFACT),)
                    }
                )
            )
            interface_identity = parent_identity if parent_identity is not None else sinks[0]
            interface = self._operation(
                interface_identity,
                inputs=interface_inputs,
                ceiling=1000,
                dependency_inputs=False,
                expose_outputs=not choices,
            )
            sink_outputs = {port.name for port in interface.outputs}
            output_bindings = tuple(
                OutputBinding(
                    source=NodeOutputRef(node=self.node_ids[sink], port=port),
                    destination=WorkflowOutputRef(port=port),
                )
                for sink in sinks
                for port in sink_outputs
                if port
                in {
                    item.name
                    for item in cast(
                        Any, next(node for node in nodes if node.id == self.node_ids[sink])
                    ).operation.outputs
                }
            )
            outcome_bindings = tuple(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=self.node_ids[sink], outcome=name),
                    destination=WorkflowOutcomeRef(outcome=name),
                )
                for sink in sinks
                for name in unique_interface_records
                if name
                in {
                    item.name
                    for item in cast(
                        Any, next(node for node in nodes if node.id == self.node_ids[sink])
                    ).operation.outcomes
                }
            )
            branch_outcomes = {
                cast(str, _object(branch)["outcome"])
                for choice_value in _array(self.support["choices"])
                for branch in _array(_object(choice_value)["branches"])
            }
            selector_fallthrough = tuple(
                OutcomeBinding(
                    source=NodeOutcomeRef(node=choice.selector, outcome=outcome.name),
                    destination=WorkflowOutcomeRef(outcome=outcome.name),
                )
                for choice in choices
                for outcome in next(node for node in nodes if node.id == choice.selector).operation.outcomes
                if outcome.name not in branch_outcomes
            )
            outcome_bindings += selector_fallthrough
            expanded = sum(
                1 + (node.body.expanded_node_count if isinstance(node, SubgraphNode) else 0) for node in nodes
            )
            depth = max(
                (1 + self._depth(node.body) if isinstance(node, SubgraphNode) else 1 for node in nodes), default=1
            )
            workflow = admit_static_workflow(
                workflow=owner,
                interface=interface,
                nodes=tuple(nodes),
                input_bindings=tuple(input_bindings),
                output_bindings=output_bindings,
                outcome_bindings=outcome_bindings,
                sequence=tuple(
                    SequenceEdge(before=self.node_ids[left], after=self.node_ids[right]) for left, right in local_pairs
                ),
                choices=tuple(choices),
                protection=(),
                limits=WorkflowLimits(
                    max_nodes=max(1, expanded),
                    max_bindings=len(input_bindings) + len(output_bindings) + len(outcome_bindings),
                    max_sequence_edges=len(local_pairs),
                    max_choices=len(choices),
                    max_branch_members=sum(len(branch.members) for choice in choices for branch in choice.branches),
                    max_subgraph_depth=depth,
                    max_choice_states=max(1, 4 ** len(choices)),
                ),
            )
            self.scopes[scope] = _BuiltScope(
                workflow=workflow,
                nodes={template: self.node_ids[scope, template] for template in self._scope_templates(scope)},
            )
        return self.scopes[()].workflow

    def _depth(self, workflow: AdmittedWorkflow) -> int:
        maximum = 1
        pending = [(workflow, 1)]
        while pending:
            current, depth = pending.pop()
            maximum = max(maximum, depth)
            pending.extend((node.body, depth + 1) for node in current.nodes if isinstance(node, SubgraphNode))
        return maximum

    def build_dynamic(self) -> AdmittedActivationWorkflow:
        root = self.build_static()
        dynamic_declaration = self.declaration if self.case["boundary"] == "dynamic_admission" else self.support
        dynamic_scopes: list[DynamicScope] = []
        maps_count = joins_count = loops_count = 0
        max_children = max_iterations = 0
        for scope, built in self.scopes.items():
            maps: dict[Identity, MapDecl] = {}
            loops: dict[Identity, LoopDecl] = {}
            joins: dict[Identity, KeyedJoinDecl] = {}
            for value in _array(dynamic_declaration["aggregates"]):
                aggregate = _object(value)
                if self._aggregate_scope(aggregate) != scope:
                    continue
                if aggregate["kind"] == "map":
                    source_identity = self._support_identity(cast(str, aggregate["parent"]))
                    join_identity = self._support_identity(cast(str, aggregate["join"]))
                    members = _array(aggregate["members"])
                    member_identity = self._support_identity(cast(str, members[0])) if members else (scope, "N1")
                    maps[source_identity] = MapDecl(
                        expander=self.node_ids[source_identity],
                        member=self.node_ids[member_identity],
                        expansion_outcomes=frozenset(
                            cast(str, item) for item in _array(aggregate["expansion_outcomes"])
                        ),
                        max_children=cast(int, aggregate["bound"]),
                    )
                    joins[source_identity] = KeyedJoinDecl(
                        source=self.node_ids[source_identity],
                        join=self.node_ids[join_identity],
                        accepted_categories=frozenset(
                            cast(OutcomeClass, item) for item in _array(aggregate["accepted_categories"])
                        ),
                        reduction="all_by_key",
                    )
                    max_children = max(max_children, cast(int, aggregate["bound"]))
                else:
                    source_identity = self._support_identity(cast(str, aggregate["starter"]))
                    join_identity = self._support_identity(cast(str, aggregate["join"]))
                    member_identity = (scope, cast(str, aggregate["member_template"]))
                    initial_value = aggregate["initial_binding"]
                    carried_value = aggregate["carried_binding"]
                    initial = ()
                    if isinstance(initial_value, dict):
                        initial = (
                            LoopInitialBinding(
                                source=WorkflowInputRef(port=cast(str, initial_value["source_port"])),
                                destination=NodeInputRef(
                                    node=self.node_ids[member_identity],
                                    port=cast(str, initial_value["destination_port"]),
                                ),
                            ),
                        )
                    carried = ()
                    if isinstance(carried_value, dict):
                        carried = (
                            LoopCarriedBinding(
                                source=NodeOutputRef(
                                    node=self.node_ids[member_identity],
                                    port=cast(str, carried_value["source_port"]),
                                ),
                                destination=NodeInputRef(
                                    node=self.node_ids[member_identity],
                                    port=cast(str, carried_value["destination_port"]),
                                ),
                            ),
                        )
                    loops[source_identity] = LoopDecl(
                        starter=self.node_ids[source_identity],
                        member=self.node_ids[member_identity],
                        join=self.node_ids[join_identity],
                        enter_outcomes=frozenset(cast(str, item) for item in _array(aggregate["enter_outcomes"])),
                        bypass_outcomes=frozenset(cast(str, item) for item in _array(aggregate["bypass_outcomes"])),
                        continue_outcomes=frozenset(cast(str, item) for item in _array(aggregate["continue_outcomes"])),
                        exit_outcomes=frozenset(cast(str, item) for item in _array(aggregate["exit_outcomes"])),
                        initial=initial,
                        carried=carried,
                        max_iterations=cast(int, aggregate["bound"]),
                    )
                    joins[source_identity] = KeyedJoinDecl(
                        source=self.node_ids[source_identity],
                        join=self.node_ids[join_identity],
                        accepted_categories=frozenset({"success"}),
                        reduction="all_by_key",
                    )
                    max_iterations = max(max_iterations, cast(int, aggregate["bound"]))
            maps_count += len(maps)
            loops_count += len(loops)
            joins_count += len(joins)
            dynamic_scopes.append(
                DynamicScope(
                    workflow=built.workflow,
                    maps=tuple(maps.values()),
                    joins=tuple(joins.values()),
                    loops=tuple(loops.values()),
                )
            )
        self.workflow = admit_activation_workflow(
            workflow=root,
            scopes=tuple(dynamic_scopes),
            limits=DynamicLimits(
                max_maps=maps_count,
                max_joins=joins_count,
                max_loops=loops_count,
                max_children_per_map=max_children,
                max_iterations_per_loop=max_iterations,
                max_dynamic_depth=max(1, max((len(scope) + 1 for scope in self.scopes), default=1)),
                max_activation_occurrences=cast(int, _object(self.support["limits"])["max_entries"]),
            ),
        )
        return self.workflow

    def key(self, label: str, invocation_label: str | None = None) -> ActivationKey:
        seed = self.seed_by_key.get(label) or self.support_seed_by_key.get(label)
        if seed is None:
            return ActivationKey(
                invocation=self.invocation(invocation_label or cast(str, self.declaration["invocation"])),
                occurrence=int(label[1:]),
                parent=None,
                iteration=None,
            )
        actual_invocation = invocation_label or cast(str, seed["invocation"])
        cache_key = (label, actual_invocation)
        if cache_key not in self.keys:
            parent_label = seed["parent"]
            self.keys[cache_key] = ActivationKey(
                invocation=self.invocation(actual_invocation),
                occurrence=int(label[1:]),
                parent=None if parent_label is None else self.key(cast(str, parent_label), actual_invocation),
                iteration=cast(int | None, seed["iteration"]),
            )
            self.activation_labels[self.keys[cache_key]] = label
        return self.keys[cache_key]

    def reservations(self) -> frozenset[ActivationSeed]:
        return frozenset(
            ActivationSeed(template=self._node_for_seed(seed), activation=self.key(cast(str, seed["key"])))
            for seed in self.seed_records
        )

    def initialize(self) -> ActivationState:
        assert self.workflow is not None
        limits = _object(self.declaration["limits"])
        return initialize_activation(
            workflow=self.workflow,
            invocation=self.invocation(cast(str, self.declaration["invocation"])),
            reservations=self.reservations(),
            limits=ActivationLimits(
                max_events=cast(int, limits["max_events"]),
                max_entries=cast(int, limits["max_entries"]),
                max_parent_depth=cast(int, limits["max_parent_depth"]),
            ),
        )

    def event(self, value: Object):
        kind = value["kind"]
        invocation = cast(str, value.get("invocation", self.declaration["invocation"]))
        if kind == "select":
            seeds = []
            for key_value in _array(value["keys"]):
                label = cast(str, key_value)
                seed = next((seed for seed in self.seed_records if seed["key"] == label), None)
                template = (
                    self._node_for_seed(seed)
                    if seed is not None
                    else self.node_ids[next(identity for identity in self.node_ids if not identity[0])]
                )
                seeds.append(
                    ActivationSeed(
                        template=template,
                        activation=self.key(label, invocation),
                    )
                )
            return Select(seeds=frozenset(seeds))
        if kind == "start":
            return Start(activation=self.key(cast(str, value["key"]), invocation))
        if cast(str, kind).startswith("terminal_"):
            return ObserveTerminal(
                activation=self.key(cast(str, value["key"]), invocation),
                outcome=cast(str | None, value["outcome"]),
                category=cast(OutcomeClass, cast(str, kind).removeprefix("terminal_")),
            )
        if cast(str, kind).startswith("close_"):
            return CloseUnstarted(
                activation=self.key(cast(str, value["key"]), invocation),
                category=cast(Any, cast(str, kind).removeprefix("close_")),
            )
        if kind in {"membership_open", "membership_close"}:
            return ObserveMembership(
                parent=self.key(cast(str, value["parent"]), invocation),
                members=frozenset(self.key(cast(str, item), invocation) for item in _array(value["members"])),
                closed=kind == "membership_close",
            )
        if kind == "membership_overflow":
            parent = value["parent"]
            return ObserveOverflow(
                parent=self.key(parent, invocation) if isinstance(parent, str) else cast(Any, parent),
                observed_count=cast(int, value["observed_count"]),
            )
        raise AssertionError(f"unknown event kind: {kind}")

    def state(self, state: ActivationState) -> Object:
        entries: list[Json] = []
        outputs: list[str] = []
        for entry in state.entries:
            label = self.activation_labels[entry.activation]
            scope, template = self.node_identity[entry.template]
            produced: list[str] = []
            if entry.outcome is not None:
                operation = next(
                    node.operation
                    for node in _scope_for_state(state, entry.template).workflow.nodes
                    if node.id == entry.template
                )
                outcome = next(item for item in operation.outcomes if item.name == entry.outcome)
                produced = sorted(outcome.produced_ports)
            if produced:
                outputs.append(label)
            entries.append(
                {
                    "activation": label,
                    "scope": list(scope),
                    "template": template,
                    "status": entry.status,
                    "category": entry.status
                    if entry.status in {"success", "failure", "cancelled", "lost", "blocked", "inconsistent"}
                    else None,
                    "outcome": entry.outcome,
                    "produced_ports": produced,
                }
            )
        result: Object = {
            "complete": state.complete,
            "entries": cast(Json, sorted(entries, key=lambda item: cast(str, cast(Object, item)["activation"]))),
            "events_applied": state.events_applied,
            "expansions": cast(
                Json,
                sorted(
                    (
                        {
                            "parent": self.activation_labels[item.parent],
                            "members": sorted(self.activation_labels[member] for member in item.members),
                            "status": item.status,
                        }
                        for item in state.expansions
                    ),
                    key=lambda item: item["parent"],
                ),
            ),
            "outputs": cast(Json, sorted(outputs)),
        }
        result["semantic_hash"] = hashlib.sha256(_canonical(cast(Json, result))).hexdigest()
        return result

    def reduce(self, events: list[Object]) -> Object:
        boundary = cast(str, self.case["boundary"])
        try:
            if boundary == "static_admission":
                self.build_static()
                return {"status": "accepted", "code": None, "state": None}
            self.build_dynamic()
            if boundary == "dynamic_admission":
                return {"status": "accepted", "code": None, "state": None}
            if boundary == "event_construction":
                self.event(events[-1])
                return {"status": "accepted", "code": None, "state": None}
            state = self.initialize()
            if boundary == "initialization":
                return {"status": "accepted", "code": None, "state": self.state(state)}
            for value in events:
                if value["kind"] == "initialize":
                    continue
                state = advance_activation(state=state, event=self.event(value))
            return {"status": "accepted", "code": None, "state": self.state(state)}
        except ContractViolation as error:
            return {"status": "rejected", "code": error.code.value, "state": None}


def _scope_for_state(state: ActivationState, template: NodeId) -> DynamicScope:
    return next(scope for scope in state.workflow.scopes if any(node.id == template for node in scope.workflow.nodes))


@pytest.mark.parametrize(
    ("_execution_id", "case"),
    EXECUTIONS,
    ids=[execution_id for execution_id, _ in EXECUTIONS],
)
def test_frozen_activation_case_matches_product(_execution_id: str, case: Object) -> None:
    adapter = _Adapter(case)
    events = [_object(value) for value in _array(case["events"])]
    assert adapter.reduce(events) == case["expected"]


def _transition_state(adapter: _Adapter, events: list[Json]) -> ActivationState:
    if adapter.workflow is None:
        adapter.build_dynamic()
    state = adapter.initialize()
    for event_value in events:
        event = _object(event_value)
        if event["kind"] != "initialize":
            state = advance_activation(state=state, event=adapter.event(event))
    return state


@pytest.mark.parametrize("case", COMMUTED_CASES, ids=lambda case: cast(str, case["case_id"]))
def test_commuted_trace_uses_same_product_identity_mapping(case: Object) -> None:
    trace = next(
        _object(value) for value in _array(case["traces"]) if _object(value)["name"] == "commute_independent_siblings"
    )
    assert trace["declaration"] == case["declaration"]
    adapter = _Adapter(case)
    original = _transition_state(adapter, _array(case["events"]))
    commuted = _transition_state(adapter, _array(trace["events"]))
    assert original == commuted
    assert hash(original) == hash(commuted)
    assert adapter.state(original)["semantic_hash"] == adapter.state(commuted)["semantic_hash"]


def test_activation_state_set_order_preserves_equality_and_hash() -> None:
    case = _case("map/007/bound_2_size_2_ss")
    adapter = _Adapter(case)
    state = _transition_state(adapter, _array(case["events"]))
    reordered = ActivationState(
        workflow=state.workflow,
        invocation=state.invocation,
        entries=frozenset(reversed(tuple(state.entries))),
        expansions=frozenset(reversed(tuple(state.expansions))),
        reservations=frozenset(reversed(tuple(state.reservations))),
        events_applied=state.events_applied,
        limits=state.limits,
    )
    assert state == reordered
    assert hash(state) == hash(reordered)
