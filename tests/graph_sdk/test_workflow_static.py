# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Conformance tests for pure static workflow composition."""

from __future__ import annotations

import copy
import hashlib
import json
import pickle
from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import TypeAlias, cast

import pytest

from anonymizer.graph._values import ContractViolation, ValidationCode
from anonymizer.graph.workflow import (
    AdmittedWorkflow,
    ArtifactType,
    CaptureMode,
    ChoiceBranch,
    ChoiceDecl,
    ContextUse,
    CoverageAtom,
    CoverageKind,
    EvidencePromise,
    InputBinding,
    InputPort,
    ModelRequirement,
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
    ProtectionRequirement,
    ResourceCeiling,
    SequenceEdge,
    StateEffect,
    StateEffectKind,
    SubgraphNode,
    WorkflowId,
    WorkflowInputRef,
    WorkflowLimits,
    WorkflowOutcomeRef,
    WorkflowOutputRef,
    admit_static_workflow,
    substitute,
)

Json: TypeAlias = str | int | bool | None | list["Json"] | dict[str, "Json"]
Object: TypeAlias = dict[str, Json]
REFERENCE_DIR = Path(__file__).parent / "reference"
CORPUS_PATH = REFERENCE_DIR / "workflow_static_v1_cases.json"
MANIFEST_PATH = REFERENCE_DIR / "workflow_static_v1_manifest.json"
FROZEN_BYTES = CORPUS_PATH.read_bytes()
CASES = cast(Object, json.loads(FROZEN_BYTES))["cases"]
MANIFEST = cast(Object, json.loads(MANIFEST_PATH.read_bytes()))


def _object(value: Json) -> Object:
    if not isinstance(value, dict):
        raise TypeError("adapter expected an object")
    return value


def _array(value: Json) -> list[Json]:
    if not isinstance(value, list):
        raise TypeError("adapter expected an array")
    return value


def _string(value: Json) -> str:
    if not isinstance(value, str):
        raise TypeError("adapter expected a string")
    return value


def _integer(value: Json) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("adapter expected an integer")
    return value


def _canonical(value: Json) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()


def _sorted(values: list[Json]) -> list[Json]:
    return sorted(values, key=_canonical)


class _Adapter:
    """Translate neutral labels and representation without deciding validity."""

    def __init__(self) -> None:
        self.workflows: dict[str, WorkflowId] = {}
        self.workflow_labels: dict[WorkflowId, str] = {}
        self.nodes: dict[tuple[str, str], NodeId] = {}
        self.node_labels: dict[NodeId, tuple[str, str]] = {}
        self.declarations: dict[AdmittedWorkflow, Object] = {}
        self.operation_representations: dict[OperationSpec, Object] = {}
        self.requirement_representations: dict[ProtectionRequirement, Object] = {}

    def workflow(self, label: str) -> WorkflowId:
        if label not in self.workflows:
            identity = WorkflowId.new()
            self.workflows[label] = identity
            self.workflow_labels[identity] = label
        return self.workflows[label]

    def node(self, value: Json) -> NodeId:
        item = _object(value)
        owner = _string(item["owner"])
        label = _string(item["label"])
        key = (owner, label)
        if key not in self.nodes:
            identity = NodeId.new(workflow=self.workflow(owner))
            self.nodes[key] = identity
            self.node_labels[identity] = key
        return self.nodes[key]

    def artifact_type(self, value: Json) -> ArtifactType:
        name, revision = _string(value).rsplit("@", 1)
        return ArtifactType(name=name, revision=int(revision))

    def operation(self, value: Json) -> OperationSpec:
        item = _object(value)
        operation = OperationSpec(
            name=_string(item["name"]),
            inputs=tuple(
                InputPort(name=_string(port["name"]), artifact_type=self.artifact_type(port["artifact_type"]))
                for port in map(_object, _array(item["inputs"]))
            ),
            outputs=tuple(
                OutputPort(name=_string(port["name"]), artifact_type=self.artifact_type(port["artifact_type"]))
                for port in map(_object, _array(item["outputs"]))
            ),
            output_dependencies=tuple(self.dependency(value) for value in _array(item["output_dependencies"])),
            outcomes=tuple(self.outcome(value) for value in _array(item["outcomes"])),
        )
        self.operation_representations[operation] = copy.deepcopy(item)
        return operation

    def dependency(self, value: Json) -> OutputDependency:
        item = _object(value)
        identity = item["identity_input"]
        return OutputDependency(
            output=_string(item["output"]),
            inputs=frozenset(_string(member) for member in _array(item["inputs"])),
            identity_input=None if identity is None else _string(identity),
        )

    def outcome(self, value: Json) -> OutcomeSpec:
        item = _object(value)
        ceiling = _object(item["ceiling"])
        return OutcomeSpec(
            name=_string(item["name"]),
            category=cast(OutcomeClass, _string(item["category"])),
            produced_ports=frozenset(_string(member) for member in _array(item["produced_ports"])),
            context=frozenset(self.context(member) for member in _array(item["context"])),
            evidence=frozenset(self.evidence(member) for member in _array(item["evidence"])),
            state_effects=frozenset(self.state(member) for member in _array(item["state_effects"])),
            model_requirements=frozenset(self.model(member) for member in _array(item["model_requirements"])),
            ceiling=ResourceCeiling(
                max_activations=_integer(ceiling["max_activations"]),
                max_model_requests=_integer(ceiling["max_model_requests"]),
                max_input_bytes=_integer(ceiling["max_input_bytes"]),
                max_output_bytes=_integer(ceiling["max_output_bytes"]),
            ),
        )

    def context(self, value: Json) -> ContextUse:
        item = _object(value)
        return ContextUse(
            port=_string(item["port"]),
            meaning=_string(item["meaning"]),
            capture=cast(CaptureMode, _string(item["capture"])),
        )

    def coverage(self, value: Json) -> CoverageAtom:
        item = _object(value)
        return CoverageAtom(kind=cast(CoverageKind, _string(item["kind"])), name=_string(item["name"]))

    def evidence(self, value: Json) -> EvidencePromise:
        item = _object(value)
        return EvidencePromise(
            name=_string(item["name"]),
            meaning=_string(item["meaning"]),
            subject_port=_string(item["subject_port"]),
            consumed_ports=frozenset(_string(member) for member in _array(item["consumed_ports"])),
            coverage=frozenset(self.coverage(member) for member in _array(item["coverage"])),
        )

    def state(self, value: Json) -> StateEffect:
        item = _object(value)
        return StateEffect(kind=cast(StateEffectKind, _string(item["kind"])), name=_string(item["name"]))

    def model(self, value: Json) -> ModelRequirement:
        item = _object(value)
        return ModelRequirement(capability=_string(item["capability"]), revision=_integer(item["revision"]))

    def source(self, value: Json) -> WorkflowInputRef | NodeOutputRef:
        item = _object(value)
        if item["kind"] == "workflow_input":
            return WorkflowInputRef(port=_string(item["port"]))
        return NodeOutputRef(node=self.node(item["node"]), port=_string(item["port"]))

    def limits(self, value: Json) -> WorkflowLimits:
        item = _object(value)
        return WorkflowLimits(
            max_nodes=_integer(item["max_nodes"]),
            max_bindings=_integer(item["max_bindings"]),
            max_sequence_edges=_integer(item["max_sequence_edges"]),
            max_choices=_integer(item["max_choices"]),
            max_branch_members=_integer(item["max_branch_members"]),
            max_subgraph_depth=_integer(item["max_subgraph_depth"]),
            max_choice_states=_integer(item["max_choice_states"]),
        )

    def admit(self, value: Json) -> AdmittedWorkflow:
        item = _object(value)
        workflow = self.workflow(_string(item["workflow"]))
        nodes: list[OperationNode | SubgraphNode] = []
        for node_value in map(_object, _array(item["nodes"])):
            identity = self.node(node_value["id"])
            operation = self.operation(node_value["operation"])
            if node_value["kind"] == "operation":
                nodes.append(OperationNode(id=identity, operation=operation))
            else:
                nodes.append(SubgraphNode(id=identity, operation=operation, body=self.admit(node_value["body"])))
        admitted = admit_static_workflow(
            workflow=workflow,
            interface=self.operation(item["interface"]),
            nodes=tuple(nodes),
            input_bindings=tuple(
                InputBinding(
                    source=self.source(binding["source"]),
                    destination=NodeInputRef(
                        node=self.node(_object(binding["destination"])["node"]),
                        port=_string(_object(binding["destination"])["port"]),
                    ),
                )
                for binding in map(_object, _array(item["input_bindings"]))
            ),
            output_bindings=tuple(
                OutputBinding(
                    source=self.source(binding["source"]),
                    destination=WorkflowOutputRef(port=_string(_object(binding["destination"])["port"])),
                )
                for binding in map(_object, _array(item["output_bindings"]))
            ),
            outcome_bindings=tuple(
                OutcomeBinding(
                    source=NodeOutcomeRef(
                        node=self.node(_object(binding["source"])["node"]),
                        outcome=_string(_object(binding["source"])["outcome"]),
                    ),
                    destination=WorkflowOutcomeRef(outcome=_string(_object(binding["destination"])["outcome"])),
                )
                for binding in map(_object, _array(item["outcome_bindings"]))
            ),
            sequence=tuple(
                SequenceEdge(before=self.node(edge["before"]), after=self.node(edge["after"]))
                for edge in map(_object, _array(item["sequence"]))
            ),
            choices=tuple(
                ChoiceDecl(
                    selector=self.node(choice["selector"]),
                    branches=tuple(
                        ChoiceBranch(
                            outcomes=frozenset(_string(member) for member in _array(branch["outcomes"])),
                            members=frozenset(self.node(member) for member in _array(branch["members"])),
                        )
                        for branch in map(_object, _array(choice["branches"]))
                    ),
                )
                for choice in map(_object, _array(item["choices"]))
            ),
            protection=tuple(self.requirement(value) for value in _array(item["protection"])),
            limits=self.limits(item["limits"]),
        )
        self.declarations[admitted] = copy.deepcopy(item)
        return admitted

    def requirement(self, value: Json) -> ProtectionRequirement:
        item = _object(value)
        requirement = ProtectionRequirement(
            outcome=_string(item["outcome"]),
            meaning=_string(item["meaning"]),
            subject_port=_string(item["subject_port"]),
            consumed_ports=frozenset(_string(member) for member in _array(item["consumed_ports"])),
            coverage=frozenset(self.coverage(member) for member in _array(item["coverage"])),
        )
        self.requirement_representations[requirement] = copy.deepcopy(item)
        return requirement

    def identity(self, node: NodeId) -> Object:
        owner, label = self.node_labels[node]
        return {"label": label, "owner": owner}

    def operation_json(self, operation: OperationSpec) -> Object:
        representation = self.operation_representations.get(operation)
        if representation is not None:
            return copy.deepcopy(representation)
        return {
            "inputs": [
                {"artifact_type": f"{port.artifact_type.name}@{port.artifact_type.revision}", "name": port.name}
                for port in operation.inputs
            ],
            "name": operation.name,
            "outcomes": [self.outcome_json(outcome) for outcome in operation.outcomes],
            "output_dependencies": [self.dependency_json(value) for value in operation.output_dependencies],
            "outputs": [
                {"artifact_type": f"{port.artifact_type.name}@{port.artifact_type.revision}", "name": port.name}
                for port in operation.outputs
            ],
        }

    def outcome_json(self, outcome: OutcomeSpec) -> Object:
        return {
            "category": outcome.category,
            "ceiling": {
                "max_activations": outcome.ceiling.max_activations,
                "max_input_bytes": outcome.ceiling.max_input_bytes,
                "max_model_requests": outcome.ceiling.max_model_requests,
                "max_output_bytes": outcome.ceiling.max_output_bytes,
            },
            "context": _sorted(
                [{"capture": use.capture, "meaning": use.meaning, "port": use.port} for use in outcome.context]
            ),
            "evidence": _sorted([self.evidence_json(value) for value in outcome.evidence]),
            "model_requirements": _sorted(
                [{"capability": value.capability, "revision": value.revision} for value in outcome.model_requirements]
            ),
            "name": outcome.name,
            "produced_ports": sorted(outcome.produced_ports),
            "state_effects": _sorted([{"kind": value.kind, "name": value.name} for value in outcome.state_effects]),
        }

    def dependency_json(self, value: OutputDependency) -> Object:
        return {"identity_input": value.identity_input, "inputs": sorted(value.inputs), "output": value.output}

    def evidence_json(self, value: EvidencePromise) -> Object:
        return {
            "consumed_ports": sorted(value.consumed_ports),
            "coverage": _sorted([{"kind": item.kind, "name": item.name} for item in value.coverage]),
            "meaning": value.meaning,
            "name": value.name,
            "subject_port": value.subject_port,
        }

    def source_json(self, value: WorkflowInputRef | NodeOutputRef) -> Object:
        if isinstance(value, WorkflowInputRef):
            return {"kind": "workflow_input", "port": value.port}
        return {"kind": "node_output", "node": self.identity(value.node), "port": value.port}

    def requirement_json(self, value: ProtectionRequirement) -> Object:
        representation = self.requirement_representations.get(value)
        if representation is not None:
            return copy.deepcopy(representation)
        return {
            "consumed_ports": sorted(value.consumed_ports),
            "coverage": _sorted([{"kind": item.kind, "name": item.name} for item in value.coverage]),
            "meaning": value.meaning,
            "outcome": value.outcome,
            "subject_port": value.subject_port,
        }

    def node_json(self, value: OperationNode | SubgraphNode) -> Object:
        result: Object = {
            "id": self.identity(value.id),
            "kind": "subgraph" if isinstance(value, SubgraphNode) else "operation",
            "operation": self.operation_json(value.operation),
        }
        if isinstance(value, SubgraphNode):
            result["body"] = copy.deepcopy(self.declarations[value.body])
        return result

    def declaration_json(self, workflow: AdmittedWorkflow) -> Object:
        return {
            **self.normalized(workflow),
            "limits": {
                "max_bindings": workflow.limits.max_bindings,
                "max_branch_members": workflow.limits.max_branch_members,
                "max_choice_states": workflow.limits.max_choice_states,
                "max_choices": workflow.limits.max_choices,
                "max_nodes": workflow.limits.max_nodes,
                "max_sequence_edges": workflow.limits.max_sequence_edges,
                "max_subgraph_depth": workflow.limits.max_subgraph_depth,
            },
            "protection": _sorted([self.requirement_json(value) for value in workflow.protection_requirements]),
            "workflow": self.workflow_labels[workflow.workflow],
        }

    def normalized(self, workflow: AdmittedWorkflow) -> Object:
        return {
            "choices": _sorted(
                [
                    {
                        "branches": [
                            {
                                "members": _sorted([self.identity(member) for member in branch.members]),
                                "outcomes": sorted(branch.outcomes),
                            }
                            for branch in choice.branches
                        ],
                        "selector": self.identity(choice.selector),
                    }
                    for choice in workflow.choices
                ]
            ),
            "expanded_node_count": workflow.expanded_node_count,
            "input_bindings": _sorted(
                [
                    {
                        "destination": {
                            "node": self.identity(binding.destination.node),
                            "port": binding.destination.port,
                        },
                        "source": self.source_json(binding.source),
                    }
                    for binding in workflow.input_bindings
                ]
            ),
            "interface": self.operation_json(workflow.interface),
            "nodes": _sorted([self.node_json(node) for node in workflow.nodes]),
            "outcome_bindings": _sorted(
                [
                    {
                        "destination": {"outcome": binding.destination.outcome},
                        "source": {"node": self.identity(binding.source.node), "outcome": binding.source.outcome},
                    }
                    for binding in workflow.outcome_bindings
                ]
            ),
            "output_bindings": _sorted(
                [
                    {
                        "destination": {"port": binding.destination.port},
                        "source": self.source_json(binding.source),
                    }
                    for binding in workflow.output_bindings
                ]
            ),
            "protection_requirements": _sorted(
                [self.requirement_json(value) for value in workflow.protection_requirements]
            ),
            "sequence": _sorted(
                [
                    {"after": self.identity(edge.after), "before": self.identity(edge.before)}
                    for edge in workflow.sequence
                ]
            ),
        }


def _execute(declaration: Json, replacement: Json) -> tuple[str, str | None, Object | None, list[str], list[Json]]:
    adapter = _Adapter()
    try:
        admitted = adapter.admit(declaration)
        item = _object(declaration)
        if replacement is not None:
            target = adapter.node(item["substitution_target"])
            admitted = substitute(workflow=admitted, target=target, replacement=adapter.admit(replacement))
    except ContractViolation as error:
        return "rejected", error.code.value, None, [], []
    return (
        "accepted",
        None,
        adapter.normalized(admitted),
        sorted(admitted.protection_eligible_outcomes),
        _sorted([adapter.requirement_json(value) for value in admitted.unmet_protection]),
    )


def _assert_case(case: Object, *, suffix: str = "") -> None:
    expected = _object(case["expected"])
    actual = _execute(case["declaration"], case["replacement"])
    wanted = (
        expected["status"],
        expected["code"],
        expected["normalized"],
        expected["protection_eligible_outcomes"],
        expected["unmet_protection"],
    )
    assert actual == wanted, f"{case['case_id']}{suffix}"


def test_all_frozen_cases_and_traces_match_product_boundaries() -> None:
    assert isinstance(CASES, list)
    compared_cases = 0
    compared_traces = 0
    for value in CASES:
        case = _object(value)
        _assert_case(case)
        compared_cases += 1
        for trace_value in _array(case["traces"]):
            trace = _object(trace_value)
            traced_case = dict(case)
            traced_case["declaration"] = trace["declaration"]
            traced_case["replacement"] = trace["replacement"]
            traced_case["expected"] = trace["expected"]
            _assert_case(traced_case, suffix=f"/{trace['transformation']}")
            compared_traces += 1
    assert compared_cases == 526
    assert compared_traces == 962


def _zero_operation(name: str = "private-value") -> OperationSpec:
    ceiling = ResourceCeiling(max_activations=1, max_model_requests=0, max_input_bytes=0, max_output_bytes=0)
    return OperationSpec(
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
                state_effects=frozenset(),
                model_requirements=frozenset(),
                ceiling=ceiling,
            ),
        ),
    )


def _one_node_factory(operation_factory: Callable[[], OperationSpec]) -> AdmittedWorkflow:
    workflow = WorkflowId.new()
    node = NodeId.new(workflow=workflow)
    operation = operation_factory()
    return admit_static_workflow(
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
        choices=(),
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


def test_values_are_private_immutable_and_copy_stable() -> None:
    operation = _zero_operation("secret-operation")
    admitted = _one_node_factory(lambda: operation)
    workflow = WorkflowId.new()
    node = NodeId.new(workflow=workflow)
    assert "secret-operation" not in repr(operation)
    assert "secret-operation" not in repr(admitted)
    assert copy.copy(operation) is operation
    assert copy.deepcopy(operation) is operation
    assert copy.copy(admitted) is admitted
    assert copy.deepcopy(admitted) is admitted
    assert copy.copy(workflow) is workflow
    assert copy.deepcopy(node) is node
    assert "secret-operation" not in repr(workflow)
    with pytest.raises(FrozenInstanceError):
        setattr(operation, "name", "changed")
    with pytest.raises(TypeError, match="serialization is not supported"):
        pickle.dumps(admitted)
    with pytest.raises(TypeError):
        replace(admitted, expanded_node_count=99)
    with pytest.raises(ContractViolation) as invalid_replacement:
        replace(ArtifactType(name="private-artifact", revision=1), revision=0)
    assert invalid_replacement.value.code is ValidationCode.INVALID_VALUE


def test_factories_share_the_same_pure_admission_boundary() -> None:
    effects = 0

    def external_factory() -> OperationSpec:
        nonlocal effects
        effects += 1
        return _zero_operation()

    external = _one_node_factory(external_factory)
    built_in_shaped = _one_node_factory(_zero_operation)
    assert effects == 1
    assert external.interface == built_in_shaped.interface
    assert external.expanded_node_count == built_in_shaped.expanded_node_count == 1


def test_malformed_values_reach_real_boundaries_and_errors_are_private() -> None:
    secret = "rejected-sensitive-name"
    with pytest.raises(ContractViolation) as captured:
        ArtifactType(name=secret, revision=True)
    assert captured.value.code is ValidationCode.INVALID_TYPE
    assert secret not in str(captured.value)
    assert secret not in repr(captured.value)
    workflow = _one_node_factory(_zero_operation)
    foreign = NodeId.new(workflow=WorkflowId.new())
    with pytest.raises(ContractViolation) as foreign_error:
        substitute(workflow=workflow, target=foreign, replacement=workflow)
    assert foreign_error.value.code is ValidationCode.FOREIGN_OWNER


def test_frozen_reference_bytes_are_unchanged() -> None:
    assert hashlib.sha256(FROZEN_BYTES).hexdigest() == MANIFEST["corpus_sha256"]
