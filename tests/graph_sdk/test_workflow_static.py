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
from typing import TypeAlias, TypeVar, cast

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
T = TypeVar("T")
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


def _label(value: Json) -> str:
    if not isinstance(value, str):
        raise TypeError("adapter expected an identity label")
    return value


def _product_tuple(value: Json, convert: Callable[[Json], T]) -> tuple[T, ...]:
    if isinstance(value, list):
        return tuple(convert(member) for member in value)
    return cast(tuple[T, ...], value)


def _product_frozenset(value: Json, convert: Callable[[Json], T]) -> frozenset[T]:
    if isinstance(value, list):
        return frozenset(convert(member) for member in value)
    return cast(frozenset[T], value)


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
        owner = _label(item["owner"])
        label = _label(item["label"])
        key = (owner, label)
        if key not in self.nodes:
            identity = NodeId.new(workflow=self.workflow(owner))
            self.nodes[key] = identity
            self.node_labels[identity] = key
        return self.nodes[key]

    def artifact_type(self, value: Json) -> ArtifactType:
        if isinstance(value, str):
            name, revision = value.rsplit("@", 1)
            return ArtifactType(name=name, revision=int(revision))
        return ArtifactType(name=cast(str, value), revision=1)

    def operation(self, value: Json) -> OperationSpec:
        if not isinstance(value, dict):
            return cast(OperationSpec, value)
        item = _object(value)
        operation = OperationSpec(
            name=cast(str, item["name"]),
            inputs=_product_tuple(item["inputs"], self.input_port),
            outputs=_product_tuple(item["outputs"], self.output_port),
            output_dependencies=_product_tuple(item["output_dependencies"], self.dependency),
            outcomes=_product_tuple(item["outcomes"], self.outcome),
        )
        self.operation_representations[operation] = copy.deepcopy(item)
        return operation

    def input_port(self, value: Json) -> InputPort:
        if not isinstance(value, dict):
            return cast(InputPort, value)
        return InputPort(name=cast(str, value["name"]), artifact_type=self.artifact_type(value["artifact_type"]))

    def output_port(self, value: Json) -> OutputPort:
        if not isinstance(value, dict):
            return cast(OutputPort, value)
        return OutputPort(name=cast(str, value["name"]), artifact_type=self.artifact_type(value["artifact_type"]))

    def dependency(self, value: Json) -> OutputDependency:
        if not isinstance(value, dict):
            return cast(OutputDependency, value)
        item = _object(value)
        identity = item["identity_input"]
        return OutputDependency(
            output=cast(str, item["output"]),
            inputs=_product_frozenset(item["inputs"], lambda member: cast(str, member)),
            identity_input=None if identity is None else cast(str, identity),
        )

    def outcome(self, value: Json) -> OutcomeSpec:
        if not isinstance(value, dict):
            return cast(OutcomeSpec, value)
        item = _object(value)
        ceiling_value = item["ceiling"]
        ceiling = (
            ResourceCeiling(
                max_activations=cast(int, ceiling_value["max_activations"]),
                max_model_requests=cast(int, ceiling_value["max_model_requests"]),
                max_input_bytes=cast(int, ceiling_value["max_input_bytes"]),
                max_output_bytes=cast(int, ceiling_value["max_output_bytes"]),
            )
            if isinstance(ceiling_value, dict)
            else cast(ResourceCeiling, ceiling_value)
        )
        return OutcomeSpec(
            name=cast(str, item["name"]),
            category=cast(OutcomeClass, item["category"]),
            produced_ports=_product_frozenset(item["produced_ports"], lambda member: cast(str, member)),
            context=_product_frozenset(item["context"], self.context),
            evidence=_product_frozenset(item["evidence"], self.evidence),
            state_effects=_product_frozenset(item["state_effects"], self.state),
            model_requirements=_product_frozenset(item["model_requirements"], self.model),
            ceiling=ceiling,
        )

    def context(self, value: Json) -> ContextUse:
        if not isinstance(value, dict):
            return cast(ContextUse, value)
        item = _object(value)
        return ContextUse(
            port=cast(str, item["port"]),
            meaning=cast(str, item["meaning"]),
            capture=cast(CaptureMode, item["capture"]),
        )

    def coverage(self, value: Json) -> CoverageAtom:
        if not isinstance(value, dict):
            return cast(CoverageAtom, value)
        item = _object(value)
        return CoverageAtom(kind=cast(CoverageKind, item["kind"]), name=cast(str, item["name"]))

    def evidence(self, value: Json) -> EvidencePromise:
        if not isinstance(value, dict):
            return cast(EvidencePromise, value)
        item = _object(value)
        return EvidencePromise(
            name=cast(str, item["name"]),
            meaning=cast(str, item["meaning"]),
            subject_port=cast(str, item["subject_port"]),
            consumed_ports=_product_frozenset(item["consumed_ports"], lambda member: cast(str, member)),
            coverage=_product_frozenset(item["coverage"], self.coverage),
        )

    def state(self, value: Json) -> StateEffect:
        if not isinstance(value, dict):
            return cast(StateEffect, value)
        item = _object(value)
        return StateEffect(kind=cast(StateEffectKind, item["kind"]), name=cast(str, item["name"]))

    def model(self, value: Json) -> ModelRequirement:
        if not isinstance(value, dict):
            return cast(ModelRequirement, value)
        item = _object(value)
        return ModelRequirement(capability=cast(str, item["capability"]), revision=cast(int, item["revision"]))

    def source(self, value: Json) -> WorkflowInputRef | NodeOutputRef:
        if not isinstance(value, dict):
            return cast(WorkflowInputRef, value)
        item = _object(value)
        if item["kind"] == "workflow_input":
            return WorkflowInputRef(port=cast(str, item["port"]))
        return NodeOutputRef(node=self.node(item["node"]), port=cast(str, item["port"]))

    def limits(self, value: Json) -> WorkflowLimits:
        if not isinstance(value, dict):
            return cast(WorkflowLimits, value)
        item = _object(value)
        return WorkflowLimits(
            max_nodes=cast(int, item["max_nodes"]),
            max_bindings=cast(int, item["max_bindings"]),
            max_sequence_edges=cast(int, item["max_sequence_edges"]),
            max_choices=cast(int, item["max_choices"]),
            max_branch_members=cast(int, item["max_branch_members"]),
            max_subgraph_depth=cast(int, item["max_subgraph_depth"]),
            max_choice_states=cast(int, item["max_choice_states"]),
        )

    def admit(self, value: Json) -> AdmittedWorkflow:
        item = _object(value)
        workflow = self.workflow(_label(item["workflow"]))
        admitted = admit_static_workflow(
            workflow=workflow,
            interface=self.operation(item["interface"]),
            nodes=_product_tuple(item["nodes"], self.node_declaration),
            input_bindings=_product_tuple(item["input_bindings"], self.input_binding),
            output_bindings=_product_tuple(item["output_bindings"], self.output_binding),
            outcome_bindings=_product_tuple(item["outcome_bindings"], self.outcome_binding),
            sequence=_product_tuple(item["sequence"], self.sequence_edge),
            choices=_product_tuple(item["choices"], self.choice),
            protection=_product_tuple(item["protection"], self.requirement),
            limits=self.limits(item["limits"]),
        )
        self.declarations[admitted] = copy.deepcopy(item)
        return admitted

    def node_declaration(self, value: Json) -> OperationNode | SubgraphNode:
        if not isinstance(value, dict):
            return cast(OperationNode, value)
        identity = self.node(value["id"])
        operation = self.operation(value["operation"])
        if value["kind"] == "operation":
            return OperationNode(id=identity, operation=operation)
        return SubgraphNode(id=identity, operation=operation, body=self.admit(value["body"]))

    def input_binding(self, value: Json) -> InputBinding:
        if not isinstance(value, dict):
            return cast(InputBinding, value)
        destination = _object(value["destination"])
        return InputBinding(
            source=self.source(value["source"]),
            destination=NodeInputRef(node=self.node(destination["node"]), port=cast(str, destination["port"])),
        )

    def output_binding(self, value: Json) -> OutputBinding:
        if not isinstance(value, dict):
            return cast(OutputBinding, value)
        destination = _object(value["destination"])
        return OutputBinding(
            source=self.source(value["source"]),
            destination=WorkflowOutputRef(port=cast(str, destination["port"])),
        )

    def outcome_binding(self, value: Json) -> OutcomeBinding:
        if not isinstance(value, dict):
            return cast(OutcomeBinding, value)
        source = _object(value["source"])
        destination = _object(value["destination"])
        return OutcomeBinding(
            source=NodeOutcomeRef(node=self.node(source["node"]), outcome=cast(str, source["outcome"])),
            destination=WorkflowOutcomeRef(outcome=cast(str, destination["outcome"])),
        )

    def sequence_edge(self, value: Json) -> SequenceEdge:
        if not isinstance(value, dict):
            return cast(SequenceEdge, value)
        return SequenceEdge(before=self.node(value["before"]), after=self.node(value["after"]))

    def choice(self, value: Json) -> ChoiceDecl:
        if not isinstance(value, dict):
            return cast(ChoiceDecl, value)
        return ChoiceDecl(
            selector=self.node(value["selector"]),
            branches=_product_tuple(value["branches"], self.choice_branch),
        )

    def choice_branch(self, value: Json) -> ChoiceBranch:
        if not isinstance(value, dict):
            return cast(ChoiceBranch, value)
        return ChoiceBranch(
            outcomes=_product_frozenset(value["outcomes"], lambda member: cast(str, member)),
            members=_product_frozenset(value["members"], self.node),
        )

    def requirement(self, value: Json) -> ProtectionRequirement:
        if not isinstance(value, dict):
            return cast(ProtectionRequirement, value)
        item = _object(value)
        requirement = ProtectionRequirement(
            outcome=cast(str, item["outcome"]),
            meaning=cast(str, item["meaning"]),
            subject_port=cast(str, item["subject_port"]),
            consumed_ports=_product_frozenset(item["consumed_ports"], lambda member: cast(str, member)),
            coverage=_product_frozenset(item["coverage"], self.coverage),
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


def _case(*, family: str, mutation: str) -> Object:
    assert isinstance(CASES, list)
    return next(
        item
        for value in CASES
        if (item := _object(value))["family"] == family and cast(str, item["case_id"]).endswith(f"/{mutation}")
    )


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


@pytest.mark.parametrize(
    ("field", "value"),
    (("name", False), ("inputs", False)),
)
def test_adapter_forwards_malformed_product_fields(field: str, value: Json) -> None:
    declaration = copy.deepcopy(_object(_case(family="ports", mutation="base")["declaration"]))
    interface = _object(declaration["interface"])
    interface[field] = value
    assert _execute(declaration, None)[:2] == ("rejected", ValidationCode.INVALID_TYPE.value)


def test_global_validation_precedence_across_complete_boundaries() -> None:
    with pytest.raises(ContractViolation) as artifact_error:
        ArtifactType(name="", revision=True)
    assert artifact_error.value.code is ValidationCode.INVALID_TYPE
    with pytest.raises(ContractViolation) as limits_error:
        WorkflowLimits(
            max_nodes=0,
            max_bindings=True,
            max_sequence_edges=0,
            max_choices=0,
            max_branch_members=0,
            max_subgraph_depth=1,
            max_choice_states=1,
        )
    assert limits_error.value.code is ValidationCode.INVALID_TYPE

    workflow = _one_node_factory(_zero_operation)
    absent_local = NodeId.new(workflow=workflow.workflow)
    with pytest.raises(ContractViolation) as substitution_error:
        substitute(workflow=workflow, target=absent_local, replacement=workflow)
    assert substitution_error.value.code is ValidationCode.FOREIGN_OWNER

    declaration = copy.deepcopy(_object(_case(family="choice", mutation="CHOICE_Z")["declaration"]))
    sequence = _array(declaration["sequence"])
    sequence[:] = [edge for edge in sequence if _object(_object(edge)["after"])["label"] != "N1"]
    sequence.append(
        {
            "after": {"label": "N1", "owner": "W0"},
            "before": {"label": "N1", "owner": "W0"},
        }
    )
    assert _execute(declaration, None)[:2] == ("rejected", ValidationCode.CYCLE.value)


def test_choice_branches_have_order_independent_semantic_state() -> None:
    declaration = copy.deepcopy(_object(_case(family="choice", mutation="CHOICE_Z")["declaration"]))
    adapter = _Adapter()
    original = adapter.admit(declaration)
    choice = _object(_array(declaration["choices"])[0])
    choice["branches"] = list(reversed(_array(choice["branches"])))
    reversed_branches = adapter.admit(declaration)
    assert original.choices == reversed_branches.choices
    assert original == reversed_branches
    assert hash(original) == hash(reversed_branches)


def test_protection_requires_every_requirement_for_an_outcome() -> None:
    declaration = copy.deepcopy(_object(_case(family="protection", mutation="exact")["declaration"]))
    requirements = _array(declaration["protection"])
    unmet = copy.deepcopy(_object(requirements[0]))
    unmet["meaning"] = "unmatched-assessment"
    requirements.append(unmet)
    admitted = _Adapter().admit(declaration)
    assert admitted.protection_eligible_outcomes == frozenset()
    assert {requirement.meaning for requirement in admitted.unmet_protection} == {"unmatched-assessment"}


def test_deep_legal_subgraphs_are_equality_hash_and_copy_safe() -> None:
    depth = 500
    limits = WorkflowLimits(
        max_nodes=depth,
        max_bindings=1,
        max_sequence_edges=0,
        max_choices=0,
        max_branch_members=0,
        max_subgraph_depth=depth,
        max_choice_states=1,
    )

    def flat() -> tuple[AdmittedWorkflow, NodeId]:
        workflow = WorkflowId.new()
        node = NodeId.new(workflow=workflow)
        operation = _zero_operation()
        return (
            admit_static_workflow(
                workflow=workflow,
                interface=operation,
                nodes=(OperationNode(id=node, operation=operation),),
                input_bindings=(),
                output_bindings=(),
                outcome_bindings=(
                    OutcomeBinding(
                        source=NodeOutcomeRef(node=node, outcome="ok"),
                        destination=WorkflowOutcomeRef(outcome="ok"),
                    ),
                ),
                sequence=(),
                choices=(),
                protection=(),
                limits=limits,
            ),
            node,
        )

    chain, _ = flat()
    final_parent: AdmittedWorkflow | None = None
    final_target: NodeId | None = None
    final_body: AdmittedWorkflow | None = None
    for _ in range(depth - 1):
        parent, target = flat()
        final_parent, final_target, final_body = parent, target, chain
        chain = substitute(workflow=parent, target=target, replacement=chain)
    assert chain.expanded_node_count == depth
    assert final_parent is not None and final_target is not None and final_body is not None
    equivalent = substitute(workflow=final_parent, target=final_target, replacement=final_body)
    assert chain is copy.copy(chain) is copy.deepcopy(chain)
    assert chain == equivalent
    assert hash(chain) == hash(equivalent)


def test_frozen_reference_bytes_are_unchanged() -> None:
    assert hashlib.sha256(FROZEN_BYTES).hexdigest() == MANIFEST["corpus_sha256"]
