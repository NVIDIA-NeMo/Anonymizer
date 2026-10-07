# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure declarations and admission for static protection workflows."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Literal, Never, SupportsIndex, TypeAlias

from anonymizer.graph._values import ContractViolation, ValidationCode

OutcomeClass: TypeAlias = Literal["success", "failure", "cancelled", "lost", "blocked", "inconsistent"]
StateEffectKind: TypeAlias = Literal["read", "write"]
CoverageKind: TypeAlias = Literal["field", "source_view", "evaluation", "absence"]
CaptureMode: TypeAlias = Literal["whole_artifact"]

_OUTCOME_CLASSES = frozenset(("success", "failure", "cancelled", "lost", "blocked", "inconsistent"))
_STATE_EFFECT_KINDS = frozenset(("read", "write"))
_COVERAGE_KINDS = frozenset(("field", "source_view", "evaluation", "absence"))
_CAPTURE_MODES = frozenset(("whole_artifact",))
_FACTORY_KEY = object()
_ADMISSION_KEY = object()
_PICKLE_ERROR = "workflow value serialization is not supported"


def _reject(code: ValidationCode) -> Never:
    raise ContractViolation(code) from None


def _validate_scalars(
    *,
    strings: tuple[object, ...] = (),
    nonnegative_integers: tuple[object, ...] = (),
    positive_integers: tuple[object, ...] = (),
) -> None:
    integers = nonnegative_integers + positive_integers
    if any(not isinstance(value, str) for value in strings) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in integers
    ):
        _reject(ValidationCode.INVALID_TYPE)
    nonnegative_values = tuple(value for value in nonnegative_integers if isinstance(value, int))
    positive_values = tuple(value for value in positive_integers if isinstance(value, int))
    if (
        any(not value for value in strings)
        or any(value < 0 for value in nonnegative_values)
        or any(value < 1 for value in positive_values)
    ):
        _reject(ValidationCode.INVALID_VALUE)


def _tuple_of(value: object, member_type: type[Any] | tuple[type[Any], ...]) -> None:
    if not isinstance(value, tuple) or any(not isinstance(member, member_type) for member in value):
        _reject(ValidationCode.INVALID_TYPE)


def _frozenset_of(value: object, member_type: type[Any] | tuple[type[Any], ...]) -> None:
    if not isinstance(value, frozenset) or any(not isinstance(member, member_type) for member in value):
        _reject(ValidationCode.INVALID_TYPE)


class _PrivateValue:
    __slots__ = ()

    def __repr__(self) -> str:
        return f"<{type(self).__name__}>"

    def __copy__(self) -> _PrivateValue:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _PrivateValue:
        del memo
        return self

    def __reduce_ex__(self, protocol: SupportsIndex, /) -> Never:
        del protocol
        raise TypeError(_PICKLE_ERROR)


class _OpaqueIdentity(_PrivateValue):
    __slots__ = ("_token",)
    _token: object

    def __new__(cls, factory_key: object, **owner: object) -> _OpaqueIdentity:
        del owner
        if factory_key is not _FACTORY_KEY:
            raise TypeError("opaque identities must be created by their factory")
        return super().__new__(cls)

    def __init__(self, factory_key: object, **owner: object) -> None:
        del factory_key, owner
        object.__setattr__(self, "_token", object())

    def __setattr__(self, name: str, value: object) -> None:
        del name, value
        raise AttributeError("opaque identities are immutable")

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _OpaqueIdentity) and type(self) is type(other) and self._token is other._token

    def __hash__(self) -> int:
        return hash(self._token)


class WorkflowId(_OpaqueIdentity):
    """Opaque process-local workflow owner."""

    __slots__ = ()

    @classmethod
    def new(cls) -> WorkflowId:
        """Create a fresh workflow owner."""
        return cls(_FACTORY_KEY)


class NodeId(_OpaqueIdentity):
    """Opaque process-local node identity owned by a workflow."""

    __slots__ = ("workflow",)
    workflow: WorkflowId

    def __init__(self, factory_key: object, *, workflow: WorkflowId) -> None:
        if not isinstance(workflow, WorkflowId):
            _reject(ValidationCode.INVALID_TYPE)
        super().__init__(factory_key)
        object.__setattr__(self, "workflow", workflow)

    @classmethod
    def new(cls, *, workflow: WorkflowId) -> NodeId:
        """Create a fresh node identity owned by ``workflow``."""
        return cls(_FACTORY_KEY, workflow=workflow)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ArtifactType(_PrivateValue):
    name: str
    revision: int

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.name,), positive_integers=(self.revision,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class InputPort(_PrivateValue):
    name: str
    artifact_type: ArtifactType

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_type, ArtifactType):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(strings=(self.name,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OutputPort(_PrivateValue):
    name: str
    artifact_type: ArtifactType

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_type, ArtifactType):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(strings=(self.name,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OutputDependency(_PrivateValue):
    output: str
    inputs: frozenset[str]
    identity_input: str | None

    def __post_init__(self) -> None:
        _frozenset_of(self.inputs, str)
        if self.identity_input is not None and not isinstance(self.identity_input, str):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(
            strings=(self.output, *self.inputs, *((self.identity_input,) if self.identity_input is not None else ()))
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ContextUse(_PrivateValue):
    port: str
    meaning: str
    capture: CaptureMode

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.port, self.meaning, self.capture))
        if self.capture not in _CAPTURE_MODES:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CoverageAtom(_PrivateValue):
    kind: CoverageKind
    name: str

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.kind, self.name))
        if self.kind not in _COVERAGE_KINDS:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class EvidencePromise(_PrivateValue):
    name: str
    meaning: str
    subject_port: str
    consumed_ports: frozenset[str]
    coverage: frozenset[CoverageAtom]

    def __post_init__(self) -> None:
        _frozenset_of(self.consumed_ports, str)
        _frozenset_of(self.coverage, CoverageAtom)
        _validate_scalars(strings=(self.name, self.meaning, self.subject_port, *self.consumed_ports))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateEffect(_PrivateValue):
    kind: StateEffectKind
    name: str

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.kind, self.name))
        if self.kind not in _STATE_EFFECT_KINDS:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ModelRequirement(_PrivateValue):
    capability: str
    revision: int

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.capability,), positive_integers=(self.revision,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ResourceCeiling(_PrivateValue):
    max_activations: int
    max_model_requests: int
    max_input_bytes: int
    max_output_bytes: int

    def __post_init__(self) -> None:
        _validate_scalars(
            nonnegative_integers=(
                self.max_activations,
                self.max_model_requests,
                self.max_input_bytes,
                self.max_output_bytes,
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OutcomeSpec(_PrivateValue):
    name: str
    category: OutcomeClass
    produced_ports: frozenset[str]
    context: frozenset[ContextUse]
    evidence: frozenset[EvidencePromise]
    state_effects: frozenset[StateEffect]
    model_requirements: frozenset[ModelRequirement]
    ceiling: ResourceCeiling

    def __post_init__(self) -> None:
        _frozenset_of(self.produced_ports, str)
        _frozenset_of(self.context, ContextUse)
        _frozenset_of(self.evidence, EvidencePromise)
        _frozenset_of(self.state_effects, StateEffect)
        _frozenset_of(self.model_requirements, ModelRequirement)
        if not isinstance(self.ceiling, ResourceCeiling):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(strings=(self.name, self.category, *self.produced_ports))
        if self.category not in _OUTCOME_CLASSES:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OperationSpec(_PrivateValue):
    name: str
    inputs: tuple[InputPort, ...]
    outputs: tuple[OutputPort, ...]
    output_dependencies: tuple[OutputDependency, ...]
    outcomes: tuple[OutcomeSpec, ...]

    def __post_init__(self) -> None:
        _tuple_of(self.inputs, InputPort)
        _tuple_of(self.outputs, OutputPort)
        _tuple_of(self.output_dependencies, OutputDependency)
        _tuple_of(self.outcomes, OutcomeSpec)
        _validate_scalars(strings=(self.name,))
        input_names = [port.name for port in self.inputs]
        output_names = [port.name for port in self.outputs]
        outcome_names = [outcome.name for outcome in self.outcomes]
        dependency_outputs = [dependency.output for dependency in self.output_dependencies]
        if any(
            len(values) != len(set(values))
            for values in (input_names + output_names, outcome_names, dependency_outputs)
        ):
            _reject(ValidationCode.DUPLICATE)
        if any(
            len(outcome.evidence) != len({promise.name for promise in outcome.evidence}) for outcome in self.outcomes
        ):
            _reject(ValidationCode.DUPLICATE)
        inputs = set(input_names)
        outputs = set(output_names)
        if set(dependency_outputs) != outputs:
            _reject(ValidationCode.MISSING)
        input_types = {port.name: port.artifact_type for port in self.inputs}
        output_types = {port.name: port.artifact_type for port in self.outputs}
        for dependency in self.output_dependencies:
            if dependency.output not in outputs or not dependency.inputs <= inputs:
                _reject(ValidationCode.MISSING)
            if dependency.identity_input is not None:
                if dependency.identity_input not in dependency.inputs:
                    _reject(ValidationCode.CONTRADICTORY)
                if input_types[dependency.identity_input] != output_types[dependency.output]:
                    _reject(ValidationCode.CONTRADICTORY)
        all_ports = inputs | outputs
        produced_any: set[str] = set()
        for outcome in self.outcomes:
            if not outcome.produced_ports <= outputs:
                _reject(ValidationCode.MISSING)
            produced_any.update(outcome.produced_ports)
            if any(use.port not in all_ports for use in outcome.context):
                _reject(ValidationCode.MISSING)
            for promise in outcome.evidence:
                if promise.subject_port not in all_ports or not promise.consumed_ports <= inputs:
                    _reject(ValidationCode.MISSING)
        if produced_any != outputs:
            _reject(ValidationCode.MISSING)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OperationNode(_PrivateValue):
    id: NodeId
    operation: OperationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.id, NodeId) or not isinstance(self.operation, OperationSpec):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SubgraphNode(_PrivateValue):
    id: NodeId
    operation: OperationSpec
    body: AdmittedWorkflow

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, NodeId)
            or not isinstance(self.operation, OperationSpec)
            or not isinstance(self.body, AdmittedWorkflow)
        ):
            _reject(ValidationCode.INVALID_TYPE)


Node: TypeAlias = OperationNode | SubgraphNode


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class WorkflowInputRef(_PrivateValue):
    port: str

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.port,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class NodeOutputRef(_PrivateValue):
    node: NodeId
    port: str

    def __post_init__(self) -> None:
        if not isinstance(self.node, NodeId):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(strings=(self.port,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class NodeInputRef(_PrivateValue):
    node: NodeId
    port: str

    def __post_init__(self) -> None:
        if not isinstance(self.node, NodeId):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(strings=(self.port,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class WorkflowOutputRef(_PrivateValue):
    port: str

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.port,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class NodeOutcomeRef(_PrivateValue):
    node: NodeId
    outcome: str

    def __post_init__(self) -> None:
        if not isinstance(self.node, NodeId):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(strings=(self.outcome,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class WorkflowOutcomeRef(_PrivateValue):
    outcome: str

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.outcome,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class InputBinding(_PrivateValue):
    source: WorkflowInputRef | NodeOutputRef
    destination: NodeInputRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, (WorkflowInputRef, NodeOutputRef)) or not isinstance(
            self.destination, NodeInputRef
        ):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OutputBinding(_PrivateValue):
    source: WorkflowInputRef | NodeOutputRef
    destination: WorkflowOutputRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, (WorkflowInputRef, NodeOutputRef)) or not isinstance(
            self.destination, WorkflowOutputRef
        ):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OutcomeBinding(_PrivateValue):
    source: NodeOutcomeRef
    destination: WorkflowOutcomeRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, NodeOutcomeRef) or not isinstance(self.destination, WorkflowOutcomeRef):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SequenceEdge(_PrivateValue):
    before: NodeId
    after: NodeId

    def __post_init__(self) -> None:
        if not isinstance(self.before, NodeId) or not isinstance(self.after, NodeId):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ChoiceBranch(_PrivateValue):
    outcomes: frozenset[str]
    members: frozenset[NodeId]

    def __post_init__(self) -> None:
        _frozenset_of(self.outcomes, str)
        _frozenset_of(self.members, NodeId)
        _validate_scalars(strings=tuple(self.outcomes))
        if not self.outcomes or not self.members:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, eq=False)
class ChoiceDecl(_PrivateValue):
    selector: NodeId
    branches: tuple[ChoiceBranch, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.selector, NodeId):
            _reject(ValidationCode.INVALID_TYPE)
        _tuple_of(self.branches, ChoiceBranch)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ChoiceDecl)
            and self.selector == other.selector
            and frozenset(self.branches) == frozenset(other.branches)
        )

    def __hash__(self) -> int:
        return hash((self.selector, frozenset(self.branches)))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProtectionRequirement(_PrivateValue):
    outcome: str
    meaning: str
    subject_port: str
    consumed_ports: frozenset[str]
    coverage: frozenset[CoverageAtom]

    def __post_init__(self) -> None:
        _frozenset_of(self.consumed_ports, str)
        _frozenset_of(self.coverage, CoverageAtom)
        _validate_scalars(strings=(self.outcome, self.meaning, self.subject_port, *self.consumed_ports))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class WorkflowLimits(_PrivateValue):
    max_nodes: int
    max_bindings: int
    max_sequence_edges: int
    max_choices: int
    max_branch_members: int
    max_subgraph_depth: int
    max_choice_states: int

    def __post_init__(self) -> None:
        _validate_scalars(
            nonnegative_integers=(
                self.max_bindings,
                self.max_sequence_edges,
                self.max_choices,
                self.max_branch_members,
            ),
            positive_integers=(self.max_nodes, self.max_subgraph_depth, self.max_choice_states),
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False, eq=False)
class AdmittedWorkflow(_PrivateValue):
    """Immutable static workflow produced only by successful admission."""

    workflow: WorkflowId
    interface: OperationSpec
    nodes: frozenset[Node]
    input_bindings: frozenset[InputBinding]
    output_bindings: frozenset[OutputBinding]
    outcome_bindings: frozenset[OutcomeBinding]
    sequence: frozenset[SequenceEdge]
    choices: frozenset[ChoiceDecl]
    protection_requirements: frozenset[ProtectionRequirement]
    protection_eligible_outcomes: frozenset[str]
    unmet_protection: frozenset[ProtectionRequirement]
    limits: WorkflowLimits
    expanded_node_count: int

    def __init__(
        self,
        *,
        _key: object,
        workflow: WorkflowId,
        interface: OperationSpec,
        nodes: frozenset[Node],
        input_bindings: frozenset[InputBinding],
        output_bindings: frozenset[OutputBinding],
        outcome_bindings: frozenset[OutcomeBinding],
        sequence: frozenset[SequenceEdge],
        choices: frozenset[ChoiceDecl],
        protection_requirements: frozenset[ProtectionRequirement],
        protection_eligible_outcomes: frozenset[str],
        unmet_protection: frozenset[ProtectionRequirement],
        limits: WorkflowLimits,
        expanded_node_count: int,
    ) -> None:
        if _key is not _ADMISSION_KEY:
            raise TypeError("admitted workflows must be created by admission")
        for name, value in locals().copy().items():
            if name not in {"self", "_key"}:
                object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, AdmittedWorkflow):
            return False
        pending: list[tuple[AdmittedWorkflow, AdmittedWorkflow]] = [(self, other)]
        seen: set[tuple[int, int]] = set()
        while pending:
            left, right = pending.pop()
            pair = (id(left), id(right))
            if pair in seen:
                continue
            seen.add(pair)
            if _workflow_local_state(left) != _workflow_local_state(right):
                return False
            left_nodes = {node.id: node for node in left.nodes}
            right_nodes = {node.id: node for node in right.nodes}
            if left_nodes.keys() != right_nodes.keys():
                return False
            for node_id, left_node in left_nodes.items():
                right_node = right_nodes[node_id]
                if type(left_node) is not type(right_node) or left_node.operation != right_node.operation:
                    return False
                if isinstance(left_node, SubgraphNode):
                    assert isinstance(right_node, SubgraphNode)
                    pending.append((left_node.body, right_node.body))
        return True

    def __hash__(self) -> int:
        pending: list[tuple[AdmittedWorkflow, bool]] = [(self, False)]
        hashes: dict[int, int] = {}
        while pending:
            workflow, children_visited = pending.pop()
            key = id(workflow)
            if key in hashes:
                continue
            children = [node.body for node in workflow.nodes if isinstance(node, SubgraphNode)]
            if not children_visited:
                pending.append((workflow, True))
                pending.extend((child, False) for child in children if id(child) not in hashes)
                continue
            node_hashes = frozenset(
                hash((type(node), node.id, node.operation, hashes[id(node.body)]))
                if isinstance(node, SubgraphNode)
                else hash((type(node), node.id, node.operation))
                for node in workflow.nodes
            )
            hashes[key] = hash((_workflow_local_state(workflow), node_hashes))
        return hashes[id(self)]


def _workflow_local_state(workflow: AdmittedWorkflow) -> tuple[object, ...]:
    return (
        workflow.workflow,
        workflow.interface,
        workflow.input_bindings,
        workflow.output_bindings,
        workflow.outcome_bindings,
        workflow.sequence,
        workflow.choices,
        workflow.protection_requirements,
        workflow.protection_eligible_outcomes,
        workflow.unmet_protection,
        workflow.limits,
        workflow.expanded_node_count,
    )


def admit_static_workflow(
    *,
    workflow: WorkflowId,
    interface: OperationSpec,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    protection: tuple[ProtectionRequirement, ...],
    limits: WorkflowLimits,
) -> AdmittedWorkflow:
    """Validate and normalize a pure static workflow declaration."""
    _validate_admission_types(
        workflow,
        interface,
        nodes,
        input_bindings,
        output_bindings,
        outcome_bindings,
        sequence,
        choices,
        protection,
        limits,
    )
    if any(choice.selector in branch.members for choice in choices for branch in choice.branches):
        _reject(ValidationCode.INVALID_VALUE)
    expanded_count = sum(1 + (node.body.expanded_node_count if isinstance(node, SubgraphNode) else 0) for node in nodes)
    depth = max((1 + _workflow_depth(node.body) if isinstance(node, SubgraphNode) else 1 for node in nodes), default=1)
    choice_states = 1
    node_operations = {node.id: node.operation for node in nodes}
    for choice in choices:
        operation = node_operations.get(choice.selector)
        choice_states *= len(operation.outcomes) if operation is not None else 1
    raw_bindings = len(input_bindings) + len(output_bindings) + len(outcome_bindings)
    branch_members = sum(len(branch.members) for choice in choices for branch in choice.branches)
    if (
        len(nodes) > limits.max_nodes
        or raw_bindings > limits.max_bindings
        or len(sequence) > limits.max_sequence_edges
        or len(choices) > limits.max_choices
        or branch_members > limits.max_branch_members
        or expanded_count > limits.max_nodes
        or depth > limits.max_subgraph_depth
        or choice_states > limits.max_choice_states
    ):
        _reject(ValidationCode.LIMIT_EXCEEDED)
    _validate_owners(workflow, nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices)
    _validate_duplicates(nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices, protection)
    incompatible_binding = _validate_references(
        interface, nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices, protection
    )
    _validate_choice_overlaps(choices)
    _validate_cycles(nodes, sequence)
    _validate_choice_reachability(sequence, choices)
    for node in nodes:
        if isinstance(node, SubgraphNode):
            if not _compatible(node.operation, node.body.interface, allow_narrower=True):
                _reject(ValidationCode.CONTRADICTORY)
    if incompatible_binding:
        _reject(ValidationCode.CONTRADICTORY)
    _validate_paths(
        interface, nodes, input_bindings, output_bindings, outcome_bindings, sequence, choices, check_cycles=False
    )
    eligible, unmet = _protection(interface, protection)
    return AdmittedWorkflow(
        _key=_ADMISSION_KEY,
        workflow=workflow,
        interface=interface,
        nodes=frozenset(nodes),
        input_bindings=frozenset(input_bindings),
        output_bindings=frozenset(output_bindings),
        outcome_bindings=frozenset(outcome_bindings),
        sequence=frozenset(sequence),
        choices=frozenset(choices),
        protection_requirements=frozenset(protection),
        protection_eligible_outcomes=eligible,
        unmet_protection=unmet,
        limits=limits,
        expanded_node_count=expanded_count,
    )


def substitute(*, workflow: AdmittedWorkflow, target: NodeId, replacement: AdmittedWorkflow) -> AdmittedWorkflow:
    """Replace one operation with a compatible, separately owned workflow."""
    if (
        not isinstance(workflow, AdmittedWorkflow)
        or not isinstance(target, NodeId)
        or not isinstance(replacement, AdmittedWorkflow)
    ):
        _reject(ValidationCode.INVALID_TYPE)
    if target.workflow != workflow.workflow:
        _reject(ValidationCode.FOREIGN_OWNER)
    if replacement.workflow == workflow.workflow:
        _reject(ValidationCode.FOREIGN_OWNER)
    target_node = next((node for node in workflow.nodes if node.id == target), None)
    if target_node is None:
        _reject(ValidationCode.MISSING)
    if not _compatible(target_node.operation, replacement.interface, allow_narrower=True):
        _reject(ValidationCode.CONTRADICTORY)
    nodes = tuple(
        SubgraphNode(id=target, operation=target_node.operation, body=replacement) if node.id == target else node
        for node in workflow.nodes
    )
    return admit_static_workflow(
        workflow=workflow.workflow,
        interface=workflow.interface,
        nodes=nodes,
        input_bindings=tuple(workflow.input_bindings),
        output_bindings=tuple(workflow.output_bindings),
        outcome_bindings=tuple(workflow.outcome_bindings),
        sequence=tuple(workflow.sequence),
        choices=tuple(workflow.choices),
        protection=tuple(workflow.protection_requirements),
        limits=workflow.limits,
    )


def _validate_admission_types(
    workflow: object,
    interface: object,
    nodes: object,
    input_bindings: object,
    output_bindings: object,
    outcome_bindings: object,
    sequence: object,
    choices: object,
    protection: object,
    limits: object,
) -> None:
    if (
        not isinstance(workflow, WorkflowId)
        or not isinstance(interface, OperationSpec)
        or not isinstance(limits, WorkflowLimits)
    ):
        _reject(ValidationCode.INVALID_TYPE)
    _tuple_of(nodes, (OperationNode, SubgraphNode))
    _tuple_of(input_bindings, InputBinding)
    _tuple_of(output_bindings, OutputBinding)
    _tuple_of(outcome_bindings, OutcomeBinding)
    _tuple_of(sequence, SequenceEdge)
    _tuple_of(choices, ChoiceDecl)
    _tuple_of(protection, ProtectionRequirement)


def _workflow_depth(workflow: AdmittedWorkflow) -> int:
    maximum = 1
    pending = [(workflow, 1)]
    while pending:
        current, depth = pending.pop()
        maximum = max(maximum, depth)
        pending.extend((node.body, depth + 1) for node in current.nodes if isinstance(node, SubgraphNode))
    return maximum


def _node_references(
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
) -> list[NodeId]:
    references: list[NodeId] = []
    for binding in (*input_bindings, *output_bindings):
        if isinstance(binding.source, NodeOutputRef):
            references.append(binding.source.node)
        if isinstance(binding, InputBinding):
            references.append(binding.destination.node)
    references.extend(binding.source.node for binding in outcome_bindings)
    references.extend(node for edge in sequence for node in (edge.before, edge.after))
    for choice in choices:
        references.append(choice.selector)
        references.extend(member for branch in choice.branches for member in branch.members)
    return references


def _validate_owners(
    workflow: WorkflowId,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
) -> None:
    if any(node.id.workflow != workflow for node in nodes) or any(
        reference.workflow != workflow
        for reference in _node_references(input_bindings, output_bindings, outcome_bindings, sequence, choices)
    ):
        _reject(ValidationCode.FOREIGN_OWNER)
    if any(isinstance(node, SubgraphNode) and node.body.workflow == workflow for node in nodes):
        _reject(ValidationCode.FOREIGN_OWNER)


def _duplicates(values: tuple[Any, ...] | list[Any]) -> bool:
    return len(values) != len(set(values))


def _validate_duplicates(
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    protection: tuple[ProtectionRequirement, ...],
) -> None:
    if (
        _duplicates([node.id for node in nodes])
        or _duplicates([binding.destination for binding in input_bindings])
        or _duplicates([binding.destination for binding in output_bindings])
        or _duplicates([binding.source for binding in outcome_bindings])
        or _duplicates(list(sequence))
        or _duplicates([choice.selector for choice in choices])
        or _duplicates(list(protection))
    ):
        _reject(ValidationCode.DUPLICATE)
    if any(_duplicates(list(choice.branches)) for choice in choices):
        _reject(ValidationCode.DUPLICATE)


def _port_type(operation: OperationSpec, port: str, *, output: bool) -> ArtifactType | None:
    ports = operation.outputs if output else operation.inputs
    return next((item.artifact_type for item in ports if item.name == port), None)


def _source_type(
    source: WorkflowInputRef | NodeOutputRef, interface: OperationSpec, nodes: dict[NodeId, OperationSpec]
) -> ArtifactType | None:
    if isinstance(source, WorkflowInputRef):
        return _port_type(interface, source.port, output=False)
    operation = nodes.get(source.node)
    return None if operation is None else _port_type(operation, source.port, output=True)


def _validate_references(
    interface: OperationSpec,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    protection: tuple[ProtectionRequirement, ...],
) -> bool:
    operations = {node.id: node.operation for node in nodes}
    interface_outputs = {port.name for port in interface.outputs}
    interface_outcomes = {outcome.name for outcome in interface.outcomes}
    incompatible_binding = False
    for binding in input_bindings:
        destination = operations.get(binding.destination.node)
        source_type = _source_type(binding.source, interface, operations)
        destination_type = (
            None if destination is None else _port_type(destination, binding.destination.port, output=False)
        )
        if source_type is None or destination_type is None:
            _reject(ValidationCode.MISSING)
        if source_type != destination_type:
            incompatible_binding = True
    for binding in output_bindings:
        source_type = _source_type(binding.source, interface, operations)
        destination_type = _port_type(interface, binding.destination.port, output=True)
        if source_type is None or destination_type is None:
            _reject(ValidationCode.MISSING)
        if source_type != destination_type:
            incompatible_binding = True
    for binding in outcome_bindings:
        operation = operations.get(binding.source.node)
        if operation is None or binding.source.outcome not in {outcome.name for outcome in operation.outcomes}:
            _reject(ValidationCode.MISSING)
        if binding.destination.outcome not in interface_outcomes:
            _reject(ValidationCode.MISSING)
    if any(edge.before not in operations or edge.after not in operations for edge in sequence):
        _reject(ValidationCode.MISSING)
    for choice in choices:
        selector = operations.get(choice.selector)
        if selector is None:
            _reject(ValidationCode.MISSING)
        selector_outcomes = {outcome.name for outcome in selector.outcomes}
        for branch in choice.branches:
            if not branch.outcomes <= selector_outcomes or not branch.members <= operations.keys():
                _reject(ValidationCode.MISSING)
    for requirement in protection:
        if (
            requirement.outcome not in interface_outcomes
            or requirement.subject_port not in ({port.name for port in interface.inputs} | interface_outputs)
            or not requirement.consumed_ports <= {port.name for port in interface.inputs}
        ):
            _reject(ValidationCode.MISSING)
    required_inputs = {(node.id, port.name) for node in nodes for port in node.operation.inputs}
    if {binding.destination for binding in input_bindings} != {
        NodeInputRef(node=node, port=port) for node, port in required_inputs
    }:
        _reject(ValidationCode.MISSING)
    if {binding.destination.port for binding in output_bindings} != interface_outputs:
        _reject(ValidationCode.MISSING)
    return incompatible_binding


def _closure(start: NodeId, edges: frozenset[SequenceEdge]) -> frozenset[NodeId]:
    reached: set[NodeId] = {start}
    changed = True
    while changed:
        changed = False
        for edge in edges:
            if edge.before in reached and edge.after not in reached:
                reached.add(edge.after)
                changed = True
    return frozenset(reached)


def _validate_choice_overlaps(choices: tuple[ChoiceDecl, ...]) -> None:
    all_members: set[NodeId] = set()
    for choice in choices:
        branch_members: set[NodeId] = set()
        branch_outcomes: set[str] = set()
        for branch in choice.branches:
            if branch_members & branch.members or branch_outcomes & branch.outcomes:
                _reject(ValidationCode.OVERLAP)
            branch_members.update(branch.members)
            branch_outcomes.update(branch.outcomes)
        if all_members & branch_members:
            _reject(ValidationCode.OVERLAP)
        all_members.update(branch_members)


def _validate_choice_reachability(sequence: tuple[SequenceEdge, ...], choices: tuple[ChoiceDecl, ...]) -> None:
    edges = frozenset(sequence)
    for choice in choices:
        reached = _closure(choice.selector, edges)
        if any(member not in reached for branch in choice.branches for member in branch.members):
            _reject(ValidationCode.CONTRADICTORY)


Vertex: TypeAlias = tuple[str, NodeId | None, str]


def _walk_endpoints(
    start: Vertex, edges: set[tuple[Vertex, Vertex]], *, backward: bool, endpoint_kind: str
) -> frozenset[str]:
    pending = [start]
    seen: set[Vertex] = set()
    endpoints: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if current[0] == endpoint_kind:
            endpoints.add(current[2])
        for source, destination in edges:
            if backward and destination == current:
                pending.append(source)
            elif not backward and source == current:
                pending.append(destination)
    return frozenset(endpoints)


def _project_one(endpoints: frozenset[str]) -> str:
    if not endpoints:
        _reject(ValidationCode.MISSING)
    if len(endpoints) > 1:
        _reject(ValidationCode.CONTRADICTORY)
    return next(iter(endpoints))


def _has_cycle(selected: frozenset[NodeId], edges: frozenset[SequenceEdge]) -> bool:
    remaining = set(selected)
    while remaining:
        roots = {
            node for node in remaining if not any(edge.after == node and edge.before in remaining for edge in edges)
        }
        if not roots:
            return True
        remaining -= roots
    return False


def _validate_cycles(nodes: tuple[Node, ...], sequence: tuple[SequenceEdge, ...]) -> None:
    if _has_cycle(frozenset(node.id for node in nodes), frozenset(sequence)):
        _reject(ValidationCode.CYCLE)


def _selected_nodes(
    node_ids: frozenset[NodeId], choices: tuple[ChoiceDecl, ...], assignment: dict[NodeId, OutcomeSpec]
) -> frozenset[NodeId]:
    branch_members = {member for choice in choices for branch in choice.branches for member in branch.members}
    selected = set(node_ids - branch_members)
    changed = True
    while changed:
        changed = False
        for choice in choices:
            if choice.selector not in selected:
                continue
            outcome = assignment[choice.selector].name
            for branch in choice.branches:
                if outcome in branch.outcomes:
                    before = len(selected)
                    selected.update(branch.members)
                    changed |= len(selected) != before
    return frozenset(selected)


def _validate_paths(
    interface: OperationSpec,
    nodes: tuple[Node, ...],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
    outcome_bindings: tuple[OutcomeBinding, ...],
    sequence: tuple[SequenceEdge, ...],
    choices: tuple[ChoiceDecl, ...],
    *,
    check_cycles: bool = True,
) -> None:
    operations = {node.id: node.operation for node in nodes}
    node_ids = frozenset(operations)
    edges = frozenset(sequence)
    if check_cycles and _has_cycle(node_ids, edges):
        _reject(ValidationCode.CYCLE)
    outcome_options = [operation.outcomes for operation in operations.values()]
    ids = tuple(operations)
    reached_interface_outcomes: set[str] = set()
    for outcomes in itertools.product(*outcome_options):
        assignment = dict(zip(ids, outcomes, strict=True))
        selected = _selected_nodes(node_ids, choices, assignment)
        selected_edges = frozenset(edge for edge in edges if edge.before in selected and edge.after in selected)
        if check_cycles and _has_cycle(selected, selected_edges):
            _reject(ValidationCode.CYCLE)
        reachable: set[NodeId] = set()
        changed = True
        while changed:
            changed = False
            for node in selected - reachable:
                predecessors = {edge.before for edge in selected_edges if edge.after == node}
                if not predecessors <= reachable:
                    continue
                bindings = [binding for binding in input_bindings if binding.destination.node == node]
                if all(
                    isinstance(binding.source, WorkflowInputRef)
                    or (
                        binding.source.node in reachable
                        and binding.source.port in assignment[binding.source.node].produced_ports
                    )
                    for binding in bindings
                ):
                    reachable.add(node)
                    changed = True
        if reachable != set(selected):
            _reject(ValidationCode.MISSING)
        sinks = {
            node
            for node in reachable
            if not any(edge.before == node and edge.after in reachable for edge in selected_edges)
        }
        if len(sinks) != 1:
            _reject(ValidationCode.MISSING)
        sink = next(iter(sinks))
        sink_outcome = assignment[sink]
        mappings = [
            binding
            for binding in outcome_bindings
            if binding.source.node == sink and binding.source.outcome == sink_outcome.name
        ]
        if len(mappings) != 1:
            _reject(ValidationCode.MISSING)
        external_outcome_name = mappings[0].destination.outcome
        reached_interface_outcomes.add(external_outcome_name)
        external_outcome = next(outcome for outcome in interface.outcomes if outcome.name == external_outcome_name)
        _validate_composition_path(
            interface,
            external_outcome,
            reachable,
            assignment,
            operations,
            input_bindings,
            output_bindings,
        )
    if reached_interface_outcomes != {outcome.name for outcome in interface.outcomes}:
        _reject(ValidationCode.MISSING)


def _validate_composition_path(
    interface: OperationSpec,
    external_outcome: OutcomeSpec,
    reachable: set[NodeId],
    assignment: dict[NodeId, OutcomeSpec],
    operations: dict[NodeId, OperationSpec],
    input_bindings: tuple[InputBinding, ...],
    output_bindings: tuple[OutputBinding, ...],
) -> None:
    identity_edges: set[tuple[Vertex, Vertex]] = set()
    dependency_edges: set[tuple[Vertex, Vertex]] = set()
    for binding in input_bindings:
        if binding.destination.node not in reachable:
            continue
        source = (
            ("wi", None, binding.source.port)
            if isinstance(binding.source, WorkflowInputRef)
            else ("out", binding.source.node, binding.source.port)
        )
        destination = ("in", binding.destination.node, binding.destination.port)
        identity_edges.add((source, destination))
        dependency_edges.add((source, destination))
    produced_external: set[str] = set()
    for binding in output_bindings:
        if isinstance(binding.source, WorkflowInputRef):
            source = ("wi", None, binding.source.port)
            exists = True
        else:
            source = ("out", binding.source.node, binding.source.port)
            exists = (
                binding.source.node in reachable
                and binding.source.port in assignment[binding.source.node].produced_ports
            )
        if exists:
            destination = ("wo", None, binding.destination.port)
            identity_edges.add((source, destination))
            dependency_edges.add((source, destination))
            produced_external.add(binding.destination.port)
    for node in reachable:
        operation = operations[node]
        produced = assignment[node].produced_ports
        for dependency in operation.output_dependencies:
            if dependency.output not in produced:
                continue
            output_vertex = ("out", node, dependency.output)
            for input_port in dependency.inputs:
                dependency_edges.add((("in", node, input_port), output_vertex))
            if dependency.identity_input is not None:
                identity_edges.add((("in", node, dependency.identity_input), output_vertex))
    contexts: set[ContextUse] = set()
    evidence: set[EvidencePromise] = set()
    state_effects: set[StateEffect] = set()
    models: set[ModelRequirement] = set()
    ceiling = [0, 0, 0, 0]
    for node in reachable:
        operation = operations[node]
        outcome = assignment[node]
        input_names = {port.name for port in operation.inputs}
        for use in outcome.context:
            endpoint = _project_one(
                _walk_endpoints(
                    ("in" if use.port in input_names else "out", node, use.port),
                    identity_edges,
                    backward=True,
                    endpoint_kind="wi",
                )
            )
            contexts.add(ContextUse(port=endpoint, meaning=use.meaning, capture=use.capture))
        for promise in outcome.evidence:
            consumed = frozenset(
                _project_one(_walk_endpoints(("in", node, port), identity_edges, backward=True, endpoint_kind="wi"))
                for port in promise.consumed_ports
            )
            if promise.subject_port in input_names:
                subject = _project_one(
                    _walk_endpoints(
                        ("in", node, promise.subject_port), identity_edges, backward=True, endpoint_kind="wi"
                    )
                )
            else:
                subject = _project_one(
                    _walk_endpoints(
                        ("out", node, promise.subject_port), identity_edges, backward=False, endpoint_kind="wo"
                    )
                )
            evidence.add(
                EvidencePromise(
                    name=promise.name,
                    meaning=promise.meaning,
                    subject_port=subject,
                    consumed_ports=consumed,
                    coverage=promise.coverage,
                )
            )
        state_effects.update(outcome.state_effects)
        models.update(outcome.model_requirements)
        values = outcome.ceiling
        ceiling[0] += values.max_activations
        ceiling[1] += values.max_model_requests
        ceiling[2] += values.max_input_bytes
        ceiling[3] += values.max_output_bytes
    dependencies = {item.output: item for item in interface.output_dependencies}
    for output in produced_external:
        vertex = ("wo", None, output)
        influence = _walk_endpoints(vertex, dependency_edges, backward=True, endpoint_kind="wi")
        identity = _walk_endpoints(vertex, identity_edges, backward=True, endpoint_kind="wi")
        if len(identity) > 1:
            _reject(ValidationCode.CONTRADICTORY)
        derived_identity = next(iter(identity)) if identity else None
        declared = dependencies[output]
        if declared.inputs != influence or declared.identity_input != derived_identity:
            _reject(ValidationCode.CONTRADICTORY)
    if (
        external_outcome.produced_ports != produced_external
        or external_outcome.context != contexts
        or external_outcome.evidence != evidence
        or external_outcome.state_effects != state_effects
        or external_outcome.model_requirements != models
    ):
        _reject(ValidationCode.CONTRADICTORY)
    actual = external_outcome.ceiling
    if any(
        supplied < derived
        for supplied, derived in zip(
            (actual.max_activations, actual.max_model_requests, actual.max_input_bytes, actual.max_output_bytes),
            ceiling,
            strict=True,
        )
    ):
        _reject(ValidationCode.CONTRADICTORY)


def _compatible(target: OperationSpec, replacement: OperationSpec, *, allow_narrower: bool) -> bool:
    if (
        target.inputs != replacement.inputs
        or target.outputs != replacement.outputs
        or frozenset(target.output_dependencies) != frozenset(replacement.output_dependencies)
    ):
        return False
    target_outcomes = {outcome.name: outcome for outcome in target.outcomes}
    replacement_outcomes = {outcome.name: outcome for outcome in replacement.outcomes}
    if target_outcomes.keys() != replacement_outcomes.keys():
        return False
    for name, expected in target_outcomes.items():
        actual = replacement_outcomes[name]
        if (
            expected.category != actual.category
            or expected.produced_ports != actual.produced_ports
            or expected.context != actual.context
            or expected.evidence != actual.evidence
            or expected.state_effects != actual.state_effects
            or expected.model_requirements != actual.model_requirements
        ):
            return False
        expected_ceiling = expected.ceiling
        actual_ceiling = actual.ceiling
        if allow_narrower:
            if any(
                replacement_value > target_value
                for replacement_value, target_value in zip(
                    (
                        actual_ceiling.max_activations,
                        actual_ceiling.max_model_requests,
                        actual_ceiling.max_input_bytes,
                        actual_ceiling.max_output_bytes,
                    ),
                    (
                        expected_ceiling.max_activations,
                        expected_ceiling.max_model_requests,
                        expected_ceiling.max_input_bytes,
                        expected_ceiling.max_output_bytes,
                    ),
                    strict=True,
                )
            ):
                return False
        elif expected_ceiling != actual_ceiling:
            return False
    return True


def _protection(
    interface: OperationSpec, protection: tuple[ProtectionRequirement, ...]
) -> tuple[frozenset[str], frozenset[ProtectionRequirement]]:
    outcomes = {outcome.name: outcome for outcome in interface.outcomes}
    affected = {requirement.outcome for requirement in protection}
    unmet: set[ProtectionRequirement] = set()
    for requirement in protection:
        matched = any(
            promise.meaning == requirement.meaning
            and promise.subject_port == requirement.subject_port
            and requirement.consumed_ports <= promise.consumed_ports
            and requirement.coverage <= promise.coverage
            for promise in outcomes[requirement.outcome].evidence
        )
        if not matched:
            unmet.add(requirement)
    ineligible = {requirement.outcome for requirement in unmet}
    eligible = affected - ineligible
    return frozenset(eligible), frozenset(unmet)
