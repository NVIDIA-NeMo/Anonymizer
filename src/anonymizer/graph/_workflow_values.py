# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow declarations and sealed admitted values."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Never, SupportsIndex, TypeAlias

from anonymizer.graph._values import ContractViolation, ValidationCode

OutcomeClass: TypeAlias = Literal["success", "failure", "cancelled", "lost", "blocked", "inconsistent"]


StateEffectKind: TypeAlias = Literal["read", "write"]


CoverageKind: TypeAlias = Literal["field", "source_view", "evaluation", "absence"]


CaptureMode: TypeAlias = Literal["whole_artifact"]


DynamicReduction: TypeAlias = Literal["all_by_key"]


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
class MapItemPort(_PrivateValue):
    """One outcome's extracted item, scoped to its map and containing path."""

    path: tuple[NodeId, ...]
    expander: NodeId
    member: NodeId
    item_input: str
    membership_port: str
    expansion_outcome: str

    def __post_init__(self) -> None:
        _tuple_of(self.path, NodeId)
        if not isinstance(self.expander, NodeId) or not isinstance(self.member, NodeId):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(strings=(self.item_input, self.membership_port, self.expansion_outcome))
        if self.expander.workflow != self.member.workflow:
            _reject(ValidationCode.FOREIGN_OWNER)
        if self.expander == self.member:
            _reject(ValidationCode.CONTRADICTORY)

    def lifted(self, container: NodeId) -> MapItemPort:
        """Preserve the map identity when exposing it through one containing node."""
        if not isinstance(container, NodeId):
            _reject(ValidationCode.INVALID_TYPE)
        return replace(self, path=(container, *self.path))

    def validate_in(self, *, workflow: WorkflowId, nodes: tuple[Node, ...]) -> None:
        """Validate the static route; dynamic admission binds its map declaration."""
        owner = workflow
        current = {node.id: node for node in nodes}
        for container in self.path:
            if container.workflow != owner:
                _reject(ValidationCode.FOREIGN_OWNER)
            node = current.get(container)
            if node is None:
                _reject(ValidationCode.MISSING)
            if not isinstance(node, SubgraphNode):
                _reject(ValidationCode.UNSUPPORTED)
            owner = node.body.workflow
            current = {child.id: child for child in node.body.nodes}
        if self.expander.workflow != owner:
            _reject(ValidationCode.FOREIGN_OWNER)
        expander = current.get(self.expander)
        member = current.get(self.member)
        if expander is None or member is None:
            _reject(ValidationCode.MISSING)
        outcome = next((item for item in expander.operation.outcomes if item.name == self.expansion_outcome), None)
        if (
            outcome is None
            or self.membership_port not in outcome.produced_ports
            or self.item_input not in {port.name for port in member.operation.inputs}
        ):
            _reject(ValidationCode.MISSING)


EvidencePort: TypeAlias = str | MapItemPort


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class EvidencePromise(_PrivateValue):
    name: str
    meaning: str
    subject_port: EvidencePort
    consumed_ports: frozenset[EvidencePort]
    coverage: frozenset[CoverageAtom]

    def __post_init__(self) -> None:
        _frozenset_of(self.consumed_ports, (str, MapItemPort))
        _frozenset_of(self.coverage, CoverageAtom)
        if not isinstance(self.subject_port, (str, MapItemPort)):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(
            strings=(
                self.name,
                self.meaning,
                *(port for port in (self.subject_port, *self.consumed_ports) if isinstance(port, str)),
            )
        )


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
                if (
                    isinstance(promise.subject_port, str)
                    and promise.subject_port not in all_ports
                    or any(isinstance(port, str) and port not in inputs for port in promise.consumed_ports)
                ):
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
        if any(
            isinstance(port, MapItemPort)
            for outcome in self.operation.outcomes
            for promise in outcome.evidence
            for port in (promise.subject_port, *promise.consumed_ports)
        ):
            _reject(ValidationCode.UNSUPPORTED)


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
class ContextInputRef(_PrivateValue):
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
    source: WorkflowInputRef | ContextInputRef | NodeOutputRef
    destination: NodeInputRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, (WorkflowInputRef, ContextInputRef, NodeOutputRef)) or not isinstance(
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
class MapDecl(_PrivateValue):
    expander: NodeId
    member: NodeId
    expansion_outcomes: frozenset[str]
    max_children: int
    item_input: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.expander, NodeId) or not isinstance(self.member, NodeId):
            _reject(ValidationCode.INVALID_TYPE)
        _frozenset_of(self.expansion_outcomes, str)
        _validate_scalars(strings=tuple(self.expansion_outcomes), nonnegative_integers=(self.max_children,))
        if self.item_input is not None:
            _validate_scalars(strings=(self.item_input,))
        if not self.expansion_outcomes:
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class KeyedJoinDecl(_PrivateValue):
    source: NodeId
    join: NodeId
    accepted_categories: frozenset[OutcomeClass]
    reduction: DynamicReduction

    def __post_init__(self) -> None:
        if not isinstance(self.source, NodeId) or not isinstance(self.join, NodeId):
            _reject(ValidationCode.INVALID_TYPE)
        _frozenset_of(self.accepted_categories, str)
        _validate_scalars(strings=(*self.accepted_categories, self.reduction))
        if not self.accepted_categories:
            _reject(ValidationCode.INVALID_VALUE)
        if not self.accepted_categories <= _OUTCOME_CLASSES or self.reduction != "all_by_key":
            _reject(ValidationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LoopInitialBinding(_PrivateValue):
    source: WorkflowInputRef | NodeOutputRef
    destination: NodeInputRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, (WorkflowInputRef, NodeOutputRef)) or not isinstance(
            self.destination, NodeInputRef
        ):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LoopCarriedBinding(_PrivateValue):
    source: NodeOutputRef
    destination: NodeInputRef

    def __post_init__(self) -> None:
        if not isinstance(self.source, NodeOutputRef) or not isinstance(self.destination, NodeInputRef):
            _reject(ValidationCode.INVALID_TYPE)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LoopDecl(_PrivateValue):
    starter: NodeId
    member: NodeId
    join: NodeId
    enter_outcomes: frozenset[str]
    bypass_outcomes: frozenset[str]
    continue_outcomes: frozenset[str]
    exit_outcomes: frozenset[str]
    initial: tuple[LoopInitialBinding, ...]
    carried: tuple[LoopCarriedBinding, ...]
    max_iterations: int

    def __post_init__(self) -> None:
        if not all(isinstance(node, NodeId) for node in (self.starter, self.member, self.join)):
            _reject(ValidationCode.INVALID_TYPE)
        for outcomes in (
            self.enter_outcomes,
            self.bypass_outcomes,
            self.continue_outcomes,
            self.exit_outcomes,
        ):
            _frozenset_of(outcomes, str)
            _validate_scalars(strings=tuple(outcomes))
            if not outcomes:
                _reject(ValidationCode.INVALID_VALUE)
        _tuple_of(self.initial, LoopInitialBinding)
        _tuple_of(self.carried, LoopCarriedBinding)
        _validate_scalars(nonnegative_integers=(self.max_iterations,))
        if self.enter_outcomes & self.bypass_outcomes or self.continue_outcomes & self.exit_outcomes:
            _reject(ValidationCode.OVERLAP)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, eq=False)
class DynamicScope(_PrivateValue):
    workflow: AdmittedWorkflow
    maps: tuple[MapDecl, ...]
    joins: tuple[KeyedJoinDecl, ...]
    loops: tuple[LoopDecl, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.workflow, AdmittedWorkflow):
            _reject(ValidationCode.INVALID_TYPE)
        _tuple_of(self.maps, MapDecl)
        _tuple_of(self.joins, KeyedJoinDecl)
        _tuple_of(self.loops, LoopDecl)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, DynamicScope)
            and self.workflow == other.workflow
            and frozenset(self.maps) == frozenset(other.maps)
            and frozenset(self.joins) == frozenset(other.joins)
            and frozenset(self.loops) == frozenset(other.loops)
        )

    def __hash__(self) -> int:
        return hash((self.workflow, frozenset(self.maps), frozenset(self.joins), frozenset(self.loops)))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class DynamicLimits(_PrivateValue):
    max_maps: int
    max_joins: int
    max_loops: int
    max_children_per_map: int
    max_iterations_per_loop: int
    max_dynamic_depth: int
    max_activation_occurrences: int

    def __post_init__(self) -> None:
        _validate_scalars(
            nonnegative_integers=(
                self.max_maps,
                self.max_joins,
                self.max_loops,
                self.max_children_per_map,
                self.max_iterations_per_loop,
            ),
            positive_integers=(self.max_dynamic_depth, self.max_activation_occurrences),
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProtectionRequirement(_PrivateValue):
    outcome: str
    meaning: str
    subject_port: EvidencePort
    consumed_ports: frozenset[EvidencePort]
    coverage: frozenset[CoverageAtom]
    candidate_port: str | None = None

    def __post_init__(self) -> None:
        _frozenset_of(self.consumed_ports, (str, MapItemPort))
        _frozenset_of(self.coverage, CoverageAtom)
        if not isinstance(self.subject_port, (str, MapItemPort)):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(
            strings=(
                self.outcome,
                self.meaning,
                *(port for port in (self.subject_port, *self.consumed_ports) if isinstance(port, str)),
            )
        )
        if self.candidate_port is not None:
            _validate_scalars(strings=(self.candidate_port,))
        elif isinstance(self.subject_port, MapItemPort):
            _reject(ValidationCode.MISSING)
        domains = {
            (port.path, port.expander, port.member, port.item_input, port.expansion_outcome, port.membership_port)
            for port in (self.subject_port, *self.consumed_ports)
            if isinstance(port, MapItemPort)
        }
        if len(domains) > 1:
            _reject(ValidationCode.CONTRADICTORY)


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

    @property
    def map_item_ports(self) -> frozenset[MapItemPort]:
        """Retained static endpoints that dynamic admission must resolve."""
        declarations = (
            *(promise for outcome in self.interface.outcomes for promise in outcome.evidence),
            *self.protection_requirements,
        )
        return frozenset(
            port
            for declaration in declarations
            for port in (declaration.subject_port, *declaration.consumed_ports)
            if isinstance(port, MapItemPort)
        )

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


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class AdmittedActivationWorkflow(_PrivateValue):
    """A statically admitted workflow with finite dynamic declarations."""

    workflow: AdmittedWorkflow
    scopes: tuple[DynamicScope, ...]
    limits: DynamicLimits
    activation_upper_bound: int
    dynamic_depth: int

    def __init__(
        self,
        *,
        _key: object,
        workflow: AdmittedWorkflow,
        scopes: tuple[DynamicScope, ...],
        limits: DynamicLimits,
        activation_upper_bound: int,
        dynamic_depth: int,
    ) -> None:
        if _key is not _ADMISSION_KEY:
            raise TypeError("admitted activation workflows must be created by admission")
        object.__setattr__(self, "workflow", workflow)
        object.__setattr__(self, "scopes", scopes)
        object.__setattr__(self, "limits", limits)
        object.__setattr__(self, "activation_upper_bound", activation_upper_bound)
        object.__setattr__(self, "dynamic_depth", dynamic_depth)
