# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Workflow declarations and sealed admitted values."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Any, Literal, Never, TypeAlias

from anonymizer.engine.graph_sdk._graph_values import _FACTORY_KEY, ContractViolation, ValidationCode
from anonymizer.engine.graph_sdk._graph_values import _OpaqueIdentity as _BaseOpaqueIdentity
from anonymizer.engine.graph_sdk._graph_values import _PrivateValue as _BasePrivateValue
from anonymizer.engine.graph_sdk._workflow_analysis import _equal, _visited_pop, _visits

OutcomeClass: TypeAlias = Literal["success", "failure", "cancelled", "lost", "blocked", "inconsistent"]


CoverageKind: TypeAlias = Literal["field", "source_view", "evaluation"]


CaptureMode: TypeAlias = Literal["whole_artifact"]


DynamicReduction: TypeAlias = Literal["all_by_key"]


_OUTCOME_CLASSES = frozenset(("success", "failure", "cancelled", "lost", "blocked", "inconsistent"))


_COVERAGE_KINDS = frozenset(("field", "source_view", "evaluation"))


_CAPTURE_MODES = frozenset(("whole_artifact",))


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
    integers = (*_visits(nonnegative_integers), *_visits(positive_integers))
    if any(not isinstance(value, str) for value in _visits(strings)) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in _visits(integers)
    ):
        _reject(ValidationCode.INVALID_TYPE)
    nonnegative_values = tuple(value for value in _visits(nonnegative_integers) if isinstance(value, int))
    positive_values = tuple(value for value in _visits(positive_integers) if isinstance(value, int))
    if (
        any(not value for value in _visits(strings))
        or any(value < 0 for value in _visits(nonnegative_values))
        or any(value < 1 for value in _visits(positive_values))
    ):
        _reject(ValidationCode.INVALID_VALUE)


def _tuple_of(value: object, member_type: type[Any] | tuple[type[Any], ...]) -> None:
    if not isinstance(value, tuple) or any(not isinstance(member, member_type) for member in _visits(value)):
        _reject(ValidationCode.INVALID_TYPE)


def _frozenset_of(value: object, member_type: type[Any] | tuple[type[Any], ...]) -> None:
    if not isinstance(value, frozenset) or any(not isinstance(member, member_type) for member in _visits(value)):
        _reject(ValidationCode.INVALID_TYPE)


class _PrivateValue(_BasePrivateValue):
    __slots__ = ()
    _pickle_error = _PICKLE_ERROR


class _OpaqueIdentity(_BaseOpaqueIdentity):
    __slots__ = ()
    _pickle_error = _PICKLE_ERROR


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
            strings=(
                self.output,
                *_visits(self.inputs),
                *(_visits((self.identity_input,) if self.identity_input is not None else ())),
            )
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
        return replace(self, path=(container, *_visits(self.path)))

    def validate_in(self, *, workflow: WorkflowId, nodes: tuple[Operation, ...]) -> None:
        """Validate the static route; dynamic admission binds its map declaration."""
        owner = workflow
        current = {node.id: node for node in _visits(nodes)}
        for container in _visits(self.path):
            if container.workflow != owner:
                _reject(ValidationCode.FOREIGN_OWNER)
            node = current.get(container)
            if node is None:
                _reject(ValidationCode.MISSING)
            if not isinstance(node, SubgraphNode):
                _reject(ValidationCode.UNSUPPORTED)
            owner = node.body.workflow
            current = {child.id: child for child in _visits(node.body.operations)}
        if self.expander.workflow != owner:
            _reject(ValidationCode.FOREIGN_OWNER)
        expander = current.get(self.expander)
        member = current.get(self.member)
        if expander is None or member is None:
            _reject(ValidationCode.MISSING)
        outcome = next(
            (item for item in _visits(expander.operation.outcomes) if item.name == self.expansion_outcome), None
        )
        if (
            outcome is None
            or self.membership_port not in outcome.produced_ports
            or self.item_input not in {port.name for port in _visits(member.operation.inputs)}
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
                *_visits(
                    (
                        port
                        for port in _visits((self.subject_port, *_visits(self.consumed_ports)))
                        if isinstance(port, str)
                    )
                ),
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ModelRequirement(_PrivateValue):
    capability: str
    revision: int

    def __post_init__(self) -> None:
        _validate_scalars(strings=(self.capability,), positive_integers=(self.revision,))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ResourceCeiling(_PrivateValue):
    """Declared ceilings; absent request or byte bounds make no corresponding hard promise."""

    max_activations: int
    max_model_requests: int | None
    max_input_bytes: int | None
    max_output_bytes: int | None

    def __post_init__(self) -> None:
        _validate_scalars(
            nonnegative_integers=(
                self.max_activations,
                *(
                    value
                    for value in (self.max_model_requests, self.max_input_bytes, self.max_output_bytes)
                    if value is not None
                ),
            )
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class OutcomeSpec(_PrivateValue):
    name: str
    category: OutcomeClass
    produced_ports: frozenset[str]
    context: frozenset[ContextUse]
    evidence: frozenset[EvidencePromise]
    model_requirements: frozenset[ModelRequirement]
    ceiling: ResourceCeiling

    def __post_init__(self) -> None:
        _frozenset_of(self.produced_ports, str)
        _frozenset_of(self.context, ContextUse)
        _frozenset_of(self.evidence, EvidencePromise)
        _frozenset_of(self.model_requirements, ModelRequirement)
        if not isinstance(self.ceiling, ResourceCeiling):
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(strings=(self.name, self.category, *_visits(self.produced_ports)))
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
        input_names = [port.name for port in _visits(self.inputs)]
        output_names = [port.name for port in _visits(self.outputs)]
        outcome_names = [outcome.name for outcome in _visits(self.outcomes)]
        dependency_outputs = [dependency.output for dependency in _visits(self.output_dependencies)]
        if any(
            len(values) != len(set(values))
            for values in _visits((input_names + output_names, outcome_names, dependency_outputs))
        ):
            _reject(ValidationCode.DUPLICATE)
        if any(
            len(outcome.evidence) != len({promise.name for promise in _visits(outcome.evidence)})
            for outcome in _visits(self.outcomes)
        ):
            _reject(ValidationCode.DUPLICATE)
        inputs = set(input_names)
        outputs = set(output_names)
        if set(dependency_outputs) != outputs:
            _reject(ValidationCode.MISSING)
        input_types = {port.name: port.artifact_type for port in _visits(self.inputs)}
        output_types = {port.name: port.artifact_type for port in _visits(self.outputs)}
        for dependency in _visits(self.output_dependencies):
            if dependency.output not in outputs or not dependency.inputs <= inputs:
                _reject(ValidationCode.MISSING)
            if dependency.identity_input is not None:
                if dependency.identity_input not in dependency.inputs:
                    _reject(ValidationCode.CONTRADICTORY)
                if input_types[dependency.identity_input] != output_types[dependency.output]:
                    _reject(ValidationCode.CONTRADICTORY)
        all_ports = inputs | outputs
        produced_any: set[str] = set()
        for outcome in _visits(self.outcomes):
            if not outcome.produced_ports <= outputs:
                _reject(ValidationCode.MISSING)
            produced_any.update(_visits(outcome.produced_ports))
            if any(use.port not in all_ports for use in _visits(outcome.context)):
                _reject(ValidationCode.MISSING)
            for promise in _visits(outcome.evidence):
                if (
                    isinstance(promise.subject_port, str)
                    and promise.subject_port not in all_ports
                    or any(isinstance(port, str) and port not in inputs for port in _visits(promise.consumed_ports))
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
            for outcome in _visits(self.operation.outcomes)
            for promise in _visits(outcome.evidence)
            for port in _visits((promise.subject_port, *_visits(promise.consumed_ports)))
        ):
            _reject(ValidationCode.UNSUPPORTED)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class SubgraphNode(_PrivateValue):
    id: NodeId
    operation: OperationSpec
    body: Graph

    def __post_init__(self) -> None:
        if (
            not isinstance(self.id, NodeId)
            or not isinstance(self.operation, OperationSpec)
            or not isinstance(self.body, Graph)
        ):
            _reject(ValidationCode.INVALID_TYPE)


Operation: TypeAlias = OperationNode | SubgraphNode


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
        _validate_scalars(strings=(*_visits(self.accepted_categories), self.reduction))
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
        if not all(isinstance(node, NodeId) for node in _visits((self.starter, self.member, self.join))):
            _reject(ValidationCode.INVALID_TYPE)
        for outcomes in _visits(
            (
                self.enter_outcomes,
                self.bypass_outcomes,
                self.continue_outcomes,
                self.exit_outcomes,
            )
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


Node: TypeAlias = Operation | ChoiceDecl | MapDecl | LoopDecl | KeyedJoinDecl


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
                *_visits(
                    (
                        port
                        for port in _visits((self.subject_port, *_visits(self.consumed_ports)))
                        if isinstance(port, str)
                    )
                ),
            )
        )
        if self.candidate_port is not None:
            _validate_scalars(strings=(self.candidate_port,))
        elif isinstance(self.subject_port, MapItemPort):
            _reject(ValidationCode.MISSING)
        domains = {
            (port.path, port.expander, port.member, port.item_input, port.expansion_outcome, port.membership_port)
            for port in _visits((self.subject_port, *_visits(self.consumed_ports)))
            if isinstance(port, MapItemPort)
        }
        if len(domains) > 1:
            _reject(ValidationCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class WorkflowLimits(_PrivateValue):
    """Declaration bounds and a separate visit budget for compiler analysis.

    ``max_choice_states`` bounds declared selector combinations. The positive
    exact integer ``max_analysis_work`` bounds declaration and relation visits
    across one compilation, including nested/replacement analysis. Exhaustion
    raises ``ANALYSIS_LIMIT_EXCEEDED`` before an admitted plan is returned.
    """

    max_nodes: int
    max_bindings: int
    max_sequence_edges: int
    max_choices: int
    max_branch_members: int
    max_subgraph_depth: int
    max_choice_states: int
    max_analysis_work: int = 1_000_000

    def __post_init__(self) -> None:
        if type(self.max_analysis_work) is not int:
            _reject(ValidationCode.INVALID_TYPE)
        _validate_scalars(
            nonnegative_integers=(
                self.max_bindings,
                self.max_sequence_edges,
                self.max_choices,
                self.max_branch_members,
            ),
            positive_integers=(self.max_nodes, self.max_subgraph_depth, self.max_choice_states, self.max_analysis_work),
        )


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False, eq=False)
class Graph(_PrivateValue):
    """Immutable graph body containing operations and finite control nodes."""

    workflow: WorkflowId
    interface: OperationSpec
    nodes: tuple[Node, ...]
    input_bindings: frozenset[InputBinding]
    output_bindings: frozenset[OutputBinding]
    outcome_bindings: frozenset[OutcomeBinding]
    sequence: frozenset[SequenceEdge]
    protection_requirements: frozenset[ProtectionRequirement]
    protection_eligible_outcomes: frozenset[str]
    unmet_protection: frozenset[ProtectionRequirement]
    limits: WorkflowLimits
    expanded_node_count: int
    dynamic_limits: DynamicLimits
    activation_upper_bound: int
    dynamic_depth: int

    @property
    def operations(self) -> tuple[Operation, ...]:
        """Executable nodes; control nodes remain in the same graph body."""
        return tuple(node for node in _visits(self.nodes) if isinstance(node, (OperationNode, SubgraphNode)))

    @property
    def choices(self) -> frozenset[ChoiceDecl]:
        return frozenset(node for node in _visits(self.nodes) if isinstance(node, ChoiceDecl))

    @property
    def maps(self) -> tuple[MapDecl, ...]:
        return tuple(node for node in _visits(self.nodes) if isinstance(node, MapDecl))

    @property
    def loops(self) -> tuple[LoopDecl, ...]:
        return tuple(node for node in _visits(self.nodes) if isinstance(node, LoopDecl))

    @property
    def joins(self) -> tuple[KeyedJoinDecl, ...]:
        return tuple(node for node in _visits(self.nodes) if isinstance(node, KeyedJoinDecl))

    def bodies(self) -> tuple[Graph, ...]:
        """Return the graph topology, visiting each shared body once."""
        pending: list[Graph] = [self]
        seen: set[int] = set()
        bodies: list[Graph] = []
        while pending:
            current = _visited_pop(pending)
            if id(current) in seen:
                continue
            seen.add(id(current))
            bodies.append(current)
            pending.extend(node.body for node in _visits(current.operations) if isinstance(node, SubgraphNode))
        return tuple(_visits(bodies))

    @property
    def map_item_ports(self) -> frozenset[MapItemPort]:
        """Retained static endpoints that dynamic admission must resolve."""
        declarations = (
            *_visits(
                (promise for outcome in _visits(self.interface.outcomes) for promise in _visits(outcome.evidence))
            ),
            *_visits(self.protection_requirements),
        )
        return frozenset(_map_item_ports(_visits(declarations)))

    def __init__(
        self,
        *,
        _key: object,
        workflow: WorkflowId,
        interface: OperationSpec,
        nodes: tuple[Node, ...],
        input_bindings: frozenset[InputBinding],
        output_bindings: frozenset[OutputBinding],
        outcome_bindings: frozenset[OutcomeBinding],
        sequence: frozenset[SequenceEdge],
        protection_requirements: frozenset[ProtectionRequirement],
        protection_eligible_outcomes: frozenset[str],
        unmet_protection: frozenset[ProtectionRequirement],
        limits: WorkflowLimits,
        expanded_node_count: int,
        dynamic_limits: DynamicLimits,
    ) -> None:
        if _key is not _ADMISSION_KEY:
            raise TypeError("admitted workflows must be created by admission")
        for name, value in locals().copy().items():
            if name not in {"self", "_key"}:
                object.__setattr__(self, name, value)

        children = tuple(node.body for node in _visits(self.operations) if isinstance(node, SubgraphNode))
        factors = {item.member: item.max_children for item in _visits(self.maps)}
        factors.update((item.member, item.max_iterations) for item in _visits(self.loops))
        object.__setattr__(
            self,
            "activation_upper_bound",
            sum(
                factors.get(node.id, 1)
                * (1 + (node.body.activation_upper_bound if isinstance(node, SubgraphNode) else 0))
                for node in _visits(self.operations)
            ),
        )
        object.__setattr__(
            self, "dynamic_depth", 1 + max((child.dynamic_depth for child in _visits(children)), default=0)
        )

    def __eq__(self, other: object) -> bool:
        if self is other:
            return True
        if not isinstance(other, Graph):
            return False
        pending: list[tuple[Graph, Graph]] = [(self, other)]
        seen: set[tuple[int, int]] = set()
        while pending:
            left, right = _visited_pop(pending)
            pair = (id(left), id(right))
            if pair in seen:
                continue
            seen.add(pair)
            if not _equal(_workflow_local_state(left), _workflow_local_state(right)):
                return False
            left_nodes = {node.id: node for node in _visits(left.operations)}
            right_nodes = {node.id: node for node in _visits(right.operations)}
            if not _equal(left_nodes.keys(), right_nodes.keys()):
                return False
            for node_id, left_node in _visits(left_nodes.items()):
                right_node = right_nodes[node_id]
                if type(left_node) is not type(right_node) or not _equal(left_node.operation, right_node.operation):
                    return False
                if isinstance(left_node, SubgraphNode):
                    assert isinstance(right_node, SubgraphNode)
                    pending.append((left_node.body, right_node.body))
        return True

    def __hash__(self) -> int:
        pending: list[tuple[Graph, bool]] = [(self, False)]
        hashes: dict[int, int] = {}
        while pending:
            workflow, children_visited = _visited_pop(pending)
            key = id(workflow)
            if key in hashes:
                continue
            children = [node.body for node in _visits(workflow.operations) if isinstance(node, SubgraphNode)]
            if not children_visited:
                pending.append((workflow, True))
                pending.extend((child, False) for child in _visits(children) if id(child) not in hashes)
                continue
            node_hashes = frozenset(
                hash((type(node), node.id, node.operation, hashes[id(node.body)]))
                if isinstance(node, SubgraphNode)
                else hash((type(node), node.id, node.operation))
                for node in _visits(workflow.operations)
            )
            hashes[key] = hash((_workflow_local_state(workflow), node_hashes))
        return hashes[id(self)]


def _map_item_ports(declarations: Iterable[EvidencePromise | ProtectionRequirement]) -> Iterator[MapItemPort]:
    """Extract dynamic endpoints; callers retain declaration traversal metering."""
    for declaration in declarations:
        for port in _visits((declaration.subject_port, *_visits(declaration.consumed_ports))):
            if isinstance(port, MapItemPort):
                yield port


def _workflow_local_state(workflow: Graph) -> tuple[object, ...]:
    return (
        workflow.workflow,
        workflow.interface,
        workflow.input_bindings,
        workflow.output_bindings,
        workflow.outcome_bindings,
        workflow.sequence,
        workflow.choices,
        frozenset(_visits(workflow.maps)),
        frozenset(_visits(workflow.loops)),
        frozenset(_visits(workflow.joins)),
        workflow.dynamic_limits,
        workflow.activation_upper_bound,
        workflow.dynamic_depth,
        workflow.protection_requirements,
        workflow.protection_eligible_outcomes,
        workflow.unmet_protection,
        workflow.limits,
        workflow.expanded_node_count,
    )


def _analysis_limit(*args: object, **kwargs: object) -> int | None:
    """Resolve the compiler envelope only from SDK-owned admission values."""
    owner = kwargs.get("workflow", args[0] if args else None)
    limits = kwargs.get("limits")
    if isinstance(limits, WorkflowLimits):
        return limits.max_analysis_work
    if isinstance(owner, Graph):
        return owner.limits.max_analysis_work
    return None
