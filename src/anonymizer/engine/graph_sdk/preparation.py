# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure capability binding and bounded preparation for admitted graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Never, SupportsIndex, TypeAlias

from anonymizer.engine.graph_sdk._workflow import reachable_workflows
from anonymizer.engine.graph_sdk.capabilities import (
    ImplementationCapability,
    ImplementationSelection,
    PreparationCode,
    PreparationRejected,
    SelectedImplementation,
    config_size,
    validate_capability,
)
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.graph._values import DatumId, PlanId
from anonymizer.graph.activation import ActivationLimits
from anonymizer.graph.workflow import (
    AdmittedActivationWorkflow,
    AdmittedWorkflow,
    ArtifactType,
    ContextInputRef,
    NodeId,
    OperationNode,
    ProtectionRequirement,
    StateEffect,
    SubgraphNode,
)

PreparationPurpose: TypeAlias = Literal["protection", "execution_only"]
_PREPARED_KEY = object()


def _reject(code: PreparationCode) -> Never:
    raise PreparationRejected(code) from None


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
        raise TypeError("preparation value serialization is not supported")


def _integer(value: object, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _reject(PreparationCode.INVALID_TYPE)
    if value < (1 if positive else 0):
        _reject(PreparationCode.INVALID_VALUE)
    return value


def _text(value: object) -> str:
    if not isinstance(value, str):
        _reject(PreparationCode.INVALID_TYPE)
    if not value:
        _reject(PreparationCode.INVALID_VALUE)
    return value


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class BoundInput(_PrivateValue):
    target: DatumId
    port: str
    source: DatumId
    artifact_type: ArtifactType

    def __post_init__(self) -> None:
        if (
            not isinstance(self.target, DatumId)
            or not isinstance(self.source, DatumId)
            or not isinstance(self.artifact_type, ArtifactType)
        ):
            _reject(PreparationCode.INVALID_TYPE)
        _text(self.port)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateRevision(_PrivateValue):
    effect: StateEffect
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.effect, StateEffect):
            _reject(PreparationCode.INVALID_TYPE)
        _integer(self.revision, positive=True)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class StateRevisionView(_PrivateValue):
    revisions: frozenset[StateRevision]

    def __post_init__(self) -> None:
        if not isinstance(self.revisions, frozenset) or any(
            not isinstance(revision, StateRevision) for revision in self.revisions
        ):
            _reject(PreparationCode.INVALID_TYPE)

    @classmethod
    def from_revisions(cls, *, revisions: tuple[StateRevision, ...]) -> StateRevisionView:
        """Validate raw revision keys before converting them to a canonical set."""
        if not isinstance(revisions, tuple) or any(not isinstance(item, StateRevision) for item in revisions):
            _reject(PreparationCode.INVALID_TYPE)
        effects = [item.effect for item in revisions]
        if len(effects) != len(set(effects)):
            _reject(PreparationCode.DUPLICATE)
        return cls(revisions=frozenset(revisions))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PreparationConfiguration(_PrivateValue):
    purpose: PreparationPurpose
    required_protection_outcomes: frozenset[str]
    hard_request_limit: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.purpose, str) or not isinstance(self.required_protection_outcomes, frozenset):
            _reject(PreparationCode.INVALID_TYPE)
        if any(not isinstance(outcome, str) for outcome in self.required_protection_outcomes):
            _reject(PreparationCode.INVALID_TYPE)
        if self.purpose not in {"protection", "execution_only"} or any(
            not outcome for outcome in self.required_protection_outcomes
        ):
            _reject(PreparationCode.INVALID_VALUE)
        if self.hard_request_limit is not None:
            _integer(self.hard_request_limit)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class PreparationLimits(_PrivateValue):
    max_bound_inputs: int
    max_state_revisions: int
    max_selections: int
    max_config_depth: int
    max_config_atoms: int
    max_total_activation_slots: int
    max_capabilities: int

    def __post_init__(self) -> None:
        for value in (
            self.max_bound_inputs,
            self.max_state_revisions,
            self.max_selections,
            self.max_config_depth,
            self.max_config_atoms,
            self.max_total_activation_slots,
            self.max_capabilities,
        ):
            _integer(value)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ReservationSlot(_PrivateValue):
    index: int
    template: NodeId
    parent_index: int | None
    iteration: int | None

    def __post_init__(self) -> None:
        _integer(self.index)
        if not isinstance(self.template, NodeId):
            _reject(PreparationCode.INVALID_TYPE)
        if self.parent_index is not None:
            _integer(self.parent_index)
        if self.iteration is not None:
            _integer(self.iteration)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TargetOccurrenceMap(_PrivateValue):
    target: DatumId
    ordinal: int
    occurrence_offset: int

    def __post_init__(self) -> None:
        if not isinstance(self.target, DatumId):
            _reject(PreparationCode.INVALID_TYPE)
        _integer(self.ordinal)
        _integer(self.occurrence_offset)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ProtectionEligibility(_PrivateValue):
    purpose: PreparationPurpose
    required_outcomes: frozenset[str]
    declared_eligible_outcomes: frozenset[str]
    unmet_requirements: frozenset[ProtectionRequirement]
    eligible_for_requested_protection: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.purpose, str)
            or not isinstance(self.required_outcomes, frozenset)
            or not isinstance(self.declared_eligible_outcomes, frozenset)
            or not isinstance(self.unmet_requirements, frozenset)
            or not isinstance(self.eligible_for_requested_protection, bool)
            or any(not isinstance(outcome, str) for outcome in self.required_outcomes)
            or any(not isinstance(outcome, str) for outcome in self.declared_eligible_outcomes)
            or any(not isinstance(item, ProtectionRequirement) for item in self.unmet_requirements)
        ):
            _reject(PreparationCode.INVALID_TYPE)
        if (
            self.purpose not in {"protection", "execution_only"}
            or any(not outcome for outcome in self.required_outcomes)
            or any(not outcome for outcome in self.declared_eligible_outcomes)
        ):
            _reject(PreparationCode.INVALID_VALUE)


@dataclass(frozen=True, slots=True, repr=False, eq=False, init=False)
class PreparedPlan(_PrivateValue):
    plan: PlanId
    data: ValidatedDataGraph
    workflow: AdmittedActivationWorkflow
    activation_limits: ActivationLimits
    bound_inputs: frozenset[BoundInput]
    configuration: PreparationConfiguration
    state: StateRevisionView
    implementations: frozenset[SelectedImplementation]
    reservation_recipe: tuple[ReservationSlot, ...]
    target_occurrences: tuple[TargetOccurrenceMap, ...]
    activation_slots_per_target: int
    total_activation_slots: int
    declared_request_upper_bound: int
    eligibility: ProtectionEligibility
    limits: PreparationLimits

    def __init__(
        self,
        *,
        _key: object,
        plan: PlanId,
        data: ValidatedDataGraph,
        workflow: AdmittedActivationWorkflow,
        activation_limits: ActivationLimits,
        bound_inputs: frozenset[BoundInput],
        configuration: PreparationConfiguration,
        state: StateRevisionView,
        implementations: frozenset[SelectedImplementation],
        reservation_recipe: tuple[ReservationSlot, ...],
        target_occurrences: tuple[TargetOccurrenceMap, ...],
        activation_slots_per_target: int,
        total_activation_slots: int,
        declared_request_upper_bound: int,
        eligibility: ProtectionEligibility,
        limits: PreparationLimits,
    ) -> None:
        if _key is not _PREPARED_KEY:
            raise TypeError("prepared plans are created by prepare")
        for name, value in locals().copy().items():
            if name not in {"self", "_key"}:
                object.__setattr__(self, name, value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, PreparedPlan) and self.plan == other.plan

    def __hash__(self) -> int:
        return hash(self.plan)


def prepare(
    *,
    data: ValidatedDataGraph,
    workflow: AdmittedActivationWorkflow,
    activation_limits: ActivationLimits,
    bound_inputs: tuple[BoundInput, ...],
    configuration: PreparationConfiguration,
    state: StateRevisionView,
    selections: tuple[ImplementationSelection, ...],
    capabilities: tuple[ImplementationCapability, ...],
    limits: PreparationLimits,
) -> PreparedPlan:
    """Prepare one immutable graph plan without execution effects."""
    _validate_outer_types(
        data, workflow, activation_limits, bound_inputs, configuration, state, selections, capabilities, limits
    )
    if len(capabilities) > limits.max_capabilities:
        _reject(PreparationCode.LIMIT_EXCEEDED)
    if any(not isinstance(capability, ImplementationCapability) for capability in capabilities):
        _reject(PreparationCode.INVALID_TYPE)

    reachable = reachable_workflows(workflow)
    operation_nodes = {node.id: node for body in reachable for node in body.nodes if isinstance(node, OperationNode)}
    recipe, map_expanders = _reservation_recipe(workflow)
    target_occurrences = tuple(
        TargetOccurrenceMap(
            target=target,
            ordinal=ordinal,
            occurrence_offset=ordinal * len(recipe),
        )
        for ordinal, target in enumerate(data.targets)
    )
    total_slots = len(recipe) * len(data.targets)
    configuration_sizes = [config_size(selection.configuration) for selection in selections]
    configuration_sizes.extend(config_size(capability.configuration) for capability in capabilities)
    maximum_depth = max((depth for depth, _ in configuration_sizes), default=0)
    total_atoms = sum(atoms for _, atoms in configuration_sizes)
    recipe_depth = _recipe_depth(recipe)
    if (
        len(bound_inputs) > limits.max_bound_inputs
        or len(state.revisions) > limits.max_state_revisions
        or len(selections) > limits.max_selections
        or maximum_depth > limits.max_config_depth
        or total_atoms > limits.max_config_atoms
        or total_slots > limits.max_total_activation_slots
        or activation_limits.max_entries < len(recipe)
        or activation_limits.max_parent_depth < recipe_depth
        or activation_limits.max_events < 3 * len(recipe) + map_expanders
    ):
        _reject(PreparationCode.LIMIT_EXCEEDED)

    datum_ids = {datum.id for datum in data.datums}
    if any(item.target.graph != data.graph or item.source.graph != data.graph for item in bound_inputs):
        _reject(PreparationCode.FOREIGN_OWNER)
    if any(selection.node.workflow not in {item.workflow for item in reachable} for selection in selections):
        _reject(PreparationCode.FOREIGN_OWNER)

    input_keys = [(item.target, item.port) for item in bound_inputs]
    revision_effects = [revision.effect for revision in state.revisions]
    selected_nodes = [selection.node for selection in selections]
    if (
        len(input_keys) != len(set(input_keys))
        or len(revision_effects) != len(set(revision_effects))
        or len(selected_nodes) != len(set(selected_nodes))
        or len(capabilities) != len(set(capabilities))
    ):
        _reject(PreparationCode.DUPLICATE)

    context_inputs = {
        binding.source.port
        for binding in workflow.workflow.input_bindings
        if isinstance(binding.source, ContextInputRef)
    }
    root_inputs = {
        port.name: port.artifact_type for port in workflow.workflow.interface.inputs if port.name not in context_inputs
    }
    expected_inputs = {(target, port) for target in data.targets for port in root_inputs}
    observed_inputs = set(input_keys)
    if observed_inputs != expected_inputs or any(item.source not in datum_ids for item in bound_inputs):
        _reject(PreparationCode.MISSING_INPUT)
    reads = {
        effect
        for node in operation_nodes.values()
        for outcome in node.operation.outcomes
        for effect in outcome.state_effects
        if effect.kind == "read"
    }
    if set(revision_effects) != reads:
        _reject(PreparationCode.MISSING_STATE)
    implementations = _select_implementations(operation_nodes, selections, capabilities)
    if configuration.hard_request_limit is not None and any(
        selected.capability.effect == "external"
        and (
            selected.capability.request_visibility != "dispatch_and_settlement"
            or selected.capability.pre_dispatch_control != "executor"
            or selected.capability.retry_owner not in {"none", "executor"}
        )
        for selected in implementations
    ):
        _reject(PreparationCode.HARD_BUDGET_INCOMPATIBLE)

    eligibility = _eligibility(workflow.workflow, configuration)
    if any(item.artifact_type != root_inputs[item.port] for item in bound_inputs) or any(
        revision.effect.kind != "read" for revision in state.revisions
    ):
        _reject(PreparationCode.CONTRADICTORY)
    for capability in capabilities:
        validate_capability(capability)
    request_per_target = sum(
        max(
            (outcome.ceiling.max_model_requests for outcome in operation_nodes[slot.template].operation.outcomes),
            default=0,
        )
        for slot in recipe
        if slot.template in operation_nodes
    )
    return PreparedPlan(
        _key=_PREPARED_KEY,
        plan=PlanId.new(),
        data=data,
        workflow=workflow,
        activation_limits=activation_limits,
        bound_inputs=frozenset(bound_inputs),
        configuration=configuration,
        state=state,
        implementations=frozenset(implementations),
        reservation_recipe=recipe,
        target_occurrences=target_occurrences,
        activation_slots_per_target=len(recipe),
        total_activation_slots=total_slots,
        declared_request_upper_bound=request_per_target * len(data.targets),
        eligibility=eligibility,
        limits=limits,
    )


def recheck_capabilities(*, prepared: PreparedPlan, capabilities: tuple[ImplementationCapability, ...]) -> None:
    """Require every selected capability to remain byte-for-byte semantic equal."""
    if not isinstance(prepared, PreparedPlan) or not isinstance(capabilities, tuple):
        _reject(PreparationCode.INVALID_TYPE)
    if len(capabilities) > prepared.limits.max_capabilities:
        _reject(PreparationCode.LIMIT_EXCEEDED)
    if any(not isinstance(capability, ImplementationCapability) for capability in capabilities):
        _reject(PreparationCode.INVALID_TYPE)
    if len(capabilities) != len(set(capabilities)):
        _reject(PreparationCode.DUPLICATE)
    for capability in capabilities:
        validate_capability(capability)
    if any(selected.capability not in capabilities for selected in prepared.implementations):
        _reject(PreparationCode.CAPABILITY_CHANGED)


def _validate_outer_types(
    data: object,
    workflow: object,
    activation_limits: object,
    bound_inputs: object,
    configuration: object,
    state: object,
    selections: object,
    capabilities: object,
    limits: object,
) -> None:
    if (
        not isinstance(data, ValidatedDataGraph)
        or not isinstance(workflow, AdmittedActivationWorkflow)
        or not isinstance(activation_limits, ActivationLimits)
        or not isinstance(bound_inputs, tuple)
        or any(not isinstance(item, BoundInput) for item in bound_inputs)
        or not isinstance(configuration, PreparationConfiguration)
        or not isinstance(state, StateRevisionView)
        or not isinstance(selections, tuple)
        or any(not isinstance(item, ImplementationSelection) for item in selections)
        or not isinstance(capabilities, tuple)
        or not isinstance(limits, PreparationLimits)
    ):
        _reject(PreparationCode.INVALID_TYPE)


def _reservation_recipe(
    workflow: AdmittedActivationWorkflow,
) -> tuple[tuple[ReservationSlot, ...], int]:
    scopes = {id(scope.workflow): scope for scope in workflow.scopes}
    slots: list[ReservationSlot] = []
    map_expanders = 0

    def emit_scope(current: AdmittedWorkflow, container_parent: int | None) -> None:
        nonlocal map_expanders
        scope = scopes[id(current)]
        dynamic_members = {item.member for item in (*scope.maps, *scope.loops)}
        nodes = {node.id: node for node in current.nodes}
        children: dict[NodeId, list[tuple[NodeId, int, bool]]] = {}
        for item in scope.maps:
            children.setdefault(item.expander, []).append((item.member, item.max_children, False))
        for item in scope.loops:
            children.setdefault(item.starter, []).append((item.member, item.max_iterations, True))

        def emit_node(template: NodeId, parent: int | None, iteration: int | None) -> None:
            nonlocal map_expanders
            index = len(slots)
            slots.append(ReservationSlot(index=index, template=template, parent_index=parent, iteration=iteration))
            node = nodes[template]
            if any(item.expander == template for item in scope.maps):
                map_expanders += 1
            if isinstance(node, SubgraphNode):
                emit_scope(node.body, index)
            for member, count, is_loop in children.get(template, ()):
                for occurrence in range(count):
                    emit_node(member, index, occurrence if is_loop else None)

        for node in current.nodes:
            if node.id not in dynamic_members:
                emit_node(node.id, container_parent, None)

    emit_scope(workflow.workflow, None)
    if len(slots) != workflow.activation_upper_bound:
        _reject(PreparationCode.CONTRADICTORY)
    return tuple(slots), map_expanders


def _recipe_depth(recipe: tuple[ReservationSlot, ...]) -> int:
    depths: list[int] = []
    for slot in recipe:
        depth = 1
        parent = slot.parent_index
        while parent is not None:
            depth += 1
            parent = recipe[parent].parent_index
        depths.append(depth)
    return max(depths, default=0)


def _select_implementations(
    operations: dict[NodeId, OperationNode],
    selections: tuple[ImplementationSelection, ...],
    capabilities: tuple[ImplementationCapability, ...],
) -> tuple[SelectedImplementation, ...]:
    by_node = {selection.node: selection for selection in selections}
    if set(by_node) != set(operations):
        _reject(PreparationCode.UNSUPPORTED_CAPABILITY)
    selected: list[SelectedImplementation] = []
    for node_id, node in operations.items():
        selection = by_node[node_id]
        matches = [
            capability
            for capability in capabilities
            if capability.implementation == selection.implementation
            and capability.configuration == selection.configuration
            and capability.operation == node.operation
        ]
        if (
            len(matches) != 1
            or matches[0].attribution == "aggregate_only"
            or matches[0].error_reporting != "typed_terminal"
        ):
            _reject(PreparationCode.UNSUPPORTED_CAPABILITY)
        selected.append(SelectedImplementation(node=node_id, capability=matches[0]))
    return tuple(selected)


def _eligibility(workflow: AdmittedWorkflow, configuration: PreparationConfiguration) -> ProtectionEligibility:
    declared = workflow.protection_eligible_outcomes
    required = configuration.required_protection_outcomes
    eligible = bool(required) and required <= declared
    if configuration.purpose == "protection" and not eligible:
        _reject(PreparationCode.PROTECTION_INELIGIBLE)
    if configuration.purpose == "execution_only" and required:
        _reject(PreparationCode.PROTECTION_INELIGIBLE)
    return ProtectionEligibility(
        purpose=configuration.purpose,
        required_outcomes=required,
        declared_eligible_outcomes=declared,
        unmet_requirements=workflow.unmet_protection,
        eligible_for_requested_protection=eligible if configuration.purpose == "protection" else False,
    )
