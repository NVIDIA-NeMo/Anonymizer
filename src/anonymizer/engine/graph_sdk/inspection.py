# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure request inspection through exact occurrence and binding ownership joins."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, fields, is_dataclass
from functools import cache
from types import UnionType
from typing import Literal, TypeVar, get_args, get_origin, get_type_hints

from anonymizer.engine.graph_sdk._continuity_values import ContinuityCommit
from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_count, require_instance
from anonymizer.engine.graph_sdk._execution_values import (
    AdmittedExecutionPlan,
    ExecutionResult,
    OperationOutputKey,
    PublicationFailureFact,
)
from anonymizer.engine.graph_sdk._ingress import inspect_artifact
from anonymizer.engine.graph_sdk.context import (
    AdmittedContextPlan,
    BindingArtifactRef,
    BindingReceipt,
    BindingResult,
    BindingTerminal,
    SourceTerminal,
)
from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
from anonymizer.engine.graph_sdk.observations import (
    _OBSERVATION_KEY,
    CorrelationRef,
    LifecyclePair,
    ObservationClose,
    ObservationDelivery,
    ObservationFailure,
    ObservationReason,
    ObservationReceipt,
    ObservationSession,
    ObservationStart,
    ObservationTerminal,
    WorkMeasurement,
    _FactoryValue,
    _width,
)
from anonymizer.engine.graph_sdk.preparation import PreparedPlan
from anonymizer.engine.graph_sdk.qualification import QualificationResult
from anonymizer.engine.graph_sdk.records import TerminalFact
from anonymizer.engine.graph_sdk.requests import (
    ArtifactValue,
    BindingAssociation,
    BindingRequestScope,
    InvocationRequestScope,
    PhysicalRequestId,
    RequestAssociation,
    RequestAssociationSummary,
    RequestReceipt,
    RequestTerminalFact,
    SemanticAssociation,
)
from anonymizer.engine.graph_sdk.resources import CleanupFact
from anonymizer.graph._values import ArtifactRef, DatumId
from anonymizer.graph.workflow import Graph


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestInspection(PrivateValue):
    """First physical acceptance alongside current owner facts, without values.

    Multiple physical attempts may join the same semantic occurrence. Their first
    acceptance facts stay distinct; publication facts belong to the occurrence.
    Binding references retain their own type and never become graph references.
    """

    request: PhysicalRequestId
    association: RequestAssociation
    physical_terminal: RequestTerminalFact | None
    acceptance: RequestAssociationSummary | None
    ownership_resolved: bool
    target: DatumId | None
    operation_terminal: TerminalFact | None = None
    binding_terminal: SourceTerminal | None = None
    artifacts: tuple[ArtifactRef, ...] = ()
    binding_artifacts: tuple[BindingArtifactRef, ...] = ()
    publication_failure: PublicationFailureFact | None = None


def inspect_binding_requests(receipt: BindingReceipt) -> tuple[RequestInspection, ...]:
    """Join a binding receipt without inventing an invocation or loading values."""
    views: list[RequestInspection] = []
    for dispatch in receipt.requests.dispatches:
        terminal = next((fact for fact in receipt.requests.terminals if fact.request == dispatch.request), None)
        for association in dispatch.associations:
            acceptance = (
                None
                if terminal is None
                else next(
                    (row for row in terminal.results if row.association == association),
                    None,
                )
            )
            sources = tuple(
                source
                for source in receipt.sources
                if isinstance(association, BindingAssociation)
                and source.identity == association.declaration
                and source.identity.binding == receipt.binding
            )
            resolved = (
                isinstance(receipt.requests.scope, BindingRequestScope)
                and receipt.requests.scope.binding == receipt.binding
                and len(sources) == 1
            )
            source = sources[0] if resolved else None
            artifacts = tuple(
                item.reference
                for item in receipt.artifacts
                if source is not None
                and item.reference.declaration == source.identity
                and item.target == source.declaration.target
                and item.source == source.declaration.source
                and item.node == source.declaration.node
                and item.port == source.declaration.port
            )
            views.append(
                RequestInspection(
                    request=dispatch.request,
                    association=association,
                    physical_terminal=terminal,
                    acceptance=acceptance,
                    ownership_resolved=resolved,
                    target=None if source is None else source.declaration.target,
                    binding_terminal=None if source is None else source.terminal,
                    binding_artifacts=artifacts,
                )
            )
    return tuple(views)


def inspect_requests(result: ExecutionResult) -> tuple[RequestInspection, ...]:
    """Resolve first acceptance against exact final occurrence publication facts."""
    views: list[RequestInspection] = []
    for dispatch in result.requests.dispatches:
        terminal = next((fact for fact in result.requests.terminals if fact.request == dispatch.request), None)
        for association in dispatch.associations:
            acceptance = (
                None
                if terminal is None
                else next(
                    (row for row in terminal.results if row.association == association),
                    None,
                )
            )
            operations = tuple(
                fact
                for fact in result.record.terminals
                if isinstance(association, SemanticAssociation)
                and fact.attempt == association.task
                and fact.activation == association.task.activation
            )
            targets = tuple(
                target.target
                for target, state in zip(
                    result._execution.context.prepared.target_occurrences, result.states, strict=True
                )
                if isinstance(association, SemanticAssociation)
                and any(entry.activation == association.task.activation for entry in state.entries)
            )
            resolved = (
                len(operations) == len(targets) == 1
                and isinstance(result.requests.scope, InvocationRequestScope)
                and isinstance(association, SemanticAssociation)
                and association.task.activation.invocation == result.requests.scope.invocation
            )
            operation = operations[0] if resolved else None
            target = targets[0] if resolved else None
            entries = tuple(
                entry
                for state in result.states
                for entry in state.entries
                if operation is not None and entry.activation == operation.activation
            )
            expected_ports: frozenset[str] = frozenset()
            if len(entries) == 1 and entries[0].outcome is not None:
                node = result._execution.nodes[entries[0].template]
                outcomes = tuple(item for item in node.operation.outcomes if item.name == entries[0].outcome)
                if len(outcomes) == 1:
                    expected_ports = outcomes[0].produced_ports
                else:
                    resolved = False
            elif len(entries) != 1:
                resolved = False
            ports = tuple(
                port
                for port in result.ports
                if operation is not None
                and port.activation == operation.activation
                and port.target == target
                and port.port in expected_ports
            )
            resolved = (
                resolved and len(ports) == len(expected_ports) and {port.port for port in ports} == expected_ports
            )
            artifacts = tuple(
                dict.fromkeys(
                    port.artifact
                    for port in ports
                    if any(
                        fact.artifact == port.artifact
                        and isinstance(fact.key, OperationOutputKey)
                        and fact.key.activation == port.activation
                        and fact.key.target == target
                        and fact.key.port == port.port
                        for fact in result.provenance
                    )
                )
            )
            failures = tuple(
                fact
                for fact in result.publication_failures
                if fact.association == association and fact.target == target
            )
            # A partial provenance join is not evidence of complete publication.
            if any(port.artifact not in artifacts for port in ports) or len(failures) > 1:
                resolved = False
            views.append(
                RequestInspection(
                    request=dispatch.request,
                    association=association,
                    physical_terminal=terminal,
                    acceptance=acceptance,
                    ownership_resolved=resolved,
                    target=target,
                    operation_terminal=operation,
                    artifacts=artifacts if resolved else (),
                    publication_failure=failures[0] if len(failures) == 1 else None,
                )
            )
    bound = result._execution.context.bound_context
    if bound is not None:
        views.extend(inspect_binding_requests(bound.receipt))
    return tuple(views)


CountKnowledge = Literal["exact", "known_upper", "unknown"]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CountEstimate(PrivateValue):
    knowledge: CountKnowledge
    value: int | None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class GraphShapeSummary(PrivateValue):
    target_count: int
    template_node_count: int
    activation_slots_per_target: int
    total_activation_slots: int
    maximum_dynamic_depth: int


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RequestForecast(PrivateValue):
    planned: CountEstimate
    conditional: CountEstimate
    upper_bound: CountEstimate


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ActualRequestCounts(PrivateValue):
    total: CountEstimate
    initial: CountEstimate
    initial_binding: CountEstimate
    adaptive_retrieval: CountEstimate
    retry: CountEstimate
    correction: CountEstimate
    repair: CountEstimate
    failover: CountEstimate


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class UsageSummary(PrivateValue):
    input_units: CountEstimate
    output_units: CountEstimate
    requests_with_unknown_usage: CountEstimate


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class TerminalCounts(PrivateValue):
    success: int
    failure: int
    cancelled: int
    lost: int
    blocked: int
    inconsistent: int


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class RouteCounts(PrivateValue):
    local: int
    external: int
    decision: int
    context: int


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ResourceCounts(PrivateValue):
    local_in_flight: int
    remote_outstanding: int
    artifact_count: int
    artifact_bytes: int
    cleanup_closed: int
    cleanup_failed: int
    cleanup_unknown: int
    caller_left_open: int


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class QualificationProxy(PrivateValue):
    assessed_targets: int
    met: int
    unmet: int
    unknown: int
    verified_evidence: int
    withheld_targets: int


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ReasonCount(PrivateValue):
    reason: ObservationReason
    count: int


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class ExplainSummary(_FactoryValue):
    schema_version: Literal[1]
    shape: GraphShapeSummary
    requests: RequestForecast
    hard_budget_support: Literal["supported", "unsupported", "unknown"]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class InspectionSummary(_FactoryValue):
    schema_version: Literal[1]
    shape: GraphShapeSummary
    requests: ActualRequestCounts
    usage: UsageSummary
    routes: RouteCounts | None
    terminals: TerminalCounts | None
    resources: ResourceCounts | None
    qualification: QualificationProxy | None
    observation_delivery: ObservationDelivery


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class DiagnosisSummary(_FactoryValue):
    schema_version: Literal[1]
    reasons: tuple[ReasonCount, ...]
    terminal_unknown: bool
    usage_unknown: bool
    remote_work_unknown: bool
    observation_failed: bool


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class InspectionLimits(PrivateValue):
    max_nodes: int
    max_depth: int
    max_pairs: int
    max_events: int
    max_reason_counts: int
    max_encoded_bytes: int
    max_count: int

    def __post_init__(self) -> None:
        for field in fields(self):
            require_count(getattr(self, field.name))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class InspectionSource(_FactoryValue):
    limits: InspectionLimits
    prepared: PreparedPlan
    context: AdmittedContextPlan | None
    execution_plan: AdmittedExecutionPlan | None
    binding: BindingResult | None
    execution: ExecutionResult | None
    qualification: QualificationResult | None
    continuity: ContinuityCommit | None
    observations: ObservationClose | None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class BindingInspectionSource(_FactoryValue):
    data: ValidatedDataGraph
    workflow: Graph
    binding: BindingResult
    limits: InspectionLimits


@dataclass(frozen=True, slots=True, kw_only=True, repr=False, init=False)
class BindingInspectionSummary(_FactoryValue):
    schema_version: Literal[1]
    terminal: BindingTerminal
    requests: ActualRequestCounts
    usage: UsageSummary
    resources: ResourceCounts


InspectionValue = (
    ExplainSummary
    | InspectionSummary
    | DiagnosisSummary
    | ObservationReceipt
    | ObservationFailure
    | BindingInspectionSummary
)
_VALUES = (
    ExplainSummary,
    InspectionSummary,
    DiagnosisSummary,
    ObservationReceipt,
    ObservationFailure,
    BindingInspectionSummary,
)
_MEMBERS = (
    *_VALUES,
    CountEstimate,
    GraphShapeSummary,
    RequestForecast,
    ActualRequestCounts,
    UsageSummary,
    TerminalCounts,
    RouteCounts,
    ResourceCounts,
    QualificationProxy,
    ReasonCount,
    CorrelationRef,
    WorkMeasurement,
    ObservationStart,
    ObservationTerminal,
    LifecyclePair,
)


@cache
def _schema(kind: type) -> dict[str, object]:
    return get_type_hints(kind)


def _members(value: object) -> tuple[tuple[str, object], ...]:
    if type(value) not in _MEMBERS:
        return ()
    assert is_dataclass(value)
    return tuple((field.name, getattr(value, field.name)) for field in fields(value))


def _string_size(value: str) -> int:
    # Size only: no arbitrary caller string enters an allocated JSON buffer.
    return 2 + sum(
        2
        if char in {'"', "\\", "\b", "\f", "\n", "\r", "\t"}
        else 6
        if ord(char) < 32
        else 1
        if ord(char) < 128
        else 2
        if ord(char) < 2048
        else 3
        if ord(char) < 65536
        else 4
        for char in value
    )


def _bounds(value: InspectionValue, limits: InspectionLimits) -> None:
    nodes = size = 0
    pending: list[tuple[object, int, str]] = [(value, 1, "")]
    while pending:
        item, depth, name = pending.pop()
        nodes += 1
        if nodes > limits.max_nodes or depth > limits.max_depth:
            reject(EffectCode.LIMIT_EXCEEDED)
        members = _members(item)
        if members:
            size += 2 + max(0, len(members) - 1) + sum(_string_size(key) + 1 for key, _ in members)
            pending.extend((child, depth + 1, key) for key, child in members)
        elif type(item) in (tuple, frozenset):
            assert isinstance(item, (tuple, frozenset))
            if name == "pairs" and (len(item) > limits.max_pairs or 2 * len(item) > limits.max_events):
                reject(EffectCode.LIMIT_EXCEEDED)
            if name == "reasons" and len(item) > limits.max_reason_counts:
                reject(EffectCode.LIMIT_EXCEEDED)
            if len(item) > limits.max_nodes - nodes:
                reject(EffectCode.LIMIT_EXCEEDED)
            size += 2 + max(0, len(item) - 1)
            pending.extend((child, depth + 1, "") for child in item)
        elif type(item) is int:
            assert isinstance(item, int)
            if name != "schema_version" and (item < 0 or item > limits.max_count):
                reject(EffectCode.LIMIT_EXCEEDED)
            size += _width(abs(item)) + int(item < 0)
        elif type(item) is str:
            assert isinstance(item, str)
            size += _string_size(item)
        elif item is None or type(item) is bool:
            size += 4 if item is None or item is True else 5
        if size > limits.max_encoded_bytes:
            reject(EffectCode.LIMIT_EXCEEDED)


def _member_types(value: object, expected: object) -> None:
    origin, args = get_origin(expected), get_args(expected)
    if origin is UnionType:
        choices = [
            kind for kind in args if kind is type(value) or get_origin(kind) is Literal and value in get_args(kind)
        ]
        if not choices:
            reject(EffectCode.INVALID_TYPE)
        _member_types(value, choices[0])
    elif origin is Literal:
        if not any(type(value) is type(option) for option in args):
            reject(EffectCode.INVALID_TYPE)
        # Version validation is deliberately later than duplicate/missing facts.
        if args != (1,) and value not in args:
            reject(EffectCode.INVALID_VALUE)
    elif origin in (tuple, frozenset):
        if type(value) is not origin:
            reject(EffectCode.INVALID_TYPE)
        assert isinstance(value, (tuple, frozenset))
        for item in value:
            _member_types(item, args[0])
    elif type(value) is not expected:
        reject(EffectCode.INVALID_TYPE)
    elif type(value) in _MEMBERS:
        for name, annotation in _schema(type(value)).items():
            _member_types(getattr(value, name), annotation)


def _all_values(value: object) -> Iterator[object]:
    yield value
    for _, child in _members(value):
        if type(child) in (tuple, frozenset):
            assert isinstance(child, (tuple, frozenset))
            for item in child:
                yield from _all_values(item)
        else:
            yield from _all_values(child)


def _duplicates(value: InspectionValue) -> None:
    for item in _all_values(value):
        if type(item) is ObservationReceipt:
            ordinals = [pair.start.correlation.ordinal for pair in item.pairs]
            if len(set(ordinals)) != len(ordinals):
                reject(EffectCode.DUPLICATE)
        elif type(item) is DiagnosisSummary:
            reasons = [reason.reason for reason in item.reasons]
            if len(set(reasons)) != len(reasons):
                reject(EffectCode.DUPLICATE)


def _consistency(value: InspectionValue) -> None:
    _duplicates(value)
    values = tuple(_all_values(value))
    for item in values:
        if type(item) in _VALUES and getattr(item, "schema_version") != 1:
            reject(EffectCode.UNSUPPORTED)
    for item in values:
        if type(item) is CountEstimate:
            if (item.knowledge == "unknown") != (item.value is None):
                reject(EffectCode.CONTRADICTORY)
        elif type(item) is ObservationReceipt:
            if (
                item.delivered_pairs > len(item.pairs)
                or item.delivery == "complete"
                and item.delivered_pairs != len(item.pairs)
                or item.delivery == "not_configured"
                and item.delivered_pairs != 0
            ):
                reject(EffectCode.CONTRADICTORY)
            for index, pair in enumerate(item.pairs):
                correlation = pair.start.correlation
                if (
                    correlation.ordinal != index
                    or correlation.parent_ordinal is not None
                    and not 0 <= correlation.parent_ordinal < index
                    or pair.start.correlation != pair.terminal.correlation
                    or pair.start.boundary != pair.terminal.boundary
                ):
                    reject(EffectCode.CONTRADICTORY)
        elif type(item) is ActualRequestCounts:
            counts = [getattr(item, field.name) for field in fields(item) if field.name != "total"]
            if any(count.knowledge == "known_upper" for count in (*counts, item.total)):
                reject(EffectCode.CONTRADICTORY)
            if all(count.knowledge == "exact" for count in counts) and item.total.knowledge != "exact":
                reject(EffectCode.CONTRADICTORY)
            if item.total.knowledge == "exact" and (
                any(count.knowledge != "exact" for count in counts)
                or item.total.value != sum(count.value for count in counts)
            ):
                reject(EffectCode.CONTRADICTORY)
        elif type(item) is UsageSummary:
            if item.input_units.knowledge != item.output_units.knowledge or any(
                count.knowledge == "known_upper"
                for count in (item.input_units, item.output_units, item.requests_with_unknown_usage)
            ):
                reject(EffectCode.CONTRADICTORY)
        elif type(item) is QualificationProxy:
            if item.assessed_targets != item.met + item.unmet + item.unknown:
                reject(EffectCode.CONTRADICTORY)


def _decimal(value: int) -> bytes:
    parts: list[int] = []
    while value >= 1_000_000_000:
        value, remainder = divmod(value, 1_000_000_000)
        parts.append(remainder)
    return (str(value) + "".join(f"{part:09d}" for part in reversed(parts))).encode("ascii")


def _chunks(value: object) -> Iterator[bytes]:
    import json

    members = _members(value)
    if members:
        yield b"{"
        for index, (key, child) in enumerate(sorted(members)):
            if index:
                yield b","
            yield json.dumps(key).encode("ascii") + b":"
            yield from _chunks(child)
        yield b"}"
    elif type(value) in (tuple, frozenset):
        assert isinstance(value, (tuple, frozenset))
        ordered = (
            tuple(reason for reason in get_args(ObservationReason) if reason in value)
            if type(value) is frozenset
            else value
        )
        if ordered and all(type(item) is ReasonCount for item in ordered):
            ordered = tuple(sorted(ordered, key=lambda item: get_args(ObservationReason).index(item.reason)))
        yield b"["
        for index, item in enumerate(ordered):
            if index:
                yield b","
            yield from _chunks(item)
        yield b"]"
    elif type(value) is int:
        yield _decimal(value)
    else:
        yield json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def encode_inspection(*, value: InspectionValue, limits: InspectionLimits) -> bytes:
    """Validate the closed schema and all bounds before allocating JSON output."""
    if type(value) not in _VALUES:
        reject(EffectCode.INVALID_TYPE)
    require_instance(limits, InspectionLimits)
    _bounds(value, limits)
    _member_types(value, type(value))
    _consistency(value)
    encoded = b"".join(_chunks(value))
    if len(encoded) > limits.max_encoded_bytes:
        reject(EffectCode.LIMIT_EXCEEDED)
    return encoded


def bind_inspection_source(
    *,
    prepared: PreparedPlan,
    context: AdmittedContextPlan | None = None,
    execution_plan: AdmittedExecutionPlan | None = None,
    binding: BindingResult | None = None,
    execution: ExecutionResult | None = None,
    qualification: QualificationResult | None = None,
    continuity: ContinuityCommit | None = None,
    observations: ObservationClose | None = None,
    observation_session: ObservationSession | None = None,
    limits: InspectionLimits,
) -> InspectionSource:
    """Bind exact semantic owners; absence never proves that work did not run."""
    require_instance(prepared, PreparedPlan)
    require_instance(limits, InspectionLimits)
    for item, kind in (
        (context, AdmittedContextPlan),
        (execution_plan, AdmittedExecutionPlan),
        (binding, BindingResult),
        (execution, ExecutionResult),
        (qualification, QualificationResult),
        (continuity, ContinuityCommit),
        (observations, (ObservationReceipt, ObservationFailure)),
        (observation_session, ObservationSession),
    ):
        if item is not None:
            require_instance(item, kind)
    if observations is not None:
        _bounds(observations, limits)
        _member_types(observations, type(observations))
    if context is not None and context.prepared is not prepared:
        reject(EffectCode.FOREIGN_OWNER)
    if binding is not None and (
        binding.context is None
        or context is None
        or context.bound_context is not binding.context
        or binding.receipt.data is not prepared.data
        or binding.receipt.workflow is not prepared.workflow
    ):
        reject(EffectCode.FOREIGN_OWNER)
    for child, parent, actual in (
        (execution_plan, context, None if execution_plan is None else execution_plan.context),
        (execution, execution_plan, None if execution is None else execution._execution),
        (qualification, execution, None if qualification is None else qualification._result),
    ):
        if child is not None and parent is not None and actual is not parent:
            reject(EffectCode.FOREIGN_OWNER)
    if continuity is not None and (
        continuity.execution is not execution or continuity.qualification is not qualification
    ):
        reject(EffectCode.FOREIGN_OWNER)
    owners = dict(
        data=prepared.data,
        workflow=prepared.workflow,
        prepared=prepared,
        context=context,
        execution_plan=execution_plan,
        binding=binding,
        execution=execution,
        qualification=qualification,
        continuity=continuity,
    )
    if isinstance(observations, ObservationReceipt):
        if observation_session is None or observation_session._closed is not observations:
            reject(EffectCode.FOREIGN_OWNER)
        if any(value is not owners.get(name) for name, value in observation_session._owner.items()):
            reject(EffectCode.FOREIGN_OWNER)
    if observations is not None:
        _duplicates(observations)
    if (
        execution_plan is not None
        and context is None
        or execution is not None
        and execution_plan is None
        or qualification is not None
        and execution is None
    ):
        reject(EffectCode.MISSING)
    if observations is not None:
        _consistency(observations)
    if execution is not None:
        if len(execution.artifacts) > limits.max_count:
            reject(EffectCode.LIMIT_EXCEEDED)
        for _, value in execution.artifacts:
            if not inspect_artifact(value):
                reject(EffectCode.INVALID_VALUE)
    return InspectionSource(
        _key=_OBSERVATION_KEY,
        limits=limits,
        prepared=prepared,
        context=context,
        execution_plan=execution_plan,
        binding=binding,
        execution=execution,
        qualification=qualification,
        continuity=continuity,
        observations=observations,
    )


def bind_binding_inspection_source(
    *, data: ValidatedDataGraph, workflow: Graph, binding: BindingResult, limits: InspectionLimits
) -> BindingInspectionSource:
    require_instance(data, ValidatedDataGraph)
    require_instance(workflow, Graph)
    require_instance(binding, BindingResult)
    require_instance(limits, InspectionLimits)
    if binding.receipt.data is not data or binding.receipt.workflow is not workflow:
        reject(EffectCode.FOREIGN_OWNER)
    return BindingInspectionSource(_key=_OBSERVATION_KEY, data=data, workflow=workflow, binding=binding, limits=limits)


def _shape(prepared: PreparedPlan) -> GraphShapeSummary:

    return GraphShapeSummary(
        target_count=len(prepared.target_occurrences),
        template_node_count=sum(len(workflow.operations) for workflow in prepared.workflow.bodies()),
        activation_slots_per_target=prepared.activation_slots_per_target,
        total_activation_slots=prepared.total_activation_slots,
        maximum_dynamic_depth=prepared.workflow.dynamic_depth,
    )


def _exact(value: int) -> CountEstimate:
    return CountEstimate(knowledge="exact", value=value)


def _unknown() -> CountEstimate:
    return CountEstimate(knowledge="unknown", value=None)


_Inspection = TypeVar("_Inspection", bound=InspectionValue)


def _validated(value: _Inspection, limits: InspectionLimits) -> _Inspection:
    _bounds(value, limits)
    _member_types(value, type(value))
    _consistency(value)
    return value


def explain(*, source: InspectionSource) -> ExplainSummary:
    require_instance(source, InspectionSource)
    ceiling = source.prepared.declared_request_upper_bound
    planned, conditional = (_exact(0), _exact(0)) if ceiling == 0 else (_unknown(), _unknown())
    if ceiling and source.execution_plan is not None:
        planned, conditional = _forecast_split(source.execution_plan)
    support: Literal["supported", "unsupported", "unknown"] = "unknown"
    if source.execution_plan is not None and source.context is not None:
        policies = [
            policy.request
            for policy in source.execution_plan.policies
            if policy.kind == "external"
            for item in policy.implementations
        ]
        policies.extend(item.request for item in source.context.context_capabilities)
        capability_control = all(
            item.capability.request_visibility == "dispatch_and_settlement"
            and item.capability.pre_dispatch_control == "executor"
            for policy in source.execution_plan.policies
            if policy.kind == "external"
            for item in policy.implementations
        )
        support = (
            "supported"
            if ceiling is not None
            and capability_control
            and all(
                policy is not None
                and policy.visibility == "dispatch_and_settlement"
                and policy.pre_dispatch_control == "executor"
                for policy in policies
            )
            else "unsupported"
        )
    return _validated(
        ExplainSummary(
            _key=_OBSERVATION_KEY,
            schema_version=1,
            shape=_shape(source.prepared),
            requests=RequestForecast(
                planned=planned,
                conditional=conditional,
                upper_bound=_unknown() if ceiling is None else CountEstimate(knowledge="known_upper", value=ceiling),
            ),
            hard_budget_support=support,
        ),
        source.limits,
    )


def _forecast_split(execution: AdmittedExecutionPlan) -> tuple[CountEstimate, CountEstimate]:
    from anonymizer.graph.workflow import SubgraphNode

    prepared = execution.context.prepared
    workflows = prepared.workflow.bodies()
    # Only a static, unconditional, unshared grammar establishes a physical
    # planned count. Dynamic scheduling or batching leaves the split unknown.
    if (
        prepared.workflow.dynamic_depth
        or execution.context.adaptive_retrievals
        or execution.decisions
        or any(policy.initial_request_purpose == "repair" for policy in execution.policies)
        or any(workflow.choices for workflow in workflows)
        or any(isinstance(node, SubgraphNode) for workflow in workflows for node in workflow.operations)
        or any(
            item.capability.attribution == "keyed_shared_request"
            for policy in execution.policies
            for item in policy.implementations
        )
    ):
        return _unknown(), _unknown()
    external = {policy.node for policy in execution.policies if policy.kind == "external"}
    # A preceding operation's non-success may suppress a dependent activation;
    # count roots only when the whole static graph has no dependency edges.
    if any(workflow.sequence for workflow in workflows):
        return _unknown(), _unknown()
    planned = sum(slot.template in external for slot in prepared.reservation_recipe) * len(prepared.target_occurrences)
    if prepared.declared_request_upper_bound is None or planned > prepared.declared_request_upper_bound:
        return _unknown(), _unknown()
    return _exact(planned), CountEstimate(
        knowledge="known_upper", value=prepared.declared_request_upper_bound - planned
    )


_PURPOSES = ("initial", "initial_binding", "adaptive_retrieval", "retry", "correction", "repair", "failover")


def _account(
    binding: RequestReceipt | None, execution: RequestReceipt | None, *, binding_only: bool = False
) -> tuple[ActualRequestCounts, UsageSummary]:

    scopes = [(binding, frozenset({"initial_binding", "retry", "correction"}))]
    if not binding_only:
        scopes.append((execution, frozenset(_PURPOSES) - {"initial_binding"}))
    return _account_receipts(scopes)


def _account_receipts(
    scopes: Sequence[tuple[RequestReceipt | None, frozenset[str]]],
) -> tuple[ActualRequestCounts, UsageSummary]:
    from anonymizer.engine.graph_sdk.requests import ExactUsage

    counts = dict.fromkeys(_PURPOSES, 0)
    missing: set[str] = set()
    seen = set()
    units_in = units_out = unknown = 0
    for receipt, purposes in scopes:
        if receipt is None:
            missing.update(purposes)
            continue
        for dispatch in receipt.dispatches:
            if dispatch.request in seen:
                reject(EffectCode.DUPLICATE)
            seen.add(dispatch.request)
            counts[dispatch.purpose] += 1
            settlement = next((item for item in receipt.settlements if item.request == dispatch.request), None)
            if settlement is None or type(settlement.usage) is not ExactUsage:
                unknown += 1
            else:
                units_in += settlement.usage.input_units
                units_out += settlement.usage.output_units
    actual = ActualRequestCounts(
        total=_unknown() if missing else _exact(len(seen)),
        **{purpose: _unknown() if purpose in missing else _exact(counts[purpose]) for purpose in _PURPOSES},
    )
    return actual, UsageSummary(
        input_units=_unknown() if missing or unknown else _exact(units_in),
        output_units=_unknown() if missing or unknown else _exact(units_out),
        requests_with_unknown_usage=_unknown() if missing else _exact(unknown),
    )


def _resources(
    receipts: tuple[RequestReceipt, ...], cleanup: tuple[CleanupFact, ...], artifact_count: int, artifact_bytes: int
) -> ResourceCounts:
    return ResourceCounts(
        local_in_flight=sum(len(receipt.local_in_flight) for receipt in receipts),
        remote_outstanding=sum(len(receipt.remote_outstanding) for receipt in receipts),
        artifact_count=artifact_count,
        artifact_bytes=artifact_bytes,
        cleanup_closed=sum(fact.disposition == "closed" for fact in cleanup),
        cleanup_failed=sum(fact.disposition == "close_failed" for fact in cleanup),
        cleanup_unknown=sum(fact.disposition == "close_unknown" for fact in cleanup),
        caller_left_open=sum(fact.owner == "caller" and fact.disposition == "left_open" for fact in cleanup),
    )


def _artifact_bytes(value: ArtifactValue) -> int:
    if not inspect_artifact(value):
        reject(EffectCode.INVALID_VALUE)
    return value.shape.content_bytes


def inspect(*, source: InspectionSource) -> InspectionSummary:
    require_instance(source, InspectionSource)
    binding = None if source.binding is None else source.binding.receipt
    execution = source.execution
    counts, usage = _account(
        None if binding is None else binding.requests,
        None if execution is None or source.prepared.declared_request_upper_bound is None else execution.requests,
    )
    routes = terminals = resources = proxy = None
    if execution is not None:
        if len(execution.artifacts) > source.limits.max_count:
            reject(EffectCode.LIMIT_EXCEEDED)
        terminals = TerminalCounts(
            **{
                name: sum(fact.category == name for fact in execution.record.terminals)
                for name in ("success", "failure", "cancelled", "lost", "blocked", "inconsistent")
            }
        )
        kinds = dict.fromkeys(("local", "external", "decision", "context"), 0)
        policies = {policy.node: policy.kind for policy in execution._execution.policies}
        adaptive = {item.node for item in execution._execution.context.adaptive_retrievals}
        states = execution.states + (() if execution.invocation_state is None else (execution.invocation_state,))
        entries = {entry.activation: entry for state in states for entry in state.entries}
        for terminal in execution.record.terminals:
            if terminal.attempt is not None:
                entry = entries[terminal.activation]
                kind = "context" if entry.template in adaptive else policies[entry.template]
                kinds["external" if kind == "delegated" else kind] += 1
        routes = RouteCounts(**kinds)
        receipts = (execution.requests,) if binding is None else (binding.requests, execution.requests)
        cleanup = execution.cleanup if binding is None else binding.cleanup + execution.cleanup
        # Execution materialization includes its bound context. Count retained
        # execution artifact identities once, rather than adding binding copies.
        resources = _resources(
            receipts, cleanup, len(execution.artifacts), sum(_artifact_bytes(value) for _, value in execution.artifacts)
        )
    if source.qualification is not None:
        qualification = source.qualification
        statuses = [status.qualification for status in qualification.record.statuses]
        proxy = QualificationProxy(
            assessed_targets=sum(status != "not_assessed" for status in statuses),
            met=statuses.count("met"),
            unmet=statuses.count("unmet"),
            unknown=statuses.count("unknown"),
            verified_evidence=len(qualification.verified),
            withheld_targets=sum(bool(item.withholding) for item in qualification.targets),
        )
    delivery = (
        source.observations.delivery
        if isinstance(source.observations, ObservationReceipt)
        else "failed"
        if isinstance(source.observations, ObservationFailure)
        else "not_configured"
    )
    return _validated(
        InspectionSummary(
            _key=_OBSERVATION_KEY,
            schema_version=1,
            shape=_shape(source.prepared),
            requests=counts,
            usage=usage,
            routes=routes,
            terminals=terminals,
            resources=resources,
            qualification=proxy,
            observation_delivery=delivery,
        ),
        source.limits,
    )


def inspect_binding(*, source: BindingInspectionSource) -> BindingInspectionSummary:
    require_instance(source, BindingInspectionSource)
    receipt = source.binding.receipt
    counts, usage = _account(receipt.requests, None, binding_only=True)
    resources = _resources(
        (receipt.requests,),
        receipt.cleanup,
        len(receipt.artifacts),
        sum(len(item.text.encode("utf-8")) for item in receipt.artifacts),
    )
    return _validated(
        BindingInspectionSummary(
            _key=_OBSERVATION_KEY,
            schema_version=1,
            terminal=receipt.terminal,
            requests=counts,
            usage=usage,
            resources=resources,
        ),
        source.limits,
    )


def _diagnosis(
    *,
    reasons: dict[str, int],
    terminal_unknown: bool,
    usage_unknown: bool,
    remote_work_unknown: bool,
    observation_failed: bool,
    limits: InspectionLimits,
) -> DiagnosisSummary:
    return _validated(
        DiagnosisSummary(
            _key=_OBSERVATION_KEY,
            schema_version=1,
            reasons=tuple(
                ReasonCount(reason=reason, count=reasons[reason])
                for reason in get_args(ObservationReason)
                if reasons.get(reason)
            ),
            terminal_unknown=terminal_unknown,
            usage_unknown=usage_unknown,
            remote_work_unknown=remote_work_unknown,
            observation_failed=observation_failed,
        ),
        limits,
    )


def diagnose_observations(*, observations: ObservationClose, limits: InspectionLimits) -> DiagnosisSummary:
    if type(observations) not in (ObservationReceipt, ObservationFailure):
        reject(EffectCode.INVALID_TYPE)
    _validated(observations, limits)
    reasons: dict[str, int] = {}
    failed = isinstance(observations, ObservationFailure) or observations.delivery == "failed"
    if isinstance(observations, ObservationReceipt):
        for pair in observations.pairs:
            for reason in pair.terminal.reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
    if failed:
        reasons["observer_failed"] = reasons.get("observer_failed", 0) + 1
    return _diagnosis(
        reasons=reasons,
        terminal_unknown=True,
        usage_unknown=True,
        remote_work_unknown=True,
        observation_failed=failed,
        limits=limits,
    )


def diagnose(*, source: InspectionSource) -> DiagnosisSummary:
    view = inspect(source=source)
    reasons: dict[str, int] = {}
    for receipt in (
        None if source.binding is None else source.binding.receipt.requests,
        None if source.execution is None else source.execution.requests,
    ):
        if receipt is not None:
            for terminal in receipt.terminals:
                category = terminal.category
                reason = (
                    "cancelled"
                    if category == "cancelled"
                    else "lost"
                    if category == "lost"
                    else "provider_failure"
                    if category == "failure"
                    else None
                )
                if reason is not None:
                    reasons[reason] = reasons.get(reason, 0) + 1
    if source.execution is not None:
        reason_map = {
            "execution_failed": "internal_failure",
            "cancel_requested": "cancelled",
            "transport_lost": "lost",
            "missing": "missing_evidence",
            "duplicate": "invalid_evidence",
            "foreign": "invalid_evidence",
            "stale": "invalid_evidence",
            "contradictory": "invalid_evidence",
            "prerequisite": "unavailable",
            "omitted_optional": "unavailable",
            "artifact_limit_exhausted": "limit_exceeded",
        }
        for terminal in source.execution.record.terminals:
            for code in terminal.reasons:
                if code == "unused_retry_capacity":
                    continue  # Successful bounded capacity is not a failure diagnosis.
                reason = reason_map[code]
                reasons[reason] = reasons.get(reason, 0) + 1
    for receipt in (
        None if source.binding is None else source.binding.receipt.requests,
        None if source.execution is None else source.execution.requests,
    ):
        if receipt is not None:
            for denial in receipt.denials:
                reason = "budget_stopped" if denial.category == "budget_stopped" else "request_limit_stopped"
                reasons[reason] = reasons.get(reason, 0) + 1
    if source.qualification is not None:
        reasons["missing_evidence"] = sum(
            bool(item.withholding & {"missing_assessment", "missing_candidate"})
            for item in source.qualification.targets
        )
        reasons["invalid_evidence"] = sum(
            bool(item.withholding & {"stale_evidence", "incomplete_coverage", "assessment_unknown"})
            for item in source.qualification.targets
        )
    if source.continuity is not None:
        reasons["continuity_conflict"] = len(source.continuity.conflicts)
    if view.resources is not None:
        reasons["cleanup_failed"] = view.resources.cleanup_failed
        reasons["cleanup_unknown"] = view.resources.cleanup_unknown
    failed = view.observation_delivery == "failed"
    if failed:
        reasons["observer_failed"] = 1
    return _diagnosis(
        reasons=reasons,
        terminal_unknown=source.execution is None,
        usage_unknown=view.usage.input_units.knowledge == "unknown",
        remote_work_unknown=source.execution is None
        or source.binding is None
        or bool(view.resources and view.resources.remote_outstanding),
        observation_failed=failed,
        limits=source.limits,
    )


def diagnose_binding(*, source: BindingInspectionSource) -> DiagnosisSummary:
    view = inspect_binding(source=source)
    reasons = {"cleanup_failed": view.resources.cleanup_failed, "cleanup_unknown": view.resources.cleanup_unknown}
    if view.terminal in ("failed", "cancelled", "lost"):
        reasons[{"failed": "provider_failure", "cancelled": "cancelled", "lost": "lost"}[view.terminal]] = 1
    return _diagnosis(
        reasons=reasons,
        terminal_unknown=False,
        usage_unknown=view.usage.input_units.knowledge == "unknown",
        remote_work_unknown=bool(view.resources.remote_outstanding),
        observation_failed=False,
        limits=source.limits,
    )
