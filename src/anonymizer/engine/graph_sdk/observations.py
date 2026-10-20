# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded content-free lifecycle values and process-local sessions."""

from __future__ import annotations

from dataclasses import InitVar, dataclass, fields
from typing import TYPE_CHECKING, Literal, Protocol, get_args

from anonymizer.engine.graph_sdk._effect_values import EffectCode, PrivateValue, reject, require_count, require_owner
from anonymizer.engine.graph_sdk._ingress_values import IngressLimits

if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk._execution_values import AdmittedExecutionPlan, MonotonicClock
    from anonymizer.engine.graph_sdk._workflow_values import Graph
    from anonymizer.engine.graph_sdk.context import BindingLimits, ContextSourceCapability, InitialContextDecl
    from anonymizer.engine.graph_sdk.data import ValidatedDataGraph
    from anonymizer.engine.graph_sdk.evidence import AdmittedQualification
    from anonymizer.engine.graph_sdk.preparation import PreparedPlan

_OBSERVATION_KEY = object()


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class _FactoryValue(PrivateValue):
    """Owner-keyed base: subclasses inherit the key and this check; none defines its own __post_init__."""

    _key: InitVar[object]

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _OBSERVATION_KEY)


ObservationBoundary = Literal[
    "preparation",
    "context_admission",
    "execution_admission",
    "initial_binding",
    "execution",
    "activation",
    "operation",
    "adaptive_retrieval",
    "physical_request",
    "decision_wait",
    "request_reconciliation",
    "cleanup",
    "qualification",
    "continuity_commit",
    "projection",
    "release",
]

ObservationOutcome = Literal["success", "failure", "cancelled", "lost", "blocked", "inconsistent", "partial", "unknown"]

ObservationReason = Literal[
    "invalid_input",
    "limit_exceeded",
    "unavailable",
    "provider_failure",
    "malformed_response",
    "cancelled",
    "lost",
    "deadline",
    "budget_stopped",
    "request_limit_stopped",
    "missing_evidence",
    "invalid_evidence",
    "continuity_conflict",
    "cleanup_failed",
    "cleanup_unknown",
    "observer_failed",
    "internal_failure",
]

RouteKind = Literal["none", "local", "external", "context"]

OperationRole = Literal["ordinary", "assessment", "repair", "context"]

ObservationDelivery = Literal["not_configured", "complete", "failed"]

ObservationFailureCode = Literal["capacity_contract", "clock_contract", "producer_contract"]


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObservationLimits(PrivateValue):
    max_pairs: int
    max_events: int
    max_nodes: int
    max_depth: int
    max_encoded_bytes: int
    max_duration_ns: int
    max_count: int

    def __post_init__(self) -> None:
        for field in fields(self):
            require_count(getattr(self, field.name))


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class CorrelationRef(PrivateValue):
    ordinal: int
    parent_ordinal: int | None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class WorkMeasurement(PrivateValue):
    role: OperationRole
    route: RouteKind
    input_count: int
    output_count: int
    artifact_count: int
    artifact_bytes: int
    request_count: int


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObservationStart(PrivateValue):
    correlation: CorrelationRef
    boundary: ObservationBoundary


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObservationTerminal(PrivateValue):
    correlation: CorrelationRef
    boundary: ObservationBoundary
    outcome: ObservationOutcome
    reasons: frozenset[ObservationReason]
    duration_ns: int
    measurement: WorkMeasurement | None


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class LifecyclePair(PrivateValue):
    start: ObservationStart
    terminal: ObservationTerminal


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObservationReceipt(_FactoryValue):
    schema_version: Literal[1]
    pairs: tuple[LifecyclePair, ...]
    delivery: ObservationDelivery
    delivered_pairs: int

    def __post_init__(self, _key: object) -> None:
        require_owner(_key, _OBSERVATION_KEY)
        ordinals = [pair.start.correlation.ordinal for pair in self.pairs]
        if len(set(ordinals)) != len(ordinals):
            reject(EffectCode.DUPLICATE)
        if (
            self.delivered_pairs > len(self.pairs)
            or self.delivery == "complete"
            and self.delivered_pairs != len(self.pairs)
            or self.delivery == "not_configured"
            and self.delivered_pairs != 0
        ):
            reject(EffectCode.CONTRADICTORY)
        for index, pair in enumerate(self.pairs):
            correlation = pair.start.correlation
            if (
                correlation.ordinal != index
                or correlation.parent_ordinal is not None
                and not 0 <= correlation.parent_ordinal < index
                or pair.start.correlation != pair.terminal.correlation
                or pair.start.boundary != pair.terminal.boundary
            ):
                reject(EffectCode.CONTRADICTORY)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObservationFailure(PrivateValue):
    schema_version: Literal[1]
    code: ObservationFailureCode

    def __post_init__(self) -> None:
        # Caller-constructible: the only inspection value that is ingress, so it is checked here, once.
        if type(self.schema_version) is not int or type(self.code) is not str:
            reject(EffectCode.INVALID_TYPE)
        if self.code not in get_args(ObservationFailureCode):
            reject(EffectCode.INVALID_VALUE)
        if self.schema_version != 1:
            reject(EffectCode.UNSUPPORTED)


@dataclass(frozen=True, slots=True, kw_only=True, repr=False)
class ObservationRequirements(_FactoryValue):
    max_pairs_required: int
    max_events_required: int
    max_nodes_required: int
    max_depth_required: int
    max_encoded_bytes_required: int
    _limits: ObservationLimits


ObservationClose = ObservationReceipt | ObservationFailure


class NodeMeasurement(Protocol):
    def terminal_measurement(self) -> WorkMeasurement: ...


class ObservationSink(Protocol):
    def observe(self, pair: LifecyclePair) -> None: ...


# Only SDK owners import this token; no public pair-producing API is exposed.
_PRODUCER = object()


def _width(value: int) -> int:
    """Decimal width without Python's process-wide integer-string setting."""
    if value == 0:
        return 1
    estimate = (value.bit_length() * 30103) // 100000
    return estimate + (value >= 10**estimate)


def _requirements(pairs: int, depth: int, limits: ObservationLimits) -> ObservationRequirements:
    """Forecast the closed worst-case receipt without allocating its pairs."""
    import json

    require_count(pairs)
    require_count(depth)
    count_width = _width(limits.max_count)
    duration_width = _width(limits.max_duration_ns)
    correlation = {"ordinal": 0, "parent_ordinal": None}
    measurement = {
        "role": "assessment",
        "route": "external",
        "input_count": 0,
        "output_count": 0,
        "artifact_count": 0,
        "artifact_bytes": 0,
        "request_count": 0,
    }
    pair = {
        "start": {"correlation": correlation, "boundary": "request_reconciliation"},
        "terminal": {
            "correlation": correlation,
            "boundary": "request_reconciliation",
            "outcome": "inconsistent",
            "reasons": list(get_args(ObservationReason)),
            "duration_ns": 0,
            "measurement": measurement,
        },
    }

    def shape(value: object) -> tuple[int, int]:
        children = list(value.values()) if isinstance(value, dict) else value if isinstance(value, list) else []
        sizes = [shape(child) for child in children]
        return 1 + sum(size[0] for size in sizes), 1 + max((size[1] for size in sizes), default=0)

    nodes, json_depth = shape(pair)
    pair_bytes = len(json.dumps(pair, separators=(",", ":"), sort_keys=True))
    pair_bytes += 7 * (count_width - 1) + 2 * (max(4, count_width) - 4) + duration_width - 1
    # The envelope includes schema, pair array, delivery, and delivered count.
    empty_bytes = len(b'{"delivered_pairs":0,"delivery":"not_configured","pairs":[],"schema_version":1}')
    return ObservationRequirements(
        _key=_OBSERVATION_KEY,
        max_pairs_required=pairs,
        max_events_required=2 * pairs,
        max_nodes_required=5 + pairs * nodes,
        max_depth_required=max(depth, 2 + json_depth if pairs else 2),
        max_encoded_bytes_required=empty_bytes + count_width - 1 + pairs * pair_bytes + max(0, pairs - 1),
        _limits=limits,
    )


def binding_observation_requirements(
    *,
    data: ValidatedDataGraph,
    workflow: Graph,
    declarations: tuple[InitialContextDecl, ...],
    capabilities: tuple[ContextSourceCapability, ...],
    binding_limits: BindingLimits,
    limits: ObservationLimits,
) -> ObservationRequirements:
    # The binding owner's admitted cumulative request/resource bounds dominate
    # retries and cleanup even when acquisition or a required source fails.
    del data, workflow, declarations, capabilities
    return _requirements(1 + 2 * binding_limits.max_requests + binding_limits.max_resources, 3, limits)


def prepared_observation_requirements(*, prepared: PreparedPlan, limits: ObservationLimits) -> ObservationRequirements:
    from anonymizer.engine.graph_sdk.preparation import _recipe_depth

    return _requirements(
        1
        + 4 * prepared.total_activation_slots
        + 2 * (prepared.declared_request_upper_bound or 0)
        + 2 * prepared.limits.max_capabilities,
        4 + 2 * _recipe_depth(prepared.reservation_recipe),
        limits,
    )


def execution_observation_requirements(
    *, execution: AdmittedExecutionPlan, limits: ObservationLimits
) -> ObservationRequirements:
    from anonymizer.engine.graph_sdk.preparation import _recipe_depth

    prepared = execution.context.prepared
    # Every selected resource may need cleanup.
    resources = sum(len(policy.implementations) for policy in execution.policies)
    return _requirements(
        1 + 4 * prepared.total_activation_slots + 2 * (prepared.declared_request_upper_bound or 0) + resources,
        4 + 2 * _recipe_depth(prepared.reservation_recipe),
        limits,
    )


def post_execution_observation_requirements(
    *,
    qualification: AdmittedQualification | None,
    projection_calls: int,
    limits: ObservationLimits,
) -> ObservationRequirements:
    require_count(projection_calls)
    return _requirements(int(qualification is not None) + 2 * projection_calls, 2, limits)


from contextvars import ContextVar
from threading import RLock

_CURRENT: ContextVar[tuple[ObservationSession, int] | None] = ContextVar("graph_observation", default=None)


class ObservationSession(PrivateValue):
    """One process-local owner chain. Only complete immutable pairs escape."""

    __slots__ = (
        "limits",
        "_clock",
        "_last_tick",
        "_sink",
        "_lock",
        "_owner",
        "_closed",
        "_failure",
        "_reserved",
        "_phase_end",
        "_phase_depth",
        "_nodes",
        "_bytes",
        "_next",
        "_open",
        "_pairs",
        "_depths",
        "_delivery",
        "_delivered",
        "_delivering",
    )

    def __init__(
        self, *, _key: object, limits: ObservationLimits, clock: MonotonicClock, sink: ObservationSink | None
    ) -> None:
        if _key is not _OBSERVATION_KEY:
            raise TypeError("observation sessions require begin_observations")
        self.limits, self._clock, self._sink = limits, clock, sink
        self._last_tick: int | None = None
        self._lock = RLock()
        self._owner: dict[str, object] = {}
        self._closed: ObservationClose | None = None
        self._failure: ObservationFailure | None = None
        self._phase_end = self._phase_depth = 0
        self._reserved = self._nodes = self._bytes = self._next = self._delivered = 0
        self._open: dict[int, tuple[ObservationStart, int]] = {}
        self._pairs: dict[int, LifecyclePair] = {}
        self._depths: dict[int, int] = {}
        self._delivery: ObservationDelivery = "not_configured" if sink is None else "complete"
        self._delivering = False

    def _disable(self, code: ObservationFailureCode) -> ObservationFailure:
        if self._failure is None:
            self._failure = ObservationFailure(schema_version=1, code=code)
        return self._failure

    def _bind(self, key: object, *, result: bool = False, **owners: object) -> None:
        with self._lock:
            if key is not _PRODUCER or self._closed is not None:
                self._disable("producer_contract")
                return
            for name, value in owners.items():
                prior = self._owner.get(name)
                if prior is not None and (prior is not value or result and name in {"execution", "binding"}):
                    self._disable("producer_contract")
                    return
            self._owner.update(owners)

    def _tick(self) -> int | None:
        try:
            tick = self._clock.now_ns()
        except BaseException:
            self._disable("clock_contract")
            return None
        if type(tick) is not int or tick < 0 or self._last_tick is not None and tick < self._last_tick:
            self._disable("clock_contract")
            return None
        self._last_tick = tick
        return tick

    def _enter(self, key: object, boundary: ObservationBoundary, parent: int | None) -> int | None:
        with self._lock:
            if self._failure is not None:
                return None
            if key is not _PRODUCER or self._closed is not None or self._delivering:
                self._disable("producer_contract")
                return None
            if self._next >= min(self._reserved, self._phase_end):
                self._disable("capacity_contract")
                return None
            if parent is not None and (type(parent) is not int or parent not in self._depths):
                self._disable("producer_contract")
                return None
            depth = 1 if parent is None else self._depths[parent] + 1
            if depth > min(self.limits.max_depth, self._phase_depth):
                self._disable("capacity_contract")
                return None
            tick = self._tick()
            if tick is None:
                return None
            ordinal = self._next
            self._next += 1
            self._depths[ordinal] = depth
            self._open[ordinal] = (
                ObservationStart(correlation=CorrelationRef(ordinal=ordinal, parent_ordinal=parent), boundary=boundary),
                tick,
            )
            return ordinal

    def _exit(
        self,
        key: object,
        ordinal: int | None,
        outcome: ObservationOutcome,
        reasons: frozenset[ObservationReason],
        measurement: WorkMeasurement | None,
    ) -> None:
        with self._lock:
            if ordinal is None or self._failure is not None:
                return
            if key is not _PRODUCER or ordinal not in self._open or self._closed is not None:
                self._disable("producer_contract")
                return
            start, tick = self._open.pop(ordinal)
            end = self._tick()
            if end is None:
                return
            if end - tick > self.limits.max_duration_ns:
                self._disable("capacity_contract")
                return
            if outcome not in get_args(ObservationOutcome) or not reasons <= frozenset(get_args(ObservationReason)):
                self._disable("producer_contract")
                return
            if measurement is not None and not _valid_measurement(measurement, self.limits.max_count):
                self._disable("capacity_contract")
                return
            terminal = ObservationTerminal(
                correlation=start.correlation,
                boundary=start.boundary,
                outcome=outcome,
                reasons=reasons,
                duration_ns=end - tick,
                measurement=measurement,
            )
            pair = LifecyclePair(start=start, terminal=terminal)
            self._pairs[ordinal] = pair
            if self._sink is not None and self._delivery == "complete":
                self._delivering = True
                try:
                    self._sink.observe(pair)
                except BaseException:
                    self._delivery = "failed"
                else:
                    self._delivered += 1
                finally:
                    self._delivering = False

    def close(self) -> ObservationClose:
        with self._lock:
            if self._closed is None:
                if self._open and self._failure is None:
                    self._disable("producer_contract")
                from anonymizer.engine.graph_sdk._effect_values import EffectRejected

                try:
                    self._closed = self._failure or ObservationReceipt(
                        _key=_OBSERVATION_KEY,
                        schema_version=1,
                        pairs=tuple(self._pairs[index] for index in sorted(self._pairs)),
                        delivery=self._delivery,
                        delivered_pairs=self._delivered,
                    )
                except EffectRejected:
                    # Receipt consistency is checked at construction; a contradictory one is a producer defect.
                    self._closed = self._disable("producer_contract")
                if isinstance(self._closed, ObservationReceipt):
                    from anonymizer.engine.graph_sdk.inspection import InspectionLimits, _validated

                    bounds = InspectionLimits(
                        max_nodes=self.limits.max_nodes,
                        max_depth=self.limits.max_depth,
                        max_pairs=self.limits.max_pairs,
                        max_events=self.limits.max_events,
                        max_reason_counts=len(get_args(ObservationReason)),
                        max_encoded_bytes=self.limits.max_encoded_bytes,
                        max_count=max(self.limits.max_count, self.limits.max_duration_ns),
                    )
                    try:
                        _validated(self._closed, bounds)
                    except EffectRejected as error:
                        self._closed = self._disable(
                            "capacity_contract" if error.code == EffectCode.LIMIT_EXCEEDED else "producer_contract"
                        )
            return self._closed


def _valid_measurement(value: object, maximum: int) -> bool:
    return (
        type(value) is WorkMeasurement
        and type(value.role) is str
        and type(value.route) is str
        and value.role in get_args(OperationRole)
        and value.route in get_args(RouteKind)
        and all(
            type(getattr(value, name)) is int and 0 <= getattr(value, name) <= maximum
            for name in ("input_count", "output_count", "artifact_count", "artifact_bytes", "request_count")
        )
    )


def begin_observations(
    *, limits: ObservationLimits, clock: MonotonicClock, sink: ObservationSink | None = None
) -> ObservationSession | ObservationFailure:
    from anonymizer.engine.graph_sdk._effect_values import require_instance

    require_instance(limits, ObservationLimits)
    session = ObservationSession(_key=_OBSERVATION_KEY, limits=limits, clock=clock, sink=sink)
    failure = reserve_observation_scope(session=session, requirements=_requirements(1, 1, limits))
    if failure is not None:
        return failure
    # Admission proves a minimum whole pair; phase reservations fund actual work.
    session._reserved = session._nodes = session._bytes = 0
    return session


def reserve_observation_scope(
    *, session: ObservationSession, requirements: ObservationRequirements
) -> ObservationFailure | None:
    from anonymizer.engine.graph_sdk._effect_values import require_instance

    require_instance(session, ObservationSession)
    require_instance(requirements, ObservationRequirements)
    with session._lock:
        if session._closed is not None:
            return session._disable("producer_contract")
        if session._failure is not None:
            return session._failure
        pairs = session._reserved + requirements.max_pairs_required
        nodes = session._nodes + requirements.max_nodes_required
        size = session._bytes + requirements.max_encoded_bytes_required
        limits = session.limits
        if (
            requirements._limits != limits
            or requirements.max_events_required != 2 * requirements.max_pairs_required
            or pairs > min(limits.max_pairs, limits.max_count)
            or 2 * pairs > limits.max_events
            or nodes > limits.max_nodes
            or requirements.max_depth_required > limits.max_depth
            or size > limits.max_encoded_bytes
        ):
            return session._disable("capacity_contract")
        session._reserved, session._nodes, session._bytes = pairs, nodes, size
        session._phase_end = session._next + requirements.max_pairs_required
        session._phase_depth = requirements.max_depth_required
    return None


class _Span:
    """Package-private complete-pair guard used at actual owner boundaries."""

    def __init__(
        self, boundary: ObservationBoundary, *, session: ObservationSession | None = None, parent: int | None = None
    ) -> None:
        current = _CURRENT.get()
        self.session = session if session is not None else current[0] if current else None
        self.parent = parent if parent is not None else current[1] if current and current[0] is self.session else None
        self.ordinal = None if self.session is None else self.session._enter(_PRODUCER, boundary, self.parent)
        self.outcome: ObservationOutcome = "success"
        self.reasons: frozenset[ObservationReason] = frozenset()
        self.measurement: WorkMeasurement | None = None
        self._done = False

    def __enter__(self) -> _Span:
        self._token = _CURRENT.set(
            (self.session, self.ordinal) if self.session is not None and self.ordinal is not None else None
        )
        return self

    def __exit__(self, kind: object, exception: object, traceback: object) -> None:
        del traceback
        if kind is not None:
            self.outcome, self.reasons = _exception_terminal(exception)
        try:
            self.finish()
        finally:
            _CURRENT.reset(self._token)

    def finish(self) -> None:
        if not self._done and self.session is not None:
            self._done = True
            self.session._exit(_PRODUCER, self.ordinal, self.outcome, self.reasons, self.measurement)


def _exception_terminal(exception: object) -> tuple[ObservationOutcome, frozenset[ObservationReason]]:
    import asyncio

    from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected

    if isinstance(exception, asyncio.CancelledError):
        return "cancelled", frozenset({"cancelled"})
    from anonymizer.engine.graph_sdk._graph_values import ContractViolation
    from anonymizer.engine.graph_sdk.capabilities import PreparationRejected

    if isinstance(exception, (EffectRejected, PreparationRejected, ContractViolation)):
        reason: ObservationReason = (
            "limit_exceeded" if exception.code.value == EffectCode.LIMIT_EXCEEDED.value else "invalid_input"
        )
        return "failure", frozenset({reason})
    return "failure", frozenset({"internal_failure"})


def observed_prepare(
    *,
    observations: ObservationSession,
    data: ValidatedDataGraph,
    workflow: Graph,
    activation_limits: ActivationLimits,
    bound_inputs: tuple[BoundInput, ...],
    configuration: PreparationConfiguration,
    selections: tuple[ImplementationSelection, ...],
    capabilities: tuple[ImplementationCapability, ...],
    limits: PreparationLimits,
) -> PreparedPlan:
    from anonymizer.engine.graph_sdk.preparation import prepare

    reserve_observation_scope(session=observations, requirements=_requirements(1, 2, observations.limits))
    observations._bind(_PRODUCER, data=data, workflow=workflow)
    with _Span("preparation", session=observations):
        result = prepare(
            data=data,
            workflow=workflow,
            activation_limits=activation_limits,
            bound_inputs=bound_inputs,
            configuration=configuration,
            selections=selections,
            capabilities=capabilities,
            limits=limits,
        )
        observations._bind(_PRODUCER, prepared=result)
        return result


def observed_context_admission(
    *,
    observations: ObservationSession,
    prepared: PreparedPlan,
    bound_context: BoundContext | None,
) -> AdmittedContextPlan:
    from anonymizer.engine.graph_sdk.context import admit_context_plan

    reserve_observation_scope(session=observations, requirements=_requirements(1, 2, observations.limits))
    from anonymizer.engine.graph_sdk.preparation import PreparedPlan

    if isinstance(prepared, PreparedPlan):
        observations._bind(_PRODUCER, prepared=prepared, data=prepared.data, workflow=prepared.workflow)
    from anonymizer.engine.graph_sdk.context import BindingResult

    binding = observations._owner.get("binding")
    if isinstance(binding, BindingResult) and binding.context is not bound_context:
        observations._disable("producer_contract")
    with _Span("context_admission", session=observations):
        result = admit_context_plan(prepared=prepared, bound_context=bound_context)
        observations._bind(_PRODUCER, context=result)
        return result


def observed_execution_admission(
    *,
    observations: ObservationSession,
    context: AdmittedContextPlan,
    capabilities: tuple[ImplementationCapability, ...],
    policies: tuple[OperationExecutionPolicy, ...],
    assessment_productions: tuple[EvidenceProductionDecl, ...],
    assessment_limits: AssessmentLimits,
    map_expansions: tuple[MapExpansionDecl, ...] = (),
    ingress_limits: IngressLimits = IngressLimits(),
) -> AdmittedExecutionPlan:
    from anonymizer.engine.graph_sdk._execution_admission import admit_execution_plan

    reserve_observation_scope(session=observations, requirements=_requirements(1, 2, observations.limits))
    from anonymizer.engine.graph_sdk.context import AdmittedContextPlan

    if isinstance(context, AdmittedContextPlan):
        observations._bind(
            _PRODUCER,
            context=context,
            prepared=context.prepared,
            data=context.prepared.data,
            workflow=context.prepared.workflow,
        )
    with _Span("execution_admission", session=observations):
        result = admit_execution_plan(
            context=context,
            capabilities=capabilities,
            policies=policies,
            assessment_productions=assessment_productions,
            assessment_limits=assessment_limits,
            map_expansions=map_expansions,
            ingress_limits=ingress_limits,
        )
        observations._bind(_PRODUCER, execution_plan=result)
        return result


def observed_qualify_retained(
    *,
    observations: ObservationSession,
    admitted: AdmittedQualification,
    result: ExecutionResult,
    current: EvidenceRevisionView,
) -> QualificationResult:
    """Observe qualification of all retained assessments without executing callbacks."""
    return observed_qualify(
        observations=observations, admitted=admitted, result=result, current=current, submissions=None
    )


def observed_qualify(
    *,
    observations: ObservationSession,
    admitted: AdmittedQualification,
    result: ExecutionResult,
    current: EvidenceRevisionView,
    submissions: tuple[AssessmentSubmission, ...] | None,
) -> QualificationResult:
    from anonymizer.engine.graph_sdk.qualification import qualify

    reserve_observation_scope(session=observations, requirements=_requirements(1, 2, observations.limits))
    from anonymizer.engine.graph_sdk.evidence import AdmittedQualification

    if isinstance(admitted, AdmittedQualification):
        observations._bind(_PRODUCER, execution=result, execution_plan=admitted.execution)
    with _Span("qualification", session=observations) as span:
        qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        observations._bind(_PRODUCER, qualification=qualified)
        if any(item.withholding for item in qualified.targets):
            span.outcome = "partial" if qualified.qualified else "blocked"
            span.reasons = frozenset({"missing_evidence"})
        return qualified


def observed_project_protected(*, observations: ObservationSession, source: ProjectionSource) -> ProtectedProjection:
    from anonymizer.engine.graph_sdk.extensions import project_protected

    reserve_observation_scope(session=observations, requirements=_requirements(2, 2, observations.limits))
    from anonymizer.engine.graph_sdk.extensions import ProjectionSource

    if isinstance(source, ProjectionSource):
        observations._bind(_PRODUCER, execution=source.execution, qualification=source.qualification)
    with _Span("projection", session=observations) as span:
        with _Span("release") as release:
            result = project_protected(source=source)
            if not result.items:
                release.outcome = "blocked"
        span.outcome = release.outcome
        return result


def observed_project_content(
    *,
    observations: ObservationSession,
    source: ProjectionSource,
    kind: ContentKind,
    artifacts: tuple[ArtifactSelection, ...],
) -> ContentProjection:
    from anonymizer.engine.graph_sdk.extensions import project_content

    reserve_observation_scope(session=observations, requirements=_requirements(1, 2, observations.limits))
    from anonymizer.engine.graph_sdk.extensions import ProjectionSource

    if isinstance(source, ProjectionSource):
        observations._bind(_PRODUCER, execution=source.execution, qualification=source.qualification)
    with _Span("projection", session=observations):
        result = project_content(source=source, kind=kind, artifacts=artifacts)
        return result


if TYPE_CHECKING:
    from anonymizer.engine.graph_sdk._execution_admission import (
        AdmittedContextPlan,
        AdmittedExecutionPlan,
        AssessmentLimits,
        EvidenceProductionDecl,
        ImplementationCapability,
        IngressLimits,
        MapExpansionDecl,
        OperationExecutionPolicy,
    )
    from anonymizer.engine.graph_sdk.context import (
        BoundContext,
        PreparedPlan,
    )
    from anonymizer.engine.graph_sdk.executor import ExecutionResult
    from anonymizer.engine.graph_sdk.extensions import (
        ArtifactSelection,
        ContentKind,
        ContentProjection,
        ProjectionSource,
        ProtectedProjection,
    )
    from anonymizer.engine.graph_sdk.preparation import (
        ActivationLimits,
        BoundInput,
        Graph,
        ImplementationSelection,
        PreparationConfiguration,
        PreparationLimits,
        ValidatedDataGraph,
    )
    from anonymizer.engine.graph_sdk.qualification import (
        AdmittedQualification,
        AssessmentSubmission,
        EvidenceRevisionView,
        QualificationResult,
    )
