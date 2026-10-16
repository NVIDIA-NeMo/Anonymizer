# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.inspection import InspectionLimits, encode_inspection
from anonymizer.engine.graph_sdk.observations import (
    _OBSERVATION_KEY,
    _PRODUCER,
    CorrelationRef,
    ObservationFailure,
    ObservationLimits,
    ObservationReceipt,
    ObservationSession,
    _requirements,
    _Span,
    begin_observations,
    reserve_observation_scope,
)
from anonymizer.graph import EffectCode, EffectRejected


class Clock:
    def __init__(self, values: tuple[int, ...] = ()) -> None:
        self.values = iter(values)
        self.count = 0

    def now_ns(self) -> int:
        self.count += 1
        return next(self.values, self.count)


class Sink:
    def __init__(self, fail_at: int | None = None) -> None:
        self.pairs = []
        self.fail_at = fail_at
        self.calls = 0

    def observe(self, pair) -> None:
        self.calls += 1
        if len(self.pairs) == self.fail_at:
            raise ValueError("PRIVATE_SINK_CANARY")
        self.pairs.append(pair)


def limits(**changes: int) -> ObservationLimits:
    return replace(
        ObservationLimits(
            max_pairs=100,
            max_events=200,
            max_nodes=10000,
            max_depth=30,
            max_encoded_bytes=100000,
            max_duration_ns=100,
            max_count=100,
        ),
        **changes,
    )


def inspection_limits(**changes: int) -> InspectionLimits:
    return replace(
        InspectionLimits(
            max_pairs=100,
            max_events=200,
            max_nodes=10000,
            max_depth=30,
            max_encoded_bytes=100000,
            max_reason_counts=17,
            max_count=100,
        ),
        **changes,
    )


def session(
    *, clock: Clock | None = None, sink: Sink | None = None, bounds: ObservationLimits | None = None, pairs: int = 3
) -> ObservationSession:
    result = begin_observations(limits=bounds or limits(), clock=clock or Clock(), sink=sink)
    assert isinstance(result, ObservationSession)
    reserve_observation_scope(session=result, requirements=_requirements(pairs, 3, result.limits))
    return result


def test_nested_complete_pairs_canonical_encoding_and_cached_close() -> None:
    observer = session()
    with _Span("execution", session=observer):
        with _Span("operation"):
            pass
    receipt = observer.close()
    assert isinstance(receipt, ObservationReceipt)
    assert observer.close() is receipt
    assert [pair.start.correlation.ordinal for pair in receipt.pairs] == [0, 1]
    assert [pair.start.correlation.parent_ordinal for pair in receipt.pairs] == [None, 0]
    assert [pair.terminal.duration_ns for pair in receipt.pairs] == [3, 1]
    encoded = encode_inspection(value=receipt, limits=inspection_limits())
    assert json.loads(encoded)["pairs"][1]["terminal"]["duration_ns"] == 1
    assert b"PRIVATE" not in encoded


@pytest.mark.parametrize("fail_at", [0, 1])
def test_sink_failure_preserves_complete_pairs_and_stops_delivery(fail_at: int) -> None:
    sink = Sink(fail_at)
    observer = session(sink=sink)
    for _ in range(3):
        with _Span("operation", session=observer):
            pass
    receipt = observer.close()
    assert isinstance(receipt, ObservationReceipt)
    assert len(receipt.pairs) == 3 and receipt.delivered_pairs == fail_at
    assert receipt.delivery == "failed" and sink.calls == fail_at + 1
    assert b"PRIVATE" not in encode_inspection(value=receipt, limits=inspection_limits())


@pytest.mark.parametrize(("ticks", "code"), [((3, 2), "clock_contract"), ((0, 101), "capacity_contract")])
def test_clock_failures_do_not_replace_semantic_body(ticks, code) -> None:
    observer = session(clock=Clock(ticks))
    body = []
    with _Span("execution", session=observer):
        body.append("semantic")
    assert body == ["semantic"]
    receipt = observer.close()
    assert isinstance(receipt, ObservationFailure) and receipt.code == code


@pytest.mark.parametrize(
    "bound", ["max_pairs", "max_events", "max_nodes", "max_depth", "max_encoded_bytes", "max_count"]
)
def test_begin_requires_a_whole_pair(bound: str) -> None:
    result = begin_observations(limits=limits(**{bound: 0}), clock=Clock())
    assert isinstance(result, ObservationFailure) and result.code == "capacity_contract"


def test_cumulative_capacity_before_next_phase_and_equal_envelope() -> None:
    sink = Sink()
    observer = session(sink=sink, bounds=limits(max_count=1), pairs=1)
    with _Span("execution", session=observer):
        pass
    failure = reserve_observation_scope(session=observer, requirements=_requirements(1, 1, replace(observer.limits)))
    assert isinstance(failure, ObservationFailure)
    assert observer.close() is failure and len(sink.pairs) == 1
    other = session(pairs=0)
    assert reserve_observation_scope(session=other, requirements=_requirements(1, 1, replace(other.limits))) is None
    assert reserve_observation_scope(session=other, requirements=_requirements(1, 1, limits(max_count=101))) is not None


def test_foreign_producer_and_unclosed_pair_fail_closed() -> None:
    first = session()
    first._enter(object(), "operation", None)
    assert isinstance(first.close(), ObservationFailure)
    second = session()
    second._enter(_PRODUCER, "operation", None)
    assert isinstance(second.close(), ObservationFailure)


def test_semantic_exception_is_preserved_without_its_message() -> None:
    observer = session()
    error = ValueError("PRIVATE_EXCEPTION_CANARY")
    with pytest.raises(ValueError) as caught:
        with _Span("operation", session=observer):
            raise error
    assert caught.value is error
    receipt = observer.close()
    assert isinstance(receipt, ObservationReceipt)
    assert receipt.pairs[0].terminal.reasons == frozenset({"internal_failure"})
    assert b"PRIVATE" not in encode_inspection(value=receipt, limits=inspection_limits())


def test_encoding_exact_nodes_depth_bytes_and_bool_correlation() -> None:
    observer = session()
    with _Span("execution", session=observer):
        pass
    receipt = observer.close()
    assert isinstance(receipt, ObservationReceipt)
    raw = encode_inspection(value=receipt, limits=inspection_limits())

    def shape(value, depth=1):
        children = value.values() if isinstance(value, dict) else value if isinstance(value, list) else ()
        nested = [shape(child, depth + 1) for child in children]
        return 1 + sum(n for n, _ in nested), max([depth] + [d for _, d in nested])

    nodes, depth = shape(json.loads(raw))
    exact = inspection_limits(max_nodes=nodes, max_depth=depth, max_encoded_bytes=len(raw))
    assert encode_inspection(value=receipt, limits=exact) == raw
    for field in ("max_nodes", "max_depth", "max_encoded_bytes"):
        with pytest.raises(EffectRejected) as caught:
            encode_inspection(value=receipt, limits=replace(exact, **{field: getattr(exact, field) - 1}))
        assert caught.value.code == EffectCode.LIMIT_EXCEEDED
    pair = receipt.pairs[0]
    invalid = replace(
        pair, terminal=replace(pair.terminal, correlation=CorrelationRef(ordinal=False, parent_ordinal=None))
    )
    mutated = ObservationReceipt(
        _key=_OBSERVATION_KEY, schema_version=1, pairs=(invalid,), delivery="not_configured", delivered_pairs=0
    )
    with pytest.raises(EffectRejected) as caught:
        encode_inspection(value=mutated, limits=inspection_limits())
    assert caught.value.code == EffectCode.INVALID_TYPE


@pytest.mark.parametrize("scenario", ["local", "nested", "adaptive", "external"])
def test_actual_execution_hooks_and_pure_inspection(monkeypatch, scenario: str) -> None:
    import asyncio

    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.engine.graph_sdk.inspection import bind_inspection_source, diagnose, explain, inspect
    from tests.graph_sdk import test_adaptive_executor as remote
    from tests.graph_sdk import test_local_executor as local

    observed = begin_observations(
        limits=limits(
            max_count=10000,
            max_pairs=1000,
            max_events=2000,
            max_nodes=100000,
            max_encoded_bytes=1000000,
            max_duration_ns=10000,
            max_depth=1000,
        ),
        clock=Clock(),
    )
    assert isinstance(observed, ObservationSession)
    captured = {}

    async def start(**kwargs):
        running = await start_execution(**kwargs, observations=observed)
        captured.update(admitted=kwargs["admitted"], running=running)
        return running

    monkeypatch.setattr(local, "start_execution", start)
    monkeypatch.setattr(remote, "start_execution", start)

    async def run():
        if scenario == "local":
            await local._assert_local_execution()
        elif scenario == "nested":
            await local._assert_shared_nested_body()
        elif scenario == "adaptive":
            await remote._assert_adaptive_retrieval()
        else:
            await remote._assert_external_retry()
        result = await captured["running"].wait()
        receipt = observed.close()
        assert isinstance(receipt, ObservationReceipt)
        boundaries = [pair.start.boundary for pair in receipt.pairs]
        assert boundaries.count("execution") == 1
        entered = sum(fact.attempt is not None for fact in result.record.terminals)
        assert boundaries.count("operation") == entered
        assert boundaries.count("physical_request") == result.requests.dispatched_count
        assert boundaries.count("request_reconciliation") == result.requests.dispatched_count
        if scenario == "adaptive":
            assert boundaries.count("adaptive_retrieval") == 1
        admitted = captured["admitted"]
        source = bind_inspection_source(
            prepared=admitted.context.prepared,
            context=admitted.context,
            execution_plan=admitted,
            execution=result,
            observations=receipt,
            observation_session=observed,
            limits=inspection_limits(
                max_count=10000,
                max_pairs=1000,
                max_events=2000,
                max_nodes=100000,
                max_depth=1000,
                max_encoded_bytes=1000000,
            ),
        )
        first = inspect(source=source)
        assert first == inspect(source=source)
        assert first.requests.total.knowledge == "unknown"  # no initial-binding receipt
        assert first.requests.initial_binding.knowledge == "unknown"
        assert first.requests.initial.value == sum(row.purpose == "initial" for row in result.requests.dispatches)
        assert explain(source=source) == explain(source=source)
        assert diagnose(source=source) == diagnose(source=source)
        for value in (receipt, first, explain(source=source), diagnose(source=source)):
            assert encode_inspection(value=value, limits=source.limits)

    asyncio.run(run())


def test_actual_initial_binding_receipt_has_its_own_exact_scope(monkeypatch) -> None:
    import asyncio

    from anonymizer.engine.graph_sdk.binding import start_initial_binding
    from anonymizer.engine.graph_sdk.inspection import bind_binding_inspection_source, diagnose_binding, inspect_binding
    from tests.graph_sdk import test_binding as fixture

    observer = begin_observations(limits=limits(), clock=Clock())
    assert isinstance(observer, ObservationSession)

    async def start(**kwargs):
        return await start_initial_binding(**kwargs, observations=observer)

    monkeypatch.setattr(fixture, "start_initial_binding", start)

    async def run():
        provider = fixture._Provider()
        running, data, workflow, _ = await fixture._initial_running(provider)
        result = await running.wait()
        receipt = observer.close()
        assert isinstance(receipt, ObservationReceipt)
        assert [pair.start.boundary for pair in receipt.pairs] == [
            "initial_binding",
            "physical_request",
            "request_reconciliation",
            "cleanup",
        ]
        source = bind_binding_inspection_source(
            data=data, workflow=workflow, binding=result, limits=inspection_limits()
        )
        view = inspect_binding(source=source)
        assert view.requests.total.value == view.requests.initial_binding.value == 1
        assert view.requests.initial.value == 0
        assert view.usage.input_units.knowledge == "exact"
        assert provider.calls == 1
        assert not diagnose_binding(source=source).usage_unknown
        assert encode_inspection(value=view, limits=source.limits)

    asyncio.run(run())


@pytest.mark.parametrize("sink_failure", [None, 0, 2])
def test_full_pure_owner_chain_commit_and_repeated_release(monkeypatch, sink_failure) -> None:
    import asyncio

    from anonymizer.engine.graph_sdk import observations as api
    from anonymizer.engine.graph_sdk.continuity import StateNamespace, create_continuity_state
    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.engine.graph_sdk.extensions import bind_projection_source
    from anonymizer.engine.graph_sdk.inspection import bind_inspection_source, inspect
    from tests.graph_sdk import continuity_graph_fixture as fixture
    from tests.graph_sdk.continuity_fixtures import limits as continuity_limits
    from tests.graph_sdk.continuity_fixtures import prepare_case

    observer = begin_observations(
        limits=limits(
            max_count=10000,
            max_pairs=1000,
            max_events=2000,
            max_nodes=100000,
            max_encoded_bytes=1000000,
            max_duration_ns=10000,
        ),
        clock=Clock(),
        sink=Sink(sink_failure),
    )
    assert isinstance(observer, ObservationSession)
    # Each replacement calls one explicit supported wrapper; no arbitrary
    # callback is represented as an SDK boundary.
    monkeypatch.setattr(fixture, "prepare", lambda **kwargs: api.observed_prepare(observations=observer, **kwargs))
    monkeypatch.setattr(
        fixture, "admit_context_plan", lambda **kwargs: api.observed_context_admission(observations=observer, **kwargs)
    )
    monkeypatch.setattr(
        fixture,
        "admit_execution_plan",
        lambda **kwargs: api.observed_execution_admission(observations=observer, **kwargs),
    )
    monkeypatch.setattr(fixture, "qualify", lambda **kwargs: api.observed_qualify(observations=observer, **kwargs))

    async def start(**kwargs):
        return await start_execution(observations=observer, **kwargs)

    monkeypatch.setattr(fixture, "start_execution", start)

    async def run():
        state = create_continuity_state(
            namespace=StateNamespace(name="PRIVATE_STATE_CANARY"), limits=continuity_limits()
        )
        graph, admitted = await prepare_case("independent", state)
        committed = api.observed_commit_qualified(
            observations=observer, admitted=admitted, execution=graph.result, qualification=graph.qualified
        )
        projection_source = bind_projection_source(
            execution=graph.result, qualification=graph.qualified, continuity=committed
        )
        first = api.observed_project_protected(observations=observer, source=projection_source)
        second = api.observed_project_protected(observations=observer, source=projection_source)
        assert first.items == second.items and {item.content for item in first.items} == {"A", "B"}
        receipt = observer.close()
        assert isinstance(receipt, ObservationReceipt)
        boundaries = [pair.start.boundary for pair in receipt.pairs]
        assert (
            boundaries.count("preparation")
            == boundaries.count("context_admission")
            == boundaries.count("execution_admission")
            == 1
        )
        assert boundaries.count("qualification") == boundaries.count("continuity_commit") == 1
        assert boundaries.count("projection") == boundaries.count("release") == 2
        assert receipt.delivery == ("complete" if sink_failure is None else "failed")
        source = bind_inspection_source(
            prepared=graph.execution.context.prepared,
            context=graph.execution.context,
            execution_plan=graph.execution,
            execution=graph.result,
            qualification=graph.qualified,
            continuity=committed,
            observations=receipt,
            observation_session=observer,
            limits=inspection_limits(
                max_count=10000, max_pairs=1000, max_events=2000, max_nodes=100000, max_encoded_bytes=1000000
            ),
        )
        view = inspect(source=source)
        assert view.qualification is not None
        assert view.qualification.met == view.qualification.assessed_targets == 2
        assert view.qualification.verified_evidence == len(graph.qualified.verified)
        assert b"PRIVATE_STATE_CANARY" not in encode_inspection(value=receipt, limits=source.limits)
        foreign = session()
        foreign.close()
        with pytest.raises(EffectRejected) as caught:
            bind_inspection_source(
                prepared=source.prepared,
                context=source.context,
                execution_plan=source.execution_plan,
                execution=source.execution,
                observations=receipt,
                observation_session=foreign,
                limits=source.limits,
            )
        assert caught.value.code == EffectCode.FOREIGN_OWNER

    asyncio.run(run())


@pytest.mark.parametrize("mode", ["valid", "raises", "wrong_route", "over_count"])
def test_selected_measurement_cannot_change_semantic_result(monkeypatch, mode: str) -> None:
    import asyncio

    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.engine.graph_sdk.observations import WorkMeasurement
    from tests.graph_sdk import test_local_executor as fixture

    observer = begin_observations(limits=limits(max_count=1000, max_duration_ns=1000), clock=Clock())
    assert isinstance(observer, ObservationSession)

    class Measurement:
        def __repr__(self):
            raise AssertionError("PRIVATE_REPR_CANARY")

        def terminal_measurement(self):
            if mode == "raises":
                raise ValueError("PRIVATE_MEASUREMENT_CANARY")
            return WorkMeasurement(
                role="repair",
                route="external" if mode == "wrong_route" else "local",
                input_count=1001 if mode == "over_count" else 1,
                output_count=1,
                artifact_count=1,
                artifact_bytes=1,
                request_count=0,
            )

    async def start(**kwargs):
        services = kwargs.pop("services")
        services = replace(
            services, handles=tuple(replace(handle, measurement=Measurement()) for handle in services.handles)
        )
        return await start_execution(**kwargs, services=services, observations=observer)

    monkeypatch.setattr(fixture, "start_execution", start)
    asyncio.run(fixture._assert_local_execution())
    receipt = observer.close()
    if mode in {"wrong_route", "over_count"}:
        assert isinstance(receipt, ObservationFailure)
    else:
        assert isinstance(receipt, ObservationReceipt)
        pairs = [pair for pair in receipt.pairs if pair.start.boundary == "operation"]
        assert len(pairs) == 2
        if mode == "raises":
            assert all(
                pair.terminal.measurement is None and "internal_failure" in pair.terminal.reasons for pair in pairs
            )
        else:
            assert all(
                pair.terminal.measurement is not None and pair.terminal.measurement.role == "repair" for pair in pairs
            )
        assert b"PRIVATE" not in encode_inspection(value=receipt, limits=inspection_limits(max_count=1000))


@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("mode", ["success", "cancel", "exception", "awaitable", "queue"])
def test_blocking_actual_starts_never_count_queued_waiters(monkeypatch, adaptive: bool, mode: str) -> None:
    from anonymizer.engine.graph_sdk.binding import start_initial_binding
    from anonymizer.engine.graph_sdk.executor import start_execution
    from tests.graph_sdk import test_adaptive_executor as later
    from tests.graph_sdk import test_binding as initial
    from tests.graph_sdk import test_blocking as blocking

    sessions = []

    async def start(**kwargs):
        observer = begin_observations(limits=limits(max_count=1000, max_duration_ns=1000), clock=Clock())
        assert isinstance(observer, ObservationSession)
        sessions.append(observer)
        return await (
            start_execution(**kwargs, observations=observer)
            if adaptive
            else start_initial_binding(**kwargs, observations=observer)
        )

    monkeypatch.setattr(
        later if adaptive else initial, "start_execution" if adaptive else "start_initial_binding", start
    )
    if mode == "queue":
        blocking.test_actual_retrieval_queue_cancellation_preserves_worker_capacity(adaptive)
    else:
        blocking.test_actual_initial_and_adaptive_blocking_provider(adaptive, mode)
    counts = []
    for observer in sessions:
        receipt = observer.close()
        assert isinstance(receipt, ObservationReceipt)
        pairs = [pair for pair in receipt.pairs if pair.start.boundary == "physical_request"]
        counts.append(len(pairs))
        if mode == "cancel":
            assert pairs[0].terminal.outcome == "lost"
        assert len([pair for pair in receipt.pairs if pair.start.boundary == "request_reconciliation"]) == len(pairs)
    assert counts == ([1, 0, 1] if mode == "queue" else [1])


@pytest.mark.parametrize("capacity", [False, True])
def test_observation_failure_never_changes_actual_local_execution(monkeypatch, capacity: bool) -> None:
    import asyncio

    from anonymizer.engine.graph_sdk.executor import start_execution
    from tests.graph_sdk import test_local_executor as fixture

    observer = begin_observations(limits=limits(max_pairs=1 if capacity else 100), clock=Clock((2, 1)))
    assert isinstance(observer, ObservationSession)

    async def start(**kwargs):
        return await start_execution(**kwargs, observations=observer)

    monkeypatch.setattr(fixture, "start_execution", start)
    asyncio.run(fixture._assert_local_execution())
    receipt = observer.close()
    assert isinstance(receipt, ObservationFailure)
    assert receipt.code == ("capacity_contract" if capacity else "clock_contract")


def test_receipt_reachable_graph_and_owned_logs_exclude_semantic_canaries(caplog) -> None:
    import logging
    from dataclasses import fields, is_dataclass

    from anonymizer.engine.graph_sdk.inspection import _MEMBERS

    canary = "PRIVATE_姓名_credential_endpoint_prompt_entity_hash"
    observer = session()
    observer._bind(_PRODUCER, data={"text": canary, "credentials": canary, "labels": canary})
    with caplog.at_level(logging.DEBUG):
        with _Span("operation", session=observer):
            pass
        receipt = observer.close()
        logging.getLogger("uncontrolled_host_fixture").warning(canary)
    assert isinstance(receipt, ObservationReceipt)
    pending = [receipt]
    while pending:
        value = pending.pop()
        if is_dataclass(value):
            assert type(value) in _MEMBERS
            pending.extend(getattr(value, field.name) for field in fields(value))
        elif type(value) in (tuple, frozenset):
            pending.extend(value)
        else:
            assert value is None or type(value) in (str, int, bool)
            assert value != canary
    owned = [
        record
        for record in caplog.records
        if record.name == "anonymizer.graph"
        or record.name.startswith("anonymizer.graph.")
        or record.name == "anonymizer.engine.graph_sdk"
        or record.name.startswith("anonymizer.engine.graph_sdk.")
    ]
    assert all(canary not in record.getMessage() for record in owned)
    assert any(
        record.name == "uncontrolled_host_fixture" and canary in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize(
    "short", [None, "max_pairs", "max_events", "max_nodes", "max_depth", "max_encoded_bytes", "max_count"]
)
def test_compiled_whole_execution_exact_and_one_less_capacity(monkeypatch, short) -> None:
    import asyncio

    from anonymizer.engine.graph_sdk.executor import start_execution
    from anonymizer.engine.graph_sdk.observations import execution_observation_requirements
    from tests.graph_sdk import test_local_executor as fixture

    captured = []

    async def start(**kwargs):
        bounds = limits(max_count=1000, max_duration_ns=1000)
        need = execution_observation_requirements(execution=kwargs["admitted"], limits=bounds)
        bounds = replace(
            bounds,
            max_pairs=need.max_pairs_required,
            max_events=need.max_events_required,
            max_nodes=need.max_nodes_required,
            max_depth=need.max_depth_required,
            max_encoded_bytes=need.max_encoded_bytes_required,
            max_count=need.max_pairs_required,
        )
        # Count widths changed; recompute concrete byte maxima with that envelope.
        need = execution_observation_requirements(execution=kwargs["admitted"], limits=bounds)
        bounds = replace(bounds, max_encoded_bytes=need.max_encoded_bytes_required)
        if short is not None:
            bounds = replace(bounds, **{short: getattr(bounds, short) - 1})
        observer = begin_observations(limits=bounds, clock=Clock())
        captured.append(observer)
        return await start_execution(
            **kwargs, observations=observer if isinstance(observer, ObservationSession) else None
        )

    monkeypatch.setattr(fixture, "start_execution", start)
    asyncio.run(fixture._assert_local_execution())
    receipt = captured[0].close() if isinstance(captured[0], ObservationSession) else captured[0]
    if short is None:
        assert isinstance(receipt, ObservationReceipt)
        assert len(receipt.pairs) == 5
    else:
        assert isinstance(receipt, ObservationFailure) and receipt.code == "capacity_contract"
        assert not isinstance(captured[0], ObservationSession) or captured[0]._next == 0


def test_worst_schema_forecast_dominates_wide_integer_encoding() -> None:
    from typing import get_args

    from anonymizer.engine.graph_sdk.observations import ObservationReason, WorkMeasurement

    maximum = 10**5000
    bounds = limits(max_count=maximum, max_duration_ns=maximum, max_encoded_bytes=1000000)
    need = _requirements(1, 1, bounds)
    observer = session(bounds=bounds, pairs=1, clock=Clock((0, maximum)))
    with _Span("request_reconciliation", session=observer) as span:
        span.outcome = "inconsistent"
        span.reasons = frozenset(get_args(ObservationReason))
        span.measurement = WorkMeasurement(
            role="assessment",
            route="external",
            input_count=maximum,
            output_count=maximum,
            artifact_count=maximum,
            artifact_bytes=maximum,
            request_count=maximum,
        )
    receipt = observer.close()
    assert isinstance(receipt, ObservationReceipt)
    raw = encode_inspection(value=receipt, limits=inspection_limits(max_count=maximum, max_encoded_bytes=1000000))
    assert len(raw) <= need.max_encoded_bytes_required
    assert b"1" + b"0" * 5000 in raw


@pytest.mark.parametrize("boundary", ["context_admission", "execution_admission", "qualification", "projection"])
def test_pure_validation_rejection_still_closes_entered_boundary(boundary: str) -> None:
    from anonymizer.engine.graph_sdk import observations as api

    observer = session(pairs=0)
    functions = {
        "context_admission": (
            api.observed_context_admission,
            dict(prepared=None, bound_context=None, adaptive_retrievals=(), context_capabilities=()),
        ),
        "execution_admission": (
            api.observed_execution_admission,
            dict(
                context=None,
                capabilities=(),
                policies=(),
                decisions=(),
                assessment_productions=(),
                assessment_limits=None,
            ),
        ),
        "qualification": (api.observed_qualify, dict(admitted=None, result=None, current=None, submissions=())),
        "projection": (api.observed_project_protected, dict(source=None)),
    }
    function, kwargs = functions[boundary]
    with pytest.raises(EffectRejected) as caught:
        cast(Any, function)(observations=observer, **kwargs)
    assert caught.value.code == EffectCode.INVALID_TYPE
    receipt = observer.close()
    assert isinstance(receipt, ObservationReceipt)
    assert receipt.pairs[0].start.boundary == boundary
    assert all(
        pair.terminal.outcome == "failure" and pair.terminal.reasons == frozenset({"invalid_input"})
        for pair in receipt.pairs
    )


def test_binding_invalid_limits_returns_original_rejection_and_complete_pair(monkeypatch) -> None:
    import asyncio

    from anonymizer.engine.graph_sdk.binding import start_initial_binding
    from tests.graph_sdk import test_binding as fixture

    observer = session(pairs=0)

    async def start(**kwargs):
        kwargs["limits"] = None
        return await start_initial_binding(**kwargs, observations=observer)

    monkeypatch.setattr(fixture, "start_initial_binding", start)
    provider = fixture._Provider()
    with pytest.raises(EffectRejected) as caught:
        asyncio.run(fixture._initial_running(provider))
    assert caught.value.code == EffectCode.INVALID_TYPE and provider.calls == 0
    receipt = observer.close()
    assert isinstance(receipt, ObservationReceipt)
    assert len(receipt.pairs) == 1 and receipt.pairs[0].terminal.reasons == frozenset({"invalid_input"})


@pytest.mark.parametrize("malformed_join", [False, True])
def test_request_pairs_use_consumer_ingress_terminal(monkeypatch, malformed_join: bool) -> None:
    import asyncio

    from anonymizer.engine.graph_sdk.executor import start_execution
    from tests.graph_sdk import test_adaptive_executor as fixture

    observer = session(bounds=limits(max_count=1000, max_duration_ns=1000), pairs=1)

    class Provider(fixture._Provider):
        async def retrieve(self, **kwargs):
            if not malformed_join:
                return object()
            response = await super().retrieve(**kwargs)
            return replace(response, source=replace(response.source, revision=response.source.revision + 1))

    async def start(**kwargs):
        return await start_execution(**kwargs, observations=observer)

    monkeypatch.setattr(fixture, "start_execution", start)

    async def run():
        running, _ = await fixture._adaptive_running(Provider())
        return await running.wait()

    result = asyncio.run(run())
    receipt = observer.close()
    assert isinstance(receipt, ObservationReceipt)
    request_pairs = [
        pair for pair in receipt.pairs if pair.start.boundary in {"physical_request", "request_reconciliation"}
    ]
    assert len(request_pairs) == 2 * len(result.requests.terminals)
    for terminal in result.requests.terminals:
        assert all(pair.terminal.outcome == terminal.category for pair in request_pairs)
    assert all(pair.terminal.outcome == ("failure" if malformed_join else "lost") for pair in request_pairs)


def test_cleanup_of_sdk_resource_with_active_worker_reports_unknown() -> None:
    import asyncio
    from threading import Event

    from anonymizer.engine.graph_sdk.blocking import BlockingPool, BlockingTransport
    from anonymizer.engine.graph_sdk.requests import TransportFailure
    from anonymizer.engine.graph_sdk.resources import ResourceLease, close_resource
    from tests.graph_sdk.test_blocking import AFFINITY, Control, launch, reserved, until

    async def run():
        observer = session(bounds=limits(max_count=1000, max_duration_ns=1000), pairs=10)
        pool = BlockingPool(max_workers=1)
        entered, release = Event(), Event()

        def callback(request):
            entered.set()
            assert release.wait(5)
            return TransportFailure(failure="permanent", settlement=None)

        adapter = BlockingTransport(callback=callback, pool=pool, affinity=AFFINITY)
        authority, request = reserved()
        # The worker fixture stops at physical completion; observe only cleanup here.
        task = launch(adapter, authority, request, Control())
        try:
            await until(entered.is_set)
            lease = ResourceLease.create(owner="sdk", safe_detachment="independent_after_dispatch", handle=adapter)
            with _Span("execution", session=observer):
                fact = await close_resource(lease)
            assert fact.owner == "sdk" and fact.disposition == "left_open"
            receipt = observer.close()
            assert isinstance(receipt, ObservationReceipt)
            cleanup = next(pair for pair in receipt.pairs if pair.start.boundary == "cleanup")
            assert cleanup.terminal.outcome == "unknown"
            assert cleanup.terminal.reasons == frozenset({"cleanup_unknown"})
        finally:
            release.set()
            await task
            await pool.close()

    asyncio.run(run())
