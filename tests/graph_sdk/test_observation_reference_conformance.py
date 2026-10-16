# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frozen neutral pair facts exercise session guards and closed encoding.

Malformed halves mutate private producer storage only in this test adapter;
SDK owners cannot submit arbitrary public pairs. Actual owner hooks are tested
separately in test_observations, rather than represented by these neutral facts.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from anonymizer.engine.graph_sdk.observations import (
    _OBSERVATION_KEY,
    _PRODUCER,
    CorrelationRef,
    ObservationFailure,
    ObservationLimits,
    ObservationReceipt,
    ObservationRequirements,
    begin_observations,
    reserve_observation_scope,
)
from tests.graph_sdk.test_observations import Clock, Sink

CORPUS = json.loads((Path(__file__).parent / "reference/observations/corpus.json").read_text())


def _limits(facts) -> ObservationLimits:
    return ObservationLimits(
        **{
            "max_" + name
            if name not in {"duration", "count"}
            else "max_duration_ns"
            if name == "duration"
            else "max_count": value
            for name, value in facts.items()
        }
    )


def _run(facts):
    bounds = _limits(facts["limits"])
    # Neutral intervals have no shared clock origin. Translate each interval
    # together, preserving its exact duration/reversal, into the real adapter's
    # sequential clock stream. Raw ticks never enter the observable receipt.
    ticks = []
    base = 0
    for phase in facts["phases"]:
        for pair in phase["pairs"]:
            start = base + pair["start_ns"]
            end = base + pair["end_ns"]
            if pair["start_present"]:
                ticks.append(start)
            if pair["terminal_present"]:
                ticks.append(end)
            base = max(start, end) + 1
    sink = Sink(facts["sink_failure_at"]) if facts["sink_configured"] else None
    session = begin_observations(limits=bounds, clock=Clock(tuple(ticks)), sink=sink)
    if isinstance(session, ObservationFailure):
        return session, 0
    owners = {0: object(), 1: object()}
    session._bind(_PRODUCER, data=owners[0])
    for phase in facts["phases"]:
        need = phase["need"]
        requirements = ObservationRequirements(
            _key=_OBSERVATION_KEY,
            max_pairs_required=need["pairs"],
            max_events_required=need["events"],
            max_nodes_required=need["nodes"],
            max_depth_required=need["depth"],
            max_encoded_bytes_required=need["encoded_bytes"],
            _limits=_limits(need["envelope"]),
        )
        reserve_observation_scope(session=session, requirements=requirements)
        session._bind(_PRODUCER, data=owners[phase["owner"]])
        for pair in phase["pairs"]:
            ordinal = (
                session._enter(_PRODUCER, pair["boundary"], pair["parent"])
                if pair["start_present"]
                else pair["ordinal"]
            )
            if ordinal is not None and ordinal in session._open:
                start, tick = session._open[ordinal]
                session._open[ordinal] = (
                    replace(start, correlation=CorrelationRef(ordinal=pair["ordinal"], parent_ordinal=pair["parent"])),
                    tick,
                )
            if pair["terminal_present"]:
                session._exit(_PRODUCER, ordinal, pair["outcome"], frozenset(pair["reasons"]), None)
                if ordinal is not None and ordinal in session._pairs:
                    complete = session._pairs[ordinal]
                    session._pairs[ordinal] = replace(
                        complete,
                        terminal=replace(
                            complete.terminal,
                            correlation=CorrelationRef(
                                ordinal=pair["terminal_ordinal"], parent_ordinal=pair["terminal_parent"]
                            ),
                            boundary=pair["terminal_boundary"],
                        ),
                    )
    return session.close(), 0 if sink is None else len(sink.pairs)


@pytest.mark.parametrize("row", CORPUS["observations"], ids=lambda row: row["id"])
def test_frozen_observation_answers(row) -> None:
    actual, delivered = _run(row["facts"])
    if isinstance(actual, ObservationFailure):
        value = asdict(actual)
        kind = "Failure"
    else:
        assert isinstance(actual, ObservationReceipt)
        kind = "Receipt"
        value = dict(
            schema_version=actual.schema_version,
            delivery=actual.delivery,
            delivered_pairs=actual.delivered_pairs,
            pairs=[
                dict(
                    ordinal=pair.start.correlation.ordinal,
                    parent=pair.start.correlation.parent_ordinal,
                    boundary=pair.start.boundary,
                    outcome=pair.terminal.outcome,
                    reasons=sorted(pair.terminal.reasons),
                    duration_ns=pair.terminal.duration_ns,
                )
                for pair in actual.pairs
            ],
        )
        # Corpus reason arrays use canonical enum order, not lexical order.
        from tests.graph_sdk.reference.observations.reference import REASONS

        for pair in value["pairs"]:
            pair["reasons"] = [reason for reason in REASONS if reason in pair["reasons"]]
    assert kind == row["expected_type"]
    assert value == row["expected"]
    assert delivered == row["delivered_pairs"]


# Adaptive retrieval is not a product request purpose. Rows that dispatch it are
# excluded; the remaining rows project out only its purpose column.
_ADAPTIVE_ACCOUNTING_IDS = frozenset({"purpose-adaptive_retrieval", "seven-purposes", "two-owners"})


def _without_adaptive_purpose(row):
    from tests.graph_sdk.reference.observations.reference import PURPOSES

    column = PURPOSES.index("adaptive_retrieval")
    facts = [
        {**fact, "possible_purposes": [p for p in fact["possible_purposes"] if p != "adaptive_retrieval"]}
        for fact in row["facts"]
    ]
    assert not any(
        dispatch["purpose"] == "adaptive_retrieval" for fact in facts for dispatch in fact["dispatches"] or ()
    )
    purposes = [value for index, value in enumerate(row["expected"]["purposes"]) if index != column]
    return {**row, "facts": facts, "expected": {**row["expected"], "purposes": purposes}}


@pytest.mark.parametrize(
    "row",
    [_without_adaptive_purpose(row) for row in CORPUS["accounting"] if row["id"] not in _ADAPTIVE_ACCOUNTING_IDS],
    ids=lambda row: row["id"],
)
def test_frozen_accounting_answers(row) -> None:
    from anonymizer.engine.graph_sdk.inspection import _PURPOSES, _account_receipts
    from anonymizer.engine.graph_sdk.requests import (
        BindingAssociation,
        BindingDeclarationId,
        BindingId,
        BindingRequestScope,
        ExactUsage,
        ExternalSettlement,
        PhysicalRequestId,
        RequestReservation,
        UnknownUsage,
        initialize_requests,
        request_receipt,
    )
    from tests.graph_sdk.test_requests import _policy

    scopes = []
    policy = _policy()
    for neutral in row["facts"]:
        possible = frozenset(neutral["possible_purposes"])
        if neutral["dispatches"] is None:
            scopes.append((None, possible))
            continue
        scope = BindingRequestScope(binding=BindingId.new())
        empty = request_receipt(initialize_requests(scope=scope, hard_limit=None, policies=frozenset({policy})))
        dispatches = []
        settlements = []
        identities = {}
        for fact in neutral["dispatches"]:
            request = identities.setdefault(fact["identity"], PhysicalRequestId.new(scope=scope))
            association = BindingAssociation(
                declaration=BindingDeclarationId.new(binding=scope.binding, ordinal=fact["identity"])
            )
            dispatches.append(
                RequestReservation(
                    request=request, purpose=fact["purpose"], associations=frozenset({association}), policy=policy
                )
            )
            usage = (
                UnknownUsage()
                if fact["usage"] is None
                else ExactUsage(input_units=fact["usage"][0], output_units=fact["usage"][1])
            )
            settlements.append(
                ExternalSettlement(request=request, disposition="completed", usage=usage, remote_stopped=True)
            )
        scopes.append(
            (
                replace(
                    empty,
                    dispatched_count=len(dispatches),
                    dispatches=tuple(dispatches),
                    settlements=tuple(settlements),
                ),
                possible,
            )
        )
    counts, usage = _account_receipts(scopes)
    actual = dict(
        purposes=[asdict(getattr(counts, purpose)) for purpose in _PURPOSES],
        total=asdict(counts.total),
        input_units=asdict(usage.input_units),
        output_units=asdict(usage.output_units),
        unknown_requests=asdict(usage.requests_with_unknown_usage),
    )
    assert actual == row["expected"]
