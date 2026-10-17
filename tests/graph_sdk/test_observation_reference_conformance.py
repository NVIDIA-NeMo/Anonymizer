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
from tests.graph_sdk.reference.observations import reference
from tests.graph_sdk.reference.observations.reference import PURPOSES as REFERENCE_PURPOSES
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


# Bool correlation ordinals are reachable only by writing into the session's private pair table; the
# closed-schema re-check that rejected them was removed with checked construction (U1.5).
PRIVATE_STATE_INJECTION_IDS = frozenset({"terminal-ordinal-bool", "terminal-parent-bool"})


@pytest.mark.parametrize(
    "row",
    [row for row in CORPUS["observations"] if row["id"] not in PRIVATE_STATE_INJECTION_IDS],
    ids=lambda row: row["id"],
)
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


# Adaptive retrieval is not a product request purpose. Its column is projected out
# of every accounting row. The only adaptive-only row is excluded. Rows that mix
# adaptive dispatches with retained purposes keep their other dispatches; their
# expected answers are recomputed by the frozen independent ``reference.account``
# over the projected facts (never from product output), with the adaptive column
# removed. The projected two-owners row leaves its second scope empty, so one
# added non-adaptive two-owner row keeps summed two-scope accounting covered.
_ADAPTIVE_ONLY_ACCOUNTING_IDS = frozenset({"purpose-adaptive_retrieval"})
_RETAINED_PURPOSES = [purpose for purpose in REFERENCE_PURPOSES if purpose != "adaptive_retrieval"]
_TWO_OWNER_RETAINED = {
    "id": "two-owners-retained",
    "facts": [
        {
            "dispatches": [{"identity": 0, "purpose": "initial_binding", "usage": [1, 2]}],
            "possible_purposes": _RETAINED_PURPOSES,
        },
        {
            "dispatches": [
                {"identity": 1, "purpose": "initial", "usage": [3, 4]},
                {"identity": 2, "purpose": "retry", "usage": [5, 6]},
            ],
            "possible_purposes": _RETAINED_PURPOSES,
        },
    ],
}


def _reference_expected(facts):
    def estimate(value):
        return {"knowledge": value.knowledge, "value": value.value}

    accounting = reference.account(
        tuple(
            reference.Scope(
                dispatches=None
                if fact["dispatches"] is None
                else tuple(
                    reference.Dispatch(
                        identity=item["identity"],
                        purpose=item["purpose"],
                        usage=None if item["usage"] is None else tuple(item["usage"]),
                    )
                    for item in fact["dispatches"]
                ),
                possible_purposes=tuple(fact["possible_purposes"]),
            )
            for fact in facts
        )
    )
    column = REFERENCE_PURPOSES.index("adaptive_retrieval")
    return {
        "purposes": [estimate(value) for index, value in enumerate(accounting.purposes) if index != column],
        "total": estimate(accounting.total),
        "input_units": estimate(accounting.input_units),
        "output_units": estimate(accounting.output_units),
        "unknown_requests": estimate(accounting.unknown_requests),
    }


def _without_adaptive_purpose(row):
    column = REFERENCE_PURPOSES.index("adaptive_retrieval")
    facts = [
        {
            **fact,
            "dispatches": None
            if fact["dispatches"] is None
            else [item for item in fact["dispatches"] if item["purpose"] != "adaptive_retrieval"],
            "possible_purposes": [p for p in fact["possible_purposes"] if p != "adaptive_retrieval"],
        }
        for fact in row["facts"]
    ]
    if facts == [
        {**fact, "possible_purposes": [p for p in fact["possible_purposes"] if p != "adaptive_retrieval"]}
        for fact in row["facts"]
    ]:
        purposes = [value for index, value in enumerate(row["expected"]["purposes"]) if index != column]
        expected = {**row["expected"], "purposes": purposes}
        assert expected == _reference_expected(facts)
    else:
        expected = _reference_expected(facts)
    return {**row, "facts": facts, "expected": expected}


_ACCOUNTING_ROWS = [
    _without_adaptive_purpose(row) for row in CORPUS["accounting"] if row["id"] not in _ADAPTIVE_ONLY_ACCOUNTING_IDS
] + [{**_TWO_OWNER_RETAINED, "expected": _reference_expected(_TWO_OWNER_RETAINED["facts"])}]


def test_projected_accounting_answers_match_reviewed_values() -> None:
    expected = {row["id"]: row["expected"] for row in _ACCOUNTING_ROWS}
    for row_id, total, units_in, units_out in (
        ("seven-purposes", 6, 19, 25),
        ("two-owners", 1, 1, 2),
        ("two-owners-retained", 3, 9, 12),
    ):
        assert expected[row_id]["total"] == {"knowledge": "exact", "value": total}
        assert expected[row_id]["input_units"] == {"knowledge": "exact", "value": units_in}
        assert expected[row_id]["output_units"] == {"knowledge": "exact", "value": units_out}


@pytest.mark.parametrize("row", _ACCOUNTING_ROWS, ids=lambda row: row["id"])
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
