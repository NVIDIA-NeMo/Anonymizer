# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constructible map and complete-loop slices of the finite predicates."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.executor import MapItemKey, OperationOutputKey
from tests.graph_sdk.effects_map_execution import _execute_loop_resolution, _execute_membership
from tests.graph_sdk.test_compact_foundation import _cases

MAPS = [
    case
    for case in _cases("map")
    if case["closed"]
    and (
        case["expansion"] == "success"
        and all(status == "success" for _, status in case["terminals"])
        and len(case["terminals"]) == len(case["members"])
        or case["closed"]
        and case["expansion"] in {"failed", "overflow"}
        and not case["members"]
    )
]
LOOP_MODES = {(): "bypass", ("continue", "exit"): "exit", ("continue", "continue"): "overflow", ("failed",): "failure"}
LOOPS = [
    case
    for case in _cases("loop")
    if case["bound"] == 2 and tuple(case["outcomes"]) in LOOP_MODES and case["entered"] == bool(case["outcomes"])
]


@pytest.mark.parametrize("case", MAPS, ids=lambda c: c["case_id"])
def test_compact_closed_map(case: dict[str, Any]) -> None:
    count = len(case["members"])
    mode = case["expansion"]
    fixture, result, callbacks = asyncio.run(
        _execute_membership(
            count if mode == "success" else 1 if mode == "failed" else 3,
            response_mode="missing" if mode == "failed" else "valid",
        )
    )
    (expansion,) = result.states[0].expansions
    assert expansion.status == ("closed" if mode == "success" else mode)
    assert len(expansion.members) == count
    terminals = {terminal.activation: terminal.category for terminal in result.record.terminals}
    assert sorted(terminals[member] for member in expansion.members) == sorted(
        status for _, status in case["terminals"]
    )
    (join,) = [entry for entry in result.states[0].entries if entry.template == fixture.join]
    assert join.status == case["expected"]
    items = [fact for fact in result.provenance if isinstance(fact.key, MapItemKey)]
    assert {fact.key.member for fact in items} == expansion.members
    assert len(callbacks[fixture.member].calls) == count
    assert len(callbacks[fixture.join].calls) == int(mode == "success")
    assert not result.requests.dispatches


@pytest.mark.parametrize("case", LOOPS, ids=lambda c: c["case_id"])
def test_compact_complete_loop(case: dict[str, Any]) -> None:
    mode = LOOP_MODES[tuple(case["outcomes"])]
    result, callbacks, starter, member, ordinary = asyncio.run(_execute_loop_resolution(mode))
    entries = result.states[0].entries
    iterations = sorted(
        (entry for entry in entries if entry.template == member), key=lambda entry: entry.activation.iteration
    )
    observed = tuple(
        "failed" if entry.status == "failure" else "continue" if entry.outcome == "again" else "exit"
        for entry in iterations
    )
    assert observed == tuple(case["outcomes"])
    assert [entry.activation.iteration for entry in iterations] == list(range(len(iterations)))
    (join,) = [entry for entry in entries if entry.template not in {starter, member, ordinary}]
    assert join.status == case["expected"][0]
    if case["expected"][1] is not None:
        exited = iterations[case["expected"][1]]
        (producer,) = [
            fact
            for fact in result.provenance
            if isinstance(fact.key, OperationOutputKey)
            and fact.key.activation == exited.activation
            and fact.key.port == "value"
        ]
        (call,) = callbacks[ordinary].calls
        assert call[0].inputs[0].artifact == producer.artifact
    else:
        assert not callbacks[ordinary].calls
    assert not result.requests.dispatches
