# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real product boundaries against the adopted finite foundation predicates."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.data import AtomicGroup, CoherenceScope, DatumDependency, SourceRelation, SourceView
from anonymizer.engine.graph_sdk.requests import (
    AcceptResult,
    AssociationResult,
    Dispatch,
    ExactUsage,
    ExternalSettlement,
    InvocationRequestScope,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    RequestPolicyBinding,
    Reserve,
    advance_requests,
    bind_request_policies,
    initialize_requests,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease, close_resource
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import ContractViolation, InvocationId, PlanId
from tests.graph_sdk.test_data_graph import _graph_with_texts, _validate
from tests.graph_sdk.test_requests import _association, _policy

CASES = json.loads((Path(__file__).parent / "reference/foundation_v2/cases.json").read_bytes())


def _cases(*families: str) -> list[dict[str, Any]]:
    return [
        case
        for case in CASES
        if case["family"] in families
        and (case["family"] != "ownership" or all(start < end for _, start, end in case["windows"]))
    ]


@pytest.mark.parametrize("case", _cases("dag", "window", "ownership"), ids=lambda c: c["case_id"])
def test_compact_data_boundaries(case: dict[str, Any]) -> None:
    accepted = True
    try:
        if case["family"] == "dag":
            graph, ids = _graph_with_texts("same", "same", "same")
            _validate(
                graph,
                ids,
                dependencies=tuple(DatumDependency(prerequisite=ids[a], dependent=ids[b]) for a, b in case["edges"]),
            )
        elif case["family"] == "window":
            graph, (source, derived) = _graph_with_texts("aé", case["derived"])
            _validate(
                graph,
                (),
                source_relations=(
                    SourceRelation(
                        derived=derived, view=SourceView(source=source, start=case["start"], end=case["end"])
                    ),
                ),
            )
        elif case["family"] == "ownership":
            windows = case["windows"]
            graph, ids = _graph_with_texts("aé", "aé", *("aé"[start:end] for _, start, end in windows))
            _validate(
                graph,
                ids[2:],
                source_relations=tuple(
                    SourceRelation(derived=target, view=SourceView(source=ids[root], start=start, end=end))
                    for target, (root, start, end) in zip(ids[2:], windows, strict=True)
                ),
            )
    except ContractViolation:
        accepted = False
    assert accepted is case["expected"]


@pytest.mark.parametrize("case", _cases("groups"), ids=lambda c: c["case_id"])
@pytest.mark.parametrize("atomic", [False, True])
def test_compact_group_partitions(case: dict[str, Any], atomic: bool) -> None:
    graph, ids = _graph_with_texts("a", "b", "foreign")
    accepted = True
    try:
        members = [tuple(ids[i] for i in group) for group in case["groups"]]
        if atomic:
            _validate(graph, ids[:2], atomic=tuple(AtomicGroup(members=group) for group in members))
        else:
            _validate(graph, ids[:2], coherence=tuple(CoherenceScope(members=group) for group in members))
    except ContractViolation:
        accepted = False
    assert accepted is case["expected"]


def _request_state(budget: int):
    invocation = InvocationId.new(plan=PlanId.new())
    scope = InvocationRequestScope(invocation=invocation)
    policy = _policy()
    associations = tuple(_association(invocation, index) for index in range(3))
    state = initialize_requests(scope=scope, hard_limit=budget, policies=frozenset({policy}))
    for association in associations:
        state = bind_request_policies(
            state=state, binding=RequestPolicyBinding.create(association=association, policies=frozenset({policy}))
        )
    return state, PhysicalRequestId.new(scope=scope), associations, policy


@pytest.mark.parametrize("case", _cases("requests"), ids=lambda c: c["case_id"])
def test_compact_request_causal_histories(case: dict[str, Any]) -> None:
    state, request, associations, policy = _request_state(case["budget"])
    legal = True
    try:
        for event in case["events"]:
            if event["kind"] == "reserve":
                command = Reserve(
                    request=request, purpose="initial", associations=frozenset({associations[0]}), policy=policy
                )
            elif event["kind"] == "dispatch":
                command = Dispatch(request=request)
            elif event["kind"] == "terminal":
                command = MarkLost(request=request)
            else:
                command = ObserveSettlement(
                    settlement=ExternalSettlement(
                        request=request,
                        disposition="completed",
                        usage=ExactUsage(input_units=event["usage"], output_units=0),
                        remote_stopped=event["stopped"],
                    )
                )
            state = advance_requests(state=state, event=command)
            if state.denials:
                legal = False
                break
    except EffectRejected:
        legal = False
    assert legal is case["expected"]
    if legal:
        (terminal,), (settlement,) = state.terminals, state.settlements
        assert isinstance(settlement.usage, ExactUsage)
        assert [terminal.category, settlement.usage.input_units, settlement.remote_stopped] == case["projection"]
        assert len(state.dispatches) == 1 and not state.reserved


@pytest.mark.parametrize("case", _cases("associations"), ids=lambda c: c["case_id"])
def test_compact_request_association_coverage(case: dict[str, Any]) -> None:
    state, request, associations, policy = _request_state(1)
    state = advance_requests(
        state=state,
        event=Reserve(request=request, purpose="initial", associations=frozenset(associations[:2]), policy=policy),
    )
    state = advance_requests(state=state, event=Dispatch(request=request))
    state = advance_requests(
        state=state,
        event=AcceptResult(
            request=request,
            results=tuple(
                AssociationResult(
                    association=associations[i], outcome="ok", outputs=(), consumed_context_ports=frozenset()
                )
                for i in case["returned"]
            ),
        ),
    )
    (terminal,) = state.terminals
    assert [terminal.category, None, False] == case["expected"]
    assert not state.settlements and len(state.dispatches) == 1


@dataclass
class _Counter:
    closes: int = 0

    async def close(self) -> None:
        self.closes += 1


@pytest.mark.parametrize("case", [c for c in _cases("resource") if not any(c["args"][1:3])], ids=lambda c: c["case_id"])
def test_compact_idle_resource_ownership(case: dict[str, Any]) -> None:
    owned, _, _, detached = case["args"]
    handle = _Counter()
    lease = ResourceLease.create(
        owner="sdk" if owned else "caller",
        safe_detachment="independent_after_dispatch" if detached else "forbidden",
        handle=handle,
    )
    fact = asyncio.run(close_resource(lease))
    assert (fact.disposition == "closed") is case["expected"]
    assert handle.closes == int(case["expected"])
