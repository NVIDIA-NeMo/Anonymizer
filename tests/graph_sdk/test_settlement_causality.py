# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Settlement requires historical dispatch, including after cancellation or Lost."""

from __future__ import annotations

from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.requests import (
    Dispatch,
    ExactUsage,
    ExternalSettlement,
    InvocationRequestScope,
    MarkLost,
    ObserveSettlement,
    PhysicalRequestId,
    RejectSettlement,
    RequestCancel,
    Reserve,
    ScopeCancel,
    StopAcknowledged,
    UnknownUsage,
    advance_requests,
    request_receipt,
)
from anonymizer.graph._values import InvocationId, PlanId
from tests.graph_sdk.test_compact_foundation import _request_state


def _settlement(request: PhysicalRequestId) -> ExternalSettlement:
    return ExternalSettlement(
        request=request, disposition="completed", usage=ExactUsage(input_units=3, output_units=0), remote_stopped=True
    )


@pytest.mark.parametrize("rejected", [False, True])
@pytest.mark.parametrize("phase", ["never", "reserved", "cancelled", "scope_cancelled"])
def test_predispatch_settlement_rejects_without_changing_authority(rejected: bool, phase: str) -> None:
    state, request, associations, policy = _request_state(1)
    if phase != "never":
        state = advance_requests(
            state=state,
            event=Reserve(request=request, purpose="initial", associations=frozenset({associations[0]}), policy=policy),
        )
    if phase == "cancelled":
        state = advance_requests(state=state, event=RequestCancel(request=request))
    elif phase == "scope_cancelled":
        state = advance_requests(state=state, event=ScopeCancel())
    before = request_receipt(state)
    reservations = state.reserved
    event = RejectSettlement(request=request) if rejected else ObserveSettlement(settlement=_settlement(request))
    with pytest.raises(EffectRejected) as failure:
        advance_requests(state=state, event=event)
    assert failure.value.code is EffectCode.MISSING
    assert request_receipt(state) == before and state.reserved == reservations
    assert not state.settlements and not state.defects and not state.dispatches


@pytest.mark.parametrize("rejected", [False, True])
def test_foreign_settlement_precedes_missing_dispatch(rejected: bool) -> None:
    state, _, _, _ = _request_state(1)
    foreign = PhysicalRequestId.new(scope=InvocationRequestScope(invocation=InvocationId.new(plan=PlanId.new())))
    event = RejectSettlement(request=foreign) if rejected else ObserveSettlement(settlement=_settlement(foreign))
    with pytest.raises(EffectRejected) as failure:
        advance_requests(state=state, event=event)
    assert failure.value.code is EffectCode.FOREIGN_OWNER
    assert not state.settlements and not state.defects


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("rejected", [False, True])
def test_late_settlement_preserves_dispatched_terminal(cancelled: bool, rejected: bool) -> None:
    state, request, associations, policy = _request_state(1)
    state = advance_requests(
        state=state,
        event=Reserve(request=request, purpose="initial", associations=frozenset({associations[0]}), policy=policy),
    )
    state = advance_requests(state=state, event=Dispatch(request=request))
    event = StopAcknowledged(request=request, usage=UnknownUsage()) if cancelled else MarkLost(request=request)
    state = advance_requests(state=state, event=event)
    terminals = state.terminals
    first = RejectSettlement(request=request) if rejected else ObserveSettlement(settlement=_settlement(request))
    settled = advance_requests(state=state, event=first)
    assert settled.terminals == terminals and terminals[0].category == ("cancelled" if cancelled else "lost")
    assert len(settled.dispatches) == 1 and not settled.local_in_flight
    assert settled.dispatched == frozenset({request})
    assert advance_requests(state=settled, event=first) == settled
    if rejected:
        assert isinstance(settled.settlements[0].usage, UnknownUsage)
        assert (request in settled.remote_outstanding) is (not cancelled)
        assert any(defect.code == "invalid_settlement" for defect in settled.defects)
    else:
        assert settled.settlements == (_settlement(request),) and not settled.remote_outstanding
    later = ObserveSettlement(settlement=replace(_settlement(request), usage=ExactUsage(input_units=9, output_units=0)))
    conflict = advance_requests(state=settled, event=later)
    assert conflict.settlements == settled.settlements and conflict.terminals == terminals
    assert any(defect.code == "conflicting_settlement" for defect in conflict.defects)
