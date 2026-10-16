# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remaining request discriminators from the frozen P7 corpus."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs, _RecoveringTransport


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_retry_reconciles_associations_independent_of_result_order(reverse: bool) -> None:
    transport = _RecoveringTransport(failure="retryable", reverse_results=reverse)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(
        _execute_assessment(external=(transport, lease), external_shared=True, target_count=2)
    )
    first, second = transport.envelopes
    first_order = tuple(item.association for item in first.associations)
    second_order = tuple(item.association for item in second.associations)
    assert len(set(first_order)) == 2 and first_order == second_order
    initial, retry = result.requests.dispatches
    assert initial.associations == retry.associations == frozenset(first_order)
    assert retry.associations == frozenset(reversed(first_order))  # Typed receipts erase list order, not membership.
    assert (initial.purpose, retry.purpose) == ("initial", "retry")
    terminal_order = tuple(item.association for item in result.requests.terminals[1].results)
    assert terminal_order == (tuple(reversed(second_order)) if reverse else second_order)
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == len(submissions) == 2
    assert all(item.qualification == "met" for item in output.record.statuses)
    assert all(not item.withholding for item in output.targets)
    assert len(result.requests.settlements) == 2
    assert transport.calls == 2 and transport.closes == 1
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    assert transport.calls == 2


def test_receipt_with_an_association_from_another_real_invocation_loses_attribution() -> None:
    results = []
    transports = []
    for _ in range(2):
        transport = _RecoveringTransport()
        lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
        results.append(asyncio.run(_execute_assessment(external=(transport, lease), target_count=2)))
        transports.append(transport)
    execution, result = results[0]
    _, foreign_result = results[1]
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    dispatch = result.requests.dispatches[0]
    foreign = foreign_result.requests.dispatches[0].associations
    assert not foreign & frozenset(
        association for item in result.requests.dispatches for association in item.associations
    )
    original = dispatch.associations
    object.__setattr__(dispatch, "associations", foreign)
    try:
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert not output.qualified
        assert all("inconsistent_attribution" in item.withholding for item in output.targets)
        assert all(item.qualification == "unknown" for item in output.record.statuses)
    finally:
        object.__setattr__(dispatch, "associations", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
    assert all(transport.calls == 2 and transport.closes == 1 for transport in transports)
