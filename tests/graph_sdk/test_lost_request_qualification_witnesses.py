# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Known physical usage cannot turn a lost request into qualified protection."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    StopConfirmed,
    TransportLost,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


@dataclass
class _LostTransport:
    settled: bool
    calls: int = 0
    closes: int = 0

    async def dispatch(self, request: DispatchEnvelope) -> TransportLost:
        self.calls += 1
        return TransportLost(
            settlement=ExternalSettlement(
                request=request.request,
                disposition="stopped",
                usage=ExactUsage(input_units=1, output_units=0),
                remote_stopped=True,
            )
            if self.settled
            else None,
        )

    async def cancel(self, request: object) -> StopConfirmed:
        return StopConfirmed(usage=ExactUsage(input_units=0, output_units=0))

    async def close(self) -> None:
        self.closes += 1


@pytest.mark.parametrize("settled", [False, True])
def test_lost_request_remains_unknown_with_or_without_exact_usage(settled: bool) -> None:
    transport = _LostTransport(settled)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(_execute_assessment(external=(transport, lease)))
    assert transport.calls == 1
    assert len(result.requests.dispatches) == 1
    assert result.requests.terminals[0].category == "lost"
    assert len(result.requests.settlements) == int(settled)
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert output.record.statuses[0].qualification == "unknown"
    assert "request_accounting" in output.targets[0].withholding
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    assert transport.calls == 1
    assert transport.closes == int(settled)
