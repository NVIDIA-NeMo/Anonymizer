# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared physical acceptance through local verification and pure qualification."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace

import pytest

from anonymizer.engine.graph_sdk.capabilities import ImplementationCapability
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import (
    AssociationResult,
    DispatchEnvelope,
    ExactUsage,
    ExternalSettlement,
    PortArtifact,
    RequestAssociation,
    StopConfirmed,
    TextArtifactValue,
    TransportSuccess,
    UnknownUsage,
)
from anonymizer.engine.graph_sdk.resources import ResourceLease
from tests.graph_sdk import evidence_fixtures as fixtures
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.workflow_construction import AdmittedActivationWorkflow


@dataclass
class _SharedProofTransport:
    malformed: bool
    unknown: bool
    bad: RequestAssociation | None = None
    groups: list[int] = field(default_factory=list)
    closes: int = 0

    async def dispatch(self, request: DispatchEnvelope) -> TransportSuccess:
        self.groups.append(len(request.associations))
        if self.bad is None:
            self.bad = request.associations[0].association
        rows = tuple(
            AssociationResult(
                association=item.association,
                outcome="ok",
                consumed_context_ports=frozenset(),
                outputs=(
                    PortArtifact(
                        port="context",
                        artifact_type=item.inputs[0].artifact_type,
                        artifact=None,
                        value=TextArtifactValue(text=chr(0xD800))
                        if self.malformed and item.association == self.bad
                        else item.inputs[0].value,
                    ),
                ),
            )
            for item in request.associations
        )
        return TransportSuccess(
            results=rows,
            settlement=ExternalSettlement(
                request=request.request,
                disposition="completed",
                usage=UnknownUsage() if self.unknown else ExactUsage(input_units=1, output_units=1),
                remote_stopped=True,
            ),
        )

    async def cancel(self, request: object) -> StopConfirmed:
        return StopConfirmed(usage=UnknownUsage())

    async def close(self) -> None:
        self.closes += 1


@pytest.mark.parametrize("malformed", [False, True])
@pytest.mark.parametrize("unknown", [False, True])
def test_shared_peer_survives_through_verification_and_qualification(
    monkeypatch: pytest.MonkeyPatch,
    malformed: bool,
    unknown: bool,
) -> None:
    original = fixtures._capability

    def capability(workflow: AdmittedActivationWorkflow, *, external: bool = False) -> ImplementationCapability:
        value = original(workflow, external=external)
        return replace(value, attribution="keyed_shared_request") if external else value

    monkeypatch.setattr(fixtures, "_capability", capability)
    transport = _SharedProofTransport(malformed, unknown)
    lease = ResourceLease.create(owner="sdk", safe_detachment="forbidden", handle=transport)
    execution, result = asyncio.run(fixtures._execute_assessment(target_count=2, external=(transport, lease)))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert transport.groups == ([2, 1] if malformed else [2])
    assert len(result.assessments) == (1 if malformed else 2)
    assert len(output.qualified) == (0 if unknown else 1 if malformed else 2)
    assert result.requests.dispatched_count == len(transport.groups)
    assert len(result.requests.settlements) == len(transport.groups) and transport.closes == 1
    assert not result.requests.remote_outstanding
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == output
    assert transport.groups == ([2, 1] if malformed else [2])
