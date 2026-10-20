# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Foreign callback values must not erase prior effects or skip safe cleanup."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import cast

import pytest

from anonymizer.engine.graph_sdk.executor import LocalAssessmentResult, LocalCompleted
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import AssociationInput, AssociationResult, ExactUsage, TextArtifactValue
from anonymizer.engine.graph_sdk.resources import ResourceLease, ResourceOwner
from tests.graph_sdk import evidence_fixtures as fixtures
from tests.graph_sdk.qualification_fixtures import _inputs, _RecoveringTransport


@pytest.mark.parametrize(
    "kind", ["valid", "unicode", "empty", "outer", "member", "collection", "assessment", "surrogate"]
)
@pytest.mark.parametrize("owner", ["sdk", "caller"])
def test_callback_ingress_preserves_prior_effects_and_resource_ownership(
    monkeypatch: pytest.MonkeyPatch, kind: str, owner: ResourceOwner
) -> None:
    transport = _RecoveringTransport()
    lease = ResourceLease.create(owner=owner, safe_detachment="forbidden", handle=transport)
    original = fixtures._AssessingLocal.run

    async def callback(self: fixtures._AssessingLocal, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        if kind == "outer":
            return cast(LocalCompleted, object())
        if kind == "member":
            return LocalCompleted(results=cast(tuple[AssociationResult, ...], (object(),)))
        if kind == "collection":
            return LocalCompleted(results=cast(tuple[AssociationResult, ...], None))
        completed = await original(self, request)
        if kind == "assessment":
            return replace(completed, assessments=cast(tuple[LocalAssessmentResult, ...], (object(),)))
        if kind in {"unicode", "empty", "surrogate"}:
            row = completed.results[0]
            text = {"unicode": "café 日本語 🌍", "empty": "", "surrogate": chr(0xD800)}[kind]
            output = replace(row.outputs[0], value=TextArtifactValue(text=text))
            return replace(completed, results=(replace(row, outputs=(output, *row.outputs[1:])),))
        return completed

    monkeypatch.setattr(fixtures._AssessingLocal, "run", callback)
    execution, result = asyncio.run(fixtures._execute_assessment(external=(transport, lease)))
    assert transport.calls == 1
    assert transport.closes == (1 if owner == "sdk" else 0)
    assert result.requests.dispatched_count == 1
    assert result.requests.terminals[0].category == "success"
    assert result.requests.settlements[0].usage == ExactUsage(input_units=1, output_units=1)
    assert not result.requests.local_in_flight
    assert not result.requests.remote_outstanding
    valid = kind in {"valid", "unicode", "empty"}
    assert all(terminal.category == "success" for terminal in result.record.terminals) == valid
    admitted, current, submissions = _inputs(execution, result)
    qualified = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert bool(qualified.qualified) == valid
