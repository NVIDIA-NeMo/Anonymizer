# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evidence ordering follows admitted ports, not callback allocation order."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.evidence import verify_evidence
from anonymizer.engine.graph_sdk.executor import LocalCompleted
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import AssociationInput
from anonymizer.graph.workflow import CoverageAtom
from tests.graph_sdk import test_qualification_subject_context as fixture
from tests.graph_sdk.test_qualification import _inputs


@pytest.mark.parametrize("reverse_outputs", [False, True])
@pytest.mark.parametrize("reverse_submissions", [False, True])
def test_verified_order_uses_admitted_target_and_port_ordinals(
    reverse_outputs: bool, reverse_submissions: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = fixture._SeparateAssessment.run

    async def run(self: fixture._SeparateAssessment, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        result = await original(self, request)
        if reverse_outputs:
            result = replace(
                result, results=tuple(replace(item, outputs=tuple(reversed(item.outputs))) for item in result.results)
            )
        return result

    monkeypatch.setattr(fixture._SeparateAssessment, "run", run)
    coverage = frozenset({CoverageAtom(kind="field", name="K0")})
    execution, result = asyncio.run(
        fixture._execute_separate_subject_context(
            coverage=coverage,
            additional_coverage=coverage,
            requirement_coverage=coverage,
            target_labels=("A", "B"),
            environment=True,
            expose_evidence=False,
        )
    )
    admitted, current, submissions = _inputs(execution, result)
    if reverse_submissions:
        submissions = tuple(reversed(submissions))
    verified = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    ports = {(port.activation, port.artifact): port for port in result.ports if port.role == "evidence"}
    observed = [
        (ports[item.activation, item.reference.artifact].target, ports[item.activation, item.reference.artifact].port)
        for item in verified
    ]
    targets = [item.target for item in execution.context.prepared.target_occurrences]
    assert observed == [(target, port) for target in targets for port in ("evidence", "evidence2")]
    for index in (0, 2):
        first, second = verified[index : index + 2]
        assert (first.reference.artifact.key > second.reference.artifact.key) is reverse_outputs
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert output.verified == verified
    assert len(output.qualified) == 2
