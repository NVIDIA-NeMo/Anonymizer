# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Ordinary versioned context preserves identity through distinct producers."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.evidence import evidence_revision_view
from anonymizer.engine.graph_sdk.executor import BoundInputKey, OperationOutputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


@pytest.mark.parametrize("corruption", ["none", "older-parent", "older-current"])
def test_versioned_context_crosses_an_ordinary_identity_hop(corruption: str) -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"b"}),
            versioned_context_ports=("b",),
            ordinary_context_hop=True,
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1 and not baseline.required_decisions
    assert len(result.record.terminals) == 2
    assert all(terminal.category == "success" for terminal in result.record.terminals)
    assert not result.pending_decisions and not any(fact.decision for fact in result.provenance)
    bound = sorted(
        (fact for fact in result.provenance if isinstance(fact.key, BoundInputKey)),
        key=lambda fact: fact.artifact.version,
    )
    older, latest = bound
    assert older.artifact.key == latest.artifact.key
    assert (older.artifact.version, latest.artifact.version) == (1, 2)
    (hop,) = (
        fact for fact in result.provenance if isinstance(fact.key, OperationOutputKey) and fact.key.port == "approved"
    )
    output = next(output for output in result.final_outputs if output.port == "result")
    final = next(fact for fact in result.provenance if fact.key == output.producer)
    assert isinstance(final.key, OperationOutputKey)
    assert isinstance(hop.key, OperationOutputKey)
    assert hop.key.activation != final.key.activation
    assert hop.artifact == latest.artifact and hop.parents == frozenset({latest.key})
    assert hop.key in final.parents and len(final.parents) == 2
    assert {ref.version for ref, _ in result.artifacts if ref.key == latest.artifact.key} == {1, 2}
    original_parents = hop.parents
    try:
        if corruption == "older-parent":
            object.__setattr__(hop, "parents", frozenset({older.key}))
            with pytest.raises(EffectRejected) as error:
                qualify(admitted=admitted, result=result, current=current, submissions=submissions)
            assert error.value.code is EffectCode.CONTRADICTORY
        elif corruption == "older-current":
            view = evidence_revision_view(
                admitted=admitted,
                result=result,
                artifacts=tuple(older.artifact if ref == latest.artifact else ref for ref in current.artifacts),
                configurations=current.configurations,
            )
            output = qualify(admitted=admitted, result=result, current=view, submissions=submissions)
            assert not output.qualified
            assert output.record.statuses[0].qualification == "unmet"
            assert "stale_evidence" in output.targets[0].withholding
    finally:
        object.__setattr__(hop, "parents", original_parents)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline
