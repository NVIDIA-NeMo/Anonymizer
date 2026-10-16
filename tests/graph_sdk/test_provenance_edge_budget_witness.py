# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Count actual provenance edges independently from retained version rows."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.evidence import admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.executor import BoundInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_five_provenance_facts_have_exactly_three_parent_edges() -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a"}),
            versioned_context_ports=("a",),
            provenance_edge_limit=3,
        )
    )
    assert len(result.provenance) == 5
    bound = [fact for fact in result.provenance if isinstance(fact.key, BoundInputKey)]
    assert len(bound) == 2 and {fact.artifact.version for fact in bound} == {1, 2}
    assert all(not fact.parents for fact in bound)
    assert sum(len(fact.parents) for fact in result.provenance) == 3
    admitted, current, submissions = _inputs(execution, result, latest=True, provenance_edges=3)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    smaller = admit_qualification(
        execution=execution,
        productions=execution.assessment_productions,
        limits=replace(admitted.limits, max_provenance_edges=2),
    )
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=smaller,
            result=result,
            artifacts=tuple(current.artifacts),
            absences=tuple(current.absences),
            configurations=current.configurations,
            state=current.state,
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED
    with pytest.raises(EffectRejected) as error:
        asyncio.run(
            _execute_separate_subject_context(
                versioned_context_dependencies=frozenset({"a"}),
                versioned_context_ports=("a",),
                provenance_edge_limit=2,
            )
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED
