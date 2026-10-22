# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact retained root, final output, and membership discriminators."""

from __future__ import annotations

import asyncio

from anonymizer.engine.graph_sdk.executor import RootInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_final_output_projects_its_own_root_input() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    final = result.final_outputs[0]
    other = next(
        fact
        for fact in result.provenance
        if isinstance(fact.key, RootInputKey) and fact.artifact != final.candidate.artifact
    )
    assert final.producer != other.key


def test_execution_only_open_root_membership_is_pending_not_assessed() -> None:
    execution, result = asyncio.run(_execute_assessment(execution_only=True))
    admitted, current, submissions = _inputs(execution, result)
    (root,) = result.record.memberships
    assert root.parent is None and root.closed


def test_owned_entry_is_reserved_and_a_retained_member() -> None:
    execution, result = asyncio.run(_execute_assessment())
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    entry = next(iter(result.states[0].entries))
    assert any(seed.activation == entry.activation for seed in result.states[0].reservations)
    assert any(entry.activation in membership.members for membership in result.record.memberships)
