# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Values from two real runs never authenticate each other (FOREIGN_OWNER), with no retained-value tampering.

These replace the reference corpus' retained-record corruptions record/{plan,graph,invocation} and
assessment/caller_{copy,replacement}, whose expected answer is ``foreign_owner``: the same mixing is reached here
by handing one run's legitimately produced values to another run's verbs.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    admit_qualification,
    evidence_revision_view,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import start_execution
from anonymizer.engine.graph_sdk.extensions import bind_projection_source
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment, _qualification_limits
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.typed_artifact_fixtures import span_graph


def _view(admitted: Any, result: Any, artifacts: tuple[Any, ...] = ()) -> Any:
    return evidence_revision_view(admitted=admitted, result=result, artifacts=artifacts, configurations=())


@pytest.fixture(scope="module")
def two_plans() -> tuple[Any, Any]:
    """Two independent prepared plans (distinct data graphs), each run once."""
    return asyncio.run(_execute_assessment()), asyncio.run(_execute_assessment())


@pytest.fixture(scope="module")
def two_invocations() -> tuple[Any, Any, Any]:
    """One admitted plan run twice: same plan and graph, two invocations."""

    async def run() -> tuple[Any, Any, Any]:
        execution, capabilities, services, _ = span_graph()
        first = await (await start_execution(admitted=execution, capabilities=capabilities, services=services)).wait()
        second = await (await start_execution(admitted=execution, capabilities=capabilities, services=services)).wait()
        return execution, first, second

    return asyncio.run(run())


def _admitted(execution: Any) -> Any:
    return admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )


@pytest.mark.parametrize("verb", ["view", "verify", "qualify"])
def test_result_of_another_plan_is_foreign(two_plans: tuple[Any, Any], verb: str) -> None:
    # record/plan and record/graph: the other run was prepared from a different plan over a different graph.
    (execution, result), (_, other) = two_plans
    assert other.record.plan != result.record.plan and other.record.graph != result.record.graph
    admitted = _admitted(execution)
    with pytest.raises(EffectRejected) as error:
        if verb == "view":
            _view(admitted, other)
        elif verb == "verify":
            verify_evidence(
                admitted=admitted, result=other, submissions=(AssessmentSubmission(fact=other.assessments[0]),)
            )
        else:
            qualify(admitted=admitted, result=other, current=_view(admitted, result), submissions=())
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_second_invocation_of_the_same_plan_cannot_borrow_the_first_ones_view(
    two_invocations: tuple[Any, Any, Any],
) -> None:
    # record/invocation: same plan and graph, different invocation.
    execution, first, second = two_invocations
    assert first.record.plan == second.record.plan and first.record.invocation != second.record.invocation
    admitted, current, submissions = _inputs(execution, first)
    qualify(admitted=admitted, result=first, current=current, submissions=submissions)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=second, current=current, submissions=submissions)
    assert error.value.code is EffectCode.FOREIGN_OWNER
    # Each invocation still qualifies on its own view.
    own_admitted, own_current, own_submissions = _inputs(execution, second)
    qualify(admitted=own_admitted, result=second, current=own_current, submissions=own_submissions)


@pytest.mark.parametrize("verb", ["verify", "qualify"])
def test_assessment_fact_from_another_run_is_foreign(two_plans: tuple[Any, Any], verb: str) -> None:
    # assessment/caller_copy and caller_replacement: a fact the result did not retain is submitted.
    (execution, result), (_, other) = two_plans
    admitted = _admitted(execution)
    submissions = (AssessmentSubmission(fact=other.assessments[0]),)
    with pytest.raises(EffectRejected) as error:
        if verb == "verify":
            verify_evidence(admitted=admitted, result=result, submissions=submissions)
        else:
            qualify(admitted=admitted, result=result, current=_view(admitted, result), submissions=submissions)
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_revision_artifact_from_another_run_is_foreign(two_plans: tuple[Any, Any]) -> None:
    (execution, result), (_, other) = two_plans
    with pytest.raises(EffectRejected) as error:
        _view(_admitted(execution), result, artifacts=(other.artifacts[0][0],))
    assert error.value.code is EffectCode.FOREIGN_OWNER


def test_projection_cannot_pair_a_result_with_another_runs_qualification(
    two_invocations: tuple[Any, Any, Any],
) -> None:
    execution, first, second = two_invocations
    admitted, current, submissions = _inputs(execution, second)
    other_qualification = qualify(admitted=admitted, result=second, current=current, submissions=submissions)
    with pytest.raises(EffectRejected) as error:
        bind_projection_source(execution=first, qualification=other_qualification)
    assert error.value.code is EffectCode.FOREIGN_OWNER
