# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every public graph verb rejects caller input at its own boundary, before any effect.

Expected codes follow the closed code meanings: a wrong input type is ``invalid_type``; a value of the right type
outside its documented domain is ``invalid_value``; a declaration the admitted plan does not hold is
``unsupported``; a declaration repeated or absent is ``duplicate`` / ``missing``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk import (
    context,
    data,
    evidence,
    executor,
    extensions,
    inspection,
    observations,
    preparation,
    qualification,
    requests,
    resources,
)
from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk._execution_values import AssessmentFinding
from anonymizer.engine.graph_sdk.capabilities import PreparationCode, PreparationRejected
from anonymizer.graph._values import ContractViolation, ValidationCode
from tests.graph_sdk.evidence_fixtures import _execute_assessment, _qualification_limits

_WRONG: Any = object()
_WRONG_OWNER: Any = "nobody"

VERBS: dict[str, tuple[Callable[[], object], type[Exception], object]] = {
    "DataGraph.add_text": (
        lambda: data.DataGraph.new().add_text(_WRONG),
        ContractViolation,
        ValidationCode.INVALID_TYPE,
    ),
    "prepare": (
        lambda: preparation.prepare(
            data=_WRONG,
            workflow=_WRONG,
            activation_limits=_WRONG,
            bound_inputs=(),
            configuration=_WRONG,
            selections=(),
            capabilities=(),
            limits=_WRONG,
        ),
        PreparationRejected,
        PreparationCode.INVALID_TYPE,
    ),
    "admit_context_plan": (
        lambda: context.admit_context_plan(prepared=_WRONG, bound_context=None),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "admit_execution_plan": (
        lambda: executor.admit_execution_plan(
            context=_WRONG, capabilities=(), policies=(), assessment_productions=(), assessment_limits=_WRONG
        ),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "start_execution": (
        lambda: asyncio.run(executor.start_execution(admitted=_WRONG, capabilities=(), services=_WRONG)),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "admit_qualification": (
        lambda: evidence.admit_qualification(execution=_WRONG, productions=(), limits=_WRONG),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "evidence_revision_view": (
        lambda: evidence.evidence_revision_view(admitted=_WRONG, result=_WRONG, artifacts=(), configurations=()),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "verify_evidence": (
        lambda: evidence.verify_evidence(admitted=_WRONG, result=_WRONG, submissions=()),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "AssessmentSubmission": (
        lambda: evidence.AssessmentSubmission(fact=_WRONG),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "qualify": (
        lambda: qualification.qualify(admitted=_WRONG, result=_WRONG, current=_WRONG, submissions=()),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "bind_projection_source": (
        lambda: extensions.bind_projection_source(execution=_WRONG, qualification=_WRONG),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "project_protected": (lambda: extensions.project_protected(source=_WRONG), EffectRejected, EffectCode.INVALID_TYPE),
    "project_content": (
        lambda: extensions.project_content(source=_WRONG, kind="research", artifacts=()),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "explain": (lambda: inspection.explain(source=_WRONG), EffectRejected, EffectCode.INVALID_TYPE),
    "inspect": (lambda: inspection.inspect(source=_WRONG), EffectRejected, EffectCode.INVALID_TYPE),
    "diagnose": (lambda: inspection.diagnose(source=_WRONG), EffectRejected, EffectCode.INVALID_TYPE),
    "diagnose_observations": (
        lambda: inspection.diagnose_observations(observations=_WRONG, limits=_WRONG),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "encode_inspection": (
        lambda: inspection.encode_inspection(value=_WRONG, limits=_WRONG),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "bind_inspection_source": (
        lambda: inspection.bind_inspection_source(prepared=_WRONG, limits=_WRONG),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "begin_observations": (
        lambda: observations.begin_observations(limits=_WRONG, clock=_WRONG, sink=None),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
    "ResourceLease.create": (
        lambda: resources.ResourceLease.create(owner=_WRONG_OWNER, safe_detachment="forbidden", handle=_WRONG),
        EffectRejected,
        EffectCode.INVALID_VALUE,
    ),
    "RequestPolicyBinding.create": (
        lambda: requests.RequestPolicyBinding.create(association=_WRONG, policies=frozenset()),
        EffectRejected,
        EffectCode.INVALID_TYPE,
    ),
}


@pytest.mark.parametrize("verb", sorted(VERBS))
def test_public_verb_rejects_wrong_input_at_its_boundary(verb: str) -> None:
    call, error_type, code = VERBS[verb]
    with pytest.raises(error_type) as error:
        call()
    assert getattr(error.value, "code", None) == code


@pytest.fixture(scope="module")
def run() -> Any:
    return asyncio.run(_execute_assessment(nested=True))


def _admit(execution: Any, productions: tuple[Any, ...]) -> Any:
    return evidence.admit_qualification(execution=execution, productions=productions, limits=_qualification_limits())


def test_admitted_productions_accept_the_plans_own_declarations(run: Any) -> None:
    execution, _ = run
    assert _admit(execution, execution.assessment_productions).productions == execution.assessment_productions


def test_repeated_production_is_a_duplicate(run: Any) -> None:
    execution, _ = run
    with pytest.raises(EffectRejected) as error:
        _admit(execution, execution.assessment_productions * 2)
    assert error.value.code is EffectCode.DUPLICATE


def test_omitting_a_required_production_is_missing(run: Any) -> None:
    execution, _ = run
    with pytest.raises(EffectRejected) as error:
        _admit(execution, ())
    assert error.value.code is EffectCode.MISSING


def test_production_the_plan_did_not_admit_is_unsupported(run: Any) -> None:
    execution, _ = run
    (production,) = execution.assessment_productions
    undeclared = replace(production, supported_findings=frozenset({AssessmentFinding(status="unsatisfied", code="x")}))
    with pytest.raises(EffectRejected) as error:
        _admit(execution, (undeclared,))
    assert error.value.code is EffectCode.UNSUPPORTED


@pytest.mark.parametrize("member", ["promise", "evidence_port"])
def test_production_naming_an_undeclared_promise_or_port_is_unsupported(run: Any, member: str) -> None:
    execution, _ = run
    (production,) = execution.assessment_productions
    with pytest.raises(EffectRejected) as error:
        _admit(execution, (replace(production, **{member: "u1-undeclared"}),))
    assert error.value.code is EffectCode.UNSUPPORTED


def test_submitted_fact_without_an_admitted_production_is_unsupported() -> None:
    # Execution-only admission may legitimately admit no productions; a retained fact then has none to verify.
    execution, result = asyncio.run(_execute_assessment(execution_only=True))
    admitted = _admit(execution, ())
    with pytest.raises(EffectRejected) as error:
        evidence.verify_evidence(
            admitted=admitted, result=result, submissions=(evidence.AssessmentSubmission(fact=result.assessments[0]),)
        )
    assert error.value.code is EffectCode.UNSUPPORTED
