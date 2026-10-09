# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Extra same-meaning promises are rejected before operation callbacks."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.evidence import admit_qualification
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph.workflow import AdmittedWorkflow, admit_static_workflow
from tests.graph_sdk import evidence_fixtures as fixture
from tests.graph_sdk.qualification_fixtures import _inputs


class _Checked(Exception):
    pass


def test_same_node_same_meaning_unrelated_subject_promise_is_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    execution, result = asyncio.run(fixture._execute_assessment())
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    callbacks = 0

    async def unexpected_callback(*args: Any, **kwargs: Any) -> Any:
        nonlocal callbacks
        callbacks += 1
        raise AssertionError("Admission must reject before callback execution")

    def static(**kwargs: Any) -> AdmittedWorkflow:
        for operation in (kwargs["interface"], *(node.operation for node in kwargs["nodes"])):
            object.__setattr__(
                operation,
                "outcomes",
                tuple(
                    replace(
                        outcome,
                        evidence=frozenset(
                            replace(promise, subject_port="input") if promise.name == "partial" else promise
                            for promise in outcome.evidence
                        ),
                    )
                    for outcome in operation.outcomes
                ),
            )
        return admit_static_workflow(**kwargs)

    async def before_execution(**kwargs: Any) -> Any:
        plan = kwargs["admitted"]
        assert len(plan.assessment_productions) == 2
        assert len({row.node for row in plan.assessment_productions}) == 1
        operation = next(iter(plan.context.prepared.workflow.workflow.nodes)).operation
        promises = operation.outcomes[0].evidence
        assert len({promise.meaning for promise in promises}) == 1
        assert len({promise.subject_port for promise in promises}) == 2
        with pytest.raises(EffectRejected) as error:
            admit_qualification(
                execution=plan, productions=plan.assessment_productions, limits=fixture._qualification_limits()
            )
        assert error.value.code.value == "unsupported"
        assert callbacks == 0
        raise _Checked

    monkeypatch.setattr(fixture, "admit_static_workflow", static)
    monkeypatch.setattr(fixture, "start_execution", before_execution)
    monkeypatch.setattr(fixture._AssessingLocal, "run", unexpected_callback)
    with pytest.raises(_Checked):
        asyncio.run(fixture._execute_assessment(partial_assessment=True))
    assert callbacks == 0
