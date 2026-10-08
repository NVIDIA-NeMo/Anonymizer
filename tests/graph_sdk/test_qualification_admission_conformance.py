# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate reference admission negatives through public SDK constructors."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.engine.graph_sdk.capabilities import PreparationRejected
from anonymizer.engine.graph_sdk.evidence import QualificationLimits, admit_qualification
from anonymizer.engine.graph_sdk.executor import AdmittedExecutionPlan, admit_execution_plan
from anonymizer.graph._values import ContractViolation
from anonymizer.graph.workflow import AdmittedWorkflow, CoverageAtom, admit_static_workflow
from tests.graph_sdk import test_qualification_subject_context as fixture
from tests.graph_sdk.test_qualification_production_conformance import CORPUS

CASE_IDS = {
    "admission/duplicate_production",
    "admission/unsupported_outcome",
    "admission/missing_promise",
    "admission/evidence_dependency_consumed_mismatch",
    "admission/fixed_point",
}


@pytest.mark.parametrize(
    "case", [case for case in CORPUS if case["case_id"] in CASE_IDS], ids=lambda case: case["case_id"]
)
def test_qualification_declaration_rejected_before_callbacks(
    case: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = CORPUS[0]["declaration"]
    declaration = case["declaration"]
    assert not case["events"]
    changed = {key for key in declaration if declaration[key] != baseline[key]}
    assert len(changed) == 1
    calls = 0

    async def forbidden_callback(self: object, request: object) -> None:
        nonlocal calls
        calls += 1
        raise AssertionError("Admission-negative case invoked implementation")

    def static(**kwargs: Any) -> AdmittedWorkflow:
        if changed == {"requirements"}:
            (raw,) = declaration["requirements"]
            (old,) = baseline["requirements"]
            (requirement,) = kwargs["protection"]
            assert {key: value for key, value in raw.items() if key not in {"meaning", "outcome"}} == {
                key: value for key, value in old.items() if key not in {"meaning", "outcome"}
            }
            kwargs["protection"] = (
                replace(
                    requirement,
                    outcome=raw["outcome"],
                    meaning=requirement.meaning if raw["meaning"] == old["meaning"] else raw["meaning"],
                ),
            )
        elif changed == {"output_dependencies"}:
            altered = [
                item for item in declaration["output_dependencies"] if item not in baseline["output_dependencies"]
            ]
            assert len(altered) == 1 and altered[0]["node"] == "N" and altered[0]["port"] == "evidence"
            (node,) = kwargs["nodes"]
            kwargs["nodes"] = (
                replace(
                    node,
                    operation=replace(
                        node.operation,
                        output_dependencies=tuple(
                            replace(dep, inputs=frozenset(altered[0]["inputs"])) if dep.output == "evidence" else dep
                            for dep in node.operation.output_dependencies
                        ),
                    ),
                ),
            )
        return admit_static_workflow(**kwargs)

    def execution(**kwargs: Any) -> AdmittedExecutionPlan:
        if changed == {"productions"}:
            assert declaration["productions"] == baseline["productions"] * 2
            kwargs["assessment_productions"] *= 2
            kwargs["assessment_limits"] = replace(
                kwargs["assessment_limits"], max_productions=declaration["limits"]["max_productions"]
            )
        return admit_execution_plan(**kwargs)

    async def before_execution(**kwargs: Any) -> None:
        assert changed == {"limits"}
        admitted = kwargs["admitted"]
        limits = QualificationLimits(
            **{field.name: declaration["limits"][field.name] for field in fields(QualificationLimits)}
        )
        admit_qualification(execution=admitted, productions=admitted.assessment_productions, limits=limits)
        raise AssertionError("Admission-negative declaration was admitted")

    monkeypatch.setattr(fixture._SeparateAssessment, "run", forbidden_callback)
    monkeypatch.setattr(fixture, "admit_static_workflow", static)
    monkeypatch.setattr(fixture, "admit_execution_plan", execution)
    monkeypatch.setattr(fixture, "start_execution", before_execution)
    try:
        asyncio.run(
            fixture._execute_separate_subject_context(
                coverage=frozenset(CoverageAtom(kind="field", name=name) for name in ("K0", "K1")),
                requirement_coverage=frozenset({CoverageAtom(kind="field", name="K0")}),
                environment=True,
                expose_evidence=False,
            )
        )
    except (EffectRejected, ContractViolation, PreparationRejected) as error:
        actual = {"status": "rejected", "code": error.code.value}
    else:
        raise AssertionError("Admission-negative declaration was admitted")
    assert calls == 0
    assert actual == case["expected"]
