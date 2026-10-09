# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned evidence aliases retain the frozen evidence/consumption coupling."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.evidence import (
    AssessmentSubmission,
    admit_qualification,
    evidence_revision_view,
    evidence_validity,
    verify_evidence,
)
from anonymizer.engine.graph_sdk.executor import LocalCompleted
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import AssociationInput
from anonymizer.graph.workflow import AdmittedWorkflow, admit_static_workflow
from tests.graph_sdk import qualification_fixtures
from tests.graph_sdk import test_qualification_subject_context as fixture
from tests.graph_sdk.evidence_fixtures import _qualification_limits


@pytest.mark.parametrize("latest", [2, 3])
def test_versioned_evidence_and_consumed_alias_are_stale(latest: int, monkeypatch: pytest.MonkeyPatch) -> None:
    original_run = fixture._SeparateAssessment.run
    provider_type = qualification_fixtures._BindingProvider

    def provider(*args: Any, **kwargs: Any) -> qualification_fixtures._BindingProvider:
        kwargs["versions"] = (1, latest)
        return provider_type(*args, **kwargs)

    def admit(**kwargs: Any) -> AdmittedWorkflow:
        for operation in (kwargs["interface"], *(node.operation for node in kwargs["nodes"])):
            object.__setattr__(
                operation,
                "output_dependencies",
                tuple(
                    replace(dep, inputs=frozenset({"b"}), identity_input="b") if dep.output == "evidence" else dep
                    for dep in operation.output_dependencies
                ),
            )
        return admit_static_workflow(**kwargs)

    async def run(self: fixture._SeparateAssessment, request: tuple[AssociationInput, ...]) -> LocalCompleted:
        completed = await original_run(self, request)
        source = next(port for port in request[0].inputs if port.port == "b")
        return replace(
            completed,
            results=tuple(
                replace(
                    row,
                    outputs=tuple(
                        replace(port, artifact=None, value=source.value) if port.port == "evidence" else port
                        for port in row.outputs
                    ),
                )
                for row in completed.results
            ),
        )

    monkeypatch.setattr(qualification_fixtures, "_BindingProvider", provider)
    monkeypatch.setattr(fixture, "admit_static_workflow", admit)
    monkeypatch.setattr(fixture._SeparateAssessment, "run", run)
    execution, result = asyncio.run(
        fixture._execute_separate_subject_context(versioned_context_dependencies=frozenset({"b"}))
    )
    assert len(result.assessments) == 1, [(row.category, row.reasons) for row in result.record.terminals]
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    submissions = (AssessmentSubmission(fact=result.assessments[0]),)
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    evidence = verified.reference.artifact
    assert evidence.version == latest
    assert verified.reference.consumed == frozenset({evidence})
    assert verified.subject.artifact.key != evidence.key
    for version in (latest, 1):
        selected = {ref.key: ref for ref, _ in sorted(result.artifacts, key=lambda row: row[0].version)}
        selected[evidence.key] = next(
            ref for ref, _ in result.artifacts if ref.key == evidence.key and ref.version == version
        )
        current = evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=tuple(selected.values()),
            absences=(),
            configurations=((result.assessments[0].node, result.assessments[0].environment.configuration),),
            state=execution.context.prepared.state,
        )
        assert evidence_validity(evidence=verified, current=current) == ("current" if version == latest else "stale")
        output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert len(output.qualified) == int(version == latest)
        if version != latest:
            assert output.record.statuses[0].qualification == "unmet"
            assert "stale_evidence" in output.targets[0].withholding
