# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional invocation and owner-projection boundary witnesses."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace

import pytest

from anonymizer.engine.graph_sdk.executor import OperationOutputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import ArtifactRef, EffectRejected
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs


def test_entry_terminal_and_final_owner_fields_are_projected() -> None:
    execution, result = asyncio.run(_execute_assessment())
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    entry = next(iter(result.states[0].entries))
    terminal = next(row for row in result.record.terminals if row.activation == entry.activation)
    final = result.final_outputs[0]
    assert "target" not in {field.name for field in fields(entry)}
    assert "target" not in {field.name for field in fields(terminal)}
    assert "node" not in {field.name for field in fields(final)}
    assert isinstance(final.producer, OperationOutputKey)
    assert final.producer.activation == entry.activation
    assert any(port.activation == entry.activation and port.node == entry.template for port in result.ports)


def test_unretained_evidence_sibling_cannot_be_claimed_current() -> None:
    from anonymizer.engine.graph_sdk.evidence import evidence_revision_view

    execution, result = asyncio.run(_execute_assessment())
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    evidence = result.assessments[0].evidence_artifact
    sibling = replace(evidence, version=evidence.version + 1)
    assert all(ref != sibling for ref, _ in result.artifacts)
    with pytest.raises(EffectRejected) as error:
        evidence_revision_view(
            admitted=admitted,
            result=result,
            artifacts=tuple(sibling if ref == evidence else ref for ref, _ in result.artifacts),
            configurations=((result.assessments[0].node, result.assessments[0].environment.configuration),),
        )
    assert error.value.code.value == "missing"


def test_missing_evidence_only_revision_is_unknown_with_dependencies_current() -> None:
    from anonymizer.engine.graph_sdk.evidence import evidence_revision_view, evidence_validity, verify_evidence
    from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context

    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    (verified,) = verify_evidence(admitted=admitted, result=result, submissions=submissions)
    assert evidence_validity(evidence=verified, current=current) == "current"
    evidence = verified.reference.artifact
    assert all(isinstance(ref, ArtifactRef) and ref.key != evidence.key for ref in verified.reference.consumed)
    assert verified.subject.artifact.key != evidence.key
    refs = tuple(ref for ref, _ in result.artifacts if ref.key != evidence.key)
    assert verified.subject.artifact in refs and verified.reference.consumed <= frozenset(refs)
    missing = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=refs,
        configurations=((result.assessments[0].node, result.assessments[0].environment.configuration),),
    )
    assert evidence_validity(evidence=verified, current=missing) == "unknown"
    output = qualify(admitted=admitted, result=result, current=missing, submissions=submissions)
    assert not output.qualified
    assert output.record.statuses[0].qualification == "unknown"


def test_output_dependency_cannot_reference_its_own_node_output(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    from anonymizer.graph import ContractViolation
    from tests.graph_sdk import test_qualification_subject_context as fixture
    from tests.graph_sdk.workflow_construction import AdmittedWorkflow, admit_static_workflow

    def static(**kwargs: Any) -> AdmittedWorkflow:
        (node,) = kwargs["nodes"]
        operation = node.operation
        assert "evidence" in {port.name for port in operation.outputs}
        assert "evidence" not in {port.name for port in operation.inputs}
        changed = replace(
            operation,
            output_dependencies=tuple(
                replace(dep, inputs=frozenset({"evidence", "subject"})) if dep.output == "result" else dep
                for dep in operation.output_dependencies
            ),
        )
        kwargs["nodes"] = (replace(node, operation=changed),)
        return admit_static_workflow(**kwargs)

    monkeypatch.setattr(fixture, "admit_static_workflow", static)
    with pytest.raises(ContractViolation) as error:
        asyncio.run(fixture._execute_separate_subject_context())
    assert error.value.code.value == "missing"
