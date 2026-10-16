# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Additional invocation and owner-projection boundary witnesses."""

from __future__ import annotations

import asyncio
from dataclasses import fields, replace

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.executor import BoundInputKey, OperationOutputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectRejected
from anonymizer.graph._values import ArtifactRef, InvocationId
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource, _inputs


def test_latest_binding_import_cannot_cross_invocations() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1), versions=(1, 2))
    execution, result = asyncio.run(
        _execute_assessment(
            initial_resources=(_initial_resource(provider),), initial_item_limit=2, initial_version_selection="latest"
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    imported = [fact.artifact for fact in result.provenance if isinstance(fact.key, BoundInputKey)]
    assert {ref.version for ref in imported} == {1, 2}
    selected = max(imported, key=lambda ref: ref.version)
    object.__setattr__(
        result,
        "artifacts",
        tuple(
            (replace(ref, invocation=InvocationId.new(plan=ref.invocation.plan)) if ref == selected else ref, value)
            for ref, value in result.artifacts
        ),
    )
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == "foreign_owner"


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


def test_output_cannot_substitute_an_undeclared_bound_root() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1), versions=(1, 2))
    execution, result = asyncio.run(
        _execute_assessment(
            initial_resources=(_initial_resource(provider),), initial_item_limit=2, initial_version_selection="latest"
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    selected = next(
        fact for fact in result.provenance if isinstance(fact.key, BoundInputKey) and fact.artifact.version == 2
    )
    key = selected.key
    assert isinstance(key, BoundInputKey)
    foreign_coordinate = replace(
        key,
        port="undeclared",
        binding_artifact=replace(
            key.binding_artifact,
            declaration=type(key.binding_artifact.declaration).new(
                binding=key.binding_artifact.declaration.binding, ordinal=999
            ),
        ),
    )
    extra = object.__new__(type(selected))
    for field in fields(selected):
        object.__setattr__(extra, field.name, getattr(selected, field.name))
    object.__setattr__(extra, "key", foreign_coordinate)
    object.__setattr__(result, "provenance", (*result.provenance, extra))
    final = result.final_outputs[0]
    output = next(fact for fact in result.provenance if fact.key == final.producer)
    assert key in output.parents
    object.__setattr__(output, "parents", frozenset({foreign_coordinate}))
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code.value == "contradictory"


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

    from anonymizer.graph._values import ContractViolation
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
