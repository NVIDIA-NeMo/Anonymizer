# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact retained-coordinate witnesses for the final partial rows."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.data import DatumDependency
from anonymizer.engine.graph_sdk.evidence import evidence_revision_view, evidence_validity
from anonymizer.engine.graph_sdk.executor import BoundInputKey, InitialCollectionKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk import test_qualification_subject_context as fixture
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_data_graph import _graph_with_texts, _validate
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_duplicate_actual_source_binding_identity() -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a"}), versioned_context_ports=("a",)
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    bound = execution.context.bound_context
    assert bound is not None
    sources = bound.receipt.sources
    assert len(sources) == 1
    object.__setattr__(bound.receipt, "sources", (*sources, replace(sources[0])))
    assert bound.receipt.sources[0].identity == bound.receipt.sources[1].identity
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.DUPLICATE
    object.__setattr__(bound.receipt, "sources", sources)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline


@pytest.mark.parametrize("submitted", [True, False])
def test_separate_candidate_version_unavailability(submitted: bool) -> None:
    execution, result = asyncio.run(_execute_separate_subject_context(initial_subject=True, expose_evidence=False))
    admitted, current, submissions = _inputs(execution, result, latest=True)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == len(baseline.verified) == 1
    bound = sorted(
        (fact for fact in result.provenance if isinstance(fact.key, BoundInputKey)),
        key=lambda fact: fact.artifact.version,
    )
    older, latest = bound
    assert older.artifact.key == latest.artifact.key and (older.artifact.version, latest.artifact.version) == (1, 2)
    subject = next(port for port in result.ports if port.port == "subject")
    context = next(port for port in result.ports if port.port == "context")
    evidence = result.assessments[0].evidence_artifact
    assert subject.artifact == latest.artifact
    assert len({subject.artifact.key, context.artifact.key, evidence.key}) == 3
    finals = result.final_outputs
    assert next(final for final in finals if final.port == "result").candidate.artifact == latest.artifact
    view = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=tuple(older.artifact if ref == latest.artifact else ref for ref in current.artifacts),
        configurations=current.configurations,
    )
    assert context.artifact in view.artifacts and evidence in view.artifacts
    output = qualify(admitted=admitted, result=result, current=view, submissions=submissions if submitted else ())
    assert len(output.verified) == int(submitted) and not output.qualified
    assert not output.targets[0].verified
    assert output.targets[0].candidate == baseline.targets[0].candidate
    assert not view.candidates
    assert not output.record.statuses[0].artifact_available
    assert not output.record.statuses[0].protection_available
    assert output.record.statuses[0].completion == "closed"
    if submitted:
        assert evidence_validity(evidence=output.verified[0], current=view) == "stale"
    assert output.record.statuses[0].qualification == "unknown"
    assert output.targets[0].withholding == frozenset(
        {"missing_candidate", "stale_evidence" if submitted else "missing_assessment"}
    )
    assert result.final_outputs is finals and len(result.assessments) == 1
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline


def test_protected_initial_collection_exact_binding_parents() -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            versioned_context_dependencies=frozenset({"a"}), versioned_context_ports=("a",), initial_collection=True
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == len(output.verified) == 1
    assert output.record.statuses[0].qualification == "met" and not output.targets[0].withholding
    collection = next(fact for fact in result.provenance if isinstance(fact.key, InitialCollectionKey))
    assert isinstance(collection.key, InitialCollectionKey)
    parents = tuple(fact for fact in result.provenance if isinstance(fact.key, BoundInputKey))
    assert len(parents) == 2 and collection.parents == frozenset(fact.key for fact in parents)
    assert {fact.artifact.version for fact in parents} == {1, 2}
    assert len({fact.artifact.key for fact in parents}) == 1
    bound = execution.context.bound_context
    assert bound is not None and len(bound.receipt.sources) == 1
    source = bound.receipt.sources[0]
    for fact in parents:
        key = fact.key
        assert isinstance(key, BoundInputKey)
        assert key.binding_artifact.declaration == source.identity == collection.key.declaration
        assert key.node == source.declaration.node and key.port == source.declaration.port == "a"
        assert key.target == source.declaration.target and key.binding_artifact.version == fact.artifact.version
    consumed = next(port for port in result.ports if port.port == "a")
    assert consumed.artifact == collection.artifact
    assert collection.artifact in output.verified[0].reference.consumed


def test_collection_required_declaration_missing_before_effects(monkeypatch: pytest.MonkeyPatch) -> None:
    original = fixture.start_initial_binding
    attempts = 0

    async def probe(**kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        declarations = kwargs["declarations"]
        assert len(declarations) == 1 and declarations[0].materialization.kind == "collection"
        providers = tuple(resource.factory() for resource in kwargs["resources"])
        assert all(provider.calls == provider.closes == 0 for provider in providers)
        omitted: dict[str, Any] = {**kwargs, "declarations": ()}
        with pytest.raises(EffectRejected) as error:
            await original(**omitted)
        assert error.value.code is EffectCode.MISSING
        assert all(provider.calls == provider.closes == 0 for provider in providers)
        return await original(**kwargs)

    monkeypatch.setattr(fixture, "start_initial_binding", probe)
    test_protected_initial_collection_exact_binding_parents()
    assert attempts == 1


def test_absent_consumed_artifact_coordinate() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 1
    port = next(item for item in result.ports if item.port == "context")
    original = port.artifact
    absent = replace(original, key=max(ref.key for ref, _ in result.artifacts) + 1)
    assert absent.invocation == original.invocation and absent not in dict(result.artifacts)
    object.__setattr__(port, "artifact", absent)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.MISSING
    object.__setattr__(port, "artifact", original)
    assert qualify(admitted=admitted, result=result, current=current, submissions=submissions) == baseline


def test_selected_same_graph_dependency_endpoint_is_valid() -> None:
    graph, (left, right) = _graph_with_texts("A", "B")
    dependency = DatumDependency(prerequisite=left, dependent=right)
    validated = _validate(graph, (left, right), dependencies=(dependency,))
    assert validated.dependencies == frozenset({dependency})
