# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared source identity retains distinct target and port occurrences."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk.data import DataGraph, DataLimits
from anonymizer.engine.graph_sdk.executor import ExecutionLimits, RootInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import ArtifactType, EffectCode, EffectRejected
from tests.graph_sdk.evidence_fixtures import _execute_assessment, _IdentityHop
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_assessment_executor import _AssessingLocal
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def _sources():
    graph = DataGraph.new()
    graph, first = graph.add_text("same")
    graph, second = graph.add_text("same")
    graph, source = graph.add_text("same")
    return graph.validate(
        targets=(first, second),
        source_relations=(),
        contexts=(),
        dependencies=(),
        coherence=(),
        atomic=(),
        output_regions=(),
        limits=DataLimits(max_datums=3, max_targets=2, max_text_bytes=12, max_declarations=0, max_group_members=0),
    ), source


@pytest.mark.parametrize("shared", [False, True])
def test_two_target_predecessors_share_only_actual_source_identity(shared: bool) -> None:
    data, source = _sources()
    unique = 1 if shared else 2
    execution, result = asyncio.run(
        _execute_assessment(
            data=data,
            target_count=2,
            bound_input_source=source if shared else None,
            predecessor_input=True,
            execution_limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=0,
                max_runtime_artifacts=unique + 2,
                max_runtime_artifact_bytes=4 * unique + 16,
                max_collection_items=0,
            ),
        )
    )
    roots = [fact for fact in result.provenance if isinstance(fact.key, RootInputKey)]
    hops = [fact for fact in result.provenance if any(parent in fact.parents for parent in (r.key for r in roots))]
    assert len(roots) == len(hops) == 2
    assert len({fact.key.target for fact in roots}) == len({fact.key.target for fact in hops}) == 2
    assert len({fact.artifact for fact in roots}) == len({fact.artifact for fact in hops}) == unique
    assert len(result.artifacts) == unique + 2
    assert len(result.ports) == 8 and len(result.provenance) == 6
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 2
    assert len(result.record.terminals) == 4
    admitted, current, submissions = _inputs(execution, result, port_facts=8)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 2
    with pytest.raises(EffectRejected) as error:
        _inputs(execution, result, port_facts=7)
    assert error.value.code is EffectCode.LIMIT_EXCEEDED


@pytest.mark.parametrize("bound", ["artifacts", "bytes"])
def test_shared_initial_capacity_one_less_rejects_before_callbacks(bound: str, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    async def unexpected(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("initial capacity rejection must precede callbacks")

    monkeypatch.setattr(_IdentityHop, "run", unexpected)
    monkeypatch.setattr(_AssessingLocal, "run", unexpected)
    data, source = _sources()
    with pytest.raises(EffectRejected) as error:
        asyncio.run(
            _execute_assessment(
                data=data,
                target_count=2,
                bound_input_source=source,
                predecessor_input=True,
                alias_output=True,
                execution_limits=ExecutionLimits(
                    max_local_in_flight=1,
                    max_remote_outstanding=0,
                    max_runtime_artifacts=0 if bound == "artifacts" else 1,
                    max_runtime_artifact_bytes=3 if bound == "bytes" else 4,
                    max_collection_items=0,
                ),
            )
        )
    assert error.value.code is EffectCode.LIMIT_EXCEEDED
    assert calls == []


@pytest.mark.parametrize("different_type", [False, True])
def test_same_source_through_distinct_ports_preserves_type_and_occurrence(different_type: bool) -> None:
    execution, result = asyncio.run(
        _execute_separate_subject_context(
            context_from_subject=True,
            context_artifact_type=ArtifactType(name="other", revision=1) if different_type else None,
        )
    )
    roots = {fact.key.port: fact for fact in result.provenance if isinstance(fact.key, RootInputKey)}
    assert set(roots) == {"subject", "context"}
    assert (roots["subject"].artifact == roots["context"].artifact) is not different_type
    assert roots["subject"].key != roots["context"].key
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1


def test_shared_artifact_does_not_erase_target_owned_producer_checks() -> None:
    data, source = _sources()
    execution, result = asyncio.run(
        _execute_assessment(data=data, target_count=2, bound_input_source=source, predecessor_input=True)
    )
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(baseline.qualified) == 2
    roots = {fact.key for fact in result.provenance if isinstance(fact.key, RootInputKey)}
    first, second = (fact for fact in result.provenance if fact.parents & roots)
    assert first.artifact == second.artifact and first.key.target != second.key.target


def test_exact_shared_initial_capacity_preserves_all_root_occurrences() -> None:
    data, source = _sources()
    execution, result = asyncio.run(
        _execute_assessment(
            data=data,
            target_count=2,
            bound_input_source=source,
            predecessor_input=True,
            alias_output=True,
            execution_only=True,
            execution_limits=ExecutionLimits(
                max_local_in_flight=1,
                max_remote_outstanding=0,
                max_runtime_artifacts=1,
                max_runtime_artifact_bytes=4,
                max_collection_items=0,
            ),
        )
    )
    assert len(result.artifacts) == 1
    assert len(result.ports) == 8 and len(result.provenance) == 6
    assert len(result.record.terminals) == 4
    assert all(item.category == "success" for item in result.record.terminals)
    assert len(execution.context.prepared.bound_inputs) == 2
