# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latest-version candidate integrity and initial binding storage limits."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.context import (
    ContextSourceRef,
    SourceItem,
)
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission, admit_qualification, evidence_revision_view
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.evidence_fixtures import _execute_assessment, _qualification_limits
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource, _inputs


def test_latest_candidate_sibling_is_unavailable() -> None:
    from anonymizer.engine.graph_sdk.executor import BoundInputKey

    provider = _BindingProvider(source=ContextSourceRef(name="version-owner", revision=1), versions=(1, 2))
    execution, result = asyncio.run(
        _execute_assessment(
            initial_resources=(_initial_resource(provider),),
            initial_item_limit=2,
            initial_version_selection="latest",
            alias_output=True,
        )
    )
    older = next(
        fact.artifact
        for fact in result.provenance
        if isinstance(fact.key, BoundInputKey) and fact.artifact.version == 1
    )
    admitted = admit_qualification(
        execution=execution, productions=execution.assessment_productions, limits=_qualification_limits()
    )
    current = evidence_revision_view(
        admitted=admitted,
        result=result,
        artifacts=(older,),
        configurations=((result.assessments[0].node, result.assessments[0].environment.configuration),),
    )
    assert not current.candidates
    submissions = (AssessmentSubmission(fact=result.assessments[0]),)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert not output.qualified
    assert "missing_candidate" in output.targets[0].withholding
    assert output.record.statuses[0].qualification == "unknown"
    assert not output.record.statuses[0].artifact_available
    without_assessment = qualify(admitted=admitted, result=result, current=current, submissions=())
    assert without_assessment.record.statuses[0].qualification == "unknown"
    assert not without_assessment.record.statuses[0].artifact_available


def test_source_versions_reject_at_typed_constructor() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="version-owner", revision=1), versions=(1, 2))
    execution, _ = asyncio.run(
        _execute_assessment(
            initial_resources=(_initial_resource(provider),),
            initial_item_limit=2,
            initial_version_selection="latest",
        )
    )
    bound = execution.context.bound_context
    assert bound is not None
    association = next(iter(bound.receipt.requests.dispatches[0].associations))
    with pytest.raises(EffectRejected) as error:
        SourceItem(association=association, key=0, version=0, text="invalid")
    assert error.value.code is EffectCode.INVALID_VALUE


@pytest.mark.parametrize("limit_name", ["max_runtime_artifacts", "max_runtime_artifact_bytes"])
def test_initial_latest_storage_counts_all_versions_at_exact_and_one_over(limit_name: str) -> None:
    from anonymizer.engine.graph_sdk.executor import ExecutionLimits

    exact = ExecutionLimits(
        max_local_in_flight=1,
        max_remote_outstanding=0,
        max_runtime_artifacts=2,
        max_runtime_artifact_bytes=2,
        max_collection_items=0,
    )
    for one_over in (True, False):
        provider = _BindingProvider(source=ContextSourceRef(name="version-owner", revision=1), versions=(1, 2))
        limits = replace(exact, **{limit_name: 1}) if one_over else exact
        coroutine = _execute_assessment(
            initial_resources=(_initial_resource(provider),),
            initial_item_limit=2,
            initial_version_selection="latest",
            alias_output=True,
            execution_limits=limits,
        )
        if one_over:
            with pytest.raises(EffectRejected) as error:
                asyncio.run(coroutine)
            assert error.value.code is EffectCode.LIMIT_EXCEEDED
        else:
            execution, result = asyncio.run(coroutine)
            assert len(result.artifacts) == 2
            assert {ref.key for ref, _ in result.artifacts} == {0}
            assert {ref.version for ref, _ in result.artifacts} == {1, 2}
            admitted, current, submissions = _inputs(execution, result, latest=True)
            assert qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified
        assert provider.calls == provider.closes == 1


def test_initial_latest_zero_parent_facts_do_not_consume_provenance_edges() -> None:
    from anonymizer.engine.graph_sdk.executor import BoundInputKey

    for limit in (0, 1):
        provider = _BindingProvider(source=ContextSourceRef(name="version-owner", revision=1), versions=(1, 2))
        coroutine = _execute_assessment(
            initial_resources=(_initial_resource(provider),),
            initial_item_limit=2,
            initial_version_selection="latest",
            assessment_edge_limit=limit,
        )
        if not limit:
            with pytest.raises(EffectRejected) as error:
                asyncio.run(coroutine)
            assert error.value.code is EffectCode.LIMIT_EXCEEDED
            continue
        execution, result = asyncio.run(coroutine)
        bound = [fact for fact in result.provenance if isinstance(fact.key, BoundInputKey)]
        assert len(bound) == 2
        assert all(not fact.parents for fact in bound)
        assert sum(len(fact.parents) for fact in result.provenance) == 1
        admitted, current, submissions = _inputs(execution, result, latest=True)
        assert qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified
