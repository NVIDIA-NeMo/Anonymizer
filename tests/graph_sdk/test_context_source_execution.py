# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executable coverage for explicit static context input sources."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectRejected
from anonymizer.graph.workflow import (
    ArtifactType,
    ContractViolation,
    CoverageAtom,
)
from tests.graph_sdk.context_source_assertions import (
    _assert_collection_root_preflight as _assert_collection_root_preflight,
)
from tests.graph_sdk.context_source_assertions import (
    _assert_context_execution,
    _assert_special_fixture_execution,
)
from tests.graph_sdk.context_source_fixtures import (
    CONTEXT_ADMISSION_CASES,
    CONTEXT_CASES,
    _assert_context_source_initial_admission,
    _nested_workflow,
    _static_fixture,
)


@pytest.mark.parametrize("case", CONTEXT_CASES, ids=lambda case: cast(str, case["id"]))
def test_context_source_static_reference_cases(case: dict[str, Any]) -> None:
    expected = cast(dict[str, object], case["expected"])
    nested = bool(cast(dict[str, object], case["input"]).get("nested"))
    if nested:
        input_name = next(iter(cast(dict[str, str], cast(dict[str, object], case["input"])["interface_inputs"])))
        input_type = ArtifactType(
            name=cast(dict[str, str], cast(dict[str, object], case["input"])["interface_inputs"])[input_name],
            revision=1,
        )
        workflow, _, _ = _nested_workflow(input_type)
        admitted = workflow.workflow
    elif expected["static"] == "accepted":
        admitted, _, _ = _static_fixture(case)
    else:
        with pytest.raises(ContractViolation) as rejected:
            _static_fixture(case)
        assert rejected.value.code.value == expected["static"]
        return
    assert admitted.workflow is not None
    if case["id"] == "context_evidence_projection":
        outcome = admitted.interface.outcomes[0]
        promise = next(iter(outcome.evidence))
        assert promise.subject_port == "context"
        assert promise.consumed_ports == frozenset({"context"})
        assert next(iter(promise.coverage)) == CoverageAtom(kind="field", name="text")
        assert next(iter(outcome.context)).port == "context"
        assert admitted.interface.output_dependencies[0].inputs == frozenset({"context"})


@pytest.mark.parametrize("case", CONTEXT_ADMISSION_CASES, ids=lambda case: cast(str, case["id"]))
def test_context_source_initial_admission_reference_cases(case: dict[str, Any]) -> None:
    asyncio.run(_assert_context_source_initial_admission(case))


def test_scalar_collection_and_nested_context_sources_execute_with_exact_identity() -> None:
    asyncio.run(_assert_context_execution("scalar"))
    asyncio.run(_assert_context_execution("collection"))
    asyncio.run(_assert_context_execution("nested"))
    asyncio.run(_assert_context_execution("substitution"))


@pytest.mark.parametrize("latest", [False, True])
def test_optional_context_omission_closes_unstarted_without_an_attempt(latest: bool) -> None:
    asyncio.run(_assert_context_execution("omitted", latest=latest))


def test_ordinary_root_and_actual_outcome_context_union_execute_at_real_boundaries() -> None:
    asyncio.run(_assert_special_fixture_execution("ordinary_context_use_preserved"))
    asyncio.run(_assert_special_fixture_execution("outcome_context_union"))


def test_initial_collection_schema_rejects_same_typed_caller_root_before_provider_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _assert_collection_root_preflight(monkeypatch)


@pytest.mark.parametrize("mode", ["collection", "nested", "substitution"])
def test_initial_materialization_retains_versions_of_one_runtime_key(mode: str) -> None:
    asyncio.run(_assert_context_execution(mode, same_key_versions=True, artifact_limit=3))


def test_initial_materialization_counts_versions_against_artifact_limit() -> None:
    with pytest.raises(EffectRejected) as rejected:
        asyncio.run(_assert_context_execution("collection", same_key_versions=True, artifact_limit=2))
    assert rejected.value.code.value == "limit_exceeded"
