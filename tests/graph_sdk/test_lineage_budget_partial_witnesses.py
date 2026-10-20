# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained versions count at actual execution's artifact storage boundary."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from anonymizer.engine.graph_sdk.executor import ExecutionLimits
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.engine.graph_sdk.requests import TextArtifactValue
from tests.graph_sdk import test_qualification_subject_context as fixture
from tests.graph_sdk.qualification_fixtures import _inputs


@pytest.mark.parametrize("field", ["max_runtime_artifacts", "max_runtime_artifact_bytes"])
def test_four_retained_versions_require_four_runtime_units(field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    for limit in (4, 3):

        def limits(**kwargs: Any) -> ExecutionLimits:
            return ExecutionLimits(**{**kwargs, field: limit})

        monkeypatch.setattr(fixture, "ExecutionLimits", limits)
        execution, result = asyncio.run(
            fixture._execute_separate_subject_context(
                versioned_context_dependencies=frozenset({"a"}), versioned_context_ports=("a",)
            )
        )
        if limit == 4:
            assert len(result.artifacts) == 4
            assert all(isinstance(value, TextArtifactValue) for _, value in result.artifacts)
            assert (
                sum(len(value.text.encode()) for _, value in result.artifacts if isinstance(value, TextArtifactValue))
                == 4
            )
            admitted, current, submissions = _inputs(execution, result, latest=True)
            output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
            assert len(output.qualified) == 1
        else:
            assert len(result.artifacts) == 3
            assert not result.assessments and not result.final_outputs
            assert result.record.terminals[0].category == "blocked"
            assert result.record.terminals[0].reasons == frozenset({"artifact_limit_exhausted"})
