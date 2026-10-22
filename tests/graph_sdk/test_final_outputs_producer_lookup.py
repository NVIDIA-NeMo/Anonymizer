# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Final outputs must name the producing fact and the artifact it published."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk._execution_state import _ExecutionFacts
from anonymizer.graph import ActivationKey, ArtifactRef, DatumId, EffectCode, EffectRejected, NodeId
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_final_output_whose_artifact_differs_from_the_producer_fact_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = _ExecutionFacts.artifact

    def advanced(
        self: _ExecutionFacts, target: DatumId, occurrence: ActivationKey | NodeId, port: str
    ) -> ArtifactRef | None:
        artifact = original(self, target, occurrence, port)
        return None if artifact is None else replace(artifact, version=artifact.version + 1)

    _, result = asyncio.run(_execute_separate_subject_context())
    assert result.final_outputs

    monkeypatch.setattr(_ExecutionFacts, "artifact", advanced)
    with pytest.raises(EffectRejected) as raised:
        asyncio.run(_execute_separate_subject_context())
    assert raised.value.code is EffectCode.MISSING
