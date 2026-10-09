# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qualification validates roles per occurrence while preserving lawful aliases."""

from __future__ import annotations

import asyncio

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.qualification import qualify
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


def test_candidate_and_evidence_can_identify_the_same_artifact() -> None:
    execution, result = asyncio.run(_execute_assessment(alias_output=True))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == len(output.verified) == 1
    candidate = output.qualified[0].candidate
    assert output.verified[0].reference.artifact == candidate.artifact
    assert output.verified[0].subject == candidate
    assert not output.required_decisions


def test_subject_occurrence_cannot_be_relabelled_as_a_decision() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    baseline = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    subject = next(port for port in result.ports if port.port == "subject")
    assert subject.role == "candidate"
    object.__setattr__(subject, "role", "decision")
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.CONTRADICTORY
    object.__setattr__(subject, "role", "candidate")
    restored = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert restored.qualified == baseline.qualified
    assert restored.required_decisions == baseline.required_decisions
