# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual typed owners for neutral assessment fields absent from submissions."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk._effect_values import EffectCode, EffectRejected
from anonymizer.engine.graph_sdk.capabilities import PreparationCode, PreparationRejected
from anonymizer.engine.graph_sdk.evidence import AssessmentSubmission
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph._values import ContractViolation, ValidationCode
from anonymizer.graph.workflow import CoverageAtom, CoverageKind, EvidencePromise
from tests.graph_sdk.test_qualification import _inputs
from tests.graph_sdk.test_qualification_subject_context import _execute_separate_subject_context


@pytest.mark.parametrize("field", ["coverage", "consumed", "subject", "target", "evidence_port"])
def test_submission_cannot_override_factory_derived_fields(field: str) -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    arguments: dict[str, Any] = {"fact": result.assessments[0], field: object()}
    with pytest.raises(TypeError):
        AssessmentSubmission(**arguments)
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert len(output.qualified) == 1
    assert output.verified[0].coverage == output.verified[0].promise.coverage


def test_coverage_is_a_closed_set_owned_by_the_admitted_promise() -> None:
    atom = CoverageAtom(kind="field", name="K0")
    arguments = {
        "name": "checked",
        "meaning": "privacy",
        "subject_port": "subject",
        "consumed_ports": frozenset({"context"}),
    }
    with pytest.raises(ContractViolation) as error:
        EvidencePromise(**arguments, coverage=cast(frozenset[CoverageAtom], (atom, atom)))
    assert error.value.code is ValidationCode.INVALID_TYPE
    promise = EvidencePromise(**arguments, coverage=frozenset((atom, atom)))
    assert promise.coverage == frozenset({atom})
    with pytest.raises(ContractViolation) as error:
        CoverageAtom(kind=cast(CoverageKind, "unsupported"), name="K0")
    assert error.value.code is ValidationCode.INVALID_VALUE


@pytest.mark.parametrize("port_name", ["subject", "context"])
def test_retained_subject_and_consumed_ports_require_exact_provenance(port_name: str) -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    port = next(item for item in result.ports if item.port == port_name)
    evidence = next(item.artifact for item in result.ports if item.port == "evidence")
    object.__setattr__(port, "artifact", evidence)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.CONTRADICTORY


@pytest.mark.parametrize("kind", ["field", "source_view"])
def test_coverage_matching_preserves_atom_kind(kind: CoverageKind) -> None:
    promised = frozenset({CoverageAtom(kind=kind, name="person")})
    required = frozenset({CoverageAtom(kind="field", name="person")})
    if kind != "field":
        with pytest.raises(PreparationRejected) as error:
            asyncio.run(_execute_separate_subject_context(coverage=promised, requirement_coverage=required))
        assert error.value.code is PreparationCode.PROTECTION_INELIGIBLE
        return
    execution, result = asyncio.run(_execute_separate_subject_context(coverage=promised, requirement_coverage=required))
    admitted, current, submissions = _inputs(execution, result)
    output = qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert output.verified[0].coverage == promised
    assert len(output.qualified) == 1
    assert not output.targets[0].withholding


def test_consumed_port_cannot_introduce_an_unsupported_role() -> None:
    execution, result = asyncio.run(_execute_separate_subject_context())
    admitted, current, submissions = _inputs(execution, result)
    port = next(item for item in result.ports if item.port == "context")
    object.__setattr__(port, "role", "other")
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    assert error.value.code is EffectCode.UNSUPPORTED
