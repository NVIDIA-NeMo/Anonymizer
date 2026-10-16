# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retained binding corruption cannot bypass qualification or repeat effects."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef, admit_context_plan
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource, _inputs


@pytest.mark.parametrize(
    "defect",
    [
        "owner-and-duplicate",
        "oversize-and-invalid",
        "receipt-only-oversize",
    ],
)
def test_retained_binding_checks_preserve_precedence_and_do_not_repeat_provider_work(defect: str) -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1), versions=(1, 2))
    execution, result = asyncio.run(
        _execute_assessment(
            initial_resources=(_initial_resource(provider),),
            initial_item_limit=2,
            initial_version_selection="latest",
        )
    )
    admitted, current, submissions = _inputs(execution, result, latest=True)
    bound = execution.context.bound_context
    assert bound is not None
    original = bound.artifacts
    original_execution = result._execution
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    foreign = replace(original[-1], source=ContextSourceRef(name="foreign", revision=1))
    if defect == "owner-and-duplicate":
        changed = (*original, foreign)
        expected = EffectCode.FOREIGN_OWNER
    else:
        changed = (None,) * (admitted.limits.max_port_facts + 1)
        expected = EffectCode.LIMIT_EXCEEDED
    if not defect.startswith("receipt-only"):
        object.__setattr__(bound, "artifacts", changed)
    object.__setattr__(bound.receipt, "artifacts", changed)
    try:
        with pytest.raises(EffectRejected) as error:
            qualify(admitted=admitted, result=result, current=current, submissions=submissions)
        assert error.value.code is expected
        assert provider.calls == provider.closes == 1
    finally:
        object.__setattr__(bound, "artifacts", original)
        object.__setattr__(bound.receipt, "artifacts", original)
        object.__setattr__(result, "_execution", original_execution)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    assert provider.calls == provider.closes == 1


def test_source_join_refactor_preserves_existing_context_admission_diagnostic() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1), versions=(1, 2))
    execution, _ = asyncio.run(
        _execute_assessment(
            initial_resources=(_initial_resource(provider),),
            initial_item_limit=2,
            initial_version_selection="latest",
        )
    )
    context = execution.context
    bound = context.bound_context
    assert bound is not None
    original = bound.artifacts
    changed = (*original[:-1], replace(original[-1], source=ContextSourceRef(name="foreign", revision=1)))
    object.__setattr__(bound, "artifacts", changed)
    object.__setattr__(bound.receipt, "artifacts", changed)
    try:
        with pytest.raises(EffectRejected) as error:
            admit_context_plan(prepared=context.prepared, bound_context=bound)
        assert error.value.code is EffectCode.CONTRADICTORY
        assert provider.calls == provider.closes == 1
    finally:
        object.__setattr__(bound, "artifacts", original)
        object.__setattr__(bound.receipt, "artifacts", original)
