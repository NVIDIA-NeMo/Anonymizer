# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Initial provenance must identify an actually retained binding artifact."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

import pytest

from anonymizer.engine.graph_sdk.context import ContextSourceRef
from anonymizer.engine.graph_sdk.executor import BoundInputKey
from anonymizer.engine.graph_sdk.qualification import qualify
from anonymizer.graph import EffectCode, EffectRejected
from anonymizer.graph._values import DatumId
from tests.graph_sdk.evidence_fixtures import _execute_assessment
from tests.graph_sdk.qualification_fixtures import _BindingProvider, _initial_resource, _inputs


def test_invocation_artifact_cannot_replace_typed_binding_artifact() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1))
    execution, result = asyncio.run(_execute_assessment(initial_resources=(_initial_resource(provider),)))
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    fact = next(item for item in result.provenance if isinstance(item.key, BoundInputKey))
    with pytest.raises(EffectRejected) as error:
        replace(cast(Any, fact.key), binding_artifact=fact.artifact)
    assert error.value.code is EffectCode.INVALID_TYPE


@pytest.mark.parametrize(
    "coordinate,contradiction",
    [
        ("key", None),
        ("version", None),
        ("key", "parents"),
        ("key", "decision"),
        ("version", "parents"),
        ("version", "decision"),
        (None, "parents"),
        (None, "decision"),
        ("key", "foreign"),
        ("version", "foreign"),
    ],
)
def test_binding_reference_checks_preserve_missing_and_owner_precedence(
    coordinate: str | None,
    contradiction: str | None,
) -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1))
    execution, result = asyncio.run(_execute_assessment(initial_resources=(_initial_resource(provider),)))
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    fact = next(item for item in result.provenance if isinstance(item.key, BoundInputKey))
    old = cast(BoundInputKey, fact.key)
    new = (
        old
        if coordinate is None
        else replace(
            old,
            binding_artifact=replace(
                old.binding_artifact, **{coordinate: getattr(old.binding_artifact, coordinate) + 1}
            ),
        )
    )
    if contradiction == "foreign":
        new = replace(new, target=DatumId.new(graph=result.record.graph))
    bound = execution.context.bound_context
    assert bound is not None
    assert any(item.reference == new.binding_artifact for item in bound.artifacts) == (coordinate is None)
    # Replace producer coordinates consistently; immutable hash keys are never mutated in place.
    for retained in result.provenance:
        if retained.key == old:
            object.__setattr__(retained, "key", new)
        object.__setattr__(retained, "parents", frozenset(new if key == old else key for key in retained.parents))
    object.__setattr__(
        result,
        "_input_parents",
        tuple(
            (target, activation, port, new if parent == old else parent)
            for target, activation, port, parent in result._input_parents
        ),
    )
    if contradiction == "parents":
        object.__setattr__(fact, "parents", frozenset({next(p.key for p in result.provenance if p is not fact)}))
    elif contradiction == "decision":
        object.__setattr__(fact, "decision", True)
    with pytest.raises(EffectRejected) as error:
        qualify(admitted=admitted, result=result, current=current, submissions=submissions)
    expected = (
        EffectCode.FOREIGN_OWNER
        if contradiction == "foreign"
        else EffectCode.CONTRADICTORY
        if coordinate is None
        else EffectCode.MISSING
    )
    assert error.value.code is expected
    assert provider.calls == provider.closes == 1
