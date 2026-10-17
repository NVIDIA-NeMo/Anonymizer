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


def test_bound_input_producer_names_a_retained_bound_artifact() -> None:
    provider = _BindingProvider(source=ContextSourceRef(name="source", revision=1))
    execution, result = asyncio.run(_execute_assessment(initial_resources=(_initial_resource(provider),)))
    admitted, current, submissions = _inputs(execution, result)
    assert len(qualify(admitted=admitted, result=result, current=current, submissions=submissions).qualified) == 1
    fact = next(item for item in result.provenance if isinstance(item.key, BoundInputKey))
    old = cast(BoundInputKey, fact.key)
    bound = execution.context.bound_context
    assert bound is not None
    assert any(item.reference == old.binding_artifact for item in bound.artifacts)
    assert provider.calls == provider.closes == 1
